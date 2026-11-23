// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <mutex>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <fmt/ranges.h>
#include "common/assert.h"
#include "common/bit_util.h"
#include "common/container/unordered_set.h"
#include "common/scope_exit.h"
#include "common/n2den_trace.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/back_thread_stats.h"
#include "video_core/query_cache/pending_reports.h"
#include "video_core/gpu_split.h"
#ifdef ARCHITECTURE_x86_64
#include "video_core/macro_turing_jit.h"
#endif
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/textures/texture.h"

namespace Tegra::Engines {

/// First register id that is actually a Macro call.
constexpr u32 MacroRegistersStart = 0xE00;

// Split GPU thread with macros on its front half (gpu_split.h, EDEN_GPU_SPLIT=2; ported from
// Eglantine). The front keeps what macros read: the registers as the back thread will hold them
// when it reaches the same point of the method stream. That is every write the front sends, with
// the shadow RAM applied, plus the few changes the back makes on its own (constant buffer offset,
// shadow_scratch[0] on falcon[4], the macro bind pointer); the front loads and binds the macro
// code itself.
struct Maxwell3D::FrontState {
    std::array<u32, Regs::NUM_REGS> regs{};
    std::array<u32, Regs::NUM_REGS> shadow{};
    Regs::ShadowRamControl control{};
    u32 executing_macro{};
    std::vector<u32> parameters;
    /// A method the draw manager handles was sent since the back last ran DrawDeferred: only
    /// then can it have a deferred draw to issue at the end of a macro.
    bool draw_pending{true};
    /// The back may hold an instanced draw that DrawDeferred has not issued yet. Every state
    /// write outside an instance loop issues it there (ProcessDirtyRegisters), even one that
    /// keeps the register's value, and resets the instance count: such writes are sent then.
    bool instance_pending{true};
    /// The running macro sent a write (the back clears the failure flags after the macro).
    bool macro_wrote{};
};

Maxwell3D::Maxwell3D(MemoryManager& memory_manager_)
    : draw_manager()
    , memory_manager{memory_manager_}
#ifdef ARCHITECTURE_x86_64
    , macro_engine(bool(Settings::values.disable_macro_jit))
#else
    , macro_engine(true)
#endif
    , upload_state{memory_manager, regs.upload}
{
    dirty.flags.flip();
    InitializeRegisterDefaults();
    execution_mask.reset();
    for (size_t i = 0; i < execution_mask.size(); i++)
        execution_mask[i] = IsMethodExecutable(u32(i));
}

Maxwell3D::~Maxwell3D() = default;

void Maxwell3D::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
    upload_state.BindRasterizer(rasterizer_);
}

void Maxwell3D::InitializeRegisterDefaults() {
    // Initializes registers to their default values - what games expect them to be at boot. This is
    // for certain registers that may not be explicitly set by games.

    // Reset all registers to zero
    std::memset(&regs, 0, sizeof(regs));

    // Depth range near/far is not always set, but is expected to be the default 0.0f, 1.0f. This is
    // needed for ARMS.
    for (auto& viewport : regs.viewports) {
        viewport.depth_range_near = 0.0f;
        viewport.depth_range_far = 1.0f;
    }
    for (auto& viewport : regs.viewport_transform) {
        viewport.swizzle.x.Assign(Regs::ViewportSwizzle::PositiveX);
        viewport.swizzle.y.Assign(Regs::ViewportSwizzle::PositiveY);
        viewport.swizzle.z.Assign(Regs::ViewportSwizzle::PositiveZ);
        viewport.swizzle.w.Assign(Regs::ViewportSwizzle::PositiveW);
    }

    // Doom and Bomberman seems to use the uninitialized registers and just enable blend
    // so initialize blend registers with sane values
    regs.blend.color_op = Regs::Blend::Equation::Add_D3D;
    regs.blend.color_source = Regs::Blend::Factor::One_D3D;
    regs.blend.color_dest = Regs::Blend::Factor::Zero_D3D;
    regs.blend.alpha_op = Regs::Blend::Equation::Add_D3D;
    regs.blend.alpha_source = Regs::Blend::Factor::One_D3D;
    regs.blend.alpha_dest = Regs::Blend::Factor::Zero_D3D;
    for (auto& blend : regs.blend_per_target) {
        blend.color_op = Regs::Blend::Equation::Add_D3D;
        blend.color_source = Regs::Blend::Factor::One_D3D;
        blend.color_dest = Regs::Blend::Factor::Zero_D3D;
        blend.alpha_op = Regs::Blend::Equation::Add_D3D;
        blend.alpha_source = Regs::Blend::Factor::One_D3D;
        blend.alpha_dest = Regs::Blend::Factor::Zero_D3D;
    }
    regs.stencil_front_op.fail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.zfail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.zpass = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.func = Regs::ComparisonOp::Always_GL;
    regs.stencil_front_func_mask = 0xFFFFFFFF;
    regs.stencil_front_mask = 0xFFFFFFFF;
    regs.stencil_two_side_enable = 1;
    regs.stencil_back_op.fail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.zfail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.zpass = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.func = Regs::ComparisonOp::Always_GL;
    regs.stencil_back_func_mask = 0xFFFFFFFF;
    regs.stencil_back_mask = 0xFFFFFFFF;

    regs.depth_test_func = Regs::ComparisonOp::Always_GL;
    regs.gl_front_face = Regs::FrontFace::CounterClockWise;
    regs.gl_cull_face = Regs::CullFace::Back;

    // TODO(Rodrigo): Most games do not set a point size. I think this is a case of a
    // register carrying a default value. Assume it's OpenGL's default (1).
    regs.point_size = 1.0f;

    // TODO(bunnei): Some games do not initialize the color masks (e.g. Sonic Mania). Assuming a
    // default of enabled fixes rendering here.
    for (auto& color_mask : regs.color_mask) {
        color_mask.R.Assign(1);
        color_mask.G.Assign(1);
        color_mask.B.Assign(1);
        color_mask.A.Assign(1);
    }

    for (auto& format : regs.vertex_attrib_format) {
        format.constant.Assign(1);
    }

    // NVN games expect these values to be enabled at boot
    regs.rasterize_enable = 1;
    regs.color_target_mrt_enable = 1;
    regs.framebuffer_srgb = 1;
    regs.line_width_aliased = 1.0f;
    regs.line_width_smooth = 1.0f;
    regs.gl_front_face = Maxwell3D::Regs::FrontFace::ClockWise;
    regs.polygon_mode_back = Maxwell3D::Regs::PolygonMode::Fill;
    regs.polygon_mode_front = Maxwell3D::Regs::PolygonMode::Fill;

    shadow_state = regs;
}

bool Maxwell3D::IsMethodExecutable(u32 method) {
    if (method >= MacroRegistersStart) {
        return true;
    }
    switch (method) {
    case NativeRootTables::SelectorMethod:
    case NativeRootTables::LoadMethod:
    case MAXWELL3D_REG_INDEX(draw.end):
    case MAXWELL3D_REG_INDEX(draw.begin):
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(index_buffer.count):
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(index_buffer32_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer16_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer8_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer32_first):
    case MAXWELL3D_REG_INDEX(index_buffer16_first):
    case MAXWELL3D_REG_INDEX(index_buffer8_first):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_first):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent):
    case MAXWELL3D_REG_INDEX(draw_texture.src_y0):
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(topology_override):
    case MAXWELL3D_REG_INDEX(clear_surface):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(sync_info):
    case MAXWELL3D_REG_INDEX(launch_dma):
    case MAXWELL3D_REG_INDEX(inline_data):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return true;
    default:
        return false;
    }
}

void Maxwell3D::ProcessMacro(Core::System& system, u32 method, const u32* base_start, u32 amount, bool is_last_call) {
    if (executing_macro == 0) {
        // A macro call must begin by writing the macro method's register, not its argument.
        ASSERT((method % 2) == 0 && "Can't start macro execution by writing to the ARGS register");
        executing_macro = method;
    }

    macro_params.insert(macro_params.end(), base_start, base_start + amount);
    macro_segments.emplace_back(current_dma_segment, amount);
    current_macro_dirty |= current_dirty;
    current_dirty = false;

    // Call the macro when there are no more parameters in the command buffer
    if (is_last_call) {
        ConsumeSink(system);
        CallMacroMethod(system, executing_macro, macro_params);
        macro_params.clear();
        macro_segments.clear();
        current_macro_dirty = false;
    }
}

namespace {

// [DEBUG] EDEN_N2DEN_PREDICATE_TRACE=1 (with EDEN_NX2_FRAMES_AHEAD=0): finds occlusion
// predicates that go wrong. Standing still: objects hidden for exactly one frame. Moving: frames
// that hide clearly more objects than both neighbours, and reads that got an earlier report. OoT's predicate macro (0x142)
// reads a 16-byte query report (count, timestamp) and draws when the count is not 0. Reads are
// keyed by report address and its n-th use in the frame (a slot is reused within a frame); a key
// whose count is 0 in one frame and not 0 in the frames before and after is logged with the three
// counts and timestamps (the frame before's timestamp again = the read saw last frame's report),
// whether the query cache still held a write for the range, and whether FlushQueryResults waited.
// Frames are split at presents (Common::N2denTrace::present_count).
class PredicateTrace {
public:
    static bool Enabled() {
        static const bool enabled = std::getenv("EDEN_N2DEN_PREDICATE_TRACE") != nullptr;
        return enabled;
    }

    void OnRead(GPUVAddr address, std::span<const u32> words, bool dirty, bool waited) {
        if (words.size() < 4) {
            return;
        }
        std::scoped_lock lock{mutex};
        const u32 present = Common::N2denTrace::present_count.load(std::memory_order_acquire);
        if (present != frame_present) {
            frame.present = frame_present;
            frame.reads = static_cast<u32>(current.size());
            EndFrame();
            frame_present = present;
        }
        const u32 use = uses[address]++;
        const u64 stamp = u64{words[2]} | (u64{words[3]} << 32);
        current.push_back(Read{(address << 8) | (use & 0xff), words[0], stamp, dirty, waited});
        // The same timestamp as this slot's previous read: no report reached the slot since. In
        // the same frame that is the report of the object read before (slots are reused).
        auto& last = last_reads[address];
        if (last.valid && last.stamp == stamp) {
            if (last.present == present) {
                ++frame.stale_in_frame;
                ++stale_logged;
                if (words[0] == 0 && (stale_logged <= 200 || stale_logged % 50 == 0)) {
                    LOG_INFO(HW_GPU,
                             "Predicate read an earlier object's report: slot {:#x} use {}, count 0, "
                             "timestamp {:#x}, query write pending={}, flush waited={} (stale {})",
                             address, use, stamp, dirty, waited, stale_logged);
                }
            } else {
                ++frame.stale_across;
            }
        }
        last = LastRead{stamp, present, true};
        frame.zeros += words[0] == 0 ? 1 : 0;
        ++stats.reads;
        stats.dirty += dirty ? 1 : 0;
        stats.waited += waited ? 1 : 0;
    }

private:
    struct Read {
        u64 key;
        u32 count;
        u64 stamp;
        bool dirty;
        bool waited;
    };

    void EndFrame() {
        if (!older.empty() && !middle.empty() && !current.empty()) {
            std::unordered_map<u64, const Read*> before;
            std::unordered_map<u64, const Read*> after;
            before.reserve(older.size());
            after.reserve(current.size());
            for (const Read& read : older) {
                before.emplace(read.key, &read);
            }
            for (const Read& read : current) {
                after.emplace(read.key, &read);
            }
            for (const Read& read : middle) {
                const auto b = before.find(read.key);
                const auto a = after.find(read.key);
                if (b == before.end() || a == after.end()) {
                    continue;
                }
                const Read& prev = *b->second;
                const Read& next = *a->second;
                if (read.count != 0 && prev.count == 0 && next.count == 0) {
                    ++stats.shows;
                    continue;
                }
                if (read.count != 0 || prev.count == 0 || next.count == 0) {
                    continue;
                }
                ++stats.hides;
                ++hides_logged;
                if (hides_logged <= 300 || hides_logged % 25 == 0) {
                    LOG_INFO(HW_GPU,
                             "Predicate hid an object for one frame: report {:#x} use {}, counts "
                             "{}/{}/{}, timestamps {:#x}/{:#x}/{:#x} ({}), query write pending={}, "
                             "flush waited={}, frame {} with {} predicate reads (hide {})",
                             read.key >> 8, read.key & 0xff, prev.count, read.count, next.count,
                             prev.stamp, read.stamp, next.stamp,
                             read.stamp == prev.stamp ? "last frame's report" : "new report",
                             read.dirty, read.waited, middle_present, middle.size(), hides_logged);
                }
            }
        }
        // A frame that hides clearly more objects than both neighbours (camera moving or not).
        frames_zero[0] = frames_zero[1];
        frames_zero[1] = frames_zero[2];
        frames_zero[2] = frame;
        if (frames_zero[0].reads != 0 && frames_zero[1].reads != 0) {
            const FrameStats& mid = frames_zero[1];
            const u32 neighbours = (std::max)(frames_zero[0].zeros, frames_zero[2].zeros);
            if (mid.zeros >= neighbours + 8 && mid.zeros * 4 >= neighbours * 5) {
                ++stats.spikes;
                ++spikes_logged;
                if (spikes_logged <= 300 || spikes_logged % 25 == 0) {
                    LOG_INFO(HW_GPU,
                             "Predicate hide spike: frame {} hid {} objects ({} before, {} after) of {} "
                             "reads; stale reads in that frame: {} of an earlier object, {} from last "
                             "frame (spike {})",
                             mid.present, mid.zeros, frames_zero[0].zeros, frames_zero[2].zeros,
                             mid.reads, mid.stale_in_frame, mid.stale_across, spikes_logged);
                }
            }
        }
        stats.stale_in_frame += frame.stale_in_frame;
        stats.stale_across += frame.stale_across;
        frame = FrameStats{};
        ++stats.frames;
        if (stats.frames >= 300) {
            LOG_INFO(HW_GPU,
                     "Predicate trace, last {} frames: {:.0f} reads a frame, {} on a pending query "
                     "write, {} waited for one, one-frame hides {}, one-frame shows {}, hide spikes {}, "
                     "stale reads {} of an earlier object / {} from last frame; flush waits "
                     "{} (timeouts {}) since start",
                     stats.frames, static_cast<double>(stats.reads) / stats.frames, stats.dirty,
                     stats.waited, stats.hides, stats.shows, stats.spikes, stats.stale_in_frame,
                     stats.stale_across,
                     Common::N2denTrace::query_flush_waits.load(std::memory_order_relaxed),
                     Common::N2denTrace::query_flush_timeouts.load(std::memory_order_relaxed));
            stats = Stats{};
        }
        older = std::move(middle);
        middle = std::move(current);
        middle_present = frame_present;
        current.clear();
        uses.clear();
    }

    struct Stats {
        u64 frames{};
        u64 reads{};
        u64 dirty{};
        u64 waited{};
        u64 hides{};
        u64 shows{};
        u64 spikes{};
        u64 stale_in_frame{};
        u64 stale_across{};
    };
    struct FrameStats {
        u32 present{};
        u32 reads{};
        u32 zeros{};
        u32 stale_in_frame{};
        u32 stale_across{};
    };
    struct LastRead {
        u64 stamp{};
        u32 present{};
        bool valid{};
    };

    std::mutex mutex;
    u32 frame_present{~0U};
    u32 middle_present{};
    std::vector<Read> older;
    std::vector<Read> middle;
    std::vector<Read> current;
    std::unordered_map<GPUVAddr, u32> uses;
    Stats stats{};
    u64 hides_logged{};
    FrameStats frame{};
    std::array<FrameStats, 3> frames_zero{};
    std::unordered_map<GPUVAddr, LastRead> last_reads;
    u64 stale_logged{};
    u64 spikes_logged{};
};

PredicateTrace predicate_trace;

/// GPU addresses that received a samples-passed report (occlusion counts). A predicate macro that
/// reads one of them decides on an occlusion count.
class OcclusionSlots {
public:
    void Add(GPUVAddr address) {
        std::scoped_lock lock{mutex};
        if (slots.size() >= 0x40000) {
            slots.clear();
            recent.fill(NO_ADDRESS);
        }
        // A game reports into the same slots every frame: an address added since the last clear is
        // already in the set, and inserting it again would only hash it and walk its bucket.
        GPUVAddr& recent_slot = recent[(address >> 2) % recent.size()];
        if (recent_slot == address && address != NO_ADDRESS) {
            return;
        }
        slots.insert(address);
        recent_slot = address;
    }
    bool Contains(GPUVAddr address) {
        std::scoped_lock lock{mutex};
        return slots.contains(address);
    }

private:
    static constexpr GPUVAddr NO_ADDRESS = ~GPUVAddr{0};

    std::mutex mutex;
    /// Flat (open addressing): Ocarina of Time reports into thousands of distinct slots a frame,
    /// and node-based inserts cost about 2% of the back thread.
    Common::unordered_set<GPUVAddr> slots;
    /// Addresses known to be in slots (direct mapped; emptied with it).
    std::array<GPUVAddr, 8192> recent = [] {
        std::array<GPUVAddr, 8192> init;
        init.fill(NO_ADDRESS);
        return init;
    }();
};

OcclusionSlots occlusion_slots;

} // Anonymous namespace

bool Maxwell3D::ReadMacroMemory(GPUVAddr address, std::span<u32> words) {
    if ((address & 3) || !memory_manager.IsFullyMappedRange(address, words.size_bytes())) {
        return false;
    }
    // Below High GPU accuracy query reports (occlusion counts, timestamps) reach guest memory only
    // when a later fence pops: a macro reading them (Ocarina of Time's predicate macros) got the
    // previous value and objects flashed white or vanished for a frame while the camera moved.
    // Write them out first, waiting for pending host results (Eglantine #215, WeslieDE). The
    // caller owns the rasterizer's caches here (the split's back is drained before a front read).
    // EDEN_N2DEN_MACRO_QUERY_FLUSH=0 reads without waiting, as before.
    static const bool flush_queries = [] {
        const char* const value = std::getenv("EDEN_N2DEN_MACRO_QUERY_FLUSH");
        return !(value && value[0] == '0');
    }();
    namespace BackStats = VideoCommon::BackStats;
    const u64 stats_start = BackStats::Enabled() ? BackStats::Stamp() : 0;
    const bool trace = PredicateTrace::Enabled();
    const bool trace_dirty =
        trace && memory_manager.IsMemoryDirty(address, words.size_bytes(), VideoCommon::CacheType::QueryCache);
    const u64 trace_waits = trace ? Common::N2denTrace::query_flush_waits.load(std::memory_order_relaxed) : 0;
    if (flush_queries && rasterizer) {
        constexpr GPUVAddr page_size = 0x1000;
        const GPUVAddr end = address + words.size_bytes();
        for (GPUVAddr begin = address; begin < end;) {
            const GPUVAddr page_end = (std::min)((begin & ~(page_size - 1)) + page_size, end);
            if (const auto device = memory_manager.GpuToCpuAddress(begin)) {
                rasterizer->FlushQueryResults(*device, page_end - begin);
            }
            begin = page_end;
        }
    }
    // Statistics: whether ReadBlock will download a GPU-modified range (and wait for the GPU).
    bool gpu_modified = false;
    if (stats_start != 0 && rasterizer) {
        if (const auto device = memory_manager.GpuToCpuAddress(address)) {
            gpu_modified = rasterizer->MustFlushRegion(*device, words.size_bytes());
        }
    }
    const u64 read_start = gpu_modified ? BackStats::Stamp() : 0;
    memory_manager.ReadBlock(address, words.data(), words.size_bytes());
    if (stats_start != 0) {
        const u64 now = BackStats::Stamp();
        BackStats::Add(BackStats::shared.macro_reads, 1);
        BackStats::Add(BackStats::shared.macro_read_words, words.size());
        BackStats::Add(BackStats::shared.macro_read_ticks, now - stats_start);
        if (gpu_modified) {
            BackStats::Add(BackStats::shared.macro_reads_gpu, 1);
            BackStats::Add(BackStats::shared.macro_read_gpu_ticks, now - read_start);
        }
    }
    if (trace) {
        const bool waited = Common::N2denTrace::query_flush_waits.load(std::memory_order_relaxed) != trace_waits;
        predicate_trace.OnRead(address, words, trace_dirty, waited);
    }
    if (occlusion_slots.Contains(address)) {
        Common::N2denOcclusion::macro_read_count = true;
    }
    static const bool draw_trace = std::getenv("EDEN_N2DEN_DRAW_TRACE") != nullptr;
    if (draw_trace && words.size() >= 4) {
        Common::N2denTrace::last_predicate_slot.store(address, std::memory_order_relaxed);
        Common::N2denTrace::last_predicate_count.store(words[0], std::memory_order_relaxed);
        Common::N2denTrace::last_predicate_stamp.store(u64{words[2]} | (u64{words[3]} << 32),
                                                       std::memory_order_relaxed);
    }
    return true;
}

void Maxwell3D::RefreshParametersImpl() {
    if (!Settings::IsGPULevelHigh()) {
        return;
    }
    size_t current_index = 0;
    for (auto& segment : macro_segments) {
        if (segment.first == 0) {
            current_index += segment.second;
            continue;
        }
        memory_manager.ReadBlock(segment.first, &macro_params[current_index],
                                 sizeof(u32) * segment.second);
        current_index += segment.second;
    }
}

u32 Maxwell3D::GetMaxCurrentVertices() {
    u32 num_vertices = 0;
    for (size_t index = 0; index < Regs::NumVertexArrays; ++index) {
        const auto& array = regs.vertex_streams[index];
        if (array.enable == 0) {
            continue;
        }
        const auto& attribute = regs.vertex_attrib_format[index];
        if (attribute.constant) {
            num_vertices = (std::max)(num_vertices, 1U);
            continue;
        }
        const auto& limit = regs.vertex_stream_limits[index];
        const GPUVAddr gpu_addr_begin = array.Address();
        const GPUVAddr gpu_addr_end = limit.Address() + 1;
        const u64 stream_size = GetNativeMacroEngine().native_turing
                                    ? regs.NativeVertexStreamSize(index)
                                    : gpu_addr_end - gpu_addr_begin;
        const u32 address_size = static_cast<u32>(std::min<u64>(stream_size, std::numeric_limits<u32>::max()));
        num_vertices = (std::max)(
            num_vertices, address_size / (std::max)(attribute.SizeInBytes(), array.stride.Value()));
        break;
    }
    return num_vertices;
}

size_t Maxwell3D::EstimateIndexBufferSize() {
    GPUVAddr start_address = regs.index_buffer.StartAddress();
    GPUVAddr end_address = regs.index_buffer.EndAddress();
    auto const byte_size = regs.index_buffer.FormatSizeInBytes();
    auto const max_size = 1ull << (byte_size * CHAR_BIT);
    auto const upper_cap = GetMaxCurrentVertices() * 4 * byte_size;
    auto const lower_cap = std::min<size_t>(size_t(end_address - start_address), upper_cap);
    return std::min<size_t>(memory_manager.GetMemoryLayoutSize(start_address, byte_size * max_size) / byte_size, lower_cap);
}

u32 Maxwell3D::ProcessShadowRam(u32 method, u32 argument) {
    // Keep track of the register value in shadow_state when requested.
    auto const c = shadow_state.shadow_ram_control;
    if (c == Regs::ShadowRamControl::Track || c == Regs::ShadowRamControl::TrackWithFilter)
        return shadow_state.reg_array[method] = argument;
    else if (c == Regs::ShadowRamControl::Replay)
        return shadow_state.reg_array[method];
    return argument;
}

void Maxwell3D::ConsumeSinkImpl(Core::System& system) {
    const auto control = shadow_state.shadow_ram_control;
    if (control == Regs::ShadowRamControl::Track || control == Regs::ShadowRamControl::TrackWithFilter) {
        for (auto [method, value] : method_sink) {
            shadow_state.reg_array[method] = value;
            ProcessDirtyRegisters(method, value);
        }
    } else if (control == Regs::ShadowRamControl::Replay) {
        for (auto [method, value] : method_sink)
            ProcessDirtyRegisters(method, shadow_state.reg_array[method]);
    } else {
        for (auto [method, value] : method_sink)
            ProcessDirtyRegisters(method, value);
    }
    method_sink.clear();
}

void Maxwell3D::ProcessDirtyRegisters(u32 method, u32 argument) {
    // The instances after the first are drawn together once their loop ends, and they must see
    // the state they were begun with: NVN's draw macros reset the base instance (and its copy in
    // c[0][0x644]) right after the loop, before the macro returns. (Eglantine PR #51.)
    if (draw_manager.HasDeferredDraw() && !DrawManager::IsInstanceLoopMethod(method)) [[unlikely]] {
        draw_manager.DrawDeferred(*this);
    }
    regs.reg_array[method] = argument;
    for (auto const& table : dirty.tables)
        dirty.flags[table[method]] = true;
}

void Maxwell3D::ProcessMethodCall(u32 method, u32 argument, u32 nonshadow_argument, bool is_last_call) {
    if (GetNativeMacroEngine().native_turing && method == MAXWELL3D_REG_INDEX(clear_surface)) {
        LOG_TRACE(HW_GPU, "Native GPU surface clear: value={:#x}", argument);
    }
    switch (method) {
    case NativeRootTables::SelectorMethod:
        if (GetNativeMacroEngine().native_turing) native_root_tables.Select(argument);
        return;
    case NativeRootTables::LoadMethod:
        if (GetNativeMacroEngine().native_turing && !native_root_tables.Load(argument)) {
            GetNativeMacroEngine().native_execution_failed = true;
            LOG_ERROR(HW_GPU, "Invalid native root-table upload");
        }
        return;
    case MAXWELL3D_REG_INDEX(wait_for_idle):
        return rasterizer->WaitForIdle();
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
        shadow_state.shadow_ram_control = static_cast<Regs::ShadowRamControl>(nonshadow_argument);
        return;
    // With macros on the split GPU thread's front (FrontWrite), the front loads and binds them.
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
        if (front_macros) return;
        return GetNativeMacroEngine().ClearCode(regs.load_mme.instruction_ptr);
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
        if (front_macros) return;
        return GetNativeMacroEngine().AddCode(regs.load_mme.instruction_ptr, argument);
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
        if (front_macros) {
            ++regs.load_mme.start_address_ptr;
            return;
        }
        return ProcessMacroBind(argument);
    case MAXWELL3D_REG_INDEX(falcon[4]):
        return ProcessFirmwareCall4();
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
        return ProcessCBData(argument);
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
        return ProcessCBBind(0);
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
        return ProcessCBBind(1);
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
        return ProcessCBBind(2);
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
        return ProcessCBBind(3);
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return ProcessCBBind(4);
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
        return ProcessQueryGet();
    case MAXWELL3D_REG_INDEX(render_enable.mode):
        return ProcessQueryCondition();
    case MAXWELL3D_REG_INDEX(clear_report_value):
        return ProcessCounterReset();
    case MAXWELL3D_REG_INDEX(sync_info):
        return ProcessSyncPoint();
    case MAXWELL3D_REG_INDEX(launch_dma):
        return upload_state.ProcessExec(regs.launch_dma.memory_layout.Value() == Regs::LaunchDMA::Layout::Pitch);
    case MAXWELL3D_REG_INDEX(inline_data):
        upload_state.ProcessData(argument, is_last_call);
        return;
    case MAXWELL3D_REG_INDEX(fragment_barrier):
        return rasterizer->FragmentBarrier();
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
        rasterizer->InvalidateGPUCache();
        return rasterizer->WaitForIdle();
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return rasterizer->TiledCacheBarrier();
    default:
        draw_manager.ProcessMethodCall(*this, method, argument);
        break;
    }
}

void Maxwell3D::CallMacroMethod(Core::System& system, u32 method, const std::vector<u32>& parameters) {
    // Reset the current macro.
    executing_macro = 0;

    // Lookup the macro offset
    const u32 entry =
        ((method - MacroRegistersStart) >> 1) % static_cast<u32>(GetNativeMacroPositions().size());

    // Execute the current macro.
    GetNativeMacroEngine().Execute(system, *this, GetNativeMacroPositions()[entry], parameters);
    draw_manager.DrawDeferred(*this);
}

void Maxwell3D::CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    // It is an error to write to a register other than the current macro's ARG register before
    // it has finished execution.
    if (executing_macro != 0) {
        ASSERT(method == executing_macro + 1);
    }

    // Methods after 0xE00 are special, they're actually triggers for some microcode that was
    // uploaded to the GPU during initialization.
    if (method >= MacroRegistersStart) {
        ProcessMacro(system, method, &method_argument, 1, is_last_call);
        return;
    }

    ASSERT(method < Regs::NUM_REGS && "Invalid Maxwell3D register, increase the size of the Regs structure");
    const u32 argument = ProcessShadowRam(method, method_argument);
    ProcessDirtyRegisters(method, argument);
    ProcessMethodCall(method, argument, method_argument, is_last_call);
}

void Maxwell3D::CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) {
    // Methods after 0xE00 are special, they're actually triggers for some microcode that was
    // uploaded to the GPU during initialization.
    if (method >= MacroRegistersStart) {
        ProcessMacro(system, method, base_start, amount, amount == methods_pending);
        return;
    }
    // As in ProcessDirtyRegisters: the paths below write without it.
    if (draw_manager.HasDeferredDraw() && !DrawManager::IsInstanceLoopMethod(method)) [[unlikely]] {
        draw_manager.DrawDeferred(*this);
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
        ProcessCBMultiData(base_start, amount);
        break;
    case MAXWELL3D_REG_INDEX(inline_data): {
        ASSERT(methods_pending == amount);
        upload_state.ProcessData(base_start, amount);
        return;
    }
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        if (shadow_state.shadow_ram_control != Regs::ShadowRamControl::Replay) {
            ProcessInlineIndexMultiData(method, base_start, amount);
            break;
        }
        [[fallthrough]];
    default:
        for (u32 i = 0; i < amount; i++) {
            CallMethod(system, method, base_start[i], methods_pending - i <= 1);
        }
        break;
    }
}

void Maxwell3D::ProcessMacroUpload(u32 data) {
    GetNativeMacroEngine().AddCode(regs.load_mme.instruction_ptr++, data);
}

void Maxwell3D::ProcessMacroBind(u32 data) {
    // A desynced pushbuffer can hand us a garbage index (Mario Kart World recorded 0x3a400040
    // against a 0x80 entry table); writing it would run past the table, so drop it.
    const std::span<u32> positions{GetNativeMacroPositions()};
    const u32 index{regs.load_mme.start_address_ptr++};
    if (index >= positions.size()) {
        static std::atomic<u32> rejected{};
        if (rejected.fetch_add(1) < 8) {
            LOG_ERROR(HW_GPU, "Macro bind index {:#x} is outside the {} entry table", index,
                      positions.size());
        }
        return;
    }
    positions[index] = data;
}

void Maxwell3D::ProcessFirmwareCall4() {
    LOG_DEBUG(HW_GPU, "(STUBBED) called");

    // Firmware call 4 is a blob that changes some registers depending on its parameters.
    // These registers don't affect emulation and so are stubbed by setting 0xd00 to 1.
    regs.shadow_scratch[0] = 1;
}

void Maxwell3D::StampQueryResult(Core::System& system, u64 payload, bool long_query) {
    const GPUVAddr sequence_address{regs.report_semaphore.Address()};
    if (long_query) {
        memory_manager.Write<u64>(sequence_address + sizeof(u64), system.GPU().GetTicks());
        memory_manager.Write<u64>(sequence_address, payload);
    } else {
        memory_manager.Write<u32>(sequence_address, static_cast<u32>(payload));
    }
}

void Maxwell3D::ProcessQueryGet() {
    VideoCommon::QueryPropertiesFlags flags{};
    if (regs.report_semaphore.query.short_query == 0) {
        flags |= VideoCommon::QueryPropertiesFlags::HasTimeout;
    }
    const GPUVAddr sequence_address{regs.report_semaphore.Address()};
    const VideoCommon::QueryType query_type =
        static_cast<VideoCommon::QueryType>(regs.report_semaphore.query.report.Value());
    if (query_type == VideoCommon::QueryType::ZPassPixelCount ||
        query_type == VideoCommon::QueryType::ZPassPixelCount64) {
        occlusion_slots.Add(sequence_address);
    }
    const u32 payload = regs.report_semaphore.payload;
    const u32 subreport = regs.report_semaphore.query.sub_report;
    switch (regs.report_semaphore.query.operation) {
    case Regs::ReportSemaphore::Operation::Release:
        if (regs.report_semaphore.query.short_query != 0) {
            flags |= VideoCommon::QueryPropertiesFlags::IsAFence;
        }
        VideoCommon::BackStats::NoteRelease(sequence_address, VideoCommon::BackStats::Release3D);
        rasterizer->Query(sequence_address, query_type, flags, payload, subreport);
        break;
    case Regs::ReportSemaphore::Operation::Acquire:
        // TODO(Blinkhawk): Under this operation, the GPU waits for the CPU to write a value that
        // matches the current payload.
        UNIMPLEMENTED_MSG("Unimplemented query operation ACQUIRE");
        break;
    case Regs::ReportSemaphore::Operation::ReportOnly:
        rasterizer->Query(sequence_address, query_type, flags, payload, subreport);
        break;
    case Regs::ReportSemaphore::Operation::Trap:
        UNIMPLEMENTED_MSG("Unimplemented query operation TRAP");
        break;
    default:
        UNIMPLEMENTED_MSG("Unknown query operation");
        break;
    }
}

void Maxwell3D::ProcessQueryCondition() {
    // [DEBUG] EDEN_N2DEN_TRACE_MACRO_READS=1 also logs render-enable updates.
    static const bool trace_condition = std::getenv("EDEN_N2DEN_TRACE_MACRO_READS") != nullptr;
    if (trace_condition) {
        static std::atomic<u64> updates{};
        const u64 n = ++updates;
        if (n <= 400 || n % 50000 == 0) {
            LOG_INFO(HW_GPU, "Render enable update n={}: override={} mode={} address={:#x}", n,
                     static_cast<u32>(regs.render_enable_override),
                     static_cast<u32>(regs.render_enable.mode), regs.render_enable.Address());
        }
    }
    // A render-enable False that a native macro derived from an occlusion count (marked by
    // Common::N2denOcclusion::MarkCondition): store the real mode and note where it came from.
    const u32 raw_mode = static_cast<u32>(regs.render_enable.mode);
    Common::N2denOcclusion::condition_from_count =
        (raw_mode & Common::N2denOcclusion::OcclusionFalseMarker) != 0;
    if (Common::N2denOcclusion::condition_from_count) {
        regs.render_enable.mode = static_cast<Regs::RenderEnable::Mode>(
            raw_mode & ~Common::N2denOcclusion::OcclusionFalseMarker);
    }
    host_conditional = rasterizer->AccelerateConditionalRendering();
    if (host_conditional) {
        execute_on = true;
        return;
    }
    const GPUVAddr condition_address{regs.render_enable.Address()};
    switch (regs.render_enable_override) {
    case Regs::RenderEnable::Override::AlwaysRender:
        execute_on = true;
        break;
    case Regs::RenderEnable::Override::NeverRender:
        execute_on = false;
        break;
    case Regs::RenderEnable::Override::UseRenderEnable: {
        switch (regs.render_enable.mode) {
        case Regs::RenderEnable::Mode::True: {
            execute_on = true;
            break;
        }
        case Regs::RenderEnable::Mode::False: {
            execute_on = false;
            break;
        }
        case Regs::RenderEnable::Mode::Conditional: {
            Regs::ReportSemaphore::Compare cmp;
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            execute_on = cmp.initial_sequence != 0U && cmp.initial_mode != 0U;
            break;
        }
        case Regs::RenderEnable::Mode::IfEqual: {
            Regs::ReportSemaphore::Compare cmp;
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            execute_on = cmp.initial_sequence == cmp.current_sequence &&
                         cmp.initial_mode == cmp.current_mode;
            break;
        }
        case Regs::RenderEnable::Mode::IfNotEqual: {
            Regs::ReportSemaphore::Compare cmp;
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            execute_on = cmp.initial_sequence != cmp.current_sequence ||
                         cmp.initial_mode != cmp.current_mode;
            break;
        }
        default: {
            UNIMPLEMENTED_MSG("Uninplemented Condition Mode!");
            execute_on = true;
            break;
        }
        }
        break;
    }
    }
}

void Maxwell3D::ProcessCounterReset() {
    const auto query_type = [clear_report = regs.clear_report_value]() {
        switch (clear_report) {
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::ZPassPixelCount:
            return VideoCommon::QueryType::ZPassPixelCount64;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::StreamingPrimitivesSucceeded:
            return VideoCommon::QueryType::StreamingPrimitivesSucceeded;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::PrimitivesGenerated:
            return VideoCommon::QueryType::PrimitivesGenerated;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::VtgPrimitivesOut:
            return VideoCommon::QueryType::VtgPrimitivesOut;
        default:
            LOG_DEBUG(HW_GPU, "Unimplemented counter reset={}", clear_report);
            return VideoCommon::QueryType::Payload;
        }
    }();
    rasterizer->ResetCounter(query_type);
}

void Maxwell3D::ProcessSyncPoint() {
    if (GetNativeMacroEngine().native_turing) {
        LOG_DEBUG(HW_GPU, "Native 3D syncpoint increment: id={}, raw={:#x}", regs.sync_info.sync_point.Value(), regs.reg_array[MAXWELL3D_REG_INDEX(sync_info)]);
    }
    const u32 sync_point = regs.sync_info.sync_point.Value();
    [[maybe_unused]] const u32 cache_flush = regs.sync_info.clean_l2.Value();
    rasterizer->SignalSyncPoint(sync_point);
}

void Maxwell3D::ProcessCBBind(size_t stage_index) {
    // Bind the buffer currently in CB_ADDRESS to the specified index in the desired shader
    // stage.
    const auto& bind_data = regs.bind_groups[stage_index];
    auto& buffer = state.shader_stages[stage_index].const_buffers[bind_data.shader_slot];
    buffer.enabled = bind_data.valid.Value() != 0;
    buffer.address = regs.const_buffer.Address();
    buffer.size = regs.const_buffer.size;

    const bool is_enabled = bind_data.valid.Value() != 0;
    if (!is_enabled) {
        rasterizer->DisableGraphicsUniformBuffer(stage_index, bind_data.shader_slot);
        return;
    }
    const GPUVAddr gpu_addr = regs.const_buffer.Address();
    const u32 size = regs.const_buffer.size;
    rasterizer->BindGraphicsUniformBuffer(stage_index, bind_data.shader_slot, gpu_addr, size);
}

void Maxwell3D::ProcessCBMultiData(const u32* start_base, u32 amount) {
    // Write the input value to the current const buffer at the current position.
    const GPUVAddr buffer_address = regs.const_buffer.Address();
    ASSERT(buffer_address != 0);

    // Don't allow writing past the end of the buffer.
    ASSERT(regs.const_buffer.offset <= regs.const_buffer.size);

    const GPUVAddr address{buffer_address + regs.const_buffer.offset};
    const size_t copy_size = amount * sizeof(u32);
    memory_manager.WriteBlockCached(address, start_base, copy_size);

    // Increment the current buffer position.
    regs.const_buffer.offset += static_cast<u32>(copy_size);
}

void Maxwell3D::ProcessCBData(u32 value) {
    ProcessCBMultiData(&value, 1);
}

void Maxwell3D::ProcessInlineIndexMultiData(u32 method, const u32* start_base, u32 amount) {
    if (amount == 0) {
        return;
    }
    const u32 argument = ProcessShadowRam(method, start_base[amount - 1]);
    ProcessDirtyRegisters(method, argument);
    draw_manager.SetInlineIndexBuffer(*this, method, start_base, amount);
}

Texture::TICEntry Maxwell3D::GetTICEntry(u32 tic_index) const {
    const GPUVAddr tic_address_gpu{regs.tex_header.Address() +
                                   tic_index * sizeof(Texture::TICEntry)};
    Texture::TICEntry tic_entry;
    memory_manager.ReadBlockUnsafe(tic_address_gpu, &tic_entry, sizeof(Texture::TICEntry));
    return tic_entry;
}

Texture::TSCEntry Maxwell3D::GetTSCEntry(u32 tsc_index) const {
    const GPUVAddr tsc_address_gpu{regs.tex_sampler.Address() +
                                   tsc_index * sizeof(Texture::TSCEntry)};
    Texture::TSCEntry tsc_entry;
    memory_manager.ReadBlockUnsafe(tsc_address_gpu, &tsc_entry, sizeof(Texture::TSCEntry));
    return tsc_entry;
}

u32 Maxwell3D::GetRegisterValue(u32 method) const {
    ASSERT(method < Regs::NUM_REGS && "Invalid Maxwell3D register");
    return regs.reg_array[method];
}

void Maxwell3D::SetHLEReplacementAttributeType(u32 bank, u32 offset,
                                               HLEReplacementAttributeType name) {
    const u64 key = (static_cast<u64>(bank) << 32) | offset;
    replace_table.emplace(key, name);
}

namespace {

// Executable methods ProcessMethodCall handles itself; the rest go to the draw manager.
bool FrontHandledOutsideDrawManager(u32 method) {
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method >= cb_data && method < cb_data + 16) {
        return true;
    }
    switch (method) {
    case NativeRootTables::SelectorMethod:
    case NativeRootTables::LoadMethod:
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(sync_info):
    case MAXWELL3D_REG_INDEX(launch_dma):
    case MAXWELL3D_REG_INDEX(inline_data):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return true;
    default:
        return false;
    }
}

} // Anonymous namespace

Maxwell3D::FrontState& Maxwell3D::Front(Core::System& system, VideoCommon::GpuSplit& split) {
    if (!front) [[unlikely]] {
        // Copy the back's state and take the macros over while it is idle (state writes it
        // still holds in its method sink first).
        split.Drain(VideoCommon::BackStats::DrainFront3D);
        ConsumeSink(system);
        front = std::make_unique<FrontState>();
        std::memcpy(front->regs.data(), regs.reg_array.data(), sizeof(front->regs));
        std::memcpy(front->shadow.data(), shadow_state.reg_array.data(), sizeof(front->shadow));
        front->control = shadow_state.shadow_ram_control;
        front->executing_macro = executing_macro;
        front->parameters = std::move(macro_params);
        macro_params.clear();
        macro_segments.clear();
        executing_macro = 0;
        front_macros = true;
        split.OnStop([this] { LeaveFront(); });
    }
    return *front;
}

void Maxwell3D::LeaveFront() {
    if (!front) {
        return;
    }
    // The back thread is gone: hand an unfinished macro call back to ProcessMacro.
    executing_macro = front->executing_macro;
    macro_params = std::move(front->parameters);
    macro_segments.clear();
    if (!macro_params.empty()) {
        macro_segments.emplace_back(GPUVAddr{0}, macro_params.size());
    }
    front.reset();
    front_macros = false;
}

void Maxwell3D::FrontWrite(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                           u32 argument, bool is_last_call, bool from_macro) {
    if (method >= Regs::NUM_REGS) [[unlikely]] {
        return;
    }
    // As ProcessShadowRam does on the back.
    u32 value = argument;
    bool shadow_kept = true;
    if (fs.control == Regs::ShadowRamControl::Track ||
        fs.control == Regs::ShadowRamControl::TrackWithFilter) {
        shadow_kept = fs.shadow[method] == argument;
        fs.shadow[method] = argument;
    } else if (fs.control == Regs::ShadowRamControl::Replay) {
        value = fs.shadow[method];
    }
    const bool instance_loop = DrawManager::IsInstanceLoopMethod(method);
    if (!execution_mask[method]) {
        // Nothing would change on the back: the register holds the value already and the shadow
        // RAM (tracked by default) too. Only dirty flags would be set, for state that is equal.
        if (fs.regs[method] == value && shadow_kept && !fs.instance_pending) {
            return;
        }
        fs.regs[method] = value;
        if (!instance_loop) {
            fs.instance_pending = false; // ProcessDirtyRegisters issues the deferred draw
        }
    } else {
        // Executable methods outside the instance loop issue a deferred draw on the back too
        // (ProcessDirtyRegisters); the loop's own methods may add instances to one.
        fs.instance_pending = instance_loop;
        fs.regs[method] = value;
        constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
        switch (method) {
        case MAXWELL3D_REG_INDEX(shadow_ram_control):
            // The back's mode is shadow_state.shadow_ram_control: the shadow copy of this very
            // register, which a later Replay write reads back.
            fs.control = static_cast<Regs::ShadowRamControl>(argument);
            fs.shadow[method] = argument;
            break;
        case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
            GetNativeMacroEngine().ClearCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)]);
            break;
        case MAXWELL3D_REG_INDEX(load_mme.instruction):
            GetNativeMacroEngine().AddCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)], value);
            break;
        case MAXWELL3D_REG_INDEX(load_mme.start_address): {
            const auto positions = GetNativeMacroPositions();
            u32& pointer = fs.regs[MAXWELL3D_REG_INDEX(load_mme.start_address_ptr)];
            if (pointer < positions.size()) {
                positions[pointer] = value;
            }
            ++pointer;
            break;
        }
        case MAXWELL3D_REG_INDEX(falcon[4]):
            fs.regs[MAXWELL3D_REG_INDEX(shadow_scratch)] = 1;
            break;
        default:
            if (method >= cb_data && method < cb_data + 16) {
                fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)] += sizeof(u32);
            } else if (!FrontHandledOutsideDrawManager(method)) {
                fs.draw_pending = true;
            }
            break;
        }
    }
    if (from_macro) {
        fs.macro_wrote = true;
        split.MacroWrite(this, method, argument);
    } else {
        split.Method(this, method, argument, is_last_call, nullptr);
    }
    if (method == MAXWELL3D_REG_INDEX(report_semaphore.query)) {
        FrontNoteReport(split, fs);
    }
}

void Maxwell3D::FrontNoteReport(VideoCommon::GpuSplit& split, FrontState& fs) {
    const u64 high = fs.regs[MAXWELL3D_REG_INDEX(report_semaphore.address_high)];
    const u64 low = fs.regs[MAXWELL3D_REG_INDEX(report_semaphore.address_low)];
    split.NoteReportWrite((high << 32) | low);
}

void Maxwell3D::FrontCallMethod(Core::System& system, VideoCommon::GpuSplit& split, u32 method,
                                u32 argument, bool is_last_call) {
    auto& fs = Front(system, split);
    if (method >= MacroRegistersStart) {
        if (fs.executing_macro == 0) {
            fs.executing_macro = method;
        }
        fs.parameters.push_back(argument);
        if (is_last_call) {
            FrontRunMacro(system, split, fs);
        }
        return;
    }
    FrontWrite(split, fs, method, argument, is_last_call, false);
}

void Maxwell3D::FrontCallMultiMethod(Core::System& system, VideoCommon::GpuSplit& split,
                                     u32 method, const u32* base_start, u32 amount,
                                     u32 methods_pending) {
    auto& fs = Front(system, split);
    if (method >= MacroRegistersStart) {
        if (fs.executing_macro == 0) {
            fs.executing_macro = method;
        }
        fs.parameters.insert(fs.parameters.end(), base_start, base_start + amount);
        if (amount == methods_pending) {
            FrontRunMacro(system, split, fs);
        }
        return;
    }
    if (method >= Regs::NUM_REGS) [[unlikely]] {
        split.MultiMethod(this, method, base_start, amount, methods_pending, nullptr);
        return;
    }
    // CallMultiMethod issues a deferred draw first for methods outside the instance loop; the
    // loop's methods may add instances to one.
    fs.instance_pending = DrawManager::IsInstanceLoopMethod(method);
    // What CallMultiMethod does to the registers on the back, then one entry for all the words.
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    const auto tracked = [&fs] {
        return fs.control == Regs::ShadowRamControl::Track ||
               fs.control == Regs::ShadowRamControl::TrackWithFilter;
    };
    if (method >= cb_data && method < cb_data + 16) {
        fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)] += amount * sizeof(u32);
    } else if (method == MAXWELL3D_REG_INDEX(inline_data)) {
        // Upload data only.
    } else if ((method == MAXWELL3D_REG_INDEX(draw_inline_index) ||
                method == MAXWELL3D_REG_INDEX(inline_index_2x16.even) ||
                method == MAXWELL3D_REG_INDEX(inline_index_4x8.index0)) &&
               fs.control != Regs::ShadowRamControl::Replay) {
        if (amount != 0) {
            const u32 last = base_start[amount - 1];
            if (tracked()) {
                fs.shadow[method] = last;
            }
            fs.regs[method] = last;
        }
        fs.draw_pending = true;
    } else {
        // CallMethod for each word, macro uploads among them (load_mme.instruction). A word may
        // change the shadow RAM mode for the next (a run of SET_SHADOW_RAM_CONTROL writes).
        for (u32 i = 0; i < amount; ++i) {
            const u32 argument = base_start[i];
            u32 value = argument;
            if (tracked()) {
                fs.shadow[method] = argument;
            } else if (fs.control == Regs::ShadowRamControl::Replay) {
                value = fs.shadow[method];
            }
            fs.regs[method] = value;
            if (!execution_mask[method]) {
                continue;
            }
            switch (method) {
            case MAXWELL3D_REG_INDEX(shadow_ram_control):
                fs.control = static_cast<Regs::ShadowRamControl>(argument);
                fs.shadow[method] = argument; // The same word on the back (see FrontWrite)
                break;
            case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
                GetNativeMacroEngine().ClearCode(value);
                break;
            case MAXWELL3D_REG_INDEX(load_mme.instruction):
                GetNativeMacroEngine().AddCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)], value);
                break;
            case MAXWELL3D_REG_INDEX(load_mme.start_address): {
                const auto positions = GetNativeMacroPositions();
                u32& pointer = fs.regs[MAXWELL3D_REG_INDEX(load_mme.start_address_ptr)];
                if (pointer < positions.size()) {
                    positions[pointer] = value;
                }
                ++pointer;
                break;
            }
            case MAXWELL3D_REG_INDEX(falcon[4]):
                fs.regs[MAXWELL3D_REG_INDEX(shadow_scratch)] = 1;
                break;
            default:
                if (!FrontHandledOutsideDrawManager(method)) {
                    fs.draw_pending = true;
                }
                break;
            }
        }
    }
    split.MultiMethod(this, method, base_start, amount, methods_pending, nullptr);
    if (method == MAXWELL3D_REG_INDEX(report_semaphore.query)) {
        FrontNoteReport(split, fs);
    }
}

bool Maxwell3D::FrontReadReportSlot(VideoCommon::GpuSplit& split, GPUVAddr address,
                                    std::span<u32> words) {
    namespace BackStats = VideoCommon::BackStats;
    const u64 size = words.size_bytes();
    if ((address & 3) || size == 0 || size > 64 || rasterizer == nullptr) {
        return false;
    }
    const auto declined = [](size_t reason) {
        if (BackStats::Enabled()) {
            BackStats::Add(BackStats::shared.macro_read_declined[reason], 1);
        }
        return false;
    };
    const auto target = split.ReportTarget(address, size);
    if (!target) {
        return declined(0);
    }
    const auto device = memory_manager.GpuToCpuAddress(address);
    const auto device_last = memory_manager.GpuToCpuAddress(address + size - 1);
    if (!device || !device_last || *device_last != *device + size - 1) {
        return declined(1);
    }
    const u64 start = BackStats::Enabled() ? BackStats::Stamp() : 0;
    const bool waited = split.WaitForPosition(*target);
    // Every report sent for the slot ran on the back; one still waiting for its host result is
    // counted until its value is in guest memory. Draws that write the slot as a storage buffer
    // are not tracked: words the buffer cache holds newer than memory send the read the old way.
    if (VideoCommon::PendingReports::Any(*device, size)) {
        return declined(2);
    }
    if (rasterizer->MustFlushRegion(*device, size,
                                    VideoCommon::CacheType::BufferCache |
                                        VideoCommon::CacheType::TextureCache)) {
        return declined(3);
    }
    memory_manager.ReadBlockUnsafe(address, words.data(), size);
    if (start != 0) {
        BackStats::Add(BackStats::shared.macro_reads_fast, 1);
        BackStats::Add(BackStats::shared.macro_read_fast_waits, waited ? 1 : 0);
        BackStats::Add(BackStats::shared.macro_read_fast_ticks, BackStats::Stamp() - start);
    }
    if (occlusion_slots.Contains(address)) {
        Common::N2denOcclusion::macro_read_count = true;
    }
    if (PredicateTrace::Enabled()) {
        predicate_trace.OnRead(address, words, false, false);
    }
    return true;
}

void Maxwell3D::FrontRunMacro(Core::System& system, VideoCommon::GpuSplit& split,
                              FrontState& fs) {
    const u32 method = fs.executing_macro;
    fs.executing_macro = 0;
    auto& engine = GetNativeMacroEngine();
    const auto positions = GetNativeMacroPositions();
    const u32 start = positions[((method - MacroRegistersStart) >> 1) % positions.size()];
    const auto read = [&fs](u32 address) {
        return address < Regs::NUM_REGS ? fs.regs[address] : 0U;
    };
    const auto emit = [&](u32 address, u32 value) {
        if (address >= MacroRegistersStart) {
            return false;
        }
        FrontWrite(split, fs, address,
                   Common::N2denOcclusion::MarkCondition(
                       MAXWELL3D_REG_INDEX(render_enable.mode), address, value),
                   true, true);
        return true;
    };
    Common::N2denOcclusion::macro_read_count = false;
    // FIFO DMA reads (indirect draw arguments a culling dispatch wrote, ...) read memory as of
    // their place in the stream: the back executes everything before them first, and the caches
    // that may hold newer words are its state (idle from here until the front sends more).
    const TuringMacroInterpreter::MemoryRead memory_read = [&](u64 address, std::span<u32> words) {
        // [TEST] EDEN_N2DEN_MACRO_READ_WAIT=1: a read of words a query report may still write
        // (OoT's predicate macro 0x142 reads occlusion counts) waits for the GPU's fences on the
        // back first, so the report holds this frame's count. Jonas (jojo-chaechae) traced the
        // stale predicate reads (Eglantine oot-jojo 50db84983f).
        static const bool wait_queries = std::getenv("EDEN_N2DEN_MACRO_READ_WAIT") != nullptr;
        if (wait_queries) {
            split.Defer([memory = &memory_manager, rasterizer_ = rasterizer, address,
                         bytes = words.size_bytes()] {
                if (memory->IsMemoryDirty(address, bytes, VideoCommon::CacheType::QueryCache)) {
                    rasterizer_->ReleaseFences(true);
                }
            });
        }
        // Report slots (OoT's predicate macro reads an occlusion report per object, ~7800 times
        // a second) are read without draining the split when nothing the back still has to
        // run can change them. EDEN_N2DEN_MACRO_READ_FAST
        // =0 always drains; =verify reads both ways and counts reads whose words differ.
        static const int fast_mode = [] {
            const char* const value = std::getenv("EDEN_N2DEN_MACRO_READ_FAST");
            if (value && value[0] == '0') return 0;
            if (value && std::string_view{value} == "verify") return 2;
            return 1;
        }();
        std::array<u32, 16> fast_words{};
        const bool fast = fast_mode != 0 && !wait_queries && words.size() <= fast_words.size() &&
                          FrontReadReportSlot(split, address, std::span{fast_words}.first(words.size()));
        // [DEBUG] EDEN_N2DEN_TRACE_MACRO_READS=1 also logs the first 300 reads of macros other
        // than 0x143 (query reads), on the fast path as well.
        static const bool trace_other = std::getenv("EDEN_N2DEN_TRACE_MACRO_READS") != nullptr;
        static std::atomic<u32> other_logged{};
        const auto trace_other_read = [&](std::span<const u32> got, const char* path) {
            if (trace_other && start != 0x143 && other_logged.fetch_add(1) < 300) {
                LOG_INFO(HW_GPU, "Macro DMA read ({}) macro={:#x} at {:#x} words={} [{:08x}]", path,
                         start, address, got.size(),
                         fmt::join(got.first(std::min<size_t>(got.size(), 8)), " "));
            }
        };
        if (fast && fast_mode == 1) {
            std::ranges::copy(std::span{fast_words}.first(words.size()), words.begin());
            trace_other_read(words, "fast");
            return true;
        }
        split.Drain(VideoCommon::BackStats::DrainMacroMemory);
        const bool ok = ReadMacroMemory(address, words);
        if (fast && ok && !std::ranges::equal(std::span{fast_words}.first(words.size()), words)) {
            static std::atomic<u32> mismatches_logged{};
            VideoCommon::BackStats::Add(VideoCommon::BackStats::shared.macro_read_mismatches, 1);
            if (mismatches_logged.fetch_add(1, std::memory_order_relaxed) < 32) {
                LOG_ERROR(HW_GPU, "Macro read without drain differs at {:#x}: [{:08x}] then [{:08x}]",
                          address, fmt::join(std::span{fast_words}.first(words.size()), " "),
                          fmt::join(words, " "));
            }
        }
        // [DEBUG] EDEN_N2DEN_TRACE_MACRO_READS=1: log macro DMA reads (predicate records, ...).
        static const bool trace_reads = std::getenv("EDEN_N2DEN_TRACE_MACRO_READS") != nullptr;
        if (ok) {
            trace_other_read(words, "drained");
        }
        if (trace_reads && ok) {
            static std::atomic<u64> reads{};
            const u64 n = ++reads;
            if (n <= 400 || n % 50000 == 0) {
                LOG_INFO(HW_GPU, "Macro DMA read n={} macro={:#x} at {:#x} words={} [{:08x}]", n,
                         start, address, words.size(),
                         fmt::join(words.first(std::min<size_t>(words.size(), 8)), " "));
            }
        }
        return ok;
    };
    auto& dram = native_interpreter.dram;
    const auto result = [&] {
#ifdef ARCHITECTURE_x86_64
        if (!engine.native_jit) {
            engine.native_jit = std::make_shared<TuringMacroJit>();
        }
        if (engine.native_jit->Prepare(engine.native_code, engine.native_code_generation)) {
            return engine.native_jit->Execute(start, fs.parameters, std::span<u32>{dram}, read,
                                              emit, memory_read);
        }
#endif
        return native_interpreter.ExecuteDecoded(engine.native_decoded_code, start, fs.parameters,
                                                 read, emit, memory_read);
    }();
    const size_t consumed = fs.parameters.size();
    fs.parameters.clear();
    // The end of the macro on the back, as CallMacroMethod and DmaPusher have it on one thread:
    // a failed macro sets the failure flag, which skips the deferred draw it ends with; the
    // deferred draw; then the failure flags clear (a failure its writes caused on the back,
    // e.g. a bad root table upload, also lasts until here).
    if (!result.success) {
        LOG_ERROR(HW_GPU, "Native MME (split front) failed: start={:#x}, pc={:#x}, consumed={}/{}, error={}",
                  start, result.pc, result.consumed, consumed, result.error);
        fs.draw_pending = false;
        fs.instance_pending = false;
        fs.macro_wrote = false;
        split.Defer([this, system_ptr = &system] {
            ConsumeSink(*system_ptr);
            GetNativeMacroEngine().native_execution_failed = true;
            draw_manager.DrawDeferred(*this);
            VideoCommon::GpuSplit::ClearNativeFailures(this);
        });
    } else if (fs.draw_pending) {
        fs.draw_pending = false;
        fs.instance_pending = false; // FinishFrontMacro issues the deferred draw
        fs.macro_wrote = false;
        split.Defer([this, system_ptr = &system] {
            FinishFrontMacro(*system_ptr);
            VideoCommon::GpuSplit::ClearNativeFailures(this);
        });
    } else if (fs.macro_wrote) {
        fs.macro_wrote = false;
        split.Defer([this] { VideoCommon::GpuSplit::ClearNativeFailures(this); });
    }
}

void Maxwell3D::FinishFrontMacro(Core::System& system) {
    // What ProcessMacro and CallMacroMethod do around the macro.
    ConsumeSink(system);
    draw_manager.DrawDeferred(*this);
}

} // namespace Tegra::Engines
