// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <unordered_set>
#include <algorithm>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string_view>
#include <thread>

#include "common/fs/path_util.h"
#include <map>
#include "common/range_sets.inc"
#include "shader_recompiler/exception.h"
#include "video_core/gpu_written_pages.h"
#include "video_core/buffer_cache/storage_range_groups.h"
#include "video_core/buffer_cache/storage_buffer_size.h"
#include "video_core/back_thread_stats.h"
#include "video_core/sync_counters.h"
#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/guest_memory.h"
#include "video_core/frame_snapshot.h"
#include "video_core/late_write_trace.h"
#include "video_core/upload_stats.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/texture_cache/util.h"

namespace VideoCommon {

using Core::DEVICE_PAGESIZE;

template <class P>
BufferCache<P>::BufferCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, Runtime& runtime_)
    : runtime{runtime_}, device_memory{device_memory_}, memory_tracker{device_memory} {
    // Ensure the first slot is used for the null buffer
    void(slot_buffers.insert(runtime, NullBufferParams{}));
    ++mark_generation; // the marks MarkWrittenBuffer skips may be gone
    gpu_modified_ranges.Clear();
    // Asked ~850,000 times a second in Fortune's Weave's crowds (IsRegionGpuModified), 87% of the
    // time about memory the GPU never wrote: a bit per 4 KiB page answers those without walking the
    // range tree. Exact: a page is unmarked only after a lookup finds it empty.
    // EDEN_NO_PAGE_FILTER=1 turns it off, for comparisons.
    if (std::getenv("EDEN_NO_PAGE_FILTER") == nullptr) {
        gpu_modified_ranges.EnablePageFilter(
            static_cast<u32>(Tegra::MaxwellDeviceTraits::device_virtual_bits),
            static_cast<u32>(Core::DEVICE_PAGEBITS));
    }
    inline_buffer_id = NULL_BUFFER_ID;
#ifdef YUZU_LEGACY
    immediately_free = (Settings::values.vram_usage_mode.GetValue() == Settings::VramUsageMode::Aggressive);
#endif
    if (!runtime.CanReportMemoryUsage()) {
        minimum_memory = DEFAULT_EXPECTED_MEMORY;
        critical_memory = DEFAULT_CRITICAL_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(runtime.GetDeviceLocalMemory());
    const s64 min_spacing_expected = device_local_memory - 1_GiB;
    const s64 min_spacing_critical = device_local_memory - 512_MiB;
    const s64 mem_threshold = (std::min)(device_local_memory, TARGET_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    minimum_memory = static_cast<u64>(
        (std::max)((std::min)(device_local_memory - min_vacancy_expected, min_spacing_expected),
                 DEFAULT_EXPECTED_MEMORY));
    critical_memory = static_cast<u64>(
        (std::max)((std::min)(device_local_memory - min_vacancy_critical, min_spacing_critical),
                 DEFAULT_CRITICAL_MEMORY));
}

template <class P>
BufferCache<P>::~BufferCache() = default;

template <class P>
void BufferCache<P>::RunGarbageCollector() {
    const bool aggressive_gc = total_used_memory >= critical_memory;
    const u64 ticks_to_destroy = aggressive_gc ? 60 : 120;
    int num_iterations = aggressive_gc ? 64 : 32;
    const auto clean_up = [this, &num_iterations](BufferId buffer_id) {
        if (num_iterations == 0) {
            return true;
        }
        --num_iterations;
        auto& buffer = slot_buffers[buffer_id];
        DownloadBufferMemory(buffer);
        DeleteBuffer(buffer_id);
        return false;
    };
    lru_cache.ForEachItemBelow(frame_tick - ticks_to_destroy, clean_up);
}

template <class P>
void BufferCache<P>::NoteDrawBindings(bool is_indexed, bool table_span) {
    using namespace VideoCommon::SyncCounters;
    if (!Enabled()) {
        return;
    }
    struct Hasher {
        u64 h = 0xcbf29ce484222325ULL;
        void operator()(u64 v) {
            h ^= v;
            h *= 0x100000001b3ULL;
            h ^= h >> 29;
        }
    };
    // Variant 0 full, 1 without uniform addresses, 2 also without written storage addresses.
    std::array<Hasher, 3> hs{};
    const auto all = [&](u64 v) { for (auto& h : hs) h(v); };
    const auto& cs = *channel_state;
    const bool native = maxwell3d->GetNativeMacroEngine().native_turing;
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        all(0x1000 + stage);
        all(cs.enabled_uniform_buffer_masks[stage]);
        ForEachEnabledBit(cs.enabled_uniform_buffer_masks[stage], [&](u32 index) {
            if (native && Tegra::NativeRootTables::IsBank(index)) return;
            const Binding& b = cs.uniform_buffers[stage][index];
            hs[0](b.device_addr);
            hs[0](b.size);
        });
        all(cs.enabled_storage_buffers[stage]);
        all(cs.written_storage_buffers[stage]);
        ForEachEnabledBit(cs.enabled_storage_buffers[stage], [&](u32 index) {
            const Binding& b = cs.storage_buffers[stage][index];
            const bool written = ((cs.written_storage_buffers[stage] >> index) & 1) != 0;
            hs[0](b.device_addr); hs[0](b.gpu_addr); hs[0](b.size); hs[0](b.write_size); hs[0](b.pointer_offset);
            hs[1](b.device_addr); hs[1](b.gpu_addr); hs[1](b.size); hs[1](b.write_size); hs[1](b.pointer_offset);
            if (!written) {
                hs[2](b.device_addr); hs[2](b.gpu_addr); hs[2](b.size);
            }
        });
        all(cs.enabled_texture_buffers[stage]);
        ForEachEnabledBit(cs.enabled_texture_buffers[stage], [&](u32 index) {
            const auto& b = cs.texture_buffers[stage][index];
            all(b.device_addr);
            all(b.size);
            all(static_cast<u64>(b.format));
        });
    }
    all(enabled_vertex_buffers_mask);
    ForEachEnabledBit(enabled_vertex_buffers_mask, [&](u32 index) {
        all(v_buffer[index].device_addr);
        all(v_buffer[index].size);
    });
    all(is_indexed);
    if (is_indexed) {
        all(cs.index_buffer.device_addr);
        all(cs.index_buffer.size);
    }
    struct State {
        u64 prev = 0;
        std::array<u64, 32> recent{};
        size_t next = 0;
        std::unordered_set<u64> frame;
    };
    static std::array<State, 3> states;
    static u64 last_frame = ~u64{0};
    const bool new_frame = last_frame != frame_tick;
    last_frame = frame_tick;
    static constexpr std::array<std::array<Counter, 3>, 3> ids{{
        {BindSamePrevFull, BindRecentFull, BindDistinctFull},
        {BindSamePrevNoUni, BindRecentNoUni, BindDistinctNoUni},
        {BindSamePrevNoWr, BindRecentNoWr, BindDistinctNoWr},
    }};
    Add(BindDraws);
    if (native) Add(BindNative);
    if constexpr (requires { runtime.UseGroupedStorage(); }) {
        if (runtime.UseGroupedStorage()) Add(BindGrouped);
    }
    if (table_span) Add(BindTableSpan);
    if (current_draw_indirect) Add(BindIndirect);
    // Per component (uniform addresses, storage, vertex, index) and per slot: repeats within
    // the recent draws, logged every 256k draws.
    {
        std::array<Hasher, 5> comp{};
        struct SlotStats {
            std::array<u64, 64> recent{};
            size_t next = 0;
            u64 slots = 0, hits = 0;
            bool Note(u64 key) {
                ++slots;
                if (std::find(recent.begin(), recent.end(), key) != recent.end()) {
                    ++hits;
                    return true;
                }
                recent[next++ % recent.size()] = key;
                return false;
            }
        };
        static std::array<SlotStats, 4> slot_stats; // uniform, storage read, storage written, vertex+index
        static std::array<SlotStats, 5> comp_stats;
        static std::array<u64, 5> comp_prev{};
        static std::array<u64, 5> comp_same{};
        static u64 draws = 0;
        const auto mix = [](u64 a, u64 b) { return (a * 0x9E3779B97F4A7C15ULL) ^ (b + (a >> 17)); };
        for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
            ForEachEnabledBit(cs.enabled_uniform_buffer_masks[stage], [&](u32 index) {
                if (native && Tegra::NativeRootTables::IsBank(index)) return;
                const Binding& b = cs.uniform_buffers[stage][index];
                comp[0](b.device_addr); comp[0](b.size);
                slot_stats[0].Note(mix(b.device_addr, b.size));
            });
            ForEachEnabledBit(cs.enabled_storage_buffers[stage], [&](u32 index) {
                const Binding& b = cs.storage_buffers[stage][index];
                const bool written = ((cs.written_storage_buffers[stage] >> index) & 1) != 0;
                comp[written ? 4 : 1](b.gpu_addr); comp[written ? 4 : 1](b.size);
                slot_stats[written ? 2 : 1].Note(mix(b.gpu_addr, mix(b.size, b.write_size)));
            });
        }
        ForEachEnabledBit(enabled_vertex_buffers_mask, [&](u32 index) {
            comp[2](v_buffer[index].device_addr); comp[2](v_buffer[index].size);
            slot_stats[3].Note(mix(v_buffer[index].device_addr, v_buffer[index].size));
        });
        if (is_indexed) {
            comp[3](cs.index_buffer.device_addr); comp[3](cs.index_buffer.size);
            slot_stats[3].Note(mix(cs.index_buffer.device_addr, cs.index_buffer.size));
        }
        for (size_t i = 0; i < 5; ++i) {
            comp_stats[i].Note(comp[i].h);
            comp_same[i] += comp[i].h == comp_prev[i];
            comp_prev[i] = comp[i].h;
        }
        if (++draws % (1u << 18) == 0) {
            const auto pc = [](u64 a, u64 b) { return b ? 100.0 * double(a) / double(b) : 0.0; };
            LOG_INFO(HW_GPU,
                     "Binding repeats per component (same-prev / recent64): uniform addrs "
                     "{:.1f}/{:.1f}%, storage read {:.1f}/{:.1f}%, storage written {:.1f}/{:.1f}%, "
                     "vertex {:.1f}/{:.1f}%, index {:.1f}/{:.1f}%; per slot recent64: uniform "
                     "{:.1f}% ({:.2f}/draw), storage read {:.1f}% ({:.2f}/draw), storage written "
                     "{:.1f}% ({:.2f}/draw), vertex+index {:.1f}% ({:.2f}/draw)",
                     pc(comp_same[0], draws), pc(comp_stats[0].hits, draws), pc(comp_same[1], draws),
                     pc(comp_stats[1].hits, draws), pc(comp_same[4], draws), pc(comp_stats[4].hits, draws),
                     pc(comp_same[2], draws), pc(comp_stats[2].hits, draws), pc(comp_same[3], draws),
                     pc(comp_stats[3].hits, draws), pc(slot_stats[0].hits, slot_stats[0].slots),
                     double(slot_stats[0].slots) / double(draws), pc(slot_stats[1].hits, slot_stats[1].slots),
                     double(slot_stats[1].slots) / double(draws), pc(slot_stats[2].hits, slot_stats[2].slots),
                     double(slot_stats[2].slots) / double(draws), pc(slot_stats[3].hits, slot_stats[3].slots),
                     double(slot_stats[3].slots) / double(draws));
            draws = 0;
            comp_same = {};
            for (auto& s : comp_stats) { s.hits = s.slots = 0; }
            for (auto& s : slot_stats) { s.hits = s.slots = 0; }
        }
    }
    for (size_t v = 0; v < 3; ++v) {
        State& st = states[v];
        const u64 h = hs[v].h;
        if (new_frame) st.frame.clear();
        if (h == st.prev) Add(ids[v][0]);
        if (std::find(st.recent.begin(), st.recent.end(), h) != st.recent.end()) Add(ids[v][1]);
        else st.recent[st.next++ % st.recent.size()] = h;
        if (st.frame.insert(h).second) Add(ids[v][2]);
        st.prev = h;
    }
}

template <class P>
void BufferCache<P>::TickFrame() {
    // Homebrew console apps don't create or bind any channels, so this will be nullptr.
    if (!channel_state) {
        return;
    }
    runtime.TickFrame(slot_buffers);

    // Calculate hits and shots and move hit bits to the right
    const u32 hits = std::reduce(channel_state->uniform_cache_hits.begin(),
                                 channel_state->uniform_cache_hits.end());
    const u32 shots = std::reduce(channel_state->uniform_cache_shots.begin(),
                                  channel_state->uniform_cache_shots.end());
    std::copy_n(channel_state->uniform_cache_hits.begin(),
                channel_state->uniform_cache_hits.size() - 1,
                channel_state->uniform_cache_hits.begin() + 1);
    std::copy_n(channel_state->uniform_cache_shots.begin(),
                channel_state->uniform_cache_shots.size() - 1,
                channel_state->uniform_cache_shots.begin() + 1);
    channel_state->uniform_cache_hits[0] = 0;
    channel_state->uniform_cache_shots[0] = 0;

    const bool skip_preferred = hits * 256 < shots * 251;
    channel_state->uniform_buffer_skip_cache_size = skip_preferred ? DEFAULT_SKIP_CACHE_SIZE : 0;

    // If we can obtain the memory info, use it instead of the estimate.
    if (runtime.CanReportMemoryUsage()) {
        total_used_memory = runtime.GetDeviceMemoryUsage();
    }
    if (total_used_memory >= minimum_memory) {
        RunGarbageCollector();
    }
    ++frame_tick;
    delayed_destruction_ring.Tick();

    for (auto& buffer : async_buffers_death_ring) {
        runtime.FreeDeferredStagingBuffer(buffer);
    }
    async_buffers_death_ring.clear();
}

template <class P>
void BufferCache<P>::UnmapGPUMemory(size_t as_id, GPUVAddr gpu_addr, size_t size) {
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        virtual_ranges.Unmap(as_id, gpu_addr, size);
    }
}

template <class P>
void BufferCache<P>::WriteMemory(DAddr device_addr, u64 size) {
    Nx2NoteListHostWrite(device_addr, size);
    if (IsRegionGpuModified(device_addr, size)) {
        ClearDownload(device_addr, size);
        ++mark_generation; // the marks MarkWrittenBuffer skips may be gone
        gpu_modified_ranges.Subtract(device_addr, size);
    }
    // Guest CPU writes come ~100,000 times a second: forget only the marks they overlap.
    for (RecentMark& mark : recent_marks) {
        if (mark.address < device_addr + size && device_addr < mark.address + mark.size) {
            mark.generation = 0;
        }
    }
    memory_tracker.MarkRegionAsCpuModified(device_addr, size);
}

template <class P>
void BufferCache<P>::CachedWriteMemory(DAddr device_addr, u64 size) {
    const bool is_dirty = IsRegionRegistered(device_addr, size);
    if (!is_dirty) {
        return;
    }
    DAddr aligned_start = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
    DAddr aligned_end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
    if (!IsRegionGpuModified(aligned_start, aligned_end - aligned_start)) {
        WriteMemory(device_addr, size);
        return;
    }

    Nx2NoteListHostWrite(device_addr, size);
    tmp_buffer.resize_destructive(size);
    device_memory.ReadBlockUnsafe(device_addr, tmp_buffer.data(), size);
    bool from_snapshot = false;
    if (VideoCommon::FrameSnapshot::Part(4)) {
        // [EXPERIMENT] The CPU's bytes as they were for the frame the back is on; the newer ones
        // are applied once it finished that frame.
        u32 tag = 0;
        if (VideoCommon::FrameSnapshot::Overlay(device_addr, tmp_buffer.data(), size, tag)) {
            VideoCommon::FrameSnapshot::Defer(device_addr, size, tag);
            from_snapshot = true;
        }
    }
    if (!from_snapshot) {
        VideoCommon::LateWriteTrace::NoteBackRead(device_addr, size, "held CPU write");
    }

    InlineMemoryImplementation(device_addr, size, tmp_buffer);
}

template <class P>
bool BufferCache<P>::OnCPUWrite(DAddr device_addr, u64 size) {
    namespace BackStats = VideoCommon::BackStats;
    const bool is_dirty = IsRegionRegistered(device_addr, size);
    if (!is_dirty) {
        if (BackStats::Enabled()) {
            BackStats::Add(BackStats::shared.cpu_write_unregistered, 1);
        }
        return false;
    }
    if (memory_tracker.IsRegionGpuModified(device_addr, size)) {
        if (BackStats::Enabled()) {
            BackStats::Add(BackStats::shared.cpu_write_held, 1);
        }
        return true;
    }
    if (BackStats::Enabled()) {
        BackStats::Add(BackStats::shared.cpu_write_written, 1);
    }
    WriteMemory(device_addr, size);
    return false;
}

template <class P>
std::optional<VideoCore::RasterizerDownloadArea> BufferCache<P>::GetFlushArea(DAddr device_addr,
                                                                              u64 size) {
    std::optional<VideoCore::RasterizerDownloadArea> area{};
    area.emplace();
    DAddr device_addr_start_aligned = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
    DAddr device_addr_end_aligned = Common::AlignUp(device_addr + size, Core::DEVICE_PAGESIZE);
    area->start_address = device_addr_start_aligned;
    area->end_address = device_addr_end_aligned;
    if (memory_tracker.IsRegionPreflushable(device_addr, size)) {
        area->preemtive = true;
        return area;
    };
    area->preemtive = !IsRegionGpuModified(device_addr_start_aligned,
                                           device_addr_end_aligned - device_addr_start_aligned);
    memory_tracker.MarkRegionAsPreflushable(device_addr_start_aligned,
                                            device_addr_end_aligned - device_addr_start_aligned);
    return area;
}

template <class P>
void BufferCache<P>::DownloadMemory(DAddr device_addr, u64 size) {
    ForEachBufferInRange(device_addr, size, [&](BufferId, Buffer& buffer) {
        DownloadBufferMemory(buffer, device_addr, size);
    });
}

template <class P>
auto BufferCache<P>::RecordDownloadSnapshot(DAddr device_addr, u64 size)
    -> std::optional<DownloadSnapshot> {
    if constexpr (!USE_MEMORY_MAPS) {
        return std::nullopt;
    } else {
        BufferId found{};
        u32 count = 0;
        ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer&) {
            found = buffer_id;
            ++count;
        });
        if (count != 1 || size == 0) {
            return std::nullopt;
        }
        Buffer& buffer = slot_buffers[found];
        if (device_addr < buffer.CpuAddr() ||
            device_addr + size > buffer.CpuAddr() + buffer.SizeBytes()) {
            return std::nullopt;
        }
        DownloadSnapshot staging = runtime.DownloadStagingBuffer(size, true);
        std::array copies{BufferCopy{
            .src_offset = device_addr - buffer.CpuAddr(),
            .dst_offset = staging.offset,
            .size = size,
        }};
        buffer.MarkUsage(copies[0].src_offset, size);
        runtime.CopyBuffer(staging.buffer, buffer, std::span<BufferCopy>(copies), true);
        return staging;
    }
}

template <class P>
void BufferCache<P>::ClearDownload(DAddr device_addr, u64 size) {
    async_downloads.DeleteAll(device_addr, size);
    ++mark_generation; // the marks MarkWrittenBuffer skips may be gone
    uncommitted_gpu_modified_ranges.Subtract(device_addr, size);
    for (auto& interval_set : committed_gpu_modified_ranges) {
        interval_set.Subtract(device_addr, size);
    }
}

template <class P>
bool BufferCache<P>::DMACopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) {
    const std::optional<DAddr> cpu_src_address = gpu_memory->GpuToCpuAddress(src_address);
    const std::optional<DAddr> cpu_dest_address = gpu_memory->GpuToCpuAddress(dest_address);
    if (!cpu_src_address || !cpu_dest_address) {
        return false;
    }
    const bool source_dirty = IsRegionRegistered(*cpu_src_address, amount);
    const bool dest_dirty = IsRegionRegistered(*cpu_dest_address, amount);
    if (!source_dirty && !dest_dirty) {
        return false;
    }

    ClearDownload(*cpu_dest_address, amount);

    BufferId buffer_a;
    BufferId buffer_b;
    do {
        channel_state->has_deleted_buffers = false;
        buffer_a = FindBuffer(*cpu_src_address, static_cast<u32>(amount), false);
        buffer_b = FindBuffer(*cpu_dest_address, static_cast<u32>(amount), false);
    } while (channel_state->has_deleted_buffers);
    auto& src_buffer = slot_buffers[buffer_a];
    auto& dest_buffer = slot_buffers[buffer_b];
    SynchronizeBuffer(src_buffer, *cpu_src_address, static_cast<u32>(amount));
    SynchronizeBuffer(dest_buffer, *cpu_dest_address, static_cast<u32>(amount));
    std::array copies{BufferCopy{
        .src_offset = src_buffer.Offset(*cpu_src_address),
        .dst_offset = dest_buffer.Offset(*cpu_dest_address),
        .size = amount,
    }};

    boost::container::small_vector<std::pair<DAddr, size_t>, 4> tmp_intervals;
    auto mirror = [&](DAddr base_address, DAddr base_address_end) {
        const u64 size = base_address_end - base_address;
        const DAddr diff = base_address - *cpu_src_address;
        const DAddr new_base_address = *cpu_dest_address + diff;
        tmp_intervals.push_back({new_base_address, size});
        QueueDownload(new_base_address, size);
    };
    gpu_modified_ranges.ForEachInRange(*cpu_src_address, amount, mirror);
    // This subtraction in this order is important for overlapping copies.
    // CPU-backed portions also replace command words, invalidating any older page copy.
    Nx2NoteListHostWrite(*cpu_dest_address, amount);
    ++mark_generation; // the marks MarkWrittenBuffer skips may be gone
    gpu_modified_ranges.Subtract(*cpu_dest_address, amount);
    const bool has_new_downloads = tmp_intervals.size() != 0;
    for (const auto& pair : tmp_intervals) {
        gpu_modified_ranges.Add(pair.first, pair.second);
        VideoCommon::MarkGpuWritten(pair.first, pair.second);
        Nx2NoteListWrite(buffer_b, pair.first, pair.second);
    }
    const auto& copy = copies[0];
    src_buffer.MarkUsage(copy.src_offset, copy.size);
    dest_buffer.MarkUsage(copy.dst_offset, copy.size);
    runtime.CopyBuffer(dest_buffer, src_buffer, copies, true);
    if (has_new_downloads) {
        memory_tracker.MarkRegionAsGpuModified(*cpu_dest_address, amount);
        const bool should_sync = Settings::IsGPUFenceBehaviorBalanced() || Settings::IsGPUFenceBehaviorAccurate();
        if (should_sync) {
            runtime.Finish();
        }
    }

    Tegra::Memory::DeviceGuestMemoryScoped<u8, Tegra::Memory::GuestMemoryFlags::UnsafeReadWrite>
        tmp(device_memory, *cpu_src_address, amount, &tmp_buffer);
    tmp.SetAddressAndSize(*cpu_dest_address, amount);
    return true;
}

template <class P>
bool BufferCache<P>::DMAClear(GPUVAddr dst_address, u64 amount, u32 value) {
    const std::optional<DAddr> cpu_dst_address = gpu_memory->GpuToCpuAddress(dst_address);
    if (!cpu_dst_address) {
        return false;
    }
    const bool dest_dirty = IsRegionRegistered(*cpu_dst_address, amount);
    if (!dest_dirty) {
        return false;
    }

    const size_t size = amount * sizeof(u32);
    // The DMA engine mirrors this fill into guest memory before its method completes.
    Nx2NoteListHostWrite(*cpu_dst_address, size);
    ClearDownload(*cpu_dst_address, size);
    ++mark_generation; // the marks MarkWrittenBuffer skips may be gone
    gpu_modified_ranges.Subtract(*cpu_dst_address, size);

    const BufferId buffer = FindBuffer(*cpu_dst_address, static_cast<u32>(size), false);
    Buffer& dest_buffer = slot_buffers[buffer];
    const u32 offset = dest_buffer.Offset(*cpu_dst_address);
    runtime.ClearBuffer(dest_buffer, offset, size, value);
    dest_buffer.MarkUsage(offset, size);
    return true;
}

template <class P>
std::pair<typename P::Buffer*, u32> BufferCache<P>::ObtainBuffer(GPUVAddr gpu_addr, u32 size,
                                                                 ObtainBufferSynchronize sync_info,
                                                                 ObtainBufferOperation post_op) {
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (!device_addr) {
        return {&slot_buffers[NULL_BUFFER_ID], 0};
    }
    return ObtainCPUBuffer(*device_addr, size, sync_info, post_op);
}

template <class P>
std::pair<typename P::Buffer*, u32> BufferCache<P>::ObtainCPUBuffer(
    DAddr device_addr, u32 size, ObtainBufferSynchronize sync_info, ObtainBufferOperation post_op) {
    const BufferId buffer_id = FindBuffer(device_addr, size, false);
    Buffer& buffer = slot_buffers[buffer_id];

    // synchronize op
    switch (sync_info) {
    case ObtainBufferSynchronize::FullSynchronize:
        SynchronizeBuffer(buffer, device_addr, size);
        break;
    default:
        break;
    }

    switch (post_op) {
    case ObtainBufferOperation::MarkAsWritten:
        MarkWrittenBuffer(buffer_id, device_addr, size);
        break;
    case ObtainBufferOperation::DiscardWrite: {
        ClearDownload(device_addr, size);
        ++mark_generation; // the marks MarkWrittenBuffer skips may be gone
        gpu_modified_ranges.Subtract(device_addr, size);
        break;
    }
    default:
        break;
    }

    return {&buffer, buffer.Offset(device_addr)};
}

template <class P>
void BufferCache<P>::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                               u32 size) {
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    const Binding binding{
        .device_addr = *device_addr,
        .gpu_addr = gpu_addr,
        .size = size,
        .buffer_id = BufferId{},
    };
    channel_state->uniform_buffers[stage][index] = binding;
}

template <class P>
void BufferCache<P>::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    channel_state->uniform_buffers[stage][index] = NULL_BINDING;
}

template <class P>
void BufferCache<P>::UpdateGraphicsBuffers(bool is_indexed) {
    do {
        channel_state->has_deleted_buffers = false;
        DoUpdateGraphicsBuffers(is_indexed);
    } while (channel_state->has_deleted_buffers);
}

template <class P>
void BufferCache<P>::UpdateComputeBuffers() {
    do {
        channel_state->has_deleted_buffers = false;
        DoUpdateComputeBuffers();
    } while (channel_state->has_deleted_buffers);
}

template <class P>
void BufferCache<P>::BindHostGeometryBuffers(bool is_indexed) {
    if (is_indexed) {
        BindHostIndexBuffer();
    } else if constexpr (!HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT) {
        const auto& draw_state = maxwell3d->draw_manager.draw_state;
        if (draw_state.topology == Maxwell::PrimitiveTopology::Quads ||
            draw_state.topology == Maxwell::PrimitiveTopology::QuadStrip) {
            runtime.BindQuadIndexBuffer(draw_state.topology, draw_state.vertex_buffer.first,
                                        draw_state.vertex_buffer.count);
        }
    }
    BindHostVertexBuffers();
    BindHostTransformFeedbackBuffers();
    if (current_draw_indirect) {
        BindHostDrawIndirectBuffers();
    }
}

template <class P>
void BufferCache<P>::BindHostStageBuffers(size_t stage) {
    BindHostGraphicsUniformBuffers(stage);
    BindHostGraphicsStorageBuffers(stage);
    BindHostGraphicsTextureBuffers(stage);
}

template <class P>
void BufferCache<P>::BindHostComputeBuffers() {
    BindHostComputeUniformBuffers();
    BindHostComputeStorageBuffers();
    BindHostComputeTextureBuffers();
    if (any_buffer_uploaded) {
        runtime.PostCopyBarrier();
        any_buffer_uploaded = false;
    }
}

template <class P>
void BufferCache<P>::SetUniformBuffersState(const std::array<u32, NUM_STAGES>& mask,
                                            const UniformBufferSizes* sizes) {
    if (channel_state->enabled_uniform_buffer_masks != mask) {
        channel_state->fast_bound_uniform_buffers.fill(0);
        if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
            channel_state->dirty_uniform_buffers.fill(~u32{0});
            channel_state->uniform_buffer_binding_sizes.fill({});
        }
    }
    channel_state->enabled_uniform_buffer_masks = mask;
    channel_state->uniform_buffer_sizes = sizes;
}

template <class P>
void BufferCache<P>::SetComputeUniformBufferState(u32 mask,
                                                  const ComputeUniformBufferSizes* sizes) {
    channel_state->enabled_compute_uniform_buffer_mask = mask;
    channel_state->compute_uniform_buffer_sizes = sizes;
}

template <class P>
void BufferCache<P>::UnbindGraphicsStorageBuffers(size_t stage) {
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers()) {
            channel_state->total_graphics_storage_buffers -=
                static_cast<u32>(std::popcount(channel_state->enabled_storage_buffers[stage]));
        }
    }
    channel_state->enabled_storage_buffers[stage] = 0;
    channel_state->written_storage_buffers[stage] = 0;
}

template <class P>
bool BufferCache<P>::BindGraphicsStorageBuffer(size_t stage, size_t ssbo_index, u32 cbuf_index,
                                               u32 cbuf_offset, bool is_written,
                                               u32 indirect_offset, u32 write_extent) {
    if (stage >= NUM_STAGES || ssbo_index >= NUM_STORAGE_BUFFERS) {
        LOG_ERROR(HW_GPU, "Invalid graphics storage buffer binding: stage {}, index {}",
                  stage, ssbo_index);
        return false;
    }
    const bool already_enabled =
        ((channel_state->enabled_storage_buffers[stage] >> ssbo_index) & 1U) != 0;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            const u32 max_bindings = runtime.GetMaxDynamicStorageBuffers();
            if (channel_state->total_graphics_storage_buffers >= max_bindings) {
                LOG_WARNING(HW_GPU,
                            "Skipping graphics storage buffer {} due to driver limit {}",
                            ssbo_index, max_bindings);
                return false;
            }
        }
    }
    channel_state->enabled_storage_buffers[stage] |= 1U << ssbo_index;
    channel_state->written_storage_buffers[stage] |= (is_written ? 1U : 0U) << ssbo_index;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            ++channel_state->total_graphics_storage_buffers;
        }
    }

    const auto& cbufs = maxwell3d->state.shader_stages[stage];
    GPUVAddr ssbo_addr = cbufs.const_buffers[cbuf_index].address + cbuf_offset;
    if (indirect_offset != ~0U) {
        // The pointer is an entry of a table the constant buffer points to.
        const GPUVAddr holder = VideoCommon::FrameSnapshot::ReadGuest<u64>(*gpu_memory, ssbo_addr);
        if (holder == 0) {
            channel_state->storage_buffers[stage][ssbo_index] = NULL_BINDING;
            return false;
        }
        ssbo_addr = holder + indirect_offset;
    }
    channel_state->storage_buffers[stage][ssbo_index] =
        StorageBufferBinding(ssbo_addr, cbuf_index, is_written,
                             maxwell3d->GetNativeMacroEngine().native_turing,
                             indirect_offset != ~0U);
    channel_state->storage_buffers[stage][ssbo_index].write_size = write_extent;
    channel_state->storage_buffers[stage][ssbo_index].dbg_cbuf =
        (cbuf_index << 16) | (cbuf_offset & 0xffff) | (indirect_offset != ~0U ? 0x80000000U : 0U);
    return (channel_state->storage_buffers[stage][ssbo_index].buffer_id != NULL_BUFFER_ID);
}

template <class P>
std::pair<GPUVAddr, u32> BufferCache<P>::BindGraphicsTableSpan(
    size_t stage, size_t ssbo_index, u32 cbuf_index, u32 cbuf_offset, u32 stride,
    u32 entry_offset, u32 slot, u32 max_ranges, u32 extent, bool is_written, u32 first,
    u32 count) {
    if (stage >= NUM_STAGES || ssbo_index >= NUM_STORAGE_BUFFERS) {
        return {};
    }
    // Enabled with a null binding even when nothing is read: the program's binding must exist.
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() &&
            ((channel_state->enabled_storage_buffers[stage] >> ssbo_index) & 1U) == 0) {
            ++channel_state->total_graphics_storage_buffers;
        }
    }
    channel_state->enabled_storage_buffers[stage] |= 1U << ssbo_index;
    channel_state->written_storage_buffers[stage] |= (is_written ? 1U : 0U) << ssbo_index;
    channel_state->storage_buffers[stage][ssbo_index] = NULL_BINDING;

    // The ranges of one draw are computed with its first slot and kept for the others.
    struct SpanKey {
        GPUVAddr table;
        u32 stride;
        u32 entry_offset;
        u32 first;
        u32 count;
        u32 max_ranges;
        u32 extent;
        auto operator<=>(const SpanKey&) const = default;
    };
    thread_local SpanKey last_key{};
    thread_local std::vector<std::pair<GPUVAddr, GPUVAddr>> ranges;
    thread_local std::vector<GPUVAddr> targets;

    const auto& cbufs = maxwell3d->state.shader_stages[stage];
    const GPUVAddr table = VideoCommon::FrameSnapshot::ReadGuest<u64>(
        *gpu_memory, cbufs.const_buffers[cbuf_index].address + cbuf_offset);
    static constexpr u32 max_entries = 1U << 16;
    if (table == 0 || count == 0 || stride == 0 || max_ranges == 0) {
        return {};
    }
    if (count > max_entries) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            LOG_WARNING(HW_GPU, "Pointer table span: {} entries, reading the first {}", count,
                        max_entries);
        }
        count = max_entries;
    }
    const SpanKey key{table, stride, entry_offset, first, count, max_ranges, extent};
    if (slot == 0 || key != last_key) {
        last_key = key;
        ranges.clear();
        targets.clear();
        const GPUVAddr entries = table + entry_offset + u64{first} * stride;
        for (u32 i = 0; i < count; ++i) {
            const GPUVAddr target =
                VideoCommon::FrameSnapshot::ReadGuest<u64>(*gpu_memory, entries + u64{i} * stride);
            if (target != 0) {
                targets.push_back(target);
            }
        }
        std::ranges::sort(targets);
        const auto [unique_end, end] = std::ranges::unique(targets);
        targets.erase(unique_end, end);
        if (!targets.empty()) {
            // A written binding is one buffer over device memory from its first address
            // (written bindings take no multi-range path), so a range must not cross a break in
            // device memory: targets whose GPU and device distances differ are always split.
            // The other cuts go to the widest gaps between consecutive targets.
            std::vector<size_t> cuts;
            std::vector<std::pair<u64, size_t>> gaps;
            gaps.reserve(targets.size());
            std::optional<DAddr> previous_device = gpu_memory->GpuToCpuAddress(targets[0]);
            for (size_t i = 1; i < targets.size(); ++i) {
                const std::optional<DAddr> device = gpu_memory->GpuToCpuAddress(targets[i]);
                const bool contiguous = device && previous_device &&
                                        *device - *previous_device == targets[i] - targets[i - 1];
                previous_device = device;
                if (!contiguous) {
                    cuts.push_back(i);
                } else {
                    gaps.emplace_back(targets[i] - targets[i - 1], i);
                }
            }
            if (cuts.size() + 1 > max_ranges) {
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    LOG_WARNING(HW_GPU,
                                "Pointer table span: targets lie in {} device memory blocks, "
                                "binding the first {}",
                                cuts.size() + 1, max_ranges);
                }
            }
            const size_t splits = cuts.size() + 1 >= max_ranges
                                      ? 0
                                      : (std::min)(gaps.size(), max_ranges - 1 - cuts.size());
            std::ranges::partial_sort(gaps, gaps.begin() + splits, std::greater{});
            for (size_t i = 0; i < splits; ++i) {
                if (gaps[i].first > extent) {
                    cuts.push_back(gaps[i].second);
                }
            }
            std::ranges::sort(cuts);
            cuts.push_back(targets.size());
            size_t begin = 0;
            for (const size_t cut : cuts) {
                ranges.emplace_back(targets[begin], targets[cut - 1] + extent);
                begin = cut;
            }
        }
        static const bool trace = [] {
            const char* const value = std::getenv("EDEN_NX2_TRACE_TABLE_SPAN");
            return value != nullptr && value[0] != '0';
        }();
        static u32 traced = 0;
        if (trace && (traced < 40 || traced % 200 == 0)) {
            u64 covered = 0;
            for (const auto& [lo, hi] : ranges) {
                covered += hi - lo;
            }
            const auto device = gpu_memory->GpuToCpuAddress(entries);
            LOG_INFO(HW_GPU,
                     "Pointer table span: table {:#x} first {} count {} stride {}: {} targets "
                     "in {} ranges covering {} KiB, first range {:#x}-{:#x}, table gpu-modified {}",
                     table, first, count, stride, targets.size(), ranges.size(), covered / 1024,
                     ranges.empty() ? 0 : ranges[0].first, ranges.empty() ? 0 : ranges[0].second,
                     device && IsRegionGpuModified(*device, u64{count} * stride));
            std::string list;
            for (const auto& [lo, hi] : ranges) {
                list += fmt::format(" {:#x}-{:#x}", lo, hi);
            }
            LOG_INFO(HW_GPU, "Pointer table span ranges:{}", list);
            if (traced >= 200) {
                // [DEBUG] the targets, for matching against zero-instance indirect records.
                std::string dump;
                for (const GPUVAddr target : targets) {
                    dump += fmt::format("{:x}", target);
                    dump += '\n';
                }
                const auto path = Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) /
                                  fmt::format("span-targets-{:x}-{}.txt", table, traced);
                if (std::FILE* const file = std::fopen(path.string().c_str(), "wb")) {
                    std::fwrite(dump.data(), 1, dump.size(), file);
                    std::fclose(file);
                }
            }
        }
        if (trace) {
            ++traced;
        }
    }
    if (slot >= ranges.size()) {
        return {};
    }
    const auto [lo, hi] = ranges[slot];
    const u32 alignment = runtime.GetStorageBufferAlignment();
    const GPUVAddr aligned_lo = Common::AlignDown(lo, alignment);
    const u64 mapped_size = gpu_memory->GetMemoryLayoutSize(aligned_lo);
    // The extent past the last target stays inside its device memory block too.
    const u32 size = static_cast<u32>(gpu_memory->DeviceContiguousSize(
        aligned_lo, (std::min)(hi - aligned_lo, mapped_size)));
    static const bool trace_clamp = [] {
        const char* const value = std::getenv("EDEN_NX2_TRACE_TABLE_SPAN");
        return value != nullptr && value[0] != '0';
    }();
    if (trace_clamp && size < hi - aligned_lo) {
        static u32 clamp_logged = 0;
        if (clamp_logged < 60) {
            ++clamp_logged;
            const auto lost = std::ranges::count_if(targets, [&](GPUVAddr target) {
                return target >= lo && target < hi && target + 20 > aligned_lo + size;
            });
            LOG_INFO(HW_GPU,
                     "Pointer table span: range {} {:#x}-{:#x} cut to {:#x} bytes by its mapping "
                     "({:#x}): {} targets outside",
                     slot, lo, hi, size, mapped_size, lost);
        }
    }
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(aligned_lo);
    if (!device_addr || size == 0) {
        return {};
    }
    channel_state->storage_buffers[stage][ssbo_index] = Binding{
        .device_addr = *device_addr,
        .gpu_addr = aligned_lo,
        .size = size,
        .buffer_id = BufferId{},
    };
    return {aligned_lo, size};
}

template <class P>
void BufferCache<P>::UnbindGraphicsTextureBuffers(size_t stage) {
    channel_state->enabled_texture_buffers[stage] = 0;
    channel_state->written_texture_buffers[stage] = 0;
    channel_state->image_texture_buffers[stage] = 0;
}

template <class P>
void BufferCache<P>::BindGraphicsTextureBuffer(size_t stage, size_t tbo_index, GPUVAddr gpu_addr,
                                               u32 size, PixelFormat format, bool is_written,
                                               bool is_image) {
    channel_state->enabled_texture_buffers[stage] |= 1U << tbo_index;
    channel_state->written_texture_buffers[stage] |= (is_written ? 1U : 0U) << tbo_index;
    if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
        channel_state->image_texture_buffers[stage] |= (is_image ? 1U : 0U) << tbo_index;
    }
    channel_state->texture_buffers[stage][tbo_index] =
        GetTextureBufferBinding(gpu_addr, size, format);
}

template <class P>
void BufferCache<P>::UnbindComputeStorageBuffers() {
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers()) {
            channel_state->total_compute_storage_buffers -=
                static_cast<u32>(std::popcount(channel_state->enabled_compute_storage_buffers));
        }
    }
    channel_state->enabled_compute_storage_buffers = 0;
    channel_state->written_compute_storage_buffers = 0;
    channel_state->image_compute_texture_buffers = 0;
}

template <class P>
void BufferCache<P>::BindComputeStorageBuffer(size_t ssbo_index, u32 cbuf_index, u32 cbuf_offset,
                                              bool is_written, u32 indirect_offset,
                                              u32 write_extent) {
    if (ssbo_index >= channel_state->compute_storage_buffers.size()) [[unlikely]] {
        LOG_ERROR(HW_GPU, "Storage buffer index {} exceeds maximum storage buffer count",
                  ssbo_index);
        return;
    }
    const bool already_enabled =
        ((channel_state->enabled_compute_storage_buffers >> ssbo_index) & 1U) != 0;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            const u32 max_bindings = runtime.GetMaxDynamicStorageBuffers();
            if (channel_state->total_compute_storage_buffers >= max_bindings) {
                LOG_WARNING(HW_GPU,
                            "Skipping compute storage buffer {} due to driver limit {}",
                            ssbo_index, max_bindings);
                return;
            }
        }
    }
    channel_state->enabled_compute_storage_buffers |= u64{1} << ssbo_index;
    channel_state->written_compute_storage_buffers |= u64{is_written ? 1U : 0U} << ssbo_index;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            ++channel_state->total_compute_storage_buffers;
        }
    }

    const auto& launch_desc = kepler_compute->launch_description;
    if (((launch_desc.const_buffer_enable_mask >> cbuf_index) & 1) == 0) {
        LOG_WARNING(HW_GPU, "Skipped binding SSBO: cbuf index {} is not enabled", cbuf_index);
        return;
    }
    ASSERT(((launch_desc.const_buffer_enable_mask >> cbuf_index) & 1) != 0);

    const auto& cbufs = launch_desc.const_buffer_config;
    GPUVAddr ssbo_addr = cbufs[cbuf_index].Address() + cbuf_offset;
    if (indirect_offset != ~0U) {
        const GPUVAddr holder = VideoCommon::FrameSnapshot::ReadGuest<u64>(*gpu_memory, ssbo_addr);
        if (holder == 0) {
            channel_state->compute_storage_buffers[ssbo_index] = NULL_BINDING;
            return;
        }
        ssbo_addr = holder + indirect_offset;
    }
    channel_state->compute_storage_buffers[ssbo_index] =
        StorageBufferBinding(ssbo_addr, cbuf_index, is_written, kepler_compute->IsNativeMode(),
                             indirect_offset != ~0U);
    channel_state->compute_storage_buffers[ssbo_index].write_size = write_extent;
}

template <class P>
void BufferCache<P>::UnbindComputeTextureBuffers() {
    channel_state->enabled_compute_texture_buffers = 0;
    channel_state->written_compute_texture_buffers = 0;
    channel_state->image_compute_texture_buffers = 0;
}

template <class P>
void BufferCache<P>::BindComputeTextureBuffer(size_t tbo_index, GPUVAddr gpu_addr, u32 size,
                                              PixelFormat format, bool is_written, bool is_image) {
    if (tbo_index >= channel_state->compute_texture_buffers.size()) [[unlikely]] {
        LOG_ERROR(HW_GPU, "Texture buffer index {} exceeds maximum texture buffer count",
                  tbo_index);
        return;
    }
    channel_state->enabled_compute_texture_buffers |= 1U << tbo_index;
    channel_state->written_compute_texture_buffers |= (is_written ? 1U : 0U) << tbo_index;
    if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
        channel_state->image_compute_texture_buffers |= (is_image ? 1U : 0U) << tbo_index;
    }
    channel_state->compute_texture_buffers[tbo_index] =
        GetTextureBufferBinding(gpu_addr, size, format);
}

template <class P>
void BufferCache<P>::FlushCachedWrites() {
    ++mark_generation; // the marks MarkWrittenBuffer skips may be gone
    memory_tracker.FlushCachedWrites();
}

template <class P>
bool BufferCache<P>::HasUncommittedFlushes() const noexcept {
    return !uncommitted_gpu_modified_ranges.Empty() || !committed_gpu_modified_ranges.empty();
}

template <class P>
void BufferCache<P>::AccumulateFlushes() {
    if (uncommitted_gpu_modified_ranges.Empty()) {
        return;
    }
    ++mark_generation; // the marks MarkWrittenBuffer skips may be gone
    committed_gpu_modified_ranges.emplace_back(std::move(uncommitted_gpu_modified_ranges));
}

template <class P>
bool BufferCache<P>::ShouldWaitAsyncFlushes() const noexcept {
    return (!async_buffers.empty() && async_buffers.front().has_value());
}

template <class P>
void BufferCache<P>::CommitAsyncFlushesHigh() {
    AccumulateFlushes();

    if (committed_gpu_modified_ranges.empty()) {
        async_buffers.emplace_back(std::optional<Async_Buffer>{});
        return;
    }

    auto it = committed_gpu_modified_ranges.begin();
    while (it != committed_gpu_modified_ranges.end()) {
        auto& current_intervals = *it;
        auto next_it = std::next(it);
        while (next_it != committed_gpu_modified_ranges.end()) {
            next_it->ForEach([&current_intervals](DAddr start, DAddr end) {
                current_intervals.Subtract(start, end - start);
            });
            next_it++;
        }
        it++;
    }

    boost::container::small_vector<std::pair<BufferCopy, BufferId>, 16> downloads;
    u64 total_size_bytes = 0;
    u64 largest_copy = 0;
    for (const Common::RangeSet<DAddr>& range_set : committed_gpu_modified_ranges) {
        range_set.ForEach([&](DAddr interval_lower, DAddr interval_upper) {
            const std::size_t size = interval_upper - interval_lower;
            const DAddr device_addr = interval_lower;
            ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer& buffer) {
                const DAddr buffer_start = buffer.CpuAddr();
                const DAddr buffer_end = buffer_start + buffer.SizeBytes();
                const DAddr new_start = (std::max)(buffer_start, device_addr);
                const DAddr new_end = (std::min)(buffer_end, device_addr + size);
                memory_tracker.ForEachDownloadRange(
                    new_start, new_end - new_start, false,
                    [&](u64 device_addr_out, u64 range_size) {
                        const DAddr buffer_addr = buffer.CpuAddr();
                        const auto add_download = [&](DAddr start, DAddr end) {
                            const u64 new_offset = start - buffer_addr;
                            const u64 new_size = end - start;
                            downloads.push_back({
                                BufferCopy{
                                    .src_offset = new_offset,
                                    .dst_offset = total_size_bytes,
                                    .size = new_size,
                                },
                                buffer_id,
                            });
                            // Align up to avoid cache conflicts
                            constexpr u64 align = 64ULL;
                            constexpr u64 mask = ~(align - 1ULL);
                            total_size_bytes += (new_size + align - 1) & mask;
                            largest_copy = (std::max)(largest_copy, new_size);
                        };

                        gpu_modified_ranges.ForEachInRange(device_addr_out, range_size,
                                                           add_download);
                    });
            });
        });
    }
    ++mark_generation; // the marks MarkWrittenBuffer skips may be gone
    committed_gpu_modified_ranges.clear();
    if (downloads.empty()) {
        async_buffers.emplace_back(std::optional<Async_Buffer>{});
        return;
    }
    auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes, true);
    boost::container::small_vector<BufferCopy, 4> normalized_copies;
    runtime.PreCopyBarrier();
    for (auto& [copy, buffer_id] : downloads) {
        copy.dst_offset += download_staging.offset;
        const std::array copies{copy};
        BufferCopy second_copy{copy};
        Buffer& buffer = slot_buffers[buffer_id];
        second_copy.src_offset = static_cast<size_t>(buffer.CpuAddr()) + copy.src_offset;
        const DAddr orig_device_addr = static_cast<DAddr>(second_copy.src_offset);
        async_downloads.Add(orig_device_addr, copy.size);
        buffer.MarkUsage(copy.src_offset, copy.size);
        runtime.CopyBuffer(download_staging.buffer, buffer, copies, false);
        normalized_copies.push_back(second_copy);
    }
    runtime.PostCopyBarrier();
    pending_downloads.emplace_back(std::move(normalized_copies));
    async_buffers.emplace_back(download_staging);
}

template <class P>
void BufferCache<P>::CommitAsyncFlushes() {
    CommitAsyncFlushesHigh();
}

template <class P>
void BufferCache<P>::PopAsyncFlushes() {
    PopAsyncBuffers();
}

template <class P>
void BufferCache<P>::PopAsyncBuffers() {
    if (async_buffers.empty()) {
        return;
    }
    if (!async_buffers.front().has_value()) {
        async_buffers.pop_front();
        return;
    }
    auto& downloads = pending_downloads.front();
    auto& async_buffer = async_buffers.front();
    u8* base = async_buffer->mapped_span.data();
    const size_t base_offset = async_buffer->offset;
    for (const auto& copy : downloads) {
        const DAddr device_addr = static_cast<DAddr>(copy.src_offset);
        const u64 dst_offset = copy.dst_offset - base_offset;
        const u8* read_mapped_memory = base + dst_offset;
        async_downloads.ForEachInRange(device_addr, copy.size, [&](DAddr start, DAddr end, s32) {
            device_memory.WriteBlockUnsafe(start, &read_mapped_memory[start - device_addr],
                                           end - start);
            VideoCommon::FrameSnapshot::OnGpuWrite(start, &read_mapped_memory[start - device_addr],
                                                   end - start);
        });
        async_downloads.Subtract(device_addr, copy.size, [&](DAddr start, DAddr end) {
            ++mark_generation; // the marks MarkWrittenBuffer skips may be gone
            gpu_modified_ranges.Subtract(start, end - start);
        });
    }
    async_buffers_death_ring.emplace_back(*async_buffer);
    async_buffers.pop_front();
    pending_downloads.pop_front();
}

template <class P>
void BufferCache<P>::NoteCpuRead(DAddr addr, size_t size) {
    if (AsyncBufferDownloads() || size == 0) {
        return;
    }
    size_t covered = 0;
    cpu_read_ranges.ForEachInRange(addr, size,
                                   [&](DAddr start, DAddr end) { covered += end - start; });
    if (covered < size) {
        cpu_read_ranges.Add(addr, size);
    }
}

template <class P>
bool BufferCache<P>::IsRegionGpuModified(DAddr addr, size_t size) {
    // One lookup instead of visiting every range inside (~4% of the GPU thread for title 1's
    // inline uploads once GPU-written ranges are no longer cleared by downloads).
    const bool modified = gpu_modified_ranges.Intersects(addr, size);
    if (VideoCommon::BackStats::Enabled()) {
        namespace BackStats = VideoCommon::BackStats;
        BackStats::Add(BackStats::shared.gpu_modified_queries, 1);
        if (modified) {
            BackStats::Add(BackStats::shared.gpu_modified_true, 1);
        }
        if ((addr >> 12) == ((addr + (size != 0 ? size - 1 : 0)) >> 12)) {
            BackStats::Add(BackStats::shared.gpu_modified_one_page, 1);
        }
    }
    return modified;
}

template <class P>
bool BufferCache<P>::IsRegionRegistered(DAddr addr, size_t size) {
    const DAddr end_addr = addr + size;
    const u64 page_end = Common::DivCeil(end_addr, CACHING_PAGESIZE);
    for (u64 page = addr >> CACHING_PAGEBITS; page < page_end;) {
        const BufferId buffer_id = page_table[page];
        if (!buffer_id) {
            ++page;
            continue;
        }
        Buffer& buffer = slot_buffers[buffer_id];
        const DAddr buf_start_addr = buffer.CpuAddr();
        const DAddr buf_end_addr = buf_start_addr + buffer.SizeBytes();
        if (buf_start_addr < end_addr && addr < buf_end_addr) {
            return true;
        }
        page = Common::DivCeil(end_addr, CACHING_PAGESIZE);
    }
    return false;
}

template <class P>
bool BufferCache<P>::IsRegionCpuModified(DAddr addr, size_t size) {
    return memory_tracker.IsRegionCpuModified(addr, size);
}

template <class P>
void BufferCache<P>::BindHostIndexBuffer() {
    Buffer& buffer = slot_buffers[channel_state->index_buffer.buffer_id];
    TouchBuffer(buffer, channel_state->index_buffer.buffer_id);
    const u32 offset = buffer.Offset(channel_state->index_buffer.device_addr);
    const u32 size = channel_state->index_buffer.size;
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    if (draw_state.inline_index_draw_indexes.empty()) {
        SynchronizeBuffer(buffer, channel_state->index_buffer.device_addr, size);
    } else {
        if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
            auto upload_staging = runtime.UploadStagingBuffer(size);
            std::array<BufferCopy, 1> copies{{BufferCopy{.src_offset = upload_staging.offset, .dst_offset = 0, .size = size}}};
            std::memcpy(upload_staging.mapped_span.data(), draw_state.inline_index_draw_indexes.data(), size);
            runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true);
        } else {
            buffer.ImmediateUpload(0, draw_state.inline_index_draw_indexes);
        }
    }
    if constexpr (HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT) {
        const u32 new_offset = offset + draw_state.index_buffer.first * u32(draw_state.index_buffer.FormatSizeInBytes());
        runtime.BindIndexBuffer(buffer, new_offset, size);
    } else {
        buffer.MarkUsage(offset, size);
        runtime.BindIndexBuffer(draw_state.topology, draw_state.index_buffer.format, draw_state.index_buffer.first, draw_state.index_buffer.count, buffer, offset, size);
    }
}

template <class P>
void BufferCache<P>::BindHostVertexBuffer(u32 index, Buffer& buffer, u32 offset, u32 size,
                                          u32 stride) {
    if constexpr (IS_OPENGL) {
        runtime.BindVertexBuffer(index, buffer, offset, size, stride);
    } else {
        runtime.BindVertexBuffer(index, buffer.Handle(), offset, size, stride);
    }
}

template <class P>
Binding& BufferCache<P>::VertexBufferSlot(u32 index) {
    ASSERT(index < NUM_VERTEX_BUFFERS);
    return v_buffer[index];
}

template <class P>
const Binding& BufferCache<P>::VertexBufferSlot(u32 index) const {
    ASSERT(index < NUM_VERTEX_BUFFERS);
    return v_buffer[index];
}

template <class P>
void BufferCache<P>::UpdateVertexBufferSlot(u32 index, const Binding& binding) {
    Binding& slot = VertexBufferSlot(index);
    if (slot.device_addr != binding.device_addr || slot.size != binding.size) {
        ++vertex_buffers_serial;
    }
    slot = binding;
    if (binding.buffer_id != NULL_BUFFER_ID && binding.size != 0) {
        enabled_vertex_buffers_mask |= (1u << index);
    } else {
        enabled_vertex_buffers_mask &= ~(1u << index);
    }
}

template <class P>
void BufferCache<P>::BindHostVertexBuffers() {

#ifdef __ANDROID__
    const bool use_optimized_vertex_buffers = Settings::values.use_optimized_vertex_buffers.GetValue();
#else
    constexpr bool use_optimized_vertex_buffers = true;
#endif

    if (use_optimized_vertex_buffers) {
        auto& flags = maxwell3d->dirty.flags;
        u32 enabled_mask = enabled_vertex_buffers_mask;
        HostBindings<Buffer> bindings{};
        u32 last_index = (std::numeric_limits<u32>::max)();
        const auto flush_bindings = [&]() {
            if (bindings.buffers.empty()) {
                return;
            }
            bindings.max_index = bindings.min_index + static_cast<u32>(bindings.buffers.size());
            runtime.BindVertexBuffers(bindings);
            bindings = HostBindings<Buffer>{};
            last_index = (std::numeric_limits<u32>::max)();
        };
        while (enabled_mask != 0) {
            const u32 index = std::countr_zero(enabled_mask);
            enabled_mask &= (enabled_mask - 1);
            const Binding& binding = VertexBufferSlot(index);
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            SynchronizeBuffer(buffer, binding.device_addr, binding.size);
            if (!flags[Dirty::VertexBuffer0 + index]) {
                flush_bindings();
                continue;
            }
            flags[Dirty::VertexBuffer0 + index] = false;
            const u32 stride = maxwell3d->regs.vertex_streams[index].stride;
            const u32 offset = buffer.Offset(binding.device_addr);
            buffer.MarkUsage(offset, binding.size);
            if (!bindings.buffers.empty() && index != last_index + 1) {
                flush_bindings();
            }
            if (bindings.buffers.empty()) {
                bindings.min_index = index;
            }
            bindings.buffers.push_back(&buffer);
            bindings.offsets.push_back(offset);
            bindings.sizes.push_back(binding.size);
            bindings.strides.push_back(stride);
            last_index = index;
        }
        flush_bindings();
    } else {
        HostBindings<typename P::Buffer> host_bindings;
        bool any_valid{false};
        auto& flags = maxwell3d->dirty.flags;
        for (u32 index = 0; index < NUM_VERTEX_BUFFERS; ++index) {
            const Binding& binding = channel_state->vertex_buffers[index];
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            SynchronizeBuffer(buffer, binding.device_addr, binding.size);
            if (!flags[Dirty::VertexBuffer0 + index]) {
                continue;
            }
            flags[Dirty::VertexBuffer0 + index] = false;

            host_bindings.min_index = (std::min)(host_bindings.min_index, index);
            host_bindings.max_index = (std::max)(host_bindings.max_index, index);
            any_valid = true;
        }

        if (any_valid) {
            host_bindings.max_index++;
            for (u32 index = host_bindings.min_index; index < host_bindings.max_index; index++) {
                flags[Dirty::VertexBuffer0 + index] = false;

                const Binding& binding = channel_state->vertex_buffers[index];
                Buffer& buffer = slot_buffers[binding.buffer_id];

                const u32 stride = maxwell3d->regs.vertex_streams[index].stride;
                const u32 offset = buffer.Offset(binding.device_addr);
                buffer.MarkUsage(offset, binding.size);

                host_bindings.buffers.push_back(&buffer);
                host_bindings.offsets.push_back(offset);
                host_bindings.sizes.push_back(binding.size);
                host_bindings.strides.push_back(stride);
            }
            runtime.BindVertexBuffers(host_bindings);
        }
    }
}

template <class P>
void BufferCache<P>::BindHostDrawIndirectBuffers() {
    const auto bind_buffer = [this](const Binding& binding) {
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        SynchronizeBuffer(buffer, binding.device_addr, binding.size);
    };
    if (current_draw_indirect->include_count) {
        bind_buffer(channel_state->count_buffer_binding);
    }
    bind_buffer(channel_state->indirect_buffer_binding);
}

template <class P>
void BufferCache<P>::BindHostGraphicsUniformBuffers(size_t stage) {
    u32 dirty = ~0U;
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        dirty = std::exchange(channel_state->dirty_uniform_buffers[stage], 0);
    }
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
        const bool needs_bind = ((dirty >> index) & 1) != 0;
        BindHostGraphicsUniformBuffer(stage, index, binding_index, needs_bind);
        if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
            ++binding_index;
        }
    });
}

template <class P>
void BufferCache<P>::BindHostGraphicsUniformBuffer(size_t stage, u32 index, u32 binding_index, bool needs_bind) {
    ++channel_state->uniform_cache_shots[0];
    if (maxwell3d->GetNativeMacroEngine().native_turing && Tegra::NativeRootTables::IsBank(index)) {
        const auto& words = maxwell3d->native_root_tables.Data(index);
        const std::span<const u8> data{reinterpret_cast<const u8*>(words.data()), sizeof(words)};
        if ((*channel_state->uniform_buffer_sizes)[stage][index] > data.size())
            throw Shader::NotImplementedException("Native root-table access exceeds table size");
        if constexpr (IS_OPENGL) {
            runtime.BindFastUniformBuffer(stage, binding_index, static_cast<u32>(data.size()));
            runtime.PushFastUniformBuffer(stage, binding_index, data);
        } else {
            const auto mapped = runtime.BindMappedUniformBuffer(stage, binding_index, static_cast<u32>(data.size()));
            std::memcpy(mapped.data(), data.data(), data.size());
        }
        channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = static_cast<u32>(data.size());
        if (VideoCommon::BackStats::Enabled()) {
            ++VideoCommon::BackStats::draw_thread.uniform_binds[0];
            VideoCommon::BackStats::draw_thread.uniform_bytes[0] += data.size();
        }
        return;
    }

    const Binding& binding = channel_state->uniform_buffers[stage][index];
    const DAddr device_addr = binding.device_addr;
    const u32 size = (std::min)(binding.size, (*channel_state->uniform_buffer_sizes)[stage][index]);
    Buffer& buffer = slot_buffers[binding.buffer_id];
    TouchBuffer(buffer, binding.buffer_id);
    const bool has_host_buffer = binding.buffer_id != NULL_BUFFER_ID;
    const u32 offset = has_host_buffer ? buffer.Offset(device_addr) : 0;
    const bool needs_alignment_stream = [&]() {
        if constexpr (IS_OPENGL) {
            return false;
        } else {
            if (!has_host_buffer) {
                return false;
            }
            const u32 alignment = runtime.GetUniformBufferAlignment();
            return alignment > 1 && (offset % alignment) != 0;
        }
    }();
    const bool use_fast_buffer = needs_alignment_stream
        || (has_host_buffer && size <= channel_state->uniform_buffer_skip_cache_size
            && !memory_tracker.IsRegionGpuModified(device_addr, size));
    if (use_fast_buffer) {
        if constexpr (IS_OPENGL) {
            if (runtime.HasFastBufferSubData()) {
                // Fast path for Nvidia
                const bool should_fast_bind =
                    !HasFastUniformBufferBound(stage, binding_index) ||
                    channel_state->uniform_buffer_binding_sizes[stage][binding_index] != size;
                if (should_fast_bind) {
                    // We only have to bind when the currently bound buffer is not the fast version
                    channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
                    channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
                    runtime.BindFastUniformBuffer(stage, binding_index, size);
                }
                const auto span = ImmediateBufferWithData(device_addr, size);
                runtime.PushFastUniformBuffer(stage, binding_index, span);
                return;
            }
        }
        channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
        // Stream buffer path to avoid stalling on non-Nvidia drivers or Vulkan
        const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
        device_memory.ReadBlockUnsafe(device_addr, span.data(), size);
        if (VideoCommon::BackStats::Enabled()) {
            ++VideoCommon::BackStats::draw_thread.uniform_binds[1];
            VideoCommon::BackStats::draw_thread.uniform_bytes[1] += size;
        }
        return;
    }
    // Classic cached path
    const bool synced = SynchronizeBuffer(buffer, device_addr, size);
    if (synced) {
        ++channel_state->uniform_cache_hits[0];
    }
    if (VideoCommon::BackStats::Enabled()) {
        ++VideoCommon::BackStats::draw_thread.uniform_binds[2];
        VideoCommon::BackStats::draw_thread.uniform_binds[3] += synced ? 1 : 0;
        VideoCommon::BackStats::draw_thread.uniform_bytes[2] += size;
    }
    // Skip binding if it's not needed and if the bound buffer is not the fast version
    // This exists to avoid instances where the fast buffer is bound and a GPU write happens
    needs_bind |= HasFastUniformBufferBound(stage, binding_index);
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        needs_bind |= channel_state->uniform_buffer_binding_sizes[stage][binding_index] != size;
    }
    if (!needs_bind) {
        return;
    }
    if constexpr (IS_OPENGL) {
        // Mark the index as dirty if offset doesn't match
        const bool is_copy_bind = offset != 0 && !runtime.SupportsNonZeroUniformOffset();
        channel_state->dirty_uniform_buffers[stage] |= (is_copy_bind ? 1U : 0U) << index;
    }
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
    }
    buffer.MarkUsage(offset, size);
    if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
        runtime.BindUniformBuffer(stage, binding_index, buffer, offset, size);
    } else {
        runtime.BindUniformBuffer(buffer, offset, size);
    }
    channel_state->fast_bound_uniform_buffers[stage] &= ~(1u << binding_index);
}

template <class P>
void BufferCache<P>::PrepareGroupedStorage(std::span<const std::pair<Binding*, bool>> bindings,
                                           std::vector<MultiRangeSegment>& pool) {
    if constexpr (requires { runtime.UseGroupedStorage(); }) {
        static const bool memo_enabled = [] {
            const char* const value = std::getenv("EDEN_GROUPED_MEMO");
            return !(value && value[0] == '0');
        }();
        // FindBuffer also waits on buffer writes under accurate GPU fences: always run it then.
        GroupedMemo* memo = nullptr;
        if (memo_enabled && !Settings::IsGPUFenceBehaviorAccurate()) {
            memo = &grouped_memo[&pool == &graphics_segments ? 0 : 1];
            const size_t as_id = gpu_memory->GetID();
            const u64 map_serial = gpu_memory->MapSerial();
            const u64 range_generation = virtual_ranges.Generation();
            const bool pending_unmaps = virtual_ranges.HasPendingUnmaps();
            bool hit = memo->valid && memo->as_id == as_id && memo->pool_base == pool.size() &&
                       memo->buffer_generation == buffer_generation &&
                       memo->range_generation == range_generation && memo->map_serial == map_serial &&
                       !pending_unmaps && memo->inputs.size() == bindings.size();
            for (size_t i = 0; hit && i < bindings.size(); ++i) {
                const Binding& binding = *bindings[i].first;
                hit = memo->inputs[i] == typename GroupedMemo::Input{binding.gpu_addr, binding.device_addr,
                                                                     binding.size, bindings[i].second};
            }
            if (hit) {
                for (size_t i = 0; i < bindings.size(); ++i) {
                    *bindings[i].first = memo->outputs[i];
                }
                pool.insert(pool.end(), memo->segments.begin(), memo->segments.end());
                return;
            }
            memo->valid = false;
            memo->as_id = as_id;
            memo->pool_base = pool.size();
            memo->buffer_generation = buffer_generation;
            // Unmaps pending now are applied by the first Query below: then the generation moves
            // and nothing is saved. A later draw with nothing pending saves.
            memo->range_generation = pending_unmaps ? ~u64{0} : range_generation;
            memo->map_serial = map_serial;
            memo->inputs.clear();
            for (const auto& [binding, written] : bindings) {
                memo->inputs.push_back({binding->gpu_addr, binding->device_addr, binding->size, written});
            }
        }
        std::vector<StorageAddressRange>& ranges = grouped_ranges;
        ranges.clear();
        for (const auto& [binding, written] : bindings) {
            if (binding->gpu_addr > std::numeric_limits<u64>::max() - binding->size)
                throw Shader::RuntimeError("Storage address range overflow");
            ranges.push_back({binding->gpu_addr, binding->size});
            binding->segment_count = binding->segment_offset = binding->segment_size = 0;
            binding->segment_written = false;
        }
        // physical_groups[0, num_physical): reused entries (their segment lists keep their storage).
        std::vector<std::pair<VirtualSegments, bool>>& physical_groups = grouped_physical;
        size_t num_physical = 0;
        const size_t num_groups = GroupStorageRanges(ranges, grouped_order, grouped_groups);
        for (size_t group_index = 0; group_index < num_groups; ++group_index) {
            const StorageRangeGroup& group = grouped_groups[group_index];
            const u64 size = group.end - group.address;
            if (size > std::numeric_limits<u32>::max())
                throw Shader::RuntimeError("Grouped storage range exceeds supported size");
            const VirtualSegments* found = virtual_ranges.Query(*gpu_memory, group.address, static_cast<u32>(size));
            if (!found || found->empty()) throw Shader::RuntimeError("Unmapped grouped storage range");
            const bool written = std::any_of(group.members.begin(), group.members.end(),
                [&](size_t member) { return bindings[member].second; });
            for (size_t previous_index = 0; previous_index < num_physical; ++previous_index) {
                const auto& [previous, previous_written] = physical_groups[previous_index];
                if (!written && !previous_written) continue;
                for (const auto& a : previous) for (const auto& b : *found)
                    if (a.device_addr < b.device_addr + b.size && b.device_addr < a.device_addr + a.size)
                        throw Shader::RuntimeError("Writable storage aliases across disjoint virtual views");
            }
            if (num_physical == physical_groups.size()) physical_groups.emplace_back();
            physical_groups[num_physical].first = *found;
            physical_groups[num_physical].second = written;
            ++num_physical;
            if (found->size() == 1) {
                for (size_t member : group.members) {
                    auto& binding = *bindings[member].first;
                    binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
                }
                continue;
            }
            const VirtualSegments segments = *found;
            const u32 first = static_cast<u32>(pool.size());
            for (const VirtualSegment& segment : segments) {
                const BufferId id = FindBuffer(segment.device_addr, segment.size, runtime.PrefersSparseSources());
                if (!id) throw Shader::RuntimeError("Grouped storage segment allocation failed");
                pool.push_back({id, segment.device_addr, segment.size});
            }
            for (size_t member : group.members) {
                auto& binding = *bindings[member].first;
                binding.segment_first = first;
                binding.segment_count = static_cast<u32>(segments.size());
                binding.segment_offset = static_cast<u32>(binding.gpu_addr - group.address);
                binding.segment_size = static_cast<u32>(size);
                binding.segment_written = written;
            }
        }
        // Keep the result only if nothing it rests on moved meanwhile: no buffer created or
        // deleted (an id handed out early may be dead, and the caller runs everything again), no
        // unmap applied or pending and no page table change (an early Query may be stale).
        if (memo && memo->buffer_generation == buffer_generation &&
            memo->range_generation == virtual_ranges.Generation() &&
            !virtual_ranges.HasPendingUnmaps() && memo->map_serial == gpu_memory->MapSerial()) {
            memo->outputs.clear();
            for (const auto& [binding, written] : bindings) {
                memo->outputs.push_back(*binding);
            }
            memo->segments.assign(pool.begin() + memo->pool_base, pool.end());
            memo->valid = true;
        }
    }
}

template <class P>
void BufferCache<P>::RefreshMultiRangeSegments(std::vector<MultiRangeSegment>& pool) {
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        // Resolving one segment can join, and so delete, the buffer of another; repeat until
        // every segment points at a live buffer.
        for (int pass = 0; pass < 16; ++pass) {
            bool resolved = false;
            for (MultiRangeSegment& segment : pool) {
                if (!segment.buffer_id) {
                    segment.buffer_id = FindBuffer(segment.device_addr, segment.size,
                                                   runtime.PrefersSparseSources());
                    resolved = true;
                }
            }
            if (!resolved) {
                return;
            }
        }
    }
}

template <class P>
void BufferCache<P>::ResolveMultiRangeStorage(Binding& binding, bool is_written,
                                              std::vector<MultiRangeSegment>& pool) {
    binding.segment_first = 0;
    binding.segment_count = binding.segment_offset = binding.segment_size = 0;
    binding.segment_written = false;
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        if (binding.gpu_addr == 0 || binding.size == 0) {
            return;
        }
        if (is_written && !runtime.PrefersSparseSources()) {
            return;
        }
        const VirtualSegments* found =
            virtual_ranges.Query(*gpu_memory, binding.gpu_addr, binding.size);
        if (!found || found->size() < 2) {
            return;
        }
        const VirtualSegments segments = *found;
        const u32 first = static_cast<u32>(pool.size());
        const bool prefer_sparse = runtime.PrefersSparseSources();
        for (const VirtualSegment& segment : segments) {
            const BufferId buffer_id =
                FindBuffer(segment.device_addr, segment.size, prefer_sparse);
            if (!buffer_id) {
                pool.resize(first);
                return;
            }
            pool.push_back(MultiRangeSegment{
                .buffer_id = buffer_id,
                .device_addr = segment.device_addr,
                .size = segment.size,
            });
        }
        binding.segment_first = first;
        binding.segment_count = static_cast<u32>(segments.size());
    }
}

template <class P>
bool BufferCache<P>::BindMultiRangeStorage(const Binding& binding, bool is_written,
                                           std::span<const MultiRangeSegment> pool) {
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        if (binding.segment_count < 2) {
            return false;
        }
        if (binding.segment_first + binding.segment_count > pool.size()) {
            return false;
        }
        for (u32 index = 0; index < binding.segment_count; ++index) {
            const BufferId id = pool[binding.segment_first + index].buffer_id;
            if (!id || !slot_buffers.contains(id)) [[unlikely]] {
                static u32 dead_logs = 0;
                if (dead_logs < 16) {
                    ++dead_logs;
                    LOG_ERROR(HW_GPU, "Multi-range segment points at a deleted buffer_id={}",
                              id.index);
                }
                return false;
            }
        }
        const bool grouped = binding.segment_size != 0;
        is_written |= grouped && binding.segment_written;
        const u64 key = (static_cast<u64>(gpu_memory->GetID()) << 48) ^ (binding.gpu_addr - binding.segment_offset);
        runtime.ResetMultiRange();
        for (u32 index = 0; index < binding.segment_count; ++index) {
            const MultiRangeSegment& segment = pool[binding.segment_first + index];
            Buffer& buffer = slot_buffers[segment.buffer_id];
            TouchBuffer(buffer, segment.buffer_id);
            if (SynchronizeBuffer(buffer, segment.device_addr, segment.size)) {
                runtime.InvalidateMultiRange(key);
            }
            const u32 offset = buffer.Offset(segment.device_addr);
            buffer.MarkUsage(offset, segment.size);
            if (is_written) {
                MarkWrittenBuffer(segment.buffer_id, segment.device_addr, segment.size);
            }
            runtime.PushMultiRangeSource(buffer, offset, segment.size);
        }
        if constexpr (requires { runtime.UseGroupedStorage(); }) {
            if (grouped)
                return runtime.BindMultiRangeStorageBuffer(key, is_written, binding.segment_offset, binding.size);
        }
        return runtime.BindMultiRangeStorageBuffer(key, is_written);
    } else {
        return false;
    }
}

template <class P>
std::pair<DAddr, u32> BufferCache<P>::StorageWriteRange(const Binding& binding, u32 size) const {
    // write_size counts from the binding's aligned-down start plus a whole alignment (the pass
    // can't know the pointer's misalignment); the bytes written start at the pointer. On with
    // the command-list word granules (VideoCommon::ListGranulesEnabled), which need it.
    static const bool imprecise = !VideoCommon::ListGranulesEnabled();
    const u32 alignment = static_cast<u32>(runtime.GetStorageBufferAlignment());
    if (imprecise || binding.write_size == ~0U || binding.write_size < alignment ||
        binding.pointer_offset >= size) {
        return {binding.device_addr, (std::min)(size, binding.write_size)};
    }
    const u32 bytes = (std::min)(binding.write_size - alignment, size - binding.pointer_offset);
    return {binding.device_addr + binding.pointer_offset, bytes};
}

template <class P>
void BufferCache<P>::BindHostGraphicsStorageBuffers(size_t stage) {
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
        const Binding& binding = channel_state->storage_buffers[stage][index];
        const bool is_written = ((channel_state->written_storage_buffers[stage] >> index) & 1) != 0;
        if (BindMultiRangeStorage(binding, is_written, graphics_segments)) {
            return;
        }
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const u32 offset = buffer.Offset(binding.device_addr);
        buffer.MarkUsage(offset, size);

        if (is_written) {
            VideoCommon::g_dbg_mark_ctx[0] = VideoCommon::g_dbg_stage_hash;
            VideoCommon::g_dbg_mark_ctx[1] = (u64{stage} << 56) | (u64{index} << 48) | binding.dbg_cbuf;
            VideoCommon::g_dbg_mark_ctx[2] = (u64{size} << 32) | binding.write_size;
            {
                const auto& ds = maxwell3d->draw_manager.draw_state;
                const bool indexed = ds.draw_indexed;
                VideoCommon::g_dbg_mark_ctx[3] =
                    (u64{indexed} << 63) |
                    (u64{indexed ? ds.index_buffer.count : ds.vertex_buffer.count} & 0xffffffff) |
                    (u64{static_cast<u32>(ds.topology)} << 40);
            }
            const auto [write_addr, write_bytes] = StorageWriteRange(binding, size);
            MarkWrittenBuffer(binding.buffer_id, write_addr, write_bytes);
            VideoCommon::g_dbg_mark_ctx[0] = VideoCommon::g_dbg_mark_ctx[1] =
                VideoCommon::g_dbg_mark_ctx[2] = VideoCommon::g_dbg_mark_ctx[3] = 0;
        }

        if constexpr (NEEDS_BIND_STORAGE_INDEX) {
            runtime.BindStorageBuffer(stage, binding_index, buffer, offset, size, is_written);
            ++binding_index;
        } else {
            runtime.BindStorageBuffer(buffer, offset, size, is_written);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostGraphicsTextureBuffers(size_t stage) {
    ForEachEnabledBit(channel_state->enabled_texture_buffers[stage], [&](u32 index) {
        const TextureBufferBinding& binding = channel_state->texture_buffers[stage][index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const bool is_written = ((channel_state->written_texture_buffers[stage] >> index) & 1) != 0;
        if (is_written) {
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
        }

        const u32 offset = buffer.Offset(binding.device_addr);
        const PixelFormat format = binding.format;
        buffer.MarkUsage(offset, size);
        if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
            if (((channel_state->image_texture_buffers[stage] >> index) & 1) != 0) {
                runtime.BindImageBuffer(buffer, offset, size, format);
            } else {
                runtime.BindTextureBuffer(buffer, offset, size, format);
            }
        } else {
            runtime.BindTextureBuffer(buffer, offset, size, format);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostTransformFeedbackBuffers() {
    if (maxwell3d->regs.transform_feedback_enabled == 0) {
        return;
    }
    HostBindings<typename P::Buffer> host_bindings;
    for (u32 index = 0; index < NUM_TRANSFORM_FEEDBACK_BUFFERS; ++index) {
        const Binding& binding = channel_state->transform_feedback_buffers[index];
        const auto& control = maxwell3d->regs.transform_feedback.controls[index];
        const bool has_layout = control.varying_count != 0 || control.stride != 0;

        Buffer* host_buffer = &slot_buffers[NULL_BUFFER_ID];
        u32 offset = 0;
        u32 size = 0;

        if (has_layout && binding.buffer_id != NULL_BUFFER_ID && binding.size != 0) {
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            size = binding.size;
            SynchronizeBuffer(buffer, binding.device_addr, size);
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
            offset = buffer.Offset(binding.device_addr);
            buffer.MarkUsage(offset, size);
            host_buffer = &buffer;
        }

        host_bindings.buffers.push_back(host_buffer);
        host_bindings.offsets.push_back(offset);
        host_bindings.sizes.push_back(size);
    }
    runtime.BindTransformFeedbackBuffers(host_bindings);
}

template <class P>
void BufferCache<P>::BindHostComputeUniformBuffers() {
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        // Mark all uniform buffers as dirty
        channel_state->dirty_uniform_buffers.fill(~u32{0});
        channel_state->fast_bound_uniform_buffers.fill(0);
    }
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_compute_uniform_buffer_mask, [&](u32 index) {
        const Binding& binding = channel_state->compute_uniform_buffers[index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        const u32 size =
            (std::min)(binding.size, (*channel_state->compute_uniform_buffer_sizes)[index]);
        const bool has_host_buffer = binding.buffer_id != NULL_BUFFER_ID;
        const u32 offset = has_host_buffer ? buffer.Offset(binding.device_addr) : 0;
        const bool needs_alignment_stream = [&]() {
            if constexpr (IS_OPENGL) {
                return false;
            } else {
                if (!has_host_buffer) {
                    return false;
                }
                const u32 alignment = runtime.GetUniformBufferAlignment();
                return alignment > 1 && (offset % alignment) != 0;
            }
        }();
        if constexpr (!IS_OPENGL) {
            if (needs_alignment_stream) {
                const std::span<u8> span =
                    runtime.BindMappedUniformBuffer(0, binding_index, size);
                device_memory.ReadBlockUnsafe(binding.device_addr, span.data(), size);
                return;
            }
        }

        SynchronizeBuffer(buffer, binding.device_addr, size);

        buffer.MarkUsage(offset, size);
        if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
            runtime.BindComputeUniformBuffer(binding_index, buffer, offset, size);
            ++binding_index;
        } else {
            runtime.BindUniformBuffer(buffer, offset, size);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostComputeStorageBuffers() {
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_compute_storage_buffers, [&](u32 index) {
        const Binding& binding = channel_state->compute_storage_buffers[index];
        const bool is_written =
            ((channel_state->written_compute_storage_buffers >> index) & 1) != 0;
        if (BindMultiRangeStorage(binding, is_written, compute_segments)) {
            return;
        }
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        const u32 size = binding.size;
        // EDEN_N2DEN_SYNC_TRACE=<size> (diagnostic): log the tracker state around the sync of read-only
        // compute storage bindings of that byte size (first 32), to see whether guest data is
        // uploaded (Fast Fusion's DLSS table: 360448).
        static const u64 trace_size = [] {
            const char* value = std::getenv("EDEN_N2DEN_SYNC_TRACE");
            return value ? std::strtoull(value, nullptr, 0) : 0ULL;
        }();
        static int trace_left = 32;
        const bool trace = trace_size != 0 && !is_written && size == trace_size && trace_left > 0;
        bool trace_cpu = false;
        u64 trace_guest = 0;
        if (trace) {
            trace_cpu = memory_tracker.IsRegionCpuModified(binding.device_addr, size);
            device_memory.ReadBlockUnsafe(binding.device_addr, &trace_guest, sizeof(trace_guest));
        }
        const bool sync_skipped = SynchronizeBuffer(buffer, binding.device_addr, size);
        if (trace) {
            --trace_left;
            LOG_WARNING(HW_GPU,
                        "sync trace addr={:#x} size={:#x} cpu_modified={} gpu_modified={} guest_u64={:#x} "
                        "backed={} buffer={:#x}+{:#x} skipped={} tick={}",
                        binding.device_addr, size, trace_cpu,
                        memory_tracker.IsRegionGpuModified(binding.device_addr, size), trace_guest,
                        device_memory.template GetPointer<u8>(binding.device_addr) != nullptr,
                        buffer.CpuAddr(), buffer.SizeBytes(), sync_skipped, frame_tick);
        }

        const u32 offset = buffer.Offset(binding.device_addr);
        buffer.MarkUsage(offset, size);

        if (is_written) {
            const auto [write_addr, write_bytes] = StorageWriteRange(binding, size);
            MarkWrittenBuffer(binding.buffer_id, write_addr, write_bytes, true);
        }

        if constexpr (NEEDS_BIND_STORAGE_INDEX) {
            runtime.BindComputeStorageBuffer(binding_index, buffer, offset, size, is_written);
            ++binding_index;
        } else {
            runtime.BindStorageBuffer(buffer, offset, size, is_written);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostComputeTextureBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_texture_buffers, [&](u32 index) {
        const TextureBufferBinding& binding = channel_state->compute_texture_buffers[index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const bool is_written =
            ((channel_state->written_compute_texture_buffers >> index) & 1) != 0;
        if (is_written) {
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
        }

        const u32 offset = buffer.Offset(binding.device_addr);
        const PixelFormat format = binding.format;
        buffer.MarkUsage(offset, size);
        if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
            if (((channel_state->image_compute_texture_buffers >> index) & 1) != 0) {
                runtime.BindImageBuffer(buffer, offset, size, format);
            } else {
                runtime.BindTextureBuffer(buffer, offset, size, format);
            }
        } else {
            runtime.BindTextureBuffer(buffer, offset, size, format);
        }
    });
}

template <class P>
void BufferCache<P>::DoUpdateGraphicsBuffers(bool is_indexed) {
    graphics_segments.clear();
    BufferOperations([&]() {
        if (is_indexed) {
            UpdateIndexBuffer();
        }
        UpdateVertexBuffers();
        UpdateTransformFeedbackBuffers();
        for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
            UpdateUniformBuffers(stage);
            UpdateStorageBuffers(stage);
            UpdateTextureBuffers(stage);
        }
        if (current_draw_indirect) {
            UpdateDrawIndirect();
        }
        if constexpr (requires { runtime.UseGroupedStorage(); }) {
            if (runtime.UseGroupedStorage()) {
                std::vector<std::pair<Binding*, bool>>& bindings = grouped_bindings;
                bindings.clear();
                for (size_t stage = 0; stage < NUM_STAGES; ++stage)
                    ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
                        bindings.emplace_back(&channel_state->storage_buffers[stage][index],
                            ((channel_state->written_storage_buffers[stage] >> index) & 1) != 0);
                    });
                PrepareGroupedStorage(bindings, graphics_segments);
            }
        }
        RefreshMultiRangeSegments(graphics_segments);
    });
}

template <class P>
void BufferCache<P>::DoUpdateComputeBuffers() {
    compute_segments.clear();
    BufferOperations([&]() {
        UpdateComputeUniformBuffers();
        UpdateComputeStorageBuffers();
        UpdateComputeTextureBuffers();
        if constexpr (requires { runtime.UseGroupedStorage(); }) {
            if (runtime.UseGroupedStorage()) {
                std::vector<std::pair<Binding*, bool>>& bindings = grouped_bindings;
                bindings.clear();
                ForEachEnabledBit(channel_state->enabled_compute_storage_buffers, [&](u32 index) {
                    bindings.emplace_back(&channel_state->compute_storage_buffers[index],
                        ((channel_state->written_compute_storage_buffers >> index) & 1) != 0);
                });
                PrepareGroupedStorage(bindings, compute_segments);
            }
        }
        RefreshMultiRangeSegments(compute_segments);
    });
}

template <class P>
void BufferCache<P>::UpdateIndexBuffer() {
    // We have to check for the dirty flags and index count
    // The index count is currently changed without updating the dirty flags
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    const auto& index_buffer_ref = draw_state.index_buffer;
    auto& flags = maxwell3d->dirty.flags;
    if (!flags[Dirty::IndexBuffer]) {
        return;
    }
    flags[Dirty::IndexBuffer] = false;
    if (!draw_state.inline_index_draw_indexes.empty()) [[unlikely]] {
        auto inline_index_size = static_cast<u32>(draw_state.inline_index_draw_indexes.size());
        u32 buffer_size = Common::AlignUp(inline_index_size, CACHING_PAGESIZE);
        if (inline_buffer_id == NULL_BUFFER_ID) [[unlikely]] {
            LOG_INFO(HW_GPU, "Creating inline index buffer of {:#x} bytes", buffer_size);
            inline_buffer_id = CreateBuffer(0, buffer_size, false);
        }
        if (slot_buffers[inline_buffer_id].SizeBytes() < buffer_size) [[unlikely]] {
            LOG_INFO(HW_GPU, "Growing inline index buffer to {:#x} bytes", buffer_size);
            // Erasing the slot directly left its page table entries and LRU item behind, so a
            // later CreateBuffer could reuse the slot and join the new buffer with itself.
            DeleteBuffer(inline_buffer_id, true);
            inline_buffer_id = CreateBuffer(0, buffer_size, false);
        }
        channel_state->index_buffer = Binding{
            .device_addr = 0,
            .size = inline_index_size,
            .buffer_id = inline_buffer_id,
        };
        return;
    }

    const GPUVAddr gpu_addr_begin = index_buffer_ref.StartAddress();
    const GPUVAddr gpu_addr_end = index_buffer_ref.EndAddress();
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr_begin);
    const u32 address_size = static_cast<u32>(gpu_addr_end - gpu_addr_begin);
    const u32 draw_size = (index_buffer_ref.count + index_buffer_ref.first) * u32(index_buffer_ref.FormatSizeInBytes());
    const u32 size = (std::min)(address_size, draw_size);
    if (size == 0 || !device_addr) {
        channel_state->index_buffer = NULL_BINDING;
        return;
    }
    channel_state->index_buffer = Binding{
        .device_addr = *device_addr,
        .size = size,
        .buffer_id = FindBuffer(*device_addr, size, false),
    };
}

template <class P>
void BufferCache<P>::UpdateVertexBuffers() {
    auto& flags = maxwell3d->dirty.flags;
    if (!maxwell3d->dirty.flags[Dirty::VertexBuffers]) {
        return;
    }
    flags[Dirty::VertexBuffers] = false;

    for (u32 index = 0; index < NUM_VERTEX_BUFFERS; ++index) {
        UpdateVertexBuffer(index);
    }
}

template <class P>
void BufferCache<P>::UpdateVertexBuffer(u32 index) {
    if (!maxwell3d->dirty.flags[Dirty::VertexBuffer0 + index]) {
        return;
    }
    const auto& array = maxwell3d->regs.vertex_streams[index];
    const auto& limit = maxwell3d->regs.vertex_stream_limits[index];
    const GPUVAddr gpu_addr_begin = array.Address();
    const bool native = maxwell3d->GetNativeMacroEngine().native_turing;
    const u64 native_size = native ? maxwell3d->regs.NativeVertexStreamSize(index) : 0;
    if (native && array.enable != 0 && native_size > (std::numeric_limits<u32>::max)()) {
        throw std::runtime_error("Native vertex stream exceeds supported 32-bit buffer size");
    }
    const GPUVAddr gpu_addr_end = native ? gpu_addr_begin + native_size : limit.Address() + 1;
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr_begin);
    const u32 address_size = static_cast<u32>(gpu_addr_end - gpu_addr_begin);
    u32 size = address_size; // TODO: Analyze stride and number of vertices
    if (array.enable == 0 || size == 0 || !device_addr) {
        channel_state->vertex_buffers[index] = NULL_BINDING;
        UpdateVertexBufferSlot(index, NULL_BINDING);
        return;
    }
    if (!gpu_memory->IsWithinGPUAddressRange(gpu_addr_end) || size >= 64_MiB) {
        size = static_cast<u32>(gpu_memory->MaxContinuousRange(gpu_addr_begin, size));
    }
    const BufferId buffer_id = FindBuffer(*device_addr, size, false);
    const Binding binding{
        .device_addr = *device_addr,
        .size = size,
        .buffer_id = buffer_id,
    };
    channel_state->vertex_buffers[index] = binding;
    UpdateVertexBufferSlot(index, binding);
}

template <class P>
void BufferCache<P>::UpdateDrawIndirect() {
    const auto update = [this](GPUVAddr gpu_addr, size_t size, Binding& binding) {
        const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
        if (!device_addr) {
            binding = NULL_BINDING;
            return;
        }
        binding = Binding{
            .device_addr = *device_addr,
            .size = static_cast<u32>(size),
            .buffer_id = FindBuffer(*device_addr, static_cast<u32>(size), false),
        };
    };
    if (current_draw_indirect->include_count) {
        update(current_draw_indirect->count_start_address, sizeof(u32),
               channel_state->count_buffer_binding);
    }
    update(current_draw_indirect->indirect_start_address, current_draw_indirect->buffer_size,
           channel_state->indirect_buffer_binding);
}

template <class P>
void BufferCache<P>::UpdateUniformBuffers(size_t stage) {
    ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
        if (maxwell3d->GetNativeMacroEngine().native_turing && Tegra::NativeRootTables::IsBank(index)) return;
        Binding& binding = channel_state->uniform_buffers[stage][index];
        if (binding.buffer_id) {
            // Already updated
            return;
        }
        // Mark as dirty
        if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
            channel_state->dirty_uniform_buffers[stage] |= 1U << index;
        }
        // Resolve buffer
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::UpdateStorageBuffers(size_t stage) {
    ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
        // Resolve buffer
        Binding& binding = channel_state->storage_buffers[stage][index];
        if constexpr (requires { runtime.UseGroupedStorage(); }) {
            if (runtime.UseGroupedStorage()) { binding.buffer_id = NULL_BUFFER_ID; return; }
        }
        const BufferId buffer_id = FindBuffer(binding.device_addr, binding.size, false);
        binding.buffer_id = buffer_id;
        const bool is_written = ((channel_state->written_storage_buffers[stage] >> index) & 1) != 0;
        ResolveMultiRangeStorage(binding, is_written, graphics_segments);
    });
}

template <class P>
void BufferCache<P>::UpdateTextureBuffers(size_t stage) {
    ForEachEnabledBit(channel_state->enabled_texture_buffers[stage], [&](u32 index) {
        Binding& binding = channel_state->texture_buffers[stage][index];
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::UpdateTransformFeedbackBuffers() {
    if (maxwell3d->regs.transform_feedback_enabled == 0) {
        return;
    }
    for (u32 index = 0; index < NUM_TRANSFORM_FEEDBACK_BUFFERS; ++index) {
        UpdateTransformFeedbackBuffer(index);
    }
}

template <class P>
void BufferCache<P>::UpdateTransformFeedbackBuffer(u32 index) {
    const auto& binding = maxwell3d->regs.transform_feedback.buffers[index];
    const GPUVAddr gpu_addr = binding.Address() + binding.start_offset;
    const u32 size = binding.size;
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (binding.enable == 0 || size == 0 || !device_addr) {
        channel_state->transform_feedback_buffers[index] = NULL_BINDING;
        return;
    }
    const BufferId buffer_id = FindBuffer(*device_addr, size, false);
    channel_state->transform_feedback_buffers[index] = Binding{
        .device_addr = *device_addr,
        .size = size,
        .buffer_id = buffer_id,
    };
}

template <class P>
void BufferCache<P>::UpdateComputeUniformBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_uniform_buffer_mask, [&](u32 index) {
        Binding& binding = channel_state->compute_uniform_buffers[index];
        binding = NULL_BINDING;
        const auto& launch_desc = kepler_compute->launch_description;
        if (((launch_desc.const_buffer_enable_mask >> index) & 1) != 0) {
            const auto& cbuf = launch_desc.const_buffer_config[index];
            const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(cbuf.Address());
            if (device_addr) {
                binding.device_addr = *device_addr;
                binding.gpu_addr = cbuf.Address();
                binding.size = cbuf.size;
            }
        }
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::UpdateComputeStorageBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_storage_buffers, [&](u32 index) {
        // Resolve buffer
        Binding& binding = channel_state->compute_storage_buffers[index];
        if constexpr (requires { runtime.UseGroupedStorage(); }) {
            if (runtime.UseGroupedStorage()) { binding.buffer_id = NULL_BUFFER_ID; return; }
        }
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
        const bool is_written =
            ((channel_state->written_compute_storage_buffers >> index) & 1) != 0;
        ResolveMultiRangeStorage(binding, is_written, compute_segments);
    });
}

template <class P>
void BufferCache<P>::UpdateComputeTextureBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_texture_buffers, [&](u32 index) {
        Binding& binding = channel_state->compute_texture_buffers[index];
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::MarkWrittenBuffer(BufferId buffer_id, DAddr device_addr, u32 size, bool needs_sync) {
    if constexpr (!IS_OPENGL) {
        if (needs_sync) {
            Buffer& buffer = slot_buffers[buffer_id];
            buffer.setWriteTick(runtime.CurrentTick());
        }
    }
    // A draw marks the same written ranges as the draws before it (Bananza: thousands a frame,
    // ~7% of the GPU back thread). Marking a range again changes nothing until a download, CPU
    // write, flush commit or unmark clears its state (mark_generation counts those).
    // EDEN_NX2_MARK_EVERY_WRITE=1 marks every time.
    static const bool mark_every = std::getenv("EDEN_NX2_MARK_EVERY_WRITE") != nullptr;
    bool repeat = false;
    if (!mark_every && size != 0) {
        for (const RecentMark& mark : recent_marks) {
            if (mark.address == device_addr && mark.size == size &&
                mark.generation == mark_generation) {
                repeat = true;
                break;
            }
        }
    }
    if (!repeat) {
        memory_tracker.MarkRegionAsGpuModified(device_addr, size);
        gpu_modified_ranges.Add(device_addr, size);
        QueueDownload(device_addr, size);
        VideoCommon::MarkGpuWritten(device_addr, size);
        recent_marks[recent_mark_next++ % recent_marks.size()] = {device_addr, size,
                                                                 mark_generation};
    }
    Nx2NoteListWrite(buffer_id, device_addr, size);
}

template <class P>
void BufferCache<P>::Nx2NoteListWrite(BufferId buffer_id, DAddr device_addr, u64 size) {
    if constexpr (!IS_OPENGL) {
        // Pairs with MarkCommandListPage: a page becoming a list page either sees this write's
        // page bit or this sees its list bit and records the words.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (size == 0 || !VideoCommon::g_has_command_list_pages.load(std::memory_order_relaxed)) {
            return;
        }
        constexpr u64 page_bits = VideoCommon::GpuWrittenPageBits;
        const u64 last = (std::min<u64>)((device_addr + size - 1) >> page_bits,
                                         VideoCommon::g_command_list_pages.size() * 64 - 1);
        for (u64 page = device_addr >> page_bits; page <= last;) {
            // Storage buffers can be large: skip 64 pages at a time where none is a list page.
            const u64 word =
                VideoCommon::g_command_list_pages[page / 64].load(std::memory_order_relaxed) >>
                (page % 64);
            if (word == 0) {
                page = (page / 64 + 1) * 64;
                continue;
            }
            if (word & 1) {
                const DAddr page_addr = page << page_bits;
#ifdef _WIN32
                // [DEBUG] EDEN_NX2_LISTWRITE_CALLERS=1: every 2000 writes noted to command list
                // pages, the most frequent (written range size, callers) as return addresses
                // (RVA + 0x140000000, for llvm-symbolizer --obj=eden.exe).
                // [DEBUG] EDEN_NX2_LISTWRITE_BINDINGS=1: every 2000, the most frequent storage
                // bindings (see g_dbg_mark_ctx) whose written range covers a list page.
                static const bool tally_bindings =
                    std::getenv("EDEN_NX2_LISTWRITE_BINDINGS") != nullptr;
                if (tally_bindings) {
                    static std::mutex bind_mutex;
                    static std::map<std::array<u64, 6>, u32> bind_tally;
                    static u32 bind_noted = 0;
                    const std::array<u64, 6> key{size, VideoCommon::g_dbg_mark_ctx[0],
                                                 VideoCommon::g_dbg_mark_ctx[1],
                                                 VideoCommon::g_dbg_mark_ctx[2],
                                                 VideoCommon::g_dbg_mark_ctx[3],
                                                 (page_addr - (std::min)(page_addr, u64{device_addr})) & ~u64{0xffff}};
                    std::scoped_lock bind_lock{bind_mutex};
                    ++bind_tally[key];
                    if (++bind_noted % 2000 == 0) {
                        std::vector<std::pair<u32, std::array<u64, 6>>> top;
                        for (const auto& [k, v] : bind_tally) {
                            top.emplace_back(v, k);
                        }
                        std::ranges::sort(top, std::greater{});
                        std::string text;
                        for (size_t i = 0; i < top.size() && i < 8; ++i) {
                            const auto& k = top[i].second;
                            text += fmt::format(
                                " | {} x size {:#x} hash {:016x} stage {} ssbo {} c{}[{:#x}]{} bound {:#x} wsize {:#x} draw {}{} topo {} page@+{:#x}",
                                top[i].first, k[0], k[1], k[2] >> 56, (k[2] >> 48) & 0xff,
                                (k[2] >> 16) & 0x7fff, k[2] & 0xffff,
                                (k[2] & 0x80000000) ? " table" : "", k[3] >> 32, k[3] & 0xffffffff,
                                (k[4] >> 63) ? "indexed " : "", k[4] & 0xffffffff, (k[4] >> 40) & 0xff, k[5]);
                        }
                        std::map<std::pair<u64, u64>, u32> per_shader;
                        for (const auto& [k, v] : bind_tally) {
                            per_shader[{k[1], (k[2] >> 48) & 0xff}] += v;
                        }
                        std::vector<std::pair<u32, std::pair<u64, u64>>> shaders;
                        for (const auto& [k, v] : per_shader) {
                            shaders.emplace_back(v, k);
                        }
                        std::ranges::sort(shaders, std::greater{});
                        text += " || totals:";
                        for (size_t i = 0; i < shaders.size() && i < 10; ++i) {
                            text += fmt::format(" {:016x}/{}={}", shaders[i].second.first,
                                                shaders[i].second.second, shaders[i].first);
                        }
                        if (!shaders.empty()) {
                            text += " || samples of the top shader:";
                            u32 shown = 0;
                            for (const auto& [k, v] : bind_tally) {
                                if (k[1] != shaders[0].second.first || shown++ >= 6) {
                                    continue;
                                }
                                text += fmt::format(
                                    " [{}x size {:#x} c{}[{:#x}]{} bound {:#x} wsize {:#x} draw {}{} topo {} page@+{:#x}]",
                                    v, k[0], (k[2] >> 16) & 0x7fff, k[2] & 0xffff,
                                    (k[2] & 0x80000000) ? " table" : "", k[3] >> 32,
                                    k[3] & 0xffffffff, (k[4] >> 63) ? "indexed " : "",
                                    k[4] & 0xffffffff, (k[4] >> 40) & 0xff, k[5]);
                            }
                        }
                        LOG_INFO(HW_GPU, "List page binding writes (last 2000):{}", text);
                        bind_tally.clear();
                    }
                }
                static const bool tally_writes = std::getenv("EDEN_NX2_LISTWRITE_CALLERS") != nullptr;
                if (tally_writes) {
                    static std::mutex tally_mutex;
                    static std::map<std::array<u64, 7>, u32> tally;
                    static u32 noted = 0;
                    std::array<u64, 7> key{size};
                    VideoCommon::DebugCaptureCallers(key.data() + 1, 6);
                    std::scoped_lock tally_lock{tally_mutex};
                    ++tally[key];
                    if (++noted % 2000 == 0) {
                        std::vector<std::pair<u32, std::array<u64, 7>>> top;
                        for (const auto& [k, v] : tally) {
                            top.emplace_back(v, k);
                        }
                        std::ranges::sort(top, std::greater{});
                        std::string text;
                        for (size_t i = 0; i < top.size() && i < 6; ++i) {
                            text += fmt::format(" | {} x size {:#x}:", top[i].first, top[i].second[0]);
                            for (size_t f = 1; f < 7; ++f) {
                                text += fmt::format(" {:#x}", top[i].second[f]);
                            }
                        }
                        LOG_INFO(HW_GPU, "List page writes (last 2000):{}", text);
                        tally.clear();
                    }
                }
#endif
                VideoCommon::NoteListGranules(page, device_addr, device_addr + size);
                if (VideoCommon::ListRunNoteWrite(page_addr)) {
                    LOG_WARNING(HW_GPU,
                                "Command list page {:#x} was written by a draw inside a synced "
                                "list run: runs are off from here (EDEN_NX2_LIST_RUN)",
                                page_addr);
                }
                {
                    std::scoped_lock lock{nx2_list_mutex};
                    Nx2ListPage& entry = nx2_list_pages[page_addr];
                    ++entry.writes;
                    if (!nx2_collect_list_copies && !entry.draw_written) {
                        entry.draw_written = true;
                        LOG_INFO(HW_GPU,
                                 "Command list page {:#x} is written by draws too: its "
                                 "lists drain the split",
                                 page_addr);
                    }
                }
                if (nx2_collect_list_copies) {
                    nx2_list_copy_requests.emplace_back(buffer_id, page_addr);
                }
            }
            ++page;
        }
    }
}

template <class P>
bool BufferCache<P>::Nx2CopyListPages(u64 tick) {
    if constexpr (IS_OPENGL) {
        return false;
    } else {
        if (nx2_list_copy_requests.empty()) {
            return false;
        }
        std::ranges::sort(nx2_list_copy_requests, {}, &std::pair<BufferId, DAddr>::second);
        constexpr DAddr page_size = DAddr{1} << VideoCommon::GpuWrittenPageBits;
        bool recorded = false;
        for (size_t i = 0; i < nx2_list_copy_requests.size(); ++i) {
            const auto [buffer_id, page] = nx2_list_copy_requests[i];
            if (i + 1 < nx2_list_copy_requests.size() &&
                nx2_list_copy_requests[i + 1].second == page) {
                continue;
            }
            Buffer& buffer = slot_buffers[buffer_id];
            const DAddr begin = (std::max<DAddr>)(page, buffer.CpuAddr());
            const DAddr end =
                (std::min<DAddr>)(page + page_size, buffer.CpuAddr() + buffer.SizeBytes());
            if (begin >= end) {
                continue;
            }
            Nx2Staging staging = runtime.DownloadStagingBuffer(page_size, true);
            const BufferCopy copy{
                .src_offset = buffer.Offset(begin),
                .dst_offset = staging.offset + (begin - page),
                .size = end - begin,
            };
            buffer.MarkUsage(copy.src_offset, copy.size);
            runtime.CopyBuffer(staging.buffer, buffer, std::span{&copy, 1}, true);
            recorded = true;
            std::scoped_lock lock{nx2_list_mutex};
            Nx2ListPage& entry = nx2_list_pages[page];
            if (entry.copy) {
                nx2_retired_list_copies.emplace_back(entry.copy_tick, std::move(*entry.copy));
            }
            entry.copy = staging;
            entry.copy_writes = entry.writes;
            entry.copy_tick = tick;
            entry.copy_begin = static_cast<u32>(begin - page);
            entry.copy_end = static_cast<u32>(end - page);
        }
        nx2_list_copy_requests.clear();
        std::erase_if(nx2_retired_list_copies, [this](auto& retired) {
            if (!runtime.IsFree(retired.first)) {
                return false;
            }
            runtime.FreeDeferredStagingBuffer(retired.second);
            return true;
        });
        return recorded;
    }
}

template <class P>
auto BufferCache<P>::Nx2ReadListCopy(DAddr device_addr, std::span<u8> out) -> Nx2ListRead {
    if constexpr (IS_OPENGL) {
        return Nx2ListRead::Unknown;
    } else {
        constexpr DAddr page_size = DAddr{1} << VideoCommon::GpuWrittenPageBits;
        const DAddr page = device_addr & ~(page_size - 1);
        const u64 offset = device_addr - page;
        if (offset + out.size() > page_size) {
            return Nx2ListRead::Unknown;
        }
        const auto current = [&](const Nx2ListPage& entry) {
            return entry.copy && !entry.draw_written && entry.copy_writes == entry.writes &&
                   offset >= entry.copy_begin && offset + out.size() <= entry.copy_end;
        };
        u64 tick{};
        u64 writes{};
        {
            std::scoped_lock lock{nx2_list_mutex};
            const auto it = nx2_list_pages.find(page);
            if (it == nx2_list_pages.end()) {
                return Nx2ListRead::Unknown;
            }
            const Nx2ListPage& entry = it->second;
            // Pages the CPU writes the lists of (GPU-written once, long before) stay synced.
            if (!entry.draw_written && entry.synced_writes == entry.writes) {
                return Nx2ListRead::Synced;
            }
            if (!current(entry)) {
                return Nx2ListRead::Unknown;
            }
            // The lists after the first of a batch find the copy done: read it at once.
            if (runtime.IsFree(entry.copy_tick)) {
                std::memcpy(out.data(), entry.copy->mapped_span.data() + offset, out.size());
                return Nx2ListRead::Copied;
            }
            tick = entry.copy_tick;
            writes = entry.copy_writes;
        }
        // Submitted with the work that wrote the page (Nx2CopyListPages's caller flushes).
        runtime.WaitSubmittedTick(tick);
        std::scoped_lock lock{nx2_list_mutex};
        const auto it = nx2_list_pages.find(page);
        if (it == nx2_list_pages.end() || !current(it->second) || it->second.copy_tick != tick ||
            it->second.copy_writes != writes) {
            return Nx2ListRead::Unknown;
        }
        std::memcpy(out.data(), it->second.copy->mapped_span.data() + offset, out.size());
        return Nx2ListRead::Copied;
    }
}

template <class P>
void BufferCache<P>::Nx2NoteListSynced(DAddr page) {
    if constexpr (!IS_OPENGL) {
        std::scoped_lock lock{nx2_list_mutex};
        Nx2ListPage& entry = nx2_list_pages[page];
        entry.synced_writes = entry.writes;
    }
}

template <class P>
void BufferCache<P>::Nx2NoteListHostWrite(DAddr device_addr, u64 size) {
    if constexpr (!IS_OPENGL) {
        if (size == 0 || !VideoCommon::g_has_command_list_pages.load(std::memory_order_relaxed)) {
            return;
        }
        constexpr u64 page_bits = VideoCommon::GpuWrittenPageBits;
        const u64 last = (std::min<u64>)((device_addr + size - 1) >> page_bits,
                                         VideoCommon::g_command_list_pages.size() * 64 - 1);
        for (u64 page = device_addr >> page_bits; page <= last; ++page) {
            if (!((VideoCommon::g_command_list_pages[page / 64].load(std::memory_order_relaxed) >>
                   (page % 64)) &
                  1)) {
                continue;
            }
            std::scoped_lock lock{nx2_list_mutex};
            const auto it = nx2_list_pages.find(page << page_bits);
            if (it == nx2_list_pages.end()) {
                continue;
            }
            Nx2ListPage& entry = it->second;
            const bool synced = entry.synced_writes == entry.writes;
            ++entry.writes;
            if (synced) {
                entry.synced_writes = entry.writes;
            }
        }
    }
}

template <class P>
BufferId BufferCache<P>::FindBuffer(DAddr device_addr, u32 size, bool sparse_compatible) {
    if (device_addr == 0) {
        return NULL_BUFFER_ID;
    }
    const u64 page = device_addr >> CACHING_PAGEBITS;
    const BufferId buffer_id = page_table[page];
    if (buffer_id) {
        Buffer& buffer = slot_buffers[buffer_id];
        if (buffer.IsInBounds(device_addr, size)) {
            const bool should_sync = Settings::IsGPUFenceBehaviorAccurate();
            if (should_sync) {
                SynchronizeBufferWrites(buffer);
            }
            bool usable = true;
            if constexpr (requires { buffer.IsSparseCompatible(); }) {
                if (sparse_compatible && !buffer.IsSparseCompatible()) {
                    usable = false;
                }
            }
            if (usable) {
                return buffer_id;
            }
        }
    }
    return CreateBuffer(device_addr, size, sparse_compatible);
}

template <class P>
void BufferCache<P>::SynchronizeBufferWrites(Buffer& buffer) {
    if constexpr (!IS_OPENGL) {
        const u64 buffer_tick = buffer.getWriteTick();
        if (!runtime.IsFree(buffer_tick)) {
            runtime.Wait(buffer_tick);
        }
    }
}

template <class P>
typename BufferCache<P>::OverlapResult BufferCache<P>::ResolveOverlaps(DAddr device_addr,
                                                                       u32 wanted_size) {
    static constexpr int STREAM_LEAP_THRESHOLD = 16;
    boost::container::small_vector<BufferId, 16> overlap_ids;
    DAddr begin = device_addr;
    DAddr end = device_addr + wanted_size;
    int stream_score = 0;
    bool has_stream_leap = false;
    auto expand_begin = [&](DAddr add_value) {
        static constexpr DAddr min_page = CACHING_PAGESIZE + Core::DEVICE_PAGESIZE;
        if (add_value > begin - min_page) {
            begin = min_page;
            device_addr = Core::DEVICE_PAGESIZE;
            return;
        }
        begin -= add_value;
        device_addr = begin - CACHING_PAGESIZE;
    };
    auto expand_end = [&](DAddr add_value) {
        static constexpr DAddr max_page = 1ULL << Tegra::MaxwellDeviceMemoryManager::AS_BITS;
        if (add_value > max_page - end) {
            end = max_page;
            return;
        }
        end += add_value;
    };
    if (begin == 0) {
        return OverlapResult{
            .ids = std::move(overlap_ids),
            .begin = begin,
            .end = end,
            .has_stream_leap = has_stream_leap,
        };
    }
    for (; device_addr >> CACHING_PAGEBITS < Common::DivCeil(end, CACHING_PAGESIZE);
         device_addr += CACHING_PAGESIZE) {
        const BufferId overlap_id = page_table[device_addr >> CACHING_PAGEBITS];
        if (!overlap_id) {
            continue;
        }
        // A stale entry points at a deleted slot, at a slot reused by a buffer that is not
        // registered yet, or at a buffer that does not cover this page. Joining it would delete
        // (or unregister) a buffer that does not own the page, so drop the entry instead.
        const bool live = slot_buffers.contains(overlap_id);
        const Buffer* const candidate = live ? &slot_buffers[overlap_id] : nullptr;
        const bool registered = candidate && candidate->getLRUID() != SIZE_MAX;
        const bool covers = candidate && candidate->CpuAddr() <= device_addr &&
                            device_addr < candidate->CpuAddr() + candidate->SizeBytes();
        if (!registered || !covers) [[unlikely]] {
            static u32 stale_logs = 0;
            if (stale_logs < 16) {
                ++stale_logs;
                LOG_ERROR(HW_GPU,
                          "Stale buffer page: device_addr={:#x} buffer_id={} live={} "
                          "registered={} range={:#x}+{:#x}",
                          device_addr, overlap_id.index, live, registered,
                          candidate ? candidate->CpuAddr() : 0,
                          candidate ? candidate->SizeBytes() : 0);
            }
            page_table[device_addr >> CACHING_PAGEBITS] = BufferId{};
            continue;
        }
        Buffer& overlap = slot_buffers[overlap_id];
        if (overlap.IsPicked()) {
            continue;
        }
        overlap_ids.push_back(overlap_id);
        overlap.Pick();
        const DAddr overlap_device_addr = overlap.CpuAddr();
        const bool expands_left = overlap_device_addr < begin;
        if (expands_left) {
            begin = overlap_device_addr;
        }
        const DAddr overlap_end = overlap_device_addr + overlap.SizeBytes();
        const bool expands_right = overlap_end > end;
        if (overlap_end > end) {
            end = overlap_end;
        }
        stream_score += overlap.StreamScore();
        if (stream_score > STREAM_LEAP_THRESHOLD && !has_stream_leap) {
            // When this memory region has been joined a bunch of times, we assume it's being used
            // as a stream buffer. Increase the size to skip constantly recreating buffers.
            has_stream_leap = true;
            if (expands_right) {
                expand_begin(CACHING_PAGESIZE * 128);
            }
            if (expands_left) {
                expand_end(CACHING_PAGESIZE * 128);
            }
        }
    }
    return OverlapResult{
        .ids = std::move(overlap_ids),
        .begin = begin,
        .end = end,
        .has_stream_leap = has_stream_leap,
    };
}

template <class P>
void BufferCache<P>::JoinOverlap(BufferId new_buffer_id, BufferId overlap_id,
                                 bool accumulate_stream_score) {
    Buffer& new_buffer = slot_buffers[new_buffer_id];
    Buffer& overlap = slot_buffers[overlap_id];
    if (accumulate_stream_score) {
        new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
    }
    boost::container::small_vector<BufferCopy, 10> copies;
    const size_t dst_base_offset = overlap.CpuAddr() - new_buffer.CpuAddr();
    copies.push_back(BufferCopy{
        .src_offset = 0,
        .dst_offset = dst_base_offset,
        .size = overlap.SizeBytes(),
    });
    new_buffer.MarkUsage(copies[0].dst_offset, copies[0].size);
    runtime.CopyBuffer(new_buffer, overlap, copies, true);
#ifdef YUZU_LEGACY
    if (immediately_free)
        runtime.Finish();
#endif
    DeleteBuffer(overlap_id, true);
}

template <class P>
BufferId BufferCache<P>::CreateBuffer(DAddr device_addr, u32 wanted_size,
                                      bool sparse_compatible) {
    DAddr device_addr_end = Common::AlignUp(device_addr + wanted_size, CACHING_PAGESIZE);
    device_addr = Common::AlignDown(device_addr, CACHING_PAGESIZE);
    wanted_size = static_cast<u32>(device_addr_end - device_addr);
    const OverlapResult overlap = ResolveOverlaps(device_addr, wanted_size);
    const u32 size = static_cast<u32>(overlap.end - overlap.begin);
    const BufferId new_buffer_id =
        slot_buffers.insert(runtime, overlap.begin, size, sparse_compatible);
    auto& new_buffer = slot_buffers[new_buffer_id];
    const size_t size_bytes = new_buffer.SizeBytes();
    runtime.ClearBuffer(new_buffer, 0, size_bytes, 0);
    new_buffer.MarkUsage(0, size_bytes);
    for (const BufferId overlap_id : overlap.ids) {
        JoinOverlap(new_buffer_id, overlap_id, !overlap.has_stream_leap);
    }
    Register(new_buffer_id);
    TouchBuffer(new_buffer, new_buffer_id);
    return new_buffer_id;
}

template <class P>
void BufferCache<P>::Register(BufferId buffer_id) {
    ChangeRegister<true>(buffer_id);
}

template <class P>
void BufferCache<P>::Unregister(BufferId buffer_id) {
    ChangeRegister<false>(buffer_id);
}

template <class P>
template <bool insert>
void BufferCache<P>::ChangeRegister(BufferId buffer_id) {
    ++buffer_generation;
    Buffer& buffer = slot_buffers[buffer_id];
    const auto size = buffer.SizeBytes();
    const DAddr device_addr_begin = buffer.CpuAddr();
    const DAddr device_addr_end = device_addr_begin + size;
    const u64 page_begin = device_addr_begin / CACHING_PAGESIZE;
    const u64 page_end = Common::DivCeil(device_addr_end, CACHING_PAGESIZE);
    if (insert) {
        total_used_memory += Common::AlignUp(size, 1024);
        buffer.setLRUID(lru_cache.Insert(buffer_id, frame_tick));
    } else {
        total_used_memory -= Common::AlignUp(size, 1024);
        if (buffer.getLRUID() != SIZE_MAX) {
            lru_cache.Free(buffer.getLRUID());
            buffer.setLRUID(SIZE_MAX);
        }
    }
    for (u64 page = page_begin; page != page_end; ++page) {
        if constexpr (insert) {
            page_table[page] = buffer_id;
        } else if (page_table[page] == buffer_id) {
            page_table[page] = BufferId{};
        }
    }
}

template <class P>
void BufferCache<P>::TouchBuffer(Buffer& buffer, BufferId buffer_id) noexcept {
    if (buffer_id != NULL_BUFFER_ID && buffer.getLRUID() != SIZE_MAX &&
        buffer.NoteLRUTouch(frame_tick)) {
        lru_cache.Touch(buffer.getLRUID(), frame_tick);
    }
}

template <class P>
bool BufferCache<P>::SynchronizeBuffer(Buffer& buffer, DAddr device_addr, u32 size,
                                       std::source_location caller) {
    if (VideoCommon::LateWriteTrace::Enabled()) [[unlikely]] {
        memory_tracker.ForEachCpuModifiedRange(device_addr, size, [&](u64 start, u64 range) {
            VideoCommon::LateWriteTrace::NoteBackRead(start, range, caller.function_name());
        });
    }
    static const bool memo_enabled = [] {
        const char* const value = std::getenv("EDEN_SYNC_MEMO");
        return !(value && value[0] == '0');
    }();
    SyncMemoEntry* memo = nullptr;
    if (memo_enabled && size != 0) {
        if (memory_tracker.ExactEpochs()) {
            const u64 key = ((device_addr >> 4) ^ (u64{size} << 36)) * 0x9E3779B97F4A7C15ULL;
            memo = &sync_memo[key >> (64 - SYNC_MEMO_BITS)];
        } else { // EDEN_SYNC_MEMO_EXACT=0: the earlier 256 slots and hash
            const u64 key = (device_addr >> 8) ^ (u64{size} * 0x9E3779B97F4A7C15ULL >> 40);
            memo = &sync_memo[key % 256];
        }
        if (memo->device_addr == device_addr && memo->size == size &&
            memory_tracker.IsRegionCpuUnchangedSince(device_addr, size, memo->epoch)) {
            ++sync_memo_hits;
            VideoCommon::SyncCounters::Add(VideoCommon::SyncCounters::SyncMemoHits);
            return true;
        }
        VideoCommon::SyncCounters::Add(VideoCommon::SyncCounters::SyncMemoMisses);
        if (memo->device_addr != device_addr || memo->size != size) {
            VideoCommon::SyncCounters::Add(VideoCommon::SyncCounters::SyncMemoKeyMisses);
        }
        if (++sync_memo_misses % (1u << 20) == 0 && std::getenv("EDEN_SYNC_MEMO_STATS")) {
            LOG_INFO(HW_GPU, "Sync memo: {} hits, {} misses", sync_memo_hits, sync_memo_misses);
        }
    }
    // After the upload ranges are collected, the range has no CPU or untracked bit left.
    const auto remember = [&] {
        if (memo) {
            *memo = SyncMemoEntry{device_addr, size, memory_tracker.CpuEpoch()};
        }
    };
#ifdef __ANDROID__
    const bool use_optimized_vertex_buffers = Settings::values.use_optimized_vertex_buffers.GetValue();
#else
    constexpr bool use_optimized_vertex_buffers = true;
#endif

    if (use_optimized_vertex_buffers) {
        upload_copies.clear();
        u64 total_size_bytes = 0;
        u64 largest_copy = 0;
        const DAddr buffer_start = buffer.cpu_addr_cached;
        const auto add_upload = [&](DAddr start, DAddr end) {
            if (start == end) return;
            const u64 range_size = end - start;
            if (VideoCommon::UploadStats::Enabled()) [[unlikely]] {
                VideoCommon::UploadStats::NoteUpload(start, range_size);
            }
            upload_copies.push_back(BufferCopy{
                .src_offset = total_size_bytes,
                .dst_offset = start - buffer_start,
                .size = range_size,
            });
            total_size_bytes += range_size;
            largest_copy = (std::max)(largest_copy, range_size);
        };
        memory_tracker.ForEachUploadRange(device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            DAddr upload_start = device_addr_out;
            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, [&](DAddr gpu_start, DAddr gpu_end) {
                add_upload(upload_start, gpu_start);
                upload_start = gpu_end;
            });
            add_upload(upload_start, device_addr_out + range_size);
        });
        remember();
        if (total_size_bytes == 0) {
            if (memo) {
                VideoCommon::SyncCounters::Add(VideoCommon::SyncCounters::SyncMemoEmptyMisses);
            }
            return true;
        }
        const std::span<BufferCopy> copies_span(upload_copies.data(), upload_copies.size());
        UploadMemory(buffer, total_size_bytes, largest_copy, copies_span);
        any_buffer_uploaded = true;
        return false;
    } else {
        boost::container::small_vector<BufferCopy, 4> copies;
        u64 total_size_bytes = 0;
        u64 largest_copy = 0;
        DAddr buffer_start = buffer.CpuAddr();
        memory_tracker.ForEachUploadRange(device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            copies.push_back(BufferCopy{
                .src_offset = total_size_bytes,
                .dst_offset = device_addr_out - buffer_start,
                .size = range_size,
            });
            total_size_bytes += range_size;
            largest_copy = (std::max)(largest_copy, range_size);
        });
        remember();
        if (total_size_bytes == 0) {
            return true;
        }
        const std::span<BufferCopy> copies_span(copies.data(), copies.size());
        UploadMemory(buffer, total_size_bytes, largest_copy, copies_span);
        return false;
    }
}

template <class P>
void BufferCache<P>::UploadMemory(Buffer& buffer, u64 total_size_bytes, u64 largest_copy,
                                  std::span<BufferCopy> copies) {
    if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
        MappedUploadMemory(buffer, total_size_bytes, copies);
    } else {
        ImmediateUploadMemory(buffer, largest_copy, copies);
    }
}

template <class P>
void BufferCache<P>::ImmediateUploadMemory([[maybe_unused]] Buffer& buffer,
                                           [[maybe_unused]] u64 largest_copy,
                                           [[maybe_unused]] std::span<const BufferCopy> copies) {
    if constexpr (!USE_MEMORY_MAPS_FOR_UPLOADS) {
        std::span<u8> immediate_buffer;
        for (const BufferCopy& copy : copies) {
            std::span<const u8> upload_span;
            const DAddr device_addr = buffer.CpuAddr() + copy.dst_offset;
            if (IsRangeGranular(device_addr, copy.size)) {
                auto* const ptr = device_memory.GetPointer<u8>(device_addr);
                if (ptr != nullptr) {
                    upload_span = std::span(ptr, copy.size);
                }
            } else {
                if (immediate_buffer.empty()) {
                    immediate_buffer = ImmediateBuffer(largest_copy);
                }
                device_memory.ReadBlockUnsafe(device_addr, immediate_buffer.data(), copy.size);
                upload_span = immediate_buffer.subspan(0, copy.size);
            }
            buffer.ImmediateUpload(copy.dst_offset, upload_span);
        }
    }
}

template <class P>
void BufferCache<P>::MappedUploadMemory([[maybe_unused]] Buffer& buffer,
                                        [[maybe_unused]] u64 total_size_bytes,
                                        [[maybe_unused]] std::span<BufferCopy> copies) {
    if constexpr (USE_MEMORY_MAPS) {
        auto upload_staging = runtime.UploadStagingBuffer(total_size_bytes);
        const std::span<u8> staging_pointer = upload_staging.mapped_span;
        const bool snapshot = VideoCommon::FrameSnapshot::Part(2);
        struct SnapshotUpload {
            DAddr address;
            u64 size;
            u32 tag;
        };
        boost::container::small_vector<SnapshotUpload, 4> from_snapshot;
        for (BufferCopy& copy : copies) {
            u8* const src_pointer = staging_pointer.data() + copy.src_offset;
            const DAddr device_addr = buffer.CpuAddr() + copy.dst_offset;
            device_memory.ReadBlockUnsafe(device_addr, src_pointer, copy.size);
            if (snapshot) {
                // [EXPERIMENT] The data as it was for the frame the back is on.
                u32 tag = 0;
                if (VideoCommon::FrameSnapshot::Overlay(device_addr, src_pointer, copy.size, tag)) {
                    from_snapshot.push_back({device_addr, copy.size, tag});
                }
            }
            // Apply the staging offset
            copy.src_offset += upload_staging.offset;
        }
        const bool can_reorder = runtime.CanReorderUpload(buffer, copies);
        runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true, can_reorder);
        for (const SnapshotUpload& upload : from_snapshot) {
            // Uploaded older than guest memory: marked CPU-written again once the back finished
            // the copy's frame (TakeDue), then uploaded at the next bind. Marking it now made
            // every later bind of the frame upload it again (48 MiB a frame in Kokiri Forest).
            VideoCommon::FrameSnapshot::Defer(upload.address, upload.size, upload.tag);
        }
    }
}

template <class P>
bool BufferCache<P>::InlineMemory(DAddr dest_address, size_t copy_size,
                                  std::span<const u8> inlined_buffer) {
    const bool is_dirty = IsRegionRegistered(dest_address, copy_size);
    if (!is_dirty) {
        return false;
    }
    DAddr aligned_start = Common::AlignDown(dest_address, DEVICE_PAGESIZE);
    DAddr aligned_end = Common::AlignUp(dest_address + copy_size, DEVICE_PAGESIZE);
    if (!IsRegionGpuModified(aligned_start, aligned_end - aligned_start)) {
        return false;
    }

    InlineMemoryImplementation(dest_address, copy_size, inlined_buffer);

    return true;
}

template <class P>
void BufferCache<P>::InlineMemoryImplementation(DAddr dest_address, size_t copy_size,
                                                std::span<const u8> inlined_buffer) {
    ClearDownload(dest_address, copy_size);
    ++mark_generation; // the marks MarkWrittenBuffer skips may be gone
    gpu_modified_ranges.Subtract(dest_address, copy_size);

    BufferId buffer_id = FindBuffer(dest_address, static_cast<u32>(copy_size), false);
    auto& buffer = slot_buffers[buffer_id];
    SynchronizeBuffer(buffer, dest_address, static_cast<u32>(copy_size));

    if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
        auto upload_staging = runtime.UploadStagingBuffer(copy_size);
        std::array copies{BufferCopy{
            .src_offset = upload_staging.offset,
            .dst_offset = buffer.Offset(dest_address),
            .size = copy_size,
        }};
        u8* const src_pointer = upload_staging.mapped_span.data();
        std::memcpy(src_pointer, inlined_buffer.data(), copy_size);
        const bool can_reorder = runtime.CanReorderUpload(buffer, copies);
        runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true, can_reorder);
    } else {
        buffer.ImmediateUpload(buffer.Offset(dest_address), inlined_buffer.first(copy_size));
    }
}

template <class P>
void BufferCache<P>::DownloadBufferMemory(Buffer& buffer) {
    DownloadBufferMemory(buffer, buffer.CpuAddr(), buffer.SizeBytes());
}

template <class P>
void BufferCache<P>::DownloadBufferMemory(Buffer& buffer, DAddr device_addr, u64 size) {
    boost::container::small_vector<BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    u64 largest_copy = 0;
    ++mark_generation; // the marks MarkWrittenBuffer skips may be gone
    memory_tracker.ForEachDownloadRangeAndClear(
        device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            const DAddr buffer_addr = buffer.CpuAddr();
            const auto add_download = [&](DAddr start, DAddr end) {
                const u64 new_offset = start - buffer_addr;
                const u64 new_size = end - start;
                copies.push_back(BufferCopy{
                    .src_offset = new_offset,
                    .dst_offset = total_size_bytes,
                    .size = new_size,
                });
                // Align up to avoid cache conflicts
                constexpr u64 align = 64ULL;
                constexpr u64 mask = ~(align - 1ULL);
                total_size_bytes += (new_size + align - 1) & mask;
                largest_copy = (std::max)(largest_copy, new_size);
            };

            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, add_download);
            ClearDownload(device_addr_out, range_size);
            ++mark_generation; // the marks MarkWrittenBuffer skips may be gone
            gpu_modified_ranges.Subtract(device_addr_out, range_size);
        });
    if (total_size_bytes == 0) {
        return;
    }

    if constexpr (USE_MEMORY_MAPS) {
        auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes);
        const u8* const mapped_memory = download_staging.mapped_span.data();
        const std::span<BufferCopy> copies_span(copies.data(), copies.data() + copies.size());
        for (BufferCopy& copy : copies) {
            // Modify copies to have the staging offset in mind
            copy.dst_offset += download_staging.offset;
            buffer.MarkUsage(copy.src_offset, copy.size);
        }
        runtime.CopyBuffer(download_staging.buffer, buffer, copies_span, true);
        runtime.Finish();
        for (const BufferCopy& copy : copies) {
            const DAddr copy_device_addr = buffer.CpuAddr() + copy.src_offset;
            // Undo the modified offset
            const u64 dst_offset = copy.dst_offset - download_staging.offset;
            const u8* copy_mapped_memory = mapped_memory + dst_offset;
            device_memory.WriteBlockUnsafe(copy_device_addr, copy_mapped_memory, copy.size);
            VideoCommon::FrameSnapshot::OnGpuWrite(copy_device_addr, copy_mapped_memory, copy.size);
        }
    } else {
        const std::span<u8> immediate_buffer = ImmediateBuffer(largest_copy);
        for (const BufferCopy& copy : copies) {
            buffer.ImmediateDownload(copy.src_offset, immediate_buffer.subspan(0, copy.size));
            const DAddr copy_device_addr = buffer.CpuAddr() + copy.src_offset;
            device_memory.WriteBlockUnsafe(copy_device_addr, immediate_buffer.data(), copy.size);
            VideoCommon::FrameSnapshot::OnGpuWrite(copy_device_addr, immediate_buffer.data(),
                                                   copy.size);
        }
    }
}

template <class P>
void BufferCache<P>::DeleteBuffer(BufferId buffer_id, bool do_not_mark) {
    if (buffer_id == inline_buffer_id) {
        // The garbage collector can delete the inline index buffer; recreate it on next use.
        inline_buffer_id = NULL_BUFFER_ID;
    }
    if constexpr (requires { runtime.OnBufferDeleted(slot_buffers[buffer_id]); }) {
        runtime.OnBufferDeleted(slot_buffers[buffer_id]);
    }
    bool dirty_index{false};
    boost::container::small_vector<u64, NUM_VERTEX_BUFFERS> dirty_vertex_buffers;
    const auto scalar_replace = [buffer_id](Binding& binding) {
        if (binding.buffer_id == buffer_id) {
            binding.buffer_id = BufferId{};
        }
    };
    const auto replace = [scalar_replace](std::span<Binding> bindings) {
        std::ranges::for_each(bindings, scalar_replace);
    };

    if (channel_state->index_buffer.buffer_id == buffer_id) {
        channel_state->index_buffer.buffer_id = BufferId{};
        dirty_index = true;
    }

    for (u32 index = 0; index < channel_state->vertex_buffers.size(); index++) {
        auto& binding = channel_state->vertex_buffers[index];
        if (binding.buffer_id == buffer_id) {
            binding.buffer_id = BufferId{};
            UpdateVertexBufferSlot(index, binding);
            dirty_vertex_buffers.push_back(index);
        }
    }
    std::ranges::for_each(channel_state->uniform_buffers, replace);
    std::ranges::for_each(channel_state->storage_buffers, replace);
    replace(channel_state->transform_feedback_buffers);
    replace(channel_state->compute_uniform_buffers);
    replace(channel_state->compute_storage_buffers);
    // Multi-range segments hold buffer ids outside the bindings; a join later in the same update
    // can delete a buffer an earlier segment resolved to.
    for (auto* pool : {&graphics_segments, &compute_segments}) {
        for (MultiRangeSegment& segment : *pool) {
            if (segment.buffer_id == buffer_id) {
                segment.buffer_id = BufferId{};
            }
        }
    }
    // The other channels keep their own bindings, and their resolved ids survive across draws.
    // Left alone, a stale id addresses whatever buffer reuses this slot next (Ocarina of Time
    // draws from several channels; the GPU faulted after the Kokiri Forest save load).
    // From Eglantine 43b6b17c3a (AManWithAPlan).
    for (const size_t other_id : active_channel_ids) {
        BufferCacheChannelInfo& other = channel_storage[other_id];
        if (&other == channel_state) {
            continue;
        }
        auto& other_flags = other.maxwell3d.dirty.flags;
        if (other.index_buffer.buffer_id == buffer_id) {
            other.index_buffer.buffer_id = BufferId{};
            other_flags[Dirty::IndexBuffer] = true;
        }
        for (u32 index = 0; index < other.vertex_buffers.size(); index++) {
            if (other.vertex_buffers[index].buffer_id == buffer_id) {
                other.vertex_buffers[index].buffer_id = BufferId{};
                other_flags[Dirty::VertexBuffers] = true;
                other_flags[Dirty::VertexBuffer0 + index] = true;
            }
        }
        std::ranges::for_each(other.uniform_buffers, replace);
        std::ranges::for_each(other.storage_buffers, replace);
        replace(other.transform_feedback_buffers);
        replace(other.compute_uniform_buffers);
        replace(other.compute_storage_buffers);
        if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
            other.dirty_uniform_buffers.fill(~u32{0});
            other.uniform_buffer_binding_sizes.fill({});
        }
        other.has_deleted_buffers = true;
    }

    // Mark the whole buffer as CPU written to stop tracking CPU writes
    if (!do_not_mark) {
        Buffer& buffer = slot_buffers[buffer_id];
        ++mark_generation; // the marks MarkWrittenBuffer skips may be gone
        memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddr(), buffer.SizeBytes());
    }

    Unregister(buffer_id);

#ifdef YUZU_LEGACY
    if (!do_not_mark || !immediately_free)
#endif
        delayed_destruction_ring.Push(std::move(slot_buffers[buffer_id]));

    slot_buffers.erase(buffer_id);

    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        channel_state->dirty_uniform_buffers.fill(~u32{0});
        channel_state->uniform_buffer_binding_sizes.fill({});
    }

    auto& flags = maxwell3d->dirty.flags;
    if (dirty_index) {
        flags[Dirty::IndexBuffer] = true;
    }

    if (dirty_vertex_buffers.size() > 0) {
        flags[Dirty::VertexBuffers] = true;
        for (auto index : dirty_vertex_buffers) {
            flags[Dirty::VertexBuffer0 + index] = true;
        }
    }
    channel_state->has_deleted_buffers = true;
}

template <class P>
Binding BufferCache<P>::StorageBufferBinding(GPUVAddr ssbo_addr, u32 cbuf_index, bool is_written,
                                             bool native_pointer, bool table_pointer) const {
    // [EXPERIMENT] The pointer as it was for the frame the back is on (frame_snapshot.h).
    const GPUVAddr gpu_addr = VideoCommon::FrameSnapshot::ReadGuest<u64>(*gpu_memory, ssbo_addr);

    if (gpu_addr == 0) {
        return NULL_BINDING;
    }

    const u64 following_word =
        native_pointer ? 0 : VideoCommon::FrameSnapshot::ReadGuest<u64>(*gpu_memory, ssbo_addr + 8);
    // Every draw rebinds each storage buffer: the layout lookup (a mutex), the contiguous-range
    // lookup (another) and two page walks cost Ocarina of Time's GPU back thread ~5-7% in Hyrule
    // Field. The answer depends only on these inputs and the page table, so it is kept per thread
    // until the page table changes (after Eglantine c8415d0e80 by Matt Artist).
    // EDEN_N2DEN_BINDING_CACHE=0 looks every binding up again.
    struct CachedBinding {
        size_t owner; ///< MemoryManager id + 1, 0 when empty
        GPUVAddr gpu_addr;
        u64 following_word;
        u64 generation;
        u32 flags;
        Binding binding;
    };
    static constexpr size_t Ways = 4;
    static const bool cache_enabled = [] {
        const char* const value = std::getenv("EDEN_N2DEN_BINDING_CACHE");
        return !(value && value[0] == '0');
    }();
    thread_local std::array<std::array<CachedBinding, Ways>, 512> binding_cache{};
    const u32 flags = (is_written ? 1U : 0U) | (native_pointer ? 2U : 0U) | (table_pointer ? 4U : 0U);
    const size_t owner = gpu_memory->GetID() + 1;
    const u64 generation = gpu_memory->MappingGeneration();
    auto& set = binding_cache[((gpu_addr ^ following_word ^ flags) * 0x9E3779B97F4A7C15ULL) >> 55];
    if (cache_enabled) {
        for (size_t way = 0; way < Ways; ++way) {
            const CachedBinding& cached = set[way];
            if (cached.owner == owner && cached.gpu_addr == gpu_addr &&
                cached.following_word == following_word && cached.flags == flags &&
                cached.generation == generation) {
                const Binding hit = cached.binding;
                std::rotate(set.begin(), set.begin() + way, set.begin() + way + 1);
                return hit;
            }
        }
    }
    const Binding binding = StorageBufferBindingUncached(gpu_addr, following_word, cbuf_index,
                                                         is_written, native_pointer, table_pointer);
    if (cache_enabled) {
        std::rotate(set.begin(), set.end() - 1, set.end());
        set[0] = {owner, gpu_addr, following_word, generation, flags, binding};
    }
    return binding;
}

template <class P>
Binding BufferCache<P>::StorageBufferBindingUncached(GPUVAddr gpu_addr, u64 following_word,
                                                     u32 cbuf_index, bool is_written,
                                                     bool native_pointer,
                                                     bool table_pointer) const {
    const u64 mapped_size = gpu_memory->GetMemoryLayoutSize(gpu_addr);
    u32 size = StorageBufferSize(mapped_size, following_word, native_pointer);
    // A pointer read from a table in memory is the base of an object pool, read anywhere in it
    // (Ocarina of Time's grass instances lie 12-18 MiB past theirs): bound to the end of its
    // mapping, up to 64 MiB (upstream 4ac9e703fe). Written pools keep the cap.
    if (table_pointer && !is_written) {
        size = static_cast<u32>((std::min)(mapped_size, u64{64_MiB}));
    }
    // A native (CUDA/Ampere) kernel's bare pointer reaches as far as the tensor it addresses: Fast
    // Fusion's DLSS network output in Ultra Quality is 480x288 pixels of 80 bytes (11 MB), and with
    // the 8 MiB guess its bottom 24% fell outside the binding for writer and reader alike (ghosting
    // on the player car). Bound native pointers to the end of their mapping up to 32 MiB; the
    // device-contiguous clamp below still keeps them inside one memory pool.
    // EDEN_N2DEN_NATIVE_SSBO_MIB overrides the limit (8 restores the old guess).
    if (native_pointer && !table_pointer) {
        static const u64 native_limit = [] {
            const char* value = std::getenv("EDEN_N2DEN_NATIVE_SSBO_MIB");
            const u64 mib = value ? std::strtoull(value, nullptr, 0) : 0;
            return (mib != 0 ? mib : 32) * 1_MiB;
        }();
        size = static_cast<u32>((std::min)(mapped_size, native_limit));
    }
    if ((native_pointer || table_pointer) && size) {
        // A bare pointer's size is a guess (up to 8 MiB of mapped GPU VA). NVN buffers live in one
        // memory pool, which is one device-contiguous allocation, so the guess must not run into
        // the next pool: a binding spanning two device ranges forces a whole-range gather/scatter
        // around every dispatch.
        size = static_cast<u32>(gpu_memory->DeviceContiguousSize(gpu_addr, size));
    }

    // Alignment only applies to the offset of the buffer
    const u32 alignment = runtime.GetStorageBufferAlignment();
    const GPUVAddr aligned_gpu_addr = Common::AlignDown(gpu_addr, alignment);
    const u32 aligned_size = static_cast<u32>(gpu_addr - aligned_gpu_addr) + size;

    const std::optional<DAddr> aligned_device_addr = gpu_memory->GpuToCpuAddress(aligned_gpu_addr);
    if (!aligned_device_addr || size == 0) {
        LOG_DEBUG(HW_GPU, "Failed to find storage buffer for cbuf index {}", cbuf_index);
        return NULL_BINDING;
    }
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    ASSERT_MSG(device_addr, "Unaligned storage buffer address not found for cbuf index {}",
               cbuf_index);
    // The end address used for size calculation does not need to be aligned
    const DAddr cpu_end = Common::AlignUp(*device_addr + size, Core::DEVICE_PAGESIZE);

    u32 binding_size = static_cast<u32>(cpu_end - *aligned_device_addr);
    if (is_written) {
        binding_size = aligned_size;
    }
    const Binding binding{
        .device_addr = *aligned_device_addr,
        .gpu_addr = aligned_gpu_addr,
        .size = binding_size,
        .buffer_id = BufferId{},
        .pointer_offset = static_cast<u32>(gpu_addr - aligned_gpu_addr),
    };
    return binding;
}

template <class P>
TextureBufferBinding BufferCache<P>::GetTextureBufferBinding(GPUVAddr gpu_addr, u32 size,
                                                             PixelFormat format) {
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    TextureBufferBinding binding;
    if (!device_addr || size == 0) {
        binding.device_addr = 0;
        binding.size = 0;
        binding.buffer_id = NULL_BUFFER_ID;
        binding.format = PixelFormat::Invalid;
    } else {
        binding.device_addr = *device_addr;
        binding.size = size;
        binding.buffer_id = BufferId{};
        binding.format = format;
    }
    return binding;
}

template <class P>
std::span<const u8> BufferCache<P>::ImmediateBufferWithData(DAddr device_addr, size_t size) {
    u8* const base_pointer = device_memory.GetPointer<u8>(device_addr);
    if (IsRangeGranular(device_addr, size) ||
        base_pointer + size == device_memory.GetPointer<u8>(device_addr + size)) {
        return std::span(base_pointer, size);
    } else {
        const std::span<u8> span = ImmediateBuffer(size);
        device_memory.ReadBlockUnsafe(device_addr, span.data(), size);
        return span;
    }
}

template <class P>
std::span<u8> BufferCache<P>::ImmediateBuffer(size_t wanted_capacity) {
    immediate_buffer_alloc.resize_destructive(wanted_capacity);
    return std::span<u8>(immediate_buffer_alloc.data(), wanted_capacity);
}

template <class P>
bool BufferCache<P>::HasFastUniformBufferBound(size_t stage, u32 binding_index) const noexcept {
    return ((channel_state->fast_bound_uniform_buffers[stage] >> binding_index) & 1u) != 0;
}

template <class P>
std::pair<typename BufferCache<P>::Buffer*, u32> BufferCache<P>::GetDrawIndirectCount() {
    auto& buffer = slot_buffers[channel_state->count_buffer_binding.buffer_id];
    return std::make_pair(&buffer, buffer.Offset(channel_state->count_buffer_binding.device_addr));
}

template <class P>
std::pair<typename BufferCache<P>::Buffer*, u32> BufferCache<P>::GetDrawIndirectBuffer() {
    auto& buffer = slot_buffers[channel_state->indirect_buffer_binding.buffer_id];
    return std::make_pair(&buffer,
                          buffer.Offset(channel_state->indirect_buffer_binding.device_addr));
}

} // namespace VideoCommon
