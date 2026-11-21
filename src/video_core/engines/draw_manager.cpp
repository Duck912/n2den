// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstring>
#include <fstream>
#include <atomic>
#include <cstdlib>
#include <algorithm>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include "common/emu_counters.h"
#include "common/n2den_trace.h"
#include "common/settings.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/texture_cache/compact_texture_table.h"
#include "video_core/memory_manager.h"

namespace Tegra::Engines {

namespace {

// [DEBUG] EDEN_N2DEN_DRAW_TRACE=1 (with EDEN_NX2_FRAMES_AHEAD=0, which splits frames at presents;
// EDEN_GPU_SPLIT=1 keeps the predicate macros in order with the draws): finds draws skipped in one
// frame and drawn in the frames before and after, i.e. an object missing for exactly one frame.
// A draw is keyed by its index or vertex range, instance range, first vertex stream and the bound
// vertex and fragment program offsets, which stay the same while the camera moves. Each frame with
// such draws gets one line with the count per reason (render-enable condition false, or pipeline
// still translating/building), and up to four of them with their data and the occlusion report
// the predicate macro read last before them.
class DrawFlickerTrace {
public:
    enum Outcome : u8 { Drawn = 1, ConditionSkip = 2, PipelineSkip = 4 };
    struct Info {
        GPUVAddr index_address{};
        GPUVAddr stream0{};
        u32 first{};
        u32 count{};
        u32 instances{};
        u32 vertex_program{};
        u32 fragment_program{};
        u64 predicate_slot{};
        u32 predicate_count{};
        u64 predicate_stamp{};
        u64 target{};
        GPUVAddr zeta{};
        u32 zeta_width{};
        u32 zeta_height{};
        u32 zeta_format{};
        GPUVAddr color0{};
        u32 color0_width{};
        u32 color0_height{};
        u32 color0_format{};
        u32 color_count{};
    };

    static bool Enabled() {
        static const bool enabled = std::getenv("EDEN_N2DEN_DRAW_TRACE") != nullptr;
        return enabled;
    }

    void OnDraw(u64 key, u8 outcome, const Info& info) {
        std::scoped_lock lock{mutex};
        // With the luma trace the frames are GPU emulation's presents (luma frame numbers minus
        // one), which also split frames when the present runs ahead.
        const u32 back = Common::N2denTrace::back_frames.load(std::memory_order_acquire);
        const u32 present =
            back != 0 ? back : Common::N2denTrace::present_count.load(std::memory_order_acquire);
        if (present != frame_present) {
            EndFrame();
            frame_present = present;
        }
        auto [it, inserted] = current.try_emplace(key, Entry{0, info});
        if (outcome != Drawn && (it->second.outcomes & ~Drawn) == 0) {
            it->second.info = info;
        }
        it->second.outcomes |= outcome;
        auto [target, new_target] = current_targets.try_emplace(info.target, TargetStats{0, 0, info});
        if (outcome == ConditionSkip) {
            ++target->second.skipped;
        } else if (outcome == Drawn) {
            ++target->second.drawn;
        }
        ++stats.draws;
        stats.condition += outcome == ConditionSkip ? 1 : 0;
        stats.pipeline += outcome == PipelineSkip ? 1 : 0;
    }

private:
    struct Entry {
        u8 outcomes;
        Info info;
    };

    void EndFrame() {
        if (!older.empty() && !middle.empty() && !current.empty()) {
            u32 by_condition = 0;
            u32 by_pipeline = 0;
            std::vector<const Entry*> samples;
            for (const auto& [key, entry] : middle) {
                if ((entry.outcomes & Drawn) != 0) {
                    continue;
                }
                const auto before = older.find(key);
                const auto after = current.find(key);
                if (before == older.end() || after == current.end() ||
                    (before->second.outcomes & Drawn) == 0 || (after->second.outcomes & Drawn) == 0) {
                    continue;
                }
                if ((entry.outcomes & ConditionSkip) != 0) {
                    ++by_condition;
                } else {
                    ++by_pipeline;
                }
                if (samples.size() < 4) {
                    samples.push_back(&entry);
                }
            }
            if (by_condition + by_pipeline != 0) {
                stats.missing_frames += 1;
                stats.missing_condition += by_condition;
                stats.missing_pipeline += by_pipeline;
                ++frames_logged;
                if (frames_logged <= 600 || frames_logged % 20 == 0) {
                    LOG_INFO(HW_GPU,
                             "One-frame missing draws: frame {} skipped {} draws that the frames before "
                             "and after drew ({} by the render-enable condition, {} while the "
                             "pipeline built); {} draw keys in the frame",
                             middle_present, by_condition + by_pipeline, by_condition, by_pipeline,
                             middle.size());
                    for (const Entry* entry : samples) {
                        const Info& i = entry->info;
                        const char* const reason =
                            (entry->outcomes & ConditionSkip) != 0 ? "condition" : "pipeline";
                        LOG_INFO(HW_GPU,
                                 "  missing ({}): index {:#x} stream0 {:#x} first {} count {} x{} "
                                 "programs vs {:#x} fs {:#x}; last predicate read: slot {:#x} count {} "
                                 "timestamp {:#x}",
                                 reason, i.index_address, i.stream0, i.first, i.count, i.instances,
                                 i.vertex_program, i.fragment_program, i.predicate_slot,
                                 i.predicate_count, i.predicate_stamp);
                    }
                }
            }
        }
        if (!older.empty() && !middle.empty() && !current.empty()) {
            // Draws made in this frame only: drawn here while the frames before and after skipped
            // them by condition (shown), or did not have them at all (new).
            u32 shown = 0;
            u32 fresh = 0;
            std::vector<std::pair<const Entry*, const char*>> samples;
            for (const auto& [key, entry] : middle) {
                if ((entry.outcomes & Drawn) == 0) {
                    continue;
                }
                const auto before = older.find(key);
                const auto after = current.find(key);
                const bool before_drawn =
                    before != older.end() && (before->second.outcomes & Drawn) != 0;
                const bool after_drawn =
                    after != current.end() && (after->second.outcomes & Drawn) != 0;
                if (before_drawn || after_drawn) {
                    continue;
                }
                const char* kind = nullptr;
                if (before != older.end() && after != current.end()) {
                    ++shown;
                    kind = "shown";
                } else if (before == older.end() && after == current.end()) {
                    ++fresh;
                    kind = "new";
                } else {
                    continue;
                }
                if (samples.size() < 6) {
                    samples.push_back({&entry, kind});
                }
            }
            if (shown + fresh != 0) {
                ++extra_logged;
                if (extra_logged <= 2000) {
                    LOG_INFO(HW_GPU,
                             "One-frame extra draws: luma frame {} drew {} draws the frames before "
                             "and after skipped by condition and {} they did not have; {} draw "
                             "keys in the frame",
                             middle_present + 1, shown, fresh, middle.size());
                    for (const auto& [entry, kind] : samples) {
                        const Info& i = entry->info;
                        LOG_INFO(HW_GPU,
                                 "  extra ({}): index {:#x} stream0 {:#x} first {} count {} x{} "
                                 "programs vs {:#x} fs {:#x}; color0 {:#x} {}x{} format {}, depth "
                                 "{:#x}; last predicate read: slot {:#x} count {}",
                                 kind, i.index_address, i.stream0, i.first, i.count, i.instances,
                                 i.vertex_program, i.fragment_program, i.color0, i.color0_width,
                                 i.color0_height, i.color0_format, i.zeta, i.predicate_slot,
                                 i.predicate_count);
                    }
                }
            }
        }
        const u64 flush_waits = Common::N2denTrace::query_flush_waits.load(std::memory_order_relaxed);
        const u64 frame_flushes = flush_waits - last_flush_waits;
        last_flush_waits = flush_waits;
        if (!older_targets.empty() && !middle_targets.empty() && !current_targets.empty()) {
            for (const auto& [key, mid] : middle_targets) {
                const auto before = older_targets.find(key);
                const auto after = current_targets.find(key);
                if (before == older_targets.end() || after == current_targets.end()) {
                    continue;
                }
                const u32 drawn_around = (std::min)(before->second.drawn, after->second.drawn);
                const u32 skipped_around = (std::max)(before->second.skipped, after->second.skipped);
                if (mid.skipped < skipped_around + 3 || mid.drawn + 3 > drawn_around) {
                    continue;
                }
                ++stats.target_spikes;
                ++target_logged;
                if (target_logged <= 600 || target_logged % 20 == 0) {
                    const Info& i = mid.info;
                    LOG_INFO(HW_GPU,
                             "Target lost draws for one frame: frame {} drew {} and skipped {} by "
                             "condition (before {}/{}, after {}/{}); depth {:#x} {}x{} format {}, "
                             "color0 {:#x} {}x{} format {}, {} color targets; query flush waits in "
                             "the frame {}",
                             middle_present, mid.drawn, mid.skipped, before->second.drawn,
                             before->second.skipped, after->second.drawn, after->second.skipped,
                             i.zeta, i.zeta_width, i.zeta_height, i.zeta_format, i.color0,
                             i.color0_width, i.color0_height, i.color0_format, i.color_count,
                             middle_flushes);
                }
            }
        }
        for (const auto& [key, t] : middle_targets) {
            auto& total = target_totals[key];
            total.drawn += t.drawn;
            total.skipped += t.skipped;
            total.info = t.info;
        }
        if (++stats.frames >= 300) {
            std::vector<std::pair<u64, TargetStats>> top(target_totals.begin(), target_totals.end());
            std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) {
                return a.second.skipped > b.second.skipped;
            });
            for (size_t n = 0; n < (std::min)(top.size(), size_t{5}); ++n) {
                const auto& t = top[n].second;
                if (t.skipped == 0) {
                    break;
                }
                LOG_INFO(HW_GPU,
                         "  target with condition skips: depth {:#x} {}x{} format {}, color0 {:#x} "
                         "{}x{} format {}, {} color targets: {} drawn, {} skipped",
                         t.info.zeta, t.info.zeta_width, t.info.zeta_height, t.info.zeta_format,
                         t.info.color0, t.info.color0_width, t.info.color0_height,
                         t.info.color0_format, t.info.color_count, t.drawn, t.skipped);
            }
            target_totals.clear();
            LOG_INFO(HW_GPU,
                     "Draw trace, last {} frames: target one-frame losses {};", stats.frames,
                     stats.target_spikes);
            LOG_INFO(HW_GPU,
                     "Draw trace, last {} frames: {:.0f} draws a frame, skips {} by condition and {} "
                     "by pipeline; frames with one-frame missing draws {} ({} by condition, {} by "
                     "pipeline)",
                     stats.frames, static_cast<double>(stats.draws) / stats.frames, stats.condition,
                     stats.pipeline, stats.missing_frames, stats.missing_condition,
                     stats.missing_pipeline);
            stats = Stats{};
        }
        older = std::move(middle);
        middle = std::move(current);
        middle_present = frame_present;
        current.clear();
        older_targets = std::move(middle_targets);
        middle_targets = std::move(current_targets);
        current_targets.clear();
        middle_flushes = frame_flushes;
    }

    struct Stats {
        u64 frames{};
        u64 draws{};
        u64 condition{};
        u64 pipeline{};
        u64 missing_frames{};
        u64 missing_condition{};
        u64 missing_pipeline{};
        u64 target_spikes{};
    };
    struct TargetStats {
        u32 drawn{};
        u32 skipped{};
        Info info{};
    };

    std::mutex mutex;
    u32 frame_present{~0U};
    u32 middle_present{};
    std::unordered_map<u64, Entry> older;
    std::unordered_map<u64, Entry> middle;
    std::unordered_map<u64, Entry> current;
    Stats stats{};
    u64 frames_logged{};
    std::unordered_map<u64, TargetStats> older_targets;
    std::unordered_map<u64, TargetStats> middle_targets;
    std::unordered_map<u64, TargetStats> current_targets;
    std::unordered_map<u64, TargetStats> target_totals;
    u64 target_logged{};
    u64 extra_logged{};
    u64 last_flush_waits{};
    u64 middle_flushes{};
};

DrawFlickerTrace draw_trace;

u64 DrawTraceKey(const Maxwell3D& maxwell3d, const Maxwell3D::DrawManager::State& state,
                 bool draw_indexed, u32 instance_count, DrawFlickerTrace::Info& info) {
    const auto& regs = maxwell3d.regs;
    info.index_address = draw_indexed ? state.index_buffer.StartAddress() : 0;
    info.stream0 = regs.vertex_streams[0].Address();
    info.first = draw_indexed ? state.index_buffer.first : state.vertex_buffer.first;
    info.count = draw_indexed ? state.index_buffer.count : state.vertex_buffer.count;
    info.instances = instance_count;
    info.vertex_program = regs.pipelines[1].offset;
    info.fragment_program = regs.pipelines[5].offset;
    info.predicate_slot = Common::N2denTrace::last_predicate_slot.load(std::memory_order_relaxed);
    info.predicate_count = Common::N2denTrace::last_predicate_count.load(std::memory_order_relaxed);
    info.predicate_stamp = Common::N2denTrace::last_predicate_stamp.load(std::memory_order_relaxed);
    info.zeta = regs.zeta_enable != 0 ? regs.zeta.Address() : 0;
    info.zeta_width = regs.zeta_size.width;
    info.zeta_height = regs.zeta_size.height;
    info.zeta_format = static_cast<u32>(regs.zeta.format);
    info.color_count = regs.rt_control.count;
    info.color0 = info.color_count != 0 ? regs.rt[0].Address() : 0;
    info.color0_width = regs.rt[0].width;
    info.color0_height = regs.rt[0].height;
    info.color0_format = static_cast<u32>(regs.rt[0].format);
    info.target = (info.zeta * 0x9E3779B97F4A7C15ULL) ^ info.color0 ^ (u64{info.color_count} << 56);
    u64 hash = 0xcbf29ce484222325ULL;
    const auto mix = [&hash](u64 value) {
        hash ^= value;
        hash *= 0x100000001b3ULL;
    };
    mix(info.index_address);
    mix(info.stream0);
    mix(info.first);
    mix(info.count);
    mix(state.base_index);
    mix(state.base_instance);
    mix(instance_count);
    mix(info.vertex_program);
    mix(info.fragment_program);
    mix(draw_indexed ? 1 : 0);
    return hash;
}

} // Anonymous namespace
namespace {
/// EDEN_TEXTURE_TABLE_STATS: draws skipped by a CPU-evaluated condition, and draws prepared under
/// host conditional rendering (which the host GPU may still discard).
void NoteDrawCondition(const Engines::Maxwell3D& maxwell3d) {
    if (!VideoCommon::CompactTextureStats::Enabled()) {
        return;
    }
    auto& stats = VideoCommon::CompactTextureStats::Get();
    if (!maxwell3d.ShouldExecute()) {
        ++stats.skipped_draws;
    } else if (maxwell3d.HostConditionalRendering()) {
        ++stats.host_conditional_draws;
    }
}
} // Anonymous namespace


void Maxwell3D::DrawManager::ProcessMethodCall(Maxwell3D& maxwell3d, u32 method, u32 argument) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(clear_surface):
        return Clear(maxwell3d, 1);
    case MAXWELL3D_REG_INDEX(draw.begin):
        return DrawBegin(maxwell3d);
    case MAXWELL3D_REG_INDEX(draw.end):
        return DrawEnd(maxwell3d);
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
        break;
    case MAXWELL3D_REG_INDEX(index_buffer.count):
        draw_state.draw_indexed = true;
        break;
    case MAXWELL3D_REG_INDEX(index_buffer32_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer16_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer8_subsequent):
        draw_state.instance_count++;
        [[fallthrough]];
    case MAXWELL3D_REG_INDEX(index_buffer32_first):
    case MAXWELL3D_REG_INDEX(index_buffer16_first):
    case MAXWELL3D_REG_INDEX(index_buffer8_first):
        return DrawIndexSmall(maxwell3d, argument);
    case MAXWELL3D_REG_INDEX(draw_inline_index):
        SetInlineIndexBuffer(maxwell3d, argument);
        break;
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
        SetInlineIndexBuffer(maxwell3d, maxwell3d.regs.inline_index_2x16.even);
        SetInlineIndexBuffer(maxwell3d, maxwell3d.regs.inline_index_2x16.odd);
        break;
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        SetInlineIndexBuffer(maxwell3d, maxwell3d.regs.inline_index_4x8.index0);
        SetInlineIndexBuffer(maxwell3d, maxwell3d.regs.inline_index_4x8.index1);
        SetInlineIndexBuffer(maxwell3d, maxwell3d.regs.inline_index_4x8.index2);
        SetInlineIndexBuffer(maxwell3d, maxwell3d.regs.inline_index_4x8.index3);
        break;
    case MAXWELL3D_REG_INDEX(vertex_array_instance_first):
        DrawArrayInstanced(maxwell3d, maxwell3d.regs.vertex_array_instance_first.topology.Value(), maxwell3d.regs.vertex_array_instance_first.start.Value(), maxwell3d.regs.vertex_array_instance_first.count.Value(), false);
        break;
    case MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent): {
        DrawArrayInstanced(maxwell3d, maxwell3d.regs.vertex_array_instance_subsequent.topology.Value(), maxwell3d.regs.vertex_array_instance_subsequent.start.Value(), maxwell3d.regs.vertex_array_instance_subsequent.count.Value(), true);
        break;
    }
    case MAXWELL3D_REG_INDEX(draw_texture.src_y0): {
        DrawTexture(maxwell3d);
        break;
    }
    default:
        break;
    }
}

void Maxwell3D::DrawManager::Clear(Maxwell3D& maxwell3d, u32 layer_count) {
    if (maxwell3d.ShouldExecute()) {
        maxwell3d.rasterizer->Clear(layer_count);
    }
}

void Maxwell3D::DrawManager::DrawDeferred(Maxwell3D& maxwell3d) {
    if (draw_state.draw_mode != DrawMode::Instance || draw_state.instance_count == 0) {
        return;
    }
    DrawEnd(maxwell3d, draw_state.instance_count + 1, true);
    draw_state.instance_count = 0;
}

bool Maxwell3D::DrawManager::IsInstanceLoopMethod(u32 method) noexcept {
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.begin):
    case MAXWELL3D_REG_INDEX(draw.end):
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(index_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer32_first):
    case MAXWELL3D_REG_INDEX(index_buffer16_first):
    case MAXWELL3D_REG_INDEX(index_buffer8_first):
    case MAXWELL3D_REG_INDEX(index_buffer32_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer16_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer8_subsequent):
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.count):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.count):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_first):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent):
        return true;
    default:
        return false;
    }
}

void Maxwell3D::DrawManager::DrawArray(Maxwell3D& maxwell3d, Maxwell3D::Regs::PrimitiveTopology topology, u32 vertex_first, u32 vertex_count, u32 base_instance, u32 num_instances) {
    draw_state.topology = topology;
    draw_state.vertex_buffer.first = vertex_first;
    draw_state.vertex_buffer.count = vertex_count;
    draw_state.base_instance = base_instance;
    ProcessDraw(maxwell3d, false, num_instances);
}

void Maxwell3D::DrawManager::DrawArrayInstanced(Maxwell3D& maxwell3d, Maxwell3D::Regs::PrimitiveTopology topology, u32 vertex_first, u32 vertex_count, bool subsequent) {
    Common::EmuCounters::draws_instance_methods.Add();
    draw_state.topology = topology;
    draw_state.vertex_buffer.first = vertex_first;
    draw_state.vertex_buffer.count = vertex_count;
    if (!subsequent) {
        draw_state.instance_count = 1;
    }
    draw_state.base_instance = draw_state.instance_count - 1;
    draw_state.draw_mode = DrawMode::Instance;
    draw_state.instance_count++;
    ProcessDraw(maxwell3d, false, 1);
}

void Maxwell3D::DrawManager::DrawIndex(Maxwell3D& maxwell3d, Maxwell3D::Regs::PrimitiveTopology topology, u32 index_first, u32 index_count, u32 base_index, u32 base_instance, u32 num_instances) {
    draw_state.topology = topology;
    draw_state.index_buffer = maxwell3d.regs.index_buffer;
    draw_state.index_buffer.first = index_first;
    draw_state.index_buffer.count = index_count;
    draw_state.base_index = base_index;
    draw_state.base_instance = base_instance;
    ProcessDraw(maxwell3d, true, num_instances);
}

void Maxwell3D::DrawManager::DrawArrayIndirect(Maxwell3D& maxwell3d, Maxwell3D::Regs::PrimitiveTopology topology) {
    draw_state.topology = topology;
    ProcessDrawIndirect(maxwell3d);
}

void Maxwell3D::DrawManager::DrawIndexedIndirect(Maxwell3D& maxwell3d, Maxwell3D::Regs::PrimitiveTopology topology, u32 index_first, u32 index_count) {
    draw_state.topology = topology;
    draw_state.index_buffer = maxwell3d.regs.index_buffer;
    draw_state.index_buffer.first = index_first;
    draw_state.index_buffer.count = index_count;
    ProcessDrawIndirect(maxwell3d);
}

void Maxwell3D::DrawManager::SetInlineIndexBuffer(Maxwell3D& maxwell3d, u32 index) {
    draw_state.inline_index_draw_indexes.push_back(u8(index & 0x000000ff));
    draw_state.inline_index_draw_indexes.push_back(u8((index & 0x0000ff00) >> 8));
    draw_state.inline_index_draw_indexes.push_back(u8((index & 0x00ff0000) >> 16));
    draw_state.inline_index_draw_indexes.push_back(u8((index & 0xff000000) >> 24));
    draw_state.draw_mode = DrawMode::InlineIndex;
}

void Maxwell3D::DrawManager::SetInlineIndexBuffer(Maxwell3D& maxwell3d, u32 method,
                                                  const u32* base_start, u32 amount) {
    auto& index_buffer = draw_state.inline_index_draw_indexes;
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw_inline_index): {
        const auto* const bytes = reinterpret_cast<const u8*>(base_start);
        index_buffer.insert(index_buffer.end(), bytes, bytes + size_t(amount) * sizeof(u32));
        break;
    }
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even): {
        const size_t offset = index_buffer.size();
        index_buffer.resize(offset + size_t(amount) * 2 * sizeof(u32));
        u8* dst = index_buffer.data() + offset;
        for (u32 i = 0; i < amount; ++i) {
            const u32 word = base_start[i];
            const u32 indexes[2]{word & 0xFFFF, word >> 16};
            std::memcpy(dst, indexes, sizeof(indexes));
            dst += sizeof(indexes);
        }
        break;
    }
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0): {
        const size_t offset = index_buffer.size();
        index_buffer.resize(offset + size_t(amount) * 4 * sizeof(u32));
        u8* dst = index_buffer.data() + offset;
        for (u32 i = 0; i < amount; ++i) {
            const u32 word = base_start[i];
            const u32 indexes[4]{word & 0xFF, (word >> 8) & 0xFF, (word >> 16) & 0xFF,
                                 word >> 24};
            std::memcpy(dst, indexes, sizeof(indexes));
            dst += sizeof(indexes);
        }
        break;
    }
    }
    draw_state.draw_mode = DrawMode::InlineIndex;
}

void Maxwell3D::DrawManager::DrawBegin(Maxwell3D& maxwell3d) {
    auto reset_instance_count = maxwell3d.regs.draw.instance_id == Maxwell3D::Regs::Draw::InstanceId::First;
    auto increment_instance_count = maxwell3d.regs.draw.instance_id == Maxwell3D::Regs::Draw::InstanceId::Subsequent;
    if (reset_instance_count) {
        DrawDeferred(maxwell3d);
        draw_state.instance_count = 0;
        draw_state.draw_mode = DrawMode::General;
    } else if (increment_instance_count) {
        draw_state.instance_count++;
        draw_state.draw_mode = DrawMode::Instance;
    }
    draw_state.topology = maxwell3d.regs.draw.topology;
}

void Maxwell3D::DrawManager::DrawEnd(Maxwell3D& maxwell3d, u32 instance_count, bool force_draw) {
    switch (draw_state.draw_mode) {
    case DrawMode::Instance:
        if (!force_draw) {
            break;
        }
        [[fallthrough]];
    case DrawMode::General:
        draw_state.base_instance = maxwell3d.regs.global_base_instance_index;
        draw_state.base_index = maxwell3d.regs.global_base_vertex_index;
        if (draw_state.draw_indexed) {
            draw_state.index_buffer = maxwell3d.regs.index_buffer;
            ProcessDraw(maxwell3d, true, instance_count);
        } else {
            draw_state.vertex_buffer = maxwell3d.regs.vertex_buffer;
            ProcessDraw(maxwell3d, false, instance_count);
        }
        draw_state.draw_indexed = false;
        break;
    case DrawMode::InlineIndex:
        Common::EmuCounters::draws_inline_index.Add();
        draw_state.base_instance = maxwell3d.regs.global_base_instance_index;
        draw_state.base_index = maxwell3d.regs.global_base_vertex_index;
        draw_state.index_buffer = maxwell3d.regs.index_buffer;
        draw_state.index_buffer.count = u32(draw_state.inline_index_draw_indexes.size() / 4);
        draw_state.index_buffer.format = Maxwell3D::Regs::IndexFormat::UnsignedInt;
        maxwell3d.dirty.flags[VideoCommon::Dirty::IndexBuffer] = true;
        ProcessDraw(maxwell3d, true, instance_count);
        draw_state.inline_index_draw_indexes.clear();
        break;
    }
}

void Maxwell3D::DrawManager::DrawIndexSmall(Maxwell3D& maxwell3d, u32 argument) {
    Common::EmuCounters::draws_instance_methods.Add();
    Maxwell3D::Regs::IndexBufferSmall index_small_params{argument};
    draw_state.base_instance = maxwell3d.regs.global_base_instance_index;
    draw_state.base_index = maxwell3d.regs.global_base_vertex_index;
    draw_state.index_buffer = maxwell3d.regs.index_buffer;
    draw_state.index_buffer.first = index_small_params.first;
    draw_state.index_buffer.count = index_small_params.count;
    draw_state.topology = index_small_params.topology;
    maxwell3d.dirty.flags[VideoCommon::Dirty::IndexBuffer] = true;
    ProcessDraw(maxwell3d, true, 1);
}

void Maxwell3D::DrawManager::DrawTexture(Maxwell3D& maxwell3d) {
    draw_texture_state.dst_x0 = f32(maxwell3d.regs.draw_texture.dst_x0) / 4096.f;
    draw_texture_state.dst_y0 = f32(maxwell3d.regs.draw_texture.dst_y0) / 4096.f;
    const auto dst_width = f32(maxwell3d.regs.draw_texture.dst_width) / 4096.f;
    const auto dst_height = f32(maxwell3d.regs.draw_texture.dst_height) / 4096.f;
    const bool lower_left{maxwell3d.regs.window_origin.mode != Maxwell3D::Regs::WindowOrigin::Mode::UpperLeft};
    if (lower_left) {
        draw_texture_state.dst_y0 = f32(maxwell3d.regs.surface_clip.height) - draw_texture_state.dst_y0;
    }
    draw_texture_state.dst_x1 = draw_texture_state.dst_x0 + dst_width;
    draw_texture_state.dst_y1 = draw_texture_state.dst_y0 + dst_height;
    draw_texture_state.src_x0 = f32(maxwell3d.regs.draw_texture.src_x0) / 4096.f;
    draw_texture_state.src_y0 = f32(maxwell3d.regs.draw_texture.src_y0) / 4096.f;
    draw_texture_state.src_x1 = (f32(maxwell3d.regs.draw_texture.dx_du) / 4294967296.f) * dst_width + draw_texture_state.src_x0;
    draw_texture_state.src_y1 = (f32(maxwell3d.regs.draw_texture.dy_dv) / 4294967296.f) * dst_height + draw_texture_state.src_y0;
    draw_texture_state.src_sampler = maxwell3d.regs.draw_texture.src_sampler;
    draw_texture_state.src_texture = maxwell3d.regs.draw_texture.src_texture;
    maxwell3d.rasterizer->DrawTexture();
}

void Maxwell3D::DrawManager::UpdateTopology(Maxwell3D& maxwell3d) {
    switch (maxwell3d.regs.primitive_topology_control) {
    case Maxwell3D::Regs::PrimitiveTopologyControl::UseInBeginMethods:
        break;
    case Maxwell3D::Regs::PrimitiveTopologyControl::UseSeparateState:
        switch (maxwell3d.regs.topology_override) {
        case Maxwell3D::Regs::PrimitiveTopologyOverride::None:
            break;
        case Maxwell3D::Regs::PrimitiveTopologyOverride::Points:
            draw_state.topology = Maxwell3D::Regs::PrimitiveTopology::Points;
            break;
        case Maxwell3D::Regs::PrimitiveTopologyOverride::Lines:
            draw_state.topology = Maxwell3D::Regs::PrimitiveTopology::Lines;
            break;
        case Maxwell3D::Regs::PrimitiveTopologyOverride::LineStrip:
            draw_state.topology = Maxwell3D::Regs::PrimitiveTopology::LineStrip;
            break;
        default:
            draw_state.topology = Maxwell3D::Regs::PrimitiveTopology(maxwell3d.regs.topology_override);
            break;
        }
        break;
    }
}

void Maxwell3D::DrawManager::ProcessDraw(Maxwell3D& maxwell3d, bool draw_indexed, u32 instance_count) {
    draw_state.draw_element_count =
        draw_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count;
    LOG_TRACE(HW_GPU, "called, topology={}, count={}", draw_state.topology, draw_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count);
    UpdateTopology(maxwell3d);
    NoteDrawCondition(maxwell3d);
    // [DEBUG] EDEN_N2DEN_IGNORE_RENDER_ENABLE=1: draw even when the render-enable condition says
    // no, and log the skipped draws' condition state, to find draws a stale condition hides.
    static const bool ignore_condition = std::getenv("EDEN_N2DEN_IGNORE_RENDER_ENABLE") != nullptr;
    if (!maxwell3d.ShouldExecute()) {
        static std::atomic<u64> condition_skips{};
        const u64 skips = ++condition_skips;
        if (skips <= 16 || skips % 4096 == 0) {
            LOG_INFO(HW_GPU, "Draw under a false render-enable condition: override={}, mode={}, "
                     "address={:#x} ({} so far){}",
                     static_cast<u32>(maxwell3d.regs.render_enable_override),
                     static_cast<u32>(maxwell3d.regs.render_enable.mode),
                     maxwell3d.regs.render_enable.Address(), skips,
                     ignore_condition ? ", drawn anyway" : "");
        }
    }
    const bool trace_draws = DrawFlickerTrace::Enabled();
    DrawFlickerTrace::Info trace_info{};
    const u64 trace_key =
        trace_draws ? DrawTraceKey(maxwell3d, draw_state, draw_indexed, instance_count, trace_info) : 0;
    if (trace_draws && !maxwell3d.ShouldExecute() && !ignore_condition) {
        draw_trace.OnDraw(trace_key, DrawFlickerTrace::ConditionSkip, trace_info);
    }
    // A draw an occlusion count culled (see Common::N2denOcclusion::MarkCondition) is still drawn
    // when the depth test can hide it: Ocarina of Time's counts read 0 for a visible object now
    // and then while the camera moves, and the object (a tree trunk, a grass bank) was missing for
    // that frame with the bright background showing through it.
    bool occlusion_override = false;
    if (!maxwell3d.ShouldExecute() && !ignore_condition &&
        Common::N2denOcclusion::condition_from_count) {
        const auto& depth_regs = maxwell3d.regs;
        const auto func = depth_regs.depth_test_func;
        occlusion_override = depth_regs.render_enable_override ==
                                 Maxwell3D::Regs::RenderEnable::Override::UseRenderEnable &&
                             depth_regs.zeta_enable != 0 && depth_regs.depth_test_enable != 0 &&
                             func != Maxwell3D::Regs::ComparisonOp::Never_D3D &&
                             func != Maxwell3D::Regs::ComparisonOp::Always_D3D &&
                             func != Maxwell3D::Regs::ComparisonOp::Never_GL &&
                             func != Maxwell3D::Regs::ComparisonOp::Always_GL;
        static std::atomic<u64> overrides{};
        static std::atomic<u64> kept{};
        const u64 n = occlusion_override ? ++overrides : overrides.load();
        const u64 k = occlusion_override ? kept.load() : ++kept;
        if ((n + k) == 1 || (n + k) % 100000 == 0) {
            LOG_INFO(HW_GPU, "Occlusion-culled draws: {} drawn depth-tested, {} kept culled (no depth test)",
                     n, k);
        }
    }
    if (maxwell3d.ShouldExecute() || ignore_condition || occlusion_override) {
        auto& native = maxwell3d.GetNativeMacroEngine();
        if (native.native_turing) {
            const auto count = draw_indexed ? draw_state.index_buffer.count : draw_state.vertex_buffer.count;
            // Zero-count draws are still prepared (their pipelines get built): Ocarina of Time's
            // mesh draws arrive with count 0. EDEN_N2DEN_KEEP_ZERO_DRAWS=0 returns early as before.
            static const bool keep_zero = [] {
                const char* const value = std::getenv("EDEN_N2DEN_KEEP_ZERO_DRAWS");
                return value == nullptr || std::strcmp(value, "0") != 0;
            }();
            if ((!count && !keep_zero) || !instance_count) return;
            if (native.native_execution_failed) return;
            if (Settings::values.renderer_backend.GetValue() != Settings::RendererBackend::Vulkan) {
                native.native_execution_failed = true;
                LOG_ERROR(HW_GPU, "Native graphics shaders require the experimental Vulkan path");
                return;
            }
        }
        const u64 pipeline_skips =
            trace_draws ? Common::N2denTrace::pipeline_skips.load(std::memory_order_relaxed) : 0;
        maxwell3d.rasterizer->Draw(draw_indexed, instance_count);
        if (trace_draws) {
            const bool skipped =
                Common::N2denTrace::pipeline_skips.load(std::memory_order_relaxed) != pipeline_skips;
            draw_trace.OnDraw(trace_key,
                              skipped ? DrawFlickerTrace::PipelineSkip : DrawFlickerTrace::Drawn,
                              trace_info);
        }
    }
}

void Maxwell3D::DrawManager::ProcessDrawIndirect(Maxwell3D& maxwell3d) {
    draw_state.draw_element_count = 0;
    LOG_TRACE(HW_GPU, "called, topology={}, is_indexed={}, includes_count={}, buffer_size={}, max_draw_count={}", draw_state.topology, indirect_state.is_indexed, indirect_state.include_count, indirect_state.buffer_size, indirect_state.max_draw_counts);
    UpdateTopology(maxwell3d);
    NoteDrawCondition(maxwell3d);
    if (maxwell3d.ShouldExecute()) {
        auto& native = maxwell3d.GetNativeMacroEngine();
        if (native.native_turing) {
            native.native_execution_failed = true;
            LOG_ERROR(HW_GPU, "Native graphics shader translation required: indirect draw");
            return;
        }
        maxwell3d.rasterizer->DrawIndirect();
    }
}
} // namespace Tegra::Engines
