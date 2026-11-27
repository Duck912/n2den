// SPDX-FileCopyrightText: Copyright 2026 n2den Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Diagnostics of the draw path (see nvn_debug.h): draw log, per-target draw table, render target
// dumps. Only reached when Debug::Tracing() (the environment's switches or a diag request).

#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <filesystem>
#include <iterator>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/nvn/state_layouts.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw.h"

namespace Vulkan::Nvn {

void DrawContext::DebugSync() {
    const u64 frame = impl.presents_recorded + 1;
    if (frame == dbg.frame) {
        return;
    }
    DebugFlushFrame();
    dbg.frame = frame;
    const Debug::Config& config = Debug::Get();
    const Debug::RequestState* request = Debug::RequestFor(frame);
    dbg.selected = (config.tracing && Debug::FrameSelected(frame)) || request != nullptr;
    dbg.draw_log = config.draw_log;
    dbg.dump_pass = config.dump_pass;
    dbg.dump_draw = config.dump_draw;
    dbg.dump_dir = config.dump_dir;
    dbg.report_pass.clear();
    dbg.report_draw.clear();
    if (request != nullptr) {
        // A diag request bound to this frame replaces the environment's selection of it.
        dbg.draw_log = 0;
        dbg.dump_pass = request->pass;
        dbg.dump_draw = request->draw;
        dbg.dump_dir.clear();
        if (request->pass != 0) {
            dbg.dump_dir = request->directory.string();
            dbg.report_pass = request->directory / fmt::format("nvn-pass-{}.txt", request->pass);
        }
        if (request->draw != 0) {
            dbg.report_draw = request->directory / fmt::format("nvn-draw-{}.txt", request->draw);
        }
        if (request->all_passes) {
            // nvn-frame: every pass's targets (dump_pass 0) and the frame's draws in the log.
            dbg.dump_dir = request->directory.string();
            dbg.draw_log = 1u << 20; // every draw of the frame
        }
    }
    dbg.logged = 0;
    dbg.draw_index = 0;
    dbg.pass_index = 0;
    dbg.pass_draws = 0;
    dbg.dumped = 0;
    dbg.targets.clear();
    dbg.break_reasons.clear();
    if (dbg.selected && !dbg.dump_dir.empty()) {
        std::error_code error;
        std::filesystem::create_directories(dbg.dump_dir, error);
    }
}

void DrawContext::DebugFlushFrame() {
    if (!dbg.selected || dbg.targets.empty()) {
        return;
    }
    u64 draws = 0;
    u64 passes = 0;
    for (const auto& [key, stat] : dbg.targets) {
        draws += stat.draws;
        passes += stat.passes;
    }
    LOG_INFO(Render_Vulkan, "[nvn] frame {}: {} draws in {} passes, {} distinct target sets", dbg.frame,
             draws, passes, dbg.targets.size());
    for (const auto& [key, stat] : dbg.targets) {
        LOG_INFO(Render_Vulkan, "[nvn]   targets {}: draws={} (indexed {}) vertices={} passes={}", key,
                 stat.draws, stat.indexed, stat.vertices, stat.passes);
    }
    for (const auto& [reason, count] : dbg.break_reasons) {
        LOG_INFO(Render_Vulkan, "[nvn]   pass break by draw preparation: {} x{}", reason, count);
    }
}

std::string DrawContext::DebugTargetKey() const {
    std::string key;
    for (u32 slot = 0; slot < color_target_count; ++slot) {
        const TargetBinding& target = color_targets[slot];
        if (!target.texture || !target.texture->storage) {
            continue;
        }
        const Storage& s = *target.texture->storage;
        const u32 level = target.ref.texture.level;
        key += fmt::format("c{}:{}x{}x{}:f{}:s{} ", slot, (std::max)(s.extent.width >> level, 1u),
                           (std::max)(s.extent.height >> level, 1u), s.type == VK_IMAGE_TYPE_3D ? s.extent.depth : s.layers,
                           static_cast<int>(s.host.format), static_cast<int>(s.samples));
    }
    if (depth_target.texture && depth_target.texture->storage) {
        const Storage& s = *depth_target.texture->storage;
        key += fmt::format("z:{}x{}:f{}", s.extent.width, s.extent.height, static_cast<int>(s.host.format));
        if (const auto pool = impl.pools.find(s.pool_serial); pool != impl.pools.end()) {
            key += fmt::format("@{:#x}+{:#x}:L{}:layer{}:lv{}x{}", pool->second->address, s.pool_offset,
                               depth_target.ref.texture.level, depth_target.ref.texture.layer, s.levels,
                               s.layers);
        }
    }
    return key;
}

void DrawContext::DebugOnPassBegin() {
    DebugSync();
    if (!dbg.selected) {
        return;
    }
    ++dbg.pass_index;
    dbg.pass_draws = 0;
    ++dbg.targets[DebugTargetKey()].passes;
}

void DrawContext::DebugOnPassEnd() {
    const Debug::Config& config = Debug::Get();
    if (!dbg.selected || dbg.dump_dir.empty() ||
        dbg.dumped >= config.dump_max ||
        (dbg.dump_pass != 0 && dbg.pass_index != dbg.dump_pass)) {
        return;
    }
    const auto dump = [&](const TargetBinding& target, const char* tag, VkImageAspectFlags aspect) {
        if (!target.texture || !target.texture->storage || !target.texture->storage->image ||
            dbg.dumped >= config.dump_max) {
            return;
        }
        Storage& s = *target.texture->storage;
        if (config.dump_width != 0 &&
            (std::max)(s.extent.width >> target.ref.texture.level, 1u) != config.dump_width) {
            return;
        }
        // EDEN_NX2_NVN_DUMP_MIN_WIDTH=N: only targets at least N texels wide (the full-size passes).
        static const u32 min_width = [] {
            const char* value = std::getenv("EDEN_NX2_NVN_DUMP_MIN_WIDTH");
            return value != nullptr ? static_cast<u32>(std::strtoul(value, nullptr, 0)) : 0u;
        }();
        if ((std::max)(s.extent.width >> target.ref.texture.level, 1u) < min_width) {
            return;
        }
        Debug::PendingDump pending;
        pending.path = fmt::format("{}/f{:05}_p{:03}_{}_t{:x}_d{}.tga", dbg.dump_dir, dbg.frame,
                                   dbg.pass_index, tag, target.texture->serial, dbg.pass_draws);
        pending.report = dbg.report_pass;
        pending.format = s.host.format;
        pending.thumb_width = config.dump_thumb;
        host->HostDumpImage(s, target.ref.texture.level, target.ref.texture.layer, aspect,
                            std::move(pending));
        ++dbg.dumped;
    };
    for (u32 slot = 0; slot < color_target_count; ++slot) {
        dump(color_targets[slot], fmt::format("c{}", slot).c_str(), VK_IMAGE_ASPECT_COLOR_BIT);
    }
    dump(depth_target, "z", VK_IMAGE_ASPECT_DEPTH_BIT);
}

void DrawContext::DebugReadbackIndirect(u64 address, const BufferRef& args, u32 bytes,
                                        const BufferRef* count) {
    DebugSync();
    if (!dbg.selected || dbg.logged >= dbg.draw_log || !args.buffer) {
        return;
    }
    InFlight& flight = host->HostFlight();
    const auto copy = [&](const BufferRef& source, u32 size, std::string tag) {
        InFlight::Watch entry;
        entry.size = size;
        entry.tag = std::move(tag);
        entry.buffer =
            impl.CreateHostBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryUsage::Download);
        const VkBufferCopy region{.srcOffset = source.offset, .dstOffset = 0, .size = size};
        host->HostCmd().CopyBuffer(source.buffer, *entry.buffer, region);
        flight.watches.push_back(std::move(entry));
    };
    const u32 draw = dbg.draw_index + 1;
    copy(args, bytes, fmt::format("draw-args f{} #{} ind={:#x}", dbg.frame, draw, address));
    if (count != nullptr && count->buffer) {
        copy(*count, sizeof(u32), fmt::format("draw-count f{} #{} ind={:#x}", dbg.frame, draw, address));
    }
    host->HostNeedHostReadBarrier();
}

bool DrawContext::DebugDrawAllowed() const {
    const Debug::Config& config = Debug::Get();
    if (!dbg.selected) {
        return true;
    }
    if (!config.only_vs.empty()) {
        const auto it = resources.program_hashes.find(stages[0].program_serial);
        const u64 hash = it == resources.program_hashes.end() ? 0 : it->second[0];
        if (std::find(config.only_vs.begin(), config.only_vs.end(), hash) == config.only_vs.end()) {
            return false;
        }
    }
    return config.only_draws.empty() ||
           std::find(config.only_draws.begin(), config.only_draws.end(), dbg.draw_index) !=
               config.only_draws.end();
}

void DrawContext::DebugOnDraw(u32 mode) {
    const bool indexed = dbg_call.indexed;
    DebugSync();
    if (!dbg.selected) {
        return;
    }
    ++dbg.draw_index;
    ++dbg.pass_draws;
    DebugTargetStat& stat = dbg.targets[DebugTargetKey()];
    ++stat.draws;
    stat.indexed += indexed ? 1 : 0;
    stat.vertices += static_cast<u64>(dbg_call.count) * (std::max)(dbg_call.instances, 1u);
    const bool log_this_draw = dbg.logged < dbg.draw_log;
    // nvn-draw / EDEN_NX2_NVN_DUMP_DRAW: draw n is logged with the CPU view of its buffers; a
    // request also writes those lines to its report file.
    // EDEN_NX2_NVN_DUMP_DRAW_VS=<hex>: every draw of the selected frames whose vertex program has
    // this hash (draw numbers move between boots; program hashes do not).
    static const u64 dump_vs = [] {
        const char* value = std::getenv("EDEN_NX2_NVN_DUMP_DRAW_VS");
        return value != nullptr ? std::strtoull(value, nullptr, 16) : u64{0};
    }();
    bool vs_match = false;
    if (dump_vs != 0) {
        const auto it = resources.program_hashes.find(stages[0].program_serial);
        vs_match = it != resources.program_hashes.end() && it->second[0] == dump_vs;
    }
    const bool dump_this_draw = (dbg.dump_draw != 0 && dbg.dump_draw == dbg.draw_index) || vs_match;
    if (!log_this_draw && !dump_this_draw) {
        return;
    }
    if (log_this_draw) {
        ++dbg.logged;
    }
    const auto emit = [&](const std::string& text) {
        LOG_INFO(Render_Vulkan, "{}", text);
        if (dump_this_draw && !dbg.report_draw.empty()) {
            Debug::AppendLine(dbg.report_draw, text);
        }
    };
    std::string line = fmt::format("[nvn] draw f{} #{} pass{} {} mode={} count={} inst={} first={}",
                                   dbg.frame, dbg.draw_index, dbg.pass_index,
                                   indexed ? "indexed" : "direct", mode, dbg_call.count,
                                   dbg_call.instances, dbg_call.first);
    if (host->HostConditional()) {
        // GPU predicate (occlusion): the GPU may discard this draw.
        line += " rc=cond";
    }
    if (dbg_call.indexed) {
        line += fmt::format(" ia={:#x}", dbg_call.index_address);
    }
    if (dbg_call.indirect != 0) {
        line += fmt::format(" cnt={:#x} max={} stride={} pool={}/{:#x}", dbg_call.count_address,
                            dbg_call.max_draws, dbg_call.stride, dbg_call.args_backing, dbg_call.args_flags);
        line += fmt::format(" IND={:#x}[{}:{},{},{},{},{}]", dbg_call.indirect, dbg_call.args_read ? "ok" : "unreadable",
                            dbg_call.args[0], dbg_call.args[1], dbg_call.args[2], dbg_call.args[3], dbg_call.args[4]);
    }
    for (u32 stage = 0; stage < GRAPHICS_STAGE_COUNT; ++stage) {
        const u64 serial = stages[stage].program_serial;
        if (serial == 0) {
            continue;
        }
        const auto it = resources.program_hashes.find(serial);
        line += fmt::format(" s{}={:x}", stage, it == resources.program_hashes.end() ? 0 : it->second[stage]);
    }
    namespace S = VC;
    line += fmt::format(" poly={:#x}(cull{} ccw{}) color={:#x} mask={:#x} blend0={:#x} ds={:#x}(zt{} zw{} zf{})",
                        fixed.polygon, S::PolygonState::CullFace(fixed.polygon),
                        S::PolygonState::FrontFace(fixed.polygon), fixed.color, fixed.channel_mask,
                        fixed.blend[0], fixed.depth_stencil, S::DepthStencilState::DepthTestEnable(fixed.depth_stencil),
                        S::DepthStencilState::DepthWriteEnable(fixed.depth_stencil),
                        S::DepthStencilState::DepthFunc(fixed.depth_stencil));
    const VC::ViewportRect& vp = viewports[0];
    const VC::ScissorRect& sc = scissors[0];
    line += fmt::format(" vp[{}]=({:.1f},{:.1f},{:.1f},{:.1f}) z=({:.2f},{:.2f}) sc[{}]=({},{},{},{}) fb={}x{}",
                        (viewports_set & 1), vp.x, vp.y, vp.width, vp.height, depth_ranges[0].near_plane,
                        depth_ranges[0].far_plane, (scissors_set & 1), sc.x, sc.y, sc.width, sc.height,
                        framebuffer_extent.width, framebuffer_extent.height);
    line += fmt::format(" off={} po=({},{},{}) dclamp={} dmode={}", S::PolygonState::OffsetEnables(fixed.polygon),
                        polygon_offset[0], polygon_offset[1], polygon_offset[2], fixed.depth_clamp,
                        impl.device_depth_mode.load(std::memory_order_relaxed));
    line += " tg=" + DebugTargetKey();
    // The layer (3D slice) each colour target is attached at: texture layer, then the view's
    // layer range when it sets one.
    for (u32 slot = 0; slot < color_target_count; ++slot) {
        const TargetBinding& target = color_targets[slot];
        if (target.texture) {
            const Storage* s = target.texture->storage.get();
            line += fmt::format(" L{}={}{} c{}={:x}@{:x}+{:x}/{:x}:th{}:st{:x}", slot,
                                target.ref.texture.layer,
                                target.ref.view.set_layers
                                    ? fmt::format("(view {}+{})", target.ref.view.min_layer,
                                                  target.ref.view.nlayers)
                                    : std::string{},
                                slot, target.texture->serial, s ? s->pool_serial : 0,
                                s ? s->pool_offset : 0, s ? s->layout.size : 0,
                                s ? s->layout.tile_h : 0, reinterpret_cast<uintptr_t>(s));
        }
    }
    emit(line);
    line = fmt::format("[nvn] draw+ f{} #{} va=[", dbg.frame, dbg.draw_index);
    for (u32 i = 0; i < MAX_VERTEX_ATTRIBUTES; ++i) {
        const u32 word = fixed.attributes[i];
        if (S::VertexAttribState::Format(word) != 0) {
            line += fmt::format("{}:f{:x}/s{}+{}{} ", i, S::VertexAttribState::Format(word),
                                S::VertexAttribState::StreamIndex(word), S::VertexAttribState::Offset(word),
                                pipeline && ((pipeline->attributes_read >> i) & 1) ? "" : "(unread)");
        }
    }
    line += "] vb=[";
    for (u32 s = 0; s < MAX_VERTEX_STREAMS; ++s) {
        if (vertex_buffers[s].address != 0) {
            line += fmt::format("{}:{:#x}+{:#x}/stride{}{} ", s, vertex_buffers[s].address,
                                vertex_buffers[s].size, fixed.stream_stride[s],
                                fixed.stream_divisor[s] != 0 ? "/inst" : "");
        }
    }
    emit(line);
    // Direct indexed draws: the CPU view of the index range and of the positions it reaches
    // (attribute 0 read as three floats), to tell bad guest data from a bad host copy.
    if (dbg_call.indexed && dbg_call.indirect == 0 && dbg_call.count != 0) {
        const u32 isize = dbg_call.index_type <= 2 ? 1u << dbg_call.index_type : 0u;
        const u32 word0 = fixed.attributes[0];
        const u32 stream = S::VertexAttribState::StreamIndex(word0);
        const u32 stride = stream < MAX_VERTEX_STREAMS ? fixed.stream_stride[stream] : 0;
        const u64 vb_address = stream < MAX_VERTEX_STREAMS ? vertex_buffers[stream].address : 0;
        const u64 vb_size = stream < MAX_VERTEX_STREAMS ? vertex_buffers[stream].size : 0;
        const u8* ib = isize != 0 ? impl.GuestPointer(dbg_call.index_address, u64{dbg_call.count} * isize)
                                  : nullptr;
        const u8* vb = vb_address != 0 ? impl.GuestPointer(vb_address, vb_size) : nullptr;
        line = fmt::format("[nvn] draw idx f{} #{} type={} base={}", dbg.frame, dbg.draw_index,
                           dbg_call.index_type, dbg_call.first);
        if (ib == nullptr) {
            line += " ib=unreadable";
        } else {
            u64 lo = ~u64{0}, hi = 0, hash = 0xcbf29ce484222325ULL;
            float mn[3]{1e30f, 1e30f, 1e30f}, mx[3]{-1e30f, -1e30f, -1e30f};
            u32 out_of_range = 0, nonfinite = 0, huge = 0;
            const u64 capacity = stride != 0 ? vb_size / stride : 0;
            const u32 offset = S::VertexAttribState::Offset(word0);
            for (u32 i = 0; i < dbg_call.count; ++i) {
                u32 index = 0;
                std::memcpy(&index, ib + u64{i} * isize, isize);
                const u32 restart = isize == 1 ? 0xffu : (isize == 2 ? 0xffffu : 0xffffffffu);
                if (index == restart) {
                    continue;
                }
                const s64 vertex = static_cast<s64>(index) + dbg_call.first;
                lo = (std::min)(lo, static_cast<u64>(vertex));
                hi = (std::max)(hi, static_cast<u64>(vertex));
                hash = (hash ^ index) * 0x100000001b3ULL;
                if (vertex < 0 || static_cast<u64>(vertex) >= capacity) {
                    ++out_of_range;
                    continue;
                }
                if (vb == nullptr) {
                    continue;
                }
                float f[3];
                std::memcpy(f, vb + static_cast<u64>(vertex) * stride + offset, sizeof(f));
                for (int c = 0; c < 3; ++c) {
                    if (!std::isfinite(f[c])) {
                        ++nonfinite;
                        continue;
                    }
                    huge += std::fabs(f[c]) > 1e5f ? 1 : 0;
                    mn[c] = (std::min)(mn[c], f[c]);
                    mx[c] = (std::max)(mx[c], f[c]);
                }
            }
            line += fmt::format(" idx=[{},{}] cap={} oor={} ihash={:016x}", lo, hi, capacity,
                                out_of_range, hash);
            if (vb == nullptr) {
                line += " vb=unreadable";
            } else {
                line += fmt::format(" pos=[{:.3g},{:.3g},{:.3g}]..[{:.3g},{:.3g},{:.3g}] nan={} huge={}",
                                    mn[0], mn[1], mn[2], mx[0], mx[1], mx[2], nonfinite, huge);
            }
        }
        emit(line);
    }
    line = fmt::format("[nvn] draw+ f{} #{} ubo=[", dbg.frame, dbg.draw_index);
    for (u32 stage = 0; stage < BIND_STAGES; ++stage) {
        for (u32 slot = 0; slot < stages[stage].uniform.size(); ++slot) {
            const BufferBinding& b = stages[stage].uniform[slot];
            if (b.address != 0) {
                line += fmt::format("{}.{}:{:#x}/{:#x} ", stage, slot, b.address, b.size);
            }
        }
    }
    line += "] ssbo=[";
    for (u32 stage = 0; stage < BIND_STAGES; ++stage) {
        for (u32 slot = 0; slot < stages[stage].storage.size(); ++slot) {
            const BufferBinding& b = stages[stage].storage[slot];
            if (b.address != 0) {
                line += fmt::format("{}.{}:{:#x}/{:#x}", stage, slot, b.address, b.size);
                // Where the buffer lives: backing 1 imported host memory, 2 mirror, 3 device local.
                BufferRef ref;
                if (ResolveBuffer(b.address, 4, false, ref, ShadowPolicy::Ignore) && ref.pool) {
                    line += fmt::format("@{}/{:#x}", static_cast<s32>(ref.pool->backing),
                                        ref.pool->flags);
                }
                line += ' ';
            }
        }
    }
    line += "]";
    if (stages[2].program_serial != 0 && stages[2].storage[0].address != 0) {
        BufferRef gs0;
        if (ResolveBuffer(stages[2].storage[0].address, 4, false, gs0, ShadowPolicy::Ignore) && gs0.pool) {
            line += fmt::format(" gs0pool={}/{:#x}", static_cast<s32>(gs0.pool->backing), gs0.pool->flags);
        }
    }
    line += " tex=[";
    for (const DrawPlan::ImageBinding& b : draw_plan.bindings) {
        const Storage& s = *b.texture->storage;
        line += fmt::format("{:x}:{}x{}:f{}{}", b.texture->serial, s.extent.width, s.extent.height,
                            static_cast<int>(s.host.format), b.scratch ? "(copy)" : "");
        // Where the bytes live and whether the image holds them: pool address+offset/flags,
        // origin (0 never uploaded, 1 CPU, 2 GPU) and contents_valid. Finds textures that are
        // never filled from guest memory (a movie frame a decoder writes there).
        if (const auto pool = impl.pools.find(s.pool_serial); pool != impl.pools.end()) {
            line += fmt::format("@{:#x}+{:#x}/{:#x}{}", pool->second->address, s.pool_offset,
                                pool->second->flags, pool->second->IsVirtual() ? "v" : "");
        } else {
            line += "@nopool";
        }
        line += fmt::format(":o{}v{} ", static_cast<int>(s.origin), s.contents_valid ? 1 : 0);
    }
    line += fmt::format("] inline_updates={} fallbacks=[{}]", inline_updates.size(), dbg_notes);
    // EDEN_NX2_NVN_PEEK_VS=<vertex program hash>: words 72..74, 126 and 155 (byte offsets 288,
    // 504, 620) of every vertex-stage storage binding of the draw (Kirby Air Riders' ambient).
    static const u64 peek_vs = [] {
        const char* env = std::getenv("EDEN_NX2_NVN_PEEK_VS");
        return env != nullptr ? std::strtoull(env, nullptr, 16) : 0ull;
    }();
    // EDEN_NX2_NVN_PEEK_VS=1: every draw whose vertex stage binds one buffer as storage slots 6
    // and 7 (the per-object ambient records).
    const bool records_draw = peek_vs == 1 && stages[0].storage.size() > 7 &&
                              stages[0].storage[7].address != 0 &&
                              stages[0].storage[7].address == stages[0].storage[6].address;
    if (peek_vs != 0 && (ProfStageHash(0) == peek_vs || records_draw)) {
        if (VC::GuestMemory* const memory = impl.guest.load(std::memory_order_acquire)) {
            for (u32 slot = 0; slot < stages[0].storage.size(); ++slot) {
                const BufferBinding& b = stages[0].storage[slot];
                if (b.address == 0 || b.size < 640) {
                    continue;
                }
                std::array<u32, 3> rgb{};
                u32 w504 = 0;
                u32 w620 = 0;
                memory->Read(b.address + 288, rgb.data(), sizeof(rgb));
                memory->Read(b.address + 504, &w504, 4);
                memory->Read(b.address + 620, &w620, 4);
                float f[3];
                std::memcpy(f, rgb.data(), sizeof(f));
                float g504;
                float g620;
                std::memcpy(&g504, &w504, 4);
                std::memcpy(&g620, &w620, 4);
                line += fmt::format(" vsssbo{}@{:#x}[288]=({:.4g},{:.4g},{:.4g}) [504]={:.4g} [620]={:.4g}",
                                    slot, b.address, f[0], f[1], f[2], g504, g620);
            }
            // Slot 7 holds the per-object ambient cubes (8 x vec4 per object) the vertex stage
            // blends: its first 256 bytes.
            if (stages[0].storage.size() > 7 && stages[0].storage[7].address != 0) {
                const BufferBinding& b = stages[0].storage[7];
                if (Debug::Get().watch_address == 3) {
                    // EDEN_NX2_NVN_WATCH=peek (+ EDEN_NX2_NVN_PEEK=0:0:0:<words>): the GPU view
                    // of these records right after this draw.
                    peek_address = b.address;
                }
                std::array<float, 1024> words{};
                const u64 bytes = (std::min)(u64{b.size}, u64{sizeof(words)});
                memory->Read(b.address, words.data(), bytes);
                line += fmt::format(" amb7@{:#x}/{:#x}=[", b.address, b.size);
                for (u64 rec = 0; rec < bytes / 128; ++rec) {
                    float max_rgb = 0.0f;
                    float sum_w = 0.0f;
                    for (u64 v = 0; v < 8; ++v) {
                        const float* f = &words[rec * 32 + v * 4];
                        max_rgb = (std::max)({max_rgb, f[0], f[1], f[2]});
                        sum_w += f[3];
                    }
                    line += fmt::format("{}{:.3g}/{:.3g}", rec == 0 ? "" : " ", max_rgb, sum_w);
                }
                line += "]";
                // The object's record index (vertex stage: slot 0 word at +460 past the object).
                if (stages[0].storage[0].address != 0) {
                    std::array<u32, 8> idx{};
                    memory->Read(stages[0].storage[0].address + 456, idx.data(), sizeof(idx));
                    line += fmt::format(" s0[456..]={} {} {} {}", idx[0], idx[1], idx[2], idx[3]);
                }
            }
        }
    }
    // Small textures the GPU wrote last: the guest bytes behind them (does the CPU fill them?).
    if (VC::GuestMemory* const memory = impl.guest.load(std::memory_order_acquire)) {
        for (const DrawPlan::ImageBinding& b : draw_plan.bindings) {
            const Storage& s = *b.texture->storage;
            if (s.extent.width * s.extent.height > 4096 || s.extent.width < 8) {
                continue;
            }
            const auto pool = impl.pools.find(s.pool_serial);
            if (pool == impl.pools.end()) {
                continue;
            }
            std::array<u32, 32> words{};
            if (!memory->Read(pool->second->address + s.pool_offset, words.data(), sizeof(words))) {
                continue;
            }
            std::string bytes;
            u32 nonzero = 0;
            for (const u32 w : words) {
                bytes += fmt::format(" {:x}", w);
                nonzero += w != 0 ? 1 : 0;
            }
            line += fmt::format(" guest[{:x} {}x{} o{} nz{}]:{}", b.texture->serial, s.extent.width,
                                s.extent.height, static_cast<int>(s.origin), nonzero, bytes);
        }
    }
    emit(line);
    if (dump_this_draw) {
        // EDEN_NX2_NVN_DUMP_DRAW: the CPU view of the draw's vertex and uniform buffers.
        const auto floats = [&](const char* what, u32 id, u64 address, u64 size) {
            std::string text = fmt::format("[nvn] draw data f{} #{} {} {} @{:#x} size {:#x}:", dbg.frame,
                                           dbg.draw_index, what, id, address, size);
            const u64 count = (std::min<u64>)(size, 0x400) / 4;
            if (const u8* base = impl.GuestPointer(address, count * 4)) {
                for (u64 i = 0; i < count; ++i) {
                    float f;
                    u32 u;
                    std::memcpy(&f, base + i * 4, 4);
                    std::memcpy(&u, base + i * 4, 4);
                    text += (i % 4 == 0 ? "\n  " : " ");
                    text += fmt::format("{:.4g}[{:x}]", f, u);
                }
            } else {
                text += " unreadable";
            }
            emit(text);
        };
        for (u32 s = 0; s < MAX_VERTEX_STREAMS; ++s) {
            if (vertex_buffers[s].address != 0) {
                floats("vb", s, vertex_buffers[s].address, vertex_buffers[s].size);
            }
        }
        for (u32 slot = 0; slot < stages[0].uniform.size(); ++slot) {
            if (stages[0].uniform[slot].address != 0) {
                floats("ubo0", slot, stages[0].uniform[slot].address, stages[0].uniform[slot].size);
            }
        }
        for (u32 slot = 0; slot < stages[1].uniform.size(); ++slot) {
            if (stages[1].uniform[slot].address != 0) {
                floats("ubo1", slot, stages[1].uniform[slot].address, stages[1].uniform[slot].size);
            }
        }
    }
}

void DrawContext::DebugEndFrame() {
    DebugFlushFrame();
    dbg.targets.clear();
    dbg.break_reasons.clear();
    dbg.selected = false;
}

void DrawContext::DiagNoteTextures() {
    if (!DiagTextures::BeginDraw()) {
        return;
    }
    diag_hold = diag_textures.size();
    for (const DrawPlan::ImageBinding& b : draw_plan.bindings) {
        Texture* game = b.game_texture ? b.game_texture.get() : b.texture;
        if (!game || !game->storage || !b.texture || !b.texture->storage) {
            continue;
        }
        const Storage& s = *game->storage;
        DiagTextures::Use use;
        use.stage = static_cast<s32>(b.stage);
        use.slot = static_cast<s32>(b.slot);
        use.nvn_format = s.desc.format;
        use.width = s.desc.width;
        use.height = s.desc.height;
        use.depth = s.desc.depth;
        use.levels = s.levels;
        use.layers = s.layers;
        if (const auto pool = impl.pools.find(s.pool_serial); pool != impl.pools.end()) {
            use.address = pool->second->address + s.pool_offset;
        }
        if (b.sampler_serial != 0) {
            if (const auto it = impl.samplers.find(b.sampler_serial); it != impl.samplers.end()) {
                use.sampler = &it->second;
            }
        }
        if (b.is_image) {
            use.note = "storage image";
        }
        if (b.scratch) {
            use.note = "render target also sampled by the draw: saved from its feedback copy";
        }
        const DiagTextures::Job job = DiagTextures::Note(use);
        if (job.capture) {
            diag_textures.push_back({b.texture->storage, job});
        }
    }
    DiagTextures::EndDraw();
}

void DrawContext::DiagFlushTextures() {
    const size_t ready = (std::min)(diag_hold, diag_textures.size());
    if (pass_open || ready == 0) {
        return;
    }
    std::vector<DiagPending> pending(std::make_move_iterator(diag_textures.begin()),
                                     std::make_move_iterator(diag_textures.begin() + ready));
    diag_textures.erase(diag_textures.begin(), diag_textures.begin() + ready);
    if (diag_hold != ~size_t{0}) {
        diag_hold -= ready;
    }
    for (const DiagPending& item : pending) {
        host->HostDumpTexture(*item.storage, item.job);
    }
}

} // namespace Vulkan::Nvn
