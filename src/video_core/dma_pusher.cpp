// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <string_view>

#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "video_core/dma_pusher.h"
#include "video_core/control/channel_state.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/back_thread_stats.h"
#include "video_core/gpu_split.h"
#include "video_core/gpu_written_pages.h"
#include "video_core/guest_memory.h"
#include "video_core/memory_manager.h"
#include "video_core/rasterizer_interface.h"

namespace Tegra {

constexpr u32 MacroRegistersStart = 0xE00;
constexpr u32 ComputeInline = 0x6D;
constexpr u32 MaxwellInline = offsetof(Engines::Maxwell3D::Regs, inline_data) / sizeof(u32);

DmaPusher::DmaPusher(Core::System& system_, MemoryManager& memory_manager_, Control::ChannelState& channel_state_)
    : system{system_}
    , memory_manager{memory_manager_}
    , channel_state{channel_state_}
    , signal_sync{false}
    , synced{false}
{}

DmaPusher::~DmaPusher() = default;

void DmaPusher::DispatchCalls() {
    dma_state.is_last_call = true;
    while (system.IsPoweredOn()) {
        if (!Step()) {
            break;
        }
    }
    if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
        split.Defer([gpu = &system.GPU()] {
            const VideoCommon::BackStats::ScopedSharedTime list_end{
                VideoCommon::BackStats::shared.list_end};
            gpu->FlushCommandsBatched();
            gpu->OnCommandListEnd();
        });
        split.Publish();
        return;
    }
    system.GPU().FlushCommandsBatched();
    system.GPU().OnCommandListEnd();
}

bool DmaPusher::HasFailedNativeWork() const {
    // With the GPU thread split the back thread clears the flags after each method it runs
    // (gpu_split.cpp), as this does on one thread.
    if (VideoCommon::GpuSplit::Get().Active()) {
        return false;
    }
    const auto& context = channel_state.native_macro_context;
    if (!channel_state.native_abi || !context ||
        !(context->engine.native_execution_failed || context->compute_engine.native_execution_failed)) {
        return false;
    }
    // Drop the failed work (a skipped draw or dispatch) and keep the game running instead of
    // hanging on its fences.
    static std::atomic<u32> reported{};
    if (reported.fetch_add(1) < 32) {
        LOG_WARNING(HW_GPU, "Native GPU work failed; skipping it and continuing the channel");
    }
    context->engine.native_execution_failed = false;
    context->compute_engine.native_execution_failed = false;
    return false;
}

bool DmaPusher::Step() {
    if (HasFailedNativeWork()) return false;
    if (!ib_enable || dma_pushbuffer.empty()) {
        return false;
    }

    if (!puller.ResumeNativeAcquire(*this)) return false;

    CommandList& command_list = dma_pushbuffer.front();

    const size_t prefetch_size = command_list.prefetch_command_list.size();
    const size_t command_list_size = command_list.command_lists.size();

    if (prefetch_size == 0 && command_list_size == 0) {
        dma_pushbuffer.pop();
        dma_pushbuffer_subindex = 0;
        return true;
    }

    if (prefetch_size > 0) {
        const std::span<const CommandHeader> commands{command_list.prefetch_command_list};
        command_resume_offset += ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        if (puller.HasPendingNativeAcquire() || HasFailedNativeWork()) return false;
        command_resume_offset = 0;
        dma_pushbuffer.pop();
        return true;
    }

    auto& current_command = command_list.command_lists[dma_pushbuffer_subindex];
    const CommandListHeader& header = current_command;
    dma_state.dma_get = header.addr;

    if (signal_sync && !synced) {
        std::unique_lock lk(sync_mutex);
        sync_cv.wait(lk, [this]() { return synced; });
        signal_sync = false;
        synced = false;
    }

    if (header.size > 0) {
        if (subchannels[dma_state.subchannel] && dma_state.method_count) {
            const auto engine = subchannel_type[dma_state.subchannel];
            const bool kepler_payload = engine == Engines::EngineTypes::KeplerCompute && dma_state.method == ComputeInline && dma_state.non_incrementing;
            const bool macro_payload = engine == Engines::EngineTypes::Maxwell3D && dma_state.method >= MacroRegistersStart;
            if (kepler_payload || macro_payload) {
                const size_t words = std::min<size_t>(dma_state.method_count, header.size);
                auto* const engine_state = subchannels[dma_state.subchannel];
                if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
                    // The caches behind IsMemoryDirty belong to the split's back thread.
                    split.Defer([engine_state, memory = &memory_manager,
                                 address = GPUVAddr{dma_state.dma_get}, bytes = words * sizeof(u32)] {
                        engine_state->current_dirty = memory->IsMemoryDirty(address, bytes);
                    });
                } else {
                    engine_state->current_dirty = memory_manager.IsMemoryDirty(dma_state.dma_get, words * sizeof(u32));
                }
            }
        }
        // GPU-written entries use completed page copies or freshly downloaded guest memory.
        const size_t entry_bytes = size_t{header.size} * sizeof(u32);
        const bool gpu_written = IsGpuWrittenCommands(dma_state.dma_get, entry_bytes);
        const bool copied = gpu_written && SyncGpuWrittenCommands(dma_state.dma_get, entry_bytes);
        // A safe read flushes the caches, which belong to the split's back thread while it runs
        // (the split only starts with unsafe reads; the settings can change mid-game).
        const bool use_safe = !VideoCommon::GpuSplit::Get().Active() &&
                              (Settings::IsDMALevelDefault() ? Settings::IsGPULevelHigh() : Settings::IsDMALevelSafe());
        // The words as submitted, unless this entry resumes after an acquire (the words after it
        // may be new) or the GPU has newer words that the safe read would flush.
        const bool submitted = command_resume_offset == 0 && !gpu_written &&
                               dma_pushbuffer_subindex < command_list.submitted_offsets.size() &&
                               !(use_safe && memory_manager.IsMemoryDirty(dma_state.dma_get,
                                                                          header.size * sizeof(u32)));
        if (copied) {
            const std::span<const CommandHeader> commands{command_headers.data(), header.size};
            command_resume_offset +=
                ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        } else if (submitted) {
            const std::span<const CommandHeader> commands{
                command_list.submitted_words.data() +
                    command_list.submitted_offsets[dma_pushbuffer_subindex],
                header.size};
            command_resume_offset += ProcessCommands(commands, 0);
        } else if (use_safe) {
            Tegra::Memory::GpuGuestMemory<Tegra::CommandHeader, Tegra::Memory::GuestMemoryFlags::SafeRead>headers(memory_manager, dma_state.dma_get, header.size, &command_headers);
            const std::span<const CommandHeader> commands{headers.data(), headers.size()};
            command_resume_offset += ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        } else {
            Tegra::Memory::GpuGuestMemory<Tegra::CommandHeader, Tegra::Memory::GuestMemoryFlags::UnsafeRead>headers(memory_manager, dma_state.dma_get, header.size, &command_headers);
            const std::span<const CommandHeader> commands{headers.data(), headers.size()};
            command_resume_offset += ProcessCommands(commands.subspan(command_resume_offset), command_resume_offset);
        }
    }

    if (puller.HasPendingNativeAcquire() || HasFailedNativeWork()) return false;
    command_resume_offset = 0;
    if (++dma_pushbuffer_subindex >= command_list_size) {
        dma_pushbuffer.pop();
        dma_pushbuffer_subindex = 0;
    } else {
        signal_sync = command_list.command_lists[dma_pushbuffer_subindex].sync && Settings::values.sync_memory_operations.GetValue();
    }

    if (signal_sync) {
        VideoCommon::GpuSplit::Get().Drain(VideoCommon::BackStats::DrainSignalSync);
        rasterizer->SignalFence([this]() {
            std::scoped_lock lk(sync_mutex);
            synced = true;
            sync_cv.notify_all();
        });
    }

    return true;
}

size_t DmaPusher::ProcessCommands(std::span<const CommandHeader> commands, size_t word_offset) {
    for (size_t index = 0; index < commands.size();) {
        // Data word of methods command
        if (dma_state.method_count && dma_state.non_incrementing) {
            auto const& command_header = commands[index]; //must ref (MUltiMethod re)
            dma_state.dma_word_offset = u64((word_offset + index) * sizeof(u32));
            u32 max_write = u32(std::min<std::size_t>(index + dma_state.method_count, commands.size()) - index);
            if (channel_state.native_abi && dma_state.method < non_puller_methods) max_write = 1;
            CallMultiMethod(&command_header.argument, max_write);
            dma_state.method_count -= max_write;
            dma_state.is_last_call = true;
            index += max_write;
        } else if (dma_state.method_count) {
            auto const command_header = commands[index]; //can copy
            dma_state.dma_word_offset = u64((word_offset + index) * sizeof(u32));
            dma_state.is_last_call = dma_state.method_count <= 1;
            CallMethod(command_header.argument);
            dma_state.method += !dma_state.non_incrementing ? 1 : 0;
            dma_state.non_incrementing |= dma_increment_once;
            dma_state.method_count--;
            index++;
        } else {
            auto const command_header = commands[index]; //can copy
            // No command active - this is the first word of a new one
            switch (command_header.mode) {
            case SubmissionMode::Increasing:
                SetState(command_header);
                dma_state.non_incrementing = false;
                dma_increment_once = false;
                break;
            case SubmissionMode::NonIncreasing:
                SetState(command_header);
                dma_state.non_incrementing = true;
                dma_increment_once = false;
                break;
            case SubmissionMode::Inline:
                dma_state.method = command_header.method;
                dma_state.subchannel = command_header.subchannel;
                dma_state.dma_word_offset = u64(-s64(dma_state.dma_get)); // negate to set address as 0
                CallMethod(command_header.arg_count);
                dma_state.non_incrementing = true;
                dma_increment_once = false;
                break;
            case SubmissionMode::IncreaseOnce:
                SetState(command_header);
                dma_state.non_incrementing = false;
                dma_increment_once = true;
                break;
            default:
                break;
            }
            index++;
        }
        if (puller.HasPendingNativeAcquire() || HasFailedNativeWork()) return index;
    }
    return commands.size();
}

void DmaPusher::SetState(const CommandHeader& command_header) {
    dma_state.method = command_header.method;
    dma_state.subchannel = command_header.subchannel;
    dma_state.method_count = command_header.method_count;
}

bool DmaPusher::WantsSegment() const {
    // current_dma_segment is read by macro parameter collection (Maxwell3D) and inline uploads
    // (KeplerCompute data_upload).
    const auto engine = subchannel_type[dma_state.subchannel];
    return engine == Engines::EngineTypes::KeplerCompute ||
           (engine == Engines::EngineTypes::Maxwell3D && dma_state.method >= MacroRegistersStart);
}

void DmaPusher::CallMethod(u32 argument) {
    if (dma_state.method < non_puller_methods) {
        puller.CallPullerMethod(*this, Engines::Puller::MethodCall{
            dma_state.method,
            argument,
            dma_state.subchannel,
            dma_state.method_count,
        });
    } else {
        auto subchannel = subchannels[dma_state.subchannel];
        auto& writer_split = VideoCommon::GpuSplit::Get();
        const auto engine_type = subchannel_type[dma_state.subchannel];
        SCOPE_EXIT {
            if (writer_split.Active() && (engine_type == Engines::EngineTypes::KeplerCompute ||
                                          engine_type == Engines::EngineTypes::MaxwellDMA ||
                                          engine_type == Engines::EngineTypes::KeplerMemory ||
                                          engine_type == Engines::EngineTypes::Fermi2D ||
                                          (engine_type == Engines::EngineTypes::Maxwell3D &&
                                           dma_state.method == MaxwellInline))) {
                writer_split.NoteWriter();
            }
        };
        if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
            const GPUVAddr segment = dma_state.dma_get + dma_state.dma_word_offset;
            if (split.FrontMacros()) {
                const auto type = subchannel_type[dma_state.subchannel];
                if (type == Engines::EngineTypes::Maxwell3D) {
                    auto* const maxwell3d = static_cast<Engines::Maxwell3D*>(subchannel);
                    if (maxwell3d->FrontRunsMacros()) {
                        maxwell3d->FrontCallMethod(system, split, dma_state.method, argument,
                                                   dma_state.is_last_call);
                        return;
                    }
                } else if (type == Engines::EngineTypes::KeplerCompute) {
                    auto* const compute = static_cast<Engines::KeplerCompute*>(subchannel);
                    if (compute->FrontRunsMacros()) {
                        compute->FrontCallMethod(system, split, dma_state.method, argument,
                                                 dma_state.is_last_call, &segment);
                        return;
                    }
                }
            }
            split.Method(subchannel, dma_state.method, argument, dma_state.is_last_call,
                         WantsSegment() ? &segment : nullptr);
            return;
        }
        if (!subchannel->execution_mask[dma_state.method]) {
            subchannel->method_sink.emplace_back(dma_state.method, argument);
        } else {
            subchannel->ConsumeSink(system);
            subchannel->current_dma_segment = dma_state.dma_get + dma_state.dma_word_offset;
            subchannel->CallMethod(system, dma_state.method, argument, dma_state.is_last_call);
        }
    }
}

void DmaPusher::CallMultiMethod(const u32* base_start, u32 num_methods) {
    if (dma_state.method < non_puller_methods) {
        puller.CallMultiMethod(*this, dma_state.method, dma_state.subchannel, base_start, num_methods, dma_state.method_count);
    } else {
        auto subchannel = subchannels[dma_state.subchannel];
        auto& writer_split = VideoCommon::GpuSplit::Get();
        const auto engine_type = subchannel_type[dma_state.subchannel];
        SCOPE_EXIT {
            if (writer_split.Active() && (engine_type == Engines::EngineTypes::KeplerCompute ||
                                          engine_type == Engines::EngineTypes::MaxwellDMA ||
                                          engine_type == Engines::EngineTypes::KeplerMemory ||
                                          engine_type == Engines::EngineTypes::Fermi2D ||
                                          (engine_type == Engines::EngineTypes::Maxwell3D &&
                                           dma_state.method == MaxwellInline))) {
                writer_split.NoteWriter();
            }
        };
        if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
            const GPUVAddr segment = dma_state.dma_get + dma_state.dma_word_offset;
            if (split.FrontMacros()) {
                const auto type = subchannel_type[dma_state.subchannel];
                if (type == Engines::EngineTypes::Maxwell3D) {
                    auto* const maxwell3d = static_cast<Engines::Maxwell3D*>(subchannel);
                    if (maxwell3d->FrontRunsMacros()) {
                        maxwell3d->FrontCallMultiMethod(system, split, dma_state.method, base_start,
                                                        num_methods, dma_state.method_count);
                        return;
                    }
                } else if (type == Engines::EngineTypes::KeplerCompute) {
                    auto* const compute = static_cast<Engines::KeplerCompute*>(subchannel);
                    if (compute->FrontRunsMacros()) {
                        compute->FrontCallMultiMethod(system, split, dma_state.method, base_start,
                                                      num_methods, dma_state.method_count, &segment);
                        return;
                    }
                }
            }
            split.MultiMethod(subchannel, dma_state.method, base_start, num_methods,
                              dma_state.method_count, WantsSegment() ? &segment : nullptr);
            return;
        }
        subchannel->ConsumeSink(system);
        subchannel->current_dma_segment = dma_state.dma_get + dma_state.dma_word_offset;
        subchannel->CallMultiMethod(system, dma_state.method, base_start, num_methods, dma_state.method_count);
    }
}

void DmaPusher::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
}

// Reading GPU-written command-list entries back from guest memory follows kraibse's change
// (6612817cb4, "restores the trees").
bool DmaPusher::IsGpuWrittenCommands(GPUVAddr address, size_t size) const {
    // EDEN_NX2_GPU_WRITTEN_LISTS=0: read every entry as submitted (the old behaviour).
    static const bool disabled = [] {
        const char* const value = std::getenv("EDEN_NX2_GPU_WRITTEN_LISTS");
        return value != nullptr && value[0] == '0';
    }();
    if (size == 0 || disabled) {
        return false;
    }
    for (GPUVAddr page = address & ~GPUVAddr{0xfff}; page < address + size; page += 0x1000) {
        if (const auto device = memory_manager.GpuToCpuAddress(page);
            device && VideoCommon::IsGpuWrittenPage(*device)) {
            return true;
        }
    }
    return false;
}

bool DmaPusher::ReadGpuWrittenCommands(GPUVAddr address, size_t size) {
    if (rasterizer == nullptr) {
        return false;
    }
    const GPUVAddr end = address + size;
    for (GPUVAddr page = address & ~GPUVAddr{0xfff}; page < end; page += 0x1000) {
        if (const auto device = memory_manager.GpuToCpuAddress(page);
            device && VideoCommon::IsGpuWrittenPage(*device)) {
            VideoCommon::MarkCommandListPage(*device);
        }
    }
    // Takoroka's copy approach (8f1435f8db): the back records and submits page copies
    // after compute and DMA writers. The front waits only through those writers, then for
    // the copy's GPU tick, without flushing the scheduler or locking the buffer cache.
    VideoCommon::GpuSplit::Get().WaitForWriters();
    command_headers.resize_destructive(size / sizeof(CommandHeader));
    auto* const out = reinterpret_cast<u8*>(command_headers.data());
    using GpuWrittenList = VideoCore::RasterizerInterface::GpuWrittenList;
    for (GPUVAddr chunk = address; chunk < end;) {
        const GPUVAddr chunk_end = (std::min)(end, (chunk & ~GPUVAddr{0xfff}) + 0x1000);
        const std::span<u8> part{out + (chunk - address), static_cast<size_t>(chunk_end - chunk)};
        const auto device = memory_manager.GpuToCpuAddress(chunk);
        if (!device) {
            return false;
        }
        const auto where = VideoCommon::IsGpuWrittenPage(*device)
                               ? rasterizer->ReadGpuWrittenCommandList(*device, part)
                               : GpuWrittenList::Synced;
        if (where == GpuWrittenList::Unknown) {
            return false;
        }
        if (where == GpuWrittenList::Synced) {
            memory_manager.ReadBlockUnsafe(chunk, part.data(), part.size());
        }
        chunk = chunk_end;
    }
    return true;
}

bool DmaPusher::SyncGpuWrittenCommands(GPUVAddr address, size_t size) {
    static const bool drain_only = [] {
        const char* const value = std::getenv("EDEN_NX2_GPU_WRITTEN_LISTS");
        return value != nullptr && std::string_view{value} == "drain";
    }();
    if (!drain_only && ReadGpuWrittenCommands(address, size)) {
        return true;
    }
    // kraibse's drain approach (6612817cb4) handles missing or stale copies and pages
    // written by draws. Download whole pages so later lists can use guest memory until a writer
    // invalidates them. EDEN_NX2_GPU_WRITTEN_LISTS=drain retains the original range flush.
    const GPUVAddr first = drain_only ? address : address & ~GPUVAddr{0xfff};
    const GPUVAddr end = drain_only ? address + size : (address + size + 0xfff) & ~GPUVAddr{0xfff};
    const bool note_synced = !drain_only;
    const auto flush = [memory = &memory_manager, renderer = rasterizer, first, end, note_synced] {
        memory->FlushRegion(first, end - first, VideoCommon::CacheType::BufferCache);
        if (note_synced && renderer != nullptr) {
            for (GPUVAddr page = first; page < end; page += 0x1000) {
                if (const auto device = memory->GpuToCpuAddress(page);
                    device && VideoCommon::IsCommandListPage(*device)) {
                    renderer->NoteGpuWrittenCommandListSynced(*device);
                }
            }
        }
    };
    auto& split = VideoCommon::GpuSplit::Get();
    if (split.Active()) {
        split.Defer(flush);
        split.Drain(VideoCommon::BackStats::DrainOther);
    } else {
        flush();
    }
    return false;
}

} // namespace Tegra
