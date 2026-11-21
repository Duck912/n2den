// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <utility>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "common/common_types.h"
#include "video_core/frame_snapshot.h"
#include "common/logging.h"
#include "core/device_memory_manager.h"
#include "video_core/back_thread_stats.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/memory_manager.h"

namespace VideoCommon {

namespace HandleProgramDetail {
/// Runs handle program `program` with read(address, size) returning the 4- or 8-byte guest word.
template <typename CbufAddress, typename Read>
u32 Run(const Shader::Info& info, u32 program, CbufAddress&& cbuf_address, Read&& read) {
    return Shader::EvaluateHandleProgram(
               info.handle_programs[program],
               [&](u32 index, u32 offset) -> u32 {
                   const GPUVAddr base{cbuf_address(index)};
                   return base != 0 ? static_cast<u32>(read(base + offset, sizeof(u32))) : 0U;
               },
               [&](u32 binding, u32 offset) -> u32 {
                   if (binding >= info.storage_buffers_descriptors.size()) {
                       return 0U;
                   }
                   const auto& descriptor{info.storage_buffers_descriptors[binding]};
                   const GPUVAddr cbuf{cbuf_address(descriptor.cbuf_index)};
                   if (cbuf == 0) {
                       return 0U;
                   }
                   GPUVAddr base{read(cbuf + descriptor.cbuf_offset, sizeof(u64))};
                   if (base != 0 && descriptor.indirect_offset != ~0U) {
                       base = read(base + descriptor.indirect_offset, sizeof(u64));
                   }
                   return base != 0 ? static_cast<u32>(read(base + offset, sizeof(u32))) : 0U;
               })
        .value_or(0U);
}
} // namespace HandleProgramDetail

/// The texture handle a shader reads from a storage buffer, computed from the current constant
/// buffer and storage buffer words (Shader::Info::handle_programs). cbuf_address(index) is the
/// bound constant buffer's address, 0 when none is bound.
template <typename CbufAddress>
u32 EvaluateTextureHandle(const Shader::Info& info, u32 program, Tegra::MemoryManager& gpu_memory,
                          CbufAddress&& cbuf_address) {
    return HandleProgramDetail::Run(info, program, cbuf_address,
                                    [&](GPUVAddr address, u32 size) -> u64 {
                                        return size == sizeof(u64)
                                                   ? FrameSnapshot::ReadGuest<u64>(gpu_memory, address)
                                                   : FrameSnapshot::ReadGuest<u32>(gpu_memory, address);
                                    });
}

/// The last result of one handle program and every input it read.
/// A handle program is a pure function of the constant buffer addresses it
/// asks for and the guest words it reads, so while all of them are unchanged its result is reused
/// instead of running the program again. Words are compared through the host pointers they were
/// read from, kept while the page table generation is the same (as the storage buffer binding
/// cache does). Owned by one pipeline and used by the thread configuring it.
struct HandleProgramMemo {
    struct Word {
        const u8* host;
        u64 value;
        u32 size;
        u64 device; ///< with frame snapshots: its device address, to check for a copy
    };
    static constexpr size_t MaxWords = 64;
    const Tegra::MemoryManager* owner{};
    u64 generation{};
    u32 result{};
    bool reusable{};
    boost::container::small_vector<std::pair<u32, GPUVAddr>, 4> cbufs; ///< Index, address used
    boost::container::small_vector<Word, 16> words;
};

enum class HandleMemoMode { Off, On, Verify };

/// EDEN_N2DEN_HANDLE_MEMO=0 runs every handle program on every draw, as before; =verify reuses
/// nothing either, but runs the program on every hit as well and counts the results that differ
/// (back thread statistics, and an error for the first ones).
[[nodiscard]] inline HandleMemoMode HandleMemo() {
    static const HandleMemoMode mode = [] {
        const char* const value = std::getenv("EDEN_N2DEN_HANDLE_MEMO");
        if (value && value[0] == '0') return HandleMemoMode::Off;
        if (value && std::string_view{value} == "verify") return HandleMemoMode::Verify;
        return HandleMemoMode::On;
    }();
    return mode;
}

/// EvaluateTextureHandle through `memo`: the last result when no input changed.
template <typename CbufAddress>
u32 EvaluateTextureHandle(const Shader::Info& info, u32 program, Tegra::MemoryManager& gpu_memory,
                          CbufAddress&& cbuf_address, HandleProgramMemo& memo) {
    const HandleMemoMode mode = HandleMemo();
    if (mode == HandleMemoMode::Off) {
        return EvaluateTextureHandle(info, program, gpu_memory, cbuf_address);
    }
    namespace BackStats = VideoCommon::BackStats;
    const u64 generation = gpu_memory.MappingGeneration();
    // [EXPERIMENT] With frame snapshots a word whose page has a copy is read from it, never reused.
    const bool snapshot = FrameSnapshot::Part(1) && LateWriteTrace::OnBackThread();
    if (memo.reusable && memo.owner == &gpu_memory && memo.generation == generation) {
        const bool same_cbufs = std::ranges::all_of(memo.cbufs, [&](const auto& entry) {
            return cbuf_address(entry.first) == entry.second;
        });
        if (same_cbufs && std::ranges::all_of(memo.words, [snapshot](const HandleProgramMemo::Word& word) {
                if (snapshot && word.device != 0 && FrameSnapshot::HasCopy(word.device)) {
                    return false;
                }
                u64 value = 0;
                std::memcpy(&value, word.host, word.size);
                return value == word.value;
            })) {
            if (BackStats::Enabled()) {
                BackStats::Add(BackStats::shared.handle_memo_hits, 1);
            }
            if (mode == HandleMemoMode::Verify) {
                const u32 fresh = EvaluateTextureHandle(info, program, gpu_memory, cbuf_address);
                if (fresh != memo.result) {
                    static std::atomic<u32> logged{};
                    if (BackStats::Enabled()) {
                        BackStats::Add(BackStats::shared.handle_memo_mismatches, 1);
                    }
                    if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                        LOG_ERROR(Render_Vulkan, "Handle memo mismatch: program {}, kept {:#x}, now {:#x}",
                                  program, memo.result, fresh);
                    }
                    memo.reusable = false;
                    return fresh;
                }
            }
            return memo.result;
        }
    }
    if (BackStats::Enabled()) {
        BackStats::Add(BackStats::shared.handle_memo_misses, 1);
    }
    memo.owner = &gpu_memory;
    memo.generation = generation;
    memo.reusable = true;
    memo.cbufs.clear();
    memo.words.clear();
    const auto recorded_cbuf = [&](u32 index) -> GPUVAddr {
        const GPUVAddr address = cbuf_address(index);
        if (std::ranges::find(memo.cbufs, index, &std::pair<u32, GPUVAddr>::first) ==
            memo.cbufs.end()) {
            memo.cbufs.emplace_back(index, address);
        }
        return address;
    };
    const auto recorded_read = [&](GPUVAddr address, u32 size) -> u64 {
        constexpr GPUVAddr page_mask = Core::DEVICE_PAGESIZE - 1;
        // The syncpoint shim is not memory, and a word across a page has no single host pointer.
        const u8* const host = (address & page_mask) <= Core::DEVICE_PAGESIZE - size &&
                                       !gpu_memory.OverlapsSyncpointShim(address, size)
                                   ? gpu_memory.GetPointer(address)
                                   : nullptr;
        if (!host) {
            memo.reusable = false;
            return size == sizeof(u64) ? gpu_memory.Read<u64>(address) : gpu_memory.Read<u32>(address);
        }
        u64 device = 0;
        if (snapshot) {
            if (const auto device_address = gpu_memory.GpuToCpuAddress(address)) {
                device = *device_address;
                FrameSnapshot::Watch(device, size);
                if (FrameSnapshot::HasCopy(device)) {
                    memo.reusable = false;
                    return size == sizeof(u64) ? FrameSnapshot::ReadGuest<u64>(gpu_memory, address)
                                               : FrameSnapshot::ReadGuest<u32>(gpu_memory, address);
                }
            }
        }
        u64 value = 0;
        std::memcpy(&value, host, size);
        if (memo.words.size() < HandleProgramMemo::MaxWords) {
            memo.words.push_back({host, value, size, device});
        } else {
            memo.reusable = false;
        }
        return value;
    };
    memo.result = HandleProgramDetail::Run(info, program, recorded_cbuf, recorded_read);
    return memo.result;
}

} // namespace VideoCommon
