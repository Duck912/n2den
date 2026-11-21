// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>

#include "common/common_types.h"

namespace VideoCommon {

// Device pages the GPU has written through the buffer cache (shader stores, buffer copies), ever.
// Switch 2 games run command lists the GPU fills in: Ocarina of Time's foliage culling writes the
// group counts of the foliage dispatches into the inline QMD words of a later command list. Such
// an entry must be read after its GPU writes complete, rather than from the words
// captured when the CPU submitted the list. The bits never clear: the dispatch that rewrites a
// page may still be queued when the command list is read. One bit per 4 KiB of a 36-bit device
// address space (2 MiB).
inline constexpr u64 GpuWrittenPageBits = 12;
inline constexpr u64 GpuWrittenPageCount = u64{1} << (36 - GpuWrittenPageBits);
inline std::array<std::atomic<u64>, GpuWrittenPageCount / 64> g_gpu_written_pages{};

inline void MarkGpuWritten(u64 device_address, u64 size) {
    if (size == 0) {
        return;
    }
    const u64 first = device_address >> GpuWrittenPageBits;
    if (first >= GpuWrittenPageCount) {
        return;
    }
    const u64 last = (std::min)((device_address + size - 1) >> GpuWrittenPageBits,
                                GpuWrittenPageCount - 1);
    for (u64 word_index = first / 64; word_index <= last / 64; ++word_index) {
        const u64 lo = (std::max)(first, word_index * 64) % 64;
        const u64 hi = (std::min)(last, word_index * 64 + 63) % 64;
        const u64 mask = (hi == 63 ? ~u64{0} : (u64{1} << (hi + 1)) - 1) & ~((u64{1} << lo) - 1);
        auto& word = g_gpu_written_pages[word_index];
        if ((word.load(std::memory_order_relaxed) & mask) != mask) {
            word.fetch_or(mask, std::memory_order_relaxed);
        }
    }
}

[[nodiscard]] inline bool IsGpuWrittenPage(u64 device_address) {
    const u64 page = device_address >> GpuWrittenPageBits;
    if (page >= GpuWrittenPageCount) {
        return false;
    }
    return (g_gpu_written_pages[page / 64].load(std::memory_order_relaxed) >> (page % 64)) & 1;
}

// Takoroka's command-list copies (8f1435f8db) track pages the pusher has read.
// The buffer cache copies these pages after compute and DMA writers.
inline std::array<std::atomic<u64>, GpuWrittenPageCount / 64> g_command_list_pages{};
inline std::atomic<bool> g_has_command_list_pages{};

inline void MarkCommandListPage(u64 device_address) {
    const u64 page = device_address >> GpuWrittenPageBits;
    if (page >= GpuWrittenPageCount) {
        return;
    }
    g_command_list_pages[page / 64].fetch_or(u64{1} << (page % 64), std::memory_order_relaxed);
    g_has_command_list_pages.store(true, std::memory_order_relaxed);
}

[[nodiscard]] inline bool IsCommandListPage(u64 device_address) {
    const u64 page = device_address >> GpuWrittenPageBits;
    return page < GpuWrittenPageCount &&
           ((g_command_list_pages[page / 64].load(std::memory_order_relaxed) >> (page % 64)) & 1);
}

/// A new emulation session starts: no page has been written by its GPU yet.
inline void ClearGpuWrittenPages() {
    for (auto& word : g_command_list_pages) {
        word.store(0, std::memory_order_relaxed);
    }
    g_has_command_list_pages.store(false, std::memory_order_relaxed);
    for (auto& word : g_gpu_written_pages) {
        word.store(0, std::memory_order_relaxed);
    }
}

} // namespace VideoCommon
