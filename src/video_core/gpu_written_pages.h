// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <vector>

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

// Synced list runs (split front). Pages written by draws make every command-list entry on them
// drain the split (Nx2ListPage::draw_written): the front can't tell whether a draw it already
// sent writes the page again. Bananza reads ~1,100 indirect draw records a frame from pages its
// argument scatter draw wrote, one entry each: ~1,100 drains a frame. After a drain has flushed
// a page, the page stays current while the front sends nothing that could write it: register
// writes and the draws of the macros those entries feed. Any other draw, compute, copy or inline
// upload closes the run. The back checks the assumption: a draw storage write to a page of an
// open run latches runs off for the session (logged). EDEN_NX2_LIST_RUN=0 disables runs,
// =verify still drains every entry and compares the words read without a drain.
struct ListRun {
    std::mutex mutex;
    std::vector<u64> pages;          ///< device pages the last drain flushed
    std::atomic<bool> open{};
    std::atomic<bool> disabled{};
    std::atomic<u64> violations{};
    bool serving{};                  ///< front only: the current entry's macro may draw
};
inline ListRun g_list_run;

/// 0 off, 1 on, 2 verify.
[[nodiscard]] inline int ListRunMode() {
    static const int mode = [] {
        const char* const value = std::getenv("EDEN_NX2_LIST_RUN");
        if (value == nullptr || value[0] == 0) {
            return 1;
        }
        if (value[0] == 'v') {
            return 2;
        }
        return value[0] == '0' ? 0 : 1;
    }();
    return mode;
}

/// Front: a drain just flushed device pages [first, end) (page aligned).
inline void ListRunOpen(u64 first, u64 end) {
    if (ListRunMode() == 0 || g_list_run.disabled.load(std::memory_order_relaxed)) {
        return;
    }
    std::scoped_lock lock{g_list_run.mutex};
    if (!g_list_run.open.load(std::memory_order_relaxed)) {
        g_list_run.pages.clear();
    }
    for (u64 page = first >> GpuWrittenPageBits; page < (end >> GpuWrittenPageBits); ++page) {
        if (std::ranges::find(g_list_run.pages, page) == g_list_run.pages.end()) {
            g_list_run.pages.push_back(page);
        }
    }
    g_list_run.open.store(true, std::memory_order_release);
}

/// Front: something that may write list pages was sent.
inline void ListRunClose() {
    if (g_list_run.open.load(std::memory_order_relaxed)) {
        std::scoped_lock lock{g_list_run.mutex};
        g_list_run.open.store(false, std::memory_order_release);
    }
}

/// Front: the device page holding `device_address` is current in guest memory.
[[nodiscard]] inline bool ListRunCovers(u64 device_address) {
    if (!g_list_run.open.load(std::memory_order_acquire) ||
        g_list_run.disabled.load(std::memory_order_relaxed)) {
        return false;
    }
    std::scoped_lock lock{g_list_run.mutex};
    return g_list_run.open.load(std::memory_order_relaxed) &&
           std::ranges::find(g_list_run.pages, device_address >> GpuWrittenPageBits) !=
               g_list_run.pages.end();
}

/// Back: a draw or dispatch stores to the device page at `page_address`. True when that broke an
/// open run (runs are now off).
[[nodiscard]] inline bool ListRunNoteWrite(u64 page_address) {
    if (!g_list_run.open.load(std::memory_order_acquire)) {
        return false;
    }
    std::scoped_lock lock{g_list_run.mutex};
    if (!g_list_run.open.load(std::memory_order_relaxed) ||
        std::ranges::find(g_list_run.pages, page_address >> GpuWrittenPageBits) ==
            g_list_run.pages.end()) {
        return false;
    }
    g_list_run.violations.fetch_add(1, std::memory_order_relaxed);
    g_list_run.disabled.store(true, std::memory_order_relaxed);
    g_list_run.open.store(false, std::memory_order_release);
    return true;
}

/// [DEBUG] Fills `out` with up to `count` return addresses of the caller's callers as
/// RVA + 0x140000000 (for llvm-symbolizer --obj=eden.exe); zeros elsewhere. Windows only.
void DebugCaptureCallers(u64* out, u32 count);

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
