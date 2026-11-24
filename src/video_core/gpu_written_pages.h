// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
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

// [DEBUG] EDEN_NX2_LISTWRITE_BINDINGS=1: the storage binding being marked written, for the
// list page write tally: {program hash, stage<<56 | index<<48 | cbuf index<<16 | cbuf offset,
// bound size<<32 | write size}. g_dbg_stage_hash: the hash of the stage being bound.
inline thread_local u64 g_dbg_mark_ctx[4]{};
inline thread_local u64 g_dbg_stage_hash{};

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

// Written words of command-list pages. Bananza's culling shaders store each draw's arguments
// (20 bytes) into the command list right before the entry that reads them; with page granularity
// every other entry of the page (the macro call before the arguments, the state after) drains the
// split as well. Pages the GPU wrote while they were command-list pages record the words written,
// 4 bytes granular; a page written before it became one (or a write the buffer cache could not
// place) counts as written everywhere. Like the page bits, the words never clear.
// On with host indirect draws (EDEN_NX2_HOST_INDIRECT=1) or EDEN_NX2_LIST_GRANULES=1; =0 off.
struct ListGranules {
    std::shared_mutex mutex;
    std::unordered_map<u64, std::array<u64, 16>> pages; ///< page -> one bit per 4 bytes
};
inline ListGranules g_list_granules;

[[nodiscard]] inline bool HostIndirectEnabled();
[[nodiscard]] inline bool ListGranulesEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_LIST_GRANULES");
        if (value && value[0] != 0) {
            return value[0] == '1';
        }
        return HostIndirectEnabled();
    }();
    return enabled;
}

/// Back: the GPU writes [first, end) of the command-list page `page`. `end` clipped to the page.
inline void NoteListGranules(u64 page, u64 first, u64 end) {
    const u64 page_base = page << GpuWrittenPageBits;
    const u64 lo = ((std::max)(first, page_base) - page_base) / 4;
    const u64 hi = ((std::min)(end, page_base + (u64{1} << GpuWrittenPageBits)) - page_base + 3) / 4;
    if (lo >= hi) {
        return;
    }
    std::unique_lock lock{g_list_granules.mutex};
    auto& bits = g_list_granules.pages[page];
    for (u64 word = lo; word < hi;) {
        const u64 bit = word % 64;
        const u64 count = (std::min)(64 - bit, hi - word);
        bits[word / 64] |= (count == 64 ? ~u64{0} : ((u64{1} << count) - 1)) << bit;
        word += count;
    }
}

/// Front: whether the GPU wrote any word of [first, end) of the written page `page` (true when
/// the page has no word record).
[[nodiscard]] inline bool ListGranulesWritten(u64 page, u64 first, u64 end) {
    if (!ListGranulesEnabled()) {
        return true;
    }
    const u64 page_base = page << GpuWrittenPageBits;
    const u64 lo = ((std::max)(first, page_base) - page_base) / 4;
    const u64 hi = ((std::min)(end, page_base + (u64{1} << GpuWrittenPageBits)) - page_base + 3) / 4;
    std::shared_lock lock{g_list_granules.mutex};
    const auto it = g_list_granules.pages.find(page);
    if (it == g_list_granules.pages.end()) {
        return true;
    }
    for (u64 word = lo; word < hi; ++word) {
        if ((it->second[word / 64] >> (word % 64)) & 1) {
            return true;
        }
    }
    return false;
}

inline void MarkCommandListPage(u64 device_address) {
    const u64 page = device_address >> GpuWrittenPageBits;
    if (page >= GpuWrittenPageCount) {
        return;
    }
    const u64 old = g_command_list_pages[page / 64].fetch_or(u64{1} << (page % 64),
                                                             std::memory_order_seq_cst);
    g_has_command_list_pages.store(true, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (((old >> (page % 64)) & 1) == 0 && IsGpuWrittenPage(device_address)) {
        // Written before it was read as a command list: which words is not known.
        NoteListGranules(page, page << GpuWrittenPageBits, (page + 1) << GpuWrittenPageBits);
    }
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

// Host indexed indirect draws. Bananza's culling geometry shader stores each object's indexed draw
// arguments (index count, instance count; 5 words) into the command list right before the
// entry that passes them to its draw macro, ~1,300 times a frame. Reading them on the CPU drains
// the split and waits for the GPU each time. Instead the split front runs the macro with the
// words as the guest wrote them (only the counts differ) and the back turns the macro's indexed
// draw into a vkCmdDrawIndexedIndirect that reads the 5 words where the GPU wrote them, which is
// VkDrawIndexedIndirectCommand's layout. Opt-in for now: EDEN_NX2_HOST_INDIRECT=1.
[[nodiscard]] inline bool HostIndirectEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_HOST_INDIRECT");
        return value && value[0] == '1';
    }();
    return enabled;
}
/// Split front: the GPU address of the GPU-written arguments of the macro call being read.
inline thread_local u64 g_host_indirect_payload{};
/// Back: the next indexed draw reads its arguments at `address` (armed by the front's macro).
struct HostIndirectDraw {
    u64 address{};
    u32 count{};   ///< the index count the macro writes (the guest's word)
    u32 first{};   ///< the first index it writes
    bool armed{};
    bool active{}; ///< inside that draw
};
inline HostIndirectDraw g_host_indirect;

/// A new emulation session starts: no page has been written by its GPU yet.
inline void ClearGpuWrittenPages() {
    for (auto& word : g_command_list_pages) {
        word.store(0, std::memory_order_relaxed);
    }
    g_has_command_list_pages.store(false, std::memory_order_relaxed);
    for (auto& word : g_gpu_written_pages) {
        word.store(0, std::memory_order_relaxed);
    }
    std::unique_lock lock{g_list_granules.mutex};
    g_list_granules.pages.clear();
}

} // namespace VideoCommon
