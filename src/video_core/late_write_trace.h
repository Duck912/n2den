// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <deque>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"
#include "common/logging.h"
#include "common/n2den_trace.h"
#include "video_core/frame_progress.h"

// [DEBUG] EDEN_N2DEN_LATE_WRITE_TRACE=1: finds guest memory the game rewrites for a later frame
// before GPU emulation has read it for the frame it belongs to (the present's frames-ahead mode).
// Each CPU write to GPU-registered memory tags its pages with the guest's present count when it
// happened. When the back uploads CPU-written data (bind-time uploads by caller, held writes,
// texture uploads), a page tagged with a present count above the frames it has finished
// (FrameProgress) was written while the game prepared a later frame, so the back is about to
// give this frame the next frame's data. Every 4 s the late reads are logged by caller with the
// pages read late most often, and the CPU write traps on pages holding query reports.
namespace VideoCommon::LateWriteTrace {

[[nodiscard]] inline bool Enabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_N2DEN_LATE_WRITE_TRACE");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

namespace Detail {
inline constexpr u64 PageBits = 12;
struct PageStats {
    u64 reads{};
    u64 bytes{};
    const char* kind{};
};
inline std::mutex mutex;
inline std::unordered_map<u64, u32> page_tags;
inline std::unordered_map<u64, PageStats> late_pages;
inline std::unordered_map<const char*, u64> late_by_kind;
inline u64 reads{};
inline u64 late_reads{};
inline u64 late_bytes{};
inline std::chrono::steady_clock::time_point report{};
inline std::unordered_map<u64, u32> report_pages; ///< page -> reports registered there
inline u64 report_page_writes{};
inline std::unordered_map<u64, u64> report_write_pages;
/// Pages the back read: marked cached once, so that CPU writes to them trap and get tagged.
inline std::unordered_set<u64> watched_pages;
inline std::function<void(u64)> watch_page;
inline thread_local bool on_back = false;

// Submissions: the guest's command submissions that advance a syncpoint, numbered in order, and
// how many of them the back has finished (it reached their syncpoint value). A page's submit tag
// is the number of submissions made when the CPU last wrote it.
struct Submit {
    u64 sequence;
    u32 id;
    u32 value;
};
inline std::deque<Submit> submits;
inline std::atomic<u64> submit_count{0};
inline std::atomic<u64> submits_finished{0};
inline std::unordered_map<u64, u64> page_submit_tags;
inline u64 submit_late_reads{}; ///< late by submission but not by present
inline u64 submit_late_bytes{};
inline std::unordered_map<const char*, u64> submit_late_by_kind;
} // namespace Detail

/// Guest side: a submission whose work ends when syncpoint `id` reaches `value`.
inline void NoteSubmit(u32 id, u32 value) {
    if (!Enabled()) {
        return;
    }
    std::scoped_lock lock{Detail::mutex};
    const u64 sequence = Detail::submit_count.fetch_add(1, std::memory_order_relaxed) + 1;
    Detail::submits.push_back({sequence, id, value});
}

/// Back side, after it advanced a syncpoint: the submissions finished in order so far.
inline void NoteSyncpoint() {
    if (!Enabled() || !FrameProgress::Detail::syncpoint_value) {
        return;
    }
    std::scoped_lock lock{Detail::mutex};
    while (!Detail::submits.empty()) {
        const Detail::Submit& submit = Detail::submits.front();
        const u32 current = FrameProgress::Detail::syncpoint_value(submit.id);
        if (static_cast<s32>(current - submit.value) < 0) {
            break;
        }
        Detail::submits_finished.store(submit.sequence, std::memory_order_relaxed);
        Detail::submits.pop_front();
    }
}

/// The rasterizer: marks a page cached (CPU writes to it trap into OnCPUWrite, NoteCpuWrite).
inline void SetPageWatcher(std::function<void(u64)>&& watch) {
    Detail::watch_page = std::move(watch);
}

/// The GPU back thread (GpuSplit::BackLoop). Only its reads count at the generic read sites
/// (descriptor tables, MemoryManager reads): the front parses later frames' command lists.
inline void MarkBackThread() {
    Detail::on_back = true;
}
[[nodiscard]] inline bool OnBackThread() {
    return Detail::on_back;
}

/// Back: a query report was registered at `address`.
inline void NoteReport(u64 address) {
    if (!Enabled()) {
        return;
    }
    std::scoped_lock lock{Detail::mutex};
    ++Detail::report_pages[address >> Detail::PageBits];
}

/// Guest side: the CPU wrote [address, address + size) of GPU-registered memory.
inline void NoteCpuWrite(u64 address, u64 size) {
    if (!Enabled() || size == 0) {
        return;
    }
    const u32 tag = Common::N2denTrace::present_count.load(std::memory_order_acquire);
    std::scoped_lock lock{Detail::mutex};
    const u64 submit_tag = Detail::submit_count.load(std::memory_order_relaxed);
    for (u64 page = address >> Detail::PageBits; page <= (address + size - 1) >> Detail::PageBits;
         ++page) {
        Detail::page_tags[page] = tag;
        Detail::page_submit_tags[page] = submit_tag;
        if (Detail::report_pages.contains(page)) {
            ++Detail::report_page_writes;
            ++Detail::report_write_pages[page];
        }
    }
}

/// Back side: CPU-written [address, address + size) is uploaded or applied now.
inline void NoteBackRead(u64 address, u64 size, const char* kind) {
    if (!Enabled() || size == 0) {
        return;
    }
    const u32 finished = FrameProgress::Finished();
    if (Detail::watch_page) {
        std::vector<u64> new_pages;
        {
            std::scoped_lock lock{Detail::mutex};
            for (u64 page = address >> Detail::PageBits;
                 page <= (address + size - 1) >> Detail::PageBits; ++page) {
                if (Detail::watched_pages.insert(page).second) {
                    new_pages.push_back(page);
                }
            }
        }
        for (const u64 page : new_pages) {
            Detail::watch_page(page << Detail::PageBits);
        }
    }
    std::scoped_lock lock{Detail::mutex};
    ++Detail::reads;
    bool late = false;
    bool submit_late = false;
    const u64 submits_finished = Detail::submits_finished.load(std::memory_order_relaxed);
    for (u64 page = address >> Detail::PageBits; page <= (address + size - 1) >> Detail::PageBits;
         ++page) {
        const auto it = Detail::page_tags.find(page);
        if (it == Detail::page_tags.end() || it->second <= finished) {
            // Written before the present of the frame the back is on. Written after the
            // submission the back is executing was made, all the same?
            const auto submit_it = Detail::page_submit_tags.find(page);
            if (submit_it != Detail::page_submit_tags.end() &&
                submit_it->second > submits_finished) {
                submit_late = true;
            }
            continue;
        }
        late = true;
        auto& stats = Detail::late_pages[page];
        ++stats.reads;
        stats.bytes += size;
        stats.kind = kind;
    }
    if (late) {
        ++Detail::late_reads;
        Detail::late_bytes += size;
        ++Detail::late_by_kind[kind];
    } else if (submit_late) {
        ++Detail::submit_late_reads;
        Detail::submit_late_bytes += size;
        ++Detail::submit_late_by_kind[kind];
    }
    const auto now = std::chrono::steady_clock::now();
    if (Detail::report == std::chrono::steady_clock::time_point{}) {
        Detail::report = now;
    }
    if (now - Detail::report < std::chrono::seconds{4}) {
        return;
    }
    Detail::report = now;
    std::vector<std::pair<u64, Detail::PageStats>> top(Detail::late_pages.begin(),
                                                       Detail::late_pages.end());
    std::ranges::sort(top, [](const auto& a, const auto& b) { return a.second.reads > b.second.reads; });
    // A caller's function name, without its class and parameters.
    const auto short_name = [](const char* name) -> std::string {
        std::string text{name ? name : "?"};
        const size_t open = text.find('(');
        if (open != std::string::npos) {
            text.resize(open);
        }
        const size_t scope = text.rfind("::");
        return scope == std::string::npos ? text : text.substr(scope + 2);
    };
    std::string list;
    for (size_t i = 0; i < (std::min)(top.size(), size_t{10}); ++i) {
        list += fmt::format(" {:#x}:{}x{}", top[i].first << Detail::PageBits, top[i].second.reads,
                            short_name(top[i].second.kind));
    }
    std::string report_list;
    for (const auto& [page, count] : Detail::report_write_pages) {
        if (report_list.size() < 200) {
            report_list += fmt::format(" {:#x}:{}", page << Detail::PageBits, count);
        }
    }
    LOG_INFO(HW_GPU, "Late write trace: {} CPU write traps on query report pages ({} report pages):{}",
             Detail::report_page_writes, Detail::report_pages.size(), report_list);
    Detail::report_page_writes = 0;
    Detail::report_write_pages.clear();
    std::string kinds;
    for (const auto& [kind_name, count] : Detail::late_by_kind) {
        kinds += fmt::format(" {}={}", short_name(kind_name), count);
    }
    LOG_INFO(HW_GPU,
             "Late write trace: {} uploads of CPU-written data, {} read pages written for a later "
             "frame ({} bytes), {} pages; by source:{}; most:{}",
             Detail::reads, Detail::late_reads, Detail::late_bytes, Detail::late_pages.size(), kinds,
             list);
    std::string submit_kinds;
    for (const auto& [kind_name, count] : Detail::submit_late_by_kind) {
        submit_kinds += fmt::format(" {}={}", short_name(kind_name), count);
    }
    LOG_INFO(HW_GPU,
             "Late write trace: {} reads of pages written before the frame's present but after "
             "the submission the back is on ({} bytes); by source:{}; submissions made {}, "
             "finished {}",
             Detail::submit_late_reads, Detail::submit_late_bytes, submit_kinds,
             Detail::submit_count.load(std::memory_order_relaxed), submits_finished);
    Detail::submit_late_reads = 0;
    Detail::submit_late_bytes = 0;
    Detail::submit_late_by_kind.clear();
    Detail::late_by_kind.clear();
    Detail::reads = 0;
    Detail::late_reads = 0;
    Detail::late_bytes = 0;
    Detail::late_pages.clear();
}

} // namespace VideoCommon::LateWriteTrace
