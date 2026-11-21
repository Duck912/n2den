// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "common/common_types.h"
#include "common/logging.h"
#include "common/n2den_trace.h"

// Guest CPU reads of query report slots whose report is still on its way wait for it.
//
// Below High GPU accuracy the guest syncpoint of a frame passes when GPU emulation processes it,
// while the frame's counter reports (occlusion sample counts) reach guest memory only when the
// host GPU finished them. When the present runs a frame ahead (EDEN_NX2_FRAMES_AHEAD=1), the game
// reads the previous frame's report slots right after that frame's fence passed: in Ocarina of
// Time's Kokiri Forest with GPU emulation behind, about 900 reads a second of slots whose report
// was queued, half of them then written with a different value, and the doorway of Link's house
// glowed white for a frame. With the present waiting
// for its own frame there were none.
//
// A page holding a queued report is marked cached, so guest CPU reads fault into GPU::OnCPURead,
// which calls OnRead: a read of a slot with a queued report waits until the report was written
// (its fence's operations ran on the release thread), at most MaxWait. The report's fence is
// queued before the frame's syncpoint passes, so the wait is for the host GPU only. Active while
// the native present runs frames ahead (Common::N2denPresent::frames_ahead not 0);
// EDEN_N2DEN_REPORT_READ_WAIT=0 or 1 turns it off or on for every title.
// [DEBUG] EDEN_N2DEN_REPORT_READ_TRACE=1 logs the reads and waits every 4 s.
namespace VideoCommon::ReportReadWait {

inline constexpr auto MaxWait = std::chrono::milliseconds{100};

namespace Detail {
inline constexpr u64 PageBits = 12;

[[nodiscard]] inline int Forced() {
    static const int forced = [] {
        const char* const value = std::getenv("EDEN_N2DEN_REPORT_READ_WAIT");
        return value == nullptr ? -1 : (value[0] == '0' ? 0 : 1);
    }();
    return forced;
}

[[nodiscard]] inline bool TraceEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_N2DEN_REPORT_READ_TRACE");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

inline std::mutex mutex;
inline std::condition_variable landed_cv;
inline std::unordered_map<u64, u32> slots;       ///< report address -> reports queued there
inline std::unordered_map<u64, u32> page_queued; ///< page -> reports queued on it
/// Pages that held a report: a read there never returns an area, so that each core's next read
/// of the page traps again (HandleRasterizerDownload skips reads inside the last area returned).
inline std::unordered_set<u64> report_pages;
inline u32 waiters{};

struct Stats {
    u64 queued{};
    u64 reads{};        ///< reads of pages that held a report
    u64 waits{};        ///< reads of slots with a queued report
    u64 wait_us{};
    u64 worst_us{};
    u64 timeouts{};
    std::chrono::steady_clock::time_point report{};
};
inline Stats stats;

/// Logs and resets the counts every 4 s. Called with the mutex held.
inline void MaybeLog() {
    if (!TraceEnabled()) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (stats.report == std::chrono::steady_clock::time_point{}) {
        stats.report = now;
    }
    if (now - stats.report < std::chrono::seconds{4}) {
        return;
    }
    LOG_INFO(HW_GPU,
             "Report read wait: {} reports queued, {} pages queued now; {} reads of report pages, "
             "{} waited for a queued report ({} us on average, worst {} us, {} gave up)",
             stats.queued, page_queued.size(), stats.reads, stats.waits,
             stats.waits ? stats.wait_us / stats.waits : 0, stats.worst_us, stats.timeouts);
    stats = Stats{.report = now};
}

[[nodiscard]] inline bool AnyQueued(u64 address, u64 size) {
    for (u64 slot = address & ~u64{3}; slot < address + size; slot += 4) {
        const auto it = slots.find(slot);
        if (it != slots.end() && it->second != 0) {
            return true;
        }
    }
    return false;
}
} // namespace Detail

/// Whether reports queued now are waited for by CPU reads.
[[nodiscard]] inline bool Active() {
    const int forced = Detail::Forced();
    if (forced >= 0) {
        return forced != 0;
    }
    return Common::N2denPresent::frames_ahead.load(std::memory_order_relaxed) != 0;
}

/// Back: a report for `address` will be written later, by an operation that calls Landed after
/// the write. Its page traps CPU reads until every report queued on it landed.
template <typename DeviceMemory>
void Queued(DeviceMemory& device_memory, u64 address) {
    std::scoped_lock lock{Detail::mutex};
    ++Detail::stats.queued;
    ++Detail::slots[address];
    const u64 page = address >> Detail::PageBits;
    Detail::report_pages.insert(page);
    if (Detail::page_queued[page]++ == 0) {
        device_memory.UpdatePagesCachedCount(page << Detail::PageBits, u64{1} << Detail::PageBits, 1);
    }
    Detail::MaybeLog();
}

/// Release or back thread: the report Queued for `address` was written, or dropped.
template <typename DeviceMemory>
void Landed(DeviceMemory& device_memory, u64 address) {
    std::scoped_lock lock{Detail::mutex};
    const auto it = Detail::slots.find(address);
    if (it == Detail::slots.end()) {
        return;
    }
    if (--it->second == 0) {
        Detail::slots.erase(it);
    }
    const u64 page = address >> Detail::PageBits;
    const auto page_it = Detail::page_queued.find(page);
    if (page_it != Detail::page_queued.end() && --page_it->second == 0) {
        Detail::page_queued.erase(page_it);
        device_memory.UpdatePagesCachedCount(page << Detail::PageBits, u64{1} << Detail::PageBits, -1);
    }
    if (Detail::waiters != 0) {
        Detail::landed_cv.notify_all();
    }
}

/// Guest CPU read of [address, address + size) trapped (any thread, no locks held). Waits while a
/// report is queued for it. Returns whether the read is on a page that held a report: the caller
/// then returns an empty area so that the next read of it traps too.
[[nodiscard]] inline bool OnRead(u64 address, u64 size) {
    std::unique_lock lock{Detail::mutex};
    if (Detail::report_pages.empty()) {
        return false;
    }
    const u64 first_page = address >> Detail::PageBits;
    const u64 last_page = (address + size - 1) >> Detail::PageBits;
    if (!Detail::report_pages.contains(first_page) && !Detail::report_pages.contains(last_page)) {
        return false;
    }
    ++Detail::stats.reads;
    if (Detail::AnyQueued(address, size)) {
        const auto start = std::chrono::steady_clock::now();
        ++Detail::waiters;
        const bool landed = Detail::landed_cv.wait_for(
            lock, MaxWait, [&] { return !Detail::AnyQueued(address, size); });
        --Detail::waiters;
        const u64 waited_us = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() -
                                                                  start)
                .count());
        ++Detail::stats.waits;
        Detail::stats.wait_us += waited_us;
        Detail::stats.worst_us = (std::max)(Detail::stats.worst_us, waited_us);
        if (!landed) {
            ++Detail::stats.timeouts;
            static u32 logged = 0;
            if (logged < 16) {
                ++logged;
                LOG_WARNING(HW_GPU, "Query report at {:#x} not written within {} ms of a CPU read",
                            address, MaxWait.count());
            }
        }
    }
    Detail::MaybeLog();
    return true;
}

} // namespace VideoCommon::ReportReadWait
