// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"
#include "common/n2den_trace.h"
#include "video_core/frame_progress.h"
#include "video_core/late_write_trace.h"
#include "video_core/memory_manager.h"

// [EXPERIMENT] Guest memory that GPU emulation reads for a frame, as it was for that frame.
//
// When the present runs a frame ahead, the game prepares the next frame while GPU emulation still
// draws the last, and in Ocarina of Time it rewrites data the last frame's draws have yet to read:
// texture descriptors, texture handles, storage buffer pointers, and buffer data uploaded at bind
// time or applied as held writes (the doorway of Link's house glowed white for a frame).
// A page the back reads that way is watched: marked cached, so
// that CPU writes to it trap. The first write to a watched page after a present, while GPU
// emulation is behind, keeps the page's contents, and the back's reads for the earlier frame come
// from that copy until it finished the frame. Descriptor, handle and pointer pages are watched when
// read; buffer pages once a read found them written for a later frame.
// EDEN_N2DEN_FRAME_SNAPSHOT=1 turns it on (off by default): it cut OoT's one-frame glows by about
// 80% only while every uploaded page was watched, and then halved its frame rate, since every CPU
// write to a watched page traps. The adaptive watching here costs about as much and did not
// reduce them. Kept for a later attempt.
namespace VideoCommon::FrameSnapshot {

inline constexpr u64 PageBits = 12;
inline constexpr u64 PageSize = u64{1} << PageBits;
inline constexpr u64 NumPages = u64{1} << (34 - PageBits); ///< the 34-bit device address space

namespace Detail {
enum : u8 { Watched = 1, HasCopy = 2 };

struct Copy {
    u32 tag; ///< present count when it was taken: holds the data of frames up to this one
    std::array<u8, PageSize> bytes;
};

/// Per page, read without the lock: whether it is watched and whether it has a copy, and the
/// present count of the last CPU write that trapped while GPU emulation was behind.
inline std::unique_ptr<std::atomic<u8>[]> flags;
inline std::unique_ptr<std::atomic<u32>[]> late_tags;

inline std::mutex mutex;
inline std::mutex watch_mutex; ///< orders the watch and unwatch of pages (never held with `mutex`)
inline std::unordered_map<u64, std::deque<std::unique_ptr<Copy>>> copies;
inline std::function<void(u64)> watch_page;
inline std::function<void(u64)> unwatch_page;
/// Mode 2: pages whose watch a CPU write took off, watched again at the next present.
inline std::vector<u64> rearm_pages;
inline std::function<const u8*(u64)> page_pointer;

struct Stats {
    u64 copies{};
    u64 served{};
    u64 watched{};
    u64 forwarded{}; ///< GPU writes copied into a page's copies
    u64 traps{};     ///< Mode 2: CPU writes that took a page's watch off
    u64 rearmed{};   ///< Mode 2: pages watched again at a present
};
inline Stats stats;
/// Overlay: pages last written for a later frame than the back is on, with and without a copy
/// to serve them from (the second kind is read live: the next frame's data).
inline std::atomic<u64> overlay_late_served{0};
inline std::atomic<u64> overlay_late_live{0};
inline std::chrono::steady_clock::time_point report{};
inline u32 dropped_through{};

[[nodiscard]] inline bool Ready(u64 page) {
    return flags != nullptr && page < NumPages;
}

/// Watches a page: marks it cached once (CPU writes to it trap from then on).
inline void WatchPage(u64 page) {
    if ((flags[page].load(std::memory_order_relaxed) & Watched) != 0) {
        return;
    }
    bool added = false;
    {
        // The flag and the page's cached count change together: a write trap taking the watch
        // off (mode 2) must not run its decrement before this increment.
        std::scoped_lock watch_lock{watch_mutex};
        if ((flags[page].fetch_or(Watched, std::memory_order_acq_rel) & Watched) == 0) {
            if (watch_page) {
                watch_page(page << PageBits);
            }
            added = true;
        }
    }
    if (added) {
        std::scoped_lock lock{mutex};
        ++stats.watched;
    }
}

/// Mode 2: takes a page's watch off. False when it was not watched.
inline bool UnwatchPage(u64 page) {
    std::scoped_lock watch_lock{watch_mutex};
    if ((flags[page].fetch_and(static_cast<u8>(~Watched), std::memory_order_acq_rel) & Watched) ==
        0) {
        return false;
    }
    unwatch_page(page << PageBits);
    return true;
}

/// Logs the counts every 4 s. Called with the mutex held.
inline void MaybeLog() {
    const auto now = std::chrono::steady_clock::now();
    if (now - report < std::chrono::seconds{4}) {
        return;
    }
    report = now;
    LOG_INFO(HW_GPU, "Frame snapshot: {} page copies, {} reads served from a copy, {} pages "
                     "newly watched, {} pages with copies, {} GPU writes forwarded to copies, "
                     "{} write traps that took a watch off, {} pages watched again at presents",
             stats.copies, stats.served, stats.watched, copies.size(), stats.forwarded, stats.traps,
             stats.rearmed);
    LOG_INFO(HW_GPU, "Frame snapshot: upload and held-write pages written for a later frame: {} "
                     "served from a copy, {} read live",
             overlay_late_served.exchange(0, std::memory_order_relaxed),
             overlay_late_live.exchange(0, std::memory_order_relaxed));
    stats = {};
}

/// Drops copies the back no longer needs. Called with the mutex held.
inline void DropFinished(u32 finished) {
    if (finished == dropped_through) {
        return;
    }
    dropped_through = finished;
    for (auto it = copies.begin(); it != copies.end();) {
        auto& list = it->second;
        while (!list.empty() && list.front()->tag <= finished) {
            list.pop_front();
        }
        if (list.empty()) {
            flags[it->first].fetch_and(static_cast<u8>(~HasCopy), std::memory_order_release);
            it = copies.erase(it);
        } else {
            ++it;
        }
    }
}

/// The copy holding the data of the frame the back is on, or null. Called with the mutex held.
[[nodiscard]] inline const Copy* CopyFor(u64 page, u32 finished) {
    DropFinished(finished);
    const auto it = copies.find(page);
    // The oldest copy newer than the finished frames holds the data of the frame the back is on.
    return it == copies.end() || it->second.empty() ? nullptr : it->second.front().get();
}
} // namespace Detail

[[nodiscard]] inline bool Active() {
    static const bool active = [] {
        const char* const value = std::getenv("EDEN_N2DEN_FRAME_SNAPSHOT");
        return value != nullptr && (value[0] == '1' || value[0] == '2');
    }();
    return active;
}

/// EDEN_N2DEN_FRAME_SNAPSHOT=2: a page traps one CPU write a frame. The first write after a
/// present keeps the page's contents (when GPU emulation is behind) and takes the watch off, so
/// the frame's other writes to it cost nothing; the next present watches it again. Every page the
/// buffer cache uploads is watched. Uploads measured 160-240 distinct pages a frame
/// (EDEN_N2DEN_UPLOAD_STATS), against a trap on every write in mode 1.
[[nodiscard]] inline bool Rearm() {
    static const bool rearm = [] {
        const char* const value = std::getenv("EDEN_N2DEN_FRAME_SNAPSHOT");
        return value != nullptr && value[0] == '2';
    }();
    return rearm;
}

/// Which reads use the copies (EDEN_N2DEN_FRAME_SNAPSHOT_PARTS, a mask, default all): 1 texture
/// descriptors, handles and storage buffer pointers, 2 buffer uploads, 4 held CPU writes.
[[nodiscard]] inline bool Part(u32 part) {
    static const u32 parts = [] {
        const char* const value = std::getenv("EDEN_N2DEN_FRAME_SNAPSHOT_PARTS");
        return value == nullptr ? 7U : static_cast<u32>(std::strtoul(value, nullptr, 0));
    }();
    return Active() && (parts & part) != 0;
}

/// The rasterizer: how to mark a device page cached and where its host memory is.
inline void SetHooks(std::function<void(u64)>&& watch, std::function<void(u64)>&& unwatch,
                     std::function<const u8*(u64)>&& pointer) {
    Detail::watch_page = std::move(watch);
    Detail::unwatch_page = std::move(unwatch);
    Detail::page_pointer = std::move(pointer);
    if (Active() && !Detail::flags) {
        Detail::flags = std::make_unique<std::atomic<u8>[]>(NumPages);
        Detail::late_tags = std::make_unique<std::atomic<u32>[]>(NumPages);
    }
}

/// CPU write trap, before the write lands: notes writes made while GPU emulation is behind the
/// frame they belong to, and keeps a watched page's contents for it.
inline void OnCpuWrite(u64 address, u64 size) {
    if (size == 0 || !Detail::page_pointer || !Detail::flags) {
        return;
    }
    const u32 tag = Common::N2denTrace::present_count.load(std::memory_order_acquire);
    const u32 finished = FrameProgress::Finished();
    const bool rearm = Rearm();
    if (tag <= finished && !rearm) {
        return;
    }
    for (u64 page = address >> PageBits; page <= (address + size - 1) >> PageBits; ++page) {
        if (page >= NumPages) {
            break;
        }
        if (tag > finished) {
            Detail::late_tags[page].store(tag, std::memory_order_relaxed);
        }
        if ((Detail::flags[page].load(std::memory_order_acquire) & Detail::Watched) == 0) {
            continue;
        }
        if (rearm) {
            // This frame's later writes to the page need no copy: stop trapping them.
            if (!Detail::UnwatchPage(page)) {
                continue;
            }
            {
                std::scoped_lock lock{Detail::mutex};
                Detail::rearm_pages.push_back(page);
                ++Detail::stats.traps;
                if (tag <= finished) {
                    Detail::MaybeLog();
                }
            }
            if (tag <= finished) {
                continue;
            }
        }
        std::scoped_lock lock{Detail::mutex};
        auto& list = Detail::copies[page];
        if (!list.empty() && list.back()->tag >= tag) {
            continue;
        }
        const u8* const host = Detail::page_pointer(page << PageBits);
        if (host == nullptr) {
            continue;
        }
        auto copy = std::make_unique<Detail::Copy>();
        copy->tag = tag;
        std::memcpy(copy->bytes.data(), host, PageSize);
        list.push_back(std::move(copy));
        Detail::flags[page].fetch_or(Detail::HasCopy, std::memory_order_release);
        ++Detail::stats.copies;
        Detail::MaybeLog();
    }
}

/// Guest side, right after a present was counted (mode 2): the pages whose watch this frame's
/// writes took off are watched again, so the next frame's first write to each keeps its contents.
inline void OnPresent() {
    if (!Rearm() || !Detail::flags) {
        return;
    }
    std::vector<u64> pages;
    {
        std::scoped_lock lock{Detail::mutex};
        pages.swap(Detail::rearm_pages);
        Detail::stats.rearmed += pages.size();
    }
    for (const u64 page : pages) {
        Detail::WatchPage(page);
    }
}

/// Whether the snapshot is set up (EDEN_N2DEN_FRAME_SNAPSHOT set and the rasterizer's hooks given).
/// A plain load, for the hot GPU write paths.
[[nodiscard]] inline bool Enabled() {
    return Detail::flags != nullptr;
}

/// GPU emulation wrote [device_address, device_address + size) of guest memory (query and
/// semaphore releases, DMA and inline uploads, buffer download write-backs), after the write
/// landed. The copies of those pages get the same bytes: the GPU's write belongs to the frame it is
/// drawing, which the copies stand for, and a read served from a copy would otherwise miss it.
/// Eglantine's run-ahead probes (5550392e8b) found every byte its snapshot still got wrong was such
/// a write (forwarded there by 07ae2a95b2).
inline void OnGpuWrite(u64 device_address, const void* src, u64 size) {
    if (size == 0 || !Enabled()) {
        return;
    }
    const u8* const bytes = static_cast<const u8*>(src);
    const u64 end = device_address + size;
    for (u64 page = device_address >> PageBits; page <= (end - 1) >> PageBits; ++page) {
        if (page >= NumPages) {
            break;
        }
        if ((Detail::flags[page].load(std::memory_order_acquire) & Detail::HasCopy) == 0) {
            continue;
        }
        const u32 finished = FrameProgress::Finished();
        std::scoped_lock lock{Detail::mutex};
        Detail::DropFinished(finished);
        const auto it = Detail::copies.find(page);
        if (it == Detail::copies.end()) {
            continue;
        }
        const u64 page_start = page << PageBits;
        const u64 start = (std::max)(page_start, device_address);
        const u64 stop = (std::min)(page_start + PageSize, end);
        for (const auto& copy : it->second) {
            std::memcpy(copy->bytes.data() + (start - page_start), bytes + (start - device_address),
                        stop - start);
        }
        ++Detail::stats.forwarded;
    }
}

/// Marks the pages of [device_address, device_address + size) watched.
inline void Watch(u64 device_address, u64 size) {
    if (size == 0) {
        return;
    }
    for (u64 page = device_address >> PageBits; page <= (device_address + size - 1) >> PageBits;
         ++page) {
        if (!Detail::Ready(page)) {
            return;
        }
        Detail::WatchPage(page);
    }
}

/// Whether the page of `device_address` may have a copy.
[[nodiscard]] inline bool HasCopy(u64 device_address) {
    const u64 page = device_address >> PageBits;
    return Detail::Ready(page) &&
           (Detail::flags[page].load(std::memory_order_acquire) & Detail::HasCopy) != 0;
}

/// GPU emulation reads [device_address, device_address + size) live: watches its page and fills
/// `dest` from the copy for the frame it is on, if any. False: read guest memory.
inline bool Read(u64 device_address, void* dest, size_t size) {
    const u64 page = device_address >> PageBits;
    if (!Detail::Ready(page) || ((device_address + size - 1) >> PageBits) != page) {
        return false;
    }
    Detail::WatchPage(page);
    if ((Detail::flags[page].load(std::memory_order_acquire) & Detail::HasCopy) == 0) {
        return false;
    }
    const u32 finished = FrameProgress::Finished();
    std::scoped_lock lock{Detail::mutex};
    const Detail::Copy* const copy = Detail::CopyFor(page, finished);
    if (copy == nullptr) {
        return false;
    }
    std::memcpy(dest, copy->bytes.data() + (device_address & (PageSize - 1)), size);
    ++Detail::stats.served;
    return true;
}

/// Back: overwrites the bytes of `dest` (guest data of [device_address, device_address + size))
/// whose pages have a copy for the frame the back is on, and watches pages it finds written for a
/// later frame without one. Returns whether any byte came from a copy, with the newest copy's tag.
inline bool Overlay(u64 device_address, u8* dest, u64 size, u32& tag) {
    if (size == 0 || !Detail::flags) {
        return false;
    }
    const u32 finished = FrameProgress::Finished();
    bool any = false;
    for (u64 page = device_address >> PageBits; page <= (device_address + size - 1) >> PageBits;
         ++page) {
        if (page >= NumPages) {
            break;
        }
        const u8 page_flags = Detail::flags[page].load(std::memory_order_acquire);
        if (Rearm() && (page_flags & Detail::Watched) == 0) {
            // Every uploaded page is watched; one a write unwatched this frame traps once more.
            Detail::WatchPage(page);
        }
        const bool late = Detail::late_tags[page].load(std::memory_order_relaxed) > finished;
        if ((page_flags & Detail::HasCopy) == 0) {
            if (late) {
                Detail::overlay_late_live.fetch_add(1, std::memory_order_relaxed);
            }
            if ((page_flags & Detail::Watched) == 0 && !Rearm() &&
                Detail::late_tags[page].load(std::memory_order_relaxed) > finished) {
                // Written for a later frame and read now: watched from now on, so that its next
                // late write keeps a copy.
                Detail::WatchPage(page);
            }
            continue;
        }
        std::scoped_lock lock{Detail::mutex};
        const Detail::Copy* const copy = Detail::CopyFor(page, finished);
        if (copy == nullptr) {
            if (late) {
                Detail::overlay_late_live.fetch_add(1, std::memory_order_relaxed);
            }
            continue;
        }
        if (late) {
            Detail::overlay_late_served.fetch_add(1, std::memory_order_relaxed);
        }
        const u64 page_start = page << PageBits;
        const u64 start = (std::max)(page_start, device_address);
        const u64 end = (std::min)(page_start + PageSize, device_address + size);
        std::memcpy(dest + (start - device_address), copy->bytes.data() + (start - page_start),
                    end - start);
        tag = any ? (std::max)(tag, copy->tag) : copy->tag;
        any = true;
        ++Detail::stats.served;
    }
    return any;
}

namespace Detail {
struct Deferred {
    u64 address;
    u64 size;
    u32 tag;
};
inline std::mutex deferred_mutex;
inline std::vector<Deferred> deferred;
inline std::atomic<u32> deferred_count{0};
} // namespace Detail

/// A held CPU write applied from a copy: applied again from guest memory once the back finished
/// the frame of the copy (TakeDue).
inline void Defer(u64 address, u64 size, u32 tag) {
    std::scoped_lock lock{Detail::deferred_mutex};
    Detail::deferred.push_back({address, size, tag});
    Detail::deferred_count.fetch_add(1, std::memory_order_relaxed);
}

/// Back: the deferred writes the back may now apply, to `apply(address, size)`.
template <typename Apply>
void TakeDue(Apply&& apply) {
    if (Detail::deferred_count.load(std::memory_order_relaxed) == 0) {
        return;
    }
    const u32 finished = FrameProgress::Finished();
    std::scoped_lock lock{Detail::deferred_mutex};
    auto& list = Detail::deferred;
    size_t kept = 0;
    for (size_t i = 0; i < list.size(); ++i) {
        if (list[i].tag <= finished) {
            apply(list[i].address, list[i].size);
        } else {
            list[kept++] = list[i];
        }
    }
    Detail::deferred_count.fetch_sub(static_cast<u32>(list.size() - kept), std::memory_order_relaxed);
    list.resize(kept);
}

/// A live read of a T at a GPU address, from the frame's copy when there is one.
template <typename T>
[[nodiscard]] T ReadGuest(const Tegra::MemoryManager& gpu_memory, GPUVAddr address) {
    if (Part(1) && LateWriteTrace::OnBackThread()) {
        if (const auto device = gpu_memory.GpuToCpuAddress(address)) {
            T value;
            if (Read(*device, &value, sizeof(T))) {
                return value;
            }
        }
    }
    return gpu_memory.Read<T>(address);
}

} // namespace VideoCommon::FrameSnapshot
