// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"

// [DEBUG] EDEN_N2DEN_UPLOAD_STATS=1: how much CPU-written guest memory the buffer cache uploads
// per frame (a frame ends at the rasterizer's TickFrame), counted in distinct 4 KiB pages.
// It sizes a capture of that data at the guest's submission instead of at the back's bind: the
// bytes are what would be copied each frame, the distinct pages are the write traps a re-armed
// page tracker would take (one per page per frame), and the pages also uploaded in the frame
// before are the ones the game rewrites every frame. Logged every 4 s.
namespace VideoCommon::UploadStats {

[[nodiscard]] inline bool Enabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_N2DEN_UPLOAD_STATS");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

namespace Detail {
inline constexpr u64 PageBits = 12;
inline std::mutex mutex;
inline std::unordered_set<u64> frame_pages;
inline std::unordered_set<u64> previous_pages;
inline std::unordered_map<u64, u32> window_pages; ///< page -> frames of the window that uploaded it
inline u64 frame_bytes{};
inline u64 frame_copies{};
inline std::vector<u64> pages_per_frame;
inline std::vector<u64> bytes_per_frame;
inline u64 window_copies{};
inline u64 window_repeats{};
inline std::chrono::steady_clock::time_point report{};

[[nodiscard]] inline u64 Percentile(std::vector<u64>& values, u64 percent) {
    if (values.empty()) {
        return 0;
    }
    const size_t index = (std::min)(values.size() - 1, values.size() * percent / 100);
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return values[index];
}
} // namespace Detail

/// The buffer cache: one range of CPU-written guest memory copied to a host buffer.
inline void NoteUpload(u64 device_addr, u64 size) {
    if (size == 0) {
        return;
    }
    std::scoped_lock lock{Detail::mutex};
    Detail::frame_bytes += size;
    ++Detail::frame_copies;
    const u64 last = (device_addr + size - 1) >> Detail::PageBits;
    for (u64 page = device_addr >> Detail::PageBits; page <= last; ++page) {
        Detail::frame_pages.insert(page);
    }
}

/// The rasterizer's TickFrame: closes the frame and logs every 4 s.
inline void TickFrame() {
    using namespace Detail;
    std::scoped_lock lock{mutex};
    u64 repeats = 0;
    for (const u64 page : frame_pages) {
        repeats += previous_pages.contains(page) ? 1 : 0;
        ++window_pages[page];
    }
    pages_per_frame.push_back(frame_pages.size());
    bytes_per_frame.push_back(frame_bytes);
    window_copies += frame_copies;
    window_repeats += repeats;
    previous_pages.swap(frame_pages);
    frame_pages.clear();
    frame_bytes = 0;
    frame_copies = 0;

    const auto now = std::chrono::steady_clock::now();
    if (report == std::chrono::steady_clock::time_point{}) {
        report = now;
    }
    if (now - report < std::chrono::seconds{4}) {
        return;
    }
    report = now;
    const u64 frames = pages_per_frame.size();
    u64 pages = 0;
    u64 bytes = 0;
    for (size_t i = 0; i < frames; ++i) {
        pages += pages_per_frame[i];
        bytes += bytes_per_frame[i];
    }
    // Pages uploaded in at least half of the window's frames: the every-frame working set.
    u64 steady = 0;
    for (const auto& [page, count] : window_pages) {
        steady += count * 2 >= frames ? 1 : 0;
    }
    LOG_INFO(Render,
             "[upload-stats] frames {}; per frame: pages mean {} p50 {} p95 {} max {}, KiB mean {} "
             "p95 {} max {}, copies {}, pages also uploaded the frame before {}; window: distinct "
             "pages {}, in half the frames or more {}",
             frames, pages / frames, Percentile(pages_per_frame, 50),
             Percentile(pages_per_frame, 95), Percentile(pages_per_frame, 100),
             bytes / frames / 1024, Percentile(bytes_per_frame, 95) / 1024,
             Percentile(bytes_per_frame, 100) / 1024, window_copies / frames,
             window_repeats / frames, window_pages.size(), steady);
    pages_per_frame.clear();
    bytes_per_frame.clear();
    window_pages.clear();
    window_copies = 0;
    window_repeats = 0;
}

} // namespace VideoCommon::UploadStats
