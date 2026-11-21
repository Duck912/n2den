// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// EDEN_TC_STATS: every 4 s, logs how many images the texture cache created (and why), joined,
// uploaded from guest memory and deleted, plus the render targets created most often. For finding
// image churn (an image created, uploaded and freed again and again).

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"

namespace VideoCommon::TextureCacheStats {

inline bool Enabled() {
    static const bool enabled = std::getenv("EDEN_TC_STATS") != nullptr;
    return enabled;
}

struct Counters {
    u64 inserts{};          ///< Images created
    u64 inserts_rt{};       ///< ... for a render target (FindRenderTargetView)
    u64 joined_overlaps{};  ///< Overlapping images merged into a new one
    u64 aliased{};          ///< Overlaps kept as aliases
    u64 bad_overlaps{};
    u64 ignored_deleted{};  ///< Overlaps deleted while joining (same address and size, remapped)
    u64 uploads{};          ///< Images uploaded from guest memory
    u64 deferred_uploads{}; ///< New render targets whose upload was left to PrepareImage
    u64 upload_bytes{};
    u64 deletes{};
    u64 deletes_gc{};       ///< ... by the garbage collector
    // (format, width, height, gpu_addr) -> render target creations
    std::map<std::tuple<u32, u32, u32, u64>, u64> rt_keys;
    // (origin 0 texture / 1 render target, format, type, width, height, levels, layers) -> creations
    std::map<std::tuple<u32, u32, u32, u32, u32, u32, u32>, u64> kinds;
    std::map<std::tuple<u32, u32, u32, u32, u32, u32, u32>, u64> kind_bytes;
    std::map<u64, u64> addresses; // gpu_addr -> creations
    std::array<u64, 6> delete_age{}; // frames since creation: <2, <10, <60, <600, <3600, more
    u64 deleted_gpu_modified{};
    bool inserting_rt{};
    std::chrono::steady_clock::time_point window{std::chrono::steady_clock::now()};
};

inline Counters& Get() {
    static Counters counters; // texture cache runs on one thread (the GPU back thread)
    return counters;
}

inline void Report() {
    auto& c = Get();
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - c.window).count();
    if (seconds < 4.0) {
        return;
    }
    std::vector<std::pair<u64, std::tuple<u32, u32, u32, u64>>> top;
    for (const auto& [key, count] : c.rt_keys) {
        top.emplace_back(count, key);
    }
    std::ranges::sort(top, std::greater{});
    std::string rts;
    for (size_t i = 0; i < (std::min<size_t>)(top.size(), 8); ++i) {
        const auto& [format, width, height, gpu_addr] = top[i].second;
        rts += fmt::format("{}fmt {} {}x{} at {:#x} {:.1f}/s", i ? "; " : "", format, width, height,
                           gpu_addr, top[i].first / seconds);
    }
    const auto per_s = [seconds](u64 value) { return value / seconds; };
    std::vector<std::pair<u64, std::tuple<u32, u32, u32, u32, u32, u32, u32>>> top_kinds;
    for (const auto& [key, count] : c.kinds) {
        top_kinds.emplace_back(count, key);
    }
    std::ranges::sort(top_kinds, std::greater{});
    std::string kinds;
    for (size_t i = 0; i < (std::min<size_t>)(top_kinds.size(), 10); ++i) {
        const auto& [origin, format, type, width, height, levels, layers] = top_kinds[i].second;
        kinds += fmt::format("{}{} fmt {} type {} {}x{} L{} A{} {:.1f}/s {:.1f} MB/s", i ? "; " : "",
                             origin ? "RT" : "tex", format, type, width, height, levels, layers,
                             top_kinds[i].first / seconds, c.kind_bytes[top_kinds[i].second] / seconds / 1e6);
    }
    u64 repeated = 0;
    for (const auto& [address, count] : c.addresses) {
        repeated += count > 1 ? count : 0;
    }
    const auto& a = c.delete_age;
    LOG_INFO(HW_GPU,
             "Texture cache kinds/s: {} distinct addresses created, {:.0f}/s at an address created again "
             "within the window; deleted by age (frames) <2 {:.0f}, <10 {:.0f}, <60 {:.0f}, <600 {:.0f}, "
             "<3600 {:.0f}, more {:.0f}; deleted drawn-into {:.0f}; created most: {}",
             c.addresses.size(), per_s(repeated), per_s(a[0]), per_s(a[1]), per_s(a[2]), per_s(a[3]),
             per_s(a[4]), per_s(a[5]), per_s(c.deleted_gpu_modified), kinds.empty() ? std::string{"none"} : kinds);
    LOG_INFO(HW_GPU,
             "Texture cache/s: created {:.0f} (render targets {:.0f}), joined overlaps {:.0f}, aliased "
             "{:.0f}, bad overlaps {:.0f}, deleted while joining {:.0f}; uploads {:.0f} ({:.1f} MB), deferred {:.0f}; "
             "deleted {:.0f} (by cleanup {:.0f}); render targets created most: {}",
             per_s(c.inserts), per_s(c.inserts_rt), per_s(c.joined_overlaps), per_s(c.aliased),
             per_s(c.bad_overlaps), per_s(c.ignored_deleted), per_s(c.uploads),
             per_s(c.upload_bytes) / 1e6, per_s(c.deferred_uploads), per_s(c.deletes), per_s(c.deletes_gc),
             rts.empty() ? std::string{"none"} : rts);
    c = {};
    c.window = now;
}

} // namespace VideoCommon::TextureCacheStats
