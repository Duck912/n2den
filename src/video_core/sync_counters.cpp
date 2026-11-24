// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/sync_counters.h"

namespace VideoCommon::SyncCounters {

namespace {

bool ReadEnabled() {
    const char* const value = std::getenv("EDEN_NX2_SYNC_COUNTERS");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

constexpr u32 NUM_METHODS = 0x1000;

std::array<std::atomic<u64>, CauseCount> cause_counts{};
std::array<std::atomic<u32>, NUM_METHODS> same_writes{};
std::array<std::atomic<u32>, NUM_METHODS> all_writes{};

using CallerKey = std::array<u64, 4>;
std::mutex caller_mutex;
std::map<CallerKey, u64> callers;

std::mutex report_mutex;
std::chrono::steady_clock::time_point window_start{};

constexpr std::array<const char*, CauseCount> CAUSE_NAMES{
    "other",        "wfi",          "fragment barrier", "framebuffer", "submit",
    "deferred clear", "copy barrier", "buffer copy",     "inline upload", "buffer clear",
    "gather",       "scatter",      "compute",          "xfb pipeline",
};

u64 Take(std::atomic<u64>& counter) {
    return counter.exchange(0, std::memory_order_relaxed);
}

} // Anonymous namespace

extern const bool enabled = ReadEnabled();
std::atomic<u64> counters[CounterCount]{};

void NoteRenderPassEnd() {
    Add(RenderPassEnds);
    cause_counts[cause].fetch_add(1, std::memory_order_relaxed);
#ifdef _WIN32
    if (cause == CauseOther) {
        void* frames[6]{};
        const USHORT n = RtlCaptureStackBackTrace(1, 6, frames, nullptr);
        const u64 base = reinterpret_cast<u64>(GetModuleHandleW(nullptr));
        CallerKey key{};
        for (USHORT i = 0; i < n && i < key.size(); ++i) {
            key[i] = reinterpret_cast<u64>(frames[i]) - base + 0x140000000ULL;
        }
        std::scoped_lock lock{caller_mutex};
        ++callers[key];
    }
#endif
}

void NoteRegisterWrite(u32 method, bool same_value) {
    if (method >= NUM_METHODS) {
        return;
    }
    Add(RegWrites);
    all_writes[method].fetch_add(1, std::memory_order_relaxed);
    if (same_value) {
        Add(RegWritesSame);
        same_writes[method].fetch_add(1, std::memory_order_relaxed);
    }
}

void MaybeReport() {
    if (!enabled) {
        return;
    }
    std::unique_lock lock{report_mutex, std::try_to_lock};
    if (!lock) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (window_start.time_since_epoch().count() == 0) {
        window_start = now;
        return;
    }
    if (now - window_start < std::chrono::seconds{1}) {
        return;
    }
    const double seconds = std::chrono::duration<double>(now - window_start).count();
    window_start = now;

    std::array<u64, CounterCount> c{};
    for (u32 i = 0; i < CounterCount; ++i) {
        c[i] = Take(counters[i]);
    }
    const double frames = static_cast<double>(c[Frames]);
    // "rate/s (per frame)"
    const auto r = [&](u64 value) {
        const double per_s = static_cast<double>(value) / seconds;
        const double per_f = frames > 0 ? static_cast<double>(value) / frames : 0.0;
        return fmt::format("{:.0f}/s ({:.1f}/f)", per_s, per_f);
    };
    LOG_INFO(Render_Vulkan,
             "Sync counters ({:.2f} s, {:.1f} fps): wfi in pass {}, wfi outside {}, wfi methods {}, "
             "fragment barriers {}; render pass begins {}, ends {}; pipeline barriers {}, xfb "
             "barriers {}; submits {} (periodic {}, finishes {})",
             seconds, frames / seconds, r(c[WfiInPass]), r(c[WfiOutsidePass]), r(c[WfiMethods]),
             r(c[FragmentBarriers]), r(c[RenderPassBegins]), r(c[RenderPassEnds]), r(c[Barriers]),
             r(c[XfbBarriers]), r(c[Submits]), r(c[PeriodicFlushes]), r(c[Finishes]));
    LOG_INFO(Render_Vulkan,
             "Sync counters: uploads reordered {}, inline {} ({:.1f} KiB/f); other main-cmdbuf "
             "copies {} ({:.1f} KiB/f); gather {} copies in {} batches; scatter {} copies in {} "
             "batches",
             r(c[ReorderedUploads]), r(c[InlineUploads]),
             frames > 0 ? static_cast<double>(c[InlineUploadBytes]) / 1024.0 / frames : 0.0,
             r(c[OtherCopies]),
             frames > 0 ? static_cast<double>(c[OtherCopyBytes]) / 1024.0 / frames : 0.0,
             r(c[GatherCopies]), r(c[GatherBatches]), r(c[ScatterCopies]), r(c[ScatterBatches]));
    const u64 memo_total = c[SyncMemoHits] + c[SyncMemoMisses];
    const u64 mark_total = c[CpuMarkNoChange] + c[CpuMarkChanged];
    LOG_INFO(Render_Vulkan,
             "Sync counters CPU: sync memo hits {} misses {} ({:.1f}% hit); cpu-modified marks "
             "no-change {} changed {} ({:.1f}% no-change); 3D register writes {}, same value {} "
             "({:.1f}%)",
             r(c[SyncMemoHits]), r(c[SyncMemoMisses]),
             memo_total ? 100.0 * static_cast<double>(c[SyncMemoHits]) / static_cast<double>(memo_total) : 0.0,
             r(c[CpuMarkNoChange]), r(c[CpuMarkChanged]),
             mark_total ? 100.0 * static_cast<double>(c[CpuMarkNoChange]) / static_cast<double>(mark_total) : 0.0,
             r(c[RegWrites]), r(c[RegWritesSame]),
             c[RegWrites] ? 100.0 * static_cast<double>(c[RegWritesSame]) / static_cast<double>(c[RegWrites]) : 0.0);
    LOG_INFO(Render_Vulkan,
             "Sync counters invalidation: ranges {}; skipped texture {} query {} shader {}; sync "
             "memo misses by other range {}, with nothing to upload {}",
             r(c[InvalRanges]), r(c[InvalTextureSkips]), r(c[InvalQuerySkips]),
             r(c[InvalShaderSkips]), r(c[SyncMemoKeyMisses]), r(c[SyncMemoEmptyMisses]));

    std::string causes;
    for (u32 i = 0; i < CauseCount; ++i) {
        const u64 count = cause_counts[i].exchange(0, std::memory_order_relaxed);
        if (count != 0) {
            causes += fmt::format(" | {} {}", CAUSE_NAMES[i], r(count));
        }
    }
    LOG_INFO(Render_Vulkan, "Sync counters render pass ends by cause:{}", causes);

    std::vector<std::pair<u64, CallerKey>> top;
    {
        std::scoped_lock caller_lock{caller_mutex};
        for (const auto& [key, count] : callers) {
            top.emplace_back(count, key);
        }
        callers.clear();
    }
    std::ranges::sort(top, std::greater{});
    std::string text;
    for (size_t i = 0; i < top.size() && i < 8; ++i) {
        text += fmt::format(" | {}", r(top[i].first));
        for (const u64 frame : top[i].second) {
            text += fmt::format(" {:#x}", frame);
        }
    }
    if (!text.empty()) {
        LOG_INFO(Render_Vulkan, "Sync counters untagged render pass ends (callers):{}", text);
    }

    std::vector<std::pair<u32, u32>> regs;
    std::vector<std::pair<u32, u32>> totals;
    for (u32 method = 0; method < NUM_METHODS; ++method) {
        const u32 same = same_writes[method].exchange(0, std::memory_order_relaxed);
        const u32 all = all_writes[method].exchange(0, std::memory_order_relaxed);
        if (same != 0) {
            regs.emplace_back(same, method);
            totals.emplace_back(method, all);
        }
    }
    std::ranges::sort(regs, std::greater{});
    std::string reg_text;
    for (size_t i = 0; i < regs.size() && i < 10; ++i) {
        const u32 method = regs[i].second;
        const auto it = std::ranges::find_if(totals, [method](const auto& t) { return t.first == method; });
        reg_text += fmt::format(" | {:#05x} {} of {}", method, r(regs[i].first),
                                it != totals.end() ? r(it->second) : std::string{"?"});
    }
    if (!reg_text.empty()) {
        LOG_INFO(Render_Vulkan, "Sync counters same-value 3D register writes (top 10):{}", reg_text);
    }
}

} // namespace VideoCommon::SyncCounters
