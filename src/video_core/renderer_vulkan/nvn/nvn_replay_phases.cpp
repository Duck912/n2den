// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <string_view>

#include <fmt/format.h>

#include "common/logging.h"
#include "common/tl_ring.h"
#include "video_core/renderer_vulkan/nvn/nvn_replay_phases.h"

namespace Vulkan::Nvn::ReplayPhases {

namespace {

constexpr std::array<const char*, static_cast<size_t>(Phase::Count)> PHASE_NAMES{
    "record",       "op-draw",      "op-dispatch",  "op-clear",      "op-state",
    "op-recorder",  "pass-break",   "submit-early", "submit",        "resolve-pipeline",
    "build-desc",   "vertex-index", "outside-pass", "vtxcov",        "pipeline-bind",
    "finish-images", "texture-pool", "push-set0",   "vertex-bind",   "draw-call",
    "end-pass",     "begin-pass",   "flush-inline", "mirror-check",  "mirror-upload",
    "d-pre",        "d-uniform",    "d-bank",       "d-storage",     "d-image",
    "d-poolplan",   "o-report",     "o-images",     "o-ranges",      "o-shadow",
    "o-feedback",   "o-closed",     "o-open",       "o-writes",     "n-all",
    "n-detect",     "n-scan",       "c-resolve",    "c-fast",        "c-compute",
    "c-walk",       "r-resolve",    "r-pass",       "r-body",        "r-follows",
    "r-note",       "r-tail",       "p-prep",       "p-framebuffer", "pl-lookup",
    "c-break",      "c-record",     "w-draw",       "w-cond",        "w-rest",
};

constexpr std::array<const char*, static_cast<size_t>(Counter::Count)> COUNTER_NAMES{
    "draws",        "conditional",  "passes",        "framebuffers", "pipeline-reuses",
    "driver-banks", "handle-banks", "push-desc",     "desc-elided",  "pool-set-binds",
    "mirror-pages", "mirror-calls", "pipeline-memo-hits", "pipeline-lookups",
    "mirror-walked", "why-untracked", "why-texel", "why-first", "why-layout", "why-writes",
    "why-images", "why-ubo-addr", "why-ubo-copy", "why-storage", "why-driver-bank",
    "why-handle-bank", "why-other-buf", "same-layout-last",
};

constexpr auto LOG_INTERVAL = std::chrono::seconds{9}; // the heartbeat asks every 10 s

struct Global {
    std::mutex mutex;
    Accumulator total;
    Accumulator logged;
    u64 logged_presents{};
    u64 logged_tsc{};
    std::chrono::steady_clock::time_point logged_at{};
};

Global& GetGlobal() {
    static Global global;
    return global;
}

} // Anonymous namespace

bool Detail::ReadEnabled() noexcept {
    const char* const value = std::getenv("EDEN_NVN_REPLAY_PHASES");
    return value != nullptr && std::string_view{value} == "1";
}

Accumulator& Local() noexcept {
    thread_local Accumulator local;
    return local;
}

namespace {
/// EDEN_NX2_NVN_TLRING: ticks already flushed out of the local accumulator, and the totals the
/// last ring snapshot saw (the replay thread's own).
thread_local std::array<u64, static_cast<size_t>(Phase::Count)> ring_flushed{};
thread_local std::array<u64, static_cast<size_t>(Phase::Count)> ring_seen{};
} // namespace

void RingSnapshot(u64 seq) {
    if (!Enabled() || !Common::TlRing::Armed()) {
        return;
    }
    const Accumulator& local = Local();
    for (size_t i = 0; i < local.ticks.size(); ++i) {
        const u64 now = ring_flushed[i] + local.ticks[i];
        if (now != ring_seen[i]) {
            Common::TlRing::Note(Common::TlRing::PhaseDelta, i, now - ring_seen[i],
                                 static_cast<u32>(seq));
            ring_seen[i] = now;
        }
    }
}

void Flush() {
    if (!Enabled()) {
        return;
    }
    Accumulator& local = Local();
    Global& global = GetGlobal();
    {
        std::scoped_lock lock{global.mutex};
        for (size_t i = 0; i < local.ticks.size(); ++i) {
            ring_flushed[i] += local.ticks[i];
            global.total.ticks[i] += local.ticks[i];
            global.total.calls[i] += local.calls[i];
        }
        for (size_t i = 0; i < local.counts.size(); ++i) {
            global.total.counts[i] += local.counts[i];
        }
        for (size_t i = 0; i < local.op_ticks.size(); ++i) {
            global.total.op_ticks[i] += local.op_ticks[i];
            global.total.op_calls[i] += local.op_calls[i];
        }
    }
    local = {};
}

void Log(u64 presents) {
    if (!Enabled()) {
        return;
    }
    Global& global = GetGlobal();
    const auto now = std::chrono::steady_clock::now();
    const u64 tsc = Now();
    Accumulator before;
    Accumulator current;
    u64 before_presents = 0;
    double seconds = 0.0;
    double ticks_per_ms = 0.0;
    {
        std::scoped_lock lock{global.mutex};
        const bool first = global.logged_at == std::chrono::steady_clock::time_point{};
        if (!first && now - global.logged_at < LOG_INTERVAL) {
            return;
        }
        seconds = std::chrono::duration<double>(now - global.logged_at).count();
        ticks_per_ms = first ? 0.0
                             : static_cast<double>(tsc - global.logged_tsc) / (seconds * 1000.0);
        before = global.logged;
        before_presents = global.logged_presents;
        current = global.total;
        global.logged = current;
        global.logged_presents = presents;
        global.logged_tsc = tsc;
        global.logged_at = now;
        if (first) {
            return;
        }
    }
    if (presents <= before_presents || ticks_per_ms <= 0.0) {
        return;
    }
    const double frames = static_cast<double>(presents - before_presents);
    std::string text;
    for (size_t i = 0; i < current.ticks.size(); ++i) {
        const double ms = static_cast<double>(current.ticks[i] - before.ticks[i]) / ticks_per_ms;
        const double calls = static_cast<double>(current.calls[i] - before.calls[i]);
        text += fmt::format(" {} {:.3f} ({:.0f})", PHASE_NAMES[i], ms / frames, calls / frames);
    }
    // Recorder ops by op number (rec-opN), only those that ran.
    for (size_t i = 0; i < current.op_ticks.size(); ++i) {
        const double calls = static_cast<double>(current.op_calls[i] - before.op_calls[i]);
        if (calls > 0.0) {
            const double ms =
                static_cast<double>(current.op_ticks[i] - before.op_ticks[i]) / ticks_per_ms;
            text += fmt::format(" rec-op{} {:.3f} ({:.0f})", i, ms / frames, calls / frames);
        }
    }
    text += ";";
    for (size_t i = 0; i < current.counts.size(); ++i) {
        const double count = static_cast<double>(current.counts[i] - before.counts[i]);
        text += fmt::format(" {} {:.1f}", COUNTER_NAMES[i], count / frames);
    }
    LOG_INFO(Render_Vulkan,
             "[nvn] replay phases: {:.0f} presents in {:.1f} s, per frame ms (calls):{}", frames,
             seconds, text);
}

} // namespace Vulkan::Nvn::ReplayPhases
