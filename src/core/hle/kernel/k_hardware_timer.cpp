// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "common/logging.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/hle/kernel/k_hardware_timer.h"
#include "core/hle/kernel/k_scheduler.h"

namespace Kernel {

namespace {
struct TimerStats {
    std::mutex mutex;
    // (svc, thread id, priority, timeout bucket) -> registrations
    std::map<std::tuple<u32, u64, s32, u32>, u64> registrations;
    std::array<u64, 9> buckets{}; // <=0 <50us <100 <250 <500 <1ms <2 <8 >=8ms
    u64 total{};
    u64 fires{};
    u64 empty_fires{};
    u64 tasks_run{};
    std::map<s64, u64> sleep_requested;   // exact requested ns (short sleeps) -> count
    std::array<u64, 9> sleep_actual{};    // actual duration of short (< 50 us) sleeps, same buckets
    u64 sleep_actual_sum{};
    u64 short_sleeps{};
    std::chrono::steady_clock::time_point window{std::chrono::steady_clock::now()};
};
TimerStats& Stats() {
    static TimerStats stats;
    return stats;
}
constexpr std::array<const char*, 9> BucketNames{"<=0", "<50us", "<100us", "<250us", "<500us",
                                                 "<1ms", "<2ms", "<8ms", ">=8ms"};
u32 Bucket(s64 ns) {
    static constexpr std::array<s64, 8> edges{1, 50'000, 100'000, 250'000, 500'000, 1'000'000, 2'000'000, 8'000'000};
    u32 bucket = 0;
    while (bucket < edges.size() && ns >= edges[bucket]) {
        ++bucket;
    }
    return bucket;
}
void MaybeReport(TimerStats& stats) {
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - stats.window).count();
    if (seconds < 4.0) {
        return;
    }
    std::string buckets;
    for (size_t i = 0; i < stats.buckets.size(); ++i) {
        buckets += fmt::format("{}{} {:.0f}", i ? ", " : "", BucketNames[i], stats.buckets[i] / seconds);
    }
    std::vector<std::pair<u64, std::tuple<u32, u64, s32, u32>>> top;
    for (const auto& [key, count] : stats.registrations) {
        top.emplace_back(count, key);
    }
    std::ranges::sort(top, std::greater{});
    std::string threads;
    for (size_t i = 0; i < std::min<size_t>(top.size(), 12); ++i) {
        const auto& [svc, tid, priority, bucket] = top[i].second;
        threads += fmt::format("{}svc {:#x} thread {} prio {} {} {:.0f}/s", i ? "; " : "", svc, tid,
                               priority, BucketNames[bucket], top[i].first / seconds);
    }
    LOG_INFO(Kernel, "Kernel timeouts/s: armed {:.0f} ({}); timer fires {:.0f}, with nothing due {:.0f}, "
                     "timeouts expired {:.0f}; top: {}",
             stats.total / seconds, buckets, stats.fires / seconds, stats.empty_fires / seconds,
             stats.tasks_run / seconds, threads);
    std::string requested;
    {
        std::vector<std::pair<u64, s64>> top_req;
        for (const auto& [ns, count] : stats.sleep_requested) {
            top_req.emplace_back(count, ns);
        }
        std::ranges::sort(top_req, std::greater{});
        for (size_t i = 0; i < std::min<size_t>(top_req.size(), 6); ++i) {
            requested += fmt::format("{}{} ns {:.0f}/s", i ? ", " : "", top_req[i].second, top_req[i].first / seconds);
        }
    }
    std::string actual;
    for (size_t i = 1; i < stats.sleep_actual.size(); ++i) {
        actual += fmt::format("{}{} {:.0f}", i > 1 ? ", " : "", BucketNames[i], stats.sleep_actual[i] / seconds);
    }
    LOG_INFO(Kernel, "Short sleeps (< 50 us requested): {:.0f}/s, requested: {}; actual (host): avg {:.1f} us, {}",
             stats.short_sleeps / seconds, requested,
             stats.short_sleeps ? stats.sleep_actual_sum / 1000.0 / stats.short_sleeps : 0.0, actual);
    stats.sleep_requested.clear();
    stats.sleep_actual = {};
    stats.sleep_actual_sum = stats.short_sleeps = 0;
    stats.registrations.clear();
    stats.buckets = {};
    stats.total = stats.fires = stats.empty_fires = stats.tasks_run = 0;
    stats.window = now;
}
} // namespace

bool KHardwareTimer::StatsEnabled() {
    static const bool enabled = std::getenv("EDEN_N2DEN_TIMER_STATS") != nullptr;
    return enabled;
}

void KHardwareTimer::NoteSleep(s64 requested, s64 actual) {
    if (requested >= 50'000) {
        return;
    }
    auto& stats = Stats();
    std::scoped_lock lock{stats.mutex};
    ++stats.sleep_requested[requested];
    ++stats.sleep_actual[Bucket(actual)];
    stats.sleep_actual_sum += static_cast<u64>(std::max<s64>(actual, 0));
    ++stats.short_sleeps;
}

void KHardwareTimer::NoteRegister(KTimerTask* task, s64 task_time) {
    auto* const thread = static_cast<KThread*>(task); // KThread is the only timer task
    const s64 relative = task_time - GetTick();
    auto& stats = Stats();
    std::scoped_lock lock{stats.mutex};
    const u32 bucket = Bucket(relative);
    ++stats.buckets[bucket];
    ++stats.total;
    ++stats.registrations[{t_stats_current_svc, thread->GetThreadId(), thread->GetPriority(), bucket}];
    MaybeReport(stats);
}

void KHardwareTimer::Initialize() {
    // Create the timing callback to register with CoreTiming.
    m_event_type = Core::Timing::CreateEvent("KHardwareTimer::Callback",
                                             [this](s64, std::chrono::nanoseconds) {
                                                 this->DoTask();
                                                 return std::nullopt;
                                             });
}

void KHardwareTimer::Finalize() {
    m_kernel.System().CoreTiming().UnscheduleEvent(m_event_type);
    m_wakeup_time = (std::numeric_limits<s64>::max)();
    m_event_type.reset();
}

void KHardwareTimer::DoTask() {
    // Handle the interrupt.
    {
        KScopedSchedulerLock slk{m_kernel};
        KScopedSpinLock lk(this->GetLock());

        //! Ignore this event if needed.
        if (!this->GetInterruptEnabled()) {
            return;
        }

        // Disable the timer interrupt while we handle this.
        // Not necessary due to core timing already having popped this event to call it.
        // this->DisableInterrupt();
        m_wakeup_time = (std::numeric_limits<s64>::max)();

        const s64 next_time = this->DoInterruptTaskImpl(GetTick());
        if (StatsEnabled()) {
            auto& stats = Stats();
            std::scoped_lock lock{stats.mutex};
            ++stats.fires;
            stats.tasks_run += last_tasks_run;
            if (last_tasks_run == 0) {
                ++stats.empty_fires;
            }
        }
        if (0 < next_time && next_time <= m_wakeup_time) {
            // We have a next time, so we should set the time to interrupt and turn the interrupt
            // on.
            this->EnableInterrupt(next_time);
        }
    }

    // Clear the timer interrupt.
    // Kernel::GetInterruptManager().ClearInterrupt(KInterruptName_NonSecurePhysicalTimer,
    //                                              GetCurrentCoreId());
}

void KHardwareTimer::EnableInterrupt(s64 wakeup_time) {
    this->DisableInterrupt();

    m_wakeup_time = wakeup_time;
    m_kernel.System().CoreTiming().ScheduleEvent(std::chrono::nanoseconds{m_wakeup_time},
                                                 m_event_type, true);
}

void KHardwareTimer::DisableInterrupt() {
    m_kernel.System().CoreTiming().UnscheduleEvent(m_event_type,
                                                   Core::Timing::UnscheduleEventType::NoWait);
    m_wakeup_time = (std::numeric_limits<s64>::max)();
}

s64 KHardwareTimer::GetTick() const {
    return m_kernel.System().CoreTiming().GetGlobalTimeNs().count();
}

bool KHardwareTimer::GetInterruptEnabled() {
    return m_wakeup_time != (std::numeric_limits<s64>::max)();
}

} // namespace Kernel
