// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "core/hle/kernel/k_hardware_timer_base.h"

namespace Core::Timing {
struct EventType;
} // namespace Core::Timing

namespace Kernel {

/// Statistics (EDEN_N2DEN_TIMER_STATS): the SVC the calling host thread is in, if any.
inline thread_local u32 t_stats_current_svc = 0xffff'ffff;

class KHardwareTimer : /* public KInterruptTask, */ public KHardwareTimerBase {
public:
    explicit KHardwareTimer(KernelCore& kernel) : KHardwareTimerBase{kernel} {}

    // Public API.
    void Initialize();
    void Finalize();

    s64 GetTick() const;

    void RegisterAbsoluteTask(KTimerTask* task, s64 task_time) {
        KScopedDisableDispatch dd{m_kernel};
        KScopedSpinLock lk{this->GetLock()};

        if (StatsEnabled()) {
            NoteRegister(task, task_time);
        }
        if (this->RegisterAbsoluteTaskImpl(task, task_time)) {
            if (task_time <= m_wakeup_time) {
                this->EnableInterrupt(task_time);
            }
        }
    }

    /// EDEN_N2DEN_TIMER_STATS: logs every 4 s which guest threads arm kernel timeouts, how long
    /// they are, and how many timer interrupts fire with nothing due.
    static bool StatsEnabled();
    /// Statistics: a SleepThread of requested ns that returned after actual ns (host time).
    static void NoteSleep(s64 requested, s64 actual);

private:
    void NoteRegister(KTimerTask* task, s64 task_time);

private:
    void EnableInterrupt(s64 wakeup_time);
    void DisableInterrupt();
    bool GetInterruptEnabled();
    void DoTask();

private:
    // Absolute time in nanoseconds
    s64 m_wakeup_time{(std::numeric_limits<s64>::max)()};
    std::shared_ptr<Core::Timing::EventType> m_event_type{};
};

} // namespace Kernel
