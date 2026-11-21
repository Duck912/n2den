// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>

#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/debugger/nx2_debug_tools.h"
#include "core/memory.h"
#include "core/debugger/debugger.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/kernel/physical_core.h"
#include "core/hle/nvn/bridge.h"
#include "core/hle/nvn/thunk_encoding.h"
#include "core/hle/kernel/svc.h"

namespace Kernel {

namespace {

/// The supervisor call path of RunThread while sched_trace records spans: the same calls, each
/// recorded as a span on the core it ends on (a call may block and resume on another core).
void TracedSupervisorCall(KernelCore& kernel, KThread* thread, Core::ArmInterface& interface,
                          u32 svc) {
    namespace Trace = Core::Nx2Debug::SchedTrace;
    auto& timing = kernel.System().CoreTiming();
    const u32 thread_id = static_cast<u32>(thread->GetThreadId());
    if (svc >= Core::NvnHle::Thunk::SvcBase) {
        const s64 start = static_cast<s64>(timing.GetClockTicks());
        const bool handled = Core::NvnHle::Bridge::Handle(kernel.System(), thread, interface, svc);
        if (handled) {
            Trace::RecordSpan(static_cast<s32>(kernel.CurrentPhysicalCoreIndex()),
                              {start, static_cast<s64>(timing.GetClockTicks()), 0, 0, thread_id,
                               Trace::SpanKind::Hle,
                               static_cast<u16>(svc - Core::NvnHle::Thunk::SvcBase)});
            return;
        }
    }
    std::array<u64, 8> args{};
    interface.GetSvcArguments(args);
    u64 arg0 = args[0];
    u64 arg1 = args[1];
    if (svc == 0x18) {
        // WaitSynchronization(out index, handles, count, timeout): name the first handle.
        arg0 = args[2] > 0 ? thread->GetOwnerProcess()->GetMemory().Read32(
                                 Common::ProcessAddress{args[1]})
                           : 0;
        arg1 = args[2];
    }
    const s64 start = static_cast<s64>(timing.GetClockTicks());
    Svc::Call(kernel.System(), svc);
    Trace::RecordSpan(static_cast<s32>(kernel.CurrentPhysicalCoreIndex()),
                      {start, static_cast<s64>(timing.GetClockTicks()), arg0, arg1, thread_id,
                       Trace::SpanKind::Svc, static_cast<u16>(svc)});
}

} // Anonymous namespace

PhysicalCore::PhysicalCore(KernelCore& kernel, std::size_t core_index)
    : m_core_index{core_index}
{
    m_is_single_core = !kernel.IsMulticore();
}
PhysicalCore::~PhysicalCore() = default;

void PhysicalCore::RunThread(KernelCore& kernel, Kernel::KThread* thread) {
    auto* process = thread->GetOwnerProcess();
    auto& system = kernel.System();
    auto* interface = process->GetArmInterface(m_core_index);

    interface->Initialize();

    const auto EnterContext = [&]() {
        // Lock the core context.
        std::scoped_lock lk{m_guard};

        // Check if we are already interrupted. If we are, we can just stop immediately.
        if (m_is_interrupted) {
            return false;
        }

        // Mark that we are running.
        m_arm_interface = interface;
        m_current_thread = thread;

        // Acquire the lock on the thread parameters.
        // This allows us to force synchronization with Interrupt.
        interface->LockThread(thread);

        return true;
    };

    const auto ExitContext = [&]() {
        // Save the JIT context back to the thread so the debugger can
        // read the current register state. Debug halt paths (step,
        // breakpoint, watchpoint) return from RunThread without going
        // through the scheduler's Unload/SaveContext.
        if (system.DebuggerEnabled()) {
            interface->GetContext(thread->GetContext());
        }

        // Unlock the thread.
        interface->UnlockThread(thread);

        // Lock the core context.
        std::scoped_lock lk{m_guard};

        // On exit, we no longer are running.
        m_arm_interface = nullptr;
        m_current_thread = nullptr;
    };

    while (true) {
        // If the thread is scheduled for termination, exit.
        if (thread->HasDpc() && thread->IsTerminationRequested()) {
            thread->Exit(kernel);
        }

        // Notify the debugger and go to sleep if a step was performed
        // and this thread has been scheduled again.
        if (thread->GetStepState() == StepState::StepPerformed) {
            system.GetDebugger().NotifyThreadStopped(thread);
            thread->RequestSuspend(kernel, SuspendType::Debug);
            return;
        }

        // Otherwise, run the thread.
        Core::HaltReason hr{};
        {
            // If we were interrupted, exit immediately.
            if (!EnterContext()) {
                return;
            }

            if (thread->GetStepState() == StepState::StepPending) {
                hr = interface->StepThread(thread);

                if (True(hr & Core::HaltReason::StepThread)) {
                    thread->SetStepState(StepState::StepPerformed);
                }
            } else if ((Core::Nx2Debug::SchedTrace::spans_enabled.load(
                           std::memory_order_relaxed) & 1u) != 0) [[unlikely]] {
                const s64 start = static_cast<s64>(system.CoreTiming().GetClockTicks());
                hr = interface->RunThread(thread);
                Core::Nx2Debug::SchedTrace::RecordSpan(
                    static_cast<s32>(m_core_index),
                    {start, static_cast<s64>(system.CoreTiming().GetClockTicks()), 0, 0,
                     static_cast<u32>(thread->GetThreadId()),
                     Core::Nx2Debug::SchedTrace::SpanKind::Jit, 0});
            } else {
                hr = interface->RunThread(thread);
            }

            ExitContext();
        }

        // Determine why we stopped.
        // If a step completed successfully, skip other halt reason handlers
        // the step takes priority (e.g. step may also set InstructionBreakpoint
        // if the next instruction happens to be a breakpoint).
        const bool step_completed = True(hr & Core::HaltReason::StepThread)
                                    && thread->GetStepState() == StepState::StepPerformed;
        const bool supervisor_call = !step_completed && True(hr & Core::HaltReason::SupervisorCall);
        const bool prefetch_abort = !step_completed && True(hr & Core::HaltReason::PrefetchAbort);
        const bool breakpoint = !step_completed && True(hr & Core::HaltReason::InstructionBreakpoint);
        const bool data_abort = !step_completed && True(hr & Core::HaltReason::DataAbort);
        const bool interrupt = !step_completed && True(hr & Core::HaltReason::BreakLoop);

        // Since scheduling may occur here, we cannot use any cached
        // state after returning from calls we make.

        // Notify the debugger and go to sleep if a breakpoint was hit,
        // or if the thread is unable to continue for any reason.
        if (breakpoint || prefetch_abort) {
            if (breakpoint) {
                interface->RewindBreakpointInstruction();
                // RewindBreakpointInstruction sets the JIT state to the
                // saved breakpoint context. Update the thread context to
                // match, since ExitContext already saved the post-execution
                // state.
                interface->GetContext(thread->GetContext());
            }
            if (system.DebuggerEnabled()) {
                system.GetDebugger().NotifyThreadStopped(thread);
            } else {
                interface->LogBacktrace(process);
            }
            thread->RequestSuspend(kernel, SuspendType::Debug);
            return;
        }

        // Notify the debugger and go to sleep on data abort.
        if (data_abort) {
            if (system.DebuggerEnabled()) {
                system.GetDebugger().NotifyThreadWatchpoint(thread, *interface->HaltedWatchpoint());
            }
            thread->RequestSuspend(kernel, SuspendType::Debug);
            return;
        }

        // Handle system calls.
        if (supervisor_call) {
            // Perform call.
            const u32 svc = interface->GetSvcNumber();
            if ((Core::Nx2Debug::SchedTrace::spans_enabled.load(std::memory_order_relaxed) &
                 6u) != 0)
                [[unlikely]] {
                TracedSupervisorCall(kernel, thread, *interface, svc);
                return;
            }
            // nvn-hle's bridge thunks (SVC 0x8000 and up) are handled on the host; anything else
            // is a kernel SVC.
            if (svc >= Core::NvnHle::Thunk::SvcBase) {
                if (Core::NvnHle::Bridge::Handle(system, thread, *interface, svc)) {
                    return;
                }
            }
            Svc::Call(system, svc);
            return;
        }

        // Handle external interrupt sources.
        if (interrupt || m_is_single_core) {
            return;
        }
    }
}

void PhysicalCore::LoadContext(const KThread* thread) {
    auto* const process = thread->GetOwnerProcess();
    if (!process) {
        // Kernel threads do not run on emulated CPU cores.
        return;
    }

    auto* interface = process->GetArmInterface(m_core_index);
    if (interface) {
        interface->SetContext(thread->GetContext());
        interface->SetTpidrroEl0(GetInteger(thread->GetTlsAddress()));
        interface->SetWatchpointArray(&process->GetWatchpoints());
    }
}

void PhysicalCore::LoadSvcArguments(const KProcess& process, std::span<const uint64_t, 8> args) {
    process.GetArmInterface(m_core_index)->SetSvcArguments(args);
}

void PhysicalCore::SaveContext(KThread* thread) const {
    auto* const process = thread->GetOwnerProcess();
    if (!process) {
        // Kernel threads do not run on emulated CPU cores.
        return;
    }

    auto* interface = process->GetArmInterface(m_core_index);
    if (interface) {
        interface->GetContext(thread->GetContext());
    }
}

void PhysicalCore::SaveSvcArguments(KProcess& process, std::span<uint64_t, 8> args) const {
    process.GetArmInterface(m_core_index)->GetSvcArguments(args);
}

void PhysicalCore::CloneFpuStatus(KThread* dst) const {
    auto* process = dst->GetOwnerProcess();

    Svc::ThreadContext ctx{};
    process->GetArmInterface(m_core_index)->GetContext(ctx);

    dst->GetContext().fpcr = ctx.fpcr;
    dst->GetContext().fpsr = ctx.fpsr;
}

void PhysicalCore::LogBacktrace(KernelCore& kernel) {
    auto* process = GetCurrentProcessPointer(kernel);
    if (!process) {
        return;
    }

    auto* interface = process->GetArmInterface(m_core_index);
    if (interface) {
        interface->LogBacktrace(process);
    }
}

void PhysicalCore::Idle() {
    std::unique_lock lk{m_guard};
    m_on_interrupt.wait(lk, [this] { return m_is_interrupted; });
}

bool PhysicalCore::GetLiveContextForDebugging(Svc::ThreadContext& ctx) const {
    if (m_arm_interface == nullptr) {
        return false;
    }
    m_arm_interface->GetContext(ctx);
    return true;
}

bool PhysicalCore::IsInterrupted() const {
    return m_is_interrupted;
}

void PhysicalCore::Interrupt() {
    // Lock core context.
    std::scoped_lock lk{m_guard};

    // Load members.
    auto* arm_interface = m_arm_interface;
    auto* thread = m_current_thread;

    // Add interrupt flag.
    m_is_interrupted = true;

    // Interrupt ourselves.
    m_on_interrupt.notify_one();

    // If there is no thread running, we are done.
    if (arm_interface == nullptr) {
        return;
    }

    // Interrupt the CPU.
    arm_interface->SignalInterrupt(thread);
}

void PhysicalCore::ClearInterrupt() {
    std::scoped_lock lk{m_guard};
    m_is_interrupted = false;
}

} // namespace Kernel
