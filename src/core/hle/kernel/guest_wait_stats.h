// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <chrono>
#include <string_view>

#include "common/common_types.h"

namespace Kernel {
class KSynchronizationObject;
}

// What guest threads block on, and for how long.
// EDEN_GUEST_WAIT_STATS=1 times every blocking SVC (WaitSynchronization,
// WaitForAddress, WaitProcessWideKeyAtomic, ArbitrateLock, SleepThread) per guest thread and per
// cause, naming events after the service that created them, and logs every 4 s the threads that
// block least (the busy ones) with their causes. Off, each SVC pays one load and a branch.
namespace Kernel::GuestWaitStats {

[[nodiscard]] bool Enabled();

/// Names a service-created event (its readable half) for the report.
void NameObject(const void* object, std::string_view name);
void ForgetObject(const void* object);

enum class Kind : u8 {
    Sync,        ///< WaitSynchronization, signaled: value is the object that ended the wait
    SyncTimeout, ///< WaitSynchronization that timed out: value is its first object
    Arbiter,     ///< WaitForAddress: value is the address, sub the arbitration type
    CondVar,     ///< WaitProcessWideKeyAtomic: value is the condition variable key
    Mutex,       ///< ArbitrateLock: value is the mutex address
    Sleep,       ///< SleepThread: value is the requested time in microseconds
    Ipc,         ///< SendSyncRequest: value is the session (named after the service handling it)
};

/// Time an HLE service spent handling one request (the guest's IPC wait, from the service side).
void NoteServiceCommand(std::string_view service, u32 command, u64 ns);

/// Times one blocking call of the current guest thread; set the cause before it ends.
class Scope {
public:
    explicit Scope(u64 thread_id_);
    ~Scope();
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

    void Set(Kind kind_, u64 value_, u32 sub_ = 0) {
        kind = kind_;
        value = value_;
        sub = sub_;
    }
    /// For Sync and SyncTimeout: the object, kept by address and kernel type.
    void SetObject(Kind kind_, KSynchronizationObject* object, u32 count);

private:
    u64 thread_id;
    std::chrono::steady_clock::time_point start;
    bool armed;
    Kind kind{Kind::Sync};
    u32 sub{};
    u64 value{};
    const char* type_name{};
};

} // namespace Kernel::GuestWaitStats
