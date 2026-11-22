// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

namespace Common {

/// Thrown by a fatal assert (UNREACHABLE) on a thread that holds a ScopedFatalAssertThrows.
struct FatalAssertError {};

/// While alive, a fatal assert on this thread throws FatalAssertError instead of aborting the process.
/// For speculative work whose failure costs nothing but the work itself -- the boot-time JIT code scan
/// translates whole code windows, data included, and one odd word must not end the emulator.
/// Only code between the scope and its catch may throw through: no noexcept frames, no held locks.
class ScopedFatalAssertThrows {
public:
    ScopedFatalAssertThrows();
    ~ScopedFatalAssertThrows();
    ScopedFatalAssertThrows(const ScopedFatalAssertThrows&) = delete;
    ScopedFatalAssertThrows& operator=(const ScopedFatalAssertThrows&) = delete;
};

} // namespace Common
