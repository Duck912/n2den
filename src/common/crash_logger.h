// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>

namespace Common::CrashLogger {

/// What EDEN_CRASH_TEST asked for. Only meant for checking the crash logger itself.
enum class TestMode {
    None,
    Worker, ///< EDEN_CRASH_TEST=av: a worker thread dereferences null a few seconds after Install.
    Main,   ///< EDEN_CRASH_TEST=av-main: the frontend calls TriggerTestCrash on its main thread.
};

#ifdef _WIN32

/// Installs the crash logger: when the process is about to die from an exception nobody handled
/// (or from abort()), a report is written to <log dir>/crash-<timestamp>.txt. Nothing is paid
/// until a crash. Safe to call more than once. Call early in main, after the single-instance and
/// startup checks. EDEN_CRASH_DUMP=1 additionally writes a small minidump beside the report.
void Install();

/// Puts the unhandled-exception filter back if a third-party DLL (a graphics driver, say)
/// replaced it after Install. Cheap; call after such DLLs have loaded.
void Reassert();

/// Tells the report that [base, base + size) is emulator guest memory. name must be a string
/// literal. Reports say whether a faulting address lies in a registered region.
void RegisterRegion(const char* name, const void* base, std::size_t size);
void UnregisterRegion(const void* base);

TestMode GetTestMode();

/// Dereferences null on the calling thread (for TestMode::Main).
[[noreturn]] void TriggerTestCrash();

#else

inline void Install() {}
inline void Reassert() {}
inline void RegisterRegion(const char*, const void*, std::size_t) {}
inline void UnregisterRegion(const void*) {}
inline TestMode GetTestMode() {
    return TestMode::None;
}
[[noreturn]] inline void TriggerTestCrash() {
    __builtin_trap();
}

#endif

} // namespace Common::CrashLogger
