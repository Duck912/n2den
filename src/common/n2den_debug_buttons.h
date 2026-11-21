// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <atomic>
#include <chrono>
#include "common/common_types.h"

// A-only diagnostic pulse. Does not replace physical input or persist controller settings.
namespace Common::N2DenDebugButtons {
inline std::atomic<s64> a_until_ns{0};
inline s64 NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline void PulseA(u32 milliseconds) {
    a_until_ns.store(NowNs() + static_cast<s64>(milliseconds) * 1000000,
                     std::memory_order_relaxed);
}
inline void ReleaseA() { a_until_ns.store(0, std::memory_order_relaxed); }
inline u64 MergeA(u64 physical, bool player_one_or_handheld) {
    if (player_one_or_handheld && a_until_ns.load(std::memory_order_relaxed) > NowNs())
        return physical | u64{1};
    return physical;
}
} // namespace Common::N2DenDebugButtons
