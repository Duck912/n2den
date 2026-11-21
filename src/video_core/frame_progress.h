// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <mutex>

#include "common/common_types.h"
#include "common/n2den_trace.h"

// How far the guest and GPU emulation are in frames. The guest has presented
// Common::N2denTrace::present_count frames, so it prepares frame present_count + 1. Each present
// records its frame's fence; GPU emulation has finished a frame once the back has processed the
// increment that reaches that fence (the guest syncpoint value, which the back advances as it
// runs the command stream). With the present running a frame ahead, the back can still be
// drawing frame N while the guest writes frame N + 1's data.
namespace VideoCommon::FrameProgress {

namespace Detail {
struct PresentFence {
    u32 frame{};
    u32 id{~0U};
    u32 value{};
};
inline std::mutex mutex;
inline std::array<PresentFence, 64> fences{};
inline u32 next_fence{};
inline std::function<u32(u32)> syncpoint_value; ///< set by the fence manager
inline std::atomic<u32> finished{0};

/// Under the mutex: raises `finished` to the newest recorded frame whose fence was reached.
inline void Update() {
    if (!syncpoint_value) {
        return;
    }
    u32 newest = finished.load(std::memory_order_relaxed);
    for (const PresentFence& fence : fences) {
        if (fence.id < 192 && fence.frame > newest && syncpoint_value(fence.id) >= fence.value) {
            newest = fence.frame;
        }
    }
    finished.store(newest, std::memory_order_release);
}
} // namespace Detail

/// The fence manager's syncpoint values as the back advances them.
inline void SetSyncpointValue(std::function<u32(u32)>&& value) {
    std::scoped_lock lock{Detail::mutex};
    Detail::syncpoint_value = std::move(value);
}

/// Guest side, at a present: frame `frame` ends when syncpoint `id` reaches `value`.
inline void NotePresentFence(u32 frame, u32 id, u32 value) {
    std::scoped_lock lock{Detail::mutex};
    Detail::fences[Detail::next_fence] = Detail::PresentFence{frame, id, value};
    Detail::next_fence = (Detail::next_fence + 1) % Detail::fences.size();
    Detail::Update();
}

/// Back side, after it incremented syncpoint `id`.
inline void NoteSyncpoint(u32 id) {
    std::scoped_lock lock{Detail::mutex};
    for (const Detail::PresentFence& fence : Detail::fences) {
        if (fence.id == id) {
            Detail::Update();
            return;
        }
    }
}

/// Frames GPU emulation has finished: the back is on frame Finished() + 1.
[[nodiscard]] inline u32 Finished() {
    return Detail::finished.load(std::memory_order_acquire);
}

/// Frames the guest has presented: it is preparing frame GuestPresents() + 1.
[[nodiscard]] inline u32 GuestPresents() {
    return Common::N2denTrace::present_count.load(std::memory_order_acquire);
}

} // namespace VideoCommon::FrameProgress
