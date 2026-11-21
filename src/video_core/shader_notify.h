// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>

namespace VideoCore {
class ShaderNotify {
public:
    [[nodiscard]] int ShadersBuilding() noexcept;

    void MarkShaderComplete() noexcept {
        ++num_complete;
    }

    void MarkShaderBuilding() noexcept {
        ++num_building;
    }

    /// Shaders queued or compiling right now.
    [[nodiscard]] int ShadersPending() const noexcept {
        return num_building.load(std::memory_order::relaxed) -
               num_complete.load(std::memory_order::relaxed);
    }

    /// Builds that hold the GPU thread until they finish: a compute pipeline compiling on a
    /// worker, which every dispatch of its program waits for.
    void MarkBlockingBuilding() noexcept {
        ++num_blocking;
    }

    void MarkBlockingComplete() noexcept {
        --num_blocking;
    }

    [[nodiscard]] int BlockingPending() const noexcept {
        return num_blocking.load(std::memory_order::relaxed);
    }

private:
    std::atomic_int num_blocking{};
    std::atomic_int num_building{};
    std::atomic_int num_complete{};
    int report_base{};

    bool completed{};
    int num_when_completed{};
    std::chrono::steady_clock::time_point complete_time;
};
} // namespace VideoCore
