// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace Nx2Sampler {
// Writes a sampling report and folded stacks next to it. Throws on failure.
std::string Sample(const char* threads, double seconds, const std::string& path);

struct RipSample {
    std::uint32_t thread; ///< Index into RipRun::threads.
    std::uint64_t rip;
};

struct RipRun {
    std::vector<std::string> threads; ///< Sampled host threads, sorted by name.
    std::uint64_t passes{};           ///< Rounds over all threads.
    std::uint64_t missed{};           ///< Samples that could not be taken (suspend/context failed).
    double seconds{};                 ///< Wall time of the run.
};

// Samples only the instruction pointer of the named host threads, about every period_us
// microseconds (no stack walk), and hands the samples to `batch` about every 200 ms and at the end,
// from the calling thread. `stop` is polled between rounds. Throws on failure.
RipRun SampleRips(const char* threads, double seconds, std::uint32_t period_us,
                  const std::function<void(std::span<const RipSample>)>& batch,
                  const std::function<bool()>& stop);

// The file name of the host module (exe or DLL) containing address, or "" when none does.
std::string HostModuleName(std::uint64_t address);
} // namespace Nx2Sampler
