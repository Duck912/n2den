// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "common/common_types.h"

/// One host thread of this process.
struct ThreadUsage {
    std::string name;   ///< As set by Common::SetCurrentThreadName; "UI" for the window's thread
    double percent = 0; ///< CPU time over the last sample, percent of one logical core
};

/// Host (PC) readings for the performance overlay. A field is empty when this machine or
/// platform cannot provide it.
struct SystemStats {
    std::string gpu_name;

    std::optional<double> cpu_process; ///< This process, percent of all logical cores
    std::optional<double> cpu_system;  ///< Whole machine, percent
    std::optional<double> cpu_kernel;  ///< Share of this process's CPU time spent in the OS kernel

    /// Every thread of this process seen in two consecutive samples
    std::vector<ThreadUsage> threads;

    std::optional<u64> ram_process; ///< Working set / resident size, bytes
    std::optional<u64> ram_used;    ///< Whole machine, bytes
    std::optional<u64> ram_total;

    std::optional<double> gpu_usage; ///< Busiest engine of the GPU, percent
    std::optional<double> gpu_temp_c;
    std::optional<u32> gpu_clock_mhz;
    std::optional<double> gpu_power_w;

    std::optional<u64> vram_process; ///< This process, bytes
    std::optional<u64> vram_used;    ///< Whole GPU, bytes
    std::optional<u64> vram_total;
};

/// Samples SystemStats once a second on a worker thread while active, so that slow OS queries
/// (performance counters, driver libraries) never run on the UI thread.
class SystemMonitor {
public:
    SystemMonitor();
    ~SystemMonitor();

    SystemMonitor(const SystemMonitor&) = delete;
    SystemMonitor& operator=(const SystemMonitor&) = delete;

    /// Sampling only runs while active (the overlay is visible).
    void SetActive(bool active);

    [[nodiscard]] SystemStats Get() const;

    struct Backend;

private:
    void Run(std::stop_token stop);

    mutable std::mutex stats_mutex;
    SystemStats stats;

    std::mutex wake_mutex;
    std::condition_variable_any wake;
    bool active = false;

    /// Native id of the thread that created the monitor (the UI thread)
    u64 ui_thread_id = 0;

    std::jthread thread;
};
