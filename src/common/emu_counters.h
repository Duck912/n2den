// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"

/// Always-on event totals for the performance overlay. Writers add with relaxed ordering; a
/// reader takes the difference of two readings. Each total has its own cache line, since they are
/// written from different threads.
namespace Common::EmuCounters {

struct alignas(64) Counter {
    std::atomic<u64> value{};

    void Add(u64 n = 1) noexcept {
        value.fetch_add(n, std::memory_order_relaxed);
    }
    [[nodiscard]] u64 Get() const noexcept {
        return value.load(std::memory_order_relaxed);
    }
};

/// Guest code blocks compiled by the JIT
inline Counter jit_blocks;
/// Requests to throw away compiled guest code (self-modifying or reloaded code)
inline Counter jit_invalidations;
/// Draw calls issued by the guest
inline Counter draws;
/// Instances in the direct draws (Draw: the instance count it is given; indirect draws not included)
inline Counter draw_instances;
/// Draws from the per-instance methods (index_buffer*_first/subsequent, vertex_array_instance_*): DrawManager's
/// DrawIndexSmall and DrawArrayInstanced, one ProcessDraw each
inline Counter draws_instance_methods;
/// Draws by kind, for the native NVN hook's accounting: task/mesh pipelines, indirect, inline-index
inline Counter draws_mesh;
inline Counter draws_indirect;
inline Counter draws_inline_index;
/// Nanoseconds the core timing thread spent running events. The rest of its CPU time is its
/// precise wait before the next event, which the OS counts as busy.
inline Counter timing_work_ns;

} // namespace Common::EmuCounters
