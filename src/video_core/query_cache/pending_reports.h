// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include <atomic>
#include <cstddef>

#include "common/common_types.h"

// Query reports registered in a query cache and not yet written to guest memory, counted per
// bucket of 16-byte device address granules. The
// GPU front reads a report slot without draining the split only while its buckets count none.
// Increments happen under the query cache's lock when a report is registered; decrements after
// its value was written, when it is unregistered. Collisions only make a bucket look busy.
namespace VideoCommon::PendingReports {

inline constexpr size_t NumBuckets = 4096;
inline std::array<std::atomic<u32>, NumBuckets> counts{};

[[nodiscard]] inline size_t Bucket(u64 device_address) {
    const u64 granule = device_address >> 4;
    return static_cast<size_t>((granule ^ (granule >> 12) ^ (granule >> 24)) % NumBuckets);
}

/// A report covers up to 16 bytes from its address (count and timestamp): every granule it
/// touches is counted.
inline void Add(u64 device_address) {
    counts[Bucket(device_address)].fetch_add(1, std::memory_order_release);
    if (((device_address + 15) >> 4) != (device_address >> 4)) {
        counts[Bucket(device_address + 15)].fetch_add(1, std::memory_order_release);
    }
}

inline void Remove(u64 device_address) {
    counts[Bucket(device_address)].fetch_sub(1, std::memory_order_release);
    if (((device_address + 15) >> 4) != (device_address >> 4)) {
        counts[Bucket(device_address + 15)].fetch_sub(1, std::memory_order_release);
    }
}

/// Whether a report anywhere in [device_address, device_address + size) may still be pending.
[[nodiscard]] inline bool Any(u64 device_address, u64 size) {
    for (u64 granule = device_address & ~u64{15}; granule < device_address + size; granule += 16) {
        if (counts[Bucket(granule)].load(std::memory_order_acquire) != 0) {
            return true;
        }
    }
    return false;
}

} // namespace VideoCommon::PendingReports
