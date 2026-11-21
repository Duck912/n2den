// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <deque>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include "common/common_types.h"

namespace Common::Log {
struct RingLine {
    u64 sequence;
    std::string text;
};
namespace Ring {
inline std::atomic<size_t> capacity{};
inline std::mutex mutex;
inline std::deque<RingLine> lines;
inline u64 sequence{};
} // namespace Ring

inline void SetRingCapacity(size_t capacity) {
    std::scoped_lock lock{Ring::mutex};
    Ring::capacity.store(capacity, std::memory_order_relaxed);
    while (Ring::lines.size() > capacity)
        Ring::lines.pop_front();
}
inline void AppendRing(std::string text) {
    if (!Ring::capacity.load(std::memory_order_relaxed))
        return;
    std::scoped_lock lock{Ring::mutex};
    const size_t capacity = Ring::capacity.load(std::memory_order_relaxed);
    if (!capacity)
        return;
    Ring::lines.push_back({++Ring::sequence, std::move(text)});
    while (Ring::lines.size() > capacity)
        Ring::lines.pop_front();
}
inline std::vector<RingLine> ReadRing(u64 after) {
    std::scoped_lock lock{Ring::mutex};
    std::vector<RingLine> result;
    for (const auto& line : Ring::lines) {
        if (line.sequence > after)
            result.push_back(line);
    }
    return result;
}
} // namespace Common::Log
