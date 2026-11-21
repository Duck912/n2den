// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <unordered_set>

#include "common/common_types.h"
#include "common/logging.h"

namespace Vulkan::Nvn {

/// Debugging: EDEN_NX2_NVN_SKIP_PROGRAMS=<hex hash>[,<hex hash>...] skips every draw and dispatch
/// that uses one of these program stages, to find which program hangs or breaks the GPU.
[[nodiscard]] inline bool SkipProgramForDebug(u64 hash, const char* what) {
    static const std::unordered_set<u64> skip = [] {
        std::unordered_set<u64> set;
        const char* p = std::getenv("EDEN_NX2_NVN_SKIP_PROGRAMS");
        while (p != nullptr && *p != '\0') {
            char* end = nullptr;
            const u64 value = std::strtoull(p, &end, 16);
            if (end == p) {
                break;
            }
            set.insert(value);
            p = *end == ',' ? end + 1 : end;
        }
        if (!set.empty()) {
            LOG_WARNING(Render_Vulkan, "[nvn] EDEN_NX2_NVN_SKIP_PROGRAMS: {} program(s) skipped", set.size());
        }
        return set;
    }();
    if (skip.empty() || hash == 0 || !skip.contains(hash)) {
        return false;
    }
    static std::atomic<u64> skipped{0};
    const u64 n = skipped.fetch_add(1, std::memory_order_relaxed);
    if (n < 8 || (n & (n - 1)) == 0) {
        LOG_INFO(Render_Vulkan, "[nvn] EDEN_NX2_NVN_SKIP_PROGRAMS: skipped {} using {:016x} (#{})", what,
                 hash, n + 1);
    }
    return true;
}

} // namespace Vulkan::Nvn
