// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

// The rest of the code base gets this from src/CMakeLists.txt, which dynarmic is added before.
#ifndef _CRT_SECURE_NO_WARNINGS
#    define _CRT_SECURE_NO_WARNINGS
#endif

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>
#include <fmt/format.h>

#include "dynarmic/backend/x64/perf_map.h"
#include "dynarmic/interface/A64/a64.h"
#include "common/common_types.h"
#if defined(__linux__) && !defined(__ANDROID__)
#    include <cstdio>
#    include <cstdlib>
#    include <mutex>
#    include <sys/types.h>
#    include <unistd.h>

namespace Dynarmic::Backend::X64 {

namespace {
std::mutex mutex;
std::FILE* file = nullptr;

void OpenFile() {
    const char* perf_dir = std::getenv("PERF_BUILDID_DIR");
    if (!perf_dir) {
        file = nullptr;
        return;
    }

    const pid_t pid = getpid();
    const std::string filename = fmt::format("{:s}/perf-{:d}.map", perf_dir, pid);

    file = std::fopen(filename.c_str(), "w");
    if (!file) {
        return;
    }

    std::setvbuf(file, nullptr, _IONBF, 0);
}
}  // anonymous namespace

namespace detail {
void PerfMapRegister(const void* start, const void* end, std::string_view friendly_name) {
    if (start == end) {
        // Nothing to register
        return;
    }

    std::lock_guard guard{mutex};

    if (!file) {
        OpenFile();
        if (!file) {
            return;
        }
    }

    const std::string line = fmt::format("{:016x} {:016x} {:s}\n", reinterpret_cast<u64>(start), reinterpret_cast<u64>(end) - reinterpret_cast<u64>(start), friendly_name);
    std::fwrite(line.data(), sizeof *line.data(), line.size(), file);
}
}  // namespace detail

void PerfMapClear() {
    std::lock_guard guard{mutex};

    if (!file) {
        return;
    }

    std::fclose(file);
    file = nullptr;
    OpenFile();
}

}  // namespace Dynarmic::Backend::X64

#endif

namespace Dynarmic::Backend::X64 {

namespace {

/// One block: host offsets from the code space start (code spaces are far below 4 GiB).
struct HotBlock {
    u32 host_offset;
    u32 host_size;
    u64 guest_pc;
    u32 guest_size;
};

struct HotSpace {
    uintptr_t start;
    uintptr_t code_begin;
    size_t size;
    s32 index;
    /// Blocks are emitted at increasing addresses until a clear; checked to be safe.
    bool sorted = true;
    std::vector<HotBlock> blocks;
};

std::mutex hot_mutex;
std::vector<HotSpace> hot_spaces;
s32 hot_next_index = 0;
u64 hot_clears = 0;

HotSpace* FindHotSpace(uintptr_t start) {
    for (HotSpace& space : hot_spaces) {
        if (space.start == start) {
            return &space;
        }
    }
    return nullptr;
}

}  // namespace

bool HotMapEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_JIT_HOT_MAP");
        return value != nullptr && std::string_view{value} == "1";
    }();
    return enabled;
}

void HotMapAdd(const void* space, const void* code_begin, size_t space_size, const void* host,
               size_t host_size, u64 guest_pc, u64 guest_end) {
    const auto start = reinterpret_cast<uintptr_t>(space);
    const auto at = reinterpret_cast<uintptr_t>(host);
    if (at < start || at - start >= space_size || host_size == 0) {
        return;
    }
    std::lock_guard guard{hot_mutex};
    HotSpace* entry = FindHotSpace(start);
    if (entry == nullptr) {
        entry = &hot_spaces.emplace_back(HotSpace{start, reinterpret_cast<uintptr_t>(code_begin),
                                                  space_size, hot_next_index++});
    }
    const HotBlock block{static_cast<u32>(at - start), static_cast<u32>(host_size), guest_pc,
                         static_cast<u32>(guest_end > guest_pc ? guest_end - guest_pc : 0)};
    if (!entry->blocks.empty() && block.host_offset < entry->blocks.back().host_offset) {
        entry->sorted = false;
    }
    entry->blocks.push_back(block);
}

void HotMapClear(const void* space) {
    std::lock_guard guard{hot_mutex};
    if (HotSpace* const entry = FindHotSpace(reinterpret_cast<uintptr_t>(space))) {
        entry->blocks.clear();
        entry->blocks.shrink_to_fit();
        entry->sorted = true;
        ++hot_clears;
    }
}

void HotMapForget(const void* space) {
    std::lock_guard guard{hot_mutex};
    std::erase_if(hot_spaces, [start = reinterpret_cast<uintptr_t>(space)](const HotSpace& entry) {
        return entry.start == start;
    });
}

}  // namespace Dynarmic::Backend::X64

namespace Dynarmic::A64::HotMap {

using Backend::X64::hot_clears;
using Backend::X64::hot_mutex;
using Backend::X64::hot_spaces;
using Backend::X64::HotBlock;

bool Enabled() {
    return Backend::X64::HotMapEnabled();
}

void Resolve(std::span<const u64> rips, std::span<Hit> out) {
    std::lock_guard guard{hot_mutex};
    for (size_t i = 0; i < rips.size() && i < out.size(); ++i) {
        const u64 rip = rips[i];
        Hit hit{};
        for (auto& space : hot_spaces) {
            if (rip < space.start || rip - space.start >= space.size) {
                continue;
            }
            hit.space = space.index;
            if (rip < space.code_begin) {
                hit.kind = Kind::Prelude;
                break;
            }
            if (!space.sorted) {
                std::ranges::sort(space.blocks, {}, &HotBlock::host_offset);
                space.sorted = true;
            }
            hit.kind = Kind::BetweenBlocks;
            const auto offset = static_cast<u32>(rip - space.start);
            const auto next = std::ranges::upper_bound(space.blocks, offset, {}, &HotBlock::host_offset);
            if (next != space.blocks.begin()) {
                const HotBlock& block = *std::prev(next);
                if (offset - block.host_offset < block.host_size) {
                    hit.kind = Kind::Block;
                    hit.guest_pc = block.guest_pc;
                    hit.guest_end = block.guest_pc + block.guest_size;
                    hit.host_size = block.host_size;
                }
            }
            break;
        }
        out[i] = hit;
    }
}

Stats GetStats() {
    std::lock_guard guard{hot_mutex};
    Stats stats{};
    stats.spaces = hot_spaces.size();
    stats.clears = hot_clears;
    for (const auto& space : hot_spaces) {
        stats.blocks += space.blocks.size();
    }
    return stats;
}

}  // namespace Dynarmic::A64::HotMap
