// SPDX-FileCopyrightText: Copyright 2026 n2den Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/nvn/guest_memory.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"
#include "video_core/renderer_vulkan/nvn/nvn_mirror_raw_stats.h"

namespace Vulkan::Nvn::MirrorRawStats {

namespace {

constexpr u64 WINDOW_PRESENTS = 600;
constexpr const char* CONSUMER_NAMES[Count] = {"other", "vertex", "index",
                                               "uniform", "storage", "indirect"};

struct PoolStats {
    std::weak_ptr<Pool> pool;
    u64 serial{};
    u64 address{};
    u64 size{};
    bool split{};
    bool sparse{};
    std::array<u64, Count> raw{};
    std::array<u64, Count> raw_changed{};
    u64 first{};
    u64 untrusted{};
    u64 untrusted_changed{};
    u64 trusted{};
    std::vector<u8> used; ///< per page: walked by a check since the last sweep
    // The last sweep.
    u64 sweep_valid{};
    u64 sweep_untrusted{};
    u64 sweep_cpu_pending{};
    u64 sweep_differ{};
    u64 sweep_differ_used{};
    u64 sweep_differ_current{}; ///< page.raw_epoch == split_raw_epoch_seen (compared after it)
    std::vector<u64> sweep_examples;
    u64 total_sweep_differ{};
    u64 total_sweep_differ_used{};
};

struct State {
    std::unordered_map<const Pool*, PoolStats> pools;
    const Pool* last_key{};
    PoolStats* last{};
    u64 adopted{};
    u64 raw_draws{};
    u64 raw_flights{};
    u64 raw_through{};
    u64 raw_unaliased{};
    std::unordered_map<u64, std::pair<u64, bool>> programs; ///< key -> draws, written through
    u64 window_start{};
    u64 sweeps{};
};

State& Get() {
    static State state;
    return state;
}

PoolStats& Of(Pool& pool) {
    State& s = Get();
    if (s.last_key == &pool && s.last != nullptr && s.last->serial == pool.serial) {
        return *s.last;
    }
    PoolStats& entry = s.pools[&pool];
    if (entry.serial != pool.serial || entry.pool.expired()) {
        entry = PoolStats{};
        entry.pool = pool.weak_from_this();
        entry.serial = pool.serial;
        entry.address = pool.address;
        entry.size = pool.size;
        entry.split = pool.split;
        entry.sparse = pool.sparse != nullptr;
    }
    entry.split = pool.split;
    if (entry.used.size() != pool.mirror_pages.size()) {
        entry.used.assign(pool.mirror_pages.size(), 0);
    }
    s.last_key = &pool;
    s.last = &entry;
    return entry;
}

void MarkUsed(PoolStats& entry, u64 page) {
    if (page < entry.used.size()) {
        entry.used[static_cast<size_t>(page)] = 1;
    }
}

u64 Sum(const std::array<u64, Count>& values) {
    u64 total = 0;
    for (const u64 value : values) {
        total += value;
    }
    return total;
}

void Sweep(const std::function<u64(Pool&, u64)>& hash, VideoCommon::Nvn::GuestMemory* memory) {
    State& s = Get();
    ++s.sweeps;
    for (auto& [key, entry] : s.pools) {
        static_cast<void>(key);
        entry.sweep_valid = entry.sweep_untrusted = entry.sweep_cpu_pending = 0;
        entry.sweep_differ = entry.sweep_differ_used = entry.sweep_differ_current = 0;
        entry.sweep_examples.clear();
        const std::shared_ptr<Pool> pool = entry.pool.lock();
        if (!pool || !pool->split || pool->backing != Pool::Backing::Mirror ||
            pool->pending.load(std::memory_order_acquire)) {
            continue;
        }
        for (u64 index = 0; index < pool->mirror_pages.size(); ++index) {
            const Pool::MirrorPage& page = pool->mirror_pages[static_cast<size_t>(index)];
            if (!page.valid) {
                continue;
            }
            ++entry.sweep_valid;
            if (!page.trusted) {
                ++entry.sweep_untrusted;
                continue;
            }
            if (hash(*pool, index) == page.hash) {
                continue;
            }
            if (memory != nullptr && pool->mirror_track != 0 &&
                memory->AnyWritten(pool->mirror_track, pool->address + index * Pool::MIRROR_PAGE,
                                   Pool::MIRROR_PAGE)) {
                ++entry.sweep_cpu_pending;
                continue;
            }
            ++entry.sweep_differ;
            const bool used = index < entry.used.size() && entry.used[static_cast<size_t>(index)];
            entry.sweep_differ_used += used ? 1 : 0;
            entry.sweep_differ_current += page.raw_epoch == pool->split_raw_epoch_seen ? 1 : 0;
            if (entry.sweep_examples.size() < 6) {
                entry.sweep_examples.push_back(index * Pool::MIRROR_PAGE);
            }
        }
        entry.total_sweep_differ += entry.sweep_differ;
        entry.total_sweep_differ_used += entry.sweep_differ_used;
        std::fill(entry.used.begin(), entry.used.end(), u8{0});
    }
}

void Log(u64 frames) {
    State& s = Get();
    const double f = static_cast<double>(frames);
    std::array<u64, Count> raw{};
    std::array<u64, Count> raw_changed{};
    u64 untrusted = 0;
    u64 untrusted_changed = 0;
    u64 first = 0;
    std::vector<PoolStats*> order;
    for (auto& [key, entry] : s.pools) {
        static_cast<void>(key);
        for (u32 c = 0; c < Count; ++c) {
            raw[c] += entry.raw[c];
            raw_changed[c] += entry.raw_changed[c];
        }
        untrusted += entry.untrusted;
        untrusted_changed += entry.untrusted_changed;
        first += entry.first;
        order.push_back(&entry);
    }
    std::string by_consumer;
    for (u32 c = 0; c < Count; ++c) {
        if (raw[c] != 0) {
            by_consumer += fmt::format(" {} {:.1f}/{:.2f}", CONSUMER_NAMES[c], raw[c] / f,
                                       raw_changed[c] / f);
        }
    }
    LOG_INFO(Render_Vulkan,
             "[nvn] mirror raw stats: {} frames, per frame: raw-epoch pages hashed {:.1f} "
             "changed {:.3f} (by consumer hashed/changed:{}); untrusted {:.1f} changed {:.1f}; "
             "first {:.1f}; epochs adopted {:.1f}; raw-storing draws {:.1f} (written through "
             "{:.1f}, not {:.1f}) in {:.2f} submissions; sweeps {}",
             frames, Sum(raw) / f, Sum(raw_changed) / f, by_consumer, untrusted / f,
             untrusted_changed / f, first / f, s.adopted / f, s.raw_draws / f, s.raw_through / f,
             s.raw_unaliased / f, s.raw_flights / f, s.sweeps);
    std::ranges::sort(order, [](const PoolStats* a, const PoolStats* b) {
        const u64 ka = Sum(a->raw) + a->sweep_differ;
        const u64 kb = Sum(b->raw) + b->sweep_differ;
        return ka != kb ? ka > kb : a->address < b->address;
    });
    size_t shown = 0;
    for (const PoolStats* entry : order) {
        if (shown == 12 || (Sum(entry->raw) == 0 && entry->sweep_differ == 0 &&
                            entry->total_sweep_differ == 0)) {
            continue;
        }
        ++shown;
        std::string consumers;
        for (u32 c = 0; c < Count; ++c) {
            if (entry->raw[c] != 0) {
                consumers += fmt::format(" {} {:.1f}/{:.2f}", CONSUMER_NAMES[c],
                                         entry->raw[c] / f, entry->raw_changed[c] / f);
            }
        }
        std::string examples;
        for (const u64 offset : entry->sweep_examples) {
            examples += fmt::format(" +{:#x}", offset);
        }
        LOG_INFO(Render_Vulkan,
                 "[nvn] mirror raw stats pool {:#x} ({:.2f} MiB, serial {}, {}{}): raw-epoch "
                 "hashed/changed per frame{}; untrusted {:.1f}/{:.2f}; trusted {:.1f}; sweep: "
                 "valid {} untrusted {} cpu-pending {} DIFFER {} (used {}, compared after the "
                 "last epoch {}){}; all sweeps differ {} used {}",
                 entry->address, entry->size / 1048576.0, entry->serial,
                 entry->split ? "split" : "not split", entry->sparse ? " sparse" : "", consumers,
                 entry->untrusted / f, entry->untrusted_changed / f, entry->trusted / f,
                 entry->sweep_valid, entry->sweep_untrusted, entry->sweep_cpu_pending,
                 entry->sweep_differ, entry->sweep_differ_used, entry->sweep_differ_current,
                 examples, entry->total_sweep_differ, entry->total_sweep_differ_used);
    }
    std::vector<std::pair<u64, std::pair<u64, bool>>> programs(s.programs.begin(),
                                                               s.programs.end());
    std::ranges::sort(programs, [](const auto& a, const auto& b) {
        return a.second.first > b.second.first;
    });
    std::string top;
    for (size_t i = 0; i < programs.size() && i < 10; ++i) {
        top += fmt::format(" {:016x}{}:{:.1f}", programs[i].first,
                           programs[i].second.second ? "(wt)" : "", programs[i].second.first / f);
    }
    LOG_INFO(Render_Vulkan, "[nvn] mirror raw stats: {} raw-storing program sets; top per frame:{}",
             programs.size(), top);
    // Reset the window counters (pools stay, with their sweep totals and used flags).
    for (auto& [key, entry] : s.pools) {
        static_cast<void>(key);
        entry.raw = {};
        entry.raw_changed = {};
        entry.first = entry.untrusted = entry.untrusted_changed = entry.trusted = 0;
    }
    s.adopted = s.raw_draws = s.raw_flights = s.raw_through = s.raw_unaliased = 0;
    s.programs.clear();
}

} // Anonymous namespace

bool Enabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NVN_MIRROR_RAW_STATS");
        return value != nullptr && std::string_view{value} == "1";
    }();
    return enabled;
}

void NotePage(Pool& pool, u64 page, u32 kind, bool changed) {
    PoolStats& entry = Of(pool);
    MarkUsed(entry, page);
    const u32 consumer = consumer_hint < Count ? consumer_hint : Other;
    switch (kind) {
    case 0:
        ++entry.first;
        break;
    case 1:
        ++entry.untrusted;
        entry.untrusted_changed += changed ? 1 : 0;
        break;
    default:
        ++entry.raw[consumer];
        entry.raw_changed[consumer] += changed ? 1 : 0;
        break;
    }
}

void NoteTrusted(Pool& pool, u64 page) {
    PoolStats& entry = Of(pool);
    MarkUsed(entry, page);
    ++entry.trusted;
}

void NoteAdopted() {
    ++Get().adopted;
}

void NoteRawDraw(u64 program, bool first_in_flight, bool written_through) {
    State& s = Get();
    ++s.raw_draws;
    s.raw_flights += first_in_flight ? 1 : 0;
    ++(written_through ? s.raw_through : s.raw_unaliased);
    auto& slot = s.programs[program];
    ++slot.first;
    slot.second = written_through;
}

void OnPresent(u64 index, const std::function<u64(Pool&, u64)>& hash,
               VideoCommon::Nvn::GuestMemory* memory) {
    State& s = Get();
    if (s.window_start == 0) {
        s.window_start = index;
        return;
    }
    if (index - s.window_start < WINDOW_PRESENTS) {
        return;
    }
    Sweep(hash, memory);
    Log(index - s.window_start);
    s.window_start = index;
}

} // namespace Vulkan::Nvn::MirrorRawStats
