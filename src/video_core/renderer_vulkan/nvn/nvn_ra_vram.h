// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Record-ahead (EDEN_NX2_NVN_RECORD_AHEAD): the frozen write view of VRAM-mirrored pools and its
// release repair. No Vulkan calls
// in here, so the rules are unit tested (src/tests/nvn/record_ahead.cpp).
//
// From a pool's first decision recorded ahead until the release, decisions recorded ahead do not
// look at the pool's CPU write tracker: they see the pool as it was at that first decision (which
// collected the CPU writes made so far, as today's first decision does). A CPU write in between
// then acts as if it came after the whole span, and the release repairs it:
//   1. snapshot the valid bits (before the CPU writes are applied);
//   2. apply the deferred CpuWrite/NotifyHostWrite commands, then collect the CPU writes;
//   3. copy guest -> mirror (prologue) the pages a decision treated as current or valid that were
//      written or flushed: (TV_take | TV_current) & changed;
//   4. set valid and arm again (TV_take | CA) & changed where the snapshot had them valid (a page
//      copied ahead is copied at GPU time, after the write);
//   5. leave pages only MirrorCurrent treated as current invalid (today leaves them invalid too,
//      or valid-but-dirty: both answer every later decision the same way).

#pragma once

#include <algorithm>
#include <bit>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/nvn/nvn_pool_mirror.h"

namespace Vulkan::Nvn {

using RaRuns = std::vector<std::pair<u64, u64>>;

/// A set of 4 KiB pages of one pool.
class RaPageSet {
public:
    static constexpr u64 PAGE = MirrorPages::PAGE;

    RaPageSet() = default;
    explicit RaPageSet(u64 pool_size)
        : size{pool_size}, bits(static_cast<size_t>((Pages(pool_size) + 63) / 64), 0) {}

    [[nodiscard]] static constexpr u64 Pages(u64 bytes) noexcept {
        return (bytes + PAGE - 1) / PAGE;
    }
    [[nodiscard]] u64 Size() const noexcept {
        return size;
    }
    [[nodiscard]] size_t Words() const noexcept {
        return bits.size();
    }
    [[nodiscard]] u64 Word(size_t index) const noexcept {
        return index < bits.size() ? bits[index] : 0;
    }
    void OrWord(size_t index, u64 value) noexcept {
        if (index < bits.size()) {
            bits[index] |= value;
        }
    }
    [[nodiscard]] bool Test(u64 page) const noexcept {
        return page / 64 < bits.size() && ((bits[page / 64] >> (page % 64)) & 1) != 0;
    }
    /// The pages [offset, offset + length) touches (clamped to the pool).
    void Set(u64 offset, u64 length) noexcept {
        if (length == 0 || offset >= size) {
            return;
        }
        const u64 first = offset / PAGE;
        const u64 last = ((std::min)(offset + length, size) - 1) / PAGE;
        for (u64 page = first; page <= last;) {
            const u64 bit = page % 64;
            const u64 count = (std::min)(64 - bit, last - page + 1);
            const u64 mask = (count == 64 ? ~u64{0} : ((u64{1} << count) - 1)) << bit;
            bits[page / 64] |= mask;
            page += count;
        }
    }
    [[nodiscard]] u64 Count() const noexcept {
        u64 count = 0;
        for (const u64 word : bits) {
            count += static_cast<u64>(std::popcount(word));
        }
        return count;
    }
    /// The runs [begin, end) of set pages (pool offsets; the last stops at the pool size).
    [[nodiscard]] RaRuns Runs() const {
        RaRuns out;
        const u64 pages = Pages(size);
        u64 page = 0;
        while (page < pages) {
            if (!Test(page)) {
                ++page;
                continue;
            }
            const u64 begin = page;
            while (page < pages && Test(page)) {
                ++page;
            }
            out.emplace_back(begin * PAGE, (std::min)(page * PAGE, size));
        }
        return out;
    }

private:
    u64 size{};
    std::vector<u64> bits;
};

/// What the decisions recorded ahead did with one VRAM-mirrored pool since its first one.
struct RaVramLog {
    RaVramLog() = default;
    RaVramLog(u64 pool_size, u64 epoch_)
        : epoch{epoch_}, tv_take{pool_size}, tv_current{pool_size}, copied{pool_size} {}

    /// PoolMirror::epoch_clean after the first decision's collect (the frozen view's epoch).
    u64 epoch{};
    RaPageSet tv_take;    ///< pages a take covered while they were valid (no copy: TV_take)
    RaPageSet tv_current; ///< pages MirrorCurrent answered current over (TV_current)
    RaPageSet copied;     ///< pages taken (copied in) ahead (CA)
    /// Runs (pool offsets) the deferred CpuWrite/NotifyHostWrite commands invalidate at the
    /// release (their effects on the mirror are applied there).
    RaRuns flushed;
    u64 decisions{};

    /// A take over [offset, offset + length): runs[first_new..] are the pages it took (sorted,
    /// page aligned, inside the range); the rest of the range was valid.
    void NoteTake(u64 offset, u64 length, const RaRuns& runs, size_t first_new) {
        const u64 size = tv_take.Size();
        if (length == 0 || offset >= size) {
            return;
        }
        const u64 begin = offset / RaPageSet::PAGE * RaPageSet::PAGE;
        const u64 end = (std::min)(offset + length, size);
        u64 cursor = begin;
        for (size_t i = first_new; i < runs.size(); ++i) {
            const auto [run_begin, run_end] = runs[i];
            if (run_begin > cursor) {
                tv_take.Set(cursor, run_begin - cursor);
            }
            copied.Set(run_begin, run_end - run_begin);
            cursor = (std::max)(cursor, run_end);
        }
        if (end > cursor) {
            tv_take.Set(cursor, end - cursor);
        }
    }
    /// A take of the invalid pages of `wanted` (whole pool): runs[first_new..] were taken, the
    /// other pages of `wanted` were valid.
    void NoteTakeWanted(const MirrorPages& wanted, const RaRuns& runs, size_t first_new) {
        const u64 size = tv_take.Size();
        u64 cursor = 0;
        const auto valid_gap = [&](u64 from, u64 to) {
            for (u64 page = from / RaPageSet::PAGE; page * RaPageSet::PAGE < to; ++page) {
                if (wanted.AllValid(page * RaPageSet::PAGE, RaPageSet::PAGE)) {
                    tv_take.Set(page * RaPageSet::PAGE, RaPageSet::PAGE);
                }
            }
        };
        for (size_t i = first_new; i < runs.size(); ++i) {
            const auto [run_begin, run_end] = runs[i];
            if (run_begin > cursor) {
                valid_gap(cursor, run_begin);
            }
            copied.Set(run_begin, run_end - run_begin);
            cursor = (std::max)(cursor, run_end);
        }
        if (size > cursor) {
            valid_gap(cursor, size);
        }
    }
    void NoteCurrent(u64 offset, u64 length) {
        tv_current.Set(offset, length);
    }
};

/// Steps 3-5 of the release repair for one pool.
struct RaVramRepair {
    RaRuns copies;     ///< prologue copy-ins guest -> mirror (step 3)
    RaRuns revalidate; ///< pages set valid again; the caller arms them in the tracker (step 4)
    u64 changed_pages{};
    u64 copy_pages{};
    u64 revalidate_pages{};
    u64 current_only_pages{}; ///< TV_current-only pages left invalid (step 5)
};

/// `before`: the mirror's valid bits before step 2. `pages`: after step 2 (the deferred commands'
/// runs invalidated, the CPU writes collected). `written`: the runs (pool offsets) the collect took.
/// With `apply`, the revalidated pages are set valid in `pages`; without it (guest memory stopped:
/// nothing is copied) they stay invalid.
[[nodiscard]] inline RaVramRepair RaVramRepairPlan(const RaVramLog& log, const MirrorPages& before,
                                                   MirrorPages& pages, const RaRuns& written,
                                                   bool apply = true) {
    RaVramRepair out;
    const u64 size = log.tv_take.Size();
    RaPageSet changed{size};
    for (const auto& [begin, end] : written) {
        if (end > begin) {
            changed.Set(begin, end - begin);
        }
    }
    for (const auto& [begin, end] : log.flushed) {
        if (end > begin) {
            changed.Set(begin, end - begin);
        }
    }
    out.changed_pages = changed.Count();
    RaPageSet copy{size};
    RaPageSet revalidate{size};
    for (size_t word = 0; word < changed.Words(); ++word) {
        const u64 ch = changed.Word(word);
        if (ch == 0) {
            continue;
        }
        const u64 take = log.tv_take.Word(word);
        const u64 current = log.tv_current.Word(word);
        const u64 ca = log.copied.Word(word);
        copy.OrWord(word, (take | current) & ch);
        u64 candidates = (take | ca) & ch;
        u64 valid_before = 0;
        while (candidates != 0) {
            const u64 bit = static_cast<u64>(std::countr_zero(candidates));
            candidates &= candidates - 1;
            const u64 page = word * 64 + bit;
            if (before.AllValid(page * RaPageSet::PAGE, RaPageSet::PAGE)) {
                valid_before |= u64{1} << bit;
            }
        }
        revalidate.OrWord(word, valid_before);
        out.current_only_pages += static_cast<u64>(std::popcount(current & ~(take | ca) & ch));
    }
    out.copies = copy.Runs();
    out.copy_pages = copy.Count();
    if (apply) {
        out.revalidate = revalidate.Runs();
        out.revalidate_pages = revalidate.Count();
        for (const auto& [begin, end] : out.revalidate) {
            pages.Set(begin, end - begin);
        }
    }
    return out;
}

} // namespace Vulkan::Nvn
