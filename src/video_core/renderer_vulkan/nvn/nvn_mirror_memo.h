// SPDX-FileCopyrightText: Copyright 2026 n2den Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Clean-range memo of a mirrored pool (Pool::mirror_memo, Recorder::MirrorNeedsUpload /
// UploadRange): up to MAX ranges of mirror pages that are all known to need no look in the
// current submission and raw epoch. Without it an indexed indirect draw whose index range is the
// rest of a 450 MiB pool (Donkey Kong Bananza) walks ~115,000 page entries twice per draw, about
// 116 million a frame, although nearly all of them were found clean by an earlier draw of the
// same submission.
//
// Invariant: every page inside a run passes the walk's skip test (valid and checked in this
// submission, or valid, trusted and of the current raw epoch). The memo is reset whenever the
// submission (stamp) or the pool's raw epoch moves; every site that clears a page's checked,
// trusted or valid takes the pages that no longer pass out of it (MirrorMemoDropCleared).
// EDEN_NVN_MIRROR_MEMO=0 turns it off, =verify also runs the full walk and counts disagreements.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "common/common_types.h"

namespace Vulkan::Nvn {

struct MirrorMemo {
    static constexpr size_t MAX = 4;
    /// Submission (seq + 1, as MirrorPage::checked) the runs hold for; 0: empty.
    u32 stamp{};
    u64 epoch{}; ///< Pool::split_raw_epoch_seen the runs hold for
    size_t count{};
    /// Page index ranges [first, second): sorted, disjoint, never adjacent.
    std::array<std::pair<u64, u64>, MAX> runs{};

    [[nodiscard]] bool Empty() const noexcept {
        return count == 0;
    }

    void Clear() noexcept {
        count = 0;
    }

    /// Makes the memo hold for (stamp, epoch), dropping its runs when it held for another.
    void Validate(u32 new_stamp, u64 new_epoch) noexcept {
        if (stamp != new_stamp || epoch != new_epoch) {
            stamp = new_stamp;
            epoch = new_epoch;
            count = 0;
        }
    }

    /// Adds [begin, end), merging with overlapping or adjacent runs. Past MAX runs the shortest
    /// run is dropped (the memo may forget, never invent).
    void Add(u64 begin, u64 end) noexcept {
        if (begin >= end) {
            return;
        }
        std::array<std::pair<u64, u64>, MAX + 1> next{};
        size_t n = 0;
        bool placed = false;
        for (size_t i = 0; i < count; ++i) {
            const auto run = runs[i];
            if (run.second < begin) {
                next[n++] = run;
            } else if (run.first > end) {
                if (!placed) {
                    next[n++] = {begin, end};
                    placed = true;
                }
                next[n++] = run;
            } else {
                begin = (std::min)(begin, run.first);
                end = (std::max)(end, run.second);
            }
        }
        if (!placed) {
            next[n++] = {begin, end};
        }
        if (n > MAX) {
            size_t shortest = 0;
            for (size_t i = 1; i < n; ++i) {
                if (next[i].second - next[i].first < next[shortest].second - next[shortest].first) {
                    shortest = i;
                }
            }
            for (size_t i = shortest; i + 1 < n; ++i) {
                next[i] = next[i + 1];
            }
            --n;
        }
        for (size_t i = 0; i < n; ++i) {
            runs[i] = next[i];
        }
        count = n;
    }

    /// Takes [begin, end) out. A split past MAX runs drops the shortest run.
    void Subtract(u64 begin, u64 end) noexcept {
        if (begin >= end || count == 0) {
            return;
        }
        std::array<std::pair<u64, u64>, MAX + 1> next{};
        size_t n = 0;
        for (size_t i = 0; i < count; ++i) {
            const auto run = runs[i];
            if (run.second <= begin || run.first >= end) {
                next[n++] = run;
                continue;
            }
            if (run.first < begin) {
                next[n++] = {run.first, begin};
            }
            if (run.second > end) {
                next[n++] = {end, run.second}; // at most one run is cut in two: n <= MAX + 1
            }
        }
        if (n > MAX) {
            size_t shortest = 0;
            for (size_t i = 1; i < n; ++i) {
                if (next[i].second - next[i].first < next[shortest].second - next[shortest].first) {
                    shortest = i;
                }
            }
            for (size_t i = shortest; i + 1 < n; ++i) {
                next[i] = next[i + 1];
            }
            --n;
        }
        for (size_t i = 0; i < n; ++i) {
            runs[i] = next[i];
        }
        count = n;
    }

    /// The parts of [begin, end) outside the runs, ascending, through `fn(first, last)`.
    /// Stops early (returns false) when `fn` returns false.
    template <typename Fn>
    bool ForEachGap(u64 begin, u64 end, Fn&& fn) const {
        u64 at = begin;
        for (size_t i = 0; i < count && at < end; ++i) {
            const auto run = runs[i];
            if (run.second <= at) {
                continue;
            }
            if (run.first >= end) {
                break;
            }
            if (run.first > at && !fn(at, run.first)) {
                return false;
            }
            at = (std::max)(at, run.second);
        }
        if (at < end) {
            return fn(at, end);
        }
        return true;
    }

    /// Pages of [begin, end) inside the runs.
    [[nodiscard]] u64 Covered(u64 begin, u64 end) const noexcept {
        u64 total = 0;
        for (size_t i = 0; i < count; ++i) {
            const u64 a = (std::max)(begin, runs[i].first);
            const u64 b = (std::min)(end, runs[i].second);
            if (a < b) {
                total += b - a;
            }
        }
        return total;
    }

    /// Pages [begin, end) had checked, trusted or valid cleared: takes out those of them inside
    /// the runs for which `passes(page)` (the walk's skip test at this memo's stamp) is false now.
    template <typename Passes>
    void DropCleared(u64 begin, u64 end, Passes&& passes) {
        if (count == 0 || begin >= end) {
            return;
        }
        // Collected first: Subtract reshapes the runs.
        std::array<std::pair<u64, u64>, MAX> cover{};
        const size_t cover_count = count;
        for (size_t i = 0; i < count; ++i) {
            cover[i] = {(std::max)(begin, runs[i].first), (std::min)(end, runs[i].second)};
        }
        for (size_t i = 0; i < cover_count; ++i) {
            const auto [a, b] = cover[i];
            u64 failing_from = ~u64{0};
            for (u64 page = a; page < b; ++page) {
                if (!passes(page)) {
                    if (failing_from == ~u64{0}) {
                        failing_from = page;
                    }
                } else if (failing_from != ~u64{0}) {
                    Subtract(failing_from, page);
                    failing_from = ~u64{0};
                }
            }
            if (failing_from != ~u64{0}) {
                Subtract(failing_from, b);
            }
        }
    }
};

/// Sorted, merged snapshot of ranges (a pool's gpu_dirty) asked mostly with ascending ranges
/// (MirrorPageSettled, UploadRange): a cursor instead of a scan over all of them per page.
struct AscendingRanges {
    std::vector<std::pair<u64, u64>> ranges;
    size_t cursor{};
    u64 last_begin{};

    /// Sorts and merges `ranges` (after they were filled).
    void Finish() {
        std::sort(ranges.begin(), ranges.end());
        size_t n = 0;
        for (const auto& range : ranges) {
            // Empty ranges are kept: like the scan this replaces, one strictly inside the asked
            // range counts as overlapping it.
            if (n != 0 && range.first <= ranges[n - 1].second) {
                ranges[n - 1].second = (std::max)(ranges[n - 1].second, range.second);
            } else {
                ranges[n++] = range;
            }
        }
        ranges.resize(n);
        cursor = 0;
        last_begin = 0;
    }

    /// Any range overlaps [begin, end). Correct for any order, fast for ascending `begin`.
    [[nodiscard]] bool Overlaps(u64 begin, u64 end) {
        if (begin < last_begin) {
            cursor = 0;
        }
        last_begin = begin;
        while (cursor < ranges.size() && ranges[cursor].second <= begin) {
            ++cursor;
        }
        return cursor < ranges.size() && ranges[cursor].first < end;
    }
};

} // namespace Vulkan::Nvn
