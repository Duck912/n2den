// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <vector>

#include "common/alignment.h"
#include "common/common_types.h"

namespace VideoCommon {

/// EDEN_USAGE_TRACKER_FAST=0 turns off UsageTracker's covered-interval fast path.
[[nodiscard]] inline bool UsageTrackerFastEnabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_USAGE_TRACKER_FAST");
        return !(value != nullptr && value[0] == '0');
    }();
    return enabled;
}

class UsageTracker {
    // PAGE_SHIFT is a macro on FreeBSD
    static constexpr size_t BUFFER_BYTES_PER_BITSHIFT = 6;
    static constexpr size_t BUFFER_PAGE_SHIFT = 6 + BUFFER_BYTES_PER_BITSHIFT;
    static constexpr size_t BUFFER_PAGE_BYTES = 1 << BUFFER_PAGE_SHIFT;
public:
    explicit UsageTracker(size_t size) : size_bytes{size} {
        const size_t num_pages = (size >> BUFFER_PAGE_SHIFT) + 1;
        pages.resize(num_pages, 0ULL);
        groups.resize(num_pages / PAGES_PER_GROUP + 1, 0);
    }

    void Reset() noexcept {
        recent.fill(NO_RANGE);
        covered.fill(Interval{});
        whole = false;
        // Only the pages tracked since the last reset can be set.
        if (dirty_begin < dirty_end) {
            std::fill(pages.begin() + dirty_begin, pages.begin() + dirty_end, 0ULL);
            std::fill(groups.begin() + dirty_begin / PAGES_PER_GROUP,
                      groups.begin() + (dirty_end - 1) / PAGES_PER_GROUP + 1, u8{0});
        }
        dirty_begin = (std::numeric_limits<size_t>::max)();
        dirty_end = 0;
    }

    void Track(u64 offset, u64 size) noexcept {
        if (size == 0) {
            return;
        }
        // Fast path: ranges inside an interval already tracked whole since the reset (merged from
        // overlapping or touching ranges) change nothing. Exact: only tracked bytes are covered.
        const bool fast = UsageTrackerFastEnabled();
        if (fast) {
            if (whole) {
                return;
            }
            for (const Interval& interval : covered) {
                if (offset >= interval.begin && offset + size <= interval.end) {
                    return;
                }
            }
        }
        // Consecutive draws keep binding the same ranges (a vertex buffer can span tens of MiB);
        // tracking a range again changes nothing until the next reset.
        for (const Range& range : recent) {
            if (range.offset == offset && range.size == size) {
                return;
            }
        }
        recent[recent_next++ % recent.size()] = Range{offset, size};
        const size_t page = offset >> BUFFER_PAGE_SHIFT;
        const size_t page_end = (offset + size) >> BUFFER_PAGE_SHIFT;
        if (page_end < page || page_end >= pages.size()) {
            return;
        }
        dirty_begin = (std::min)(dirty_begin, page);
        dirty_end = (std::max)(dirty_end, page_end + 1);
        if (fast) {
            Cover(offset, offset + size);
        }
        TrackPage(page, offset, size);
        if (page == page_end) {
            return;
        }
        // Whole pages in between: a group of PAGES_PER_GROUP pages entirely inside is marked
        // in the coarse level only (large vertex and storage buffers bound by thousands of draws
        // a frame filled every page of them each time).
        size_t i = page + 1;
        while (i < page_end) {
            if (i % PAGES_PER_GROUP == 0 && i + PAGES_PER_GROUP <= page_end) {
                groups[i / PAGES_PER_GROUP] = 1;
                i += PAGES_PER_GROUP;
            } else {
                pages[i++] = ~u64{0};
            }
        }
        const size_t offset_end = offset + size;
        const size_t offset_end_page_aligned = Common::AlignDown(offset_end, BUFFER_PAGE_BYTES);
        if (offset_end > offset_end_page_aligned) {
            TrackPage(page_end, offset_end_page_aligned, offset_end - offset_end_page_aligned);
        }
    }

    [[nodiscard]] bool IsUsed(u64 offset, u64 size) const noexcept {
        if (size == 0) {
            return false;
        }
        const size_t page = offset >> BUFFER_PAGE_SHIFT;
        const size_t page_end = (offset + size) >> BUFFER_PAGE_SHIFT;
        if (page_end < page || page_end >= pages.size()) {
            return false;
        }
        if (IsPageUsed(page, offset, size)) {
            return true;
        }
        if (page == page_end) {
            return false;
        }
        for (size_t i = page + 1; i < page_end; i++) {
            if (pages[i] != 0 || groups[i / PAGES_PER_GROUP] != 0) {
                return true;
            }
        }
        const size_t offset_end = offset + size;
        const size_t offset_end_page_aligned = Common::AlignDown(offset_end, BUFFER_PAGE_BYTES);
        return offset_end > offset_end_page_aligned &&
               IsPageUsed(page_end, offset_end_page_aligned, offset_end - offset_end_page_aligned);
    }

private:
    struct Interval {
        u64 begin{};
        u64 end{};
    };

    /// Records [begin, end) as tracked: joined to an interval it overlaps or touches, else in
    /// the next slot. `whole` once an interval spans the buffer.
    void Cover(u64 begin, u64 end) noexcept {
        Interval* target = nullptr;
        for (Interval& interval : covered) {
            if (interval.end > interval.begin && begin <= interval.end && end >= interval.begin) {
                interval.begin = (std::min)(interval.begin, begin);
                interval.end = (std::max)(interval.end, end);
                target = &interval;
                break;
            }
        }
        if (target == nullptr) {
            target = &covered[covered_next++ % covered.size()];
            *target = Interval{begin, end};
        } else {
            // The grown interval may now reach another one: join those too.
            for (Interval& other : covered) {
                if (&other != target && other.end > other.begin && other.begin <= target->end &&
                    other.end >= target->begin) {
                    target->begin = (std::min)(target->begin, other.begin);
                    target->end = (std::max)(target->end, other.end);
                    other = Interval{};
                }
            }
        }
        if (target->begin == 0 && target->end >= size_bytes) {
            whole = true;
        }
    }

    /// Bits of a page's 64-byte chunks that [offset, offset + size) touches. Partial chunks
    /// count, and a size of zero is no bits (the old shift by 64 was undefined).
    /// From Eglantine c4573fabb1.
    [[nodiscard]] static u64 PageMask(u64 offset, u64 size) noexcept {
        if (size == 0) {
            return 0;
        }
        const size_t offset_in_page = offset % BUFFER_PAGE_BYTES;
        const size_t end_in_page = (std::min<size_t>)(BUFFER_PAGE_BYTES, offset_in_page + size);
        const size_t first_bit = offset_in_page >> BUFFER_BYTES_PER_BITSHIFT;
        const size_t end_bit = (end_in_page + (size_t{1} << BUFFER_BYTES_PER_BITSHIFT) - 1) >>
                               BUFFER_BYTES_PER_BITSHIFT;
        const size_t num_bits = end_bit - first_bit;
        if (num_bits == 0) {
            return 0;
        }
        const u64 mask = num_bits >= 64 ? ~u64{0} : ((u64{1} << num_bits) - 1);
        return mask << first_bit;
    }

    void TrackPage(u64 page, u64 offset, u64 size) noexcept {
        pages[page] |= PageMask(offset, size);
    }

    bool IsPageUsed(u64 page, u64 offset, u64 size) const noexcept {
        return groups[page / PAGES_PER_GROUP] != 0 || (pages[page] & PageMask(offset, size)) != 0;
    }

private:
    struct Range {
        u64 offset;
        u64 size;
    };
    static constexpr Range NO_RANGE{(std::numeric_limits<u64>::max)(), 0};

    std::vector<u64> pages;
    /// One entry per PAGES_PER_GROUP pages: nonzero when every byte of them is used.
    static constexpr size_t PAGES_PER_GROUP = 64;
    std::vector<u8> groups;
    /// A draw can bind more than four ranges of one buffer (storage buffers carved from one pool):
    /// with too few slots each draw forgot the others and filled their pages again.
    std::array<Range, 16> recent = [] {
        std::array<Range, 16> ranges;
        ranges.fill(NO_RANGE);
        return ranges;
    }();
    size_t recent_next{};
    /// Byte intervals tracked whole since the reset (the fast path's memory).
    std::array<Interval, 4> covered{};
    size_t covered_next{};
    u64 size_bytes{};
    bool whole{};
    size_t dirty_begin{(std::numeric_limits<size_t>::max)()};
    size_t dirty_end{};
};

} // namespace VideoCommon
