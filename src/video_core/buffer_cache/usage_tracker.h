// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

#include "common/alignment.h"
#include "common/common_types.h"

namespace VideoCommon {

class UsageTracker {
    // PAGE_SHIFT is a macro on FreeBSD
    static constexpr size_t BUFFER_BYTES_PER_BITSHIFT = 6;
    static constexpr size_t BUFFER_PAGE_SHIFT = 6 + BUFFER_BYTES_PER_BITSHIFT;
    static constexpr size_t BUFFER_PAGE_BYTES = 1 << BUFFER_PAGE_SHIFT;
public:
    explicit UsageTracker(size_t size) {
        const size_t num_pages = (size >> BUFFER_PAGE_SHIFT) + 1;
        pages.resize(num_pages, 0ULL);
    }

    void Reset() noexcept {
        recent.fill(NO_RANGE);
        // Only the pages tracked since the last reset can be set.
        if (dirty_begin < dirty_end) {
            std::fill(pages.begin() + dirty_begin, pages.begin() + dirty_end, 0ULL);
        }
        dirty_begin = (std::numeric_limits<size_t>::max)();
        dirty_end = 0;
    }

    void Track(u64 offset, u64 size) noexcept {
        if (size == 0) {
            return;
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
        TrackPage(page, offset, size);
        if (page == page_end) {
            return;
        }
        for (size_t i = page + 1; i < page_end; i++) {
            pages[i] = ~u64{0};
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
            if (pages[i] != 0) {
                return true;
            }
        }
        const size_t offset_end = offset + size;
        const size_t offset_end_page_aligned = Common::AlignDown(offset_end, BUFFER_PAGE_BYTES);
        return offset_end > offset_end_page_aligned &&
               IsPageUsed(page_end, offset_end_page_aligned, offset_end - offset_end_page_aligned);
    }

private:
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
        return (pages[page] & PageMask(offset, size)) != 0;
    }

private:
    struct Range {
        u64 offset;
        u64 size;
    };
    static constexpr Range NO_RANGE{(std::numeric_limits<u64>::max)(), 0};

    std::vector<u64> pages;
    /// A draw can bind more than four ranges of one buffer (storage buffers carved from one pool):
    /// with too few slots each draw forgot the others and filled their pages again.
    std::array<Range, 16> recent = [] {
        std::array<Range, 16> ranges;
        ranges.fill(NO_RANGE);
        return ranges;
    }();
    size_t recent_next{};
    size_t dirty_begin{(std::numeric_limits<size_t>::max)()};
    size_t dirty_end{};
};

} // namespace VideoCommon
