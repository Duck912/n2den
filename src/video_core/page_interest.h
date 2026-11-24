// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstdlib>

#include "common/common_types.h"

namespace VideoCommon {

/// Lock-free count of the objects a cache keeps per 1 MiB of device address space. A cache adds
/// every object it registers over the object's whole byte range and removes it with the same
/// range, so a write can only touch one of its objects where the count is not zero: a write whose
/// pages all count zero can skip that cache (and its lock) altogether.
class PageInterest {
public:
    static constexpr u64 PAGE_BITS = 20;
    static constexpr u64 ADDRESS_BITS = 34; // MaxwellDeviceTraits::device_virtual_bits
    static constexpr u64 NUM_PAGES = 1ULL << (ADDRESS_BITS - PAGE_BITS);

    /// Counts an object over [addr, addr + size) (at least the page of addr).
    void Add(u64 addr, u64 size) noexcept {
        Change(addr, size, 1);
    }

    /// Uncounts an object added with the same range.
    void Remove(u64 addr, u64 size) noexcept {
        Change(addr, size, ~u32{0});
    }

    /// Counts an object on one page (page = addr >> PAGE_BITS).
    void AddPage(u64 page) noexcept {
        ChangePage(page, 1);
    }

    void RemovePage(u64 page) noexcept {
        ChangePage(page, ~u32{0});
    }

    /// False only when no counted object can overlap [addr, addr + size).
    [[nodiscard]] bool Any(u64 addr, u64 size) const noexcept {
        if (outside.load(std::memory_order_seq_cst) != 0) {
            return true;
        }
        const u64 first = addr >> PAGE_BITS;
        const u64 last = (addr + (size == 0 ? 1 : size) - 1) >> PAGE_BITS;
        if (last >= NUM_PAGES || last < first) {
            return true;
        }
        for (u64 page = first; page <= last; ++page) {
            if (counts[page].load(std::memory_order_seq_cst) != 0) {
                return true;
            }
        }
        return false;
    }

    /// EDEN_NX2_INVAL_FILTER=0 sends every write to every cache, as before the filter.
    [[nodiscard]] static bool FilterEnabled() noexcept {
        static const bool enabled = [] {
            const char* const value = std::getenv("EDEN_NX2_INVAL_FILTER");
            return !(value && value[0] == '0');
        }();
        return enabled;
    }

private:
    void Change(u64 addr, u64 size, u32 delta) noexcept {
        const u64 first = addr >> PAGE_BITS;
        const u64 last = (addr + (size == 0 ? 1 : size) - 1) >> PAGE_BITS;
        if (last < first) {
            outside.fetch_add(delta, std::memory_order_seq_cst);
            return;
        }
        for (u64 page = first; page <= last; ++page) {
            ChangePage(page, delta);
        }
    }

    void ChangePage(u64 page, u32 delta) noexcept {
        if (page >= NUM_PAGES) {
            // Images without a mapped address get fake addresses past ADDRESS_BITS
            // (TextureCache::InsertImage): no write can reach them, and counting them in
            // `outside` turned the filter off while one existed. Any() answers true for queries
            // past ADDRESS_BITS anyway.
            return;
        }
        counts[page].fetch_add(delta, std::memory_order_seq_cst);
    }

    std::array<std::atomic<u32>, NUM_PAGES> counts{};
    std::atomic<u32> outside{}; ///< Objects reaching past ADDRESS_BITS (or wrapping): always "any"
};

} // namespace VideoCommon
