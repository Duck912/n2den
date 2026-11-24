// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/renderer_vulkan/nvn/nvn_mirror_memo.h"

using Vulkan::Nvn::AscendingRanges;
using Vulkan::Nvn::MirrorMemo;

namespace {

/// The memo's runs as a page bitmap of `pages` pages.
std::vector<bool> Bits(const MirrorMemo& memo, u64 pages) {
    std::vector<bool> bits(pages, false);
    for (size_t i = 0; i < memo.count; ++i) {
        for (u64 p = memo.runs[i].first; p < memo.runs[i].second; ++p) {
            bits[p] = true;
        }
    }
    return bits;
}

bool WellFormed(const MirrorMemo& memo) {
    for (size_t i = 0; i < memo.count; ++i) {
        if (memo.runs[i].first >= memo.runs[i].second) {
            return false;
        }
        if (i != 0 && memo.runs[i - 1].second >= memo.runs[i].first) {
            return false; // overlapping or adjacent
        }
    }
    return memo.count <= MirrorMemo::MAX;
}

} // namespace

TEST_CASE("MirrorMemo: add merges and subtract splits", "[video_core][nvn]") {
    MirrorMemo memo;
    memo.Validate(1, 0);
    memo.Add(10, 20);
    memo.Add(20, 30); // adjacent: merged
    REQUIRE(memo.count == 1);
    REQUIRE(memo.runs[0] == std::pair<u64, u64>{10, 30});
    memo.Subtract(15, 16);
    REQUIRE(memo.count == 2);
    REQUIRE(memo.Covered(0, 100) == 19);
    memo.Validate(2, 0); // a new submission forgets everything
    REQUIRE(memo.Empty());
}

TEST_CASE("MirrorMemo: never claims a page that was not added (random)", "[video_core][nvn]") {
    std::mt19937_64 rng{1234};
    constexpr u64 PAGES = 256;
    for (int round = 0; round < 200; ++round) {
        MirrorMemo memo;
        memo.Validate(1, 0);
        std::vector<bool> truth(PAGES, false); // pages that may be claimed
        for (int op = 0; op < 60; ++op) {
            const u64 a = rng() % PAGES;
            const u64 b = a + rng() % (PAGES - a + 1);
            if (rng() % 3 == 0) {
                memo.Subtract(a, b);
                for (u64 p = a; p < b; ++p) {
                    truth[p] = false;
                }
            } else {
                memo.Add(a, b);
                for (u64 p = a; p < b; ++p) {
                    truth[p] = true;
                }
            }
            REQUIRE(WellFormed(memo));
            const std::vector<bool> bits = Bits(memo, PAGES);
            for (u64 p = 0; p < PAGES; ++p) {
                REQUIRE((!bits[p] || truth[p]));
            }
            // The gaps of any range are exactly its pages outside the runs, ascending.
            const u64 qa = rng() % PAGES;
            const u64 qb = qa + rng() % (PAGES - qa + 1);
            std::vector<bool> gap(PAGES, false);
            u64 last = 0;
            memo.ForEachGap(qa, qb, [&](u64 x, u64 y) {
                REQUIRE(x >= last);
                REQUIRE(x < y);
                last = y;
                for (u64 p = x; p < y; ++p) {
                    gap[p] = true;
                }
                return true;
            });
            for (u64 p = 0; p < PAGES; ++p) {
                REQUIRE(gap[p] == (p >= qa && p < qb && !bits[p]));
            }
        }
    }
}

TEST_CASE("MirrorMemo: DropCleared takes out exactly the failing pages", "[video_core][nvn]") {
    MirrorMemo memo;
    memo.Validate(1, 0);
    memo.Add(0, 100);
    std::vector<bool> passes(100, true);
    passes[5] = passes[6] = passes[50] = false;
    memo.DropCleared(0, 60, [&](u64 p) { return passes[p]; });
    const std::vector<bool> bits = Bits(memo, 100);
    for (u64 p = 0; p < 100; ++p) {
        REQUIRE(bits[p] == passes[p]);
    }
}

TEST_CASE("AscendingRanges: overlaps matches a linear scan", "[video_core][nvn]") {
    std::mt19937_64 rng{99};
    for (int round = 0; round < 100; ++round) {
        AscendingRanges ranges;
        std::vector<std::pair<u64, u64>> raw;
        const int n = static_cast<int>(rng() % 12);
        for (int i = 0; i < n; ++i) {
            const u64 a = rng() % 1000;
            raw.emplace_back(a, a + rng() % 80);
        }
        ranges.ranges = raw;
        ranges.Finish();
        for (int q = 0; q < 200; ++q) {
            // Mostly ascending, sometimes not.
            const u64 a = (q % 37 == 0) ? rng() % 1000 : static_cast<u64>(q) * 5;
            const u64 b = a + 1 + rng() % 8;
            bool expect = false;
            for (const auto& [x, y] : raw) {
                expect |= x < b && y > a;
            }
            REQUIRE(ranges.Overlaps(a, b) == expect);
        }
    }
}
