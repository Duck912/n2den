// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <random>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include "common/common_types.h"
#include "video_core/textures/astc.h"

using Tegra::Texture::ASTC::Channels;

// The smart BC choice trusts ClassifyChannels: a texture it calls alpha-free gets a BC1 stand-in,
// which has no alpha. So every block it calls alpha-free must decode (with the same decoder the
// stand-ins are made from) to alpha 255 in every texel. Random blocks cover the endpoint encodings
// broadly; some use an alpha endpoint mode with every alpha endpoint at 255 (the opaque RGBA
// blocks the classifier now counts as alpha-free), and the test requires that it saw some.
TEST_CASE("ASTC classification calls a block alpha-free only when it decodes opaque", "[astc]") {
    constexpr u32 Footprint = 8; // Mario Kart World's ASTC 8x8
    std::mt19937_64 random{0x5eed'a57c'0000'0001ULL};
    std::array<u8, 16> block{};
    std::vector<u8> texels(Footprint * Footprint * 4);
    u32 alpha_free = 0;
    u32 alpha_mode_but_opaque = 0;
    for (u32 round = 0; round < 400000; ++round) {
        for (u32 i = 0; i < 16; i += 8) {
            const u64 bits = random();
            std::memcpy(block.data() + i, &bits, 8);
        }
        bool empty = false;
        const Channels channels = Tegra::Texture::ASTC::ClassifyChannels(block, Footprint, Footprint, &empty);
        if (empty || (static_cast<u32>(channels) & 1) != 0) {
            continue;
        }
        ++alpha_free;
        if (!Tegra::Texture::ASTC::IsOpaque(block)) {
            ++alpha_mode_but_opaque; // an alpha endpoint mode: only the new endpoint check let it through
        }
        Tegra::Texture::ASTC::Decompress(block, Footprint, Footprint, 1, Footprint, Footprint, texels);
        for (u32 texel = 0; texel < Footprint * Footprint; ++texel) {
            INFO("round " << round << " texel " << texel);
            REQUIRE(texels[texel * 4 + 3] == 255);
        }
    }
    INFO("alpha-free blocks " << alpha_free << ", of them with an alpha endpoint mode " << alpha_mode_but_opaque);
    REQUIRE(alpha_free > 1000);
    REQUIRE(alpha_mode_but_opaque > 0);
}
