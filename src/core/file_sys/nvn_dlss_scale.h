// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// The DLSS quality setting's scene size (nvn_dlss_quality) for the games whose code it can patch:
// a per-build table, as file_sys/fefw_ultrawide.h. Today one build: The Legend of Zelda: Ocarina
// of Time (Switch 2), main executable E0F3CCFA... Any other build or title is left alone, and
// the setting then does nothing.
//
// Ocarina of Time draws its 3D scene at a share of the output that its GPU-time controller picks
// (dynamic resolution) unless an engine flag sends it down a fixed path whose share is the
// constant 1.0. Two instructions of main:
// - main+0x805bd0, `tbz w8, #0, main+0x80640c`: the flag test; `b main+0x80640c` always takes the
//   fixed path. This is the one record of the "Max Resolution" mod (IPS offset 0x805cd0,
//   0f020014), byte for byte.
// - main+0x806420, `movz w9, #0x3f80, lsl #16`: the fixed path's share, 1.0f. Its place and
//   meaning come from Longshot (Hookshot fork, nx2/oot_patches.py: hos_fixed_scale_bits replaces
//   this constant); the "Fixed 1080p (4K output)" mod sets the same instruction to 0.75. Writing
//   another float's upper half here (`movz w9, #<hi16>, lsl #16`) is the scene's share.
// No code was taken from either; only the two sites. Longshot also forces a scale recompute in
// main+0x31d3200 because it switches to the fixed path while the game runs; taken from the first
// frame (as here) the fixed path's share holds without it (Kokiri: a steady 1707x960 at 0x3f2a).
//
// With the "Max Resolution" mod on, its record and this branch are the same bytes. A mod that sets
// the share itself (the 4K mod's 0.75) is overridden: the built-in patch is applied after the mod
// folders, so the setting wins, and the share is of whatever output size the game then has.

#pragma once

#include <cstring>
#include <optional>
#include <string_view>
#include <vector>

#include "common/common_types.h"
#include "common/settings_enums.h"

namespace FileSys::NvnDlssScale {

/// Ocarina of Time (Switch 2)'s main executable, as PatchNSO names build IDs (trailing zeros cut).
constexpr std::string_view OotMainBuildId = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348";

/// Offsets in PatchNSO's image (the 0x100-byte NSO header, then main): main + 0x100.
constexpr std::size_t BranchOffset = 0x805cd0;
constexpr std::size_t ScaleOffset = 0x806520;
/// tbz w8, #0, +0x83c (as shipped) and b +0x83c (the fixed path).
constexpr u32 BranchShipped = 0x360041e8;
constexpr u32 BranchFixed = 0x1400020f;

/// movz w9, #<hi16>, lsl #16 (any immediate): the shipped 1.0 or a mod's share.
[[nodiscard]] constexpr bool IsScaleInstruction(u32 instruction) {
    return (instruction & 0xffe0001fu) == 0x52a00009u;
}

[[nodiscard]] constexpr u32 ScaleInstruction(u16 float_high) {
    return 0x52a00009u | (u32{float_high} << 5);
}

/// The scene's share for a quality level as the upper half of a float (the lower half is zero, so
/// the shares are a little under DLSS's nominal ones), or nothing for Native.
[[nodiscard]] constexpr std::optional<u16> FloatHigh(Settings::NvnDlssQuality quality) {
    switch (quality) {
    case Settings::NvnDlssQuality::Quality:
        return u16{0x3f2a}; // 0.6640625: 1707x960 of 2560x1440
    case Settings::NvnDlssQuality::Balanced:
        return u16{0x3f15}; // 0.58203125
    case Settings::NvnDlssQuality::Performance:
        return u16{0x3f00}; // 0.5
    case Settings::NvnDlssQuality::UltraPerformance:
        return u16{0x3eaa}; // 0.33203125
    case Settings::NvnDlssQuality::Native:
        break;
    }
    return std::nullopt;
}

/// Whether the build has an entry (the quality setting can do something for it).
[[nodiscard]] inline bool Supports(std::string_view build_id) {
    return build_id == OotMainBuildId;
}

enum class Outcome { Applied, Native, OtherBuild, Mismatch };

struct Result {
    Outcome outcome{Outcome::Native};
    u32 branch_found{}; ///< the words found at the two sites (logged on a mismatch)
    u32 scale_found{};
};

/// Writes the fixed path and the quality's share into `image` (PatchNSO's image) when the build is
/// in the table and both sites hold what this table expects (shipped, or a mod's version of the
/// same instructions); otherwise leaves it untouched.
[[nodiscard]] inline Result Apply(std::vector<u8>& image, std::string_view build_id,
                                  Settings::NvnDlssQuality quality) {
    Result result;
    const std::optional<u16> high = FloatHigh(quality);
    if (!high) {
        return result;
    }
    if (!Supports(build_id)) {
        result.outcome = Outcome::OtherBuild;
        return result;
    }
    if (image.size() < ScaleOffset + sizeof(u32)) {
        result.outcome = Outcome::Mismatch;
        return result;
    }
    std::memcpy(&result.branch_found, image.data() + BranchOffset, sizeof(u32));
    std::memcpy(&result.scale_found, image.data() + ScaleOffset, sizeof(u32));
    if ((result.branch_found != BranchShipped && result.branch_found != BranchFixed) ||
        !IsScaleInstruction(result.scale_found)) {
        result.outcome = Outcome::Mismatch;
        return result;
    }
    const u32 branch = BranchFixed;
    const u32 scale = ScaleInstruction(*high);
    std::memcpy(image.data() + BranchOffset, &branch, sizeof(u32));
    std::memcpy(image.data() + ScaleOffset, &scale, sizeof(u32));
    result.outcome = Outcome::Applied;
    return result;
}

} // namespace FileSys::NvnDlssScale
