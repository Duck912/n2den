// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Which of the NVN-HLE upscaling settings do something for which game, so the settings pages show
// a game only the options that work for it, and the DLSS-network code knows where the DLSS quality
// setting acts. The Upscaler (nvn_upscaler) applies to every Switch 2 game; the two below only to
// the games listed here.
//
// DLSS quality (nvn_dlss_quality): the share of the output the 3D scene is drawn at. A game sizes
// its scene in one of two ways:
// - CodePatch: by its own controller, which a per-build patch of its code fixes at the setting's
//   share (file_sys/nvn_dlss_scale.h; Ocarina of Time, one build).
// - RenderSizes: from the render sizes nvnDlssNetworkBuilderGetOptimalSettings (and the network's
//   Initialize/Resize) give it, which then report the setting's share
//   (core/hle/nvn/functions_dlss.cpp, GameRenderSizes): Fast Fusion.
// Fire Emblem: Fortune's Weave is not supported: it draws its scene at half the DLSS output
// whatever sizes it is given (checked 2026-11-24: told 2560x1440, it still drew 1280x720), so it
// would need a patch of its own code (where main+0x18df354 gets the input height from).
//
// Display resolution (display_resolution): the docked display a Switch 2 game is told it is on.
// It changes the game's output size only for the games that ask the system for the display
// (Fire Emblem through the graphics supervisor's mode query, Fast Fusion through
// GetDefaultDisplayResolution); Xenoblade Chronicles 2 and Ocarina of Time do not.

#pragma once

#include "common/common_types.h"

namespace FileSys::NvnGameUpscaling {

constexpr u64 OcarinaOfTime = 0x0400486024DEC000ULL;
constexpr u64 FireEmblemFortunesWeave = 0x04007EF01F8AE000ULL;
constexpr u64 FastFusion = 0x040041E0212AE000ULL;

enum class DlssQualityPath {
    None,        ///< the setting does nothing for the game
    CodePatch,   ///< file_sys/nvn_dlss_scale.h (for the builds it knows)
    RenderSizes, ///< the DLSS render sizes the game is given
};

/// A program and its sub-programs (the low 12 bits) count as one game.
[[nodiscard]] constexpr bool SameGame(u64 program_id, u64 game) {
    return (program_id & ~u64{0xfff}) == game;
}

[[nodiscard]] constexpr DlssQualityPath DlssQualityPathOf(u64 program_id) {
    if (SameGame(program_id, OcarinaOfTime)) {
        return DlssQualityPath::CodePatch;
    }
    if (SameGame(program_id, FastFusion)) {
        return DlssQualityPath::RenderSizes;
    }
    return DlssQualityPath::None;
}

[[nodiscard]] constexpr bool DlssQualityApplies(u64 program_id) {
    return DlssQualityPathOf(program_id) != DlssQualityPath::None;
}

[[nodiscard]] constexpr bool DisplayResolutionApplies(u64 program_id) {
    return SameGame(program_id, FireEmblemFortunesWeave) || SameGame(program_id, FastFusion);
}

} // namespace FileSys::NvnGameUpscaling
