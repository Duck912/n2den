// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Built-in code patches for The Legend of Zelda: Ocarina of Time (Switch 2), main executable
// E0F3CCFA... (the only build these fit; any other build or title is left alone):
//
// - Dynamic FPS (setting oot_dynamic_fps): the game advances time by a fixed 1/60 s per frame, so
//   below 60 fps it runs in slow motion. One instruction of the time manager's step
//   (main+0x6e6cc4, `fdiv s8, s1, s0`) becomes a call into a routine in the zero padding after
//   .text (main+0x3c46200, mapped with the text segment) that re-runs the divide, then measures
//   the real frame time with the guest counter (8-frame average, capped at 1/20 s) and uses it
//   when it is more than 3 % over the game's own step. At 60 fps the step is the game's. Its state
//   lives past the end of .bss (main+0x4e9fe00, zeroed, inside the module's last page). Same bytes
//   as the "Dynamic FPS" mod (IPS records 0x6e6dc4 and 0x3c46300).
// - Max Resolution (setting oot_max_resolution): the game sizes its 3D scene by measured GPU time
//   (dynamic resolution), which under emulation looks slow, so it sits at its floor (854x480 of
//   1440p). main+0x805bd0, `tbz w8, #0`, becomes `b`: the game's own fixed path, share 1.0. Same
//   bytes as the "Max Resolution" mod (IPS record 0x805cd0); the DLSS scene size
//   (nvn_dlss_scale.h) writes the same branch plus a share.
//
// Applied before the mod folders, so a mod folder patching the same code (another variant) wins.
// Each patch checks the code it replaces and is skipped, with a log line, on anything else.

#pragma once

#include <array>
#include <cstring>
#include <string_view>
#include <vector>

#include "common/common_types.h"

namespace FileSys::OotBuiltinMods {

/// Ocarina of Time (Switch 2)'s main executable, as PatchNSO names build IDs (trailing zeros cut).
constexpr std::string_view MainBuildId = "E0F3CCFAF50065AE4EA01EB5AEB13B1A45DA3348";

/// Offsets in PatchNSO's image (the 0x100-byte NSO header, then main): main + 0x100.
constexpr std::size_t HookOffset = 0x6e6dc4;
constexpr u32 HookShipped = 0x1e201828; ///< fdiv s8, s1, s0
constexpr u32 HookPatched = 0x94d57d4f; ///< bl main+0x3c46200
constexpr std::size_t CaveOffset = 0x3c46300;
constexpr std::array<u32, 30> Cave{
    0x1e201828, 0xb00092ca, 0x9138014a, 0xd53be02b, 0xf940014c, 0xf900014b,
    0xb40002cc, 0xcb0c016b, 0xd53be00d, 0xd280028e, 0x9ace09ae, 0xeb0e017f,
    0x9a8b81cb, 0xf940054c, 0xb500004c, 0xaa0b03ec, 0xcb0c016e, 0x8b8e0d8c,
    0xf900054c, 0x9e230182, 0x9e2301a3, 0x1e231842, 0x1e210842, 0x1c0000c3,
    0x1e230903, 0x1e232040, 0x5400004d, 0x1e204048, 0xd65f03c0, 0x3f83d70a,
};

constexpr std::size_t ResolutionOffset = 0x805cd0;
constexpr u32 ResolutionShipped = 0x360041e8; ///< tbz w8, #0, +0x83c
constexpr u32 ResolutionFixed = 0x1400020f;   ///< b +0x83c

enum class Outcome { Applied, AlreadyApplied, Mismatch };

[[nodiscard]] inline u32 ReadWord(const std::vector<u8>& image, std::size_t offset) {
    u32 word{};
    std::memcpy(&word, image.data() + offset, sizeof(word));
    return word;
}

inline void WriteWord(std::vector<u8>& image, std::size_t offset, u32 word) {
    std::memcpy(image.data() + offset, &word, sizeof(word));
}

/// Dynamic FPS: the hook must hold the shipped divide and the cave must be empty padding (or both
/// already hold this patch, from a mod folder applied earlier).
[[nodiscard]] inline Outcome ApplyDynamicFps(std::vector<u8>& image) {
    if (image.size() < CaveOffset + sizeof(u32) * Cave.size()) {
        return Outcome::Mismatch;
    }
    const u32 hook = ReadWord(image, HookOffset);
    bool cave_empty = true;
    bool cave_ours = true;
    for (std::size_t i = 0; i < Cave.size(); ++i) {
        const u32 word = ReadWord(image, CaveOffset + i * sizeof(u32));
        cave_empty = cave_empty && word == 0;
        cave_ours = cave_ours && word == Cave[i];
    }
    if (hook == HookPatched && cave_ours) {
        return Outcome::AlreadyApplied;
    }
    if (hook != HookShipped || !cave_empty) {
        return Outcome::Mismatch;
    }
    for (std::size_t i = 0; i < Cave.size(); ++i) {
        WriteWord(image, CaveOffset + i * sizeof(u32), Cave[i]);
    }
    WriteWord(image, HookOffset, HookPatched);
    return Outcome::Applied;
}

/// Max Resolution: the dynamic-resolution flag test becomes a branch to the fixed path.
[[nodiscard]] inline Outcome ApplyMaxResolution(std::vector<u8>& image) {
    if (image.size() < ResolutionOffset + sizeof(u32)) {
        return Outcome::Mismatch;
    }
    const u32 word = ReadWord(image, ResolutionOffset);
    if (word == ResolutionFixed) {
        return Outcome::AlreadyApplied;
    }
    if (word != ResolutionShipped) {
        return Outcome::Mismatch;
    }
    WriteWord(image, ResolutionOffset, ResolutionFixed);
    return Outcome::Applied;
}

} // namespace FileSys::OotBuiltinMods
