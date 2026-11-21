// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <span>

#include "common/common_types.h"
#include "shader_recompiler/frontend/ampere/instruction.h"

namespace Shader::Ampere {

// Names a captured native program in logs, tools and caches: FNV-1a over its instruction words.
// On its own so users that only name programs (nvn-hle) do not pull in the Maxwell lowering.
inline u64 NativeProgramKey(std::span<const Instruction> source) {
    u64 hash = 14695981039346656037ULL;
    for (const auto& instruction : source) {
        for (u32 word : instruction.words) {
            for (u32 byte = 0; byte < 4; ++byte) hash = (hash ^ ((word >> (byte * 8)) & 255)) * 1099511628211ULL;
        }
    }
    return hash;
}

} // namespace Shader::Ampere
