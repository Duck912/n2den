// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <string_view>
#include <nlohmann/json_fwd.hpp>
#include "common/common_types.h"

// Why programs leave the native Ampere front end, and how much work is lost for it.
// Always on: one table entry per rejected
// program and one relaxed atomic per lost draw. Logged at shutdown and returned by the debug
// server's `shaders` method.
namespace VideoCore::NativeCensus {

enum class Site : u8 {
    Native,    ///< The native front end refused the program (auto mode lowers it instead)
    Lowering,  ///< The Maxwell lowering refused it as well: the program is dropped
    Build,     ///< Rejected elsewhere while reading or building it (stage, environment, translation)
    Collected, ///< EDEN_NATIVE_CENSUS=1: every hole of a refused program, not only the first
    Count,
};

enum class Drop : u8 {
    GraphicsRejected,    ///< Draws lost because one of their programs is rejected
    ComputeRejected,     ///< Dispatches lost the same way
    PipelineTranslating, ///< Draws skipped: the pipeline missed the wait budget in translation
    PipelineBuilding,    ///< ... or in the driver compile
    Count,
};

inline constexpr u32 NoOpcode = ~0U;
inline constexpr u32 NoStage = ~0U;

/// Records one rejection. `program` identifies the program (native key, or address for Build);
/// repeats of the same program at the same site and reason count as events, not programs.
/// `thrown`: the caller throws right after, so NoteBuildFailure must not count it again.
void NoteReject(Site site, u64 program, u32 stage, u32 opcode, std::string_view reason,
                bool thrown = false);

/// Build-site rejection seen where a program is first remembered as rejected. Skipped when this
/// thread's last NoteReject was `thrown` (it is the same exception).
void NoteBuildFailure(u64 program, std::string_view what);

void NoteDrop(Drop drop);

/// EDEN_NATIVE_CENSUS=1: collect every unsupported instruction of a refused program.
[[nodiscard]] bool CollectAll();

[[nodiscard]] nlohmann::json Snapshot();
void LogSummary();
/// Tests only.
void Reset();

} // namespace VideoCore::NativeCensus
