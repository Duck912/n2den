// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"

/// What last wrote a replaced upscaling network's output under NVN-HLE, for the status bar.
/// Written by the sink's replay thread, read by the frontend; None until a network is replaced
/// (the game's own network runs) and after Reset.
namespace VideoCommon::Nvn::UpscalerStatus {

enum class Stage : u8 {
    None,
    Copy,
    Fsr1,
    Dlaa,
    DlssUltraQuality,
    DlssQuality,
    DlssBalanced,
    DlssPerformance,
    DlssUltraPerformance,
    GameNetwork, ///< the game's own upscaling network ran as shipped
};

/// Which upscale wrote: a replaced upscaling network (NDSR, the sink's stand-in) or the game's own
/// nvnCommandBufferEvaluateDlss. A frame may run both (XC2: the game's DLSS to 1080p, then NDSR
/// to 4K), so each has its own slot.
enum class Slot : u8 {
    Network,
    Game,
};

struct Value {
    Stage stage{Stage::None};
    /// DLSS / DLAA was given jitter and motion vectors (always false for the other stages).
    bool temporal{};
    Stage game_stage{Stage::None};
    bool game_temporal{};
};

namespace Detail {
inline std::atomic<u32> packed{0};
} // namespace Detail

inline void Note(Stage stage, bool temporal = false, Slot slot = Slot::Network) {
    const u32 shift = slot == Slot::Game ? 16u : 0u;
    const u32 mask = 0x1ffu << shift;
    const u32 value = (static_cast<u32>(stage) | (temporal ? 0x100u : 0u)) << shift;
    u32 old = Detail::packed.load(std::memory_order_relaxed);
    while (!Detail::packed.compare_exchange_weak(old, (old & ~mask) | value,
                                                 std::memory_order_relaxed)) {
    }
}

inline void Reset() {
    Detail::packed.store(0, std::memory_order_relaxed);
}

[[nodiscard]] inline Value Read() {
    const u32 value = Detail::packed.load(std::memory_order_relaxed);
    return Value{static_cast<Stage>(value & 0xffu), (value & 0x100u) != 0,
                 static_cast<Stage>((value >> 16) & 0xffu), (value & 0x1000000u) != 0};
}

} // namespace VideoCommon::Nvn::UpscalerStatus
