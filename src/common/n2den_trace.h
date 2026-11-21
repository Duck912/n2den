// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <string_view>

#include "common/common_types.h"
#include "common/logging.h"

/// [DEBUG] Counters shared between the HLE services (core) and GPU emulation (video_core) for
/// live diagnostics. Each is a plain counter; readers compare values over time.
namespace Common::N2denTrace {

/// Presents that returned to the game (incremented after the present's frame wait). With
/// EDEN_NX2_FRAMES_AHEAD=0 every GPU command processed between two increments belongs to one frame.
inline std::atomic<u32> present_count{0};

/// Frames GPU emulation presented, counted where the luma trace numbers them (with
/// EDEN_N2DEN_LUMA_TRACE). Draws made while it is N belong to luma frame N + 1, in any
/// frames-ahead mode.
inline std::atomic<u32> back_frames{0};

/// RasterizerVulkan::FlushQueryResults calls that had to wait for pending host query results,
/// and those whose wait ran out.
inline std::atomic<u64> query_flush_waits{0};
inline std::atomic<u64> query_flush_timeouts{0};

/// Draws the Vulkan pipeline cache skipped because their pipeline was still being translated or
/// built (PipelineCache::NoteSkippedDraw).
inline std::atomic<u64> pipeline_skips{0};

/// The last occlusion report a predicate macro read (slot address, count, timestamp), for the
/// draw trace. Only in order with the draws when the macros run on the GPU back thread
/// (EDEN_GPU_SPLIT=1 or 0).
inline std::atomic<u64> last_predicate_slot{0};
inline std::atomic<u32> last_predicate_count{0};
inline std::atomic<u64> last_predicate_stamp{0};

} // namespace Common::N2denTrace

namespace Common::N2denPresent {

/// The native present's frames-ahead mode for the running title (0, 1, or -1 for unlimited), set
/// at each present. While it is not 0, guest CPU reads of query report slots wait for reports
/// still on their way (VideoCommon::ReportReadWait).
inline std::atomic<int> frames_ahead{0};

} // namespace Common::N2denPresent

/// Occlusion predicates of native (NX2) titles: a native MME macro that read an occlusion count
/// (a samples-passed report) and then writes render enable False marks the write with
/// OcclusionFalseMarker. Maxwell3D::ProcessQueryCondition removes the mark again and notes that
/// the condition came from an occlusion count; ProcessDraw then still draws depth-tested draws
/// (the depth test hides them where they really are occluded). All on the thread that runs them.
namespace Common::N2denOcclusion {

inline constexpr u32 OcclusionFalseMarker = 0x80000000U;

/// The macro running on this thread read a samples-passed report.
inline thread_local bool macro_read_count = false;

/// The render-enable False now in effect on this thread came from an occlusion count.
inline thread_local bool condition_from_count = false;

/// EDEN_N2DEN_OCCLUSION_PREDICATES=honor keeps the game's occlusion culling (one-frame missing
/// objects in Ocarina of Time while the camera moves). Default: culled draws are drawn
/// depth-tested.
inline bool DrawCulled() {
    static const bool draw = [] {
        const char* const value = std::getenv("EDEN_N2DEN_OCCLUSION_PREDICATES");
        const bool honor = value != nullptr && std::string_view{value} == "honor";
        LOG_INFO(HW_GPU, "Occlusion predicates of native macros: {}",
                 honor ? "honored (EDEN_N2DEN_OCCLUSION_PREDICATES=honor)"
                       : "culled draws are still drawn when depth-tested");
        return !honor;
    }();
    return draw;
}

/// A native macro's register write: render enable False after the macro read an occlusion
/// count gets the marker. `mode_method`: the render_enable.mode register index.
inline u32 MarkCondition(u32 mode_method, u32 method, u32 value) {
    if (method == mode_method && value == 0 && macro_read_count && DrawCulled()) {
        return OcclusionFalseMarker;
    }
    return value;
}

} // namespace Common::N2denOcclusion
