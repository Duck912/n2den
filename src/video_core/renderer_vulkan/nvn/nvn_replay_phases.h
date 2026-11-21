// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>

#include "common/common_types.h"

#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

/// Replay thread phase timers (EDEN_NVN_REPLAY_PHASES=1): rdtsc ticks and call counts per phase of
/// the replay's recording, plus event counts, kept in plain thread-local accumulators, added to a
/// process-wide total once per replay step (Flush) and logged every 10 s per presented frame
/// (Log). Off: one predictable branch per site and nothing else.
namespace Vulkan::Nvn::ReplayPhases {

enum class Phase : u8 {
    Record,         ///< Recorder::RecordList as a whole (the recording of every op)
    OpDraw,         ///< draw ops (Draw, DrawIndexed, DrawMeshTasks) through the draw context
    OpDispatch,     ///< Dispatch through the draw context
    OpClear,        ///< ClearColor / ClearDepthStencil through the draw context
    OpState,        ///< every other draw-context op (binds, state, barriers, inline updates)
    OpRecorder,     ///< ops the recorder handles itself (copies, syncs, reports, conditions)
    PassBreak,      ///< EndDrawPass in front of an op that cannot run inside a pass
    SubmitEarly,    ///< MaybeSubmitEarly checks (and the early submissions they make)
    Submit,         ///< SubmitSegments (end + vkQueueSubmit of the pieces)
    ResolvePipeline,
    BuildDescriptors,
    VertexIndex,    ///< vertex layout + vertex / index buffer resolution
    OutsidePass,    ///< PrepareOutsidePass + BeginPass (with the condition suspend / resume)
    VtxCov,         ///< staged vertex / vtxcov serving after the pass began
    PipelineBind,   ///< pipeline bind, dynamic fixed state, vertex input, dynamic state
    FinishImages,
    TexturePool,    ///< FinishTexturePool + the set 1 bind
    PushSetZero,    ///< push constants + push descriptors (or set allocation) of set 0
    VertexBind,     ///< vertex / index buffer binds
    DrawCall,       ///< the vkCmdDraw* / vkCmdDrawMeshTasks call
    EndPass,        ///< DrawContext::EndPass
    BeginPassOnly,  ///< DrawContext::BeginPass (inside OutsidePass for draws)
    FlushInline,    ///< DrawContext::FlushInlineUpdates
    MirrorCheck,    ///< Recorder::MirrorNeedsUpload
    MirrorUpload,   ///< Recorder::UploadRange
    // BuildDescriptors split (per descriptor binding of the layout):
    DescPre,        ///< raw table / alias table / storage-vram preamble
    DescUniform,    ///< uniform buffer bindings of a stage slot (buffer_info)
    DescBank,       ///< driver bank, handle bank and program constant bindings
    DescStorage,    ///< storage buffer bindings
    DescImage,      ///< texture and storage image bindings
    DescPoolPlan,   ///< PlanTexturePool
    // PrepareOutsidePass split:
    OutReport,      ///< pending counter report check
    OutImages,      ///< per image: needs-contents / layout decision
    OutRanges,      ///< per range: mirror needs-upload check (contains mirror-check)
    OutShadow,      ///< ClassifyShadows
    OutFeedback,    ///< PruneFreshFeedback
    OutClosed,      ///< pass closed: feedback copies, contents, transitions, uploads, barriers
    OutOpen,        ///< pass open: touched notes, shadows
    OutWrites,      ///< pool write notes, raw write notes
    // Context::Impl::StorageNeedsCommands split:
    NeedsAll,       ///< StorageNeedsCommands as a whole
    NeedsDetect,    ///< DetectGuestChange (guest signature checks)
    NeedsScan,      ///< the overlap scan of the pool's storages (write_gen moved)
    // SetRenderCondition split (kind 0, a GPU condition):
    CondResolve,    ///< the report pair's two ResolvePoolRange
    CondFast,       ///< predicate computed ahead: lookup and checks
    CondCompute,    ///< not computed ahead: pass break, uploads, PrefetchConditions
    CondWalk,       ///< PrefetchConditions' walk of the records ahead
    // ReportCounter split:
    RepResolve,     ///< ResolvePoolRange
    RepPass,        ///< deferred-zero check, ReserveCounterEvents, pass break
    RepBody,        ///< the per-counter-type recording (queries, copies, fills)
    RepFollows,     ///< SamplesReportFollows
    RepNote,        ///< NotePoolWrite
    RepTail,        ///< pool keepalive + WriteTimestamp
    // BeginPass split (passes actually begun):
    PassPrep,       ///< attachments, contents, transitions, mirrors, vertex staging
    PassFramebuffer,///< views, render pass, framebuffer creation, vkCmdBeginRenderPass
    PipeLookup,     ///< ResolvePipeline's pipeline cache lookup (GetGraphics)
    CondBreak,      ///< CondCompute's pass break
    CondRecord,     ///< PrefetchConditions after the walk (slots, copies, dispatch, entries)
    WriteDraw,      ///< NotePoolWrite: DrawContext::OnPoolWrite
    WriteCond,      ///< NotePoolWrite: condition prefetch invalidation
    WriteRest,      ///< NotePoolWrite: the rest (sparse, gpu writes, snapshots, mirrors)
    Count,
};

enum class Counter : u8 {
    Draws,           ///< PrepareDraw calls
    ConditionalDraws,///< of them under a render condition (RenderState::Conditional)
    Passes,          ///< render passes begun
    Framebuffers,    ///< VkFramebuffers created
    PipelineReuses,  ///< ResolvePipeline fast path
    DriverBanks,     ///< graphics driver bank (c0) rebuilds
    HandleBanks,     ///< handle bank (c2) rebuilds
    PushDescriptors, ///< vkCmdPushDescriptorSetKHR calls
    DescriptorElided,///< set 0 pushes skipped (the pass had them)
    PoolSetBinds,    ///< texture pool set (set 1) binds
    MirrorPages,     ///< mirror pages in the ranges MirrorNeedsUpload / UploadRange walked
    MirrorCalls,     ///< MirrorNeedsUpload / UploadRange calls on mirror pools
    PipelineMemoHits,///< ResolvePipeline found the pipeline in its memo
    PipelineLookups, ///< ResolvePipeline asked the pipeline cache
    Count,
};

struct Accumulator {
    std::array<u64, static_cast<size_t>(Phase::Count)> ticks{};
    std::array<u64, static_cast<size_t>(Phase::Count)> calls{};
    std::array<u64, static_cast<size_t>(Counter::Count)> counts{};
    /// OpRecorder split by op number (VC::Op).
    static constexpr size_t OPS = 96;
    std::array<u64, OPS> op_ticks{};
    std::array<u64, OPS> op_calls{};
};

namespace Detail {
[[nodiscard]] bool ReadEnabled() noexcept;
inline const bool enabled = ReadEnabled();
} // namespace Detail

/// EDEN_NVN_REPLAY_PHASES=1, read once at start-up.
[[nodiscard]] inline bool Enabled() noexcept {
    return Detail::enabled;
}

/// The calling thread's accumulator (the replay thread's).
[[nodiscard]] Accumulator& Local() noexcept;

/// Adds the calling thread's accumulator to the process-wide total and clears it.
void Flush();

/// EDEN_NX2_NVN_TLRING: one ring record per phase with the calling thread's ticks since its
/// previous call (called at each vkQueueSubmit of the replay).
void RingSnapshot(u64 seq);

/// Logs `[nvn] replay phases: ...` per frame over the presents since the last line, at most every
/// 10 s (any thread).
void Log(u64 presents);

[[nodiscard]] inline u64 Now() noexcept {
    return __rdtsc();
}

inline void Count(Counter counter, u64 n = 1) noexcept {
    if (Enabled()) [[unlikely]] {
        Local().counts[static_cast<size_t>(counter)] += n;
    }
}

/// Times its lifetime into `phase` when the timers are on.
class Scope {
public:
    explicit Scope(Phase phase_) noexcept : phase{phase_}, start{Enabled() ? Now() : 0} {}
    ~Scope() {
        Stop();
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

    /// Ends the timing early (idempotent).
    void Stop() noexcept {
        if (start != 0) [[unlikely]] {
            Accumulator& local = Local();
            local.ticks[static_cast<size_t>(phase)] += Now() - start;
            ++local.calls[static_cast<size_t>(phase)];
            start = 0;
        }
    }

private:
    Phase phase;
    u64 start;
};

/// Times its lifetime into OpRecorder and into the op's own slot.
class OpScope {
public:
    explicit OpScope(u32 op_) noexcept : op{op_}, start{Enabled() ? Now() : 0} {}
    ~OpScope() {
        if (start != 0) [[unlikely]] {
            Accumulator& local = Local();
            const u64 ticks = Now() - start;
            local.ticks[static_cast<size_t>(Phase::OpRecorder)] += ticks;
            ++local.calls[static_cast<size_t>(Phase::OpRecorder)];
            const size_t slot = op < Accumulator::OPS ? op : Accumulator::OPS - 1;
            local.op_ticks[slot] += ticks;
            ++local.op_calls[slot];
        }
    }
    OpScope(const OpScope&) = delete;
    OpScope& operator=(const OpScope&) = delete;

private:
    u32 op;
    u64 start;
};

} // namespace Vulkan::Nvn::ReplayPhases
