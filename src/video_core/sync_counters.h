// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"

// [DEBUG] EDEN_NX2_SYNC_COUNTERS=1: how often the back end synchronizes (guest wait-for-idle,
// render pass ends by cause, pipeline barriers, uploads that cannot move to the upload command
// buffer, grouped-storage copies, submits) and how much CPU-side work is redundant (buffer sync
// memo, CPU-modified marks that change no bit, Maxwell3D register writes of the current value).
// One INFO line group about every second. Off, every probe is one load of a static bool.
namespace VideoCommon::SyncCounters {

extern const bool enabled;

[[nodiscard]] inline bool Enabled() {
    return enabled;
}

enum Counter : u32 {
    WfiInPass,          ///< RasterizerVulkan::WaitForIdle with a render pass active
    WfiOutsidePass,     ///< ... without one
    FragmentBarriers,   ///< RasterizerVulkan::FragmentBarrier
    WfiMethods,         ///< Maxwell3D wait_for_idle method (0x110)
    RenderPassEnds,     ///< Scheduler::EndRenderPass that ended an active pass
    RenderPassBegins,
    Barriers,           ///< vkCmdPipelineBarrier(2) recorded (worker thread)
    XfbBarriers,        ///< Transform-feedback barriers at render pass ends
    ReorderedUploads,   ///< Stream-buffer uploads recorded into the upload command buffer
    InlineUploads,      ///< Stream-buffer uploads that could not be reordered (main cmdbuf)
    InlineUploadBytes,
    OtherCopies,        ///< Other buffer copies recorded in the main command buffer
    OtherCopyBytes,
    GatherCopies,       ///< Grouped storage: copies into the gathered buffer
    GatherBatches,
    ScatterCopies,      ///< Grouped storage: copies back (FlushStorageWrites)
    ScatterBatches,
    Submits,            ///< Scheduler::SubmitExecution
    Finishes,           ///< Scheduler::Finish (submit and wait)
    PeriodicFlushes,    ///< RasterizerVulkan::FlushWork submits (every N draws)
    SyncMemoHits,       ///< BufferCache::SynchronizeBuffer memo
    SyncMemoMisses,
    CpuMarkNoChange,    ///< MarkRegionAsCpuModified calls that set no new CPU bit
    CpuMarkChanged,     ///< ... that set at least one
    RegWrites,          ///< Maxwell3D::ProcessDirtyRegisters
    RegWritesSame,      ///< ... of the value the register already holds
    Frames,             ///< RasterizerVulkan::TickFrame
    CounterCount,
};

/// Why a render pass ends: set by the code that asks for it (the outermost tag wins).
enum Cause : u32 {
    CauseOther,          ///< Untagged: tallied by return address
    CauseWfi,
    CauseFragmentBarrier,
    CauseFramebuffer,    ///< RequestRenderpass with another framebuffer
    CauseSubmit,         ///< SubmitExecution (Flush, Finish)
    CauseDeferredClear,
    CauseCopyBarrier,    ///< BufferCacheRuntime::Pre/PostCopyBarrier (outside grouped storage)
    CauseBufferCopy,     ///< BufferCacheRuntime::CopyBuffer in the main command buffer
    CauseInlineUpload,   ///< ... a stream-buffer upload that could not be reordered
    CauseBufferClear,
    CauseGather,         ///< Grouped storage gather
    CauseScatter,        ///< Grouped storage scatter (FlushStorageWrites)
    CauseCompute,        ///< RasterizerVulkan::DispatchCompute
    CauseXfbPipeline,    ///< GraphicsPipeline configure with transform feedback enabled
    CauseCount,
};

inline thread_local Cause cause = CauseOther;

extern std::atomic<u64> counters[CounterCount];

inline void Add(Counter counter, u64 value = 1) {
    if (enabled) {
        counters[counter].fetch_add(value, std::memory_order_relaxed);
    }
}

/// Tags render pass ends in its scope with `cause`, unless an outer scope already did.
class ScopedCause {
public:
    explicit ScopedCause(Cause new_cause) {
        if (enabled && cause == CauseOther) {
            cause = new_cause;
            set = true;
        }
    }
    ~ScopedCause() {
        if (set) {
            cause = CauseOther;
        }
    }
    ScopedCause(const ScopedCause&) = delete;
    ScopedCause& operator=(const ScopedCause&) = delete;

private:
    bool set = false;
};

/// Counts an ended render pass under the current cause (and its callers when untagged).
void NoteRenderPassEnd();

/// Counts a Maxwell3D register write (back thread).
void NoteRegisterWrite(u32 method, bool same_value);

/// Logs and resets the counters about once a second (call once a frame, any one thread).
void MaybeReport();

} // namespace VideoCommon::SyncCounters
