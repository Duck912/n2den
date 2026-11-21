// SPDX-FileCopyrightText: Copyright 2026 n2den Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>

#include "common/common_types.h"

/// Frame-time statistics for the Switch 2 paths (legacy GPU emulation and NVN HLE).
///
/// Off unless `EDEN_NX2_FRAME_STATS` is set (anything but `0`); then every guest present calls
/// `OnPresent` and the subsystems add the time they spent stalling the frame with `AddStall` or
/// `Scope`. A value other than `1` is also the path of a CSV with one line per present
/// (`present,t_ms,dt_us,<cause us...>,<counter...>`). Every 600 presents one log line gives the
/// present-to-present percentiles, the counts over 16.7/33/50/100 ms and, for the frames over 33 ms, which causes
/// were active and for how long.
namespace Common::FrameStats {

enum class Cause : u32 {
    GuestAcquireWait, ///< HLE: guest blocked in AcquireTexture
    GuestSyncWait,    ///< HLE: guest blocked in SyncWait / QueueFinish / EventWait
    GuestPace,        ///< speed limiter / vblank pacing on the present thread
    LegacyFence,      ///< legacy: present thread waiting for the GPU thread's fence
    LegacyComposite,  ///< legacy: composite on the present thread
    PipelineCompile,  ///< replay: pipeline translation/creation on the critical path
    PoolBacking,      ///< replay: host import / device-local backing
    RawTableRebuild,  ///< raw global address table: replay rebuild; legacy GPU build of the guest-memory
                      ///< ranges, including the wait for their imports (RawGuestMemory::Ranges)
    TextureUpload,    ///< replay: texture upload, unswizzle, ASTC decode
    CoherencyCheck,   ///< replay: guest-write detection (signatures, hashes)
    HeldBack,         ///< replay: submission parked on a missing fence
    ReplayExecute,    ///< replay: ExecuteSubmission (recording) busy time
    QueueSubmit,      ///< replay: vkQueueSubmit
    PresentBlit,      ///< replay: present recording, acquire, submit
    PresentIdle,      ///< replay: waiting for the next frame's work
    PresentAcquire,   ///< replay: AcquireFrame (waiting for a free present frame); inside present_blit
    PresentHandOver,  ///< replay: PresentFrame (swapchain acquire, blit submit, vkQueuePresent when
                      ///< not on the present thread); inside present_blit
    // Work time summed over every thread that did it, on the critical path or not:
    JitTranslate,     ///< guest JIT: block translated, optimized, emitted (and stored) on a guest core
    JitNative,        ///< guest JIT: block placed from cached x64 code (load, hash check, relocation)
    JitIr,            ///< guest JIT: block from cached IR, emitted again
    PipeTranslate,    ///< pipeline compile (any thread): guest shaders to SPIR-V
    PipeCreate,       ///< pipeline compile (any thread): vkCreate*Pipelines
    PipePreload,      ///< disk cache preload workers: pipelines rebuilt from cached SPIR-V
    FsRead,           ///< fs service: IFile/IStorage reads (guest asset loading)
    Count,
};

constexpr std::array<const char*, static_cast<size_t>(Cause::Count)> CauseNames{
    "guest_acquire", "guest_syncwait", "guest_pace",     "legacy_fence", "legacy_composite",
    "pipeline",      "pool_backing",   "raw_table",      "tex_upload",   "coherency",
    "held_back",     "replay_exec",    "queue_submit",   "present_blit", "present_idle",
    "present_acquire", "present_handover", "jit_translate", "jit_native",  "jit_ir",
    "pipe_translate", "pipe_create",    "pipe_preload",  "fs_read",
};

/// Event counts per frame, written after the causes in the CSV.
enum class Counter : u32 {
    JitTranslate, ///< blocks translated
    JitNative,    ///< blocks placed from cached x64 code
    JitIr,        ///< blocks emitted from cached IR
    PipeCompile,  ///< pipelines compiled (translated and created), any thread
    PipeAsync,    ///< of which on the asynchronous workers
    DrawSkipped,  ///< draws/dispatches skipped while their pipeline compiles asynchronously
    PipePreload,  ///< pipelines rebuilt by the disk preload
    FsReadKiB,    ///< KiB read through IFile/IStorage
    Count,
};

constexpr std::array<const char*, static_cast<size_t>(Counter::Count)> CounterNames{
    "n_jit_translate", "n_jit_native", "n_jit_ir",       "n_pipe_compile",
    "n_pipe_async",    "n_draw_skipped", "n_pipe_preload", "fs_read_kib",
};

namespace Detail {
extern const bool g_enabled;
extern std::array<std::atomic<u64>, static_cast<size_t>(Cause::Count)> g_stall_ns;
extern std::array<std::atomic<u64>, static_cast<size_t>(Counter::Count)> g_counts;
} // namespace Detail

[[nodiscard]] inline bool Enabled() {
    return Detail::g_enabled;
}

inline void AddStall(Cause cause, u64 ns) {
    if (Detail::g_enabled) {
        Detail::g_stall_ns[static_cast<size_t>(cause)].fetch_add(ns, std::memory_order_relaxed);
    }
}

inline void AddCount(Counter counter, u64 n = 1) {
    if (Detail::g_enabled) {
        Detail::g_counts[static_cast<size_t>(counter)].fetch_add(n, std::memory_order_relaxed);
    }
}

/// Adds the lifetime of the object to `cause`.
class Scope {
public:
    explicit Scope(Cause cause_) : cause{cause_}, active{Detail::g_enabled} {
        if (active) {
            start = std::chrono::steady_clock::now();
        }
    }
    ~Scope() {
        if (active) {
            AddStall(cause, static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                 std::chrono::steady_clock::now() - start)
                                                 .count()));
        }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Cause cause;
    bool active;
    std::chrono::steady_clock::time_point start{};
};

/// Called by the guest present on either path (the frame boundary). No-op when disabled.
void OnPresent(std::chrono::steady_clock::time_point now);
/// Guest presents observed by this instance, for request/event logs.
[[nodiscard]] u64 PresentCount();

/// The `stats` diagnostic request: logs the current window (without resetting it), the stall causes
/// since start and a frame-time histogram, and writes them to `<directory>/frame-stats.txt`. No-op
/// when the statistics are off.
void DumpNow(const std::filesystem::path& directory);

} // namespace Common::FrameStats
