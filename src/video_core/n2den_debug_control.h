// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <chrono>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <vector>
#include <nlohmann/json_fwd.hpp>
#include "common/common_types.h"

// Opt-in, bounded diagnostic inventory. Not a frame replay or a replacement renderer.
namespace VideoCore::N2DenDebug {
bool Enabled() noexcept;
void SetActive(bool active);
void RecordProgram(u64 key, u64 address, u32 stage, std::span<const u8> bytes);
u64 AssociateHash(u64 hash, u64 address);
std::vector<u64> NativeKeys(std::span<const u64> hashes);
void NoteResult(u64 key, std::string_view frontend, std::string_view error, u32 error_offset);
/// Why the native front end refused the program; kept when the lowering then takes it.
void NoteNativeReject(u64 key, std::string_view error, u32 error_offset);
bool RecordDraw(u64 pipeline, std::span<const u64> shaders, bool compute);
void FrameEnd();
nlohmann::json Status();
nlohmann::json Programs(size_t offset, size_t limit, bool failed_only);
nlohmann::json Draws(size_t offset, size_t limit);
nlohmann::json SetRejected(const std::vector<u64>& hashes, bool pipelines);
std::vector<u8> ProgramBytes(u64 key);

// Compute inspection: one merged snapshot per host shader hash (translation, bindings, cbufs).
void NoteCompute(u64 hash, const nlohmann::json& update);
nlohmann::json ComputeState(u64 id);
bool WatchedCompute(u64 hash);
nlohmann::json SetWatchedCompute(const std::vector<u64>& ids);
bool TakeComputeRebuild(u64 hash);
nlohmann::json RequestComputeRebuild(const std::vector<u64>& ids);
// Read-only GPU virtual memory access, provided by the renderer while it is alive.
using GpuReader = std::function<bool(u64 address, std::span<u8> out)>;
void SetGpuReader(GpuReader reader);
nlohmann::json ReadGpu(u64 address, size_t size);

// Per-program frontend override (native key -> native Ampere / Maxwell lowering), applied the next
// time the program is read: the shader cache drops its entries (TakeInvalidations) so the next draw
// or dispatch reads, captures and translates it again.
enum class FrontendOverride { None, Native, Lower };
FrontendOverride ForcedFrontend(u64 key);
nlohmann::json SetFrontend(const std::vector<u64>& keys, FrontendOverride mode);
nlohmann::json Frontends();
/// Host shader hashes whose shader cache entries must be dropped; empties the queue.
std::vector<u64> TakeInvalidations();

// Live IR / SPIR-V capture: programs named here are dumped the next time any pipeline using them is
// translated, with the live environment (header, constant buffers, BRX tables).
nlohmann::json RequestDump(const std::vector<u64>& keys);
/// Native key when `hash` has a pending dump request (not yet dumped for this hash).
std::optional<u64> PendingDump(u64 hash);
bool AnyPendingDump(std::span<const u64> hashes);
void NoteDump(u64 key, u64 hash, const nlohmann::json& entry);
nlohmann::json Dumps();

// QMD trace: every compute QMD event the game's PCAS signals cause (load, dependency counts,
// launch, zero-grid completion), with the program address and grid read from the descriptor.
// `qmd` holds the descriptor's 64 words when one was read, else null.
void RecordQmd(std::string_view event, u64 qmd_address, u64 address_space, s32 value,
               const u32* qmd);
nlohmann::json QmdTrace(size_t limit, bool last_frame_only, bool clear);

// Activity timeline: long waits on the GPU thread, guest channel semaphores, pipeline work, frames.
class ActivityScope {
public:
    ActivityScope(const char* label, u64 detail = 0);
    ~ActivityScope();
    ActivityScope(const ActivityScope&) = delete;
    ActivityScope& operator=(const ActivityScope&) = delete;

private:
    const char* label;
    u64 detail;
    bool armed;
    std::chrono::steady_clock::time_point start;
};
void NoteActivity(const char* label, u64 detail, std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point end);
void ChannelWaitBegin(u32 channel, u64 address, u64 required);
void ChannelWaitEnd(u32 channel, u64 address);
nlohmann::json Activity(double min_ms, size_t limit, bool clear);
} // namespace VideoCore::N2DenDebug
