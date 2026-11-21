// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "common/common_types.h"
#include "core/debugger/nx2_debug_server.h"
#include "hid_core/frontend/emulated_controller.h"

namespace Core {
class System;
}

namespace Core::Memory {
class Memory;
}

namespace Core::Nx2Debug {

class Server;
struct Endpoint;
struct ToolState;

/// Additional handlers and pending requests for one boot of the debug server.
class Tools {
public:
    Tools(System& system, std::string title, u64 program_id, std::filesystem::path directory);
    ~Tools();
    Handlers MakeHandlers() const;
    void Stop();
    void NoteLogFilter(std::string filter);

private:
    std::shared_ptr<ToolState> state;
};

/// A reject_shaders hash: exactly 16 hex digits, optionally after 0x. Throws std::runtime_error.
[[nodiscard]] u64 ParseShaderHash(std::string_view text);

/// search_memory's pattern: hex digit pairs, optionally after 0x, spaces allowed. Throws
/// std::runtime_error.
[[nodiscard]] std::vector<u8> ParseHexBytes(std::string_view text);

/// read_memory's formats: {"hex": "..."} or {"u32"|"u64"|"f32": [...]}; u64 values are hex strings
/// and non-finite floats are "nan", "inf" or "-inf". Throws std::runtime_error when the format is
/// unknown or the size is not a multiple of its element.
[[nodiscard]] nlohmann::json FormatMemory(std::span<const u8> bytes, std::string_view as);

/// Copies guest memory from a thread that is not a guest CPU core.
/// Returns the first address that is not mapped; out is then partly written.
[[nodiscard]] std::optional<u64> ReadGuestMemory(Memory::Memory& memory, u64 address,
                                                 std::span<u8> out);

struct GuestSearch {
    std::vector<u64> hits;
    /// Where to resume: the first address no match was looked for at; end when complete.
    u64 next;
    bool complete;
    /// Mapped bytes read.
    u64 scanned;
};

/// Finds the pattern in the mapped pages of [start, end), as ReadGuestMemory reads them. Stops at
/// max_hits or the deadline. Matches may span pages, but not unmapped gaps.
[[nodiscard]] GuestSearch SearchGuestMemory(Memory::Memory& memory, std::span<const u8> pattern,
                                            u64 start, u64 end, size_t max_hits,
                                            std::chrono::steady_clock::time_point deadline);

/// One step of input or input_sequence: what to hold, for how many presented frames.
struct InputStep {
    HID::InjectedInput input;
    u64 frames;
};

/// {buttons?, left_stick?, right_stick?, frames}. Sticks are [x, y] in -1..1. Throws
/// std::runtime_error on unknown keys, buttons or out-of-range values.
[[nodiscard]] InputStep ParseInputStep(const nlohmann::json& step);

/// The input of the step active at present count now, for steps starting at present count start;
/// nothing before start and from the end of the last step on.
[[nodiscard]] std::optional<HID::InjectedInput> InputAt(std::span<const InputStep> steps, u64 start,
                                                        u64 now);

/// sched_trace's recorder: guest context switches and presented frames, for a window of at most a
/// few seconds. Off, it costs the scheduler one relaxed load per switch.
namespace SchedTrace {

/// On while sched_trace records.
extern std::atomic<bool> enabled;

/// One context switch on a core. Thread ids: 0 is the core's idle thread.
struct Switch {
    s64 ticks;
    u32 out_id;
    u32 in_id;
};

/// kind 0: the present was counted (Core::g_nx2_present_count); 1: present pacing returned.
struct Present {
    s64 ticks;
    u64 count;
    u64 thread_id;
    u32 kind;
};

/// From KScheduler::SwitchThread on the core's own host thread (one writer per core): appends to the
/// core's preallocated ring without locks; drops and counts when it is full.
void RecordSwitch(s32 core, s64 ticks, u64 out_id, u64 in_id);
/// From the present path (any thread); takes a small mutex, only while recording.
void RecordPresent(u32 kind, s64 ticks, u64 count, u64 thread_id);

/// Span kinds (bit 1 << SpanKind) sched_trace records with its `spans` parameter: what a guest
/// thread did while it was on a core. Zero (off), PhysicalCore::RunThread pays one relaxed load
/// per loop.
extern std::atomic<u32> spans_enabled;

enum class SpanKind : u16 {
    Jit = 0, ///< inside ArmInterface::RunThread (JIT code, its dispatcher and memory callbacks)
    Hle = 1, ///< an nvn-hle bridge call (Bridge::Handle); id = thunk slot
    Svc = 2, ///< a kernel SVC (Svc::Call), including time blocked in it; id = SVC number
};

/// One span of a guest thread. args: SVC x0 and x1 (WaitSynchronization: the first handle, then
/// the handle count); zero for other kinds.
struct Span {
    s64 start;
    s64 end;
    u64 arg0;
    u64 arg1;
    u32 thread_id;
    SpanKind kind;
    u16 id;
};

/// From the core's own host thread at the span's end (one writer per core ring).
void RecordSpan(s32 core, const Span& span);

/// Host-side spans of the renderer's own threads (no guest thread), recorded while sched_trace
/// records, in steady_clock time; the tool maps them to CoreTiming ticks with the presents.
enum class HostSpanKind : u32 {
    ReplayIdle = 0,    ///< nvn replay thread waiting for work
    ReplayStep = 1,    ///< a submission's replay step; arg = seq
    ReplayPresent = 2, ///< present blit on the replay thread; arg = seq
    GpuWait = 3,       ///< completion thread waiting for a submission's semaphore; arg = seq
    JitCompile = 4,    ///< a guest core's JIT producing a block; arg = kind (0 translated, 1 cached
                       ///< x64 code, 2 cached IR) | guest core << 8
    PipelineCompile = 5, ///< nvn pipeline compile; arg = 0 on the replay thread, 1 async worker,
                         ///< 2 disk preload (| 0x10 compute)
    FsRead = 6,        ///< fs service IFile/IStorage read; arg = bytes
};

struct HostSpan {
    s64 start_ns;
    s64 end_ns;
    u64 arg;
    HostSpanKind kind;
};

/// Any thread; takes a small mutex, only while recording.
void RecordHostSpan(HostSpanKind kind, std::chrono::steady_clock::time_point start,
                    std::chrono::steady_clock::time_point end, u64 arg);

/// Records its lifetime as a host span when sched_trace was recording at construction.
class HostSpanScope {
public:
    explicit HostSpanScope(HostSpanKind kind_, u64 arg_ = 0)
        : kind{kind_}, arg{arg_}, active{enabled.load(std::memory_order_relaxed)} {
        if (active) {
            start = std::chrono::steady_clock::now();
        }
    }
    ~HostSpanScope() {
        if (active) {
            RecordHostSpan(kind, start, std::chrono::steady_clock::now(), arg);
        }
    }
    HostSpanScope(const HostSpanScope&) = delete;
    HostSpanScope& operator=(const HostSpanScope&) = delete;

private:
    HostSpanKind kind;
    u64 arg;
    bool active;
    std::chrono::steady_clock::time_point start{};
};

} // namespace SchedTrace

} // namespace Core::Nx2Debug
