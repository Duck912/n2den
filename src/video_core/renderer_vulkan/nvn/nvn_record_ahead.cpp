// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>

#include "common/assert.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/tl_ring.h"
#include "video_core/nvn/guest_memory.h"
#include "video_core/nvn/prerec_probe.h"
#include "video_core/nvn/record.h"
#include "video_core/renderer_vulkan/nvn/nvn_debug.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw_resources.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"
#include "video_core/renderer_vulkan/nvn/nvn_rescale.h"
#include "video_core/renderer_vulkan/nvn/nvn_texture_pool.h"
#include "video_core/renderer_vulkan/nvn/nvn_upload.h"

namespace Vulkan::Nvn {

namespace {

namespace VC = VideoCommon::Nvn;
namespace PreRec = VideoCommon::Nvn::PreRec;

/// Ahead-recording caps (a gate that runs into them stops recording ahead).
constexpr u64 HELD_CMDBUF_CAP = 64;
constexpr auto AHEAD_TIME_CAP = std::chrono::milliseconds{12};
/// A word becomes a gate word once the guest signalled it from the CPU this often, each change
/// satisfying every wait that had blocked on it.
constexpr u64 LEARN_CHANGES = 8;

bool EnvOn(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != 0 && value[0] != '0';
}

/// Guest memory wrapper (switch on only): counts replay-thread guest accesses while recording
/// ahead that no NoteRead covered (the choke point check of the read log).
class RaGuest final : public VC::GuestMemory {
public:
    explicit RaGuest(VC::GuestMemory* inner_) : inner{inner_} {}

    void Set(VC::GuestMemory* inner_) {
        inner.store(inner_, std::memory_order_release);
    }

    bool Read(u64 va, void* dst, size_t size) override {
        Check(va, size);
        VC::GuestMemory* const m = Get();
        return m != nullptr && m->Read(va, dst, size);
    }
    bool Write(u64 va, const void* src, size_t size) override {
        VC::GuestMemory* const m = Get();
        return m != nullptr && m->Write(va, src, size);
    }
    u8* Pointer(u64 va) override {
        Check(va, 1);
        VC::GuestMemory* const m = Get();
        return m != nullptr ? m->Pointer(va) : nullptr;
    }
    u8* MappingPointer(u64 va) override {
        // A host-mapping lookup, not a read of guest bytes (GuestMemory::MappingPointer).
        PreRec::RaMapping(va);
        VC::GuestMemory* const m = Get();
        return m != nullptr ? m->MappingPointer(va) : nullptr;
    }
    const u8* GuestViewPointer(u64 va) override {
        // Never dereferenced (GuestMemory::GuestViewPointer): a residency query, not a read.
        PreRec::RaMapping(va);
        VC::GuestMemory* const m = Get();
        return m != nullptr ? m->GuestViewPointer(va) : nullptr;
    }
    u64 Track(u64 va, u64 size) override {
        VC::GuestMemory* const m = Get();
        return m != nullptr ? m->Track(va, size) : 0;
    }
    void Untrack(u64 id) override {
        if (VC::GuestMemory* const m = Get()) {
            m->Untrack(id);
        }
    }
    void Arm(u64 id, u64 va, u64 size) override {
        if (VC::GuestMemory* const m = Get()) {
            m->Arm(id, va, size);
        }
    }
    void TakeWritten(u64 id, u64 va, u64 size, std::vector<std::pair<u64, u64>>& runs) override {
        if (VC::GuestMemory* const m = Get()) {
            m->TakeWritten(id, va, size, runs);
        }
    }
    bool AnyWritten(u64 id, u64 va, u64 size) override {
        VC::GuestMemory* const m = Get();
        return m != nullptr && m->AnyWritten(id, va, size);
    }
    std::shared_ptr<const std::atomic<u64>> WriteEpoch(u64 id) override {
        VC::GuestMemory* const m = Get();
        return m != nullptr ? m->WriteEpoch(id) : nullptr;
    }
    bool Stopped() const override {
        VC::GuestMemory* const m = inner.load(std::memory_order_acquire);
        return m != nullptr && m->Stopped();
    }
    bool SupportsAccessTraps() const override {
        VC::GuestMemory* const m = inner.load(std::memory_order_acquire);
        return m != nullptr && m->SupportsAccessTraps();
    }
    void SetAccessCallback(u64 id, std::function<void(u64 va, u64 size)> callback) override {
        if (VC::GuestMemory* const m = Get()) {
            m->SetAccessCallback(id, std::move(callback));
        }
    }
    void ArmAccessTrap(u64 id, u64 va, u64 size) override {
        if (VC::GuestMemory* const m = Get()) {
            m->ArmAccessTrap(id, va, size);
        }
    }
    void DisarmAccessTrap(u64 id, u64 va, u64 size) override {
        if (VC::GuestMemory* const m = Get()) {
            m->DisarmAccessTrap(id, va, size);
        }
    }
    u64 BeginImportGate(u64 va, u64 size) override {
        VC::GuestMemory* const m = Get();
        return m != nullptr ? m->BeginImportGate(va, size) : 0;
    }
    void EndImportGate(u64 token) override {
        if (VC::GuestMemory* const m = Get()) {
            m->EndImportGate(token);
        }
    }

private:
    VC::GuestMemory* Get() const {
        return inner.load(std::memory_order_acquire);
    }
    static void Check(u64 va, u64 size) {
        PreRec::RaCheck(va, size);
    }
    std::atomic<VC::GuestMemory*> inner;
};

/// Ops that touch resources but read no guest memory and take no decision from guest bytes:
/// counter reports and resets (GPU writes into pools, values written at completion) and barriers
/// (recorder-local ordering flags while no pass is open; no draw can open one ahead).
bool TouchingAhead(u16 op) {
    switch (static_cast<VC::Op>(op)) {
    case VC::Op::ReportCounter:
    case VC::Op::ResetCounter:
    case VC::Op::ReportValue:
        return true;
    default:
        return false;
    }
}

/// Ops a held queue records ahead in this build: state, sync, barriers and counter ops (no guest
/// memory read, no decision from guest bytes).
bool AllowedAhead(u16 op) {
    if (TouchingAhead(op) || op == static_cast<u16>(VC::Op::Barrier)) {
        return true;
    }
    return RaPureOp(op);
}

} // Anonymous namespace

bool RaPureOp(u16 op) noexcept {
    switch (static_cast<VC::Op>(op)) {
    case VC::Op::End:
    case VC::Op::CallList: // expanded by the recorder
    case VC::Op::FenceSync:
    case VC::Op::SignalEvent:
    case VC::Op::WaitSync:
    case VC::Op::WaitEvent:
    case VC::Op::SetViewports:
    case VC::Op::SetScissors:
    case VC::Op::SetDepthRanges:
    case VC::Op::BindProgram:
    case VC::Op::BindState:
    case VC::Op::BindVertexAttribs:
    case VC::Op::BindVertexStreams:
    case VC::Op::BindVertexBuffers:
    case VC::Op::BindUniformBuffers:
    case VC::Op::BindStorageBuffers:
    case VC::Op::BindTextures:
    case VC::Op::BindSeparateTextures:
    case VC::Op::BindSeparateSamplers:
    case VC::Op::BindImages:
    case VC::Op::SetStencil:
    case VC::Op::SetPolygonOffset:
    case VC::Op::SetTexturePool:
    case VC::Op::SetSamplerPool:
    case VC::Op::SetBlendColor:
    case VC::Op::SetDepthClamp:
    case VC::Op::SetAlphaRef:
    case VC::Op::PushConstants:
    case VC::Op::SetPrimitiveRestart:
        return true;
    default:
        return false;
    }
}

namespace {

bool RaRunsContain(const RaRuns& runs, u64 at) {
    for (const auto& [begin, end] : runs) {
        if (at >= begin && at < end) {
            return true;
        }
    }
    return false;
}

u64 NsSince(std::chrono::steady_clock::time_point start) {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count());
}

} // Anonymous namespace

void Context::Impl::RaInit() {
    ra.verify = EnvOn("EDEN_NX2_NVN_RECORD_AHEAD_VERIFY");
    ra.enabled = ra.verify || EnvOn("EDEN_NX2_NVN_RECORD_AHEAD");
    if (const char* inject = std::getenv("EDEN_NX2_NVN_RA_INJECT"); ra.verify && inject) {
        if (std::strcmp(inject, "mispredict") == 0) {
            ra.inject = 1;
            LOG_WARNING(Render_Vulkan, "[nvn] record-ahead fault injection: one release in 60 is "
                                       "deferred to the next command");
        } else if (std::strcmp(inject, "unlogged") == 0) {
            ra.inject = 2;
            LOG_WARNING(Render_Vulkan, "[nvn] record-ahead fault injection: one unlogged guest "
                                       "read every 4096 ops recorded ahead");
        } else if (std::strcmp(inject, "prologue") == 0) {
            ra.inject = 3;
            LOG_WARNING(Render_Vulkan, "[nvn] record-ahead fault injection: every release gets a "
                                       "prologue that copies one scratch page");
        } else if (std::strcmp(inject, "vramepoch") == 0) {
            ra.inject = 4;
            LOG_WARNING(Render_Vulkan, "[nvn] record-ahead fault injection: every logged VRAM "
                                       "pool is repaired at the release as if every page a "
                                       "decision used was written (real copy-ins of equal bytes)");
        }
    }
    ra.validate_only = ra.enabled && EnvOn("EDEN_NX2_NVN_RA_VALIDATE_ONLY");
    if (ra.validate_only) {
        LOG_WARNING(Render_Vulkan,
                    "[nvn] record-ahead VALIDATE-ONLY (EDEN_NX2_NVN_RA_VALIDATE_ONLY, measurement "
                    "only): draws and dispatches are recorded ahead, their Backing::Mirror and "
                    "texture reads are checked at the release; a change can show one wrong frame "
                    "before record-ahead latches off");
    }
    if (ra.enabled) {
        LOG_INFO(Render_Vulkan,
                 "[nvn] record-ahead ON (EDEN_NX2_NVN_RECORD_AHEAD{}): work behind a CPU-gated "
                 "event wait is recorded early and held until the guest's signal",
                 ra.verify ? "_VERIFY" : "");
    }
}

VC::GuestMemory* Context::Impl::RaWrapGuest(VC::GuestMemory* memory) {
    if (!ra.enabled) {
        return memory;
    }
    if (!ra_guest) {
        ra_guest = std::make_unique<RaGuest>(memory);
    } else {
        static_cast<RaGuest*>(ra_guest.get())->Set(memory);
    }
    return memory != nullptr ? ra_guest.get() : nullptr;
}

void Context::Impl::RaNoteBlocked(u64 address, u32 mode, u32 value) {
    if (!ra.enabled) {
        return;
    }
    RaWord& word = ra.words[address];
    if (word.pending.size() < 16) {
        word.pending.emplace_back(mode, value);
    }
}

bool Context::Impl::RaLearned(u64 address) const {
    const auto it = ra.words.find(address);
    return it != ra.words.end() && it->second.cpu_changes >= LEARN_CHANGES &&
           it->second.stable >= LEARN_CHANGES;
}

bool Context::Impl::RaMayPass(u64 address, u32 queue_index, bool snapshot) {
    if (!ra.enabled || ra.latched_off) {
        return false;
    }
    if (!RaLearned(address)) {
        return false;
    }
    if (ra.Active()) {
        // Joining: a queue that is not held yet becomes held, which needs recording ahead to be
        // live (a stopped gate only waits for its signal).
        if (ra.Stopped(queue_index)) {
            return false;
        }
        if (snapshot) {
            RaStopNow(RaStop::Snapshot);
            return false;
        }
        return true;
    }
    // Frame-level preconditions (design section 3, layer 1).
    const bool blocked =
        snapshot || invalidate_wait_ms != 0 || Debug::Get().astc_verify || Debug::Tracing() ||
        costsplit != nullptr || Rescale::CurrentFactor().Active() || mario_kart_world ||
        device_lost.load(std::memory_order_acquire) ||
        sparse_split_wanted.load(std::memory_order_acquire) ||
        (counters_engine != nullptr && !counters_engine->GpuConditions()) ||
        // Stage 2: the texture gate's skip path takes no hash, so nothing is logged to check.
        (ra.validate_only && tex_prehash != nullptr);
    if (blocked) {
        ++ra.stops[static_cast<size_t>(RaStop::Precondition)];
        return false;
    }
    return true;
}

void Context::Impl::RaPass(u64 address, u32 mode, u32 value, u64 seq, u32 queue_index,
                           u64 first_held) {
    if (!ra.Active()) {
        ra.stopped = false;
        ra.stop_reason = RaStop::None;
        ra.stopped_queues.clear();
        ra.start = std::chrono::steady_clock::now();
        ra.held_cmdbufs = 0;
        ra.frame_ops = 0;
        ra.frame_draws = 0;
        ra.frame_mispredicted = false;
        ra.held_touched = false;
        ra.inject_deferred = false;
        ra.handle_words.clear();
        ra.uniform_copies.clear();
        ra.texpool_gen.clear();
        ra.frame_dispatches = 0;
        ra.ns_gate_ops = 0;
        ++ra.gates;
    }
    ra.gate.Pass({address, mode, value, queue_index, seq}, first_held);
    ++ra.provisional_waits;
    u64 words = 0;
    const auto& waits = ra.gate.ProvisionalWaits();
    for (size_t i = 0; i < waits.size(); ++i) {
        bool first = true;
        for (size_t j = 0; j < i; ++j) {
            first = first && waits[j].address != waits[i].address;
        }
        words += first ? 1 : 0;
    }
    ra.gate_words_max = (std::max)(ra.gate_words_max, words);
}

bool Context::Impl::RaContagion(u32 queue_index, u64 first_held) {
    if (ra.stopped || ra.latched_off) {
        ++ra.contagion_blocks;
        return false;
    }
    ra.gate.HoldQueue(queue_index, first_held);
    ++ra.contagion_holds;
    return true;
}

RaStop Context::Impl::RaOpCheck(u32 queue_index, u16 op, const void* payload, size_t payload_size) {
    RaOpTimerStop(); // the previous op recorded ahead ends here
    if (ra.stopped) {
        return ra.stop_reason;
    }
    for (const auto& [stopped_queue, why] : ra.stopped_queues) {
        if (stopped_queue == queue_index) {
            return why;
        }
    }
    // Yield: a command is waiting (the gate's EventCpu, a CpuWrite...): let ReplayMain take it.
    if (commands_enqueued.load(std::memory_order_acquire) !=
        commands_taken.load(std::memory_order_acquire)) {
        ++ra.yields;
        return RaStop::Count;
    }
    // Stage 2 (validate-only): the draw-path ops, direct only; a CPU index conversion is
    // predicted from the draw mode (GenIndex has no repair).
    const RaAdmit admit =
        ra.validate_only ? RaAdmitDrawOp(op, payload, payload_size) : RaAdmit::NotDrawOp;
    if (admit == RaAdmit::NotDrawOp) {
        if (!AllowedAhead(op)) {
            RaStopNow(RaStop::Op, queue_index);
            return RaStop::Op;
        }
    } else if (admit == RaAdmit::GenIndex) {
        RaStopNow(RaStop::RareClass, queue_index);
        return RaStop::RareClass;
    } else if (admit != RaAdmit::Yes) {
        RaStopNow(RaStop::Op, queue_index);
        return RaStop::Op;
    }
    // Event ops: the model reads guest memory for an event it has not seen; a GPU signal of a
    // gate word would let the model satisfy a provisional wait with held work.
    if (op == static_cast<u16>(VC::Op::WaitEvent) && payload_size >= sizeof(VC::WaitEventPayload)) {
        const auto* p = static_cast<const VC::WaitEventPayload*>(payload);
        if (!events.Known(p->event_address)) {
            RaStopNow(RaStop::UnseenEvent, queue_index);
            return RaStop::UnseenEvent;
        }
    } else if (op == static_cast<u16>(VC::Op::SignalEvent) &&
               payload_size >= sizeof(VC::SignalEventPayload)) {
        const auto* p = static_cast<const VC::SignalEventPayload*>(payload);
        if (!events.Known(p->event_address)) {
            RaStopNow(RaStop::UnseenEvent, queue_index);
            return RaStop::UnseenEvent;
        }
        if (RaLearned(p->event_address) || ra.gate.IsGateWord(p->event_address)) {
            RaStopNow(RaStop::GateSignal, queue_index);
            return RaStop::GateSignal;
        }
    }
    if (ra.held_cmdbufs >= HELD_CMDBUF_CAP) {
        RaStopNow(RaStop::CmdBufCap);
        return RaStop::CmdBufCap;
    }
    if (std::chrono::steady_clock::now() - ra.start > AHEAD_TIME_CAP) {
        RaStopNow(RaStop::TimeCap);
        return RaStop::TimeCap;
    }
    const bool draw_like =
        op == static_cast<u16>(VC::Op::Draw) || op == static_cast<u16>(VC::Op::DrawIndexed) ||
        op == static_cast<u16>(VC::Op::DrawMeshTasks);
    const bool dispatch = op == static_cast<u16>(VC::Op::Dispatch);
    if (admit == RaAdmit::Yes &&
        (pools_upload_pending.load(std::memory_order_acquire) != 0 || SparseSplitGrowthDue())) {
        // A GPU-only pool's whole upload (EnsureBacking / the raw global table) or a sparse split
        // pool's growth from tracked CPU writes would run in this op: both read guest memory with
        // no repair (rare classes), so the queue stops before the op.
        RaStopNow(RaStop::PoolContents, queue_index);
        return RaStop::PoolContents;
    }
    if (admit == RaAdmit::Yes && (draw_like || dispatch)) {
        // The queue's texture pool set: a resolve generation move from GPU feedback, mark aging
        // or a release (inputs that depend on when the draw is recorded) stops this queue.
        const u64 generation = RaTexPoolGeneration(queue_index);
        const auto it = std::ranges::find_if(
            ra.texpool_gen, [queue_index](const auto& entry) { return entry.first == queue_index; });
        if (it == ra.texpool_gen.end()) {
            ra.texpool_gen.emplace_back(queue_index, generation);
        } else if (it->second != generation) {
            ++ra.checks[static_cast<size_t>(RaCheck::TexPoolGen)].mismatches;
            RaStopNow(RaStop::TexPoolGen, queue_index);
            return RaStop::TexPoolGen;
        }
    }
    if (TouchingAhead(op) || (admit == RaAdmit::Yes && RaDrawTouching(op))) {
        ra.held_touched = true;
    }
    ++ra.frame_ops;
    ++ra.ops_ahead;
    if (op < std::size(ra.ops_by_kind)) {
        ++ra.ops_by_kind[op];
    }
    if (draw_like) {
        ++ra.frame_draws;
        ++ra.draws_ahead;
    } else if (dispatch) {
        ++ra.frame_dispatches;
        ++ra.dispatches_ahead;
    }
    ra.op_t0 = std::chrono::steady_clock::now();
    ra.op_timer = true;
    ra.last_op = op;
    return RaStop::None;
}

void Context::Impl::RaStopNow(RaStop why, u32 queue_index) {
    if (!ra.Active() || ra.stopped) {
        return;
    }
    if (queue_index != RA_ALL_QUEUES) {
        // This queue only: the other held queues keep recording ahead, and queues may still
        // join or become held (its suspended stream resumes after the release).
        if (ra.Stopped(queue_index)) {
            return;
        }
        ra.stopped_queues.emplace_back(queue_index, why);
        ++ra.stops[static_cast<size_t>(why)];
        ++ra.queue_stops;
        return;
    }
    ra.stopped = true;
    ra.stop_reason = why;
    ++ra.stops[static_cast<size_t>(why)];
    ra.ns_ahead += NsSince(ra.start);
}

void Context::Impl::RaHold(u32 queue_index, u64 seq, RaHeldBatch&& batch) {
    ra.held_cmdbufs += batch.segs.size();
    ra.held_segments += batch.segs.size();
    ++ra.held_batches;
    ra.gate.Hold(queue_index, seq, std::move(batch));
}

void Context::Impl::RaCheckWaits(std::span<const VkSemaphore> waits, std::span<const u64> values) {
    for (const auto& [queue_index, from] : ra.gate.HeldQueues()) {
        const VkSemaphore timeline = *GetQueue(queue_index).timeline;
        for (size_t i = 0; i < waits.size() && i < values.size(); ++i) {
            if (waits[i] == timeline && values[i] >= from) {
                ++ra.order_violations;
                if (ra.order_violations == 1) {
                    LOG_ERROR(Render_Vulkan,
                              "[nvn] record-ahead: a submission waits for held value {} of queue "
                              "{} (held from {}): contagion missed",
                              values[i], queue_index, from);
                }
            }
        }
    }
}

void Context::Impl::RaOnCpuChange(u64 address, bool forget) {
    if (!ra.enabled) {
        return;
    }
    if (forget) {
        // EventInitialize: not a signal (the model re-reads the word at its next use).
        if (ra.Active() && ra.gate.IsGateWord(address)) {
            RaStopNow(RaStop::Tripwire); // keeps holding; the parked timeout releases
        }
        return;
    }
    RaWord& word = ra.words[address];
    ++word.cpu_changes;
    if (!word.pending.empty()) {
        bool all = true;
        std::vector<VC::SignalPoint> deps;
        for (const auto& [mode, value] : word.pending) {
            deps.clear();
            all = all && events.Wait(address, mode, value, deps).satisfied;
        }
        word.stable = all ? word.stable + 1 : 0;
        word.pending.clear();
    }
    if (!ra.Active() || !ra.gate.IsGateWord(address)) {
        return;
    }
    // Release only when every provisional wait (all gate words) is satisfied; otherwise keep
    // holding: a later change of a gate word releases, or the parked timeout.
    const bool all = ra.gate.AllSatisfied([this](u64 a) { return events.Value(a); });
    if (!all) {
        // Mispredicted only when this word's own waits are not satisfied by its change (another
        // gate word not signalled yet is the normal case of a multi-word gate).
        bool own = true;
        for (const auto& wait : ra.gate.ProvisionalWaits()) {
            if (wait.address == address) {
                own = own && VC::EventModel::Satisfies(events.Value(address), wait.mode, wait.value);
            }
        }
        if (!own && !ra.frame_mispredicted) {
            ra.frame_mispredicted = true;
            ++ra.mispredicts;
            if (ra.mispredicts >= 3) {
                ra.latched_off = true;
                LOG_WARNING(Render_Vulkan, "[nvn] record-ahead: 3 mispredicted gates: off for "
                                           "this boot");
            }
        }
        return;
    }
    if (ra.inject == 1 && !ra.inject_deferred && ra.gates % 60 == 0) {
        // Fault injection: hold one more command (as a misprediction would), then release.
        ra.inject_deferred = true;
        ++ra.injected;
        return;
    }
    RaRelease(false);
}

namespace {
constexpr std::array<const char*, HeadTrace::CLASSES> TRACE_CLS_NAMES{
    "MirrorCheck", "MirrorCopy", "VramMirror",  "VramWanted",  "TexSig",     "TexFull",
    "TexBytes",    "TexGather",  "TexTakeover", "UniformWord", "HandleMem",  "OracleUniform",
    "UniformCopy", "GenIndex",   "PoolUpload",  "EventWord",   "TexReadback", "SparseGrowth",
};
static_assert(TRACE_CLS_NAMES.size() == static_cast<size_t>(PreRec::Cls::Count));

float TraceMs(std::chrono::steady_clock::time_point from, std::chrono::steady_clock::time_point to) {
    return std::chrono::duration<float, std::milli>(to - from).count();
}
} // Anonymous namespace

void Context::Impl::HeadTraceStart(u64 address) {
    HeadTrace& ht = head_trace;
    const u32 value = events.Value(address); // just written by the CPU change: no guest read
    bool unblocks = false;
    for (u32 q = 0; q < signal_yield.blocked.size(); ++q) {
        const auto& blocked = signal_yield.blocked[q];
        if (blocked.valid && blocked.address == address &&
            VC::EventModel::Satisfies(value, blocked.mode, blocked.value)) {
            unblocks = true;
            break;
        }
    }
    if (!unblocks) {
        return;
    }
    if (ht.on) {
        HeadTraceFlush();
        ++ht.heads;
    }
    ht.on = true;
    ht.t0 = std::chrono::steady_clock::now();
    ht.have_prev = false;
    ht.seen.fill(0);
    PreRec::Detail::ra_trace_fired = 0;
}

void Context::Impl::HeadTraceFlush() {
    HeadTrace& ht = head_trace;
    if (ht.have_prev) {
        const auto now = std::chrono::steady_clock::now();
        const u64 ns = static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - ht.prev_t).count());
        const u32 q = (std::min)(ht.prev_queue, HeadTrace::QUEUES - 1);
        const u32 fired = PreRec::Detail::ra_trace_fired;
        if (ht.prev_ms < HeadTrace::HEAD_MS) {
            const u32 op = (std::min<u32>)(ht.prev_op, HeadTrace::OPS - 1);
            ++ht.op_count[q][op];
            ht.op_ns[q][op] += ns;
            ht.rec_ns[q] += ns;
            for (u32 c = 0; c < HeadTrace::CLASSES; ++c) {
                if ((fired >> c) & 1) {
                    ht.cls_ns[q][c] += ns;
                }
            }
        }
        const u32 fresh = fired & ~ht.seen[q];
        for (u32 c = 0; c < HeadTrace::CLASSES; ++c) {
            if (((fresh >> c) & 1) && ht.cls_first[q][c].size() < 4096) {
                ht.cls_first[q][c].push_back(ht.prev_ms);
            }
        }
        ht.seen[q] |= fresh;
        ht.have_prev = false;
    }
    PreRec::Detail::ra_trace_fired = 0;
    PreRec::Detail::ra_trace = false;
}

void Context::Impl::HeadTraceOp(u32 queue_index, u16 op) {
    HeadTrace& ht = head_trace;
    HeadTraceFlush();
    const auto now = std::chrono::steady_clock::now();
    const float ms = TraceMs(ht.t0, now);
    if (ms > HeadTrace::END_MS) {
        ht.on = false;
        ++ht.heads;
        return;
    }
    ht.have_prev = true;
    ht.prev_queue = queue_index;
    ht.prev_op = op;
    ht.prev_ms = ms;
    ht.prev_t = now;
    PreRec::Detail::ra_trace = true;
}

void Context::Impl::HeadTraceLog(u64 presents) {
    HeadTrace& ht = head_trace;
    if (presents < ht.last_log_present + 600 || ht.heads == 0) {
        return;
    }
    ht.last_log_present = presents;
    const double heads = static_cast<double>(ht.heads);
    for (u32 q = 0; q < HeadTrace::QUEUES; ++q) {
        if (ht.rec_ns[q] == 0) {
            continue;
        }
        std::array<u32, HeadTrace::OPS> order{};
        for (u32 o = 0; o < HeadTrace::OPS; ++o) {
            order[o] = o;
        }
        std::ranges::sort(order, [&](u32 a, u32 b) { return ht.op_ns[q][a] > ht.op_ns[q][b]; });
        std::string ops;
        for (const u32 o : order) {
            if (ht.op_count[q][o] == 0) {
                break;
            }
            ops += fmt::format(" {}={:.1f}/{:.3f}", o, ht.op_count[q][o] / heads,
                               ht.op_ns[q][o] / 1e6 / heads);
        }
        std::string classes;
        for (u32 c = 0; c < HeadTrace::CLASSES; ++c) {
            auto& first = ht.cls_first[q][c];
            if (first.empty() && ht.cls_ns[q][c] == 0) {
                continue;
            }
            float median = 0.0f;
            if (!first.empty()) {
                std::nth_element(first.begin(), first.begin() + first.size() / 2, first.end());
                median = first[first.size() / 2];
            }
            classes += fmt::format(" {}={:.0f}%@{:.3f}:{:.3f}", TRACE_CLS_NAMES[c],
                                   100.0 * first.size() / heads, median,
                                   ht.cls_ns[q][c] / 1e6 / heads);
        }
        LOG_INFO(Render_Vulkan,
                 "[nvn] head trace: present {} heads {} q{} rec {:.3f} ms/head in the first {} ms "
                 "| ops (kind=count/ms per head, by time):{} | classes (share of heads@median ms "
                 "of first use:ms of ops using it per head):{}",
                 presents, ht.heads, q, ht.rec_ns[q] / 1e6 / heads, HeadTrace::HEAD_MS,
                 ops.empty() ? " none" : ops, classes.empty() ? " none" : classes);
    }
    ht.heads = 0;
    ht.rec_ns.fill(0);
    for (u32 q = 0; q < HeadTrace::QUEUES; ++q) {
        ht.op_count[q].fill(0);
        ht.op_ns[q].fill(0);
        ht.cls_ns[q].fill(0);
        for (auto& v : ht.cls_first[q]) {
            v.clear();
        }
    }
}

void Context::Impl::RaInjectRead() {
    if (ra.ops_ahead % 4096 != 1 || ra.gate.ProvisionalWaits().empty()) {
        return;
    }
    // Read the gate word (a mapped guest address) without a note.
    if (VC::GuestMemory* const memory = guest.load(std::memory_order_acquire)) {
        u32 word = 0;
        static_cast<void>(memory->Read(ra.gate.ProvisionalWaits().front().address, &word,
                                       sizeof(word)));
        ++ra.injected;
    }
}

void Context::Impl::RaNoteHandleWord(u64 address, u32 value, bool ok) {
    if (!ra.Active()) {
        return;
    }
    ra.handle_words.push_back({address, value, ok});
    ++ra.handle_words_logged;
}

bool Context::Impl::RaValidateHandleWords() {
    VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    bool same = true;
    for (const RecordAhead::HandleWord& word : ra.handle_words) {
        u32 now = 0;
        const bool ok = memory != nullptr && memory->Read(word.address, &now, sizeof(now));
        if (ok == word.ok && (!ok || now == word.value)) {
            continue;
        }
        same = false;
        ++ra.handle_word_mismatches;
        if (!ra.latched_off) {
            ra.latched_off = true;
            LOG_WARNING(Render_Vulkan,
                        "[nvn] record-ahead: a handle word read ahead changed before the signal "
                        "(address {:#x}, read ahead {:#x}{}, at the release {:#x}{}): off for this "
                        "boot",
                        word.address, word.value, word.ok ? "" : " (unreadable)", now,
                        ok ? "" : " (unreadable)");
        }
    }
    ra.handle_words.clear();
    return same;
}

void Context::Impl::RaNoteUniformCopy(RaUniformCopy&& copy) {
    if (!ra.Active()) {
        return;
    }
    ra.uniform_copies.push_back(std::move(copy));
    ++ra.uniform_copies_logged;
}

void Context::Impl::RaLateBindUniformCopies() {
    const auto begin = std::chrono::steady_clock::now();
    std::vector<u8> want;
    for (RaUniformCopy& copy : ra.uniform_copies) {
        ra.ucopy_bytes += copy.size;
        want.assign(static_cast<size_t>(copy.size), 0);
        if (const u8* source = GuestPointer(copy.address, copy.size)) {
            std::memcpy(want.data(), source, want.size());
        }
        for (const auto& [offset, bytes] : copy.patches) {
            std::memcpy(want.data() + offset, bytes.data(), bytes.size());
        }
        // `written` holds exactly the bytes in `pointer` (written from it at record time, nothing
        // wrote `pointer` since): compared here instead of reading the mapped upload memory back.
        if (copy.written.size() != want.size() ||
            std::memcmp(copy.written.data(), want.data(), want.size()) != 0) {
            ++ra.uniform_copies_changed;
            std::memcpy(copy.pointer, want.data(), want.size());
            // A2 (the re-take landed) reads the mapped memory back: VERIFY boots only.
            if (ra.verify) {
                ra.uniform_copies_different +=
                    std::memcmp(copy.pointer, want.data(), want.size()) != 0 ? 1 : 0;
            }
        }
    }
    ra.uniform_copies.clear();
    ra.ns_release_late_bind += NsSince(begin);
}

void Context::Impl::RaAfterCommand() {
    if (ra.inject_deferred && ra.Active()) {
        ra.inject_deferred = false;
        RaRelease(false);
    }
}

void Context::Impl::RaNoteCommand(const char* kind, bool host_write) {
    if (!ra.Active()) {
        return;
    }
    ++ra.e8_commands;
    if (!ra.held_touched) {
        return; // held work is state and sync only: commutes
    }
    if (host_write && ra.validate_only) {
        // Stage 2 E8 rule for NotifyHostWrite: applied now, as today (it precedes the recording
        // there). On a VRAM-logged pool its mirror effects are deferred to the release repair
        // (NotifyCpuWriteNow); on a Backing::Mirror pool it untrusts the pages, so the release
        // check hashes every page a decision used again; a texture checked or uploaded ahead is
        // hashed again at the release; uniform copies are late-bound and handle words compared.
        // So a host write that changed anything held work captured is found at the release
        // (validate-only: counted as a latch there).
        ++ra.e8_validated;
        return;
    }
    ++ra.e8_failures;
    ra.latched_off = true;
    LOG_WARNING(Render_Vulkan,
                "[nvn] record-ahead: {} arrived while held work had touched resources: off for "
                "this boot",
                kind);
}

void Context::Impl::RaRelease(bool timeout) {
    if (!ra.Active()) {
        return;
    }
    const auto begin = std::chrono::steady_clock::now();
    ra.rel_flag_ahead += PreRec::Detail::ra_ahead ? 1 : 0;
    if (!ra.stopped) {
        ra.ns_ahead += NsSince(ra.start);
    }
    if (ra.frame_ops != 0) {
        ++ra.frames_with_ahead;
    }
    // Values read ahead that chose what held commands bind (not repairable: checked only).
    auto phase = std::chrono::steady_clock::now();
    const auto split = [&phase](u64& into) {
        const auto now = std::chrono::steady_clock::now();
        into += static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - phase).count());
        phase = now;
    };
    const auto words_begin = phase;
    if (!ra.handle_words.empty()) {
        static_cast<void>(RaValidateHandleWords());
    }
    split(ra.ns_rel_handles);
    // Bytes captured on the CPU into held memory: taken again now (late binding).
    if (!ra.uniform_copies.empty()) {
        RaLateBindUniformCopies();
    }
    split(ra.ns_rel_ucopies);
    ra.ns_rel_words += NsSince(words_begin);
    // Stage 2: the validate-only checks (and the rare-class reads made ahead).
    if (ra.validate_only) {
        RaValidateAll();
    }
    phase = std::chrono::steady_clock::now();
    RaOpTimerStop();
    if (ra.frame_draws != 0 || ra.frame_dispatches != 0) {
        ++ra.gates_with_draws;
        ra.ns_ops_ahead_draw_gates += ra.ns_gate_ops;
    }
    // The gate ends first: whatever happens below, no later recording is held.
    auto held = ra.gate.Take();
    ra.held_cmdbufs = 0;
    ra.stopped = false;
    ra.stopped_queues.clear();
    ra.held_touched = false;
    ra.inject_deferred = false;
    ++ra.releases;
    if (timeout) {
        ++ra.releases_timeout;
    }
    RaRetirePrologues();
    const auto fail_all = [&] {
        for (auto& entry : held) {
            FailFlight(*entry.batch.flight);
        }
    };
    if (device_lost.load(std::memory_order_acquire)) {
        RaVramAbandon();
        RaValidateDrop();
        ra.prologue_copies.clear();
        ++ra.dropped;
        fail_all();
        return;
    }
    phase = std::chrono::steady_clock::now();
    if (!ra_vram.empty()) {
        RaVramRepairAll(); // copy-ins into ra.prologue_copies
    }
    split(ra.ns_rel_vram);
    if (ra.inject == 3) {
        // Fault injection: a prologue at every release (one scratch page copied).
        try {
            if (!ra.inject_scratch) {
                ra.inject_scratch = CreateHostBuffer(
                    2 * 4096, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    MemoryUsage::DeviceLocal);
            }
            ra.prologue_copies.push_back(
                {*ra.inject_scratch, *ra.inject_scratch, VkBufferCopy{0, 4096, 4096}});
            ++ra.injected;
        } catch (const std::exception& exception) {
            LOG_ERROR(Render_Vulkan, "[nvn] record-ahead: injection scratch buffer: {}",
                      exception.what());
        }
    }
    // The prologue runs before every held batch: its signal value is a wait of the first held
    // segment of every held queue (patched now: the segments' wait arrays are CPU-side).
    VkCommandBuffer prologue = VK_NULL_HANDLE;
    u64 prologue_value = 0;
    if (!ra.prologue_copies.empty()) {
        prologue = RaRecordPrologue();
        if (prologue != VK_NULL_HANDLE) {
            prologue_value = ++ra.prologue_value;
            std::vector<u32> patched;
            for (auto& entry : held) {
                if (entry.batch.segs.empty() ||
                    std::ranges::find(patched, entry.queue) != patched.end()) {
                    continue;
                }
                patched.push_back(entry.queue);
                RaHeldBatch::Seg& first = entry.batch.segs.front();
                first.waits.push_back(*ra.prologue_timeline);
                first.wait_values.push_back(prologue_value);
                ++ra.prologue_waits_patched;
            }
        }
    }
    split(ra.ns_rel_prologue);
    const size_t head = prologue != VK_NULL_HANDLE ? 1 : 0;
    size_t count = head;
    for (const auto& entry : held) {
        count += entry.batch.segs.size();
    }
    std::vector<VkTimelineSemaphoreSubmitInfo> timelines(count);
    std::vector<std::vector<VkPipelineStageFlags>> stages(count);
    std::vector<VkSemaphore> signals(count);
    std::vector<u64> signal_values(count);
    std::vector<VkSubmitInfo> submits(count);
    if (head != 0) {
        signals[0] = *ra.prologue_timeline;
        signal_values[0] = prologue_value;
        timelines[0] = {
            .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreValueCount = 0,
            .pWaitSemaphoreValues = nullptr,
            .signalSemaphoreValueCount = 1,
            .pSignalSemaphoreValues = &signal_values[0],
        };
        submits[0] = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = &timelines[0],
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = nullptr,
            .pWaitDstStageMask = nullptr,
            .commandBufferCount = 1,
            .pCommandBuffers = &prologue,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &signals[0],
        };
    }
    size_t k = head;
    for (const auto& entry : held) {
        for (const RaHeldBatch::Seg& seg : entry.batch.segs) {
            signals[k] = entry.batch.timeline;
            signal_values[k] = seg.signal_value;
            stages[k].assign(seg.waits.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            timelines[k] = {
                .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                .pNext = nullptr,
                .waitSemaphoreValueCount = static_cast<u32>(seg.wait_values.size()),
                .pWaitSemaphoreValues = seg.wait_values.data(),
                .signalSemaphoreValueCount = 1,
                .pSignalSemaphoreValues = &signal_values[k],
            };
            submits[k] = {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .pNext = &timelines[k],
                .waitSemaphoreCount = static_cast<u32>(seg.waits.size()),
                .pWaitSemaphores = seg.waits.data(),
                .pWaitDstStageMask = stages[k].data(),
                .commandBufferCount = 1,
                .pCommandBuffers = &seg.raw,
                .signalSemaphoreCount = 1,
                .pSignalSemaphores = &signals[k],
            };
            ++k;
        }
    }
    if (count != 0) {
        VkResult result = VK_SUCCESS;
        u64 submit_tsc = 0;
        const u64 first_seq = held.empty() ? 0 : held.front().seq;
        {
            Common::TlRing::Note(Common::TlRing::RQSubB, first_seq, count);
            std::scoped_lock lock{submit_mutex};
            submit_tsc = Common::TlRing::Now();
            result = queue.Submit(submits);
            Common::TlRing::Note(Common::TlRing::RQSubE, first_seq, count);
        }
        split(ra.ns_rel_submit);
        ra.rel_segments += count;
        if (result == VK_SUCCESS && (Debug::Timeline() || Common::TlRing::On())) {
            // The GPU profiler pairs each command buffer with its submit time (the frame
            // timeline's GPU clock fit uses it): the held ones were submitted now.
            const u64 now_us = static_cast<u64>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count());
            for (const auto& entry : held) {
                if (entry.batch.flight->gpu_prof) {
                    entry.batch.flight->gpu_prof->NoteSubmitted(now_us, entry.queue, submit_tsc);
                }
            }
        }
        if (result != VK_SUCCESS) {
            // Never thrown from here (Abort and destructors call this): the flights fail.
            if (result == VK_ERROR_DEVICE_LOST) {
                NoteDeviceLost("record-ahead release");
            }
            LOG_ERROR(Render_Vulkan, "[nvn] record-ahead: release submit failed ({})",
                      static_cast<int>(result));
            ++ra.dropped;
            fail_all();
            return;
        }
        if (head != 0) {
            ra.prologue_retire.emplace_back(prologue, prologue_value);
            ++ra.prologue_submits;
        }
        for (const auto& entry : held) {
            if (!entry.batch.segs.empty()) {
                QueueState& state = GetQueue(entry.queue);
                state.gpu_submitted_value =
                    (std::max)(state.gpu_submitted_value, entry.batch.segs.back().signal_value);
            }
        }
        if (options.on_queue_submit) {
            options.on_queue_submit();
        }
    }
    phase = std::chrono::steady_clock::now();
    for (auto& entry : held) {
        if (entry.batch.mark) {
            MarkSubmitted(*entry.batch.flight, entry.batch.timeline, entry.batch.mark_value);
        }
    }
    split(ra.ns_rel_mark);
    ra.ns_release += NsSince(begin);
}

bool Context::Impl::RaVramAhead(Pool& pool, PoolMirror& mirror) {
    // Suspend / Finish of a held queue record into held segments too (after every decision of
    // the span, before the release): the frozen view applies there as well.
    const bool ahead = ra.Active() && (VC::PreRec::Detail::ra_ahead || ra.tail_held);
    if (mirror.ra_log != nullptr) {
        if (ahead) {
            ++mirror.ra_log->decisions;
            ++ra.vram_frozen_decisions;
            return true;
        }
        // Not recorded ahead (a non-held queue, or held work recorded outside an op): it sees
        // the tracker, the frozen decisions did not. Counted; no gate opens again this boot.
        ++ra.vram_unheld_decisions;
        if (!ra.latched_off) {
            ra.latched_off = true;
            LOG_ERROR(Render_Vulkan,
                      "[nvn] record-ahead: a mirror decision not recorded ahead used pool {:#x}, "
                      "whose write view is frozen for the open gate: off for this boot",
                      pool.address);
        }
        return false;
    }
    if (!ahead) {
        return false;
    }
    // The pool's first decision recorded ahead: what the CPU wrote so far is collected (today's
    // first decision would collect it too), then the pool's write view is frozen until the
    // release.
    MirrorCollectCpuWrites(pool, mirror, 0, pool.size);
    auto entry = std::make_unique<RaVramPool>();
    entry->pool = pool.shared_from_this();
    entry->log = RaVramLog{pool.size, mirror.write_epoch ? mirror.epoch_clean : 0};
    mirror.ra_log = &entry->log;
    ++mirror.ra_log->decisions;
    ra_vram.push_back(std::move(entry));
    ++ra.vram_logged_pools;
    ++ra.vram_frozen_decisions;
    // Mirror pages are replay state other queues decide from: their resource ops go behind.
    ra.held_touched = true;
    return true;
}

void Context::Impl::RaVramDeferCpuWrite(Pool& pool, u64 begin, u64 size, bool host_write) {
    RaVramLog* const log = pool.vram_mirror->ra_log;
    const auto it = std::ranges::find_if(
        ra_vram, [log](const std::unique_ptr<RaVramPool>& entry) { return &entry->log == log; });
    ASSERT(it != ra_vram.end());
    RaVramPool::Deferred deferred{.begin = begin, .size = size, .host_write = host_write};
    // The runs as today at the command (with `flush_tracked` the tracker's dirty bits of the
    // range are taken now; the release adds the runs to the changed pages).
    CpuWriteMirrorRuns(pool, begin, size, host_write, deferred.runs);
    // Draw contexts of queues not held are refreshed now, as today; a held queue has no live
    // entry while a command is taken (its stream yielded), its refresh is applied at the release.
    for (auto& [owner_queue, draw] : draw_contexts) {
        if (ra.Held(owner_queue)) {
            deferred.held_queues.push_back(owner_queue);
            continue;
        }
        for (const auto& [run_begin, run_end] : deferred.runs) {
            draw->OnCpuWrite(pool, run_begin, run_end - run_begin);
        }
    }
    for (const auto& run : deferred.runs) {
        log->flushed.push_back(run);
    }
    (*it)->deferred.push_back(std::move(deferred));
    ++ra.vram_deferred_commands;
}

void Context::Impl::RaVramRepairAll() {
    VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    // Emulation stopping: guest memory is unreadable, nothing is copied and no page is made
    // valid (the pools copy in again at their next use, if any).
    const bool stopped = memory == nullptr || memory->Stopped();
    for (const auto& entry_ptr : ra_vram) {
        RaVramPool& entry = *entry_ptr;
        Pool& pool = *entry.pool;
        PoolMirror* const mirror = pool.vram_mirror.get();
        const bool logged = mirror != nullptr && mirror->ra_log == &entry.log;
        if (logged) {
            mirror->ra_log = nullptr; // the frozen view ends
        }
        if (!logged || std::ranges::find(vram_mirrored, &pool) == vram_mirrored.end()) {
            ++ra.vram_lost_pools;
            continue;
        }
        const bool moved = !mirror->write_epoch ||
                           mirror->write_epoch->load(std::memory_order_acquire) != entry.log.epoch;
        if (!moved && entry.deferred.empty() && ra.inject != 4) {
            continue; // nothing written since the first decision: the decisions saw today's state
        }
        ++ra.vram_repairs;
        // 1. The valid bits before the CPU writes are applied.
        const MirrorPages before = mirror->pages;
        // 2. The deferred commands (as NotifyCpuWriteNow would have applied them), then the CPU
        //    writes (the collect sets epoch_clean).
        for (const RaVramPool::Deferred& deferred : entry.deferred) {
            for (const u32 held_queue : deferred.held_queues) {
                const auto draw = draw_contexts.find(held_queue);
                if (draw == draw_contexts.end()) {
                    continue;
                }
                for (const auto& [run_begin, run_end] : deferred.runs) {
                    draw->second->OnCpuWrite(pool, run_begin, run_end - run_begin);
                }
            }
            CpuWriteMirrorPageReset(pool, deferred.begin, deferred.size, deferred.host_write);
            for (const auto& [run_begin, run_end] : deferred.runs) {
                MirrorInvalidate(pool, run_begin, run_end - run_begin);
            }
        }
        RaRuns written;
        MirrorCollectCpuWrites(pool, *mirror, 0, pool.size, &written);
        // For A2: the write epoch the collect read before its TakeWritten (epoch_clean: a page
        // dirty later moved the epoch past it), and, for the diagnosis, the epoch right after the
        // collect and when.
        const u64 epoch_collect = mirror->epoch_clean;
        const u64 epoch_after_collect =
            mirror->write_epoch ? mirror->write_epoch->load(std::memory_order_acquire) : 0;
        const auto collect_time = std::chrono::steady_clock::now();
        if (ra.inject == 4) {
            // Fault injection: every valid page a decision used is repaired as if written.
            const RaRuns extra = RaVramInjectRuns(entry.log, before);
            written.insert(written.end(), extra.begin(), extra.end());
            ++ra.injected;
        }
        // 3-5. Prologue copies, pages valid and armed again, TV_current-only pages left invalid.
        const RaVramRepair repair =
            RaVramRepairPlan(entry.log, before, mirror->pages, written, !stopped);
        ra.vram_changed_pages += repair.changed_pages;
        ra.vram_current_only_pages += repair.current_only_pages;
        if (stopped) {
            ++ra.vram_stopped_skips;
            continue;
        }
        ra.vram_copy_pages += repair.copy_pages;
        ra.vram_revalidated_pages += repair.revalidate_pages;
        for (const auto& [begin, end] : repair.copies) {
            ra.prologue_copies.push_back(
                {pool.Handle(), *mirror->buffer, VkBufferCopy{begin, begin, end - begin}});
        }
        for (const auto& [begin, end] : repair.revalidate) {
            memory->Arm(mirror->track_id, pool.address + begin, end - begin);
        }
        if (ra.validate_only || ra.inject == 4) {
            // VRAM A2: a page a decision used that the repair left valid must not be written
            // (side-effect free: the tracker is only asked).
            RaCheckStats& a2 = ra.checks[static_cast<size_t>(RaCheck::VramA2)];
            for (const auto& [begin, end] : RaVramA2Pages(entry.log)) {
                for (u64 at = begin; at < end; at += RaPageSet::PAGE) {
                    const u64 length = (std::min)(RaPageSet::PAGE, end - at);
                    ++a2.checked;
                    const bool valid = mirror->pages.AllValid(at, length);
                    if (valid && memory->AnyWritten(mirror->track_id, pool.address + at, length)) {
                        const u64 epoch_now = mirror->write_epoch
                                                  ? mirror->write_epoch->load(std::memory_order_acquire)
                                                  : 0;
                        const bool collected = RaRunsContain(written, at);
                        // A dirty page with an epoch moved since the collect read it: the guest
                        // wrote it after the collect (the guest runs during the release); the next
                        // decision collects it, as without record-ahead. An unmoved epoch breaks
                        // the tracker invariant (RaVramA2Classify): latched.
                        const RaVramA2Verdict verdict =
                            RaVramA2Classify(true, true, mirror->write_epoch != nullptr,
                                             epoch_collect, epoch_now);
                        const bool after_collect = verdict == RaVramA2Verdict::WrittenAfterCollect;
                        // Turned dirty inside the collect itself (after its epoch read, before the
                        // read right after it): what an epoch read after the collect misses.
                        const bool in_collect = after_collect && epoch_after_collect == epoch_now;
                        if (in_collect) {
                            ++ra.vram_a2_in_collect;
                        }
                        if (after_collect && ra.vram_a2_after_collect++ < 8) {
                            LOG_INFO(Render_Vulkan,
                                     "[nvn] record-ahead VRAM A2: pool {:#x} +{:#x} written after "
                                     "the repair's collect read the write epoch (epoch {} at the "
                                     "collect's read, {} after the collect, {} now, {:.1f} us "
                                     "later; inside the collect {}; in the collected runs {}, "
                                     "copied by the repair {}): not latched",
                                     pool.address, at, epoch_collect, epoch_after_collect,
                                     epoch_now, NsSince(collect_time) / 1e3, in_collect ? 1 : 0,
                                     collected ? 1 : 0, RaRunsContain(repair.copies, at) ? 1 : 0);
                        }
                        if (!after_collect) {
                            LOG_WARNING(Render_Vulkan,
                                        "[nvn] record-ahead VRAM A2 positive: pool {:#x} +{:#x}: "
                                        "write epoch {} at the collect's read, {} after the "
                                        "collect, {} now ({:.1f} us later); page in the collected "
                                        "runs {}, copied by the repair {}",
                                        pool.address, at, epoch_collect, epoch_after_collect,
                                        epoch_now, NsSince(collect_time) / 1e3, collected ? 1 : 0,
                                        RaRunsContain(repair.copies, at) ? 1 : 0);
                            RaLatch(RaCheck::VramA2, pool.address, at,
                                    "a VRAM mirror page a decision used is valid but written "
                                    "after the repair");
                        }
                    }
                }
            }
        }
    }
    ra_vram.clear();
}

void Context::Impl::RaVramAbandon() {
    for (const auto& entry_ptr : ra_vram) {
        RaVramPool& entry = *entry_ptr;
        Pool& pool = *entry.pool;
        PoolMirror* const mirror = pool.vram_mirror.get();
        if (mirror == nullptr || mirror->ra_log != &entry.log) {
            continue;
        }
        mirror->ra_log = nullptr;
        for (const RaVramPool::Deferred& deferred : entry.deferred) {
            CpuWriteMirrorPageReset(pool, deferred.begin, deferred.size, deferred.host_write);
        }
        mirror->pages.InvalidateAll();
        ++ra.vram_abandoned;
    }
    ra_vram.clear();
}

VkCommandBuffer Context::Impl::RaRecordPrologue() {

    try {
        if (!ra.prologue_timeline) {
            const VkSemaphoreTypeCreateInfo type_ci{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                .pNext = nullptr,
                .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                .initialValue = 0,
            };
            ra.prologue_timeline = device.GetLogical().CreateSemaphore({
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                .pNext = &type_ci,
                .flags = 0,
            });
        }
        const VkCommandBuffer raw = AcquireCommandBuffer();
        const vk::CommandBuffer cmd{raw, dld};
        cmd.Begin({
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        });
        // Leading barrier: after everything earlier on the queue (frame N's work may still read
        // the mirror pages or write the guest bytes the copies read).
        const VkMemoryBarrier before{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        };
        cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                            before);
        for (const RaPrologueCopy& copy : ra.prologue_copies) {
            cmd.CopyBuffer(copy.src, copy.dst, copy.region);
            ++ra.prologue_copies_total;
            ra.prologue_bytes += copy.region.size;
        }
        // And before everything after it on the queue (held segments also wait for its signal;
        // a segment submitted after the release without a held piece relies on this barrier).
        const VkMemoryBarrier after{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                            after);
        cmd.End();
        ra.prologue_copies.clear();
        return raw;
    } catch (const std::exception& exception) {
        ra.prologue_copies.clear();
        ++ra.prologue_failed;
        if (!ra.latched_off) {
            ra.latched_off = true;
            LOG_ERROR(Render_Vulkan,
                      "[nvn] record-ahead: the release prologue could not be recorded ({}): off "
                      "for this boot",
                      exception.what());
        }
        return VK_NULL_HANDLE;
    }
}

void Context::Impl::RaRetirePrologues() {
    if (ra.prologue_retire.empty() || !ra.prologue_timeline) {
        return;
    }
    u64 reached = 0;
    try {
        reached = ra.prologue_timeline.GetCounter();
    } catch (const std::exception&) {
        return; // device lost: the buffers stay with the command pool
    }
    std::scoped_lock lock{free_commands_mutex};
    std::erase_if(ra.prologue_retire, [&](const std::pair<VkCommandBuffer, u64>& entry) {
        if (entry.second > reached) {
            return false;
        }
        free_command_buffers.push_back(entry.first);
        return true;
    });
}

void Context::Impl::RaEmergency(const char* where) {

    if (!ra.Active()) {
        return;
    }
    bool shutting_down = false;
    {
        std::scoped_lock lock{state_mutex};
        shutting_down = stopping;
    }
    if (shutting_down || device_lost.load(std::memory_order_acquire)) {
        // Nothing held may reach the GPU: the flights complete without running.
        ra.handle_words.clear();
        ra.uniform_copies.clear();
        auto held = ra.gate.Take();
        ra.stopped = false;
        ra.stopped_queues.clear();
        RaVramAbandon();
        RaValidateDrop();
        ra.prologue_copies.clear();
        ra.held_cmdbufs = 0;
        ++ra.dropped;
        for (auto& entry : held) {
            FailFlight(*entry.batch.flight);
        }
        return;
    }
    ++ra.tripwires;
    ++ra.stops[static_cast<size_t>(RaStop::Tripwire)];
    ra.latched_off = true;
    LOG_WARNING(Render_Vulkan,
                "[nvn] record-ahead: {} ran while work behind a CPU-gated wait was held: released "
                "early, record-ahead off for this boot",
                where);
    RaRelease(false);
}

void Context::Impl::RaBeforeCompletionWait(u64 seq) {
    if (!ra.Active() || completed_seq.load(std::memory_order_acquire) >= seq ||
        seq < ra.gate.MinHeldSeq()) {
        return;
    }
    RaEmergency("a CPU wait for a completion behind held work");
}

void Context::Impl::RaPeriodic(bool present, bool final) {
    if (!ra.enabled) {
        return;
    }
    if (!final) {
        RaAfterCommand(); // an injected deferral never waits longer than one idle pump
        if (ra.Active() && std::chrono::steady_clock::now() - ra.start > options.parked_timeout) {
            RaRelease(true);
        }
        if (!present) {
            return;
        }
    }
    const u64 presents = stats.presents.load(std::memory_order_relaxed);
    if (!final && presents < ra.last_log_present + 600) {
        return;
    }
    ra.last_log_present = presents;
    std::string stops;
    for (size_t i = 1; i < ra.stops.size(); ++i) {
        if (ra.stops[i] != 0) {
            stops += fmt::format(" {}={}", RA_STOP_NAMES[i], ra.stops[i]);
        }
    }
    std::string kinds;
    for (size_t i = 0; i < std::size(ra.ops_by_kind); ++i) {
        if (ra.ops_by_kind[i] != 0) {
            kinds += fmt::format(" {}={}", i, ra.ops_by_kind[i]);
        }
    }
    const PreRec::RaReadStats& reads = PreRec::RaStats();
    u64 read_calls = 0;
    for (const u64 n : reads.calls) {
        read_calls += n;
    }
    LOG_INFO(Render_Vulkan,
             "[nvn] record-ahead{}: present {} gates {} releases {} (timeout {}) frames-with-ahead {} "
             "ops {} draws {} ahead {:.3f} ms/gate release {:.3f} ms/gate held batches {} segments "
             "{} provisional {} gate-words-max {} contagion {} (blocked {}) order-violations {} "
             "ordering-holds {} e8 {} (failures {}) injected {} "
             "yields {} mispredicts {} tripwires {} dropped {} immediate-submits {} latched {} "
             "reads-ahead {} unlogged {} (first {:#x}) handle-words {} (mismatches {}) uniform-copies {} (changed {} "
             "different-after {}) queue-stops {} prologue {} (copies {} bytes {} waits-patched {} "
             "failed {}) vram: pools {} frozen-decisions {} unheld-decisions {} deferred {} "
             "repairs {} changed {} copied {} revalidated {} current-only {} stopped {} lost {} "
             "abandoned {} stops:{} "
             "ops-ahead:{}",
             final ? " (replay end)" : "", presents, ra.gates, ra.releases, ra.releases_timeout,
             ra.frames_with_ahead,
             ra.ops_ahead, ra.draws_ahead,
             ra.gates != 0 ? static_cast<double>(ra.ns_ahead) / 1e6 / ra.gates : 0.0,
             ra.releases != 0 ? static_cast<double>(ra.ns_release) / 1e6 / ra.releases : 0.0,
             ra.held_batches, ra.held_segments, ra.provisional_waits, ra.gate_words_max,
             ra.contagion_holds, ra.contagion_blocks, ra.order_violations, ra.ordering_holds,
             ra.e8_commands, ra.e8_failures, ra.injected, ra.yields,
             ra.mispredicts, ra.tripwires, ra.dropped, ra.immediate_submits,
             ra.latched_off ? 1 : 0, read_calls, reads.unlogged, reads.first_unlogged,
             ra.handle_words_logged, ra.handle_word_mismatches, ra.uniform_copies_logged,
             ra.uniform_copies_changed, ra.uniform_copies_different, ra.queue_stops,
             ra.prologue_submits, ra.prologue_copies_total, ra.prologue_bytes,
             ra.prologue_waits_patched, ra.prologue_failed, ra.vram_logged_pools,
             ra.vram_frozen_decisions, ra.vram_unheld_decisions, ra.vram_deferred_commands,
             ra.vram_repairs, ra.vram_changed_pages, ra.vram_copy_pages,
             ra.vram_revalidated_pages, ra.vram_current_only_pages, ra.vram_stopped_skips,
             ra.vram_lost_pools, ra.vram_abandoned, stops.empty() ? " none" : stops,
             kinds.empty() ? " none" : kinds);
    if (ra.validate_only || ra.inject == 4) {
        RaLogStage2(final);
    }
}

// --- Stage 2: validate-only checks (nvn_ra_validate.h) ---

void Context::Impl::RaOpTimerStop() {
    if (const PreRec::RaReadStats& reads = PreRec::RaStats(); reads.unlogged != ra.unlogged_seen)
        [[unlikely]] {
        const size_t k = ra.last_op < 80 ? ra.last_op : 80;
        ra.unlogged_by_op[k] += reads.unlogged - ra.unlogged_seen;
        ra.unlogged_va[k] = reads.last_unlogged;
        ra.unlogged_size[k] = reads.last_unlogged_size;
        ra.unlogged_seen = reads.unlogged;
    }
    ra.last_op = 0xFFFF;
    if (!ra.op_timer) {
        return;
    }
    ra.op_timer = false;
    const u64 ns = NsSince(ra.op_t0);
    ra.ns_ops_ahead += ns;
    ra.ns_gate_ops += ns;
}

u64 Context::Impl::RaTexPoolGeneration(u32 queue_index) const {
    if (!draw_resources) {
        return 0;
    }
    u64 sum = 0;
    for (const TexturePoolSet* set : draw_resources->texture_pools) {
        if (set != nullptr && set->Queue() == queue_index) {
            sum += set->TimedGeneration();
        }
    }
    return sum;
}

void Context::Impl::RaNoteMirrorPage(Pool& pool, u64 page) {
    RecordAhead::MirrorPool* entry = nullptr;
    for (auto it = ra.mirror_pools.rbegin(); it != ra.mirror_pools.rend(); ++it) {
        if (it->pool.get() == &pool) {
            entry = &*it;
            break;
        }
    }
    if (entry == nullptr) {
        entry = &ra.mirror_pools.emplace_back();
        entry->pool = pool.shared_from_this();
    }
    if (entry->pages.Insert(page)) {
        ++ra.checks[static_cast<size_t>(RaCheck::MirrorPage)].logged;
    }
}

void Context::Impl::RaNoteMirrorCopy(Pool& pool, u8* pointer, u64 begin, u64 length) {
    ra.mirror_copies.push_back({pointer, pool.shared_from_this(), begin, length});
    ++ra.checks[static_cast<size_t>(RaCheck::MirrorCopy)].logged;
}

void Context::Impl::RaNoteTexCheck(const Pool& pool, u64 storage_address, u64 offset, u64 size,
                                   u64 signature, bool full, u64 full_hash) {
    ra.tex_checks.push_back({pool.shared_from_this(), storage_address, offset, size, signature,
                             full_hash, full});
    ++ra.checks[static_cast<size_t>(RaCheck::TexCheck)].logged;
}

void Context::Impl::RaNoteTexBytes(const Pool& pool, u64 offset, u64 size, const u8* data) const {
    if (data == nullptr || size == 0) {
        return;
    }
    ra.tex_bytes.push_back({pool.shared_from_this(), offset, size, HashGuestBytes(data, size)});
    ++ra.checks[static_cast<size_t>(RaCheck::TexBytes)].logged;
}

void Context::Impl::RaLatch(RaCheck check, u64 address, u64 offset, const char* detail) {
    ++ra.checks[static_cast<size_t>(check)].mismatches;
    if (ra.latches++ < 32) {
        LOG_WARNING(Render_Vulkan,
                    "[nvn] record-ahead latch: class {} address {:#x} offset {:#x} present {} gate "
                    "{}: {}{}",
                    RA_CHECK_NAMES[static_cast<size_t>(check)], address, offset,
                    stats.presents.load(std::memory_order_relaxed), ra.gates, detail,
                    ra.latched_off ? "" : " (record-ahead off for this boot)");
    }
    ra.latched_off = true;
}

void Context::Impl::RaValidateDrop() {
    ra.mirror_pools.clear();
    ra.mirror_copies.clear();
    ra.tex_checks.clear();
    ra.tex_bytes.clear();
    ra.texpool_gen.clear();
}

namespace {

/// Recorder::GuestPageHash, at the release.
u64 RaPageHash(const Context::Impl& impl, const Pool& pool, u64 page, VC::GuestMemory* memory) {
    constexpr u64 UNREADABLE = 0x9E3779B97F4A7C15ull;
    const u64 begin = page * Pool::MIRROR_PAGE;
    if (begin >= pool.size) {
        return UNREADABLE;
    }
    const u64 length = (std::min)(Pool::MIRROR_PAGE, pool.size - begin);
    const u8* bytes = pool.HasHost() ? pool.HostRead(PreRec::Cls::MirrorCheck, begin, length)
                                     : impl.GuestPointer(pool.address + begin, length);
    std::array<u8, Pool::MIRROR_PAGE> local;
    if (bytes == nullptr) {
        if (memory == nullptr || !memory->Read(pool.address + begin, local.data(), length)) {
            return UNREADABLE;
        }
        bytes = local.data();
    }
    return GuestContentHash(bytes, length);
}

} // Anonymous namespace

void Context::Impl::RaValidateAll() {
    // The release can run inside an op recorded ahead (a tripwire): its own reads are not ahead.
    const bool was_ahead = std::exchange(PreRec::Detail::ra_ahead, false);
    const auto begin = std::chrono::steady_clock::now();
    VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    const auto alive = [this](const Pool& pool) {
        const auto it = pools.find(pool.serial);
        return it != pools.end() && it->second.get() == &pool;
    };
    const auto stat = [this](RaCheck check) -> RaCheckStats& {
        return ra.checks[static_cast<size_t>(check)];
    };
    if (memory == nullptr || memory->Stopped()) {
        // Emulation stopping: guest memory is unreadable (the gate's work is dropped or never
        // shown); nothing to compare.
        ++ra.validate_skipped;
        RaValidateDrop();
        PreRec::Detail::ra_ahead = was_ahead;
        return;
    }
    std::vector<u8> scratch;
    // Backing::Mirror staging copies made ahead: taken again (late binding) and compared. The
    // page hash is not recomputed in this stage: a change also fails the page check below.
    auto t0 = std::chrono::steady_clock::now();
    for (const RecordAhead::MirrorCopy& copy : ra.mirror_copies) {
        const Pool& pool = *copy.pool;
        if (!alive(pool) || copy.begin + copy.length > pool.size) {
            ++ra.lost_entries;
            continue;
        }
        ++stat(RaCheck::MirrorCopy).checked;
        const u8* source = nullptr;
        if (pool.HasHost()) {
            source = pool.HostRead(PreRec::Cls::MirrorCopy, copy.begin, copy.length);
        } else {
            scratch.assign(static_cast<size_t>(copy.length), 0);
            static_cast<void>(memory->Read(pool.address + copy.begin, scratch.data(), copy.length));
            source = scratch.data();
        }
        if (source != nullptr && std::memcmp(copy.pointer, source, copy.length) != 0) {
            std::memcpy(copy.pointer, source, copy.length);
            RaLatch(RaCheck::MirrorCopy, pool.address, copy.begin,
                    "Backing::Mirror bytes copied ahead changed before the signal (taken again)");
        }
    }
    stat(RaCheck::MirrorCopy).ns += NsSince(t0);
    ra.mirror_copies.clear();
    // Backing::Mirror pages: the decision today would take now, side-effect free.
    t0 = std::chrono::steady_clock::now();
    const u64 epoch_now = split_raw_epoch.load(std::memory_order_acquire);
    for (RecordAhead::MirrorPool& entry : ra.mirror_pools) {
        Pool& pool = *entry.pool;
        if (!alive(pool)) {
            ra.lost_entries += entry.pages.Count();
            continue;
        }
        const bool tracked = mirror_hash_tracked && pool.mirror_track != 0;
        const bool clean = tracked && pool.mirror_write_epoch &&
                           pool.mirror_write_epoch->load(std::memory_order_acquire) ==
                               pool.mirror_epoch_clean;
        RaCheckStats& st = stat(RaCheck::MirrorPage);
        entry.pages.ForEach([&](u64 index) {
            if (index >= pool.mirror_pages.size()) {
                return;
            }
            const Pool::MirrorPage& page = pool.mirror_pages[static_cast<size_t>(index)];
            const u64 at = index * Pool::MIRROR_PAGE;
            ++st.checked;
            const RaMirrorPageState state{
                .valid = page.valid,
                .trusted = page.trusted,
                .tracked = tracked,
                .written = tracked && !clean &&
                           memory->AnyWritten(pool.mirror_track, pool.address + at,
                                              (std::min)(Pool::MIRROR_PAGE, pool.size - at)),
                .split = pool.split,
                .page_raw_epoch = page.raw_epoch,
                .epoch_now = epoch_now,
            };
            const RaPageVerdict verdict = RaMirrorPageVerdict(state);
            if (verdict == RaPageVerdict::Unchanged) {
                return;
            }
            if (verdict == RaPageVerdict::EpochOnly && pool.prehash &&
                index < pool.prehash->count) {
                // The pre-hash worker's record for the current epoch (seqlock read).
                const auto& slot = pool.prehash->slots[static_cast<size_t>(index)];
                const u64 e1 = slot.epoch.load(std::memory_order_acquire);
                const u64 h = slot.hash.load(std::memory_order_acquire);
                const u64 e2 = slot.epoch.load(std::memory_order_acquire);
                if (e1 == e2 && e1 == epoch_now && h == page.hash) {
                    return;
                }
            }
            ++st.hashed;
            if (RaPageHash(*this, pool, index, memory) != page.hash) {
                RaLatch(RaCheck::MirrorPage, pool.address, at,
                        page.valid ? "Backing::Mirror page decided ahead differs from the guest "
                                     "bytes at the release"
                                   : "Backing::Mirror page decided ahead is not valid at the "
                                     "release and differs");
            }
        });
    }
    stat(RaCheck::MirrorPage).ns += NsSince(t0);
    ra.mirror_pools.clear();
    // Texture coherency checks made ahead: the same signature (and full hash) now.
    t0 = std::chrono::steady_clock::now();
    for (const RecordAhead::TexCheck& check : ra.tex_checks) {
        const Pool& pool = *check.pool;
        if (!alive(pool)) {
            ++ra.lost_entries;
            continue;
        }
        RaCheckStats& st = stat(RaCheck::TexCheck);
        ++st.checked;
        ++st.hashed;
        const u8* data = GuestBytes(pool, check.offset, check.size, scratch);
        const char* what = nullptr;
        if (data == nullptr) {
            what = "texture checked ahead is unreadable at the release";
        } else if (SampleGuestBytes(data, check.size, mario_kart_world) != check.signature) {
            what = "texture signature differs at the release";
        } else if (check.full && HashGuestBytes(data, check.size) != check.full_hash) {
            what = "texture full hash differs at the release";
        }
        if (what != nullptr) {
            RaLatch(RaCheck::TexCheck, check.storage_address, check.offset, what);
        }
    }
    stat(RaCheck::TexCheck).ns += NsSince(t0);
    ra.tex_checks.clear();
    // Texture bytes read ahead (uploads, CPU copies).
    t0 = std::chrono::steady_clock::now();
    for (const RecordAhead::TexBytes& bytes : ra.tex_bytes) {
        const Pool& pool = *bytes.pool;
        if (!alive(pool)) {
            ++ra.lost_entries;
            continue;
        }
        RaCheckStats& st = stat(RaCheck::TexBytes);
        ++st.checked;
        ++st.hashed;
        const u8* data = GuestBytes(pool, bytes.offset, bytes.size, scratch);
        if (data == nullptr || HashGuestBytes(data, bytes.size) != bytes.hash) {
            RaLatch(RaCheck::TexBytes, pool.address + bytes.offset, bytes.offset,
                    "texture bytes read ahead differ at the release");
        }
    }
    stat(RaCheck::TexBytes).ns += NsSince(t0);
    ra.tex_bytes.clear();
    // Rare classes read ahead (no repair, no check).
    const PreRec::RaReadStats& reads = PreRec::RaStats();
    for (size_t c = 0; c < reads.calls.size(); ++c) {
        if (!RaUnadmittedClass(static_cast<PreRec::Cls>(c)) || reads.calls[c] == ra.rare_calls[c]) {
            continue;
        }
        stat(RaCheck::Unadmitted).logged += reads.calls[c] - ra.rare_calls[c];
        ra.rare_calls[c] = reads.calls[c];
        RaLatch(RaCheck::Unadmitted, reads.last_va[c], c, TRACE_CLS_NAMES[c]);
    }
    ra.texpool_gen.clear();
    ra.ns_validate += NsSince(begin);
    PreRec::Detail::ra_ahead = was_ahead;
}

void Context::Impl::RaLogStage2(bool final) {
    std::string checks;
    for (size_t i = 0; i < ra.checks.size(); ++i) {
        const RaCheckStats& c = ra.checks[i];
        checks += fmt::format(" {}={}/{}/{}/{}/{:.3f}", RA_CHECK_NAMES[i], c.logged, c.checked,
                              c.hashed, c.mismatches,
                              ra.releases != 0 ? c.ns / 1e6 / ra.releases : 0.0);
    }
    std::string unlogged;
    for (size_t k = 0; k < ra.unlogged_by_op.size(); ++k) {
        if (ra.unlogged_by_op[k] != 0) {
            unlogged += fmt::format(" {}={}@{:#x}+{}", k, ra.unlogged_by_op[k], ra.unlogged_va[k],
                                    ra.unlogged_size[k]);
        }
    }
    const double rel = ra.releases != 0 ? 1e6 * static_cast<double>(ra.releases) : 1.0;
    LOG_INFO(Render_Vulkan,
             "[nvn] record-ahead stage2 diag{}: release split ms/gate: words {:.3f} (handles {:.3f} "
             "uniform copies {:.3f}, {:.0f} copy bytes/gate; entered with ra_ahead set {}) "
             "| mapping lookups ahead {} (raw table rebuilds ahead {}, last at present {}; pools "
             "backed ahead {}) | vram-a2 written after the collect {} (inside it {}) | vram {:.3f} prologue {:.3f} submit {:.3f} ({:.1f} segments/gate) mark {:.3f} | unlogged by op "
             "(kind=count@last va+size, 80 = outside an op):{}",
             final ? " (replay end)" : "", ra.ns_rel_words / rel, ra.ns_rel_handles / rel,
             ra.ns_rel_ucopies / rel,
             ra.releases != 0 ? static_cast<double>(ra.ucopy_bytes) / ra.releases : 0.0,
             ra.rel_flag_ahead, PreRec::RaStats().mapping_lookups, ra.raw_rebuilds_ahead,
             ra.raw_rebuild_last_present, ra.pools_backed_ahead, ra.vram_a2_after_collect, ra.vram_a2_in_collect,
             ra.ns_rel_vram / rel,
             ra.ns_rel_prologue / rel, ra.ns_rel_submit / rel,
             ra.releases != 0 ? static_cast<double>(ra.rel_segments) / ra.releases : 0.0,
             ra.ns_rel_mark / rel, unlogged.empty() ? " none" : unlogged);
    if (final && !unlogged.empty()) {
        LOG_INFO(Render_Vulkan, "[nvn] record-ahead unlogged call sites (replay end):{}",
                 PreRec::RaUnloggedSites());
    }
    if (final && PreRec::RaStats().mapping_lookups != 0) {
        LOG_INFO(Render_Vulkan, "[nvn] record-ahead mapping lookup call sites (replay end):{}",
                 PreRec::RaMappingSites());
    }
    LOG_INFO(Render_Vulkan,
             "[nvn] record-ahead stage2{}: present {} validate-only {} gates {} with-draws {} "
             "draws {} dispatches {} head-moved {:.3f} ms/gate ({:.3f} ms per gate with draws) "
             "release {:.3f} ms/gate (checks {:.3f}) latches {} latched {} skipped {} lost {} "
             "e8-validated {} | checks (logged/checked/hashed/mismatches/ms per release):{}",
             final ? " (replay end)" : "", stats.presents.load(std::memory_order_relaxed),
             ra.validate_only ? 1 : 0, ra.gates, ra.gates_with_draws, ra.draws_ahead,
             ra.dispatches_ahead, ra.gates != 0 ? ra.ns_ops_ahead / 1e6 / ra.gates : 0.0,
             ra.gates_with_draws != 0 ? ra.ns_ops_ahead_draw_gates / 1e6 / ra.gates_with_draws
                                      : 0.0,
             ra.releases != 0 ? ra.ns_release / 1e6 / ra.releases : 0.0,
             ra.releases != 0 ? ra.ns_validate / 1e6 / ra.releases : 0.0, ra.latches,
             ra.latched_off ? 1 : 0, ra.validate_skipped, ra.lost_entries, ra.e8_validated,
             checks);
}

} // namespace Vulkan::Nvn
