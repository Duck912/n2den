// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <string_view>

#include "common/common_types.h"
#include "common/tl_ring.h"

namespace VideoCommon::Nvn {
class GuestMemory;
}

/// Pre-record probe (EDEN_NX2_NVN_PREREC_PROBE=<mode>, measurement only, off by default; needs
/// EDEN_NX2_NVN_TLRING for its window and dump). Question it answers: would recording the lists
/// that sit behind the frame's CPU-event wait before the game's CPU signal read different guest
/// bytes on the CPU than recording them after it, as today.
///
/// The replay's record-time guest reads (every place the replay thread reads guest bytes on the
/// CPU, or decides from guest write tracking which guest bytes a GPU copy brings in) call
/// NoteRead with a class. The gating waits are the event waits the replay suspends on whose event
/// the game also signals from the CPU; the reads between two passes of one form the frame's read
/// set R(f).
///  - Decision classes (pool mirrors, Backing::Mirror pools, takeover and sparse-split trackers):
///    the replay's own tracker write epochs (pages that turned dirty) are sampled at the
///    suspension, the game's wake from its blocking nvnSyncWait, the CPU signal and the pass, and
///    polled every ~100 us with a time stamp; kept per pool.
///  - Byte-captured classes, tier 1 (uniform/handle/copy/index/event words): a fast probe thread
///    snapshots the predicted read set (union of the last complete read sets) at the trigger
///    (mode 1: the suspension; mode 3, sanity check: right after the previous signal) and diffs
///    and refreshes it at the wake, the signal and the end of R(f).
///  - Tier 2 (texture lines and bytes, mirror copy pages, uploads): snapshot right after the
///    previous signal (a superset of W), diffed at the suspension and the signal.
/// EDEN_NX2_NVN_PREREC_WATCH=<va hex>:<size> also polls one range with time stamps. Results are
/// written next to the TLRING dump (<tlring path>.prerec.tsv) at the TLRING's last present.
/// Preallocated tables, no log line per event, nothing
/// recorded or submitted changes, no pipeline key or shader change.
namespace VideoCommon::Nvn::PreRec {

/// Classes of record-time guest reads (see the dump's legend and the inventory).
enum class Cls : u8 {
    MirrorCheck = 0,   ///< Backing::Mirror range compared (hash/trust) at record time (pages)
    MirrorCopy = 1,    ///< Backing::Mirror range whose changed pages are memcpy'd to staging (pages)
    VramMirror = 2,    ///< VRAM pool mirror currency decision (tracker) before a GPU copy-in
    VramWanted = 3,    ///< VRAM pool mirror wanted-set copy-in decision (whole pool)
    TexSig = 4,        ///< texture coherency signature lines (256 x 64 B) or a small whole storage
    TexFull = 5,       ///< texture coherency full hash (whole storage)
    TexBytes = 6,      ///< texture bytes read through GuestBytes (upload, CPU copy, classify)
    TexGather = 7,     ///< texture bytes gathered from a virtual pool's physical pools
    TexTakeover = 8,   ///< overlap rule: tracker asked whether the CPU rewrote GPU-taken bytes
    UniformWord = 9,   ///< a uniform word read on the CPU (bindless handle, ref, offset words)
    HandleMem = 10,    ///< a handle word the shader loads from a storage table, read on the CPU
    OracleUniform = 11, ///< uniform ranges frozen for a background pipeline compile
    UniformCopy = 12,  ///< private uniform copy (inline-updated or misaligned binding)
    GenIndex = 13,     ///< game indices read for CPU index conversion (quads, fans, loops)
    PoolUpload = 14,   ///< GPU-only pool upload / host-write upload read on the CPU
    EventWord = 15,    ///< event word the event model read from guest memory
    TexReadback = 16,  ///< ReadTexels read-modify-write of a storage's guest bytes
    SparseGrowth = 17, ///< sparse split granules imported because the tracker saw a CPU write
    Count = 18,
};

namespace Detail {
extern const int mode;
/// EDEN_NX2_NVN_RECORD_AHEAD: this (replay) thread records ahead of a CPU-gated wait right now.
extern thread_local bool ra_ahead;
/// EDEN_NX2_NVN_HEADTRACE: this (replay) thread records an op of a traced head (the read classes
/// it touches are collected in ra_trace_fired, a bit per Cls).
extern thread_local bool ra_trace;
extern thread_local u32 ra_trace_fired;
} // namespace Detail

/// Record-ahead read log: every NoteRead while recording ahead lands here (per class counts) and
/// becomes this thread's read scope: the range of the latest note of the op being recorded. Every
/// replay-thread guest access while recording ahead (guest memory wrapper, Impl::GuestPointer
/// with its real size, Pool::HostRead / HostRaw) must lie inside the scope, else it is counted as
/// unlogged. The scope is cleared at every op boundary (RaResetScope).
void RaNote(Cls cls, u64 va, u64 size);
/// Whether [va, va + size) lies in the current scope.
[[nodiscard]] bool RaCovered(u64 va, u64 size) noexcept;
/// The access [va, va + size) is about to happen: counted as unlogged when this thread records
/// ahead and the scope does not cover it.
void RaCheck(u64 va, u64 size) noexcept;
/// Ends the scope (op boundary).
void RaResetScope() noexcept;
/// A host-mapping lookup of `va` (GuestMemory::MappingPointer / GuestViewPointer: the address is
/// used as an identity or for a host import, no guest byte is read through it on the CPU): not a
/// read, never unlogged; counted (mapping_lookups) while this thread records ahead.
void RaMapping(u64 va) noexcept;
/// Diagnostic: the call sites (stack offsets into the executable) of this thread's unlogged
/// accesses / mapping lookups made ahead so far, one entry each with a count.
[[nodiscard]] std::string RaUnloggedSites();
[[nodiscard]] std::string RaMappingSites();
struct RaReadStats {
    std::array<u64, static_cast<size_t>(Cls::Count)> calls{};
    std::array<u64, static_cast<size_t>(Cls::Count)> bytes{};
    std::array<u64, static_cast<size_t>(Cls::Count)> last_va{}; ///< identity of the latest note
    u64 unlogged{};
    u64 first_unlogged{};
    u64 last_unlogged{}; ///< va of the latest unlogged access
    u64 last_unlogged_size{};
    u64 mapping_lookups{}; ///< host-mapping lookups while recording ahead (RaMapping)
};
/// Counts so far (replay thread).
[[nodiscard]] RaReadStats& RaStats() noexcept;

[[nodiscard]] inline bool On() noexcept {
    return Detail::mode != 0;
}

/// Recording reads now (probe on and the TLRING window armed).
[[nodiscard]] inline bool Active() noexcept {
    return Detail::mode != 0 && Common::TlRing::Armed();
}

void NoteReadSlow(Cls cls, u64 va, u64 size, u64 id);
void NoteLinesSlow(Cls cls, u64 va, u64 size, u64 id);

/// A record-time read of [va, va + size) (guest VA). `id`: pool serial or another identity.
inline void NoteRead(Cls cls, u64 va, u64 size, u64 id = 0) noexcept {
    if (Detail::ra_ahead) [[unlikely]] {
        RaNote(cls, va, size);
    }
    if (Detail::ra_trace) [[unlikely]] {
        Detail::ra_trace_fired |= 1u << static_cast<u32>(cls);
    }
    if (Active()) [[unlikely]] {
        NoteReadSlow(cls, va, size, id);
    }
}

/// Whether a NoteRead would be used by anyone (the probe, the record-ahead read log or the head
/// trace): for call sites that compute the noted range only when it is wanted.
[[nodiscard]] inline bool NoteWanted() noexcept {
    return Detail::ra_ahead || Detail::ra_trace || Active();
}

/// The 64-byte lines SampleGuestBytes reads of a storage of `size` bytes at `va` (the whole
/// range when it is at most 256 lines).
inline void NoteLines(Cls cls, u64 va, u64 size, u64 id = 0) noexcept {
    if (Detail::ra_ahead) [[unlikely]] {
        RaNote(cls, va, size);
    }
    if (Detail::ra_trace) [[unlikely]] {
        Detail::ra_trace_fired |= 1u << static_cast<u32>(cls);
    }
    if (Active()) [[unlikely]] {
        NoteLinesSlow(cls, va, size, id);
    }
}

/// While one exists on this thread, NoteRead(TexBytes) calls are dropped (the caller notes the
/// bytes it really reads itself).
class SuppressTexBytes {
public:
    SuppressTexBytes();
    ~SuppressTexBytes();
    SuppressTexBytes(const SuppressTexBytes&) = delete;
    SuppressTexBytes& operator=(const SuppressTexBytes&) = delete;
};
[[nodiscard]] bool TexBytesSuppressed() noexcept;

/// Guest memory the probe thread reads (the replay's; null detaches).
void SetGuest(GuestMemory* memory);

/// A guest write tracker the replay decides from (its write epoch: pages that turned dirty).
/// Sampled at the suspension, the game's wake, the CPU signal and the pass of the gating wait.
void RegisterEpoch(u64 va, u64 size, u64 serial, Cls cls,
                   std::shared_ptr<const std::atomic<u64>> epoch);

/// Game side, once per present, after Common::TlRing::OnPresent.
void OnPresent(u64 count);

// Replay thread.
void OnWaitBlocked(u64 address, u32 mode, u32 value, u64 seq, u32 queue);
void OnWaitPassed(u64 address, u32 value);
void OnEventCpu(u64 address);
void NoteOpSlow(u16 op);
inline void NoteOp(u16 op) noexcept {
    if (Active()) [[unlikely]] {
        NoteOpSlow(op);
    }
}

// Game side.
void OnSyncWaitBegin(u64 fence, bool blocking, u32 tid);
void OnSyncWaitEnd(u64 fence, u32 tid);
void NoteSignalTid(u32 tid);
/// nvnEventSignal: `value` is the word written (after an ADD), before the write lands.
void OnEventSignal(u64 address, u32 written, u32 mode);
void OnGameCall(const void* handler, u32 tid);
/// nvnMemoryPoolFlushMappedRange (flush) / InvalidateMappedRange: guest VA range, guest thread.
void OnMappedRange(bool flush, u64 va, u64 size, u32 tid);
void NameHandler(const void* handler, std::string_view name);

} // namespace VideoCommon::Nvn::PreRec
