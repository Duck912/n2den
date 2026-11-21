// SPDX-FileCopyrightText: Copyright 2026 n2den Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "common/common_types.h"

namespace VideoCommon::Nvn {
class GuestMemory;
}

namespace Vulkan::Nvn {

/// One CPU-origin storage the texture coherency gate looks after (EDEN_NVN_TEX_PREHASH). Shared by
/// the storage (TexGateLink), the replay's registry and the worker's published list; the worker
/// never touches the Storage itself, only guest memory at `address` through GuestMemory::Read (a
/// storage or pool destroyed meanwhile leaves the worker hashing bytes nobody asks about).
struct TexPrehashSlot {
    TexPrehashSlot(u64 address_, u64 size_, bool batch_lines_)
        : address{address_}, size{size_}, batch_lines{batch_lines_} {}
    // Fixed (a storage's pool, offset and size never change).
    const u64 address;
    const u64 size;
    const bool batch_lines; ///< Impl::mario_kart_world (SampleGuestBytes' batch_lines)

    // Written by the replay thread, read by the worker (release / acquire).
    /// Generation of the last Arm of the storage's pages in its pool's `tex_track`; 0: not armed
    /// (nothing may skip). The worker reads it before the bytes.
    std::atomic<u64> arm_gen{0};
    /// The newest submission that writes guest bytes overlapping the storage (Recorder::
    /// NotePoolWrite, at record time). A pre-hash counts only if Impl::completed_seq had reached
    /// it before the worker read the bytes.
    std::atomic<u64> block_seq{0};
    /// The check schedule, for the worker to choose what to hash (hints only: a pre-hash of the
    /// wrong kind is a miss, never a different decision).
    std::atomic<u64> next_check_tick{0};
    std::atomic<u64> next_full_tick{~u64{0}};
    std::atomic<bool> every_check{false};
    /// The check clock at the storage's last use (DetectGuestChange, before its schedule test):
    /// the worker hashes only storages used recently (a due storage nobody draws with is not
    /// checked, so hashing it is wasted).
    std::atomic<u64> last_use_tick{0};
    std::atomic<bool> dead{false}; ///< the storage is gone

    // The worker's record: a seqlock (odd while written; the worker is the only writer).
    std::atomic<u32> rec_seq{0};
    std::atomic<u64> rec_arm_gen{0};    ///< arm_gen read before the bytes (0: no record)
    std::atomic<u64> rec_block_ok{0};   ///< Impl::completed_seq read before the bytes
    std::atomic<u64> rec_epoch{0};      ///< Impl::split_raw_epoch read before the bytes
    std::atomic<u64> rec_signature{0};  ///< SampleGuestBytes of the bytes
    std::atomic<u64> rec_full{0};       ///< HashGuestBytes of the bytes (when rec_has_full)
    std::atomic<bool> rec_has_full{false};
    std::atomic<u64> rec_read_ns{0};    ///< steady clock just before the worker read the bytes

    // Replay thread only.
    u32 listed_frame{};      ///< TexPrehash::frame it was put on this frame's check list
    u64 last_check_tick{};   ///< for the registry's pruning
    bool registered{false};  ///< in TexPrehash::registry
};

/// Storage member: marks the slot dead when the storage goes.
struct TexGateLink {
    std::shared_ptr<TexPrehashSlot> slot;
    TexGateLink() = default;
    TexGateLink(const TexGateLink&) = delete;
    TexGateLink& operator=(const TexGateLink&) = delete;
    ~TexGateLink() {
        if (slot) {
            slot->dead.store(true, std::memory_order_relaxed);
        }
    }
};

/// Texture coherency gate (EDEN_NVN_TEX_PREHASH=1, off by default; EDEN_NVN_NO_TEX_PREHASH=1
/// wins). Today every check of a CPU-origin storage (Impl::DetectGuestChange) hashes its guest
/// bytes on the replay thread (a ~16 KiB signature, and the whole storage for small uncompressed
/// ones and on a schedule): ~220 checks and ~16 MiB a frame in Ocarina of Time, and the checks
/// find nothing changed (cpu changes: 2 per boot). Here a worker thread hashes the storages due
/// soon, after every raw-storing submission completed (Impl::split_raw_epoch) and every present,
/// with exactly the functions and bytes today's check uses, recording what it read before the
/// bytes; the replay skips the hashing at a check only when it can prove the bytes did not change
/// between the worker's read and the check, and the worker's values equal the stored ones (the
/// "unchanged" result today's check would give). Every decision and every piece of bookkeeping is
/// then the one today's "unchanged" branch takes (backoff, unchanged_checks, next_full_tick, the
/// full-compare byte budget); the replay decides `full` exactly as before and a pre-hash without
/// the full hash cannot stand in for a full compare.
///
/// The proof, per check (Context::Impl::TexGateDecide):
///  (a) guest CPU writes: the storage's pages are armed in a GuestMemory::Track registration of
///      its pool (`Pool::tex_track`, separate from every other registration) before the worker's
///      read (rec_arm_gen == arm_gen) and none of them is dirty at the check (the registration's
///      write epoch unchanged since the arm, else GuestMemory::AnyWritten). Pages of a storage are
///      only re-armed together with disarming every other gated storage over the same pages, so
///      "not dirty" means "not written since this storage's arm". Remaps mark armed pages dirty.
///  (b) GPU writes this replay records into guest memory (copies, fills, reports, written storage
///      ranges, mirror write-backs: Recorder::NotePoolWrite) raise block_seq of every gated
///      storage they overlap (and the pool's tex_last_write_seq, which a newly armed storage
///      starts from); a pre-hash counts only if completed_seq had reached block_seq before the
///      worker's read. completed_seq is published after the write-backs and counter reports.
///  (c) raw GPU stores (untracked, any imported pool): rec_epoch must equal split_raw_epoch at
///      the check. A raw-storing submission still running at the check races with today's hash
///      as much as with the pre-hash (the verify mode calls that the race class).
///  (d) untracked host writes: a guest flush or host service write (NotifyCpuWriteNow) and the
///      texel flush (FlushStorage) disarm every storage they overlap.
///  (e) emulation stop (GuestMemory::Stopped): today's path. Virtual pools, sparse, GPU-origin
///      (`watching`) and dynamic storages: today's path, never armed. A storage the CPU keeps
///      writing (MAX_WRITE_REARMS re-arms after writes) is not armed again.
/// EDEN_NVN_TEX_PREHASH_VERIFY=1: whenever the gate skips, today's hashes are taken too and every
/// "today says changed" is counted with the storage's identity (decisions follow the gate, as
/// with the mirror pre-hash). EDEN_NVN_TEX_PREHASH_INJECT=1 (with VERIFY): every INJECT_PRESENTS
/// presents one guest byte of a gated storage is flipped through the host pointer (untracked)
/// right before such a verify hash, which must count it, and restored after.
class TexPrehash {
public:
    static constexpr u32 MAX_WRITE_REARMS = 3;

    /// Null unless the switch is on.
    [[nodiscard]] static std::unique_ptr<TexPrehash> Create(
        std::atomic<u64>& split_raw_epoch, std::atomic<u64>& completed_seq,
        std::atomic<VideoCommon::Nvn::GuestMemory*>& guest);
    TexPrehash(std::atomic<u64>& split_raw_epoch, std::atomic<u64>& completed_seq,
               std::atomic<VideoCommon::Nvn::GuestMemory*>& guest, bool verify, bool inject);
    ~TexPrehash();
    TexPrehash(const TexPrehash&) = delete;
    TexPrehash& operator=(const TexPrehash&) = delete;

    /// Completion thread, after it moved split_raw_epoch.
    void Wake();
    /// Replay thread, once per present (`tick`: the check clock's present count).
    void OnPresent(u64 present_index, u64 tick);
    /// Replay thread: the slot was looked at by a check (puts it on this frame's list and in the
    /// registry).
    void NoteCheck(const std::shared_ptr<TexPrehashSlot>& slot, u64 tick);
    /// Replay thread: a fresh arm generation.
    [[nodiscard]] u64 NextArmGen() {
        return ++arm_gen_clock;
    }

    [[nodiscard]] bool Verify() const noexcept {
        return verify;
    }
    /// The present count the worker chooses the due storages by.
    [[nodiscard]] u64 TickHint() const noexcept {
        return tick_hint.load(std::memory_order_relaxed);
    }

    /// A consistent copy of the worker's record.
    struct Record {
        u64 arm_gen{};
        u64 block_ok{};
        u64 epoch{};
        u64 signature{};
        u64 full{};
        bool has_full{};
        u64 read_ns{};
    };
    [[nodiscard]] static bool ReadRecord(const TexPrehashSlot& slot, Record& out);

    // Counters (replay thread), one window per WINDOW_PRESENTS presents.
    struct Counts {
        u64 checks{};        ///< checks of storages that reached the gate (not watching/virtual)
        u64 tick_offset[3]{}; ///< check tick - TickHint(): 0, 1, other
        u64 eligible{};      ///< of them: armed-eligible storages (CPU origin, not dynamic...)
        u64 skips{};         ///< hashing skipped
        u64 skip_full{};     ///< of them standing in for a full compare
        u64 not_armed{};
        u64 dirty{};         ///< the tracker saw a write since the arm
        u64 any_written_calls{};
        u64 no_record{};
        u64 arm_mismatch{};
        u64 epoch_mismatch{};
        u64 blocked{};       ///< completed_seq had not reached block_seq at the worker's read
        u64 no_full{};       ///< full compare due, the record has no full hash
        u64 said_changed{};  ///< the record differs from the stored hashes: today's check
        u64 stopped{};
        u64 arms{};
        u64 rearms_after_write{};
        u64 gave_up{};       ///< storages not armed again (written too often)
        u64 disarmed_by_overlap{};
        u64 disarmed_by_host_write{};
        u64 blocks_raised{};
        u64 verified{};
        u64 bug{};
        u64 race{};
        u64 verify_stopped{};
        u64 injected{};
        u64 injected_found{};
        u64 injected_missed{};
    };
    Counts window{};
    Counts total{};
    /// Tracker faults: sum of write-epoch growth of the tex_track registrations (set by the
    /// replay at each window from the live pools plus the ones destroyed).
    u64 faults_window{};
    u64 faults_total{};

    // Verify / inject (replay thread).
    /// Whether an injection is due now; if so, flips the byte at `host` (guest view, untracked).
    bool BeginInject(u8* host);
    /// Restores the byte; `found`: today's check said changed.
    void EndInject(bool found);
    struct DisagreementInfo {
        u64 pool_serial;
        u64 address;
        u64 size;
        u32 width;
        u32 height;
        s32 format;
        bool full;
        const Record* record;
        u64 epoch_now;
        u64 completed_now;
        u64 raw_submitted;
        u64 block_seq;
        bool stopped;
        bool signature_differs;
    };
    /// Counts and logs a "today says changed" after a skip. Returns the class name.
    void NoteDisagreement(const DisagreementInfo& info);

private:
    void WorkerMain();
    void LogWindow(u64 frames);
    void FoldWindow();

    std::atomic<u64>& epoch;
    std::atomic<u64>& completed;
    std::atomic<VideoCommon::Nvn::GuestMemory*>& guest;
    const bool verify;
    const bool inject;

    // Replay thread.
    u64 arm_gen_clock{0};
    u32 frame{1};
    std::vector<std::shared_ptr<TexPrehashSlot>> checked_this_frame;
    std::vector<std::shared_ptr<TexPrehashSlot>> registry;
    u64 window_start{};
    u64 next_inject{};
    bool inject_due{false};
    u8* inject_where{};
    u8 inject_old{};
    u64 identities_logged{};

    // Worker.
    using List = std::vector<std::shared_ptr<TexPrehashSlot>>;
    std::mutex mutex;
    std::condition_variable cv;
    bool stop{false};
    std::shared_ptr<const List> published; ///< under `mutex`
    u64 published_gen{};                   ///< under `mutex`
    std::atomic<u64> tick_hint{0};
    std::atomic<u64> worker_passes{0};
    std::atomic<u64> worker_hashed{0};
    std::atomic<u64> worker_full{0};
    std::atomic<u64> worker_bytes{0};
    std::atomic<u64> worker_busy_ns{0};
    std::thread worker;
};

} // namespace Vulkan::Nvn
