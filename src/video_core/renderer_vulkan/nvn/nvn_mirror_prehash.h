// SPDX-FileCopyrightText: Copyright 2026 n2den Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
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

struct Pool;

/// Per split mirror pool: the worker's pre-hashes, one per Pool::MIRROR_PAGE page. Kept alive by
/// the pool (Pool::prehash) and by the worker's page list, so a pool destroyed while the worker
/// walks it leaves the worker writing slots nobody reads.
struct MirrorPrehashSlots {
    static constexpr u64 INVALID = ~u64{0};
    struct Slot {
        /// Impl::split_raw_epoch the worker read before it read the page's bytes; INVALID while
        /// `hash` is being written (a seqlock: the worker is the only writer).
        std::atomic<u64> epoch{INVALID};
        std::atomic<u64> hash{0};
    };
    /// EDEN_NVN_MIRROR_PREHASH_VERIFY only: when the worker read the page and a hash of each
    /// STRIPES-th of it, so a disagreement can say how much of the page differs.
    static constexpr size_t STRIPES = 8;
    struct VerifyInfo {
        std::atomic<u64> read_ns{0}; ///< steady clock just before the worker read the bytes
        std::array<std::atomic<u32>, STRIPES> stripes{};
    };
    MirrorPrehashSlots(size_t count_, bool verify)
        : count{count_}, slots{std::make_unique<Slot[]>(count_)},
          verify_info{verify ? std::make_unique<VerifyInfo[]>(count_) : nullptr},
          listed(count_, 0) {}
    const size_t count;
    const std::unique_ptr<Slot[]> slots;
    const std::unique_ptr<VerifyInfo[]> verify_info; ///< null unless verify
    // Replay thread only.
    std::vector<u32> listed; ///< frame stamp the page was put on the building list
    u32 group_frame{};       ///< frame stamp of `group_index`
    u32 group_index{};       ///< this pool's group in the building list
};

/// Split-pool mirror pre-hash (on by default; EDEN_NVN_MIRROR_PREHASH=0 opts out,
/// EDEN_NVN_NO_MIRROR_PREHASH=1 wins over everything; tested in Ocarina of Time Kokiri only). Every completed raw-storing submission moves Impl::split_raw_epoch, after which every
/// trusted page of a split mirror pool is hashed again before its next use (Ocarina of Time:
/// ~10,000 pages a frame, none of them changed). A worker thread, woken by that bump, hashes the
/// pages the replay hashed in that class during the last frame, in their use order, recording the
/// epoch it read BEFORE it read any byte. The replay (MirrorNeedsUpload / UploadRange), for a page
/// of that class whose pre-hash carries the pool's current split_raw_epoch_seen and equals the
/// page's stored hash, takes it as its hash; anything else (no pre-hash, an older or newer epoch, a
/// pre-hash that differs) is hashed on the replay thread as before. The bookkeeping written from
/// the hash is unchanged.
///
/// Why this takes the same decisions: the pre-hash's epoch equal to the pool's means the worker
/// started reading after every raw-storing submission the replay counted had completed. Between
/// the worker's read and the use a page can still change through (a) a guest CPU write: the take
/// before the walk untrusts it, so it is not of this class; (b) a raw store of a submission that
/// completes later: its bump makes the epochs differ once the replay adopts it, and before that
/// the replay's own hash is not ordered after those stores either; (c) a writer neither the
/// tracker nor the epoch sees (a counter report written at completion onto a trusted page): today
/// the re-hash would catch it, here it waits for the next epoch. (d) Unmapping or remapping guest
/// pages of a live pool: Memory::UnmapRegion / MapMemoryRegion mark the tracker's armed pages
/// dirty (PageTrackerState::OnRemap) like a CPU write, so (a) applies. A pool destroyed and
/// another made at the same address do not share pre-hashes (the slots belong to the Pool).
/// (e) Emulation stop: guest reads fail from Runtime::Shutdown on; once GuestMemory::Stopped the
/// replay hashes itself (today's path). EDEN_NVN_MIRROR_PREHASH_VERIFY=1
/// hashes again at the use whenever a pre-hash decided and counts every disagreement (with pool,
/// page, consumer, the race facts of VerifyContext and what the bytes and mapping look like);
/// EDEN_NVN_MIRROR_PREHASH_INJECT=1 (with VERIFY) flips one guest byte of such a page right before
/// its verify hash every INJECT_PRESENTS presents, which the verify must count, and restores it.
class MirrorPrehash {
public:
    /// Null unless the switch is on.
    [[nodiscard]] static std::unique_ptr<MirrorPrehash> Create(
        std::atomic<u64>& split_raw_epoch, std::atomic<VideoCommon::Nvn::GuestMemory*>& guest);
    MirrorPrehash(std::atomic<u64>& split_raw_epoch,
                  std::atomic<VideoCommon::Nvn::GuestMemory*>& guest, bool verify, bool inject);
    ~MirrorPrehash();
    MirrorPrehash(const MirrorPrehash&) = delete;
    MirrorPrehash& operator=(const MirrorPrehash&) = delete;

    /// Completion thread, after it moved split_raw_epoch.
    void Wake();

    enum class Result : u8 {
        Unchanged, ///< a current pre-hash equals the stored hash: the page is unchanged
        Changed,   ///< a current pre-hash differs: hash on the replay thread as before
        Miss,      ///< no current pre-hash: hash on the replay thread as before
    };
    /// Replay thread, for a raw-epoch page (valid, trusted, its raw_epoch behind the pool's
    /// split_raw_epoch_seen `seen`) whose stored hash is `stored`. Lists the page for the next
    /// pass either way.
    [[nodiscard]] Result Use(Pool& pool, u64 page, u64 stored, u64 seen);
    /// Verify mode, after Use returned Unchanged: BeginVerify flips one guest byte of the page when
    /// an injection is due (returns whether it did); EndVerify takes today's hash `now` (computed
    /// in between), counts a disagreement and restores the byte.
    /// `raw_outstanding`: a raw-storing submission was recorded and has not completed;
    /// `consumer`: MirrorRawStats::Consumer of the range.
    bool BeginVerify(Pool& pool, u64 page);
    struct VerifyContext {
        u64 seen;      ///< the pool's split_raw_epoch_seen (= the pre-hash's epoch)
        u64 completed; ///< Impl::split_raw_epoch, loaded before `submitted`
        u64 submitted; ///< Impl::raw_flights_submitted
        u8 consumer;   ///< MirrorRawStats::Consumer of the range
        bool stopped;  ///< GuestMemory::Stopped after today's hash was taken
    };
    /// Returns false when the replay must decide with `now` instead (the emulation was found
    /// stopped: counted as `stopped`, not as a disagreement of the pre-hash).
    bool EndVerify(const Pool& pool, u64 page, u64 stored, u64 now, bool injected,
                   const VerifyContext& context);
    /// The emulation is stopping: a raw-epoch page was hashed on the replay thread, no pre-hash
    /// looked at.
    void NoteStopped() {
        ++window.stopped_fallback;
    }
    /// Verify mode, after Use returned Changed and today's hash equals the stored one.
    void NoteReverse() {
        ++window.reverse;
    }

    /// Replay thread, once per present: publishes the pages listed since the last present.
    void OnPresent(u64 present_index);

    [[nodiscard]] bool Verify() const noexcept {
        return verify;
    }

private:
    struct Group {
        std::shared_ptr<MirrorPrehashSlots> slots;
        u64 address{};
        u64 size{};
        std::vector<u32> pages; ///< in first-use order
    };
    struct List {
        std::vector<Group> groups;
    };

    // Replay thread.
    void ListPage(Pool& pool, u64 page);
    void LogWindow(u64 frames);
    void FoldWindow();

    void WorkerMain();

    std::atomic<u64>& epoch;
    std::atomic<VideoCommon::Nvn::GuestMemory*>& guest;
    const bool verify;
    const bool inject;

    // Replay thread.
    std::unique_ptr<List> building = std::make_unique<List>();
    u32 frame{1};
    u64 window_start{};
    u64 next_inject{};
    bool inject_due{false};
    u8* inject_where{};
    u8 inject_old{};
    struct Counts {
        u64 pages{};          ///< raw-epoch pages seen
        u64 hits{};           ///< decided by a pre-hash (it equalled the stored hash)
        u64 said_changed{};   ///< a current pre-hash differed from the stored hash: hashed again
        u64 miss_none{};      ///< no pre-hash for this page (never hashed, or being written)
        u64 miss_older{};     ///< the pre-hash is of an older epoch (the worker had not got there)
        u64 miss_newer{};     ///< the pre-hash is of a newer epoch than the pool adopted
        u64 verified{};
        u64 bug{};            ///< verify: pre-hash said unchanged, today's hash changed, and
                              ///< no raw-storing work could have stored in between
        u64 race{};           ///< the same while raw-storing work could have (VerifyContext:
                              ///< epoch past `seen`, or submitted > completed)
        u64 stopped{};        ///< the same with the emulation found stopped: decided by today's
                              ///< hash
        u64 stopped_fallback{}; ///< raw-epoch pages hashed on the replay thread: stopping
        u64 reverse{};        ///< pre-hash said changed, today's hash unchanged (no harm)
        u64 injected{};
        u64 injected_found{};
        u64 injected_missed{};
    };
    Counts window{};
    Counts total{};
    u64 identities_logged{};

    // Worker.
    std::mutex mutex;
    std::condition_variable cv;
    bool stop{false};
    std::shared_ptr<const List> published; ///< under `mutex`
    u64 published_gen{};                   ///< under `mutex`
    std::atomic<u64> worker_passes{0};
    std::atomic<u64> worker_restarts{0};
    std::atomic<u64> worker_pages{0};
    std::atomic<u64> worker_busy_ns{0};
    std::atomic<u64> worker_wake_to_done_ns{0};
    std::atomic<u64> worker_wake_to_done_count{0};
    std::atomic<u64> wake_ns{0}; ///< steady clock of the last Wake
    std::thread worker;
};

} // namespace Vulkan::Nvn
