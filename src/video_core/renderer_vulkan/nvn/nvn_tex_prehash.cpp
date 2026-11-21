// SPDX-FileCopyrightText: Copyright 2026 n2den Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string_view>

#include "common/logging.h"
#include "common/thread.h"
#include "video_core/nvn/guest_memory.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"
#include "video_core/renderer_vulkan/nvn/nvn_tex_prehash.h"
#include "video_core/renderer_vulkan/nvn/nvn_upload.h"

namespace Vulkan::Nvn {
namespace {

constexpr u64 WINDOW_PRESENTS = 600;
/// EDEN_NVN_TEX_PREHASH_INJECT: one injected byte flip per this many presents.
constexpr u64 INJECT_PRESENTS = 300;
constexpr u64 MAX_IDENTITIES = 48;
/// The worker hashes storages due by the replay's last present count first, then the ones due one
/// tick later (the next frame's checks see that count or one more).
constexpr u64 DUE_AHEAD = 1;
/// Full hashes of larger storages are left to the replay thread (worker copy buffer bound).
constexpr u64 MAX_WORKER_FULL = u64{16} << 20;
/// The worker skips storages not used within this many ticks of its hint.
constexpr u64 USED_WITHIN = 2;
/// A registry slot not checked for this many ticks leaves the registry (it comes back at its
/// next check).
constexpr u64 REGISTRY_AGE = 128;

bool EnvIs1(const char* name) {
    const char* const value = std::getenv(name);
    return value != nullptr && std::string_view{value} == "1";
}

u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

} // Anonymous namespace

std::unique_ptr<TexPrehash> TexPrehash::Create(std::atomic<u64>& split_raw_epoch,
                                               std::atomic<u64>& completed_seq,
                                               std::atomic<VideoCommon::Nvn::GuestMemory*>& guest) {
    const bool verify = EnvIs1("EDEN_NVN_TEX_PREHASH_VERIFY");
    if (!EnvIs1("EDEN_NVN_TEX_PREHASH") && !verify) {
        return nullptr;
    }
    if (EnvIs1("EDEN_NVN_NO_TEX_PREHASH")) {
        LOG_INFO(Render_Vulkan, "[nvn] tex prehash: off (EDEN_NVN_NO_TEX_PREHASH=1)");
        return nullptr;
    }
    if (EnvIs1("EDEN_NX2_NVN_RECORD_AHEAD")) {
        // Record-ahead writes guest memory of its own and reads ahead of the signal: not combined.
        LOG_INFO(Render_Vulkan, "[nvn] tex prehash: off (not combined with record-ahead)");
        return nullptr;
    }
    const bool inject = verify && EnvIs1("EDEN_NVN_TEX_PREHASH_INJECT");
    LOG_INFO(Render_Vulkan,
             "[nvn] tex prehash: on (texture coherency checks proven unchanged skip their "
             "hashing; a worker thread pre-hashes the storages due){}{}",
             verify ? "; VERIFY: every skipped check is hashed again on the replay thread" : "",
             inject ? "; INJECT: one guest byte flipped before a verify hash every 300 presents"
                    : "");
    return std::make_unique<TexPrehash>(split_raw_epoch, completed_seq, guest, verify, inject);
}

TexPrehash::TexPrehash(std::atomic<u64>& split_raw_epoch, std::atomic<u64>& completed_seq,
                       std::atomic<VideoCommon::Nvn::GuestMemory*>& guest_, bool verify_,
                       bool inject_)
    : epoch{split_raw_epoch}, completed{completed_seq}, guest{guest_}, verify{verify_},
      inject{inject_} {
    worker = std::thread([this] {
        Common::SetCurrentThreadName("NvnTexPrehash");
        WorkerMain();
    });
}

TexPrehash::~TexPrehash() {
    {
        std::scoped_lock lock{mutex};
        stop = true;
    }
    cv.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
    FoldWindow();
    const Counts& t = total;
    LOG_INFO(Render_Vulkan,
             "[nvn] tex prehash totals (whole boot, last partial window included): checks {} "
             "eligible {} skips {} (full {}) | not armed {} dirty {} no record {} arm mismatch {} "
             "epoch mismatch {} blocked {} no full {} said changed {} stopped {} | arms {} "
             "re-arms after write {} gave up {} disarmed overlap {} host write {} blocks {} | "
             "tracker faults {} | verify: verified {} bug {} race {} stopped {} injected {} found "
             "{} missed {}",
             t.checks, t.eligible, t.skips, t.skip_full, t.not_armed, t.dirty, t.no_record,
             t.arm_mismatch, t.epoch_mismatch, t.blocked, t.no_full, t.said_changed, t.stopped,
             t.arms, t.rearms_after_write, t.gave_up, t.disarmed_by_overlap,
             t.disarmed_by_host_write, t.blocks_raised, faults_total, t.verified, t.bug, t.race,
             t.verify_stopped, t.injected, t.injected_found, t.injected_missed);
}

void TexPrehash::Wake() {
    {
        std::scoped_lock lock{mutex};
    }
    cv.notify_one();
}

bool TexPrehash::ReadRecord(const TexPrehashSlot& slot, Record& out) {
    const u32 first = slot.rec_seq.load(std::memory_order_acquire);
    if ((first & 1) != 0) {
        return false;
    }
    out.arm_gen = slot.rec_arm_gen.load(std::memory_order_relaxed);
    out.block_ok = slot.rec_block_ok.load(std::memory_order_relaxed);
    out.epoch = slot.rec_epoch.load(std::memory_order_relaxed);
    out.signature = slot.rec_signature.load(std::memory_order_relaxed);
    out.full = slot.rec_full.load(std::memory_order_relaxed);
    out.has_full = slot.rec_has_full.load(std::memory_order_relaxed);
    out.read_ns = slot.rec_read_ns.load(std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    return slot.rec_seq.load(std::memory_order_relaxed) == first && out.arm_gen != 0;
}

void TexPrehash::NoteCheck(const std::shared_ptr<TexPrehashSlot>& slot, u64 tick) {
    slot->last_check_tick = tick;
    if (slot->listed_frame != frame) {
        slot->listed_frame = frame;
        checked_this_frame.push_back(slot);
    }
    if (!slot->registered) {
        slot->registered = true;
        registry.push_back(slot);
    }
}

void TexPrehash::OnPresent(u64 present_index, u64 tick) {
    tick_hint.store(tick, std::memory_order_relaxed);
    // Published list: this frame's checks in their order (the next frame checks in about the
    // same order), then the rest of the registry.
    auto list = std::make_shared<List>();
    list->reserve(registry.size());
    for (std::shared_ptr<TexPrehashSlot>& slot : checked_this_frame) {
        list->push_back(std::move(slot));
    }
    checked_this_frame.clear();
    std::erase_if(registry, [&](const std::shared_ptr<TexPrehashSlot>& slot) {
        if (slot->dead.load(std::memory_order_relaxed) ||
            slot->last_check_tick + REGISTRY_AGE < tick) {
            slot->registered = false;
            return true;
        }
        return false;
    });
    for (const std::shared_ptr<TexPrehashSlot>& slot : registry) {
        if (slot->listed_frame != frame) {
            list->push_back(slot);
        }
    }
    {
        std::scoped_lock lock{mutex};
        published = std::move(list);
        ++published_gen;
    }
    cv.notify_one();
    ++frame;
    if (inject && present_index >= next_inject) {
        inject_due = true;
        next_inject = present_index + INJECT_PRESENTS;
    }
    if (window_start == 0) {
        window_start = present_index;
        return;
    }
    if (present_index - window_start >= WINDOW_PRESENTS) {
        LogWindow(present_index - window_start);
        window_start = present_index;
    }
}

bool TexPrehash::BeginInject(u8* host) {
    if (!inject || !inject_due || host == nullptr) {
        return false;
    }
    inject_due = false;
    inject_where = host;
    inject_old = *static_cast<volatile u8*>(host);
    *static_cast<volatile u8*>(host) = static_cast<u8>(~inject_old);
    ++window.injected;
    return true;
}

void TexPrehash::EndInject(bool found) {
    *static_cast<volatile u8*>(inject_where) = inject_old;
    inject_where = nullptr;
    ++(found ? window.injected_found : window.injected_missed);
    if (identities_logged < MAX_IDENTITIES) {
        ++identities_logged;
        LOG_INFO(Render_Vulkan, "[nvn] tex prehash injected: verify {}",
                 found ? "found it" : "MISSED it");
    }
}

void TexPrehash::NoteDisagreement(const DisagreementInfo& info) {
    const bool race = info.epoch_now != info.record->epoch || info.raw_submitted > info.epoch_now;
    const char* const cls = info.stopped ? "stopped" : race ? "race" : "bug";
    ++(info.stopped ? window.verify_stopped : race ? window.race : window.bug);
    if (identities_logged >= MAX_IDENTITIES) {
        return;
    }
    ++identities_logged;
    const u64 now_ns = NowNs();
    LOG_WARNING(Render_Vulkan,
                "[nvn] tex prehash DISAGREEMENT ({} class): pool serial {} address {:#x} size "
                "{:#x} {}x{} fmt {:#x} check {} ({} differs) | record arm {} block_ok {} epoch {} "
                "read {} us before | now epoch {} completed {} raw submitted {} block_seq {}",
                cls, info.pool_serial, info.address, info.size, info.width, info.height,
                info.format, info.full ? "full" : "signature",
                info.signature_differs ? "signature" : "full hash", info.record->arm_gen,
                info.record->block_ok, info.record->epoch,
                info.record->read_ns != 0 && info.record->read_ns <= now_ns
                    ? (now_ns - info.record->read_ns) / 1000
                    : ~u64{0},
                info.epoch_now, info.completed_now, info.raw_submitted, info.block_seq);
}

void TexPrehash::LogWindow(u64 frames) {
    const double f = static_cast<double>(frames);
    const u64 passes = worker_passes.exchange(0, std::memory_order_relaxed);
    const u64 hashed = worker_hashed.exchange(0, std::memory_order_relaxed);
    const u64 fulls = worker_full.exchange(0, std::memory_order_relaxed);
    const u64 bytes = worker_bytes.exchange(0, std::memory_order_relaxed);
    const u64 busy = worker_busy_ns.exchange(0, std::memory_order_relaxed);
    const Counts& w = window;
    LOG_INFO(Render_Vulkan,
             "[nvn] tex prehash: {} frames, per frame: checks {:.1f} (tick - hint 0: {:.1f} 1: "
             "{:.1f} other: {:.1f}) eligible {:.1f} skips {:.1f} "
             "({:.1f}% of checks; full {:.1f}) | not armed {:.1f} dirty {:.2f} no record {:.1f} "
             "arm mismatch {:.2f} epoch mismatch {:.1f} blocked {:.2f} no full {:.2f} said "
             "changed {:.2f} | AnyWritten calls {:.1f} | arms {} re-arms after write {} gave up "
             "{} disarmed overlap {} host write {} blocks {} | tracker faults {} | worker passes "
             "{:.2f} hashed {:.1f} (full {:.1f}) {:.2f} MiB busy {:.3f} ms | registry {} | "
             "stopped {} | verify: verified {} bug {} race {} stopped {} injected {} found {} "
             "missed {}",
             frames, w.checks / f, w.tick_offset[0] / f, w.tick_offset[1] / f,
             w.tick_offset[2] / f, w.eligible / f, w.skips / f,
             w.checks != 0 ? 100.0 * static_cast<double>(w.skips) / static_cast<double>(w.checks)
                           : 0.0,
             w.skip_full / f, w.not_armed / f, w.dirty / f, w.no_record / f, w.arm_mismatch / f,
             w.epoch_mismatch / f, w.blocked / f, w.no_full / f, w.said_changed / f,
             w.any_written_calls / f, w.arms, w.rearms_after_write, w.gave_up,
             w.disarmed_by_overlap, w.disarmed_by_host_write, w.blocks_raised, faults_window,
             passes / f, hashed / f, fulls / f, static_cast<double>(bytes) / f / 1048576.0,
             busy / f / 1e6, registry.size(), w.stopped, w.verified, w.bug, w.race,
             w.verify_stopped, w.injected, w.injected_found, w.injected_missed);
    FoldWindow();
}

void TexPrehash::FoldWindow() {
    const Counts& w = window;
    Counts& t = total;
    t.checks += w.checks;
    t.eligible += w.eligible;
    t.skips += w.skips;
    t.skip_full += w.skip_full;
    t.not_armed += w.not_armed;
    t.dirty += w.dirty;
    t.any_written_calls += w.any_written_calls;
    t.no_record += w.no_record;
    t.arm_mismatch += w.arm_mismatch;
    t.epoch_mismatch += w.epoch_mismatch;
    t.blocked += w.blocked;
    t.no_full += w.no_full;
    t.said_changed += w.said_changed;
    t.stopped += w.stopped;
    t.arms += w.arms;
    t.rearms_after_write += w.rearms_after_write;
    t.gave_up += w.gave_up;
    t.disarmed_by_overlap += w.disarmed_by_overlap;
    t.disarmed_by_host_write += w.disarmed_by_host_write;
    t.blocks_raised += w.blocks_raised;
    t.verified += w.verified;
    t.bug += w.bug;
    t.race += w.race;
    t.verify_stopped += w.verify_stopped;
    t.injected += w.injected;
    t.injected_found += w.injected_found;
    t.injected_missed += w.injected_missed;
    faults_total += faults_window;
    faults_window = 0;
    window = Counts{};
}

void TexPrehash::WorkerMain() {
    u64 done_epoch = ~u64{0};
    u64 done_gen = 0;
    std::shared_ptr<const List> list;
    std::vector<u8> bytes;
    while (true) {
        {
            std::unique_lock lock{mutex};
            cv.wait(lock, [&] {
                return stop || epoch.load(std::memory_order_acquire) != done_epoch ||
                       published_gen != done_gen;
            });
            if (stop) {
                return;
            }
            list = published;
            done_gen = published_gen;
        }
        done_epoch = epoch.load(std::memory_order_acquire);
        VideoCommon::Nvn::GuestMemory* const memory = guest.load(std::memory_order_acquire);
        if (!list || memory == nullptr) {
            continue;
        }
        const u64 start = NowNs();
        const u64 tick = tick_hint.load(std::memory_order_relaxed);
        u64 hashed = 0;
        u64 fulls = 0;
        u64 hashed_bytes = 0;
        for (u64 round = 0; round <= DUE_AHEAD; ++round)
        for (const std::shared_ptr<TexPrehashSlot>& slot_ptr : *list) {
            TexPrehashSlot& slot = *slot_ptr;
            if (slot.dead.load(std::memory_order_relaxed)) {
                continue;
            }
            const u64 next_check = slot.next_check_tick.load(std::memory_order_relaxed);
            if (round == 0 ? next_check > tick : next_check != tick + round) {
                continue;
            }
            if (slot.last_use_tick.load(std::memory_order_relaxed) + USED_WITHIN < tick) {
                continue;
            }
            const bool want_full =
                slot.size <= MAX_WORKER_FULL &&
                (slot.every_check.load(std::memory_order_relaxed) ||
                 slot.next_full_tick.load(std::memory_order_relaxed) <= tick + round);
            // Everything the replay compares against is read BEFORE the bytes: a store, write or
            // arm after these loads makes them differ from what the replay sees at its check.
            const u64 arm = slot.arm_gen.load(std::memory_order_acquire);
            if (arm == 0) {
                continue;
            }
            const u64 block = slot.block_seq.load(std::memory_order_acquire);
            const u64 raw_epoch = epoch.load(std::memory_order_acquire);
            const u64 done_seq = completed.load(std::memory_order_acquire);
            if (block > done_seq) {
                continue; // a write of it is still running: the record would not count
            }
            Record current;
            if (ReadRecord(slot, current) && current.arm_gen == arm && current.epoch == raw_epoch &&
                current.block_ok >= block && (current.has_full || !want_full)) {
                continue; // still good
            }
            const u64 read_ns = NowNs();
            bool readable = true;
            const u64 signature = SampleGuestBytesByQuiet(
                slot.size,
                [&](u64 offset, u64 count, u8* destination) {
                    if (!memory->Read(slot.address + offset, destination,
                                      static_cast<size_t>(count))) {
                        readable = false;
                        std::memset(destination, 0, static_cast<size_t>(count));
                    }
                },
                slot.batch_lines);
            u64 full = 0;
            bool has_full = false;
            if (readable && want_full) {
                bytes.resize(static_cast<size_t>(slot.size));
                if (memory->Read(slot.address, bytes.data(), bytes.size())) {
                    full = GuestContentHash(bytes.data(), slot.size);
                    has_full = true;
                    ++fulls;
                    hashed_bytes += slot.size;
                } else {
                    readable = false;
                }
            }
            hashed_bytes += (std::min)(slot.size, u64{16384});
            ++hashed;
            const u32 seq = slot.rec_seq.load(std::memory_order_relaxed);
            slot.rec_seq.store(seq + 1, std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_release);
            // Unreadable: no record (arm 0), the replay's own check decides.
            slot.rec_arm_gen.store(readable ? arm : 0, std::memory_order_relaxed);
            slot.rec_block_ok.store(done_seq, std::memory_order_relaxed);
            slot.rec_epoch.store(raw_epoch, std::memory_order_relaxed);
            slot.rec_signature.store(signature, std::memory_order_relaxed);
            slot.rec_full.store(full, std::memory_order_relaxed);
            slot.rec_has_full.store(has_full, std::memory_order_relaxed);
            slot.rec_read_ns.store(read_ns, std::memory_order_relaxed);
            slot.rec_seq.store(seq + 2, std::memory_order_release);
            if ((hashed & 15) == 0) {
                std::scoped_lock lock{mutex};
                if (stop) {
                    return;
                }
            }
        }
        worker_passes.fetch_add(1, std::memory_order_relaxed);
        worker_hashed.fetch_add(hashed, std::memory_order_relaxed);
        worker_full.fetch_add(fulls, std::memory_order_relaxed);
        worker_bytes.fetch_add(hashed_bytes, std::memory_order_relaxed);
        worker_busy_ns.fetch_add(NowNs() - start, std::memory_order_relaxed);
    }
}

} // namespace Vulkan::Nvn
