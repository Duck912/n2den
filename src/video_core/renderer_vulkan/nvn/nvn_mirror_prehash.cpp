// SPDX-FileCopyrightText: Copyright 2026 n2den Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <string_view>

#include "common/logging.h"
#include "common/thread.h"
#include "video_core/nvn/guest_memory.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"
#include "video_core/renderer_vulkan/nvn/nvn_mirror_prehash.h"

namespace Vulkan::Nvn {

namespace {

constexpr u64 WINDOW_PRESENTS = 600;
/// EDEN_NVN_MIRROR_PREHASH_INJECT: one injected byte flip per this many presents.
constexpr u64 INJECT_PRESENTS = 300;
/// Recorder::GuestPageHash's value for a page it cannot read.
constexpr u64 UNREADABLE = 0x9E3779B97F4A7C15ull;
constexpr u64 MAX_IDENTITIES = 48;

bool EnvIs1(const char* name) {
    const char* const value = std::getenv(name);
    return value != nullptr && std::string_view{value} == "1";
}

u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

constexpr const char* CONSUMER_NAMES[] = {"other",   "vertex",  "index",
                                          "uniform", "storage", "indirect"};

} // Anonymous namespace

std::unique_ptr<MirrorPrehash> MirrorPrehash::Create(
    std::atomic<u64>& split_raw_epoch, std::atomic<VideoCommon::Nvn::GuestMemory*>& guest) {
    const bool verify = EnvIs1("EDEN_NVN_MIRROR_PREHASH_VERIFY");
    // On by default; EDEN_NVN_MIRROR_PREHASH=0 opts out (VERIFY turns it on regardless).
    const char* const switch_value = std::getenv("EDEN_NVN_MIRROR_PREHASH");
    if (switch_value != nullptr && std::string_view{switch_value} == "0" && !verify) {
        LOG_INFO(Render_Vulkan, "[nvn] mirror prehash: off (EDEN_NVN_MIRROR_PREHASH=0 opts out)");
        return nullptr;
    }
    if (EnvIs1("EDEN_NVN_NO_MIRROR_PREHASH")) {
        LOG_INFO(Render_Vulkan, "[nvn] mirror prehash: off (EDEN_NVN_NO_MIRROR_PREHASH=1)");
        return nullptr;
    }
    const bool inject = verify && EnvIs1("EDEN_NVN_MIRROR_PREHASH_INJECT");
    LOG_INFO(Render_Vulkan,
             "[nvn] mirror prehash: on (split mirror pages hashed again after a raw-storing "
             "submission are pre-hashed on a worker thread){}{}",
             verify ? "; VERIFY: every page a pre-hash decided is hashed again at its use" : "",
             inject ? "; INJECT: one guest byte flipped before a verify hash every 300 presents"
                    : "");
    return std::make_unique<MirrorPrehash>(split_raw_epoch, guest, verify, inject);
}

MirrorPrehash::MirrorPrehash(std::atomic<u64>& split_raw_epoch,
                             std::atomic<VideoCommon::Nvn::GuestMemory*>& guest_, bool verify_,
                             bool inject_)
    : epoch{split_raw_epoch}, guest{guest_}, verify{verify_}, inject{inject_} {
    worker = std::thread([this] {
        Common::SetCurrentThreadName("NvnPrehash");
        WorkerMain();
    });
}

MirrorPrehash::~MirrorPrehash() {
    {
        std::scoped_lock lock{mutex};
        stop = true;
    }
    cv.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
    // The last partial window (the replay and completion threads are gone: nothing else writes
    // `window` any more).
    FoldWindow();
    LOG_INFO(Render_Vulkan,
             "[nvn] mirror prehash totals (whole boot, last partial window included): raw-epoch "
             "pages {} hits {} changed {} miss none {} older {} newer {}; stopping: hashed on the "
             "replay thread {}; verify: verified {} bug {} race {} stopped {} reverse {} injected "
             "{} found {} missed {}",
             total.pages, total.hits, total.said_changed, total.miss_none, total.miss_older,
             total.miss_newer, total.stopped_fallback, total.verified, total.bug, total.race,
             total.stopped, total.reverse, total.injected, total.injected_found,
             total.injected_missed);
}

void MirrorPrehash::Wake() {
    wake_ns.store(NowNs(), std::memory_order_relaxed);
    {
        // Taken so the worker cannot miss the bump between its predicate and its wait.
        std::scoped_lock lock{mutex};
    }
    cv.notify_one();
}

void MirrorPrehash::ListPage(Pool& pool, u64 page) {
    std::shared_ptr<MirrorPrehashSlots>& slots = pool.prehash;
    if (!slots || slots->count != pool.mirror_pages.size()) {
        slots = std::make_shared<MirrorPrehashSlots>(pool.mirror_pages.size(), verify);
    }
    if (slots->listed[static_cast<size_t>(page)] == frame) {
        return;
    }
    slots->listed[static_cast<size_t>(page)] = frame;
    if (slots->group_frame != frame) {
        slots->group_frame = frame;
        slots->group_index = static_cast<u32>(building->groups.size());
        building->groups.push_back(Group{slots, pool.address, pool.size, {}});
    }
    building->groups[slots->group_index].pages.push_back(static_cast<u32>(page));
}

MirrorPrehash::Result MirrorPrehash::Use(Pool& pool, u64 page, u64 stored, u64 seen) {
    ListPage(pool, page);
    ++window.pages;
    MirrorPrehashSlots::Slot& slot = pool.prehash->slots[static_cast<size_t>(page)];
    // Seqlock read: the worker writes INVALID, the hash, then the epoch it read before the bytes.
    const u64 first = slot.epoch.load(std::memory_order_acquire);
    if (first == MirrorPrehashSlots::INVALID) {
        ++window.miss_none;
        return Result::Miss;
    }
    if (first != seen) {
        ++(first < seen ? window.miss_older : window.miss_newer);
        return Result::Miss;
    }
    const u64 hash = slot.hash.load(std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_acquire);
    if (slot.epoch.load(std::memory_order_relaxed) != first) {
        ++window.miss_none;
        return Result::Miss;
    }
    if (hash != stored) {
        ++window.said_changed;
        return Result::Changed;
    }
    ++window.hits;
    return Result::Unchanged;
}

bool MirrorPrehash::BeginVerify(Pool& pool, u64 page) {
    if (!inject || !inject_due) {
        return false;
    }
    VideoCommon::Nvn::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    if (memory == nullptr) {
        return false;
    }
    // Through the host pointer: a write the page tracker does not see (as a raw GPU store).
    u8* const where = memory->Pointer(pool.address + page * Pool::MIRROR_PAGE);
    if (where == nullptr) {
        return false;
    }
    inject_due = false;
    inject_where = where;
    inject_old = *static_cast<volatile u8*>(where);
    *static_cast<volatile u8*>(where) = static_cast<u8>(~inject_old);
    ++window.injected;
    return true;
}

bool MirrorPrehash::EndVerify(const Pool& pool, u64 page, u64 stored, u64 now, bool injected,
                              const VerifyContext& context) {
    ++window.verified;
    const char* const name = CONSUMER_NAMES[context.consumer < 6 ? context.consumer : 0];
    if (injected) {
        *static_cast<volatile u8*>(inject_where) = inject_old;
        inject_where = nullptr;
        ++(now != stored ? window.injected_found : window.injected_missed);
        if (identities_logged < MAX_IDENTITIES) {
            ++identities_logged;
            LOG_INFO(Render_Vulkan,
                     "[nvn] mirror prehash injected: pool {:#x} serial {} page +{:#x} consumer {} "
                     "-> verify {}",
                     pool.address, pool.serial, page * Pool::MIRROR_PAGE, name,
                     now != stored ? "found it" : "MISSED it");
        }
        return true;
    }
    if (now == stored) {
        return true;
    }
    const u64 current = epoch.load(std::memory_order_acquire);
    const bool race = current != context.seen || context.submitted > context.completed;
    const char* const cls = context.stopped ? "stopped" : race ? "race" : "bug";
    ++(context.stopped ? window.stopped : race ? window.race : window.bug);
    if (identities_logged >= MAX_IDENTITIES) {
        return !context.stopped;
    }
    ++identities_logged;
    // What the page looks like now, against what the worker read.
    const u64 begin = page * Pool::MIRROR_PAGE;
    const u64 length = (std::min)(Pool::MIRROR_PAGE, pool.size - begin);
    VideoCommon::Nvn::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    std::array<u8, Pool::MIRROR_PAGE> bytes{};
    const bool readable =
        memory != nullptr && memory->Read(pool.address + begin, bytes.data(), length);
    const bool stopped_now = memory != nullptr && memory->Stopped();
    const MirrorPrehashSlots* const slots = pool.prehash.get();
    u64 pre_epoch = MirrorPrehashSlots::INVALID;
    u64 read_age_us = ~u64{0};
    u32 stripes_differ = 0;
    u64 first_differ = ~u64{0};
    if (slots != nullptr && page < slots->count) {
        pre_epoch = slots->slots[page].epoch.load(std::memory_order_acquire);
        if (slots->verify_info) {
            const MirrorPrehashSlots::VerifyInfo& info = slots->verify_info[page];
            const u64 read_ns = info.read_ns.load(std::memory_order_relaxed);
            const u64 now_ns = NowNs();
            read_age_us = read_ns != 0 && read_ns <= now_ns ? (now_ns - read_ns) / 1000 : ~u64{0};
            if (readable) {
                const u64 stripe = Pool::MIRROR_PAGE / MirrorPrehashSlots::STRIPES;
                for (size_t i = 0; i < MirrorPrehashSlots::STRIPES; ++i) {
                    const u64 at = i * stripe;
                    if (at >= length) {
                        break;
                    }
                    const u32 value = static_cast<u32>(
                        GuestContentHash(bytes.data() + at, (std::min)(stripe, length - at)));
                    if (value != info.stripes[i].load(std::memory_order_relaxed)) {
                        ++stripes_differ;
                        first_differ = (std::min)(first_differ, at);
                    }
                }
            }
        }
    }
    // The host page the guest address maps to now, against the one recorded at backing (sparse
    // split pools keep their host runs; others: unknown).
    const u8* const pointer_now =
        memory != nullptr ? memory->Pointer(pool.address + begin) : nullptr;
    const char* mapping = "unknown";
    if (pool.sparse) {
        mapping = "no run";
        for (const Pool::SparseSplit::HostRun& run : pool.sparse->runs) {
            if (begin >= run.offset && begin < run.offset + run.size) {
                mapping = run.host == nullptr      ? "run without host"
                          : pointer_now == nullptr ? "unmapped now"
                          : pointer_now == run.host + (begin - run.offset) ? "same as at backing"
                                                                           : "MOVED";
                break;
            }
        }
    }
    LOG_WARNING(Render_Vulkan,
                "[nvn] mirror prehash DISAGREEMENT ({} class): pool {:#x} serial {} page +{:#x} "
                "consumer {} | stopped at verify {} now {} | today's hash {} | guest Read now {} | "
                "mapping {} | pool released {} | pre-hash epoch {} seen {} current epoch {} "
                "raw submitted {} completed {} | worker read {} us before | stripes differing {}/{} "
                "first +{:#x}",
                cls, pool.address, pool.serial, begin, name, context.stopped, stopped_now,
                now == UNREADABLE ? "UNREADABLE" : "readable", readable ? "ok" : "FAILED", mapping,
                pool.released.load(std::memory_order_relaxed), pre_epoch, context.seen, current,
                context.submitted, context.completed, read_age_us,
                readable ? stripes_differ : 0u, MirrorPrehashSlots::STRIPES,
                first_differ == ~u64{0} ? u64{0} : first_differ);
    return !context.stopped;
}

void MirrorPrehash::OnPresent(u64 present_index) {
    if (!building->groups.empty()) {
        // An empty frame (no epoch moved) keeps the last list.
        std::shared_ptr<const List> list{std::move(building)};
        {
            std::scoped_lock lock{mutex};
            published = std::move(list);
            ++published_gen;
        }
        cv.notify_one();
        building = std::make_unique<List>();
    }
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

void MirrorPrehash::LogWindow(u64 frames) {
    const double f = static_cast<double>(frames);
    const u64 passes = worker_passes.exchange(0, std::memory_order_relaxed);
    const u64 restarts = worker_restarts.exchange(0, std::memory_order_relaxed);
    const u64 pages = worker_pages.exchange(0, std::memory_order_relaxed);
    const u64 busy = worker_busy_ns.exchange(0, std::memory_order_relaxed);
    const u64 latency = worker_wake_to_done_ns.exchange(0, std::memory_order_relaxed);
    const u64 latency_n = worker_wake_to_done_count.exchange(0, std::memory_order_relaxed);
    const Counts& w = window;
    LOG_INFO(Render_Vulkan,
             "[nvn] mirror prehash: {} frames, per frame: raw-epoch pages {:.1f} hits {:.1f} "
             "({:.1f}%) changed {:.2f} miss none {:.1f} older {:.1f} newer {:.1f}; worker passes "
             "{:.2f} restarts {:.2f} pages {:.1f} busy {:.3f} ms, wake to pass end {:.3f} ms; "
             "stopping: hashed on the replay thread {}; verify: verified {} bug {} race {} "
             "stopped {} reverse {} injected {} found {} missed {}",
             frames, w.pages / f, w.hits / f,
             w.pages != 0 ? 100.0 * static_cast<double>(w.hits) / static_cast<double>(w.pages)
                          : 0.0,
             w.said_changed / f, w.miss_none / f, w.miss_older / f, w.miss_newer / f, passes / f,
             restarts / f, pages / f, busy / f / 1e6,
             latency_n != 0 ? static_cast<double>(latency) / static_cast<double>(latency_n) / 1e6
                            : 0.0,
             w.stopped_fallback, w.verified, w.bug, w.race, w.stopped, w.reverse, w.injected,
             w.injected_found, w.injected_missed);
    FoldWindow();
}

void MirrorPrehash::FoldWindow() {
    const Counts& w = window;
    total.pages += w.pages;
    total.hits += w.hits;
    total.said_changed += w.said_changed;
    total.miss_none += w.miss_none;
    total.miss_older += w.miss_older;
    total.miss_newer += w.miss_newer;
    total.verified += w.verified;
    total.bug += w.bug;
    total.race += w.race;
    total.stopped += w.stopped;
    total.stopped_fallback += w.stopped_fallback;
    total.reverse += w.reverse;
    total.injected += w.injected;
    total.injected_found += w.injected_found;
    total.injected_missed += w.injected_missed;
    window = Counts{};
}

void MirrorPrehash::WorkerMain() {
    u64 done_epoch = MirrorPrehashSlots::INVALID;
    u64 done_gen = 0;
    std::shared_ptr<const List> list;
    std::array<u8, Pool::MIRROR_PAGE> bytes;
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
        // Read before any byte: every raw-storing submission counted in it completed before.
        const u64 current = epoch.load(std::memory_order_acquire);
        done_epoch = current;
        VideoCommon::Nvn::GuestMemory* const memory = guest.load(std::memory_order_acquire);
        if (!list || memory == nullptr) {
            continue;
        }
        const u64 start = NowNs();
        const u64 woken = wake_ns.load(std::memory_order_relaxed);
        u64 hashed = 0;
        bool restarted = false;
        for (const Group& group : list->groups) {
            MirrorPrehashSlots& slots = *group.slots;
            for (const u32 page : group.pages) {
                if ((hashed & 15) == 0 && epoch.load(std::memory_order_relaxed) != current) {
                    restarted = true;
                    break;
                }
                if (page >= slots.count) {
                    continue;
                }
                MirrorPrehashSlots::Slot& slot = slots.slots[page];
                if (slot.epoch.load(std::memory_order_relaxed) == current) {
                    continue;
                }
                const u64 begin = u64{page} * Pool::MIRROR_PAGE;
                if (begin >= group.size) {
                    continue;
                }
                const u64 length = (std::min)(Pool::MIRROR_PAGE, group.size - begin);
                const u64 read_ns = slots.verify_info ? NowNs() : 0;
                const bool readable = memory->Read(group.address + begin, bytes.data(), length);
                const u64 hash = readable ? GuestContentHash(bytes.data(), length) : UNREADABLE;
                slot.epoch.store(MirrorPrehashSlots::INVALID, std::memory_order_relaxed);
                std::atomic_thread_fence(std::memory_order_release);
                slot.hash.store(hash, std::memory_order_relaxed);
                if (slots.verify_info) [[unlikely]] {
                    MirrorPrehashSlots::VerifyInfo& info = slots.verify_info[page];
                    info.read_ns.store(read_ns, std::memory_order_relaxed);
                    const u64 stripe = Pool::MIRROR_PAGE / MirrorPrehashSlots::STRIPES;
                    for (size_t i = 0; i < MirrorPrehashSlots::STRIPES; ++i) {
                        const u64 at = i * stripe;
                        const u32 value =
                            readable && at < length
                                ? static_cast<u32>(GuestContentHash(
                                      bytes.data() + at, (std::min)(stripe, length - at)))
                                : 0;
                        info.stripes[i].store(value, std::memory_order_relaxed);
                    }
                }
                slot.epoch.store(current, std::memory_order_release);
                ++hashed;
            }
            if (restarted) {
                break;
            }
        }
        const u64 end = NowNs();
        worker_passes.fetch_add(1, std::memory_order_relaxed);
        worker_pages.fetch_add(hashed, std::memory_order_relaxed);
        worker_busy_ns.fetch_add(end - start, std::memory_order_relaxed);
        if (restarted) {
            worker_restarts.fetch_add(1, std::memory_order_relaxed);
        } else if (hashed != 0 && woken != 0 && woken <= end) {
            worker_wake_to_done_ns.fetch_add(end - woken, std::memory_order_relaxed);
            worker_wake_to_done_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

} // namespace Vulkan::Nvn
