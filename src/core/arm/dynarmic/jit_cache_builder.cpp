// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/arm/dynarmic/jit_cache_builder.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"
#include "common/thread.h"
#include "core/arm/dynarmic/arm_dynarmic_64.h"
#include "core/arm/dynarmic/jit_block_cache.h"
#include "core/arm/dynarmic/jit_code_discovery.h"
#include "core/arm/dynarmic/jit_prebuild_list.h"

namespace Core {

namespace {

using Result = Dynarmic::A64::PrecompileResult;
constexpr size_t result_kinds = 5;

// File-only blocks are dropped from the building JIT's code buffer after this many builds (one full clear),
// so a multi-million-block list never holds more than a few tens of MiB of code per JIT.
constexpr size_t transient_clear_interval = 32768;

// The boot preload stops filling a core's code cache when this much of it is left, for the blocks the game
// reaches that the manifest does not list (and for code that is invalidated and placed again).
constexpr size_t preload_keep_free = size_t{96} << 20;

size_t ThreadCount(size_t jits) {
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    size_t count = std::min<size_t>(jits, hw > 2 ? hw - 2 : 1);
    if (const char* env = std::getenv("EDEN_JIT_CACHE_REBUILD_THREADS"); env && env[0] != '\0') {
        const long requested = std::strtol(env, nullptr, 10);
        if (requested > 0) {
            count = std::min<size_t>(jits, static_cast<size_t>(requested));
        }
    }
    return std::max<size_t>(count, 1);
}

struct Item {
    u64 descriptor;
    u64 end_pc;
    u64 code_hash;
};

struct PhaseCounts {
    std::array<std::atomic<size_t>, result_kinds> results{};
    std::atomic<u64> ns{0};  // summed over threads
};

} // namespace

void RebuildJitCache(JitBlockCache& cache, std::span<ArmDynarmic64* const> jits,
                     const JitCacheBuildProgress& progress) {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    const auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
    };
    if (jits.empty()) {
        return;
    }

    // Resident items: blocks seen in play (manifest, most placed first), then the list's executed entries. They
    // stay in the JIT that built them. File-only items: the rest of the list within the scope; written to the
    // records file and dropped from the JIT.
    const std::vector<JitBlockCache::ManifestEntry> manifest = cache.ManifestSnapshot();
    std::vector<Item> resident;
    resident.reserve(manifest.size());
    std::vector<u64> learned;  // manifest descriptors, sorted: list entries already listed there are skipped
    learned.reserve(manifest.size());
    for (const auto& e : manifest) {
        resident.push_back({e.descriptor, e.end_pc, e.code_hash});
        learned.push_back(e.descriptor);
    }
    std::sort(learned.begin(), learned.end());

    const JitBlockCache::PrebuildScope scope = JitBlockCache::GetPrebuildScope();
    std::string list_error;
    // Precompile "previously played areas" builds the manifest only; "entire game" adds the list.
    const std::unique_ptr<JitPrebuildList> list =
        cache.UsesPrebuildList() ? JitPrebuildList::Open(cache.PrebuildListPath(), cache.ProgramId(), &list_error)
                                 : nullptr;
    if (!list && !list_error.empty()) {
        LOG_WARNING(Core_ARM, "JIT prebuild list {} not used: {}", cache.PrebuildListPath().string(), list_error);
    }
    std::vector<u32> file_only;  // list indices
    size_t list_in_manifest = 0;
    size_t list_out_of_scope = 0;
    if (list) {
        using F = JitPrebuildList::Flags;
        for (size_t i = 0; i < list->Size(); ++i) {
            const JitPrebuildList::Entry e = list->At(i);
            if (std::binary_search(learned.begin(), learned.end(), e.descriptor)) {
                ++list_in_manifest;
            } else if (e.flags & F::Executed) {
                resident.push_back({e.descriptor, e.end_pc, e.guest_hash});
            } else if (scope == JitBlockCache::PrebuildScope::Full ||
                       (scope == JitBlockCache::PrebuildScope::Played && (e.flags & F::PlayedFunction))) {
                file_only.push_back(static_cast<u32>(i));
            } else {
                ++list_out_of_scope;
            }
        }
    }
    const size_t total = resident.size() + file_only.size();
    if (total == 0) {
        return;
    }
    const size_t thread_count = ThreadCount(jits.size());
    LOG_INFO(Core_ARM,
             "JIT cache build: {} blocks on {} threads before the game starts: {} resident ({} from the manifest, "
             "{} executed entries of the prebuild list), {} file-only from the prebuild list (scope {}; {} list "
             "entries already in the manifest, {} out of scope); records file {} MiB",
             total, thread_count, resident.size(), manifest.size(), resident.size() - manifest.size(),
             file_only.size(), JitBlockCache::PrebuildScopeName(scope), list_in_manifest, list_out_of_scope,
             cache.FileSize() >> 20);

    cache.BeginBulkStore();
    std::atomic<size_t> next_file{0};
    std::atomic<size_t> next_resident{0};
    std::atomic<size_t> done{0};
    std::atomic<size_t> running{thread_count};
    PhaseCounts file_counts;
    PhaseCounts resident_counts;
    // List entries built into a JIT and left there, per thread: added to the manifest afterwards
    // (JitBlockCache::AddResidentBlocks), because running them on that core never reports a placement.
    std::vector<std::vector<JitBlockCache::ManifestEntry>> resident_built(thread_count);
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (size_t t = 0; t < thread_count; ++t) {
        threads.emplace_back([&, t, jit = jits[t]] {
            Common::SetCurrentThreadName("JitCacheBuild");
            constexpr size_t chunk = 256;
            const auto run_phase = [&](std::atomic<size_t>& next, size_t count, PhaseCounts& counts, bool transient,
                                       auto&& item_at) {
                const auto phase_start = Clock::now();
                std::array<size_t, result_kinds> local{};
                size_t since_clear = 0;
                for (;;) {
                    const size_t begin = next.fetch_add(chunk, std::memory_order_relaxed);
                    if (begin >= count) {
                        break;
                    }
                    const size_t end = std::min(count, begin + chunk);
                    for (size_t i = begin; i < end; ++i) {
                        const Item item = item_at(i);
                        const Result r = jit->PrecompileBlock(item.descriptor, item.end_pc, item.code_hash);
                        ++local[std::min<size_t>(static_cast<size_t>(r), result_kinds - 1)];
                        if (!transient && r == Result::Built && i >= manifest.size()) {
                            resident_built[t].push_back({item.descriptor, item.end_pc, item.code_hash, 0, 0});
                        }
                        if (transient && r == Result::Built && ++since_clear >= transient_clear_interval) {
                            jit->ClearInstructionCache();  // carried out by the next Precompile
                            since_clear = 0;
                        }
                    }
                    done.fetch_add(end - begin, std::memory_order_relaxed);
                }
                if (transient) {
                    jit->ClearInstructionCache();  // drop the last file-only blocks before resident ones are built
                }
                for (size_t k = 0; k < result_kinds; ++k) {
                    counts.results[k].fetch_add(local[k], std::memory_order_relaxed);
                }
                counts.ns.fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                         Clock::now() - phase_start).count()),
                                    std::memory_order_relaxed);
            };
            if (list) {
                run_phase(next_file, file_only.size(), file_counts, true, [&](size_t i) {
                    const JitPrebuildList::Entry e = list->At(file_only[i]);
                    return Item{e.descriptor, e.end_pc, e.guest_hash};
                });
            }
            run_phase(next_resident, resident.size(), resident_counts, false,
                      [&](size_t i) { return resident[i]; });
            running.fetch_sub(1, std::memory_order_release);
        });
    }

    auto last_log = start;
    while (running.load(std::memory_order_acquire) != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
        const size_t now_done = std::min(done.load(std::memory_order_relaxed), total);
        if (progress) {
            progress(now_done, total);
        }
        if (Clock::now() - last_log >= std::chrono::seconds{2}) {
            last_log = Clock::now();
            LOG_INFO(Core_ARM, "JIT cache build: {} / {} blocks ({} ms); records file {} MiB", now_done, total,
                     ms(start, last_log), cache.FileSize() >> 20);
        }
    }
    for (auto& thread : threads) {
        thread.join();
    }
    const auto built_at = Clock::now();
    cache.EndBulkStore();
    size_t resident_added = 0;
    for (const auto& built : resident_built) {
        resident_added += cache.AddResidentBlocks(built);
    }
    if (resident_added != 0) {
        LOG_INFO(Core_ARM, "JIT cache build: {} prebuild list blocks left resident added to the manifest",
                 resident_added);
    }
    if (list) {
        cache.MarkPrebuilt(list->Header(), scope);
    }
    if (progress) {
        progress(total, total);
    }
    const auto line = [&](const char* name, size_t count, const PhaseCounts& c) {
        const auto get = [&](Result r) { return c.results[static_cast<size_t>(r)].load(); };
        LOG_INFO(Core_ARM,
                 "JIT cache build, {}: {} blocks, {} ms of thread time; {} built, {} already cached, {} skipped "
                 "(guest code changed), {} skipped (not mapped), {} not cacheable",
                 name, count, c.ns.load() / 1'000'000, get(Result::Built), get(Result::Present),
                 get(Result::GuestChanged), get(Result::Unreadable), get(Result::NotCacheable));
    };
    line("file-only", file_only.size(), file_counts);
    line("resident", resident.size(), resident_counts);
    const auto sum = [&](Result r) {
        return file_counts.results[static_cast<size_t>(r)].load() +
               resident_counts.results[static_cast<size_t>(r)].load();
    };
    // Same wording as the stage 1 rebuild line (scripts read it).
    LOG_INFO(Core_ARM,
             "JIT cache rebuild: {} ms ({} ms building, {} ms writing/indexing); {} blocks built, {} already "
             "cached, {} skipped (guest code changed), {} skipped (not mapped), {} not cacheable; {} threads; "
             "records file {} MiB, index {} MiB",
             ms(start, Clock::now()), ms(start, built_at), ms(built_at, Clock::now()), sum(Result::Built),
             sum(Result::Present), sum(Result::GuestChanged), sum(Result::Unreadable), sum(Result::NotCacheable),
             thread_count, cache.FileSize() >> 20, cache.IndexBytes() >> 20);
}

void PreloadJitCache(JitBlockCache& cache, std::span<ArmDynarmic64* const> jits,
                     const JitCacheBuildProgress& progress) {
    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    if (jits.empty()) {
        return;
    }
    if (!cache.TakePreloadRequest()) {
        return;
    }
    if (const char* env = std::getenv("EDEN_JIT_CACHE_PRELOAD"); env && std::string_view{env} == "0") {
        LOG_INFO(Core_ARM, "JIT cache preload: off (EDEN_JIT_CACHE_PRELOAD=0)");
        return;
    }
    const std::vector<JitBlockCache::ManifestEntry> manifest = cache.ManifestSnapshot();
    if (manifest.empty()) {
        return;
    }
    const size_t thread_count = ThreadCount(jits.size());
    const size_t total = manifest.size() * jits.size();
    constexpr size_t result_count = result_kinds + 1;  // the Precompile results and NoSpace
    std::vector<std::array<size_t, result_count>> results(jits.size());
    std::vector<size_t> code_before(jits.size());
    std::vector<u64> jit_ms(jits.size());
    for (size_t j = 0; j < jits.size(); ++j) {
        code_before[j] = jits[j]->CodeSpaceRemaining();
    }
    std::atomic<size_t> done{0};
    std::atomic<size_t> running{thread_count};
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (size_t t = 0; t < thread_count; ++t) {
        threads.emplace_back([&, t] {
            Common::SetCurrentThreadName("JitCachePreload");
            // Each thread owns whole JITs (a JIT is not thread-safe): t, t + thread_count, ...
            for (size_t j = t; j < jits.size(); j += thread_count) {
                const auto jit_start = Clock::now();
                auto& local = results[j];
                size_t since_report = 0;
                size_t handled = 0;
                for (const auto& e : manifest) {
                    const Result r = jits[j]->PlaceCachedBlock(e.descriptor, preload_keep_free);
                    ++local[std::min<size_t>(static_cast<size_t>(r), result_count - 1)];
                    ++handled;
                    if (++since_report == 1024) {
                        done.fetch_add(since_report, std::memory_order_relaxed);
                        since_report = 0;
                    }
                    if (r == Result::NoSpace) {
                        break;  // the rest of the list (less placed) is left to the game
                    }
                }
                done.fetch_add(since_report + (manifest.size() - handled), std::memory_order_relaxed);
                jit_ms[j] = static_cast<u64>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - jit_start).count());
            }
            running.fetch_sub(1, std::memory_order_release);
        });
    }
    while (running.load(std::memory_order_acquire) != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
        if (progress) {
            progress(std::min(done.load(std::memory_order_relaxed), total), total);
        }
    }
    for (auto& thread : threads) {
        thread.join();
    }
    if (progress) {
        progress(total, total);
    }
    std::array<size_t, result_count> sum{};
    std::string per_jit;
    for (size_t j = 0; j < jits.size(); ++j) {
        for (size_t k = 0; k < result_count; ++k) {
            sum[k] += results[j][k];
        }
        const size_t used = code_before[j] - std::min(code_before[j], jits[j]->CodeSpaceRemaining());
        per_jit += fmt::format("{}core {}: {} placed, {} MiB, {} ms{}", j == 0 ? "" : "; ", j,
                               results[j][static_cast<size_t>(Result::Built)], used >> 20, jit_ms[j],
                               results[j][static_cast<size_t>(Result::NoSpace)] != 0 ? ", code cache full" : "");
    }
    LOG_INFO(Core_ARM,
             "JIT cache preload: {} manifest blocks into {} cores on {} threads in {} ms: {} placed, {} already "
             "there, {} guest code changed, {} not mapped, {} without usable x64 code, {} cores stopped at the "
             "code cache reserve ({})",
             manifest.size(), jits.size(), thread_count,
             std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count(),
             sum[static_cast<size_t>(Result::Built)], sum[static_cast<size_t>(Result::Present)],
             sum[static_cast<size_t>(Result::GuestChanged)], sum[static_cast<size_t>(Result::Unreadable)],
             sum[static_cast<size_t>(Result::NotCacheable)], sum[static_cast<size_t>(Result::NoSpace)], per_jit);
}

void PrebuildJitCache(JitBlockCache& cache, Kernel::KProcess& process, std::span<ArmDynarmic64* const> jits,
                      const JitCacheBuildProgress& progress) {
    using Clock = std::chrono::steady_clock;
    const auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
    };
    if (jits.empty()) {
        return;
    }
    // The loaded code is captured only when a list has to be checked or made (Entire game).
    std::unique_ptr<JitCodeImage> image;
    const auto capture = [&] {
        if (!image) {
            const auto start = Clock::now();
            image = JitCodeImage::Capture(process);
            LOG_INFO(Core_ARM, "JIT code image: {} windows, {} MiB of code, image hash {:016X}; captured in {} ms",
                     image->Windows().size(), image->CodeBytes() >> 20, image->Hash(), ms(start, Clock::now()));
        }
    };
    // The manifest's FPCR mix, for the check of a generated list's common FPCR (taken once, when needed).
    std::optional<std::vector<std::pair<u32, std::size_t>>> mix;
    const JitBlockCache::ListCheck check = [&](const JitPrebuildList& list, bool generated, std::string& why) {
        capture();
        if (!mix) {
            mix = JitManifestFpcrMix(cache.ManifestSnapshot());
        }
        return JitPrebuildListMatches(list, generated, *image, *jits[0], *mix, why);
    };
    if (!cache.TakeRebuildRequest(check)) {
        // A launch with nothing to build: the blocks earlier sessions ran are placed in every core now, behind the
        // loading screen, instead of one by one as the game first reaches them on each core.
        PreloadJitCache(cache, jits, progress);
        return;
    }
    if (cache.GeneratesList()) {
        capture();
        // The common FPCR: the manifest's most frequent one; with no manifest (a first boot), what the code's own
        // FPCR writes say (JitStaticFpcrHint), else the FPCR guest threads start with.
        const std::vector<JitBlockCache::ManifestEntry> manifest = cache.ManifestSnapshot();
        u32 default_fpcr = jit_code_list_first_boot_fpcr;
        std::string default_from = "the thread-start FPCR";
        if (manifest.empty()) {
            const auto hint_start = Clock::now();
            const JitFpcrHint hint = JitStaticFpcrHint(*image);
            default_fpcr = hint.fpcr;
            default_from = fmt::format("the code's FPCR writes ({} found, {} resolved)", hint.writes, hint.resolved);
            LOG_INFO(Core_ARM,
                     "JIT code list: static FPCR hint {:08X}: {} MSR FPCR writes, {} set or clear a key bit ({}); "
                     "{} ms",
                     hint.fpcr, hint.writes, hint.resolved, hint.detail, ms(hint_start, Clock::now()));
        }
        const auto entries = GenerateJitCodeList(*image, jits, manifest, default_fpcr, default_from,
                                                 cache.ProgramId(), cache.PrebuildListPath(), progress);
        // The image (every module copied whole, hundreds of MiB) is not used after generation: free it before
        // the drop, which holds the key set and the index while it streams the records file.
        image.reset();
        // Records the new list and the manifest do not name (an earlier list's blocks: the other FPCR mode, or
        // code that changed) only fill the file toward its 4 GiB limit: drop them before the build.
        if (entries && cache.RecordCount() != 0) {
            const auto drop_start = Clock::now();
            std::string list_error;
            if (const auto list = JitPrebuildList::Open(cache.PrebuildListPath(), cache.ProgramId(), &list_error)) {
                std::vector<u64> keys;
                keys.reserve(list->Size() + manifest.size());
                for (size_t i = 0; i < list->Size(); ++i) {
                    keys.push_back(jits[0]->CacheKey(list->At(i).descriptor));
                }
                for (const auto& e : manifest) {
                    keys.push_back(jits[0]->CacheKey(e.descriptor));
                }
                std::sort(keys.begin(), keys.end());
                keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
                std::string what;
                const auto removed = cache.DropRecords(
                    [&](u64 key) { return std::binary_search(keys.begin(), keys.end(), key); }, u64{64} << 20,
                    what);
                LOG_INFO(Core_ARM, "JIT cache: records not in the new list or the manifest{}: {}; {} ms",
                         removed.value_or(0) != 0 ? " dropped" : " not dropped", what,
                         ms(drop_start, Clock::now()));
            } else {
                LOG_WARNING(Core_ARM, "JIT cache: cannot reopen the new list to drop old records: {}", list_error);
            }
        }
    }
    image.reset();
    RebuildJitCache(cache, jits, progress);
    // The build leaves each manifest block in one core only; the other cores get it here. With an empty manifest
    // (a first boot) this does nothing.
    PreloadJitCache(cache, jits, progress);
}

} // namespace Core
