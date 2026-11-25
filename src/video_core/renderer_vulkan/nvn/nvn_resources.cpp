// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 n2den Contributors
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>

#include <fmt/format.h>

#include "common/diag_request.h"
#include "common/settings.h"

#include "common/frame_stats.h"
#include "common/logging.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/raw_coverage_layout.h"
#include "video_core/nvn/prerec_probe.h"
#include "video_core/nvn/shader/nvn_driver_layout.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw_resources.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"
#include "video_core/renderer_vulkan/nvn/nvn_sparse.h"
#include "video_core/vulkan_common/vma.h"

namespace Vulkan::Nvn {

bool UsesNativeGobOrder(const Storage& s) {
    if (s.desc.pkg_block_h != 0) {
        return true; // a packaged layout is the Switch 2 tools'
    }
    if (s.type == VK_IMAGE_TYPE_3D) {
        return false;
    }
    // Non-packaged 2D textures: the Switch 2 order, except for Xenoblade Chronicles 2, whose
    // engine (from Switch 1) tiles its UI atlases and fonts itself in the classic order.
    // EDEN_NX2_NVN_CLASSIC_GOB_2D=1 takes the classic order for every game, =0 for none.
    static const int forced = [] {
        const char* value = std::getenv("EDEN_NX2_NVN_CLASSIC_GOB_2D");
        if (value == nullptr || *value == '\0') {
            return -1;
        }
        return value[0] == '0' ? 0 : 1;
    }();
    if (forced >= 0) {
        return forced == 0;
    }
    constexpr u64 Xenoblade2Switch2Edition = 0x0400E95004038000ULL;
    return (Settings::GetCurrentProgramID() & ~u64{0xF}) != Xenoblade2Switch2Edition;
}
namespace {

/// The import gate (GuestMemory::BeginImportGate) around the vkAllocateMemory of an import of
/// guest [va, va + size): on unless EDEN_NVN_NO_IMPORT_GATE is set (to anything but 0), for A/B.
/// An empty hook gates nothing.
HostImporter::AllocateHook ImportGateHook(VC::GuestMemory* memory, u64 va, u64 size) {
    static const bool enabled = [] {
        const char* const env = std::getenv("EDEN_NVN_NO_IMPORT_GATE");
        const bool on = env == nullptr || *env == '\0' || std::strcmp(env, "0") == 0;
        LOG_INFO(Render_Vulkan, "[nvn] import gate: {}",
                 on ? "on (guest stores to a range wait while it is imported)"
                    : "off (EDEN_NVN_NO_IMPORT_GATE)");
        return on;
    }();
    if (!enabled || memory == nullptr || size == 0) {
        return {};
    }
    return [memory, va, size, token = u64{0}](bool begin) mutable {
        if (begin) {
            token = memory->BeginImportGate(va, size);
        } else if (token != 0) {
            memory->EndImportGate(token);
            token = 0;
        }
    };
}

/// Alias import: a CPU-visible pool whose guest storage is several host runs is imported through
/// a contiguous alias view of them (Pool::alias) instead of becoming a split pool with a
/// device-local mirror. On by default for Donkey Kong Bananza only (its 450 MiB pool lands on a
/// 1 GiB heap block seam in most boots: 5 fps mirrored, 24-25 fps imported); untested elsewhere
/// (one Kirby boot crashed with it on, cause unknown). EDEN_NVN_ALIAS_IMPORT=1 turns it on for
/// every game, =0 off. EDEN_NVN_ALIAS_IMPORT_MAX_MB (default 1024): larger pools keep the split
/// path (an import pins every page; Ocarina of Time's 2.4 GiB pool stays sparse).
bool AliasImportOn() {
    static const int forced = [] {
        const char* const env = std::getenv("EDEN_NVN_ALIAS_IMPORT");
        if (env == nullptr || *env == 0) {
            return -1;
        }
        return std::strcmp(env, "0") == 0 ? 0 : 1;
    }();
    if (forced >= 0) {
        return forced == 1;
    }
    constexpr u64 DonkeyKongBananza = 0x040074A01BF12000ULL;
    return Settings::GetCurrentProgramID() == DonkeyKongBananza;
}
u64 AliasImportMaxBytes() {
    static const u64 bytes = [] {
        u64 mib = 1024;
        if (const char* const env = std::getenv("EDEN_NVN_ALIAS_IMPORT_MAX_MB"); env && *env) {
            mib = std::strtoull(env, nullptr, 10);
        }
        LOG_INFO(Render_Vulkan,
                 "[nvn] alias import: {} (EDEN_NVN_ALIAS_IMPORT, default Donkey Kong Bananza only), pools up to {} MiB "
                 "(EDEN_NVN_ALIAS_IMPORT_MAX_MB)",
                 AliasImportOn() ? "on" : "off", mib);
        return mib << 20;
    }();
    return bytes;
}
/// Most host runs an alias view may have (each is at least one view).
constexpr u64 ALIAS_MAX_RUNS = 64;

/// The first alias imports check, inside the import gate (the guest's accesses to the pool wait),
/// that the alias reaches the very bytes the guest's own host pointer does: a marker written
/// through Pointer() on both sides of every host run seam (at most 8 seams) must read back
/// through the alias, and the first 64 bytes of every page must agree. Restores what it wrote.
bool AliasSelfTest(VC::GuestMemory* memory, const Pool& pool, const u8* alias, std::string& report) {
    constexpr u64 PAGE = 4096;
    if (memory == nullptr) {
        report = "no guest memory";
        return false;
    }
    u64 markers = 0;
    u64 markers_ok = 0;
    std::vector<u64> probes;
    for (size_t i = 1; i < pool.alias_runs.size() && i <= 8; ++i) {
        probes.push_back(pool.alias_runs[i] - PAGE); // last page of the run before the seam
        probes.push_back(pool.alias_runs[i]);        // first page of the run after it
    }
    for (const u64 offset : probes) {
        u8* const backing = memory->Pointer(pool.address + offset + 1);
        if (backing == nullptr) {
            continue;
        }
        volatile u8* const b = backing;
        volatile const u8* const a = alias + offset + 1;
        const u8 old = *b;
        const u8 mark = static_cast<u8>(old ^ 0xA5);
        *b = mark;
        const bool ok = *a == mark;
        *b = old;
        ++markers;
        markers_ok += ok ? 1 : 0;
    }
    u64 pages = 0;
    u64 pages_equal = 0;
    for (u64 offset = 0; offset < pool.size; offset += PAGE) {
        const u8* const backing = memory->Pointer(pool.address + offset);
        if (backing == nullptr) {
            continue;
        }
        ++pages;
        pages_equal += std::memcmp(backing, alias + offset, 64) == 0 ? 1 : 0;
    }
    report = fmt::format("{} seams, markers {}/{} read back through the alias, pages equal {}/{}",
                         pool.alias_runs.size() - 1, markers_ok, markers, pages_equal, pages);
    return markers_ok == markers && pages_equal == pages;
}

bool IsAllZero(const u8* data, u64 size) {
    const u8* const end = data + size;
    while (data < end && reinterpret_cast<uintptr_t>(data) % alignof(u64) != 0) {
        if (*data++ != 0) {
            return false;
        }
    }
    const u64* words = reinterpret_cast<const u64*>(data);
    const u64* const words_end = words + static_cast<size_t>((end - data) / sizeof(u64));
    for (; words < words_end; ++words) {
        if (*words != 0) {
            return false;
        }
    }
    for (data = reinterpret_cast<const u8*>(words_end); data < end; ++data) {
        if (*data != 0) {
            return false;
        }
    }
    return true;
}

bool IsLayered(TexTarget target) {
    switch (target) {
    case TexTarget::Dim1DArray:
    case TexTarget::Dim2DArray:
    case TexTarget::Dim2DMultisampleArray:
    case TexTarget::Cubemap:
    case TexTarget::CubemapArray:
        return true;
    default:
        return false;
    }
}

} // Anonymous namespace

vk::Buffer Context::Impl::CreateHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                           MemoryUsage usage_hint, AllocationKind kind) const {
    return allocator->CreateBuffer(
        VkBufferCreateInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .size = size,
            .usage = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
        },
        usage_hint, kind);
}

namespace {
/// GuestPointer / GuestMappingPointer: `at(va)` is GuestMemory::Pointer or MappingPointer.
template <typename At>
const u8* ContiguousHostPointer(const Context::Impl& impl, VC::GuestMemory* memory, u64 address,
                                u64 size, At&& at) {
    if (!memory || size == 0) {
        return nullptr;
    }
    const u8* pointer = at(address);
    if (!pointer) {
        return nullptr;
    }
    if (size > 1 && at(address + size - 1) != pointer + (size - 1)) {
        return nullptr;
    }
    // Every page between, too: the first and last page can sit where one mapping would put them
    // while a run in the middle lies elsewhere (a hole in the host pages refilled from another
    // block of the same size). Importing such a range whole made the sink read and sample other
    // memory in place of the game's for the whole run (2.4 GiB texture pool of Ocarina of Time).
    constexpr u64 PAGE = 4096;
    for (u64 offset = PAGE - (address & (PAGE - 1)); offset < size; offset += PAGE) {
        if (at(address + offset) != pointer + offset) {
            static std::atomic<u32> logged{0};
            if (size >= (u64{1} << 20) && logged.fetch_add(1, std::memory_order_relaxed) < 16) {
                u64 runs = 0;
                u64 first_break = size;
                impl.CountHostRuns(address, size, runs, first_break);
                LOG_INFO(Render_Vulkan,
                         "[nvn] guest {:#x} ({:#x} bytes): first and last page map like one host "
                         "range but it is {} host runs (first break at +{:#x}): not contiguous",
                         address, size, runs, first_break);
            }
            return nullptr;
        }
    }
    return pointer;
}
} // namespace

const u8* Context::Impl::GuestPointer(u64 address, u64 size) const {
    // Record-ahead read log: the caller reads [address, address + size) through the pointer.
    VC::PreRec::RaCheck(address, size);
    VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    return ContiguousHostPointer(*this, memory, address, size,
                                 [memory](u64 va) -> const u8* { return memory->Pointer(va); });
}

const u8* Context::Impl::GuestMappingPointer(u64 address, u64 size) const {
    VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    return ContiguousHostPointer(*this, memory, address, size, [memory](u64 va) -> const u8* {
        return memory->MappingPointer(va);
    });
}

void Context::Impl::CountHostRuns(u64 address, u64 size, u64& runs, u64& first_break) const {
    runs = 0;
    first_break = size;
    VC::GuestMemory* memory = guest.load(std::memory_order_acquire);
    if (!memory) {
        return;
    }
    constexpr u64 PAGE = 4096;
    const u8* previous = nullptr;
    for (u64 offset = 0; offset < size; offset += PAGE) {
        const u8* page = memory->MappingPointer(address + offset);
        if (page == nullptr || previous == nullptr || page != previous + PAGE) {
            if (++runs == 2) {
                first_break = offset;
            }
        }
        previous = page;
    }
}

bool Context::Impl::ImportSplitRuns(Pool& pool, std::string& error) {
    // A run per import: each costs a VkDeviceMemory, and a storage split page by page is left to
    // the mirror alone.
    constexpr u64 MAX_RUNS = 64;
    constexpr u64 PAGE = 4096;
    VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    if (memory == nullptr || !importer || !importer->Available()) {
        error = "host import not available";
        return false;
    }
    std::vector<Pool::SplitRun> runs;
    std::vector<u8*> hosts;
    for (u64 offset = 0; offset < pool.size;) {
        const u64 guest_address = pool.address + offset;
        const u64 length = (std::min)(PAGE - guest_address % PAGE, pool.size - offset);
        u8* const host = const_cast<u8*>(memory->MappingPointer(guest_address));
        if (host == nullptr) {
            error = fmt::format("guest page at +{:#x} has no host mapping", offset);
            return false;
        }
        if (!runs.empty() && host == hosts.back() + runs.back().size) {
            runs.back().size += length;
        } else {
            if (runs.size() == MAX_RUNS) {
                error = fmt::format("more than {} host runs", MAX_RUNS);
                return false;
            }
            runs.push_back({.offset = offset, .size = length});
            hosts.push_back(host);
        }
        offset += length;
    }
    // A large pool is imported granule by granule, only where it is in use (Pool::SparseSplit).
    if (SetUpSparseSplit(pool, runs, hosts)) {
        pool.split = true;
        return true;
    }
    try {
        for (size_t i = 0; i < runs.size(); ++i) {
            Pool::SplitRun& run = runs[i];
            if (!importer->IsAligned(hosts[i], run.size)) {
                error = fmt::format("run at +{:#x} ({:#x} bytes) is not aligned to {:#x}",
                                    run.offset, run.size, importer->Alignment());
                return false;
            }
            std::string run_error;
            const HostImporter::AllocateHook gate =
                ImportGateHook(memory, pool.address + run.offset, run.size);
            const VkResult result = importer->Import(hosts[i], run.size, pool_usage, run.imported,
                                                     run_error, &gate);
            if (result != VK_SUCCESS) {
                error = fmt::format("run at +{:#x} ({:#x} bytes): {} ({})", run.offset, run.size,
                                    run_error, static_cast<int>(result));
                return false;
            }
        }
    } catch (const vk::Exception& exception) {
        error = exception.what();
        return false;
    }
    pool.split_runs = std::move(runs);
    pool.split = true;
    return true;
}

bool Context::Impl::SetUpSparseSplit(Pool& pool, std::span<const Pool::SplitRun> runs,
                                     std::span<u8* const> hosts) {
    // EDEN_NVN_SPARSE_SPLIT_IMPORT (default on; 0: every split pool is imported whole, as
    // before), EDEN_NVN_SPARSE_SPLIT_GRANULE_MB (default 4, 1..64: the grid; smaller pins less
    // unused memory around what the game uses and costs more imports) and
    // EDEN_NVN_SPARSE_SPLIT_MIN_MB (default 64: smaller pools are imported whole; they are what
    // the game uses, and every sparse pool costs a tracker registration and a sweep).
    struct Config {
        bool enabled{true};
        u64 granule{u64{4} << 20};
        u64 min_size{u64{64} << 20};
    };
    static const Config config = [] {
        Config result;
        if (const char* const env = std::getenv("EDEN_NVN_SPARSE_SPLIT_IMPORT"); env && *env) {
            result.enabled = std::strcmp(env, "0") != 0;
        }
        if (const char* const env = std::getenv("EDEN_NVN_SPARSE_SPLIT_GRANULE_MB");
            env && *env) {
            result.granule = (std::clamp<u64>)(std::strtoull(env, nullptr, 10), 1, 64) << 20;
        }
        if (const char* const env = std::getenv("EDEN_NVN_SPARSE_SPLIT_MIN_MB"); env && *env) {
            result.min_size = (std::min<u64>)(std::strtoull(env, nullptr, 10), 1u << 20) << 20;
        }
        LOG_INFO(Render_Vulkan,
                 "[nvn] sparse split import: {}, granule {} MiB, pools of {} MiB and more",
                 result.enabled ? "on" : "off (EDEN_NVN_SPARSE_SPLIT_IMPORT=0)",
                 result.granule >> 20, result.min_size >> 20);
        return result;
    }();
    VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    const u64 align = importer->Alignment();
    if (!config.enabled || memory == nullptr || runs.empty() || pool.size < config.min_size ||
        align == 0 || config.granule % align != 0 ||
        pool.released.load(std::memory_order_relaxed)) {
        return false;
    }
    // Grid lines cut the runs: a piece is importable when its run is, and when the run's guest
    // address is aligned like its host address.
    for (size_t i = 0; i < runs.size(); ++i) {
        if ((pool.address + runs[i].offset) % align != 0 ||
            !importer->IsAligned(hosts[i], runs[i].size)) {
            return false;
        }
    }
    // The tracker tells which granules the guest writes from now on; without it only the slow
    // sweep would, and a draw could read a granule the game just filled as zeros for seconds.
    const u64 track = memory->Track(pool.address, pool.size);
    if (track == 0) {
        LOG_INFO(Render_Vulkan,
                 "[nvn] pool {:#x} ({:#x} bytes): no write tracking, split runs are imported whole",
                 pool.address, pool.size);
        return false;
    }
    using Clock = std::chrono::steady_clock;
    const auto begin = Clock::now();
    auto sparse = std::make_unique<Pool::SparseSplit>();
    sparse->granule = config.granule;
    sparse->first = pool.address / config.granule;
    sparse->count = static_cast<size_t>((pool.address + pool.size - 1) / config.granule -
                                        sparse->first + 1);
    sparse->state = std::make_unique<std::atomic<u8>[]>(sparse->count);
    for (size_t i = 0; i < sparse->count; ++i) {
        sparse->state[i].store(Pool::SparseSplit::Untouched, std::memory_order_relaxed);
    }
    sparse->remaining = sparse->count;
    sparse->runs.reserve(runs.size());
    for (size_t i = 0; i < runs.size(); ++i) {
        sparse->runs.push_back({runs[i].offset, runs[i].size, hosts[i]});
    }
    // Armed before the look below: a page the guest writes after it shows in the tracker, one
    // it wrote before has a host page. (An armed page traps the guest's first write to it, once:
    // nothing collects, so nothing arms it again.)
    memory->Arm(track, pool.address, pool.size);
    sparse->track.store(track, std::memory_order_relaxed);
    sparse->write_epoch = memory->WriteEpoch(track);
    VC::PreRec::RegisterEpoch(pool.address, pool.size, pool.serial, VC::PreRec::Cls::SparseGrowth,
                              sparse->write_epoch);
    sparse->epoch_seen =
        sparse->write_epoch ? sparse->write_epoch->load(std::memory_order_acquire) : 0;
    pool.sparse = std::move(sparse);
    Pool::SparseSplit& state = *pool.sparse;
    std::vector<size_t> used;
    for (size_t i = 0; i < state.count; ++i) {
        if (SparseGranuleResident(pool, i, *memory)) {
            used.push_back(i);
        }
    }
    const auto looked = Clock::now();
    const u64 imported = ImportSparseGranules(pool, used);
    sparse_split_pools_made.fetch_add(1, std::memory_order_relaxed);
    sparse_split_pool_bytes.fetch_add(pool.size, std::memory_order_relaxed);
    if (!state.complete.load(std::memory_order_relaxed)) {
        sparse_split_pools.push_back(pool.weak_from_this());
        sparse_split_live.store(static_cast<u32>(sparse_split_pools.size()),
                                std::memory_order_release);
        // The backing thread sweeps while sparse pools exist.
        backing_cv.notify_one();
    }
    LOG_INFO(Render_Vulkan,
             "[nvn] pool {:#x} ({:#x} bytes, flags {:#x}) is a sparse split pool: {} host runs, {} "
             "granules of {} MiB, {} in use now: {:.1f} of {:.1f} MiB imported in {} pieces "
             "(residency look {:.1f} ms, imports {:.1f} ms); the rest is imported when used",
             pool.address, pool.size, pool.flags, state.runs.size(), state.count,
             state.granule >> 20, used.size(), imported / 1048576.0, pool.size / 1048576.0,
             pool.split_runs.size(),
             std::chrono::duration<double, std::milli>(looked - begin).count(),
             std::chrono::duration<double, std::milli>(Clock::now() - looked).count());
    return true;
}

std::pair<u64, u64> Context::Impl::SparseGranuleRange(const Pool& pool, size_t index) {
    const Pool::SparseSplit& sparse = *pool.sparse;
    const u64 begin = (std::max)((sparse.first + index) * sparse.granule, pool.address);
    const u64 end =
        (std::min)((sparse.first + index + 1) * sparse.granule, pool.address + pool.size);
    return {begin, end - begin};
}

bool Context::Impl::SparseGranuleResident(const Pool& pool, size_t index,
                                          VC::GuestMemory& memory) {
    const auto [address, size] = SparseGranuleRange(pool, index);
    // The backing's own address: the kernel, services and nvn-hle write through it.
    for (const Pool::SparseSplit::HostRun& run : pool.sparse->runs) {
        const u64 run_begin = pool.address + run.offset;
        const u64 part_begin = (std::max)(address, run_begin);
        const u64 part_end = (std::min)(address + size, run_begin + run.size);
        if (part_begin < part_end &&
            HostImporter::AnyResident(run.host + (part_begin - run_begin), part_end - part_begin)) {
            return true;
        }
    }
    // The guest's view: what its own code read or wrote has its page there.
    const u8* const view = memory.GuestViewPointer(address);
    return view != nullptr && HostImporter::AnyResident(view, size);
}

u64 Context::Impl::ImportSparseGranules(Pool& pool, std::span<const size_t> granules) {
    using State = Pool::SparseSplit;
    Pool::SparseSplit& sparse = *pool.sparse;
    VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    std::vector<size_t> rest;
    // The imports this call would make at most: one per host run for every range of neighbours.
    size_t ranges = granules.empty() ? 0 : 1;
    for (size_t i = 1; i < granules.size(); ++i) {
        ranges += granules[i] != granules[i - 1] + 1 ? 1 : 0;
    }
    if (pool.split_runs.size() + ranges * sparse.runs.size() > Pool::SparseSplit::MAX_PIECES) {
        // Enough device allocations for one pool: what is left goes in as few pieces as the gaps
        // between the imported ones allow, used or not.
        for (size_t i = 0; i < sparse.count; ++i) {
            const u8 state = sparse.state[i].load(std::memory_order_acquire);
            if (state != State::Imported && state != State::Failed) {
                rest.push_back(i);
            }
        }
        granules = rest;
        LOG_INFO(Render_Vulkan,
                 "[nvn] sparse split pool {:#x}: {} pieces imported, the remaining {} granules are "
                 "imported whole",
                 pool.address, pool.split_runs.size(), rest.size());
    }
    u64 imported = 0;
    u64 pieces = 0;
    for (size_t at = 0; at < granules.size();) {
        size_t last = at;
        while (last + 1 < granules.size() && granules[last + 1] == granules[last] + 1) {
            ++last;
        }
        const u64 begin = SparseGranuleRange(pool, granules[at]).first;
        const auto [last_address, last_size] = SparseGranuleRange(pool, granules[last]);
        const u64 end = last_address + last_size;
        bool failed = false;
        for (const Pool::SparseSplit::HostRun& run : sparse.runs) {
            const u64 run_begin = pool.address + run.offset;
            const u64 part_begin = (std::max)(begin, run_begin);
            const u64 part_end = (std::min)(end, run_begin + run.size);
            if (part_begin >= part_end) {
                continue;
            }
            Pool::SplitRun piece{.offset = part_begin - pool.address,
                                 .size = part_end - part_begin};
            std::string error;
            VkResult result = VK_ERROR_UNKNOWN;
            try {
                // The gate, as for every import of live guest memory.
                const HostImporter::AllocateHook gate =
                    ImportGateHook(memory, part_begin, piece.size);
                result = importer->Import(run.host + (part_begin - run_begin), piece.size,
                                          pool_usage, piece.imported, error, &gate);
            } catch (const vk::Exception& exception) {
                error = exception.what();
            }
            if (result != VK_SUCCESS) {
                failed = true;
                stats.host_import_failures.fetch_add(1, std::memory_order_relaxed);
                LOG_WARNING(Render_Vulkan,
                            "[nvn] sparse split pool {:#x}: import of +{:#x} ({:#x} bytes) failed: "
                            "{} ({}); raw global accesses there read 0",
                            pool.address, piece.offset, piece.size, error,
                            static_cast<int>(result));
                continue;
            }
            imported += piece.size;
            ++pieces;
            pool.split_runs.push_back(std::move(piece));
        }
        for (size_t i = at; i <= last; ++i) {
            sparse.state[granules[i]].store(failed ? State::Failed : State::Imported,
                                            std::memory_order_release);
        }
        sparse.remaining -= (std::min)(sparse.remaining, last - at + 1);
        at = last + 1;
    }
    sparse_split_imported_bytes.fetch_add(imported, std::memory_order_relaxed);
    sparse_split_pieces.fetch_add(pieces, std::memory_order_relaxed);
    stats.split_runs.fetch_add(pieces, std::memory_order_relaxed);
    if (sparse.remaining == 0 && !sparse.complete.exchange(true, std::memory_order_acq_rel)) {
        // Nothing left to watch.
        if (const u64 track = sparse.track.exchange(0, std::memory_order_acq_rel);
            track != 0 && memory != nullptr) {
            memory->Untrack(track);
        }
    }
    return imported;
}

bool Context::Impl::WantSparseSplit(Pool& pool, u64 offset, u64 size) {
    Pool::SparseSplit* const sparse = pool.sparse.get();
    if (sparse == nullptr || size == 0 || offset >= pool.size ||
        sparse->complete.load(std::memory_order_relaxed)) {
        return false;
    }
    const u64 last_byte = pool.address + (std::min)(size, pool.size - offset) + offset - 1;
    const u64 first = (pool.address + offset) / sparse->granule - sparse->first;
    const u64 last = last_byte / sparse->granule - sparse->first;
    bool wanted = false;
    for (u64 index = first; index <= last && index < sparse->count; ++index) {
        std::atomic<u8>& state = sparse->state[static_cast<size_t>(index)];
        u8 expected = Pool::SparseSplit::Untouched;
        if (state.load(std::memory_order_relaxed) == expected &&
            state.compare_exchange_strong(expected, Pool::SparseSplit::Wanted,
                                          std::memory_order_acq_rel)) {
            wanted = true;
        }
    }
    if (wanted) {
        sparse_split_wanted.store(true, std::memory_order_release);
    }
    return wanted;
}

bool Context::Impl::SparseSplitNeedsGrowth() {
    const u32 live = sparse_split_live.load(std::memory_order_acquire);
    if (live == 0 && sparse_split_view.empty()) {
        return false;
    }
    // A pool backed since the last look is not in the view yet.
    if (live != sparse_split_view.size() ||
        sparse_split_wanted.load(std::memory_order_acquire)) {
        return true;
    }
    // The tracker: once per submission. What a draw reads, the game wrote before it submitted.
    const u64 seq = replay_seq.load(std::memory_order_relaxed);
    if (seq == sparse_split_checked_seq) {
        return false;
    }
    sparse_split_checked_seq = seq;
    for (const std::weak_ptr<Pool>& weak : sparse_split_view) {
        const std::shared_ptr<Pool> pool = weak.lock();
        if (!pool || !pool->sparse) {
            continue;
        }
        const Pool::SparseSplit& sparse = *pool->sparse;
        if (!sparse.write_epoch ||
            sparse.write_epoch->load(std::memory_order_acquire) != sparse.epoch_seen) {
            return true;
        }
    }
    return false;
}

bool Context::Impl::SparseSplitGrowthDue() const {
    const u32 live = sparse_split_live.load(std::memory_order_acquire);
    if (live == 0 && sparse_split_view.empty()) {
        return false;
    }
    if (live != sparse_split_view.size() ||
        sparse_split_wanted.load(std::memory_order_acquire)) {
        return true;
    }
    if (replay_seq.load(std::memory_order_relaxed) == sparse_split_checked_seq) {
        return false;
    }
    for (const std::weak_ptr<Pool>& weak : sparse_split_view) {
        const std::shared_ptr<Pool> pool = weak.lock();
        if (!pool || !pool->sparse) {
            continue;
        }
        const Pool::SparseSplit& sparse = *pool->sparse;
        if (!sparse.write_epoch ||
            sparse.write_epoch->load(std::memory_order_acquire) != sparse.epoch_seen) {
            return true;
        }
    }
    return false;
}

bool Context::Impl::GrowSparseSplitPools() {
    if (sparse_split_pools.empty() && sparse_split_view.empty()) {
        return false;
    }
    using State = Pool::SparseSplit;
    // Cleared before the scan: a granule wanted while it runs is seen by the next call.
    sparse_split_wanted.store(false, std::memory_order_release);
    VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    bool grew = false;
    std::vector<std::weak_ptr<Pool>> live;
    std::vector<size_t> granules;
    for (const std::weak_ptr<Pool>& weak : sparse_split_pools) {
        const std::shared_ptr<Pool> pool = weak.lock();
        if (!pool || !pool->sparse || pool->released.load(std::memory_order_relaxed) ||
            pool->sparse->complete.load(std::memory_order_relaxed)) {
            continue;
        }
        Pool::SparseSplit& sparse = *pool->sparse;
        // Read before the tracker is asked: a write in between shows as a change next time.
        const u64 epoch =
            sparse.write_epoch ? sparse.write_epoch->load(std::memory_order_acquire) : 0;
        const u64 track = sparse.track.load(std::memory_order_relaxed);
        const bool ask = memory != nullptr && track != 0 &&
                         (!sparse.write_epoch || epoch != sparse.epoch_seen);
        granules.clear();
        for (size_t i = 0; i < sparse.count; ++i) {
            const u8 state = sparse.state[i].load(std::memory_order_acquire);
            if (state == State::Wanted) {
                granules.push_back(i);
            } else if (state == State::Untouched && ask) {
                const auto [address, size] = SparseGranuleRange(*pool, i);
                if (memory->AnyWritten(track, address, size)) {
                    granules.push_back(i);
                    // EDEN_NX2_NVN_PREREC_PROBE: imported because the CPU wrote it (a write
                    // before this check decides what raw global loads can reach).
                    VC::PreRec::NoteRead(VC::PreRec::Cls::SparseGrowth, address, size,
                                         pool->serial);
                }
            }
        }
        sparse.epoch_seen = epoch;
        if (!granules.empty()) {
            const size_t pieces_before = pool->split_runs.size();
            const auto begin = std::chrono::steady_clock::now();
            const u64 bytes = ImportSparseGranules(*pool, granules);
            grew = grew || bytes != 0;
            const u64 logged = sparse.growths_logged++;
            if (logged < 16 || logged % 32 == 0) {
                u64 total = 0;
                for (const Pool::SplitRun& piece : pool->split_runs) {
                    total += piece.size;
                }
                LOG_INFO(Render_Vulkan,
                         "[nvn] sparse split pool {:#x}: growth #{}: {} granules now in use, {:.1f} "
                         "MiB in {} imports, {:.2f} ms; {:.1f} of {:.1f} MiB imported in {} "
                         "pieces, {} granules left",
                         pool->address, logged + 1, granules.size(), bytes / 1048576.0,
                         pool->split_runs.size() - pieces_before,
                         std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - begin)
                             .count(),
                         total / 1048576.0, pool->size / 1048576.0, pool->split_runs.size(),
                         sparse.remaining);
            }
        }
        if (!sparse.complete.load(std::memory_order_relaxed)) {
            live.push_back(weak);
        }
    }
    sparse_split_pools = std::move(live);
    sparse_split_view = sparse_split_pools;
    sparse_split_live.store(static_cast<u32>(sparse_split_pools.size()), std::memory_order_release);
    if (grew) {
        ++pool_generation;
    }
    return grew;
}

void Context::Impl::SweepSparseSplitPools() {
    // Granules looked at per pool and sweep: at four sweeps a second and 4 MiB granules, a
    // 2.4 GiB pool nobody uses is looked through in about five seconds.
    constexpr size_t BUDGET = 32;
    std::vector<std::shared_ptr<Pool>> targets;
    {
        std::scoped_lock lock{backing_mutex};
        for (const std::weak_ptr<Pool>& weak : sparse_split_pools) {
            if (std::shared_ptr<Pool> pool = weak.lock();
                pool && pool->sparse && !pool->released.load(std::memory_order_relaxed)) {
                targets.push_back(std::move(pool));
            }
        }
    }
    VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    if (memory == nullptr) {
        return;
    }
    u64 found = 0;
    for (const std::shared_ptr<Pool>& pool : targets) {
        Pool::SparseSplit& sparse = *pool->sparse;
        size_t looked = 0;
        for (size_t step = 0; step < sparse.count && looked < BUDGET; ++step) {
            const size_t index = sparse.sweep_cursor;
            sparse.sweep_cursor = (index + 1) % sparse.count;
            std::atomic<u8>& state = sparse.state[index];
            if (state.load(std::memory_order_acquire) != Pool::SparseSplit::Untouched) {
                continue;
            }
            ++looked;
            if (!SparseGranuleResident(*pool, index, *memory)) {
                continue;
            }
            u8 expected = Pool::SparseSplit::Untouched;
            if (state.compare_exchange_strong(expected, Pool::SparseSplit::Wanted,
                                              std::memory_order_acq_rel)) {
                ++found;
            }
        }
    }
    if (found != 0) {
        // Somebody used these granules past the tracker (or read them): imported with the next
        // raw table. Until then raw loads there read 0.
        const u64 total = sparse_split_sweep_hits.fetch_add(found, std::memory_order_relaxed) + found;
        sparse_split_wanted.store(true, std::memory_order_release);
        if (total == found || total / 64 != (total - found) / 64) {
            LOG_INFO(Render_Vulkan,
                     "[nvn] sparse split sweep: {} granules have host pages the write tracker "
                     "did not report ({} so far); imported with the next raw table",
                     found, total);
        }
    }
}

void Context::Impl::ImmediateSubmit(const std::function<void(vk::CommandBuffer)>& record) {
    if (ra.Active()) [[unlikely]] {
        ++ra.immediate_submits; // touches never-used resources only (design G)
    }
    const VkCommandBuffer raw = AcquireCommandBuffer();
    const vk::CommandBuffer cmd{raw, dld};
    cmd.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    record(cmd);
    cmd.End();
    vk::Fence fence = device.GetLogical().CreateFence({
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    });
    const VkSubmitInfo submit_info{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = nullptr,
        .pWaitDstStageMask = nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers = &raw,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr,
    };
    {
        std::scoped_lock lock{submit_mutex};
        vk::Check(queue.Submit(submit_info, *fence));
    }
    vk::Check(fence.Wait());
    std::scoped_lock lock{free_commands_mutex};
    free_command_buffers.push_back(raw);
}

bool Context::Impl::ResolveBuffer(u64 address, u64 size, Pool*& pool, u64& offset) const {
    const u64 generation = pool_generation.load(std::memory_order_acquire);
    ExactResolveSlot* exact = nullptr;
    if (mario_kart_world) {
        const u64 key = ((address >> 4) ^ (size * 0xd6e8feb86659fd93ULL)) *
                        0x9e3779b97f4a7c15ULL;
        exact = &mkw_resolve_cache[key >> 56];
        if (exact->generation == generation && exact->mappings == mkw_mapping_generation &&
            exact->address == address && exact->size == size) {
            pool = exact->pool;
            offset = exact->offset;
            ++mkw_resolve_hits;
            return true;
        }
        ++mkw_resolve_misses;
    }
    const auto remember = [&] {
        if (exact) {
            *exact = {generation, mkw_mapping_generation, address, size, pool, offset};
        }
    };
    for (const ResolveSlot& slot : resolve_cache) {
        if (slot.generation == generation && address >= slot.begin && address < slot.end) {
            const Pool& cached = *slot.pool;
            if (size <= cached.size - (address - cached.address)) {
                pool = slot.pool;
                offset = address - cached.address;
                remember();
                return true;
            }
            break; // the slow path decides
        }
    }
    auto it = pools_by_address.upper_bound(address);
    if (it == pools_by_address.begin()) {
        return false;
    }
    const auto next = it;
    --it;
    Pool& candidate = *it->second;
    if (address < candidate.address || address - candidate.address > candidate.size ||
        size > candidate.size - (address - candidate.address)) {
        return false;
    }
    if (candidate.IsVirtual()) {
        if (!ResolveVirtualRange(candidate, address - candidate.address, size, pool, offset)) {
            return false;
        }
        remember();
        return true;
    }
    pool = &candidate;
    offset = address - candidate.address;
    const u64 limit = next != pools_by_address.end() ? next->first : ~u64{0};
    resolve_cache[resolve_next++ % resolve_cache.size()] = {
        generation, candidate.address, (std::min)(candidate.address + candidate.size, limit),
        &candidate};
    remember();
    return true;
}

void Context::Impl::CreatePoolNow(const VC::PoolDesc& desc) {
    auto pool = std::make_shared<Pool>();
    pool->serial = desc.serial;
    pool->address = desc.guest_address;
    pool->size = desc.size;
    pool->flags = desc.flags;
    pools[desc.serial] = pool;
    ++pool_generation;
    if (desc.size == 0) {
        return;
    }
    pools_by_address[desc.guest_address] = pool;

    if (desc.flags & PoolFlag::Virtual) {
        // No guest storage and no buffer: a table of mapped ranges (MapVirtual). Buffers placed in
        // the pool resolve to the physical pool bytes behind them (ResolveBuffer); textures read
        // their bytes through the same table.
        pool->backing = Pool::Backing::Virtual;
        return;
    }

    // Backing the pool (a host import of the chunk around it, a few ms of driver time, or a
    // device-local buffer, about a millisecond) happens on the backing
    // thread, or at the first command that needs the pool if that comes first. Most of DK's
    // ~1,700 pools are descriptor tables or CPU-only data the GPU never reads.
    pool->pending.store(true, std::memory_order_relaxed);
    if (desc.flags & PoolFlag::CpuNoAccess) {
        stats.pools_device_local.fetch_add(1, std::memory_order_relaxed);
    }
    if (options.background_backing) {
        {
            std::scoped_lock lock{backing_queue_mutex};
            backing_queue.push_back(pool);
        }
        backing_cv.notify_one();
    }
}

bool Context::Impl::AliasImportWanted(u64 size) const {
    return AliasImportOn() && size <= AliasImportMaxBytes() && options.allow_host_import &&
           importer && importer->Available();
}

std::shared_ptr<u8> Context::Impl::MapPoolAlias(const Pool& pool, std::vector<u64>& runs) const {
    VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    // The buffer starts at the alias: pool address and size must suit the import alignment.
    const u64 align = importer->Alignment();
    if (memory == nullptr || pool.address % align != 0 || pool.size % align != 0) {
        return nullptr;
    }
    std::shared_ptr<u8> alias = memory->MapContiguousAlias(pool.address, pool.size,
                                                           ALIAS_MAX_RUNS, &runs);
    if (alias && !importer->IsAligned(alias.get(), pool.size)) {
        alias.reset();
    }
    return alias;
}

std::optional<Context::Impl::ChunkWindow> Context::Impl::ImportWindow(u64 address, u64 size,
                                                                      u8* host) const {
    // EDEN_NVN_IMPORT_CHUNK_MB (default 256; 0: the pool's own pages only): the most a window
    // may reach, half each side of the pool (see the granule below). On the NVIDIA Windows
    // driver vkAllocateMemory of a host pointer reads byte 0 of every imported page and writes it
    // back, not atomically: a guest store to that byte in between is lost. The smaller the window,
    // the less guest memory that is not a pool is exposed to it.
    static const u64 ChunkSize = [] {
        u64 mib = 256;
        if (const char* const env = std::getenv("EDEN_NVN_IMPORT_CHUNK_MB"); env && *env) {
            mib = (std::min<u64>)(std::strtoull(env, nullptr, 10), 1024);
        }
        return mib << 20;
    }();
    const u64 align = importer->Alignment();
    VC::GuestMemory* memory = guest.load(std::memory_order_acquire);
    // Guest and host addresses must differ by a multiple of the import alignment.
    if (!memory || (reinterpret_cast<uintptr_t>(host) - address) % align != 0) {
        return std::nullopt;
    }
    const auto align_down = [align](u64 v) { return v & ~(align - 1); };
    const auto align_up = [align](u64 v) { return (v + align - 1) & ~(align - 1); };
    // Up to half a chunk of guest space on each side of the pool (so pools next to each other,
    // whichever side of any fixed grid line they are on, land in one chunk), cut down to the
    // contiguous run of mapped pages that contains the pool.
    const u64 pool_begin = align_down(address);
    const u64 pool_end = align_up(address + size);
    u64 window_begin = pool_begin > ChunkSize / 2 ? align_down(pool_begin - ChunkSize / 2) : 0;
    u64 window_end = pool_end + ChunkSize / 2;
    // EDEN_NVN_IMPORT_GRANULE_MB (default 4; 0: the +-half-chunk window above). Every page of a
    // window gets RAM and is pinned by the import, and n2den's guest backing gives a page RAM
    // only once somebody writes it: half a chunk each side of every pool made resident heap the
    // game had mapped but never used. The window is the pool rounded out to a fixed grid of
    // guest addresses instead, so a chunk costs at most two granules of such pages, pools of
    // one granule still share a chunk, and the number of chunks stays bounded by the guest
    // space the pools ever cover (windows that float with the pool would not be: pools come
    // and go at new addresses and chunks live as long as the context).
    static const u64 Granule = [] {
        u64 mib = 4;
        if (const char* const env = std::getenv("EDEN_NVN_IMPORT_GRANULE_MB"); env && *env) {
            mib = (std::min<u64>)(std::strtoull(env, nullptr, 10), 1024);
        }
        LOG_INFO(Render_Vulkan, "[nvn] import windows: chunk {} MiB, granule {} MiB{}",
                 ChunkSize >> 20, mib,
                 mib == 0 ? " (off: half a chunk each side of the pool)" : "");
        return mib << 20;
    }();
    if (Granule != 0 && ChunkSize != 0) {
        // A granule is a multiple of 1 MiB, so of the import alignment too.
        const u64 granule_begin = pool_begin / Granule * Granule;
        const u64 granule_end = (pool_end + Granule - 1) / Granule * Granule;
        window_begin = (std::max)(window_begin, align_down(granule_begin));
        window_end = (std::min)(window_end, align_up(granule_end));
    }
    u8* const base = host - (address - pool_begin);
    const auto mapped_like_base = [&](u64 page) {
        return memory->MappingPointer(page) == base + (page - pool_begin) &&
               memory->MappingPointer(page + align - 1) ==
                   base + (page - pool_begin) + align - 1;
    };
    u64 low = pool_begin;
    while (low > window_begin && mapped_like_base(low - align)) {
        low -= align;
    }
    u64 high = pool_end;
    while (high < window_end && mapped_like_base(high)) {
        high += align;
    }
    return ChunkWindow{base + (low - pool_begin), low, high - low};
}

VkResult Context::Impl::ImportPoolBuffer(Pool& pool, u8* host, std::string& error) {
    const u64 address = pool.address;
    const u64 size = pool.size;
    const auto per_pool_import = [&]() {
        pool_import_fallbacks.fetch_add(1, std::memory_order_relaxed);
        const auto t = std::chrono::steady_clock::now();
        const HostImporter::AllocateHook gate =
            ImportGateHook(guest.load(std::memory_order_acquire), address, size);
        const VkResult result =
            importer->Import(host, size, pool_usage, pool.imported, error, &gate);
        chunk_imports.fetch_add(1, std::memory_order_relaxed);
        chunk_import_ns.fetch_add(
            static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now() - t)
                                 .count()),
            std::memory_order_relaxed);
        chunk_import_bytes.fetch_add(size, std::memory_order_relaxed);
        return result;
    };
    if (pool.alias) {
        // An alias view: guest and host addresses are not congruent (ImportWindow's assumption),
        // so the pool is imported on its own. The import gate is keyed by guest address as for
        // any pool; the first alias imports check the alias inside it (AliasSelfTest).
        static std::atomic<u32> self_tests{0};
        if (self_tests.fetch_add(1, std::memory_order_relaxed) >= 4) {
            return per_pool_import();
        }
        VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
        const HostImporter::AllocateHook gate = ImportGateHook(memory, address, size);
        bool tested = false;
        bool passed = false;
        std::string report;
        const HostImporter::AllocateHook hook = [&](bool begin) {
            if (gate) {
                gate(begin);
            }
            if (begin && !tested) {
                tested = true;
                passed = AliasSelfTest(memory, pool, host, report);
            }
        };
        pool_import_fallbacks.fetch_add(1, std::memory_order_relaxed);
        const auto t = std::chrono::steady_clock::now();
        VkResult result = importer->Import(host, size, pool_usage, pool.imported, error, &hook);
        chunk_imports.fetch_add(1, std::memory_order_relaxed);
        chunk_import_ns.fetch_add(
            static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now() - t)
                                 .count()),
            std::memory_order_relaxed);
        chunk_import_bytes.fetch_add(size, std::memory_order_relaxed);
        LOG_INFO(Render_Vulkan, "[nvn] alias self-test of pool {:#x} ({:#x} bytes): {} ({})",
                 address, size, tested ? (passed ? "passed" : "FAILED") : "not run",
                 tested ? report : std::string{"the import did not get to vkAllocateMemory"});
        if (result == VK_SUCCESS && tested && !passed) {
            pool.imported.buffer = {}; // the buffer before its memory
            pool.imported.memory = {};
            pool.imported.size = 0;
            error = "alias self-test failed";
            result = VK_ERROR_INITIALIZATION_FAILED;
        }
        return result;
    }
    // Guest and host addresses must differ by a multiple of the import alignment.
    if (!guest.load(std::memory_order_acquire) ||
        (reinterpret_cast<uintptr_t>(host) - address) % importer->Alignment() != 0) {
        return per_pool_import();
    }
    std::shared_ptr<ImportedChunk> chunk;
    for (const auto& candidate : import_chunks) {
        if (host >= candidate->host && host + size <= candidate->host + candidate->size) {
            chunk = candidate;
            break;
        }
    }
    if (!chunk) {
        const std::optional<ChunkWindow> window = ImportWindow(address, size, host);
        if (!window) {
            return per_pool_import();
        }
        const u64 low = window->guest_low;
        const u64 high = low + window->size;
        const auto t = std::chrono::steady_clock::now();
        auto imported = std::make_shared<ImportedChunk>();
        HostImporter::ImportTiming timing{};
        // The window is one run of guest pages mapped like the pool: guest [low, high).
        const HostImporter::AllocateHook gate =
            ImportGateHook(guest.load(std::memory_order_acquire), low, high - low);
        const VkResult result = importer->ImportChunk(window->host, window->size, pool_usage,
                                                      *imported, error, &timing, &gate);
        const u64 ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::steady_clock::now() - t)
                                            .count());
        imported->guest_base = low;
        chunk_imports.fetch_add(1, std::memory_order_relaxed);
        chunk_import_ns.fetch_add(ns, std::memory_order_relaxed);
        chunk_import_bytes.fetch_add(high - low, std::memory_order_relaxed);
        if (chunk_imports.load() <= 32) {
            LOG_INFO(Render_Vulkan,
                     "[nvn] import #{}: chunk guest {:#x}..{:#x} ({:.1f} MiB) for pool {:#x} "
                     "({:#x} bytes): {:.2f} ms (make resident {:.2f} ms, pointer properties "
                     "{:.2f} ms, allocate {:.2f} ms), result {}",
                     chunk_imports.load(), low, high, (high - low) / 1048576.0, address, size,
                     ns / 1e6, timing.resident_ns / 1e6, timing.props_ns / 1e6,
                     timing.alloc_ns / 1e6, static_cast<int>(result));
        }
        if (result != VK_SUCCESS) {
            return per_pool_import();
        }
        chunk = std::move(imported);
        import_chunks.push_back(chunk);
        import_chunks_live.fetch_add(1, std::memory_order_relaxed);
        import_chunks_live_bytes.fetch_add(high - low, std::memory_order_relaxed);
    }
    const VkResult result = importer->CreateBufferInChunk(
        *chunk, static_cast<u64>(host - chunk->host), size, pool_usage,
        pool.imported.buffer, error);
    if (result != VK_SUCCESS) {
        return per_pool_import();
    }
    pool.imported.size = size;
    pool.chunk = std::move(chunk);
    pools_in_chunks.fetch_add(1, std::memory_order_relaxed);
    return VK_SUCCESS;
}

bool Context::Impl::InImportedChunk(const u8* host, u64 size) const {
    for (const auto& chunk : import_chunks) {
        if (host >= chunk->host && host + size <= chunk->host + chunk->size) {
            return true;
        }
    }
    return false;
}

void Context::Impl::EnsureBacking(Pool& pool) {
    // Called for every buffer range an op touches: the common case (backed long ago) must not
    // even start the stall timer.
    const bool was_pending = pool.pending.load(std::memory_order_acquire);
    const bool was_upload = pool.upload_pending;
    if (!was_pending && !was_upload) {
        return;
    }
    const Common::FrameStats::Scope stall{Common::FrameStats::Cause::PoolBacking};
    const bool log = Debug::Get().upload_log;
    const auto begin = log ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (was_pending) {
        // Named in the status line: a slow host import (or the wait for the backing thread's)
        // otherwise shows under the stage of whatever ran before.
        const char* const stage_before =
            replay_stage.exchange("backing a pool (host import)", std::memory_order_relaxed);
        {
            std::scoped_lock lock{backing_mutex};
            BackPoolLocked(pool);
        }
        replay_stage.store(stage_before, std::memory_order_relaxed);
    }
    if (pool.upload_pending) {
        Pool* const one[] = {&pool};
        UploadWholePools(one);
    }
    if (log && (was_pending || was_upload)) {
        LOG_INFO(Render_Vulkan,
                 "[nvn] ensure backing: present {} pool {:#x} size {:.2f} MB flags {:#x} pending {} "
                 "upload {}: {:.2f} ms",
                 stats.presents.load(std::memory_order_relaxed), pool.address,
                 pool.size / 1048576.0, pool.flags, was_pending, was_upload,
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                     .count());
    }
}

void Context::Impl::BackingMain() {
    for (;;) {
        std::shared_ptr<Pool> pool;
        {
            std::unique_lock lock{backing_queue_mutex};
            const auto ready = [this] { return backing_stop || !backing_queue.empty(); };
            // While sparse split pools have granules left, an idle thread wakes four times a
            // second for their sweep (SweepSparseSplitPools).
            backing_cv.wait(lock, [&] {
                return ready() || sparse_split_live.load(std::memory_order_acquire) != 0;
            });
            if (!ready()) {
                backing_cv.wait_for(lock, std::chrono::milliseconds{250}, ready);
            }
            if (backing_stop) {
                return;
            }
            if (!backing_queue.empty()) {
                pool = std::move(backing_queue.front());
                backing_queue.pop_front();
            }
        }
        if (!pool) {
            SweepSparseSplitPools();
            continue;
        }
        std::unique_lock lock{backing_mutex};
        const auto skip = [&] {
            if (!pool->pending.load(std::memory_order_relaxed) ||
                pool->released.load(std::memory_order_relaxed)) {
                return true; // used (and backed) already, or destroyed
            }
            // A pool inside a chunk imported since needs no buffer of its own.
            const u8* host = (pool->flags & PoolFlag::CpuNoAccess)
                                 ? nullptr
                                 : GuestMappingPointer(pool->address, pool->size);
            return host != nullptr && InImportedChunk(host, pool->size);
        };
        if (skip()) {
            continue;
        }
        if (!(pool->flags & PoolFlag::CpuNoAccess) && options.allow_host_import && importer &&
            importer->Available()) {
            // The pages of the chunk the import will cover get their host pages now, without
            // backing_mutex (the replay thread may build the raw global table meanwhile): see
            // HostImporter::MakeResident. The import below then finds them resident.
            u8* const host = const_cast<u8*>(GuestMappingPointer(pool->address, pool->size));
            if (const auto window = host ? ImportWindow(pool->address, pool->size, host)
                                         : std::nullopt) {
                lock.unlock();
                HostImporter::MakeResident(window->host, window->size, true);
                lock.lock();
                if (skip()) {
                    continue;
                }
            } else if (host == nullptr && !pool->alias && AliasImportWanted(pool->size)) {
                // A split pool imported through an alias view (EDEN_NVN_ALIAS_IMPORT): the view is
                // mapped and its pages made resident here, without backing_mutex (the import
                // would otherwise fault them in under the driver lock: a 40-210 ms hitch for
                // DK's 450 MiB pool). Mapping the view reads the guest page table only.
                lock.unlock();
                std::vector<u64> runs;
                std::shared_ptr<u8> alias = MapPoolAlias(*pool, runs);
                if (alias) {
                    HostImporter::MakeResident(alias.get(), pool->size, false);
                }
                lock.lock();
                if (skip()) {
                    continue; // (the alias goes with `alias`)
                }
                if (alias && !pool->alias) {
                    pool->alias = std::move(alias);
                    pool->alias_runs = std::move(runs);
                }
            }
        }
        BackPoolLocked(*pool);
        pools_backed_background.fetch_add(1, std::memory_order_relaxed);
    }
}

void Context::Impl::BackPoolLocked(Pool& pool_ref) {
    if (!pool_ref.pending.load(std::memory_order_relaxed)) {
        return;
    }
    if (VC::PreRec::Detail::ra_ahead) [[unlikely]] {
        ++ra.pools_backed_ahead; // replay thread only (the flag is thread-local)
    }
    const VC::PoolDesc desc{pool_ref.serial, pool_ref.address, pool_ref.size, pool_ref.flags, 0};
    Pool* const pool = &pool_ref;
    using Clock = std::chrono::steady_clock;
    const auto pool_begin = Clock::now();
    const auto since = [](Clock::time_point t) {
        return static_cast<u64>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t).count());
    };
    u64 ns_pointer = 0;
    u64 ns_import = 0;
    u64 ns_buffer = 0;
    const char* path = "none";
    // Totals and the slowest pools are reported by DescribeState / the log.
    const auto report = [&] {
        // The raw global table covers backed pools; the state above is complete before
        // `pending` is cleared (a reader that sees it clear sees the backing).
        ++pool_generation;
        pool_ref.pending.store(false, std::memory_order_release);
        const u64 total = since(pool_begin);
        pool_ns_pointer.fetch_add(ns_pointer, std::memory_order_relaxed);
        pool_ns_import.fetch_add(ns_import, std::memory_order_relaxed);
        pool_ns_buffer.fetch_add(ns_buffer, std::memory_order_relaxed);
        pool_bytes.fetch_add(desc.size, std::memory_order_relaxed);
        if (total > 20'000'000 && pool_slow_logged.fetch_add(1) < 16) {
            LOG_INFO(Render_Vulkan,
                     "[nvn] slow pool create: {:#x} bytes flags {:#x} path {} total {:.1f} ms "
                     "(pointer {:.1f}, import {:.1f}, buffer {:.1f})",
                     desc.size, desc.flags, path, total / 1e6, ns_pointer / 1e6, ns_import / 1e6,
                     ns_buffer / 1e6);
        }
    };
    struct ScopeExit {
        std::function<void()> fn;
        ~ScopeExit() {
            fn();
        }
    } scope_exit{report};
    try {
        const bool cpu_access = !(desc.flags & PoolFlag::CpuNoAccess);
        if (cpu_access) {
            auto t = Clock::now();
            u8* host = const_cast<u8*>(GuestMappingPointer(desc.guest_address, desc.size));
            // EDEN_NVN_ALIAS_IMPORT: several host runs, imported through a contiguous alias view
            // of them (mapped by the backing thread already, or now).
            const bool via_alias = host == nullptr && AliasImportWanted(desc.size);
            if (via_alias) {
                if (!pool->alias) {
                    pool->alias = MapPoolAlias(*pool, pool->alias_runs);
                }
                host = pool->alias.get();
            }
            ns_pointer = since(t);
            pool->host_bytes = host;
            std::string import_error = "not attempted";
            VkResult import_result = VK_NOT_READY;
            if (host && options.allow_host_import && importer && importer->Available()) {
                t = Clock::now();
                import_result = ImportPoolBuffer(*pool, host, import_error);
                ns_import = since(t);
                path = pool->alias ? "import (alias view)" : "import";
                if (import_result == VK_SUCCESS) {
                    pool->backing = Pool::Backing::Imported;
                    stats.pools_imported.fetch_add(1, std::memory_order_relaxed);
                    if (pool->alias) {
                        const u64 n = pools_alias_imported.fetch_add(1) + 1;
                        LOG_INFO(Render_Vulkan,
                                 "[nvn] pool {:#x} ({:#x} bytes, flags {:#x}) imported through a "
                                 "contiguous alias view of its {} host runs at {:p} (pointer "
                                 "{:.1f} ms, import {:.1f} ms; alias pools {}, fallbacks {})",
                                 desc.guest_address, desc.size, desc.flags,
                                 pool->alias_runs.size(), static_cast<const void*>(host),
                                 ns_pointer / 1e6, ns_import / 1e6, n,
                                 pools_alias_failed.load());
                    }
                    return;
                }
                stats.host_import_failures.fetch_add(1, std::memory_order_relaxed);
                LOG_WARNING(Render_Vulkan,
                            "[nvn] host import of pool {:#x} ({:#x} bytes{}) failed: {} ({}); using "
                            "a device-local mirror",
                            desc.guest_address, desc.size, pool->alias ? ", alias view" : "",
                            import_error, static_cast<int>(import_result));
            }
            if (via_alias) {
                // Back to what a split pool gets today.
                if (!pool->alias) {
                    LOG_INFO(Render_Vulkan,
                             "[nvn] pool {:#x} ({:#x} bytes): no alias view (alignment, more "
                             "than {} host runs, or the host mapping failed)",
                             desc.guest_address, desc.size, ALIAS_MAX_RUNS);
                }
                pools_alias_failed.fetch_add(1, std::memory_order_relaxed);
                pool->imported.buffer = {}; // the buffer before its memory
                pool->imported.memory = {};
                pool->imported.size = 0;
                pool->alias.reset();
                pool->alias_runs.clear();
                host = nullptr;
                pool->host_bytes = nullptr;
            }
            t = Clock::now();
            pool->buffer = CreateHostBuffer(desc.size, pool_usage, MemoryUsage::DeviceLocal,
                                            AllocationKind::PoolBuffer);
            ns_buffer = since(t);
            path = "mirror";
            pool->backing = Pool::Backing::Mirror;
            pool->mirror_pages.assign(static_cast<size_t>((desc.size + Pool::MIRROR_PAGE - 1) /
                                                          Pool::MIRROR_PAGE),
                                      Pool::MirrorPage{});
            stats.pools_mirrored.fetch_add(1, std::memory_order_relaxed);
            if (host == nullptr) {
                stats.pools_mirrored_discontiguous.fetch_add(1, std::memory_order_relaxed);
                u64 runs = 0;
                u64 first_break = desc.size;
                CountHostRuns(desc.guest_address, desc.size, runs, first_break);
                // Raw global memory reaches the guest bytes only through imports: without them a
                // raw access into this pool reads 0 (its stores are dropped).
                std::string split_error = "raw_split_import off";
                if (raw_split_import.load(std::memory_order_relaxed)) {
                    if (options.allow_host_import && importer && importer->Available()) {
                        const auto split_begin = Clock::now();
                        if (ImportSplitRuns(*pool, split_error)) {
                            stats.pools_split_imported.fetch_add(1, std::memory_order_relaxed);
                            if (!pool->sparse) { // (a sparse pool's pieces count as they come)
                                stats.split_runs.fetch_add(pool->split_runs.size(),
                                                           std::memory_order_relaxed);
                            }
                            path = "split";
                            LOG_INFO(Render_Vulkan,
                                     "[nvn] pool {:#x} ({:#x} bytes, flags {:#x}) split-imported "
                                     "as {} {} (first break at +{:#x}, {:.1f} ms): raw global "
                                     "memory reads the guest bytes, draws bind the device-local "
                                     "copy",
                                     desc.guest_address, desc.size, desc.flags,
                                     pool->split_runs.size(),
                                     pool->sparse ? "sparse pieces so far" : "runs", first_break,
                                     since(split_begin) / 1e6);
                            return;
                        }
                    } else {
                        split_error = "host import not available";
                    }
                }
                LOG_INFO(Render_Vulkan,
                         "[nvn] pool {:#x} ({:#x} bytes, flags {:#x}) is a device-local mirror: the "
                         "guest storage is {} host runs (first break at +{:#x}), not one "
                         "contiguous host mapping; split import: {}",
                         desc.guest_address, desc.size, desc.flags, runs, first_break,
                         split_error);
            } else {
                stats.pools_mirrored_no_import.fetch_add(1, std::memory_order_relaxed);
                LOG_INFO(Render_Vulkan,
                         "[nvn] pool {:#x} ({:#x} bytes, flags {:#x}) is a device-local mirror: "
                         "host import not available",
                         desc.guest_address, desc.size, desc.flags);
            }
            return;
        }
        const auto t = Clock::now();
        pool->buffer = CreateHostBuffer(desc.size, pool_usage, MemoryUsage::DeviceLocal,
                                        AllocationKind::PoolBuffer);
        ns_buffer = since(t);
        path = "device-local";
        pool->backing = Pool::Backing::DeviceLocal;
        // The guest storage of a CPU_NO_ACCESS pool can still hold bytes the game placed there
        // before (and nxnative uses them): upload it once, at the first use (batched by the raw
        // table, or by EnsureBacking), not at creation.
        if (!pool->upload_pending) {
            pool->upload_pending = true;
            pools_upload_pending.fetch_add(1, std::memory_order_acq_rel);
        }
    } catch (const vk::Exception& exception) {
        pool->backing = Pool::Backing::None;
        stats.pools_unsupported.fetch_add(1, std::memory_order_relaxed);
        LOG_ERROR(Render_Vulkan, "[nvn] pool {:#x} ({:#x} bytes) allocation failed: {}",
                  desc.guest_address, desc.size, exception.what());
    }
}

void Context::Impl::UploadWholePools(std::span<Pool* const> targets) {
    if (targets.empty()) {
        return;
    }
    std::vector<PoolUpload> ranges;
    ranges.reserve(targets.size());
    for (Pool* const pool : targets) {
        if (pool->upload_pending) {
            pool->upload_pending = false;
            pools_upload_pending.fetch_sub(1, std::memory_order_acq_rel);
        }
        ranges.push_back({pool, 0, pool->size});
    }
    if (!UploadPoolRanges(ranges, false)) {
        for (Pool* const pool : targets) {
            pool->backing = Pool::Backing::None;
        }
        stats.pools_unsupported.fetch_add(targets.size(), std::memory_order_relaxed);
    }
}

bool Context::Impl::UploadPoolRanges(std::span<const PoolUpload> ranges, bool in_use) {
    using Clock = std::chrono::steady_clock;
    const auto upload_begin = Clock::now();
    u64 scan_ns = 0;
    u64 wait_ns = 0;
    u64 flushes = 0;
    u64 zero_bytes = 0;
    u64 copied_bytes = 0;
    constexpr u64 STAGING_SIZE = u64{64} << 20;
    VC::GuestMemory* memory = guest.load(std::memory_order_acquire);
    u64 total = 0;
    for (const PoolUpload& range : ranges) {
        total += range.size;
    }
    const u64 staging_size = (std::clamp)(total, u64{4096}, STAGING_SIZE);
    try {
        vk::Buffer staging =
            CreateHostBuffer(staging_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryUsage::Upload,
                             AllocationKind::Staging);
        u8* const mapped = staging.Mapped().data();
        struct Copy {
            VkBuffer destination;
            u64 destination_offset;
            u64 staging_offset;
            u64 size;
            bool zero;
        };
        std::vector<Copy> copies;
        u64 used = 0;
        const auto flush = [&] {
            if (copies.empty()) {
                return;
            }
            const auto wait_begin = Clock::now();
            staging.Flush();
            ImmediateSubmit([&](vk::CommandBuffer cmd) {
                // A pool the replay already uses: earlier submissions may still read or write the
                // range, later ones read the new bytes.
                const VkMemoryBarrier before{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
                    .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                };
                if (in_use) {
                    cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, before);
                }
                for (const Copy& copy : copies) {
                    if (copy.zero) {
                        // Buffers start with undefined contents: zero memory has to be written too.
                        cmd.FillBuffer(copy.destination, copy.destination_offset, copy.size, 0);
                    } else {
                        cmd.CopyBuffer(*staging, copy.destination,
                                       VkBufferCopy{.srcOffset = copy.staging_offset,
                                                    .dstOffset = copy.destination_offset,
                                                    .size = copy.size});
                    }
                }
                const VkMemoryBarrier after{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
                };
                if (in_use) {
                    cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, after);
                }
            });
            wait_ns += static_cast<u64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - wait_begin)
                    .count());
            ++flushes;
            copies.clear();
            used = 0;
        };
        for (const PoolUpload& range : ranges) {
            Pool* const pool = range.pool;
            const VkBuffer destination = pool->Handle();
            const u64 end = (std::min)(range.offset + range.size, pool->size);
            for (u64 offset = range.offset; offset < end; offset += staging_size) {
                const u64 count = (std::min)(staging_size, end - offset);
                const auto scan_begin = Clock::now();
                VC::PreRec::NoteRead(VC::PreRec::Cls::PoolUpload, pool->address + offset, count,
                                     pool->serial);
                const u8* source = GuestPointer(pool->address + offset, count);
                bool zero = source != nullptr ? IsAllZero(source, count) : memory == nullptr;
                if (!zero) {
                    if (used + count > staging_size) {
                        flush();
                    }
                    if (source) {
                        std::memcpy(mapped + used, source, count);
                    } else if (!memory->Read(pool->address + offset, mapped + used, count)) {
                        zero = true;
                    }
                }
                scan_ns += static_cast<u64>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - scan_begin)
                        .count());
                (zero ? zero_bytes : copied_bytes) += count;
                if (zero) {
                    copies.push_back({destination, offset, 0, count, true});
                    continue;
                }
                copies.push_back({destination, offset, used, count, false});
                used += (count + 15) / 16 * 16;
            }
        }
        flush();
        if (Debug::Get().upload_log) {
            LOG_INFO(Render_Vulkan,
                     "[nvn] pool contents upload: {} pools, {:.1f} MiB zero, {:.1f} MiB copied, {} "
                     "submits: {:.1f} ms total, {:.1f} ms scanning/copying (minus the waits), "
                     "{:.1f} ms in the submit waits",
                     ranges.size(), zero_bytes / 1048576.0, copied_bytes / 1048576.0, flushes,
                     std::chrono::duration<double, std::milli>(Clock::now() - upload_begin).count(),
                     scan_ns / 1e6 - wait_ns / 1e6, wait_ns / 1e6);
        }
        return true;
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "[nvn] uploading {} guest ranges into GPU-only pools failed: {}",
                  ranges.size(), exception.what());
        return false;
    }
}

void Context::Impl::CreateTextureNow(const VC::TextureDesc& d) {
    textures[d.serial] = BuildTexture(d);
}

bool Context::Impl::MkwBc3AtlasSupported() {
    if (mkw_bc3_atlas_supported) {
        return *mkw_bc3_atlas_supported;
    }
    bool supported = astc_standin != AstcStandin::Rgba8 &&
                     device.IsSparseResidencyImageSupported(VK_IMAGE_TYPE_2D) &&
                     device.CanUseUnbackedSparseImages();
    for (VkFormat format : {VK_FORMAT_BC3_UNORM_BLOCK, VK_FORMAT_BC3_SRGB_BLOCK}) {
        if (!supported) {
            break;
        }
        auto image = allocator->CreateUnbackedImage({
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = format,
            .extent = {8192, 8192, 1},
            .mipLevels = 14,
            .arrayLayers = 68,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        });
        supported = static_cast<bool>(image);
        if (image) {
            const auto requirements = device.GetLogical().GetImageSparseMemoryRequirements(*image);
            const auto memory = device.GetLogical().GetImageMemoryRequirements(*image);
            supported = memory.alignment == SPARSE_TILE_BYTES &&
                        std::ranges::any_of(requirements, [](const auto& requirement) {
                            const auto& props = requirement.formatProperties;
                            return props.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT &&
                                   props.imageGranularity.width == 256 &&
                                   props.imageGranularity.height == 256 &&
                                   props.imageGranularity.depth == 1 &&
                                   !(props.flags & VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT);
                        });
        }
    }
    mkw_bc3_atlas_supported = supported;
    LOG_INFO(Render_Vulkan, "[nvn] Mario Kart World: sparse ASTC 8x8 atlas backing {}",
             supported ? "BC3 (one quarter of RGBA8 backing, full guest dimensions)"
                       : "RGBA8 (BC3 sparse format/usage unavailable or recompression disabled)");
    return supported;
}

std::shared_ptr<Texture> Context::Impl::BuildTexture(const VC::TextureDesc& d) {
    auto texture = std::make_shared<Texture>();
    texture->serial = d.serial;
    texture->desc = d;

    if (d.view_of != 0) {
        const auto base = textures.find(d.view_of);
        if (base == textures.end()) {
            LOG_ERROR(Render_Vulkan, "[nvn] view {:#x} of unknown texture {:#x}", d.serial,
                      d.view_of);
            return texture;
        }
        texture->storage = base->second->storage;
        texture->host = base->second->host;
        if (d.view.set_format) {
            if (const auto host = ResolveHostFormat(device, d.view.format)) {
                texture->host = *host;
                // An ASTC view of a BC stand-in reads it as that BC format (its own sRGB-ness).
                if (texture->storage && texture->storage->host.bc_standin) {
                    Storage& s = *texture->storage;
                    texture->host = ToBcStandin(*host, BcStandinOf(s.host));
                    // BC4/BC5 have no sRGB form: a smart stand-in with an sRGB view takes BC1/BC3
                    // as an sRGB texture would. One that is BC4/BC5 already changes at an upload
                    // made at once (RefitStandin); until then the view has no format (ViewKeyOf).
                    if (s.smart_standin && !s.standin_srgb_view &&
                        VideoCore::Surface::IsPixelFormatSRGB(host->pixel_format)) {
                        s.standin_srgb_view = true;
                        const AstcStandin kind = BcStandinOf(s.host);
                        if (kind == AstcStandin::Bc4 || kind == AstcStandin::Bc5) {
                            s.force_upload = true;
                        }
                    }
                }
            }
        }
        return texture;
    }

    TextureLayoutDesc ld{
        .target = d.target,
        .format = d.format,
        .flags = d.flags,
        .samples = d.samples,
        .width = d.width,
        .height = d.height,
        .depth = d.depth,
        .levels = d.levels,
        .stride = d.stride,
        .pkg_block_h = d.pkg_block_h,
    };
    const auto host = ResolveHostFormat(device, d.format);
    if (!host) {
        stats.textures_failed.fetch_add(1, std::memory_order_relaxed);
        if (unsupported_formats_logged.insert(d.format).second) {
            LOG_WARNING(Render_Vulkan, "[nvn] texture format {:#x} has no host counterpart",
                        d.format);
        }
        return texture;
    }
    texture->host = *host;
    if (d.target == static_cast<s32>(TexTarget::Buffer)) {
        return texture; // texture buffers are views of a buffer range, not images
    }
    if (host->needs_transcode && !VideoCore::Surface::IsPixelFormatASTC(host->pixel_format) &&
        !VideoCore::Surface::IsPixelFormatBCn(host->pixel_format)) {
        // The device cannot hold the format and no decoder exists: nothing can fill the image.
        stats.textures_transcode_unsupported.fetch_add(1, std::memory_order_relaxed);
        if (unsupported_formats_logged.insert(d.format).second) {
            LOG_WARNING(Render_Vulkan,
                        "[nvn] texture format {:#x} needs a transcode to {} on this device and "
                        "has no decoder; its guest contents are not uploaded",
                        d.format, static_cast<int>(host->format));
        }
    }

    const TextureLayout layout = ComputeTextureLayout(ld);
    const BlockInfo block = FormatBlockInfo(d.format);
    const u64 begin = d.pool_offset;
    const u64 end = begin + layout.size;

    // The same bytes in the same shape share one storage, which outlives the textures that use it:
    // DK re-creates its transient render targets over the same pool bytes every frame.
    const StorageKey key{
        .pool_serial = d.pool_serial,
        .offset = begin,
        .size = layout.size,
        .target = d.target,
        .width = d.width,
        .height = d.height,
        .depth = d.depth,
        .levels = d.levels,
        .samples = d.samples,
        .flags_class = d.flags & (TexFlag::Sparse | TexFlag::Linear | TexFlag::LinearRenderTarget |
                                   TexFlag::MinimalLayout),
        .stride = d.stride,
        .pkg_block_h = d.pkg_block_h,
        .bytes_per_block = layout.bytes_per_block,
        .block_width = block.width,
        .block_height = block.height,
        .identity = host->needs_transcode ? d.format : 0,
        .vk_format = (host->depth || host->stencil) ? static_cast<u32>(host->format) : 0u,
        .host_caps = (host->attachable ? 1u : 0u) | (host->storage ? 2u : 0u),
    };
    if (const auto cached = storage_cache.find(key); cached != storage_cache.end()) {
        texture->storage = cached->second;
        if (texture->storage->host.bc_standin) {
            texture->host = texture->storage->host; // the key holds the format: the same stand-in
        }
        // The bytes under a re-created texture may have been rewritten since the last upload.
        if (texture->storage->origin == StorageOrigin::Cpu) {
            texture->storage->force_check = true;
        }
        stats.storages_reused.fetch_add(1, std::memory_order_relaxed);
        stats.textures_created.fetch_add(1, std::memory_order_relaxed);
        return texture;
    }

    auto storage = std::make_shared<Storage>();
    storage->host = *host;
    storage->desc = ld;
    storage->layout = layout;
    storage->pool_serial = d.pool_serial;
    storage->pool_offset = begin;
    const auto target = static_cast<TexTarget>(d.target);
    storage->type = (target == TexTarget::Dim1D || target == TexTarget::Dim1DArray)
                        ? VK_IMAGE_TYPE_1D
                    : target == TexTarget::Dim3D ? VK_IMAGE_TYPE_3D
                                                 : VK_IMAGE_TYPE_2D;
    storage->extent = {
        .width = (std::max)(d.width, 1u),
        .height = storage->type == VK_IMAGE_TYPE_1D ? 1u : (std::max)(d.height, 1u),
        .depth = storage->type == VK_IMAGE_TYPE_3D ? (std::max)(d.depth, 1u) : 1u,
    };
    storage->layers = IsLayered(target) ? (std::max)(layout.layers, 1u) : 1u;
    const u32 largest = (std::max)({storage->extent.width, storage->extent.height,
                                    storage->type == VK_IMAGE_TYPE_3D ? storage->extent.depth : 1u});
    const u32 max_levels = static_cast<u32>(std::bit_width(largest));
    storage->levels = std::clamp(d.levels ? d.levels : 1u, 1u, max_levels);
    storage->samples = d.samples > 1 ? static_cast<VkSampleCountFlagBits>(d.samples)
                                     : VK_SAMPLE_COUNT_1_BIT;
    storage->aspect = (host->depth ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) |
                      (host->stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
    if (!storage->aspect) {
        storage->aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    }

    storage->create_flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    if ((target == TexTarget::Cubemap || target == TexTarget::CubemapArray) &&
        storage->extent.width == storage->extent.height && storage->layers % 6 == 0) {
        storage->create_flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    }
    // Native shaders may sample six layers of a square 2D array as a cube (a point-light atlas):
    // such storages also take cube views.
    if (target == TexTarget::Dim2DArray && storage->extent.width == storage->extent.height &&
        storage->layers >= 6 && storage->samples == VK_SAMPLE_COUNT_1_BIT) {
        storage->create_flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    }
    if (storage->type == VK_IMAGE_TYPE_3D) {
        // A slice (or a range of slices) of a 3D render target attaches as a 2D / 2D_ARRAY view.
        storage->create_flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    }
    storage->usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT;
    if (host->attachable) {
        storage->usage |= (host->depth || host->stencil)
                              ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                              : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    }
    if (host->storage) {
        storage->usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    } else if (host->storage_view != VK_FORMAT_UNDEFINED || host->storage_alias) {
        // Storage-image views use the UNORM format of the same bits (HostFormat::storage_view) or
        // the format of an NVN view of the same texel size (HostFormat::storage_alias): the
        // image's own format cannot claim the usage.
        storage->usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        storage->create_flags |= VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
    }
    if (astc_gpu_enabled && host->needs_transcode &&
        VideoCore::Surface::IsPixelFormatASTC(host->pixel_format) &&
        (host->format == VK_FORMAT_A8B8G8R8_SRGB_PACK32 ||
         host->format == VK_FORMAT_A8B8G8R8_UNORM_PACK32) &&
        storage->type == VK_IMAGE_TYPE_2D && storage->samples == VK_SAMPLE_COUNT_1_BIT) {
        // The GPU decoder writes the levels through UNORM storage views (sRGB stand-ins have no
        // storage support of their own: the views carry the usage, see DrawResources::GetView).
        storage->usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        storage->create_flags |= VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
        storage->astc_gpu = true;
    }
    {
        // The size of the image once created (the cache budget counts it): every level and layer
        // at the host texel size.
        u64 texels = 0;
        for (u32 level = 0; level < storage->levels; ++level) {
            texels += static_cast<u64>((std::max)(storage->extent.width >> level, 1u)) *
                      (std::max)(storage->extent.height >> level, 1u) *
                      (storage->type == VK_IMAGE_TYPE_3D
                           ? (std::max)(storage->extent.depth >> level, 1u)
                           : 1u);
        }
        const u64 texel_bytes =
            host->needs_transcode ? 4u
                                  : (std::max)(layout.bytes_per_block / (std::max)(block.width * block.height, 1u), 1u);
        storage->vram_bytes = texels * storage->layers * texel_bytes *
                              (std::max)(static_cast<u32>(storage->samples), 1u);
    }

    // A large sparse texture in a virtual pool is bound and uploaded tile by tile (nvn_sparse.h);
    // the tiles the pool maps by now are marked, later remaps mark theirs (MapVirtualNow).
    if (const auto pool_it = pools.find(d.pool_serial);
        pool_it != pools.end() && pool_it->second->IsVirtual()) {
        const Pool& virtual_pool = *pool_it->second;
        const char* reason = "";
        if (std::shared_ptr<SparseBacking> sparse =
                SparseBacking::Make(*storage, device, virtual_pool.size, reason)) {
            sparse->Remap(
                0, layout.size,
                [&](u64 offset, u64 size) {
                    return VirtualRangeMapped(virtual_pool, begin + offset, size);
                },
                [&](u64 offset) { return SparseTilePhysical(virtual_pool, begin + offset); });
            storage->sparse = std::move(sparse);
            if (sparse_virtual_pools.insert(d.pool_serial).second) {
                mkw_sparse_users_dirty = true;
            }
        } else if (layout.size >= (u64{4} << 20) && virtual_storages_logged++ < 24) {
            LOG_INFO(Render_Vulkan,
                     "[nvn] texture {}x{}x{} ({} levels, format {:#x}, flags {:#x}, {:.1f} MiB) in "
                     "virtual pool {:#x} +{:#x} is not tile backed: {}; a remap uploads it whole",
                     d.width, d.height, storage->layers, storage->levels, d.format, d.flags,
                     layout.size / 1048576.0, d.pool_serial, begin, reason);
        }
    }

    // Every MKW ASTC 8x8 copy source and atlas uses the same fixed BC3 format. Smart per-image
    // formats would make streaming copies incompatible, and recreating an atlas loses GPU writes.
    const bool mkw_atlas_copy_format = mario_kart_world && IsAstc8x8(host->pixel_format);
    const bool mkw_bc3 = mkw_atlas_copy_format && host->needs_transcode &&
                         storage->type == VK_IMAGE_TYPE_2D &&
                         storage->samples == VK_SAMPLE_COUNT_1_BIT && MkwBc3AtlasSupported();
    if (mkw_bc3) {
        storage->astc_gpu = false;
        storage->usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                         VK_IMAGE_USAGE_SAMPLED_BIT;
        storage->create_flags &= ~VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
        SetStandin(*storage, AstcStandin::Bc3);
        texture->host = storage->host;
    }
    // Other sampled ASTC images retain the selected stand-in policy.
    if (!mkw_atlas_copy_format && astc_standin != AstcStandin::Rgba8 && host->needs_transcode &&
        VideoCore::Surface::IsPixelFormatASTC(host->pixel_format) &&
        storage->type == VK_IMAGE_TYPE_2D && storage->samples == VK_SAMPLE_COUNT_1_BIT &&
        !storage->sparse) {
        storage->astc_gpu = false;
        storage->usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                         VK_IMAGE_USAGE_SAMPLED_BIT;
        storage->create_flags &= ~VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
        // Smart selection: BC3 until EnsureImage reads the guest bytes (ChooseSmartStandin).
        storage->smart_standin = smart_bc.enabled;
        storage->host = *host;
        SetStandin(*storage, astc_standin);
        texture->host = storage->host;
    }

    // Overlap census: other storages over the same bytes (no data moves yet; that happens when one
    // is read after the other was written, see Recorder::ResolveOverlaps).
    PoolIndex& index = storage_index[d.pool_serial];
    {
        const u64 scan_from = begin > index.max_size ? begin - index.max_size : 0;
        bool overlaps = false;
        for (auto it = index.spans.lower_bound(scan_from);
             it != index.spans.end() && it->first < end; ++it) {
            if (it->first + it->second->layout.size > begin) {
                overlaps = true;
                break;
            }
        }
        if (overlaps) {
            const u64 seen = stats.alias_overlaps_seen.fetch_add(1, std::memory_order_relaxed);
            if (seen < 8) {
                LOG_INFO(Render_Vulkan,
                         "[nvn] texture {:#x} ({}x{}, format {:#x}, pool {:#x} +{:#x}, {:#x} "
                         "bytes) overlaps another texture's storage",
                         d.serial, d.width, d.height, d.format, d.pool_serial, begin,
                         layout.size);
            }
        }
    }
    index.spans.emplace(begin, storage.get());
    index.max_size = (std::max)(index.max_size, layout.size);
    storage->cached = true;
    storage_cache.emplace(key, storage);
    texture->storage = std::move(storage);
    stats.textures_created.fetch_add(1, std::memory_order_relaxed);
    return texture;
}

std::shared_ptr<Texture> Context::Impl::TargetAlias(const Texture& texture,
                                                    VC::RenderTargetRef& ref) {
    const Storage* const storage = texture.storage.get();
    const s32 format = RenderTargetFormat(texture, ref);
    if (storage == nullptr || !IsBlockView(storage->desc.format, format)) {
        return nullptr;
    }
    // A 3D level's alias holds all its slices: the target keeps its slice range on it.
    const bool volume = storage->desc.target == static_cast<s32>(TexTarget::Dim3D);
    if (!volume && ref.view.set_layers && ref.view.nlayers > 1) {
        return nullptr; // one layer per alias: a layered block-view target is not expressed
    }
    const u32 level = ref.texture.level;
    const u32 layer = volume ? 0 : ref.texture.layer;
    const auto key = std::make_tuple(texture.serial, level, layer, format);
    std::shared_ptr<Texture> built;
    if (const auto it = target_aliases.find(key); it != target_aliases.end()) {
        built = it->second;
    } else {
        const std::optional<BlockViewAlias> alias =
            BlockViewAliasOf(storage->desc, format, level, layer);
        if (!alias) {
            return nullptr;
        }
        VC::TextureDesc d{};
        d.target = alias->desc.target;
        d.format = format;
        d.flags = alias->desc.flags;
        d.samples = alias->desc.samples;
        d.width = alias->desc.width;
        d.height = alias->desc.height;
        d.depth = alias->desc.depth;
        d.levels = alias->desc.levels;
        d.stride = alias->desc.stride;
        d.pkg_block_h = alias->desc.pkg_block_h;
        d.serial = next_alias_serial--;
        d.pool_serial = storage->pool_serial;
        d.pool_offset = storage->pool_offset + alias->offset;
        built = BuildTexture(d);
        if (!built->storage) {
            return nullptr;
        }
        target_aliases.emplace(key, built);
    }
    VC::TextureViewDesc view{};
    if (volume) {
        view.set_layers = ref.view.set_layers;
        view.min_layer = ref.view.min_layer;
        view.nlayers = ref.view.nlayers;
    }
    ref.texture.level = 0;
    ref.texture.layer = volume ? ref.texture.layer : 0;
    ref.view = view;
    return built;
}

void Context::Impl::NoteUse(u64 serial, u64 seq) {
    UseSlot& slot = use_cache[(serial * 0x9E3779B97F4A7C15ull) >> 56];
    if (slot.serial == serial && slot.seq >= seq) {
        return;
    }
    u64& last = last_use[serial];
    last = (std::max)(last, seq);
    slot = {serial, last};
}

void Context::Impl::DestroyNow(u64 serial) {
    u64 after = 0;
    if (const auto it = last_use.find(serial); it != last_use.end()) {
        after = it->second;
        last_use.erase(it);
    }
    if (UseSlot& slot = use_cache[(serial * 0x9E3779B97F4A7C15ull) >> 56]; slot.serial == serial) {
        slot = {};
    }
    if (const auto it = pools.find(serial); it != pools.end()) {
        std::shared_ptr<Pool> pool = std::move(it->second);
        pools.erase(it);
        pool->released.store(true, std::memory_order_relaxed);
        if (pool->upload_pending) {
            pool->upload_pending = false;
            pools_upload_pending.fetch_sub(1, std::memory_order_acq_rel);
        }
        ReleaseVramMirror(*pool);
        if (pool->takeover_track != 0) {
            if (VC::GuestMemory* const memory = guest.load(std::memory_order_acquire)) {
                memory->Untrack(pool->takeover_track);
            }
            pool->takeover_track = 0;
        }
        if (pool->mirror_track != 0) {
            if (VC::GuestMemory* const memory = guest.load(std::memory_order_acquire)) {
                memory->Untrack(pool->mirror_track);
            }
            pool->mirror_track = 0;
        }
        if (pool->tex_track != 0) {
            if (tex_prehash && pool->tex_write_epoch) {
                tex_prehash->faults_window +=
                    pool->tex_write_epoch->load(std::memory_order_relaxed) -
                    pool->tex_faults_counted;
            }
            if (VC::GuestMemory* const memory = guest.load(std::memory_order_acquire)) {
                memory->Untrack(pool->tex_track);
            }
            pool->tex_track = 0;
        }
        {
            // A sparse split pool's write tracking. The backing thread may be backing the pool
            // right now (it holds backing_mutex from its look at `released` to the end): wait
            // for it, so the registration it makes is dropped here.
            std::unique_lock<std::mutex> lock{backing_mutex, std::defer_lock};
            if (pool->pending.load(std::memory_order_acquire)) {
                lock.lock();
            }
            if (pool->sparse) {
                if (const u64 track =
                        pool->sparse->track.exchange(0, std::memory_order_acq_rel);
                    track != 0) {
                    if (VC::GuestMemory* const memory = guest.load(std::memory_order_acquire)) {
                        memory->Untrack(track);
                    }
                }
            }
        }
        ++pool_generation;
        const auto by_address = pools_by_address.find(pool->address);
        if (by_address != pools_by_address.end() && by_address->second == pool) {
            pools_by_address.erase(by_address);
        }
        EvictPoolStorages(serial);
        if (pool->device_address != 0) {
            retired_addresses[retired_address_count++ % retired_addresses.size()] = {
                serial, pool->address, pool->device_address, pool->size, after,
                replay_seq.load(std::memory_order_relaxed)};
        }
        // (An alias view, Pool::alias, is unmapped with the pool's last reference, after its
        // buffer and imported memory: member order.)
        retired.push_back({after, [pool = std::move(pool)]() mutable { pool.reset(); }});
        return;
    }
    if (const auto it = textures.find(serial); it != textures.end()) {
        std::shared_ptr<Texture> texture = std::move(it->second);
        textures.erase(it);
        ++texture_epoch;
        // The image views of the draw path go with the texture, after the last submission that
        // used them completed.
        std::shared_ptr<TextureViews> views =
            draw_resources ? draw_resources->TakeViews(serial) : nullptr;
        // Its render-target aliases (TargetAlias) were last used with it (SetRenderTargets notes
        // the texture's serial).
        std::vector<std::pair<std::shared_ptr<Texture>, std::shared_ptr<TextureViews>>> aliases;
        for (auto alias = target_aliases.lower_bound(std::make_tuple(serial, 0u, 0u, s32{INT32_MIN}));
             alias != target_aliases.end() && std::get<0>(alias->first) == serial;
             alias = target_aliases.erase(alias)) {
            std::shared_ptr<TextureViews> alias_views =
                draw_resources ? draw_resources->TakeViews(alias->second->serial) : nullptr;
            aliases.emplace_back(std::move(alias->second), std::move(alias_views));
        }
        // So are the views of its scaled twin (Resolution setting), made under a serial of their own.
        if (texture->scaled) {
            std::shared_ptr<TextureViews> twin_views =
                draw_resources ? draw_resources->TakeViews(texture->scaled->serial) : nullptr;
            aliases.emplace_back(std::move(texture->scaled), std::move(twin_views));
        }
        retired.push_back({after, [texture = std::move(texture), views = std::move(views),
                                   aliases = std::move(aliases)]() mutable {
                               aliases.clear();
                               views.reset();
                               texture.reset();
                           }});
        return;
    }
    if (samplers.erase(serial) != 0 || programs.erase(serial) != 0) {
        return;
    }
    if (syncs.contains(serial)) {
        retired.push_back({after, [this, serial] { syncs.erase(serial); }});
    }
}

const Context::Impl::RawGlobalTable& Context::Impl::RawGlobalMemoryTable() {
    if (raw_table.generation == pool_generation.load(std::memory_order_acquire) &&
        !SparseSplitNeedsGrowth()) {
        return raw_table;
    }
    using Clock = std::chrono::steady_clock;
    if (VC::PreRec::Detail::ra_ahead) [[unlikely]] {
        ++ra.raw_rebuilds_ahead;
        ra.raw_rebuild_last_present = stats.presents.load(std::memory_order_relaxed);
    }
    const Common::FrameStats::Scope stall{Common::FrameStats::Cause::RawTableRebuild};
    const auto begin = Clock::now();
    // The backing thread backs pools (and bumps the generation) only under this lock: the pools,
    // the chunks and the generation the table is built from are one consistent state.
    std::scoped_lock backing_lock{backing_mutex};
    const auto locked_at = Clock::now();
    // Sparse split pools: the granules the game began to use are imported before the table is
    // built, so the draw that asked reads them. Nothing new and no other change: the table stays.
    if (!GrowSparseSplitPools() &&
        raw_table.generation == pool_generation.load(std::memory_order_acquire)) {
        return raw_table;
    }
    const auto has_own_buffer = [](const Pool& pool) {
        return (pool.backing == Pool::Backing::Imported && !pool.chunk) ||
               pool.backing == Pool::Backing::DeviceLocal;
    };

    // 1. Every pool is reachable through a chunk (one range for the pools it holds: CPU-visible
    // pools are guest memory the chunk maps) or has a buffer of its own. A pool inside an existing
    // chunk needs nothing; the others get their backing now (most of them already have it: the
    // backing thread works on every pool right after its creation). The GPU-only pools' guest
    // contents go to the GPU in one batch.
    u64 covered_by_chunk = 0;
    u64 backed_now = 0;
    std::vector<Pool*> virtual_pools;
    std::vector<Pool*> uploads;
    for (auto& [address, pool] : pools_by_address) {
        static_cast<void>(address);
        if (pool->backing == Pool::Backing::Virtual) {
            virtual_pools.push_back(pool.get());
            continue;
        }
        if (pool->pending) {
            if (!(pool->flags & PoolFlag::CpuNoAccess)) {
                const u8* host = GuestMappingPointer(pool->address, pool->size);
                if (host != nullptr && InImportedChunk(host, pool->size)) {
                    ++covered_by_chunk;
                    continue;
                }
            }
            BackPoolLocked(*pool);
            ++backed_now;
        }
        if (pool->upload_pending) {
            uploads.push_back(pool.get());
        }
    }
    const auto backed_at = Clock::now();
    UploadWholePools(uploads);
    const auto uploaded_at = Clock::now();

    // 2. The ranges. A pool with a buffer of its own wins over the chunk its guest range may lie
    // in (a CPU_NO_ACCESS pool next to imported ones); FlattenRawGlobalRanges cuts the chunk
    // ranges around such pools.
    RawGlobalTable table;
    std::vector<RawGlobalSource> sources;
    u64 skipped = 0;
    u64 split_pools = 0;
    u64 own_pools = 0;
    u64 own_bytes = 0;
    u64 own_small = 0;
    u64 own_in_chunk = 0;
    for (auto& [address, pool] : pools_by_address) {
        static_cast<void>(address);
        if (pool->backing == Pool::Backing::Virtual || pool->pending || pool->size == 0) {
            continue;
        }
        if (try_storage_vram && pool->vram_mirror && pool->vram_mirror->storage) {
            const u64 mirror_address =
                device.GetLogical().GetBufferDeviceAddress(*pool->vram_mirror->buffer);
            sources.push_back({pool->address, mirror_address, pool->size, 2});
            table.pool_serials.push_back(pool->serial);
            table.imported_pool = pool.get();
            ++own_pools;
            own_bytes += pool->size;
            continue;
        }
        if (pool->backing == Pool::Backing::Mirror && pool->split) {
            // The guest bytes of each imported host run (of a sparse pool: each piece imported
            // so far; the rest reads 0 until it is used), not the mirror the draws bind.
            for (Pool::SplitRun& run : pool->split_runs) {
                if (run.device_address == 0) {
                    run.device_address =
                        device.GetLogical().GetBufferDeviceAddress(*run.imported.buffer);
                }
                sources.push_back({pool->address + run.offset, run.device_address, run.size, 1});
            }
            table.pool_serials.push_back(pool->serial);
            ++split_pools;
            continue;
        }
        if (pool->backing == Pool::Backing::Mirror || pool->backing == Pool::Backing::None) {
            // A non-VRAM mirror pool's GPU copy is not the guest memory.
            ++skipped;
            continue;
        }
        if (!has_own_buffer(*pool)) {
            continue; // reached through its chunk
        }
        if (pool->device_address == 0) {
            pool->device_address = device.GetLogical().GetBufferDeviceAddress(pool->Handle());
        }
        sources.push_back({pool->address, pool->device_address, pool->size, 1});
        table.pool_serials.push_back(pool->serial);
        if (pool->backing == Pool::Backing::Imported) {
            table.imported_pool = pool.get();
        }
        ++own_pools;
        own_bytes += pool->size;
        if (pool->size < (u64{1} << 20)) {
            ++own_small;
        }
        if (const u8* host = GuestMappingPointer(pool->address, pool->size);
            host && InImportedChunk(host, pool->size)) {
            ++own_in_chunk;
        }
    }
    // A virtual pool's mapped runs resolve to ranges of physical pools.
    for (Pool* const virtual_pool : virtual_pools) {
        for (const auto& [offset, run] : virtual_pool->virtual_runs) {
            const auto physical_it = pools.find(run.physical_serial);
            Pool* const physical = physical_it != pools.end() ? physical_it->second.get() : nullptr;
            if (physical == nullptr) {
                ++skipped;
                continue;
            }
            if (physical->pending) {
                BackPoolLocked(*physical);
            }
            if (physical->upload_pending) {
                Pool* const one[] = {physical};
                UploadWholePools(one);
            }
            if (physical->backing == Pool::Backing::Mirror && physical->split) {
                // The parts of the mapped range in each host run of the physical pool.
                const u64 run_end = run.physical_offset + run.size;
                for (Pool::SplitRun& split : physical->split_runs) {
                    const u64 part_begin = (std::max)(run.physical_offset, split.offset);
                    const u64 part_end = (std::min)(run_end, split.offset + split.size);
                    if (part_begin >= part_end) {
                        continue;
                    }
                    if (split.device_address == 0) {
                        split.device_address =
                            device.GetLogical().GetBufferDeviceAddress(*split.imported.buffer);
                    }
                    sources.push_back({virtual_pool->address + offset +
                                           (part_begin - run.physical_offset),
                                       split.device_address + (part_begin - split.offset),
                                       part_end - part_begin, 1});
                }
                table.pool_serials.push_back(physical->serial);
                continue;
            }
            if (physical->backing != Pool::Backing::Imported &&
                physical->backing != Pool::Backing::DeviceLocal) {
                ++skipped;
                continue;
            }
            if (physical->device_address == 0) {
                physical->device_address =
                    device.GetLogical().GetBufferDeviceAddress(physical->Handle());
            }
            sources.push_back({virtual_pool->address + offset,
                               physical->device_address + run.physical_offset, run.size, 1});
            table.pool_serials.push_back(physical->serial);
        }
        table.pool_serials.push_back(virtual_pool->serial);
    }
    u64 chunk_ranges = 0;
    u64 chunk_buffers_made = 0;
    const auto chunks_at = Clock::now();
    for (const auto& chunk : import_chunks) {
        if (!chunk->address_buffer) {
            ++chunk_buffers_made;
            std::string error;
            if (importer->CreateBufferInChunk(*chunk, 0, chunk->size, pool_usage,
                                              chunk->address_buffer, error) != VK_SUCCESS) {
                LOG_WARNING(Render_Vulkan,
                            "[nvn] raw global memory: chunk {:#x} (+{:#x}) has no address buffer: {}",
                            chunk->guest_base, chunk->size, error);
                ++skipped;
                continue;
            }
            chunk->device_address = device.GetLogical().GetBufferDeviceAddress(*chunk->address_buffer);
        }
        // Below every pool range: a pool with a buffer of its own inside the chunk wins.
        sources.push_back({chunk->guest_base, chunk->device_address, chunk->size, 0});
        ++chunk_ranges;
    }
    // Pools in chunks stay alive as long as their chunk (import_chunks owns it). One of them
    // stands for "the host sees the writes" when a draw writes.
    if (table.imported_pool == nullptr) {
        for (auto& [address, pool] : pools_by_address) {
            static_cast<void>(address);
            if (!pool->pending && pool->backing == Pool::Backing::Imported) {
                table.imported_pool = pool.get();
                break;
            }
        }
    }

    const auto flatten_at = Clock::now();
    table.entries = FlattenRawGlobalRanges(std::move(sources));
    const size_t total_ranges = table.entries.size();
    table.device = UploadRawTable(table.entries);
    if (!table.device && !table.entries.empty()) {
        // No usable table: every raw access is dropped (counted).
        stats.raw_global_ranges_dropped.fetch_add(table.entries.size(), std::memory_order_relaxed);
    }
    // A gauge: the pools the current table leaves out (mirror or unbacked), not a running sum.
    stats.raw_global_pools_skipped.store(skipped, std::memory_order_relaxed);
    table.generation = pool_generation;
    const u64 rebuilds = stats.raw_global_rebuilds.fetch_add(1, std::memory_order_relaxed) + 1;
    const auto end = Clock::now();
    const auto ms = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    if (rebuilds <= 8 || rebuilds % 64 == 0 || ms(begin, end) >= 20.0) {
        u64 bytes = 0;
        for (const RawGlobalRange& entry : table.entries) {
            bytes += entry.size;
        }
        LOG_INFO(Render_Vulkan,
                 "[nvn] raw global memory table #{}: {} ranges ({:.1f} MiB): {} chunks, {} pools "
                 "with their own buffer, {} split-imported pools, {} virtual pools; {} pools in "
                 "chunks, {} backed now, {} skipped (mirror/unbacked), {:.1f} ms; own buffers: "
                 "{:.1f} MiB, {} under 1 MiB, {} inside chunks; parts (ms): lock wait {:.1f}, backing "
                 "{:.1f}, whole-pool uploads {:.1f} ({} pools), pool ranges {:.1f}, chunk ranges {:.1f} ({} "
                 "address buffers made), flatten + table upload {:.1f}",
                 rebuilds, total_ranges, bytes / 1048576.0, chunk_ranges, own_pools, split_pools,
                 virtual_pools.size(), covered_by_chunk, backed_now, skipped,
                 std::chrono::duration<double, std::milli>(Clock::now() - begin).count(),
                 own_bytes / 1048576.0, own_small, own_in_chunk, ms(begin, locked_at),
                 ms(locked_at, backed_at), ms(backed_at, uploaded_at), uploads.size(),
                 ms(uploaded_at, chunks_at), ms(chunks_at, flatten_at), chunk_buffers_made,
                 ms(flatten_at, end));
    }
    raw_table = std::move(table);
    return raw_table;
}

const Context::Impl::RawWriteThroughTable& Context::Impl::RawStoreAliasTable() {
    const bool read_only_enabled = Common::DiagRequest::TrySwitchEnabled("TRY_RO_MIRROR_EXACT");
    const bool storage_enabled = Common::DiagRequest::TrySwitchEnabled("TRY_STORAGE_RAW_WRITE_THROUGH");
    if (raw_write_through_table.mirror_generation == ro_alias_generation &&
        raw_write_through_table.read_only_enabled == read_only_enabled &&
        raw_write_through_table.storage_enabled == storage_enabled) {
        return raw_write_through_table;
    }
    struct AliasSource {
        Pool* pool;
        RawGlobalRange range;
    };
    std::vector<AliasSource> sources;
    for (Pool* const pool : vram_mirrored) {
        PoolMirror& mirror = *pool->vram_mirror;
        mirror.written_through = false;
        u64 target = 0;
        if (!mirror.storage && read_only_enabled) {
            target = device.GetLogical().GetBufferDeviceAddress(*mirror.buffer);
        } else if (mirror.storage && storage_enabled && pool->backing == Pool::Backing::Imported) {
            if (pool->device_address == 0) {
                pool->device_address = device.GetLogical().GetBufferDeviceAddress(pool->Handle());
            }
            target = pool->device_address;
        }
        if (target != 0) {
            sources.push_back({pool, {pool->address, target, pool->size}});
        }
    }
    std::ranges::stable_sort(sources, {}, [](const AliasSource& source) {
        return source.range.guest;
    });
    std::vector<RawGlobalRange> entries;
    std::vector<Pool*> included;
    u64 end = 0;
    for (const AliasSource& source : sources) {
        if (!entries.empty() && source.range.guest < end) {
            continue;
        }
        entries.push_back(source.range);
        included.push_back(source.pool);
        end = source.range.guest + source.range.size;
    }
    RawWriteThroughTable table;
    table.generation = ++raw_write_through_table_generation;
    table.mirror_generation = ro_alias_generation;
    table.read_only_enabled = read_only_enabled;
    table.storage_enabled = storage_enabled;
    table.device = UploadRawTable(entries);
    if (table.device) {
        table.low = entries.front().guest;
        table.high = entries.back().guest + entries.back().size;
        for (Pool* const pool : included) {
            pool->vram_mirror->written_through = true;
            table.pool_serials.push_back(pool->serial);
        }
    }
    stats.vram_raw_alias_rebuilds.fetch_add(1, std::memory_order_relaxed);
    raw_write_through_table = std::move(table);
    return raw_write_through_table;
}

std::shared_ptr<RawTableBuffer> Context::Impl::UploadRawTable(
    std::span<const RawGlobalRange> entries) {
    if (entries.empty()) {
        return nullptr;
    }
    try {
        auto table = std::make_shared<RawTableBuffer>();
        const VkDeviceSize bytes = entries.size() * sizeof(RawGlobalRange);
        table->count = static_cast<u32>(entries.size());
        table->buffer = CreateHostBuffer(
            bytes, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            MemoryUsage::DeviceLocal);
        table->address = device.GetLogical().GetBufferDeviceAddress(*table->buffer);
        vk::Buffer staging =
            CreateHostBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryUsage::Upload,
                             AllocationKind::Staging);
        std::memcpy(staging.Mapped().data(), entries.data(), bytes);
        staging.Flush();
        ImmediateSubmit([&](vk::CommandBuffer cmd) {
            cmd.CopyBuffer(*staging, *table->buffer, VkBufferCopy{.size = bytes});
            const VkMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                0, barrier, {}, {});
        });
        return table;
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "[nvn] raw global memory table ({} ranges) failed: {}",
                  entries.size(), exception.what());
        return nullptr;
    }
}

void AppendRawGlobalRange(std::vector<RawGlobalRange>& ranges, u64 guest, u64 device, u64 size) {
    if (size == 0) {
        return;
    }
    if (!ranges.empty()) {
        RawGlobalRange& last = ranges.back();
        if (last.guest + last.size == guest && last.device + last.size == device) {
            last.size += size;
            return;
        }
    }
    ranges.push_back({guest, device, size});
}

std::vector<RawGlobalRange> FlattenRawGlobalRanges(std::vector<RawGlobalSource> sources) {
    std::vector<RawGlobalRange> out;
    sources.erase(std::remove_if(sources.begin(), sources.end(),
                                 [](const RawGlobalSource& s) { return s.size == 0; }),
                  sources.end());
    if (sources.empty()) {
        return out;
    }
    // The index breaks priority ties: the earlier source wins.
    std::vector<u32> order(sources.size());
    for (u32 i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::ranges::sort(order, [&](u32 a, u32 b) { return sources[a].guest < sources[b].guest; });
    std::vector<u64> bounds;
    bounds.reserve(sources.size() * 2);
    for (const RawGlobalSource& source : sources) {
        bounds.push_back(source.guest);
        bounds.push_back(source.guest + source.size);
    }
    std::ranges::sort(bounds);
    bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());
    std::vector<u32> active;
    size_t next = 0;
    for (size_t i = 0; i + 1 < bounds.size(); ++i) {
        const u64 begin = bounds[i];
        const u64 end = bounds[i + 1];
        while (next < order.size() && sources[order[next]].guest <= begin) {
            active.push_back(order[next++]);
        }
        std::erase_if(active, [&](u32 index) {
            return sources[index].guest + sources[index].size <= begin;
        });
        u32 best = ~0u;
        for (const u32 index : active) {
            if (best == ~0u || sources[index].priority > sources[best].priority ||
                (sources[index].priority == sources[best].priority && index < best)) {
                best = index;
            }
        }
        if (best != ~0u) {
            const RawGlobalSource& source = sources[best];
            AppendRawGlobalRange(out, begin, source.device + (begin - source.guest), end - begin);
        }
    }
    return out;
}

void WriteRawGlobalTable(const RawTableBuffer* table, u64 coverage_ring, std::span<u8> bank,
                         const RawAliasHeader& alias) {
    static_assert(Shader::RawCoverage::BANK_RING_OFFSET + sizeof(u64) <=
                  Shader::Profile::RAW_GLOBAL_ALIAS_COUNT_OFFSET);
    static_assert(Shader::Profile::RAW_GLOBAL_ALIAS_HIGH_OFFSET + sizeof(u64) <= 0x1000u);
    // The header is ours, not NVN's: it sits past every stage's storage table (the mesh one is
    // the last) and the words the draw macros write.
    static_assert([] {
        for (u32 stage = 0; stage < VC::ShaderStageCount; ++stage) {
            if (VC::StorageBufferBase(static_cast<VC::ShaderStage>(stage)) +
                    VC::StorageBufferSlots * VC::StorageBufferStride >
                Shader::Profile::RAW_GLOBAL_TABLE_OFFSET) {
                return false;
            }
        }
        return true;
    }());
    static_assert(VC::DriverMeshDimsOffset + 3 * sizeof(u32) <=
                  Shader::Profile::RAW_GLOBAL_TABLE_OFFSET);
    static_assert(sizeof(RawGlobalRange) == Shader::Profile::RAW_GLOBAL_TABLE_ENTRY_SIZE);
    const u32 count = table != nullptr ? table->count : 0;
    const u64 address = table != nullptr ? table->address : 0;
    std::memcpy(bank.data() + Shader::Profile::RAW_GLOBAL_TABLE_OFFSET, &count, sizeof(count));
    std::memcpy(bank.data() + Shader::Profile::RAW_GLOBAL_TABLE_ADDRESS_OFFSET, &address,
                sizeof(address));
    std::memcpy(bank.data() + Shader::RawCoverage::BANK_RING_OFFSET, &coverage_ring,
                sizeof(coverage_ring));
    const bool has_alias = alias.table != nullptr && alias.table->count != 0;
    const u32 alias_count = has_alias ? alias.table->count : 0;
    const u64 alias_address = has_alias ? alias.table->address : 0;
    const u64 alias_low = has_alias ? alias.low : 0;
    const u64 alias_high = has_alias ? alias.high : 0;
    const u32 pad = 0;
    std::memcpy(bank.data() + Shader::Profile::RAW_GLOBAL_ALIAS_COUNT_OFFSET, &alias_count,
                sizeof(alias_count));
    std::memcpy(bank.data() + Shader::Profile::RAW_GLOBAL_ALIAS_COUNT_OFFSET + 4, &pad, sizeof(pad));
    std::memcpy(bank.data() + Shader::Profile::RAW_GLOBAL_ALIAS_ADDRESS_OFFSET, &alias_address,
                sizeof(alias_address));
    std::memcpy(bank.data() + Shader::Profile::RAW_GLOBAL_ALIAS_LOW_OFFSET, &alias_low,
                sizeof(alias_low));
    std::memcpy(bank.data() + Shader::Profile::RAW_GLOBAL_ALIAS_HIGH_OFFSET, &alias_high,
                sizeof(alias_high));
}

namespace {

/// The NVN replay's flights as the coverage rings' queue: a ring used by the flights up to `seq`
/// can be read once the completion thread has published `seq` (it waits for each flight's
/// timeline value in seq order); every segment that used a ring ends with a host barrier.
class FlightCoverageQueue final : public ::Vulkan::RawCoverageRings::Queue {
public:
    FlightCoverageQueue(const std::atomic<u64>& completed_seq_, u64 seq_)
        : completed_seq{completed_seq_}, seq{seq_} {}

    [[nodiscard]] u64 CurrentTick() override {
        return seq;
    }
    void HostReadBarrier() override {}
    [[nodiscard]] bool IsDone(u64 tick) override {
        return completed_seq.load(std::memory_order_acquire) >= tick;
    }

private:
    const std::atomic<u64>& completed_seq;
    u64 seq;
};

/// Why the raw global table (RawGlobalMemoryTable) leaves out a pool's byte.
const char* RawTableGapReason(const Pool& pool) {
    if (pool.pending.load(std::memory_order_acquire)) {
        return "pending";
    }
    switch (pool.backing) {
    case Pool::Backing::None:
        return "unbacked";
    case Pool::Backing::Imported:
        return "imported";
    case Pool::Backing::Mirror:
        return "mirror";
    case Pool::Backing::DeviceLocal:
        return "device-local";
    case Pool::Backing::Virtual:
        return "virtual";
    }
    return "?";
}

} // Anonymous namespace

u64 Context::Impl::RawCoverageRingFor(u64 seq) {
    if (!raw_coverage) {
        raw_coverage.emplace(device, *allocator, "hle");
    }
    if (!raw_coverage->Valid()) {
        return 0;
    }
    FlightCoverageQueue flights{completed_seq, seq};
    if (const auto* const window = raw_coverage->Poll(std::chrono::steady_clock::now(), flights)) {
        VideoCommon::ParseRawCoverageRing(window->bytes, raw_coverage_window);
        if (VideoCommon::RawLastHitVerifyEnabled()) {
            // Run totals (the replay thread is the only caller).
            static u64 run_mismatches = 0;
            static u64 run_samples = 0;
            static u64 run_windows = 0;
            static u64 logged_records = 0;
            run_mismatches += raw_coverage_window.lasthit_mismatches;
            run_samples += raw_coverage_window.lasthit_samples;
            ++run_windows;
            LOG_INFO(HW_GPU,
                     "[raw] lasthit verify window: mismatches={} sampled_hits={} records={} | run "
                     "windows={} mismatches={} sampled_hits={}",
                     raw_coverage_window.lasthit_mismatches, raw_coverage_window.lasthit_samples,
                     raw_coverage_window.lasthit_records.size(), run_windows, run_mismatches,
                     run_samples);
            for (const auto& record : raw_coverage_window.lasthit_records) {
                if (logged_records >= 64) {
                    break;
                }
                ++logged_records;
                LOG_INFO(HW_GPU,
                         "[raw] lasthit mismatch prog={:016x} stage={} guest={:#x} present={} "
                         "lasthit_low={:#010x} search_low={:#010x}",
                         record.program, VideoCommon::RawCoverageStageName(record.stage),
                         record.guest, record.present, record.fast_low, record.search_low);
            }
        }
        // A miss in a granule of a sparse split pool that is not imported: somebody uses it
        // unseen (a shader stores into memory the CPU never wrote). Imported with the next table.
        if (sparse_split_live.load(std::memory_order_acquire) != 0) {
            u64 asked = 0;
            for (const VideoCommon::RawCoverageMiss& miss : raw_coverage_window.misses) {
                Pool* pool = nullptr;
                u64 offset = 0;
                if (miss.guest >= VideoCommon::RAW_COVERAGE_NULL_LIMIT &&
                    ResolveBuffer(miss.guest, 1, pool, offset) &&
                    !pool->pending.load(std::memory_order_acquire) &&
                    WantSparseSplit(*pool, offset, 1)) {
                    ++asked;
                }
            }
            if (asked != 0) {
                sparse_split_miss_hits.fetch_add(asked, std::memory_order_relaxed);
                LOG_INFO(Render_Vulkan,
                         "[nvn] sparse split: raw global misses in {} granules that were not "
                         "imported; imported with the next raw table",
                         asked);
            }
        }
        // Classified against the table and the pools of now, up to a window after the accesses
        // (an address the table holds by now is `late`).
        std::vector<VideoCommon::RawGlobalRange> ranges;
        ranges.reserve(raw_table.entries.size());
        for (const RawGlobalRange& entry : raw_table.entries) {
            ranges.push_back({entry.guest, entry.device, entry.size});
        }
        const auto classify = [&](u64 guest_address) {
            return ClassifyHleRawMiss(guest_address, ranges, pools_by_address, pools);
        };
        for (const std::string& line : raw_coverage_report.AddWindow(
                 raw_coverage_window, window->dispatches, window->seconds, classify)) {
            LOG_INFO(HW_GPU, "{}", line);
        }
        raw_coverage->Recycle();
    }
    return raw_coverage->Use(flights);
}

VideoCommon::RawMissVerdict ClassifyHleRawMiss(
    u64 guest, std::span<const VideoCommon::RawGlobalRange> ranges,
    const std::map<u64, std::shared_ptr<Pool>>& pools_by_address,
    const std::unordered_map<u64, std::shared_ptr<Pool>>& pools) {
    const Pool* pool = nullptr;
    const Pool* physical = nullptr;
    bool physical_missing = false;
    if (auto it = pools_by_address.upper_bound(guest); it != pools_by_address.begin()) {
        const Pool& candidate = *std::prev(it)->second;
        if (guest - candidate.address < candidate.size) {
            pool = &candidate;
        }
    }
    // A pending pool's backing is still being decided (the backing thread publishes it with
    // `pending`): only a backed pool can be a virtual one.
    if (pool != nullptr && !pool->pending.load(std::memory_order_acquire) &&
        pool->backing == Pool::Backing::Virtual) {
        const u64 offset = guest - pool->address;
        const auto run_it = pool->virtual_runs.upper_bound(offset);
        const bool mapped = run_it != pool->virtual_runs.begin() &&
                            offset - std::prev(run_it)->first < std::prev(run_it)->second.size;
        if (!mapped) {
            pool = nullptr;
        } else if (const auto found = pools.find(std::prev(run_it)->second.physical_serial);
                   found != pools.end()) {
            physical = found->second.get();
        } else {
            physical_missing = true;
        }
    }
    VideoCommon::RawMissVerdict verdict =
        VideoCommon::ClassifyRawMiss(guest, ranges, [&](u64) { return pool != nullptr; });
    if (verdict.kind != VideoCommon::RawMissClass::MappedGap || pool == nullptr) {
        return verdict;
    }
    verdict.range_guest = pool->address;
    verdict.range_size = pool->size;
    if (physical_missing) {
        verdict.pool = "virtual->destroyed";
    } else if (physical == nullptr) {
        verdict.pool = RawTableGapReason(*pool);
    } else {
        static constexpr std::array<const char*, 5> on_physical{
            "virtual->unbacked", "virtual->imported", "virtual->mirror", "virtual->device-local",
            "virtual->virtual"};
        static_assert(static_cast<size_t>(Pool::Backing::None) == 0 &&
                      static_cast<size_t>(Pool::Backing::Imported) == 1 &&
                      static_cast<size_t>(Pool::Backing::Mirror) == 2 &&
                      static_cast<size_t>(Pool::Backing::DeviceLocal) == 3 &&
                      static_cast<size_t>(Pool::Backing::Virtual) == 4);
        verdict.pool = physical->pending.load(std::memory_order_acquire)
                           ? "virtual->pending"
                           : on_physical[static_cast<size_t>(physical->backing)];
    }
    return verdict;
}

} // namespace Vulkan::Nvn
