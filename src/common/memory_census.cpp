// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/memory_census.h"

#ifdef _WIN32

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string_view>
#include <utility>

#include <intrin.h>

#include <fmt/format.h>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "common/common_types.h"

namespace Common {
namespace {

constexpr u64 PageBytes = 4096;

/// QueryWorkingSetEx (psapi), looked up by name: kernel32 exports it as K32QueryWorkingSetEx.
struct WorkingSetEntry {
    void* address;
    ULONG_PTR attributes; // bit 0: the page is valid (in the working set)
};
using QueryWorkingSetFn = BOOL(WINAPI*)(HANDLE, PVOID, DWORD);

QueryWorkingSetFn QueryWorkingSet() {
    static const QueryWorkingSetFn function = [] {
        FARPROC address = nullptr;
        if (const HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll")) {
            address = GetProcAddress(kernel32, "K32QueryWorkingSetEx");
        }
        return reinterpret_cast<QueryWorkingSetFn>(address);
    }();
    return function;
}

/// Bytes of [base, base + size) in the working set (0 without QueryWorkingSetEx).
u64 ResidentBytes(uintptr_t base, u64 size) {
    const QueryWorkingSetFn query = QueryWorkingSet();
    if (query == nullptr) {
        return 0;
    }
    constexpr u64 Batch = 16384;
    thread_local std::vector<WorkingSetEntry> entries(Batch);
    u64 resident = 0;
    const u64 pages = size / PageBytes;
    for (u64 first = 0; first < pages; first += Batch) {
        const u64 count = (std::min)(Batch, pages - first);
        for (u64 i = 0; i < count; ++i) {
            entries[i].address = reinterpret_cast<void*>(base + (first + i) * PageBytes);
            entries[i].attributes = 0;
        }
        if (!query(GetCurrentProcess(), entries.data(),
                   static_cast<DWORD>(count * sizeof(WorkingSetEntry)))) {
            continue;
        }
        for (u64 i = 0; i < count; ++i) {
            resident += (entries[i].attributes & 1) != 0 ? PageBytes : 0;
        }
    }
    return resident;
}

const char* ProtectName(DWORD protect) {
    switch (protect & 0xFF) {
    case PAGE_NOACCESS:
        return "NA";
    case PAGE_READONLY:
        return "R";
    case PAGE_READWRITE:
        return "RW";
    case PAGE_WRITECOPY:
        return "WC";
    case PAGE_EXECUTE:
        return "X";
    case PAGE_EXECUTE_READ:
        return "RX";
    case PAGE_EXECUTE_READWRITE:
        return "RWX";
    case PAGE_EXECUTE_WRITECOPY:
        return "XWC";
    default:
        return "?";
    }
}

bool Executable(DWORD protect) {
    return (protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
                       PAGE_EXECUTE_WRITECOPY)) != 0;
}

const char* TypeName(DWORD type) {
    return type == MEM_PRIVATE ? "private" : type == MEM_MAPPED ? "mapped" : "image";
}

struct Allocation {
    u64 committed{};
    u64 resident{};
    DWORD type{};
    DWORD protect{}; ///< of its largest committed region
    u64 largest_region{};
    u32 regions{};
};

u64 MiB(u64 bytes) {
    return bytes >> 20;
}

std::mutex census_mutex;
std::map<uintptr_t, std::pair<u64, u64>> previous; ///< allocation base -> committed, resident

// --- Allocation tracking (EDEN_MEMORY_TRACK_ALLOC=1) ---
// Commits of 1 MiB and more, per calling module: a fixed table, filled lock-free by the hooks.
constexpr size_t TrackedModules = 64;
struct ModuleTally {
    std::atomic<HMODULE> module{nullptr};
    std::atomic<u64> bytes{0};
    std::atomic<u64> count{0};
};
std::array<ModuleTally, TrackedModules> tallies;
std::atomic<u64> untallied_bytes{0};
std::array<std::pair<u64, u64>, TrackedModules> tallies_seen{}; ///< at the previous census

using VirtualAllocFn = LPVOID(WINAPI*)(LPVOID, SIZE_T, DWORD, DWORD);
using NtAllocateFn = LONG(NTAPI*)(HANDLE, PVOID*, ULONG_PTR, PSIZE_T, ULONG, ULONG);
VirtualAllocFn real_virtual_alloc = nullptr;
NtAllocateFn real_nt_allocate = nullptr;

// The return addresses of the largest commits (64 MiB and up), for symbolizing their callers.
constexpr size_t BigSlots = 64;
std::array<std::atomic<uintptr_t>, BigSlots> big_callers{};
std::array<std::atomic<u64>, BigSlots> big_sizes{};
std::atomic<u32> big_next{0};

void Tally(void* return_address, u64 bytes) {
    if (bytes >= (u64{64} << 20)) {
        const u32 slot = big_next.fetch_add(1, std::memory_order_relaxed);
        if (slot < BigSlots) {
            big_callers[slot].store(reinterpret_cast<uintptr_t>(return_address), std::memory_order_relaxed);
            big_sizes[slot].store(bytes, std::memory_order_relaxed);
        }
    }
    HMODULE module = nullptr;
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       static_cast<LPCWSTR>(return_address), &module);
    for (ModuleTally& tally : tallies) {
        HMODULE current = tally.module.load(std::memory_order_acquire);
        if (current == nullptr) {
            HMODULE expected = nullptr;
            if (tally.module.compare_exchange_strong(expected, module)) {
                current = module;
            } else {
                current = expected;
            }
        }
        if (current == module) {
            tally.bytes.fetch_add(bytes, std::memory_order_relaxed);
            tally.count.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    untallied_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

LPVOID WINAPI TrackedVirtualAlloc(LPVOID address, SIZE_T size, DWORD type, DWORD protect) {
    LPVOID const result = real_virtual_alloc(address, size, type, protect);
    if (result != nullptr && (type & MEM_COMMIT) != 0 && size >= (SIZE_T{1} << 20)) {
        Tally(_ReturnAddress(), size);
    }
    return result;
}

LONG NTAPI TrackedNtAllocate(HANDLE process, PVOID* address, ULONG_PTR zero_bits, PSIZE_T size,
                             ULONG type, ULONG protect) {
    const LONG status = real_nt_allocate(process, address, zero_bits, size, type, protect);
    if (status >= 0 && (type & MEM_COMMIT) != 0 && size != nullptr &&
        *size >= (SIZE_T{1} << 20) && (process == GetCurrentProcess() || process == nullptr)) {
        Tally(_ReturnAddress(), *size);
    }
    return status;
}

/// Points the import-table slots of `module` named VirtualAlloc / NtAllocateVirtualMemory at the
/// tracking functions.
void PatchModule(HMODULE module) {
    auto* const base = reinterpret_cast<u8*>(module);
    const auto* const dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) {
        return;
    }
    const auto* const nt = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
    const IMAGE_DATA_DIRECTORY& directory =
        nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (directory.VirtualAddress == 0) {
        return;
    }
    for (auto* descriptor = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base + directory.VirtualAddress);
         descriptor->Name != 0; ++descriptor) {
        if (descriptor->OriginalFirstThunk == 0) {
            continue;
        }
        auto* names = reinterpret_cast<IMAGE_THUNK_DATA*>(base + descriptor->OriginalFirstThunk);
        auto* slots = reinterpret_cast<IMAGE_THUNK_DATA*>(base + descriptor->FirstThunk);
        for (; names->u1.AddressOfData != 0; ++names, ++slots) {
            if (IMAGE_SNAP_BY_ORDINAL(names->u1.Ordinal)) {
                continue;
            }
            const auto* const by_name =
                reinterpret_cast<const IMAGE_IMPORT_BY_NAME*>(base + names->u1.AddressOfData);
            const char* const name = reinterpret_cast<const char*>(by_name->Name);
            ULONG_PTR replacement = 0;
            if (std::strcmp(name, "VirtualAlloc") == 0) {
                replacement = reinterpret_cast<ULONG_PTR>(&TrackedVirtualAlloc);
            } else if (std::strcmp(name, "NtAllocateVirtualMemory") == 0) {
                replacement = reinterpret_cast<ULONG_PTR>(&TrackedNtAllocate);
            } else {
                continue;
            }
            DWORD old_protect = 0;
            if (VirtualProtect(&slots->u1.Function, sizeof(ULONG_PTR), PAGE_READWRITE, &old_protect)) {
                slots->u1.Function = replacement;
                VirtualProtect(&slots->u1.Function, sizeof(ULONG_PTR), old_protect, &old_protect);
            }
        }
    }
}

bool tracking_enabled = false;
std::vector<HMODULE> patched_modules;

/// Patches the modules loaded since the last call (under census_mutex).
void PatchNewModules() {
    using EnumModulesFn = BOOL(WINAPI*)(HANDLE, HMODULE*, DWORD, LPDWORD);
    static const auto enum_modules = reinterpret_cast<EnumModulesFn>(
        GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "K32EnumProcessModules"));
    if (enum_modules == nullptr) {
        return;
    }
    std::vector<HMODULE> modules(1024);
    DWORD needed = 0;
    if (!enum_modules(GetCurrentProcess(), modules.data(),
                      static_cast<DWORD>(modules.size() * sizeof(HMODULE)), &needed)) {
        return;
    }
    modules.resize((std::min<size_t>)(modules.size(), needed / sizeof(HMODULE)));
    // The system's own layers stay as they are: kernelbase's VirtualAlloc calls
    // NtAllocateVirtualMemory itself, which would count each allocation twice under kernelbase.
    const std::array<HMODULE, 3> system{GetModuleHandleW(L"kernel32.dll"),
                                        GetModuleHandleW(L"kernelbase.dll"),
                                        GetModuleHandleW(L"ntdll.dll")};
    for (const HMODULE module : modules) {
        if (std::find(patched_modules.begin(), patched_modules.end(), module) !=
            patched_modules.end()) {
            continue;
        }
        patched_modules.push_back(module);
        if (std::find(system.begin(), system.end(), module) == system.end()) {
            PatchModule(module);
        }
    }
}

std::string ModuleName(HMODULE module) {
    wchar_t path[MAX_PATH]{};
    if (module == nullptr || GetModuleFileNameW(module, path, MAX_PATH) == 0) {
        return "?";
    }
    std::wstring_view view{path};
    const size_t slash = view.find_last_of(L"\\/");
    if (slash != std::wstring_view::npos) {
        view = view.substr(slash + 1);
    }
    std::string name;
    for (const wchar_t c : view) {
        name += c < 128 ? static_cast<char>(c) : '?';
    }
    return name;
}

} // Anonymous namespace

void TrackLargeAllocations() {
    const char* const env = std::getenv("EDEN_MEMORY_TRACK_ALLOC");
    if (env == nullptr || std::strcmp(env, "1") != 0) {
        return;
    }
    std::scoped_lock lock{census_mutex};
    if (!tracking_enabled) {
        const HMODULE kernelbase = GetModuleHandleW(L"kernelbase.dll");
        const HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
        real_virtual_alloc = reinterpret_cast<VirtualAllocFn>(
            GetProcAddress(kernelbase != nullptr ? kernelbase : kernel32, "VirtualAlloc"));
        real_nt_allocate = reinterpret_cast<NtAllocateFn>(
            GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtAllocateVirtualMemory"));
        if (real_virtual_alloc == nullptr || real_nt_allocate == nullptr) {
            return;
        }
        tracking_enabled = true;
    }
    PatchNewModules();
}

std::vector<std::string> ProcessMemoryCensus() {
    std::scoped_lock lock{census_mutex};
    std::vector<std::string> lines;
    SYSTEM_INFO system{};
    GetSystemInfo(&system);
    std::map<uintptr_t, Allocation> allocations;
    struct Kind {
        u64 committed{};
        u64 resident{};
        u64 count{};
    };
    Kind private_kind, mapped_kind, image_kind, executable_kind;
    uintptr_t address = reinterpret_cast<uintptr_t>(system.lpMinimumApplicationAddress);
    const uintptr_t end = reinterpret_cast<uintptr_t>(system.lpMaximumApplicationAddress);
    while (address < end) {
        MEMORY_BASIC_INFORMATION info{};
        if (VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)) == 0) {
            break;
        }
        const uintptr_t region_base = reinterpret_cast<uintptr_t>(info.BaseAddress);
        const u64 region_size = info.RegionSize;
        if (info.State == MEM_COMMIT) {
            Allocation& allocation = allocations[reinterpret_cast<uintptr_t>(info.AllocationBase)];
            const u64 resident = ResidentBytes(region_base, region_size);
            allocation.committed += region_size;
            allocation.resident += resident;
            allocation.type = info.Type;
            if (region_size > allocation.largest_region) {
                allocation.largest_region = region_size;
                allocation.protect = info.Protect;
            }
            ++allocation.regions;
            Kind& kind = info.Type == MEM_PRIVATE  ? private_kind
                         : info.Type == MEM_MAPPED ? mapped_kind
                                                   : image_kind;
            kind.committed += region_size;
            kind.resident += resident;
            if (info.Type == MEM_PRIVATE && Executable(info.Protect)) {
                executable_kind.committed += region_size;
                executable_kind.resident += resident;
            }
        }
        if (region_base + region_size <= address) {
            break;
        }
        address = region_base + region_size;
    }
    for (const auto& [base, allocation] : allocations) {
        Kind& kind = allocation.type == MEM_PRIVATE  ? private_kind
                     : allocation.type == MEM_MAPPED ? mapped_kind
                                                     : image_kind;
        ++kind.count;
    }

    // Heaps: committed bytes per heap (HeapSummary), the largest three.
    u64 heap_committed = 0;
    std::vector<u64> heap_sizes;
    const DWORD heap_count = GetProcessHeaps(0, nullptr);
    std::vector<HANDLE> heaps(heap_count + 16);
    const DWORD got = GetProcessHeaps(static_cast<DWORD>(heaps.size()), heaps.data());
    const HANDLE default_heap = GetProcessHeap();
    u64 default_heap_committed = 0;
    for (DWORD i = 0; i < (std::min)(got, static_cast<DWORD>(heaps.size())); ++i) {
        HEAP_SUMMARY summary{};
        summary.cb = sizeof(summary);
        if (HeapSummary(heaps[i], 0, &summary)) {
            heap_committed += summary.cbCommitted;
            heap_sizes.push_back(summary.cbCommitted);
            if (heaps[i] == default_heap) {
                default_heap_committed = summary.cbCommitted;
            }
        }
    }
    std::sort(heap_sizes.rbegin(), heap_sizes.rend());
    std::string largest_heaps;
    for (size_t i = 0; i < (std::min<size_t>)(3, heap_sizes.size()); ++i) {
        largest_heaps += fmt::format("{}{}", i == 0 ? "" : ", ", MiB(heap_sizes[i]));
    }
    lines.push_back(fmt::format(
        "census MiB committed/resident: private {}/{} ({} allocations), mapped {}/{} ({}), image "
        "{}/{} ({}); executable private {}/{}; heaps {} committed {} (process heap {}; largest "
        "{})",
        MiB(private_kind.committed), MiB(private_kind.resident), private_kind.count,
        MiB(mapped_kind.committed), MiB(mapped_kind.resident), mapped_kind.count,
        MiB(image_kind.committed), MiB(image_kind.resident), image_kind.count,
        MiB(executable_kind.committed), MiB(executable_kind.resident), got,
        MiB(heap_committed), MiB(default_heap_committed), largest_heaps));

    // The largest allocations by resident bytes.
    std::vector<std::pair<uintptr_t, const Allocation*>> by_resident;
    by_resident.reserve(allocations.size());
    for (const auto& [base, allocation] : allocations) {
        by_resident.emplace_back(base, &allocation);
    }
    std::sort(by_resident.begin(), by_resident.end(), [](const auto& a, const auto& b) {
        return a.second->resident > b.second->resident;
    });
    std::string largest;
    for (size_t i = 0; i < (std::min<size_t>)(12, by_resident.size()); ++i) {
        const Allocation& a = *by_resident[i].second;
        largest += fmt::format("{}{:#x} {} {} {}/{}", i == 0 ? "" : "; ", by_resident[i].first,
                               TypeName(a.type), ProtectName(a.protect), MiB(a.committed),
                               MiB(a.resident));
    }
    lines.push_back("largest by resident (base kind protection committed/resident MiB): " +
                    largest);

    // Growth since the previous census, per allocation (new ones count from zero).
    struct Growth {
        uintptr_t base;
        s64 resident;
        s64 committed;
        const Allocation* allocation;
    };
    std::vector<Growth> growth;
    s64 freed_resident = 0;
    u64 freed_count = 0;
    u64 new_count = 0;
    if (!previous.empty()) {
        for (const auto& [base, allocation] : allocations) {
            const auto it = previous.find(base);
            const u64 before_committed = it != previous.end() ? it->second.first : 0;
            const u64 before_resident = it != previous.end() ? it->second.second : 0;
            new_count += it == previous.end() ? 1 : 0;
            growth.push_back({base,
                              static_cast<s64>(allocation.resident) - static_cast<s64>(before_resident),
                              static_cast<s64>(allocation.committed) - static_cast<s64>(before_committed),
                              &allocation});
        }
        for (const auto& [base, sizes] : previous) {
            if (!allocations.contains(base)) {
                freed_resident += static_cast<s64>(sizes.second);
                ++freed_count;
            }
        }
        std::sort(growth.begin(), growth.end(),
                  [](const Growth& a, const Growth& b) { return a.resident > b.resident; });
        s64 grown_resident = 0;
        s64 grown_committed = 0;
        for (const Growth& g : growth) {
            grown_resident += g.resident;
            grown_committed += g.committed;
        }
        std::string top;
        for (size_t i = 0; i < (std::min<size_t>)(10, growth.size()); ++i) {
            const Growth& g = growth[i];
            if (g.resident <= 0) {
                break;
            }
            top += fmt::format("{}{:#x} {} {} +{}/{:+} MiB (now {}/{})", i == 0 ? "" : "; ",
                               g.base, TypeName(g.allocation->type),
                               ProtectName(g.allocation->protect), g.resident >> 20,
                               g.committed >> 20, MiB(g.allocation->committed),
                               MiB(g.allocation->resident));
        }
        lines.push_back(fmt::format(
            "since the last census: resident {:+} MiB, committed {:+} MiB over live allocations "
            "({} new), {} freed holding {} MiB resident; grown most (resident/committed): {}",
            grown_resident >> 20, grown_committed >> 20, new_count, freed_count,
            freed_resident >> 20, top.empty() ? std::string{"none"} : top));
        // A heap's large blocks (C++ new / malloc above ~0.5 MiB) get regions of their own that
        // start with list links to the heap's other large blocks (or its list head, inside the
        // heap): two pointer-sized values pointing into other private allocations. Anything else
        // came from VirtualAlloc directly (the driver, Vulkan host memory, JIT caches).
        const auto points_into = [&](u64 value) {
            auto it = allocations.upper_bound(static_cast<uintptr_t>(value));
            if (it == allocations.begin()) {
                return false;
            }
            --it;
            return it->second.type == MEM_PRIVATE && value < it->first + it->second.committed + 0x10000;
        };
        const auto heap_block = [&](uintptr_t base) {
            MEMORY_BASIC_INFORMATION info{};
            if (VirtualQuery(reinterpret_cast<const void*>(base), &info, sizeof(info)) == 0 ||
                info.State != MEM_COMMIT || (info.Protect & (PAGE_READWRITE | PAGE_READONLY)) == 0 ||
                (info.Protect & PAGE_GUARD) != 0 || ResidentBytes(base, PageBytes) == 0) {
                return false; // reading it would fault it in or fault
            }
            const u64* const links = reinterpret_cast<const u64*>(base);
            return links[0] != 0 && links[1] != 0 && links[0] != links[1] && points_into(links[0]) &&
                   points_into(links[1]);
        };
        // The new private allocations by committed size: a size that recurs names its producer.
        std::map<u64, std::pair<u64, u64>> new_sizes; // MiB -> count, resident bytes
        u64 new_private_committed = 0;
        u64 new_private_resident = 0;
        u64 heap_new_committed = 0;
        u64 heap_new_resident = 0;
        u64 heap_new_count = 0;
        for (const auto& [base, allocation] : allocations) {
            if (allocation.type != MEM_PRIVATE || previous.contains(base)) {
                continue;
            }
            if (allocation.committed >= (u64{1} << 20) && heap_block(base)) {
                heap_new_committed += allocation.committed;
                heap_new_resident += allocation.resident;
                ++heap_new_count;
            }
            auto& bucket = new_sizes[(allocation.committed + (u64{1} << 19)) >> 20];
            ++bucket.first;
            bucket.second += allocation.resident;
            new_private_committed += allocation.committed;
            new_private_resident += allocation.resident;
        }
        std::vector<std::pair<u64, std::pair<u64, u64>>> sizes(new_sizes.begin(), new_sizes.end());
        std::sort(sizes.begin(), sizes.end(), [](const auto& a, const auto& b) {
            return a.first * a.second.first > b.first * b.second.first;
        });
        std::string histogram;
        for (size_t i = 0; i < (std::min<size_t>)(12, sizes.size()); ++i) {
            histogram += fmt::format("{}{} MiB x{} ({} resident)", i == 0 ? "" : ", ", sizes[i].first,
                                     sizes[i].second.first, MiB(sizes[i].second.second));
        }
        lines.push_back(fmt::format("new private allocations: {} MiB committed, {} MiB resident; "
                                    "of them heap large blocks (1 MiB and up) {} MiB/{} MiB in {}; "
                                    "by size (largest total first): {}",
                                    MiB(new_private_committed), MiB(new_private_resident),
                                    MiB(heap_new_committed), MiB(heap_new_resident), heap_new_count,
                                    histogram.empty() ? std::string{"none"} : histogram));
    }
    previous.clear();
    for (const auto& [base, allocation] : allocations) {
        previous.emplace(base, std::make_pair(allocation.committed, allocation.resident));
    }
    if (tracking_enabled) {
        PatchNewModules(); // modules loaded since (the Vulkan driver loads after boot)
        struct Row {
            HMODULE module;
            u64 bytes;
            u64 count;
            u64 delta_bytes;
            u64 delta_count;
        };
        std::vector<Row> rows;
        for (size_t i = 0; i < tallies.size(); ++i) {
            const HMODULE module = tallies[i].module.load(std::memory_order_acquire);
            if (module == nullptr) {
                break;
            }
            const u64 bytes = tallies[i].bytes.load(std::memory_order_relaxed);
            const u64 count = tallies[i].count.load(std::memory_order_relaxed);
            rows.push_back({module, bytes, count, bytes - tallies_seen[i].first,
                            count - tallies_seen[i].second});
            tallies_seen[i] = {bytes, count};
        }
        std::sort(rows.begin(), rows.end(),
                  [](const Row& a, const Row& b) { return a.delta_bytes > b.delta_bytes; });
        std::string text;
        for (const Row& row : rows) {
            text += fmt::format("{}{} +{} MiB in {} (all {} MiB in {})", text.empty() ? "" : "; ",
                                ModuleName(row.module), MiB(row.delta_bytes), row.delta_count,
                                MiB(row.bytes), row.count);
        }
        lines.push_back(fmt::format(
            "commits of 1 MiB and more by calling module since the last census: {}{}",
            text.empty() ? std::string{"none"} : text,
            untallied_bytes.load() != 0
                ? fmt::format("; untallied {} MiB", MiB(untallied_bytes.load()))
                : std::string{}));
        // Callers of the commits of 64 MiB and more, as module+offset (symbolize with the PDB).
        std::string big;
        const u32 count = (std::min<u32>)(big_next.load(), static_cast<u32>(BigSlots));
        for (u32 i = 0; i < count; ++i) {
            const uintptr_t caller = big_callers[i].load(std::memory_order_relaxed);
            HMODULE module = nullptr;
            GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCWSTR>(caller), &module);
            big += fmt::format("{}{}+{:#x} {} MiB", big.empty() ? "" : "; ", ModuleName(module),
                               caller - reinterpret_cast<uintptr_t>(module),
                               MiB(big_sizes[i].load(std::memory_order_relaxed)));
        }
        if (!big.empty()) {
            lines.push_back("commits of 64 MiB and more so far, by caller: " + big);
        }
    }
    return lines;
}

} // namespace Common

#else

namespace Common {
std::vector<std::string> ProcessMemoryCensus() {
    return {};
}
void TrackLargeAllocations() {}
} // namespace Common

#endif
