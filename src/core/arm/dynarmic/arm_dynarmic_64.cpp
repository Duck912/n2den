// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <map>
#include <mutex>
#include <vector>
#include <cstring>
#include <ranges>
#include <unordered_map>

#include "common/emu_counters.h"
#include "common/settings.h"
#include "core/arm/debug.h"
#include "core/arm/symbols.h"
#include "core/arm/dynarmic/arm_dynarmic.h"
#include "core/arm/dynarmic/arm_dynarmic_64.h"
#include "core/arm/dynarmic/jit_block_cache.h"
#include "core/arm/dynarmic/dynarmic_exclusive_monitor.h"
#include "core/core_timing.h"
#include "core/guest_clock.h"
#include "core/hle/kernel/k_process.h"
#include "dynarmic/interface/A64/config.h"

namespace Core {

using namespace Common::Literals;

DynarmicCallbacks64::DynarmicCallbacks64(ArmDynarmic64& parent, Kernel::KProcess* process)
    : m_parent{parent}, m_memory(process->GetMemory())
    , m_process(process), m_debugger_enabled{parent.m_system.DebuggerEnabled()}
    , m_check_memory_access{m_debugger_enabled || !Settings::values.cpuopt_ignore_memory_aborts.GetValue()}
{}

u64 DynarmicCallbacks64::MemoryRead(u64 vaddr, size_t size) {
    // EDEN_DIAG_UNMAPPED_BT=1 (diagnostic, off by default): the guest registers and backtrace at
    // the first few reads below 0x08000000, where no 64-bit program has memory (a read through a
    // null or truncated pointer). The PC is the one the JIT last stored, so it can lag inside the
    // block; LR and the registers are exact.
    static const bool diag_unmapped_bt = [] {
        const char* const value = std::getenv("EDEN_DIAG_UNMAPPED_BT");
        return value != nullptr && value[0] != '0';
    }();
    if (diag_unmapped_bt && vaddr < 0x08000000) [[unlikely]] {
        static std::atomic<u32> logged{};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 4) {
            LOG_CRITICAL(Core_ARM, "EDEN_DIAG_UNMAPPED_BT: read of {} bytes at {:#x}", size, vaddr);
            m_parent.LogBacktrace(m_process);
        }
    }
    CheckMemoryAccess(vaddr, size, Kernel::DebugWatchpointType::Read);
    switch (size) {
    case sizeof(u64): return m_memory.Read64(vaddr);
    case sizeof(u32): return m_memory.Read32(vaddr);
    case sizeof(u16): return m_memory.Read16(vaddr);
    case sizeof(u8): return m_memory.Read8(vaddr);
    default: UNREACHABLE();
    }
}
Dynarmic::A64::Vector DynarmicCallbacks64::MemoryRead128(u64 vaddr) {
    CheckMemoryAccess(vaddr, 16, Kernel::DebugWatchpointType::Read);
    return {m_memory.Read64(vaddr), m_memory.Read64(vaddr + 8)};
}

namespace {

// EDEN_NVN_PROBE=1: step 1 of a native NVN mode. A Switch 2 game's sdk module exports the NVN driver's C entry points
// (nvnBootstrapLoader and the nvn* procs) in its dynamic symbol table, which is how a recomp hooks them without
// per-game addresses. This lists the NVN functions the game's modules define and logs each one the first time the JIT
// translates its entry, so the log shows which NVN calls the game makes. It reads only: the game runs as before. Run
// it with EDEN_JIT_CACHE=0, or blocks loaded from the cache are never translated.
bool NvnProbeOn() {
    static const bool on = [] {
        const char* value = std::getenv("EDEN_NVN_PROBE");
        return value != nullptr && *value != 0 && *value != '0';
    }();
    return on;
}

struct NvnProbe {
    std::mutex mutex;
    bool built = false;
    std::unordered_map<u64, std::string> entries;   // entry address -> symbol
    std::unordered_map<u64, bool> seen;
    u64 first_run = 0;
};

NvnProbe& GetNvnProbe() {
    static NvnProbe probe;
    return probe;
}

bool IsNvnSymbol(std::string_view name) {
    if (name.starts_with("nvn")) {
        return true;
    }
    // C++ names: _Z<length>nvn<...> (the Switch 2 driver exports some, nvnNativeImageInitialize among them)
    if (!name.starts_with("_Z")) {
        return false;
    }
    std::size_t i = 2;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
        ++i;
    }
    return i > 2 && name.substr(i).starts_with("nvn");
}

// The driver's proc table: nvnDeviceGetProcAddress looks names up in it, so it holds a pointer to each "nvn..." name
// and, next to it, the function. Find the names in the module, the pointers to them, and a code address beside each.
void ScanNvnProcTable(NvnProbe& probe, Kernel::KProcess* process, Core::Memory::Memory& memory, u64 base) {
    const u64 end = GetInteger(Core::GetModuleEnd(process, base));
    if (end <= base || end - base > 512_MiB) {
        LOG_INFO(Core_ARM, "NVN probe: module {:#x} end {:#x}: not scanned", base, end);
        return;
    }
    std::vector<u8> image(end - base);
    memory.ReadBlock(base, image.data(), image.size());
    // code range: the first region (read-execute) of the module
    Kernel::KMemoryInfo info{};
    Kernel::Svc::PageInfo page_info{};
    u64 code_end = base;
    if (R_SUCCEEDED(process->GetPageTable().QueryInfo(std::addressof(info), std::addressof(page_info), base))) {
        const auto svc = info.GetSvcMemoryInfo();
        code_end = svc.base_address + svc.size;
    }
    std::unordered_map<u64, std::string> names;   // address -> name
    for (std::size_t i = 0; i + 4 < image.size(); ++i) {
        if (image[i] == 'n' && image[i + 1] == 'v' && image[i + 2] == 'n' && image[i + 3] >= 'A' &&
            image[i + 3] <= 'Z' && (i == 0 || image[i - 1] == 0)) {
            std::size_t j = i;
            while (j < image.size() && image[j] != 0 && j - i < 128) {
                ++j;
            }
            if (j < image.size() && image[j] == 0) {
                names.emplace(base + i, std::string(reinterpret_cast<const char*>(&image[i]), j - i));
                i = j;
            }
        }
    }
    std::size_t pointers = 0, pairs = 0, added = 0;
    for (std::size_t i = 0; i + 16 <= image.size(); i += 8) {
        u64 value;
        std::memcpy(&value, &image[i], 8);
        const auto it = names.find(value);
        if (it == names.end()) {
            continue;
        }
        ++pointers;
        for (const std::ptrdiff_t delta : {8, -8}) {
            const std::ptrdiff_t at = static_cast<std::ptrdiff_t>(i) + delta;
            if (at < 0 || static_cast<std::size_t>(at) + 8 > image.size()) {
                continue;
            }
            u64 fn;
            std::memcpy(&fn, &image[at], 8);
            if (fn >= base && fn < code_end) {
                ++pairs;
                if (probe.entries.emplace(fn, it->second).second) {
                    ++added;
                }
                break;
            }
        }
    }
    LOG_INFO(Core_ARM, "NVN probe: module {:#x}: {} nvn names, {} pointers to them, {} beside code, {} new entry points",
             base, names.size(), pointers, pairs, added);
}

void BuildNvnProbe(NvnProbe& probe, Kernel::KProcess* process, Core::Memory::Memory& memory) {
    probe.built = true;
    // Every executable code region, named or not (FindModules skips modules without a path after their code, and
    // found none in the Switch 2 Ocarina of Time).
    std::map<u64, std::string> modules;
    for (const auto& [base, module_name] : Core::FindModules(process)) {
        modules.emplace(base, std::string(module_name));
    }
    LOG_INFO(Core_ARM, "NVN probe: FindModules named {} modules", modules.size());
    auto& page_table = process->GetPageTable();
    u64 address = 0;
    while (true) {
        Kernel::KMemoryInfo info{};
        Kernel::Svc::PageInfo page_info{};
        if (R_FAILED(page_table.QueryInfo(std::addressof(info), std::addressof(page_info), address))) {
            break;
        }
        const auto svc = info.GetSvcMemoryInfo();
        if (svc.permission == Kernel::Svc::MemoryPermission::ReadExecute &&
            (svc.state == Kernel::Svc::MemoryState::Code ||
             svc.state == Kernel::Svc::MemoryState::AliasCode)) {
            modules.emplace(svc.base_address, fmt::format("code@{:#x}", svc.base_address));
        }
        const u64 next = svc.base_address + svc.size;
        if (next <= address) {
            break;
        }
        address = next;
    }
    for (const auto& [base, module_name] : modules) {
        const auto symbols = Core::Symbols::GetSymbols(base, memory, true);
        std::size_t defined = 0, imported = 0;
        for (const auto& [name, value] : symbols) {
            if (!IsNvnSymbol(name)) {
                continue;
            }
            if (value.first == 0) {
                ++imported;
                continue;
            }
            ++defined;
            probe.entries.emplace(base + value.first, name);
            LOG_INFO(Core_ARM, "NVN probe: exported {} (+{:#x})", name, value.first);
            if (name == "nvnBootstrapLoader" || name == "nvnDeviceGetProcAddress" ||
                name == "nvnBootstrapLoaderInternal") {
                LOG_INFO(Core_ARM, "NVN probe: {} at {:#x} in {} (+{:#x}, {} bytes)", name, base + value.first,
                         module_name, value.first, value.second);
            }
        }
        LOG_INFO(Core_ARM, "NVN probe: module {} at {:#x}: {} symbols, {} NVN functions defined, {} NVN imported",
                 module_name, base, symbols.size(), defined, imported);
        if (defined != 0) {
            ScanNvnProcTable(probe, process, memory, base);
        }
    }
    LOG_INFO(Core_ARM, "NVN probe: watching {} NVN entry points", probe.entries.size());
}

void NvnProbeCode(Kernel::KProcess* process, Core::Memory::Memory& memory, u64 vaddr) {
    auto& probe = GetNvnProbe();
    std::scoped_lock lock{probe.mutex};
    // (the first code fetch can come before every module is mapped: rebuild while nothing was found)
    static u64 fetches = 0;
    if (!probe.built || (probe.entries.empty() && (++fetches & 0xFFFF) == 0)) {
        probe.entries.clear();
        BuildNvnProbe(probe, process, memory);
    }
    const auto it = probe.entries.find(vaddr);
    if (it == probe.entries.end() || probe.seen.contains(vaddr)) {
        return;
    }
    probe.seen.emplace(vaddr, true);
    ++probe.first_run;
    LOG_INFO(Core_ARM, "NVN probe: first run #{}: {} ({:#x})", probe.first_run, it->second, vaddr);
}

} // Anonymous namespace

std::optional<u32> DynarmicCallbacks64::MemoryReadCode(u64 vaddr) {
    if (NvnProbeOn()) {
        NvnProbeCode(m_process, m_memory, vaddr);
    }
    if (!m_memory.IsValidVirtualAddressRange(vaddr, sizeof(u32)))
        return std::nullopt;
    auto const aligned_vaddr = vaddr & ~Core::Memory::YUZU_PAGEMASK;
    if (last_code_addr != aligned_vaddr) {
        m_memory.ReadBlock(aligned_vaddr, &cached_code_page, sizeof(cached_code_page));
        last_code_addr = aligned_vaddr;
    }
    return cached_code_page.inst[(vaddr & Core::Memory::YUZU_PAGEMASK) / sizeof(u32)];
}

void DynarmicCallbacks64::MemoryWrite(Dynarmic::A64::VAddr vaddr, u64 value, std::size_t size) {
    if (CheckMemoryAccess(vaddr, size, Kernel::DebugWatchpointType::Write)) {
        switch (size) {
        case sizeof(u64): return m_memory.Write64(vaddr, u64(value));
        case sizeof(u32): return m_memory.Write32(vaddr, u32(value));
        case sizeof(u16): return m_memory.Write16(vaddr, u16(value));
        case sizeof(u8): return m_memory.Write8(vaddr, u8(value));
        default: UNREACHABLE();
        }
    }
}
void DynarmicCallbacks64::MemoryWrite128(u64 vaddr, Dynarmic::A64::Vector value) {
    if (CheckMemoryAccess(vaddr, 16, Kernel::DebugWatchpointType::Write)) {
        m_memory.Write64(vaddr, value[0]);
        m_memory.Write64(vaddr + 8, value[1]);
    }
}

bool DynarmicCallbacks64::MemoryWriteExclusive8(u64 vaddr, std::uint8_t value, std::uint8_t expected) {
    return CheckMemoryAccess(vaddr, 1, Kernel::DebugWatchpointType::Write) &&
            m_memory.WriteExclusive8(vaddr, value, expected);
}
bool DynarmicCallbacks64::MemoryWriteExclusive16(u64 vaddr, std::uint16_t value, std::uint16_t expected) {
    return CheckMemoryAccess(vaddr, 2, Kernel::DebugWatchpointType::Write) &&
            m_memory.WriteExclusive16(vaddr, value, expected);
}
bool DynarmicCallbacks64::MemoryWriteExclusive32(u64 vaddr, std::uint32_t value, std::uint32_t expected) {
    return CheckMemoryAccess(vaddr, 4, Kernel::DebugWatchpointType::Write) &&
            m_memory.WriteExclusive32(vaddr, value, expected);
}
bool DynarmicCallbacks64::MemoryWriteExclusive64(u64 vaddr, std::uint64_t value, std::uint64_t expected) {
    return CheckMemoryAccess(vaddr, 8, Kernel::DebugWatchpointType::Write) &&
            m_memory.WriteExclusive64(vaddr, value, expected);
}
bool DynarmicCallbacks64::MemoryWriteExclusive128(u64 vaddr, Dynarmic::A64::Vector value, Dynarmic::A64::Vector expected) {
    return CheckMemoryAccess(vaddr, 16, Kernel::DebugWatchpointType::Write) &&
            m_memory.WriteExclusive128(vaddr, value, expected);
}

void DynarmicCallbacks64::InstructionCacheOperationRaised(Dynarmic::A64::InstructionCacheOperation op, u64 value) {
    last_code_addr = u64(-1); //invalidate cached page
    switch (op) {
    case Dynarmic::A64::InstructionCacheOperation::InvalidateByVAToPoU: {
        static constexpr u64 ICACHE_LINE_SIZE = 64;
        const u64 cache_line_start = value & ~(ICACHE_LINE_SIZE - 1);
        m_parent.InvalidateCacheRange(cache_line_start, ICACHE_LINE_SIZE);
        break;
    }
    case Dynarmic::A64::InstructionCacheOperation::InvalidateAllToPoU:
        m_parent.ClearInstructionCache();
        break;
    case Dynarmic::A64::InstructionCacheOperation::InvalidateAllToPoUInnerSharable:
    default:
        LOG_DEBUG(Core_ARM, "Unprocesseed instruction cache operation: {}", op);
        break;
    }
    m_parent.m_jit->HaltExecution(Dynarmic::HaltReason::CacheInvalidation);
}

void DynarmicCallbacks64::ExceptionRaised(u64 pc, Dynarmic::A64::Exception exception) {
    switch (exception) {
    case Dynarmic::A64::Exception::WaitForInterrupt:
    case Dynarmic::A64::Exception::WaitForEvent:
    case Dynarmic::A64::Exception::SendEvent:
    case Dynarmic::A64::Exception::SendEventLocal:
    case Dynarmic::A64::Exception::Yield:
        LOG_TRACE(Core_ARM, "ExceptionRaised(exception = {}, pc = {:08X}, code = {:08X}, cached = {:08X})", std::size_t(exception), pc, m_memory.Read32(pc), MemoryReadCode(pc).value_or(0));
        return;
    case Dynarmic::A64::Exception::NoExecuteFault:
        LOG_CRITICAL(Core_ARM, "Cannot execute instruction at unmapped address {:#016x}", pc);
        ReturnException(pc, PrefetchAbort);
        return;
    default:
        if (m_debugger_enabled) {
            ReturnException(pc, InstructionBreakpoint);
        } else {
            m_parent.LogBacktrace(m_process);
            LOG_CRITICAL(Core_ARM, "ExceptionRaised(exception = {}, pc = {:08X}, code = {:08X})", static_cast<std::size_t>(exception), pc, m_memory.Read32(pc));
        }
    }
}

void DynarmicCallbacks64::CallSVC(u32 svc) {
    m_parent.m_svc = svc;
    m_parent.m_jit->HaltExecution(SupervisorCall);
}

void DynarmicCallbacks64::AddTicks(u64 ticks) {
    ASSERT(!m_parent.m_uses_wall_clock && "Dynarmic ticking disabled");
    // Divide the number of ticks by the amount of CPU cores. TODO(Subv): This yields only a
    // rough approximation of the amount of executed ticks in the system, it may be thrown off
    // if not all cores are doing a similar amount of work. Instead of doing this, we should
    // device a way so that timing is consistent across all cores without increasing the ticks 4
    // times.
    // Stays at the Switch 1 core count so guest time does not change with the extra emulated cores.
    u64 amortized_ticks = ticks / Core::Hardware::NUM_LEGACY_CPU_CORES;
    // Always execute at least one tick.
    amortized_ticks = std::max<u64>(amortized_ticks, 1);
    m_parent.m_system.CoreTiming().AddTicks(amortized_ticks);
}

u64 DynarmicCallbacks64::GetTicksRemaining() {
    ASSERT(!m_parent.m_uses_wall_clock && "Dynarmic ticking disabled");
    return std::max<s64>(m_parent.m_system.CoreTiming().downcount, 0);
}

u64 DynarmicCallbacks64::GetCNTPCT() {
    return Timing::GetGuestCounterTicks(m_parent.m_system.CoreTiming().GetClockTicks(),
                                        m_process && m_process->UsesPAuthReturns());
}

bool DynarmicCallbacks64::CheckMemoryAccess(u64 addr, u64 size, Kernel::DebugWatchpointType type) {
    if (!m_check_memory_access) {
        return true;
    }

    if (!m_memory.IsValidVirtualAddressRange(addr, size)) {
        LOG_CRITICAL(Core_ARM, "Stopping execution due to unmapped memory access at {:#x}",
                        addr);
        m_parent.m_jit->HaltExecution(PrefetchAbort);
        return false;
    }

    if (!m_debugger_enabled) {
        return true;
    }

    const auto match{m_parent.MatchingWatchpoint(addr, size, type)};
    if (match) {
        m_parent.m_halted_watchpoint = match;
        m_parent.m_jit->HaltExecution(DataAbort);
        return false;
    }

    return true;
}

void DynarmicCallbacks64::ReturnException(u64 pc, Dynarmic::HaltReason hr) {
    m_parent.GetContext(m_parent.m_breakpoint_context);
    m_parent.m_breakpoint_context.pc = pc;
    m_parent.m_jit->HaltExecution(hr);
}

void ArmDynarmic64::MakeJit(Common::PageTable* page_table, std::size_t address_space_bits) {
    Dynarmic::A64::UserConfig config;

    // Callbacks
    config.callbacks = std::addressof(*m_cb);
    config.enable_pauth_returns = m_cb->m_process && m_cb->m_process->UsesPAuthReturns();
    // Translated blocks persist across runs (EDEN_JIT_CACHE=0 disables).
    config.block_cache = m_cb->m_process ? m_cb->m_process->GetJitBlockCache() : nullptr;
    config.block_cache_native = config.block_cache && JitBlockCache::KeepsHostCode();
    // EDEN_JIT_CACHE_VERIFY=1: compare cached x64 code with freshly generated code (testing).
    if (const char* verify = std::getenv("EDEN_JIT_CACHE_VERIFY"); verify && verify[0] == '1') {
        config.block_cache_verify_native = config.block_cache_native;
    }

    // Memory
    if (page_table) {
        // Dynarmic will not write to the page table, const_cast is safe here
        config.page_table = reinterpret_cast<void**>(
            const_cast<Common::PageTable::PageEntryData*>(page_table->entries.data()));
        config.page_table_address_space_bits = std::uint32_t(address_space_bits);
        config.page_table_pointer_mask = Common::PageTable::ATTRIBUTE_MASK;
        config.page_table_marked_bit = uint8_t(0);
        config.silently_mirror_page_table = false;
        config.absolute_offset_page_table = true;
        config.detect_misaligned_access_via_page_table = 16 | 32 | 64 | 128;
        config.only_detect_misalignment_via_page_table_on_page_boundary = true;

        config.fastmem_pointer = page_table->fastmem_arena ?
            std::optional<uintptr_t>{reinterpret_cast<uintptr_t>(page_table->fastmem_arena)} :
            std::nullopt;
        config.fastmem_address_space_bits = std::uint32_t(address_space_bits);
        config.silently_mirror_fastmem = false;

        config.fastmem_exclusive_access = config.fastmem_pointer != std::nullopt;
        config.recompile_on_exclusive_fastmem_failure = true;

        if (reinterpret_cast<u64>(m_system.DeviceMemory().buffer.BackingBasePointer() +
            Kernel::Board::Nintendo::Nx::KSystemControl::Init::GetIntendedMemorySize()) < (1ULL << 39)) {
            // Systems like FreeBSD allocate memory really low by default, and since we pack our page table entries,
            // we have to manually sign extend when our actual pointer is negative.
            config.page_table_sign_extension = std::uint8_t(Common::PageTable::SIGN_BIT);
        }
    }

    // Multi-process state
    config.processor_id = std::uint8_t(m_core_index);
    config.global_monitor = &m_exclusive_monitor.monitor;

    // System registers
    config.tpidrro_el0 = &m_cb->m_tpidrro_el0;
    config.tpidr_el0 = &m_cb->m_tpidr_el0;
    config.dczid_el0 = 4;
    config.ctr_el0 = 0x8444c004;
    config.cntfrq_el0 = Timing::GetGuestCounterFrequency(config.enable_pauth_returns);

    // Unpredictable instructions
    config.define_unpredictable_behaviour = true;

    // Timing
    config.wall_clock_cntpct = m_uses_wall_clock;
    config.enable_cycle_counting = !m_uses_wall_clock;

    // Code cache size
#if defined(ARCHITECTURE_arm64) || defined(__sun__) || defined(__NetBSD__) || defined(__DragonFly__) || defined(__OpenBSD__)
    config.code_cache_size = std::uint32_t(128_MiB);
#else
    config.code_cache_size = std::uint32_t(512_MiB);
#endif

    // Allow memory fault handling to work
    if (m_system.DebuggerEnabled()) {
        config.check_halt_on_memory_access = true;
    }

    // null_jit
    if (!page_table) {
        // Don't waste too much memory on null_jit
        config.code_cache_size = std::uint32_t(8_MiB);
    }

    switch (Settings::values.cpu_accuracy.GetValue()) {
    // Debug mode
    case Settings::CpuAccuracy::Debugging:
        if (!Settings::values.cpuopt_page_tables) {
            config.page_table = nullptr;
        }
        if (!Settings::values.cpuopt_block_linking) {
            config.optimizations &= ~Dynarmic::OptimizationFlag::BlockLinking;
        }
        if (!Settings::values.cpuopt_return_stack_buffer) {
            config.optimizations &= ~Dynarmic::OptimizationFlag::ReturnStackBuffer;
        }
        if (!Settings::values.cpuopt_fast_dispatcher) {
            config.optimizations &= ~Dynarmic::OptimizationFlag::FastDispatch;
        }
        if (!Settings::values.cpuopt_context_elimination) {
            config.optimizations &= ~Dynarmic::OptimizationFlag::GetSetElimination;
        }
        if (!Settings::values.cpuopt_const_prop) {
            config.optimizations &= ~Dynarmic::OptimizationFlag::ConstProp;
        }
        if (!Settings::values.cpuopt_misc_ir) {
            config.optimizations &= ~Dynarmic::OptimizationFlag::MiscIROpt;
        }
        if (!Settings::values.cpuopt_reduce_misalign_checks) {
            config.only_detect_misalignment_via_page_table_on_page_boundary = false;
        }
        if (!Settings::values.cpuopt_fastmem) {
            config.fastmem_pointer = std::nullopt;
            config.fastmem_exclusive_access = false;
        }
        if (!Settings::values.cpuopt_fastmem_exclusives) {
            config.fastmem_exclusive_access = false;
        }
        if (!Settings::values.cpuopt_recompile_exclusives) {
            config.recompile_on_exclusive_fastmem_failure = false;
        }
        if (!Settings::values.cpuopt_ignore_memory_aborts) {
            config.check_halt_on_memory_access = true;
        }
        break;
    // Unsafe optimizations
    case Settings::CpuAccuracy::Unsafe:
        config.unsafe_optimizations = true;
        if (!Settings::values.cpuopt_unsafe_host_mmu) {
            config.fastmem_pointer = std::nullopt;
            config.fastmem_exclusive_access = false;
        }
        if (Settings::values.cpuopt_unsafe_unfuse_fma) {
            config.optimizations |= Dynarmic::OptimizationFlag::Unsafe_UnfuseFMA;
        }
        if (Settings::values.cpuopt_unsafe_reduce_fp_error) {
            config.optimizations |= Dynarmic::OptimizationFlag::Unsafe_ReducedErrorFP;
        }
        if (Settings::values.cpuopt_unsafe_inaccurate_nan) {
            config.optimizations |= Dynarmic::OptimizationFlag::Unsafe_InaccurateNaN;
        }
        if (Settings::values.cpuopt_unsafe_fastmem_check) {
            config.fastmem_address_space_bits = 64;
        }
        if (Settings::values.cpuopt_unsafe_ignore_global_monitor) {
            config.optimizations |= Dynarmic::OptimizationFlag::Unsafe_IgnoreGlobalMonitor;
        }
        break;
    // Safe optimisations
    case Settings::CpuAccuracy::Auto:
        config.unsafe_optimizations = true;
        config.optimizations |= Dynarmic::OptimizationFlag::Unsafe_UnfuseFMA;
        config.fastmem_address_space_bits = 64;
        config.optimizations |= Dynarmic::OptimizationFlag::Unsafe_IgnoreGlobalMonitor;
        break;
    // Paranoia mode for debugging optimizations
    case Settings::CpuAccuracy::Paranoid:
        config.unsafe_optimizations = false;
        config.optimizations = Dynarmic::no_optimizations;
        break;
    case Settings::CpuAccuracy::Accurate:
    default:
        break;
    }
    if (!Settings::IsFastmemEnabled()) {
        config.fastmem_pointer = std::nullopt;
        config.fastmem_exclusive_access = false;
    }
    m_jit.emplace(config);
}

namespace {
// EDEN_JIT_CACHE_STATS=1: per-core JIT compile counters, every 30 s and on shutdown.
bool JitStatsEnabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("EDEN_JIT_CACHE_STATS");
        return env && env[0] == '1';
    }();
    return enabled;
}

void LogJitStats(const Dynarmic::A64::Jit& jit, std::size_t core) {
    const auto s = jit.GetBlockCacheStats();
    LOG_INFO(Core_ARM, "JIT core {}: {} blocks from cache ({} as x64 code), {} translated, {} stale | translate {} ms, cache load {} ms, emit {} ms"
             " | x64 code: {} stored, {} not relocatable, {} rejected, {} verified, {} mismatches",
             core, s.hits, s.native_hits, s.misses, s.stale, s.translate_ns / 1'000'000, s.load_ns / 1'000'000, s.emit_ns / 1'000'000,
             s.native_stored, s.native_not_relocatable, s.native_rejected, s.native_verified, s.native_verify_mismatches);
    if (s.native_hits != 0) {
        LOG_INFO(Core_ARM, "JIT core {}: per x64 block loaded: storage {} ns, parse {} ns, guest code hash {} ns, placing {} ns",
                 core, s.load_storage_ns / s.native_hits, s.load_parse_ns / s.native_hits, s.load_hash_ns / s.native_hits,
                 s.load_place_ns / s.native_hits);
    }
}
}  // namespace

HaltReason ArmDynarmic64::RunThread(Kernel::KThread* thread) {
    if (JitStatsEnabled()) {
        const auto now = std::chrono::steady_clock::now();
        if (now - m_last_stats_log >= std::chrono::seconds{30}) {
            m_last_stats_log = now;
            LogJitStats(*m_jit, m_core_index);
        }
    }
    m_jit->ClearExclusiveState();
    return TranslateHaltReason(m_jit->Run());
}

HaltReason ArmDynarmic64::StepThread(Kernel::KThread* thread) {
    m_jit->ClearExclusiveState();
    return TranslateHaltReason(m_jit->Step());
}

u32 ArmDynarmic64::GetSvcNumber() const {
    return m_svc;
}

void ArmDynarmic64::GetSvcArguments(std::span<uint64_t, 8> args) const {
    Dynarmic::A64::Jit const& j = *m_jit;
    for (size_t i = 0; i < 8; i++)
        args[i] = j.GetRegister(i);
}

void ArmDynarmic64::SetSvcArguments(std::span<const uint64_t, 8> args) {
    Dynarmic::A64::Jit& j = *m_jit;
    for (size_t i = 0; i < 8; i++)
        j.SetRegister(i, args[i]);
}

const Kernel::DebugWatchpoint* ArmDynarmic64::HaltedWatchpoint() const {
    return m_halted_watchpoint;
}

void ArmDynarmic64::RewindBreakpointInstruction() {
    this->SetContext(m_breakpoint_context);
}

ArmDynarmic64::ArmDynarmic64(System& system, bool uses_wall_clock, Kernel::KProcess* process, DynarmicExclusiveMonitor& exclusive_monitor, std::size_t core_index)
    : ArmInterface{uses_wall_clock}, m_system{system}, m_exclusive_monitor{exclusive_monitor}
    , m_cb(std::make_optional<DynarmicCallbacks64>(*this, process))
    , m_core_index{core_index}
{
    auto& page_table = process->GetPageTable().GetBasePageTable();
    auto& page_table_impl = page_table.GetImpl();
    MakeJit(&page_table_impl, page_table.GetAddressSpaceWidth());
}

ArmDynarmic64::~ArmDynarmic64() {
    if (m_jit && JitStatsEnabled()) {
        LogJitStats(*m_jit, m_core_index);
    }
}

std::optional<ArmInterface::JitCacheCounters> ArmDynarmic64::GetJitCacheCounters() const {
    if (!m_jit) {
        return std::nullopt;
    }
    // Read while the core runs: the counters are only displayed.
    const auto s = m_jit->GetBlockCacheStats();
    return JitCacheCounters{s.hits, s.native_hits, s.misses};
}

Dynarmic::A64::PrecompileResult ArmDynarmic64::PrecompileBlock(u64 descriptor, u64 end_pc, u64 code_hash) {
    if (!m_jit) {
        return Dynarmic::A64::PrecompileResult::NotCacheable;
    }
    return m_jit->Precompile(descriptor, end_pc, code_hash);
}

Dynarmic::A64::PrecompileResult ArmDynarmic64::PlaceCachedBlock(u64 descriptor, std::size_t keep_free) {
    if (!m_jit) {
        return Dynarmic::A64::PrecompileResult::NotCacheable;
    }
    return m_jit->PlaceCached(descriptor, keep_free);
}

std::size_t ArmDynarmic64::CodeSpaceRemaining() const {
    return m_jit ? m_jit->CodeSpaceRemaining() : 0;
}

std::optional<Dynarmic::A64::ScanResult> ArmDynarmic64::ScanBlock(u64 descriptor) {
    if (!m_jit) {
        return std::nullopt;
    }
    return m_jit->ScanBlock(descriptor);
}

std::optional<u64> ArmDynarmic64::GuestCodeHash(u64 begin, u64 end) const {
    if (!m_jit) {
        return std::nullopt;
    }
    return m_jit->GuestCodeHash(begin, end);
}

u64 ArmDynarmic64::CacheKey(u64 descriptor) const {
    return m_jit ? m_jit->CacheKey(descriptor) : 0;
}

void ArmDynarmic64::SetTpidrroEl0(u64 value) {
    m_cb->m_tpidrro_el0 = value;
}

void ArmDynarmic64::GetContext(Kernel::Svc::ThreadContext& ctx) const {
    Dynarmic::A64::Jit const& j = *m_jit;
    auto gpr = j.GetRegisters();
    auto fpr = j.GetVectors();
    // TODO: this is inconvenient
    for (size_t i = 0; i < 29; i++)
        ctx.r[i] = gpr[i];
    ctx.fp = gpr[29];
    ctx.lr = gpr[30];
    ctx.sp = j.GetSP();
    ctx.pc = j.GetPC();
    ctx.pstate = j.GetPstate();
    ctx.v = fpr;
    ctx.fpcr = j.GetFpcr();
    ctx.fpsr = j.GetFpsr();
    ctx.tpidr = m_cb->m_tpidr_el0;
}

void ArmDynarmic64::SetContext(const Kernel::Svc::ThreadContext& ctx) {
    Dynarmic::A64::Jit& j = *m_jit;
    // TODO: this is inconvenient
    std::array<u64, 31> gpr;
    for (size_t i = 0; i < 29; i++)
        gpr[i] = ctx.r[i];
    gpr[29] = ctx.fp;
    gpr[30] = ctx.lr;
    j.SetRegisters(gpr);
    j.SetSP(ctx.sp);
    j.SetPC(ctx.pc);
    j.SetPstate(ctx.pstate);
    j.SetVectors(ctx.v);
    j.SetFpcr(ctx.fpcr);
    j.SetFpsr(ctx.fpsr);
    m_cb->m_tpidr_el0 = ctx.tpidr;
}

void ArmDynarmic64::SignalInterrupt(Kernel::KThread* thread) {
    m_jit->HaltExecution(BreakLoop);
}

void ArmDynarmic64::ClearInstructionCache() {
    m_cb->last_code_addr = u64(-1);
    m_jit->ClearCache();
}

void ArmDynarmic64::InvalidateCacheRange(u64 addr, std::size_t size) {
    // Refetch the source page too, including instructions restored by the debugger.
    Common::EmuCounters::jit_invalidations.Add();
    m_cb->last_code_addr = u64(-1);
    m_jit->InvalidateCacheRange(addr, size);
}

} // namespace Core
