// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include <cstring>
#include <memory>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <span>
#include <string>
#include <mutex>

#include <boost/icl/interval_set.hpp>
#include "common/assert.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/common/llvm_disassemble.h"
#include <bit>

#include "dynarmic/backend/x64/a64_emit_x64.h"
#include "dynarmic/backend/x64/a64_jitstate.h"
#include "dynarmic/backend/x64/block_of_code.h"
#include "dynarmic/backend/x64/devirtualize.h"
#include "dynarmic/backend/x64/exclusive_monitor_friend.h"
#include "dynarmic/backend/x64/native_cache.h"
#include "dynarmic/backend/x64/jitstate_info.h"
#include "dynarmic/common/atomic.h"
#include "dynarmic/frontend/A64/translate/a64_translate.h"
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/block_serialization.h"
#include "dynarmic/ir/opt_passes.h"

namespace Dynarmic::A64 {

using namespace Backend::X64;

static RunCodeCallbacks GenRunCodeCallbacks(A64::UserCallbacks* cb, CodePtr (*LookupBlock)(void* lookup_block_arg), void* arg, const A64::UserConfig& conf) {
    return RunCodeCallbacks{
        ArgCallback(LookupBlock, reinterpret_cast<u64>(arg)),
        ArgCallback(Devirtualize<&A64::UserCallbacks::AddTicks>(cb)),
        ArgCallback(Devirtualize<&A64::UserCallbacks::GetTicksRemaining>(cb)),
        conf.enable_cycle_counting,
    };
}

static std::function<void(BlockOfCode&)> GenRCP(const A64::UserConfig& conf) {
    return [conf](BlockOfCode& code) {
        if (conf.page_table) {
            code.mov(code.r14, std::bit_cast<u64>(conf.page_table));
        }
        if (conf.fastmem_pointer) {
            code.mov(code.r13, *conf.fastmem_pointer);
        }
    };
}

static Optimization::PolyfillOptions GenPolyfillOptions(const BlockOfCode& code) {
    return Optimization::PolyfillOptions{
        .sha256 = !code.HasHostFeature(HostFeature::SHA),
        .vector_multiply_widen = true,
    };
}

struct Jit::Impl final {
public:
    Impl(Jit* jit, UserConfig conf)
        : conf(conf)
        , block_of_code(GenRunCodeCallbacks(conf.callbacks, &GetCurrentBlockThunk, this, conf), JitStateInfo{jit_state}, conf.code_cache_size, GenRCP(conf))
        , emitter(block_of_code, conf, jit)
        , polyfill_options(GenPolyfillOptions(block_of_code))
        , cache_fingerprint(ComputeCacheFingerprint())
        , native_fingerprint(ComputeNativeFingerprint())
    {
        ASSERT(conf.page_table_address_space_bits >= 12 && conf.page_table_address_space_bits <= 64);
    }

    ~Impl() = default;

    HaltReason Run() {
        ASSERT(!is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&jit_state.halt_reason)));
        is_executing = true;
        // TODO: Check code alignment
        const CodePtr current_code_ptr = [this] {
            // RSB optimization
            const u64 location = jit_state.GetUniqueHash();
            const u32 new_rsb_ptr = (jit_state.rsb_ptr - 1) & A64JitState::RSB_PTR_MASK;
            if (location == jit_state.rsb_location_descriptors[new_rsb_ptr]) {
                jit_state.rsb_ptr = new_rsb_ptr;
                return CodePtr(jit_state.rsb_codeptrs[new_rsb_ptr]);
            }
            // The entry below the top: the embedder returned to the caller of a frame whose own
            // return was never executed (a host-side handler of a guest call reached through an
            // SVC, which pushed its pc+4, returning straight to lr). Pop both. Any entry whose
            // location matches is a valid entry point, as for the top.
            const u32 below_rsb_ptr = (jit_state.rsb_ptr - 2) & A64JitState::RSB_PTR_MASK;
            if (location == jit_state.rsb_location_descriptors[below_rsb_ptr]) {
                jit_state.rsb_ptr = below_rsb_ptr;
                return CodePtr(jit_state.rsb_codeptrs[below_rsb_ptr]);
            }
            return CodePtr((uintptr_t(GetCurrentBlock()) + 15) & ~uintptr_t(15));
        }();
        const HaltReason hr = block_of_code.RunCode(&jit_state, current_code_ptr);
        PerformRequestedCacheInvalidation(hr);
        is_executing = false;
        return hr;
    }

    HaltReason Step() {
        ASSERT(!is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&jit_state.halt_reason)));
        is_executing = true;
        const HaltReason hr = block_of_code.StepCode(&jit_state, GetCurrentSingleStep());
        PerformRequestedCacheInvalidation(hr);
        is_executing = false;
        return hr;
    }

    void ClearCache() {
        std::unique_lock lock{invalidation_mutex};
        invalidate_entire_cache = true;
        HaltExecution(HaltReason::CacheInvalidation);
    }

    void InvalidateCacheRange(u64 start_address, size_t length) {
        std::unique_lock lock{invalidation_mutex};
        const auto end_address = static_cast<u64>(start_address + length - 1);
        const auto range = boost::icl::discrete_interval<u64>::closed(start_address, end_address);
        invalid_cache_ranges.add(range);
        HaltExecution(HaltReason::CacheInvalidation);
    }

    void Reset() {
        ASSERT(!is_executing);
        jit_state = {};
    }

    void HaltExecution(HaltReason hr) {
        Atomic::Or(&jit_state.halt_reason, static_cast<u32>(hr));
    }

    void ClearHalt(HaltReason hr) {
        Atomic::And(&jit_state.halt_reason, ~static_cast<u32>(hr));
    }

    u64 GetSP() const {
        return jit_state.sp;
    }

    void SetSP(u64 value) {
        jit_state.sp = value;
    }

    u64 GetPC() const {
        return jit_state.pc;
    }

    void SetPC(u64 value) {
        jit_state.pc = value;
    }

    u64 GetRegister(size_t index) const {
        if (index == 31)
            return GetSP();
        return jit_state.reg.at(index);
    }

    void SetRegister(size_t index, u64 value) {
        if (index == 31)
            return SetSP(value);
        jit_state.reg.at(index) = value;
    }

    std::array<u64, 31> GetRegisters() const {
        return jit_state.reg;
    }

    void SetRegisters(const std::array<u64, 31>& value) {
        jit_state.reg = value;
    }

    Vector GetVector(size_t index) const {
        return {jit_state.vec.at(index * 2), jit_state.vec.at(index * 2 + 1)};
    }

    void SetVector(size_t index, Vector value) {
        jit_state.vec.at(index * 2) = value[0];
        jit_state.vec.at(index * 2 + 1) = value[1];
    }

    std::array<Vector, 32> GetVectors() const {
        std::array<Vector, 32> ret;
        static_assert(sizeof(ret) == sizeof(jit_state.vec));
        std::memcpy(ret.data(), jit_state.vec.data(), sizeof(jit_state.vec));
        return ret;
    }

    void SetVectors(const std::array<Vector, 32>& value) {
        static_assert(sizeof(value) == sizeof(jit_state.vec));
        std::memcpy(jit_state.vec.data(), value.data(), sizeof(jit_state.vec));
    }

    u32 GetFpcr() const {
        return jit_state.GetFpcr();
    }

    void SetFpcr(u32 value) {
        jit_state.SetFpcr(value);
    }

    u32 GetFpsr() const {
        return jit_state.GetFpsr();
    }

    void SetFpsr(u32 value) {
        jit_state.SetFpsr(value);
    }

    u32 GetPstate() const {
        return jit_state.GetPstate();
    }

    void SetPstate(u32 value) {
        jit_state.SetPstate(value);
    }

    void ClearExclusiveState() {
        jit_state.exclusive_state = 0;
    }

    bool IsExecuting() const {
        return is_executing;
    }

    std::string Disassemble() const {
        const size_t size = reinterpret_cast<const char*>(block_of_code.getCurr()) - reinterpret_cast<const char*>(block_of_code.GetCodeBegin());
        auto const* p = reinterpret_cast<const char*>(block_of_code.GetCodeBegin());
        return Common::DisassembleX64(p, p + size);
    }

private:
    static CodePtr GetCurrentBlockThunk(void* thisptr) {
        Jit::Impl* this_ = static_cast<Jit::Impl*>(thisptr);
        return this_->GetCurrentBlock();
    }

    IR::LocationDescriptor GetCurrentLocation() const {
        return IR::LocationDescriptor{jit_state.GetUniqueHash()};
    }

    CodePtr GetCurrentBlock() {
        return GetBlock(GetCurrentLocation());
    }

    CodePtr GetCurrentSingleStep() {
        return GetBlock(A64::LocationDescriptor{GetCurrentLocation()}.SetSingleStepping(true));
    }

public:
    BlockCacheStats GetBlockCacheStats() const {
        return cache_stats;
    }

private:

    CodePtr GetBlock(IR::LocationDescriptor descriptor) {
        if (auto block = emitter.GetBasicBlock(descriptor))
            return block->entrypoint;

        constexpr size_t MINIMUM_REMAINING_CODESIZE = 1 * 1024 * 1024;
        if (block_of_code.SpaceRemaining() < MINIMUM_REMAINING_CODESIZE) {
            // Immediately evacuate cache
            invalidate_entire_cache = true;
            PerformRequestedCacheInvalidation(HaltReason::CacheInvalidation);
        }
        block_of_code.EnsureMemoryCommitted(MINIMUM_REMAINING_CODESIZE);

        // JIT Compile
        const auto get_code = [this](u64 vaddr) { return conf.callbacks->MemoryReadCode(vaddr); };
        // LocationDescriptor ctor() does important ops (like tflags) do not skip
        auto const arch_descriptor = A64::LocationDescriptor{descriptor};
        using Clock = std::chrono::steady_clock;
        const auto elapsed_ns = [](Clock::time_point since) {
            return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - since).count());
        };
        const auto compile_start = Clock::now();
        const auto note_compiled = [&](BlockCacheStorage::CompileKind kind) {
            if (conf.block_cache) {
                conf.block_cache->OnBlockCompiled(kind, elapsed_ns(compile_start));
            }
        };
        const u64 cache_key = IR::HashCombine(cache_fingerprint, descriptor.Value());
        const bool use_native = conf.block_cache && conf.block_cache_native;

        if (conf.block_cache) {
            const auto start = Clock::now();
            auto record = conf.block_cache->Load(cache_key);
            const auto after_storage = Clock::now();
            if (record) {
                const auto parsed = ParseRecord(*record);
                const auto after_parse = Clock::now();
                if (!parsed) {
                    ++cache_stats.stale;
                } else {
                    // The record's own location must be the requested one: keys are hashes, and two FPCR variants
                    // of one PC have the same guest code, so the guest hash alone cannot tell them apart.
                    const bool native_usable = use_native && parsed->native && parsed->native_fingerprint == native_fingerprint
                                            && IR::PeekBlockLocation(parsed->ir) == std::optional<u64>{descriptor.Value()};
                    // Host code: no translation and no code generation.
                    if (native_usable && !conf.block_cache_verify_native) {
                        const bool code_matches = HashGuestCode(arch_descriptor.PC(), parsed->end_pc) == std::optional<u64>{parsed->code_hash};
                        const auto after_hash = Clock::now();
                        if (!code_matches) {
                            ++cache_stats.stale;
                        } else if (auto placed = emitter.LoadNative(descriptor, parsed->end_pc, *parsed->native)) {
                            const auto ns = [](auto a, auto b) { return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count()); };
                            cache_stats.load_storage_ns += ns(start, after_storage);
                            cache_stats.load_parse_ns += ns(after_storage, after_parse);
                            cache_stats.load_hash_ns += ns(after_parse, after_hash);
                            cache_stats.load_place_ns += elapsed_ns(after_hash);
                            cache_stats.load_ns += elapsed_ns(start);
                            ++cache_stats.hits;
                            ++cache_stats.native_hits;
                            conf.block_cache->OnBlockPlaced(descriptor.Value(), parsed->end_pc, parsed->code_hash);
                            note_compiled(BlockCacheStorage::CompileKind::Native);
                            return placed->entrypoint;
                        } else {
                            ++cache_stats.native_rejected;
                        }
                    }
                    // IR: no translation.
                    if (LoadCachedIR(*parsed, arch_descriptor)) {
                        cache_stats.load_ns += elapsed_ns(start);
                        ++cache_stats.hits;
                        const u64 ir_end_pc = A64::LocationDescriptor{ir_block.EndLocation()}.PC();
                        const CodePtr entry = EmitTimed(elapsed_ns);
                        conf.block_cache->OnBlockPlaced(descriptor.Value(), ir_end_pc, parsed->code_hash);
                        if (use_native) {
                            if (native_usable && conf.block_cache_verify_native) {
                                VerifyNative(*parsed->native, descriptor, entry);
                            }
                            if (!native_usable) {
                                StoreRecord(cache_key, parsed->code_hash, parsed->ir, parsed->end_pc);  // add host code
                            }
                        }
                        note_compiled(BlockCacheStorage::CompileKind::Ir);
                        return entry;
                    }
                    ++cache_stats.stale;
                }
            }
        }

        const auto start = Clock::now();
        ir_block.Reset(arch_descriptor);
        A64::Translate(ir_block, arch_descriptor, get_code, TranslationOptions());
        Optimization::Optimize(ir_block, conf, polyfill_options);
        cache_stats.translate_ns += elapsed_ns(start);
        ++cache_stats.misses;

        const CodePtr entry = EmitTimed(elapsed_ns);
        if (conf.block_cache && IsCacheableIR(ir_block)) {
            const u64 begin = A64::LocationDescriptor{ir_block.Location()}.PC();
            const u64 end = A64::LocationDescriptor{ir_block.EndLocation()}.PC();
            if (const auto code_hash = HashGuestCode(begin, end)) {
                StoreRecord(cache_key, *code_hash, IR::SerializeBlock(ir_block), end);
                conf.block_cache->OnBlockPlaced(descriptor.Value(), end, *code_hash);
            }
        }
        note_compiled(BlockCacheStorage::CompileKind::Translated);
        return entry;
    }

public:
    PrecompileResult Precompile(u64 value, u64 end_pc, u64 code_hash) {
        ASSERT(!is_executing);
        // A ClearCache or InvalidateCacheRange requested since the last Run or Precompile takes effect first, as
        // at the start of Run (a builder clears transient blocks this way between batches).
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&jit_state.halt_reason)));
        const IR::LocationDescriptor descriptor{value};
        if (emitter.GetBasicBlock(descriptor)) {
            return PrecompileResult::Present;
        }
        if (!conf.block_cache) {
            return PrecompileResult::NotCacheable;
        }
        const u64 cache_key = IR::HashCombine(cache_fingerprint, value);
        if (conf.block_cache->Contains(cache_key)) {
            return PrecompileResult::Present;
        }
        const auto arch_descriptor = A64::LocationDescriptor{descriptor};
        // The guest check comes first: a block whose code changed is not translated at all.
        const auto hash = HashGuestCode(arch_descriptor.PC(), end_pc);
        if (!hash) {
            return PrecompileResult::Unreadable;
        }
        if (*hash != code_hash) {
            return PrecompileResult::GuestChanged;
        }
        constexpr size_t MINIMUM_REMAINING_CODESIZE = 1 * 1024 * 1024;
        if (block_of_code.SpaceRemaining() < MINIMUM_REMAINING_CODESIZE) {
            invalidate_entire_cache = true;
            PerformRequestedCacheInvalidation(HaltReason::CacheInvalidation);
        }
        block_of_code.EnsureMemoryCommitted(MINIMUM_REMAINING_CODESIZE);
        const auto get_code = [this](u64 vaddr) { return conf.callbacks->MemoryReadCode(vaddr); };
        ir_block.Reset(arch_descriptor);
        A64::Translate(ir_block, arch_descriptor, get_code, TranslationOptions());
        Optimization::Optimize(ir_block, conf, polyfill_options);
        if (!IsCacheableIR(ir_block)) {
            return PrecompileResult::NotCacheable;
        }
        // The translator decides where the block ends; if this build ends it elsewhere than the build that
        // listed it, the record carries the hash of the range it really covers.
        const u64 end = A64::LocationDescriptor{ir_block.EndLocation()}.PC();
        const auto stored_hash = end == end_pc ? hash : HashGuestCode(arch_descriptor.PC(), end);
        if (!stored_hash) {
            return PrecompileResult::Unreadable;
        }
        emitter.Emit(ir_block);
        StoreRecord(cache_key, *stored_hash, IR::SerializeBlock(ir_block), end);
        return PrecompileResult::Built;
    }

    PrecompileResult PlaceCached(u64 value, size_t keep_free) {
        ASSERT(!is_executing);
        PerformRequestedCacheInvalidation(static_cast<HaltReason>(Atomic::Load(&jit_state.halt_reason)));
        const IR::LocationDescriptor descriptor{value};
        if (emitter.GetBasicBlock(descriptor)) {
            return PrecompileResult::Present;
        }
        if (!conf.block_cache || !conf.block_cache_native || conf.block_cache_verify_native) {
            return PrecompileResult::NotCacheable;
        }
        constexpr size_t MINIMUM_REMAINING_CODESIZE = 1 * 1024 * 1024;
        if (block_of_code.SpaceRemaining() < keep_free + MINIMUM_REMAINING_CODESIZE) {
            return PrecompileResult::NoSpace;
        }
        block_of_code.EnsureMemoryCommitted(MINIMUM_REMAINING_CODESIZE);
        const auto record = conf.block_cache->Load(IR::HashCombine(cache_fingerprint, value));
        if (!record) {
            return PrecompileResult::NotCacheable;
        }
        // As on a hit in GetBlock: the record's own location must be the requested one (FPCR variants of one PC
        // share guest code), and the guest code must still hash to the recorded value.
        const auto parsed = ParseRecord(*record);
        if (!parsed || !parsed->native || parsed->native_fingerprint != native_fingerprint
            || IR::PeekBlockLocation(parsed->ir) != std::optional<u64>{value}) {
            return PrecompileResult::NotCacheable;
        }
        const auto arch_descriptor = A64::LocationDescriptor{descriptor};
        const auto hash = HashGuestCode(arch_descriptor.PC(), parsed->end_pc);
        if (!hash) {
            return PrecompileResult::Unreadable;
        }
        if (*hash != parsed->code_hash) {
            return PrecompileResult::GuestChanged;
        }
        if (!emitter.LoadNative(descriptor, parsed->end_pc, *parsed->native)) {
            return PrecompileResult::NotCacheable;
        }
        return PrecompileResult::Built;
    }

    size_t CodeSpaceRemaining() const {
        return block_of_code.SpaceRemaining();
    }

    std::optional<ScanResult> ScanBlock(u64 value) {
        ASSERT(!is_executing);
        const IR::LocationDescriptor descriptor{value};
        const auto arch_descriptor = A64::LocationDescriptor{descriptor};
        if (!conf.callbacks->MemoryReadCode(arch_descriptor.PC())) {
            return std::nullopt;
        }
        const auto get_code = [this](u64 vaddr) { return conf.callbacks->MemoryReadCode(vaddr); };
        // Own block: ir_block may hold the block Run is about to emit.
        scan_block.Reset(arch_descriptor);
        A64::Translate(scan_block, arch_descriptor, get_code, TranslationOptions());
        Optimization::Optimize(scan_block, conf, polyfill_options);
        ScanResult r;
        r.end_pc = A64::LocationDescriptor{scan_block.EndLocation()}.PC();
        if (const auto hash = HashGuestCode(arch_descriptor.PC(), r.end_pc)) {
            r.code_hash = *hash;
            r.hashed = true;
        }
        r.cacheable = IsCacheableIR(scan_block);
        const auto add = [&r](u64 pc) {
            if (r.successor_count < std::size(r.successors)) {
                r.successors[r.successor_count++] = pc;
            }
        };
        const auto leaf = [&](const IR::Term::LeafTerminal& l) {
            if (const auto* x = std::get_if<IR::Term::LinkBlock>(&l)) {
                add(A64::LocationDescriptor{x->next}.PC());
            } else if (const auto* y = std::get_if<IR::Term::LinkBlockFast>(&l)) {
                add(A64::LocationDescriptor{y->next}.PC());
            }
        };
        const auto& t = scan_block.GetTerminal();
        if (const auto* l = std::get_if<IR::Term::LeafTerminal>(&t)) {
            leaf(*l);
        } else if (const auto* i = std::get_if<IR::Term::If>(&t)) {
            leaf(i->then_);
            leaf(i->else_);
        } else if (const auto* c = std::get_if<IR::Term::CheckBit>(&t)) {
            leaf(c->then_);
            leaf(c->else_);
        } else if (const auto* h = std::get_if<IR::Term::CheckHalt>(&t)) {
            leaf(h->else_);
        }
        for (const auto& inst : scan_block.instructions) {
            const auto op = inst.GetOpcode();
            if (op == IR::Opcode::PushRSB && inst.GetArg(0).IsImmediate()) {
                add(A64::LocationDescriptor{IR::LocationDescriptor{inst.GetArg(0).GetU64()}}.PC());
            } else if (op == IR::Opcode::A64SetPC && inst.GetArg(0).IsImmediate()) {
                add(inst.GetArg(0).GetU64());
            }
        }
        return r;
    }

    std::optional<u64> GuestCodeHash(u64 begin, u64 end) const {
        return HashGuestCode(begin, end);
    }

    u64 CacheKey(u64 descriptor) const {
        return IR::HashCombine(cache_fingerprint, descriptor);
    }

private:

    A64::TranslationOptions TranslationOptions() const {
        return {.define_unpredictable_behaviour = conf.define_unpredictable_behaviour,
                .wall_clock_cntpct = conf.wall_clock_cntpct,
                .enable_pauth_returns = conf.enable_pauth_returns,
                .address_space_bits = conf.page_table_address_space_bits};
    }

    template<typename ElapsedFn>
    CodePtr EmitTimed(ElapsedFn elapsed_ns) {
        const auto start = std::chrono::steady_clock::now();
        const CodePtr entry = emitter.Emit(ir_block).entrypoint;
        cache_stats.emit_ns += elapsed_ns(start);
        return entry;
    }

    /// Blocks that embed host addresses in their IR cannot outlive the process.
    static bool IsCacheableIR(const IR::Block& block) {
        for (const auto& inst : block.instructions) {
            if (inst.GetOpcode() == IR::Opcode::CallHostFunction) {
                return false;
            }
        }
        return true;
    }

    /// Hash of the guest instructions in [begin, end), or nothing if any of them cannot be read.
    std::optional<u64> HashGuestCode(u64 begin, u64 end) const {
        if (end <= begin || end - begin > 64 * 1024) {
            return std::nullopt;
        }
        u64 hash = IR::HashCombine(begin, end);
        for (u64 pc = begin; pc < end; pc += 4) {
            const auto word = conf.callbacks->MemoryReadCode(pc);
            if (!word) {
                return std::nullopt;
            }
            hash = IR::HashCombine(hash, *word);
        }
        return hash;
    }

    // Record: code hash (8 bytes LE), IR length (4 bytes LE), IR (ir/block_serialization), and when host code is
    // kept: native fingerprint (8 bytes LE), end PC, then the host code and its relocations, patch sites and
    // fastmem entries (LEB128 varints).
    struct ParsedRecord {
        u64 code_hash = 0;
        std::span<const u8> ir;
        u64 end_pc = 0;
        u64 native_fingerprint = 0;
        std::optional<NativeBlock> native;
    };

    static std::optional<ParsedRecord> ParseRecord(const std::vector<u8>& record) {
        size_t pos = 0;
        const auto fixed = [&](size_t bytes) -> std::optional<u64> {
            if (pos + bytes > record.size()) {
                return std::nullopt;
            }
            u64 v = 0;
            for (size_t i = 0; i < bytes; ++i) {
                v |= u64{record[pos + i]} << (8 * i);
            }
            pos += bytes;
            return v;
        };
        const auto var = [&]() -> std::optional<u64> {
            u64 v = 0;
            for (unsigned shift = 0; shift < 64 && pos < record.size(); shift += 7) {
                const u8 b = record[pos++];
                v |= u64(b & 0x7F) << shift;
                if ((b & 0x80) == 0) {
                    return v;
                }
            }
            return std::nullopt;
        };
        ParsedRecord r;
        const auto hash = fixed(8);
        const auto ir_len = fixed(4);
        if (!hash || !ir_len || pos + *ir_len > record.size()) {
            return std::nullopt;
        }
        r.code_hash = *hash;
        r.ir = std::span<const u8>{record}.subspan(pos, static_cast<size_t>(*ir_len));
        pos += static_cast<size_t>(*ir_len);
        if (pos == record.size()) {
            return r;
        }
        const auto fp = fixed(8);
        const auto end_pc = var();
        const auto size = var();
        if (!fp || !end_pc || !size || pos + *size > record.size()) {
            return std::nullopt;
        }
        r.native_fingerprint = *fp;
        r.end_pc = *end_pc;
        NativeBlock nb;
        nb.code.assign(record.begin() + static_cast<ptrdiff_t>(pos), record.begin() + static_cast<ptrdiff_t>(pos + *size));
        pos += static_cast<size_t>(*size);
        const auto nrel = var();
        if (!nrel || *nrel > nb.code.size()) {
            return std::nullopt;
        }
        for (u64 i = 0; i < *nrel; ++i) {
            const auto off = var();
            const auto kind = fixed(1);
            const auto value = var();
            if (!off || !kind || !value) {
                return std::nullopt;
            }
            nb.relocs.push_back({static_cast<u32>(*off), static_cast<NativeRelocKind>(*kind), *value});
        }
        const auto npatch = var();
        if (!npatch || *npatch > nb.code.size()) {
            return std::nullopt;
        }
        for (u64 i = 0; i < *npatch; ++i) {
            const auto off = var();
            const auto kind = var();
            const auto target = var();
            if (!off || !kind || !target) {
                return std::nullopt;
            }
            nb.patch_sites.push_back({static_cast<u32>(*off), static_cast<u32>(*kind), *target});
        }
        const auto nfast = var();
        if (!nfast || *nfast > nb.code.size()) {
            return std::nullopt;
        }
        for (u64 i = 0; i < *nfast; ++i) {
            const auto rip_ = var();
            const auto resume = var();
            const auto cb = var();
            const auto name = var();
            const auto recompile = fixed(1);
            const auto location = var();
            if (!rip_ || !resume || !cb || !name || !recompile || !location) {
                return std::nullopt;
            }
            nb.fastmem.push_back({static_cast<u32>(*rip_), static_cast<u32>(*resume), static_cast<u32>(*cb), static_cast<u32>(*name), *recompile != 0, *location});
        }
        if (pos != record.size()) {
            return std::nullopt;
        }
        r.native = std::move(nb);
        return r;
    }

    /// Stores the IR of a block and, when kept and available, the host code just emitted for it.
    void StoreRecord(u64 key, u64 code_hash, std::span<const u8> ir, u64 end_pc) {
        std::vector<u8> out;
        const auto fixed = [&](u64 v, size_t bytes) {
            for (size_t i = 0; i < bytes; ++i) {
                out.push_back(static_cast<u8>(v >> (8 * i)));
            }
        };
        const auto var = [&](u64 v) {
            while (v >= 0x80) {
                out.push_back(static_cast<u8>(v | 0x80));
                v >>= 7;
            }
            out.push_back(static_cast<u8>(v));
        };
        fixed(code_hash, 8);
        fixed(ir.size(), 4);
        out.insert(out.end(), ir.begin(), ir.end());
        if (conf.block_cache_native) {
            if (const auto& nb = emitter.last_native) {
                fixed(native_fingerprint, 8);
                var(end_pc);
                var(nb->code.size());
                out.insert(out.end(), nb->code.begin(), nb->code.end());
                var(nb->relocs.size());
                for (const auto& rel : nb->relocs) {
                    var(rel.offset);
                    fixed(static_cast<u64>(rel.kind), 1);
                    var(rel.value);
                }
                var(nb->patch_sites.size());
                for (const auto& site : nb->patch_sites) {
                    var(site.offset);
                    var(site.kind);
                    var(site.target);
                }
                var(nb->fastmem.size());
                for (const auto& f : nb->fastmem) {
                    var(f.rip);
                    var(f.resume_rip);
                    var(f.callback);
                    var(f.inst_name);
                    fixed(f.recompile ? 1 : 0, 1);
                    var(f.location);
                }
                ++cache_stats.native_stored;
            } else {
                ++cache_stats.native_not_relocatable;
            }
        }
        conf.block_cache->Store(key, std::move(out));
    }

    /// Rebuilds ir_block from a parsed record if it was translated from the guest code now in memory.
    bool LoadCachedIR(const ParsedRecord& record, A64::LocationDescriptor descriptor) {
        ir_block.Reset(descriptor);
        if (!IR::DeserializeBlock(record.ir, ir_block)) {
            return false;
        }
        const u64 begin = A64::LocationDescriptor{ir_block.Location()}.PC();
        const u64 end = A64::LocationDescriptor{ir_block.EndLocation()}.PC();
        return HashGuestCode(begin, end) == std::optional<u64>{record.code_hash};
    }

    /// Verification mode: the cached host code, relocated to where the fresh code was placed, must be identical
    /// to the fresh code, apart from links to other blocks (rewritten on load).
    void VerifyNative(const NativeBlock& cached, IR::LocationDescriptor descriptor, CodePtr entry) {
        ++cache_stats.native_verified;
        const auto report = [&](const std::string& what) {
            ++cache_stats.native_verify_mismatches;
            std::fprintf(stderr, "dynarmic native cache verify: block %016llx: %s\n",
                         static_cast<unsigned long long>(A64::LocationDescriptor{descriptor}.PC()), what.c_str());
        };
        const auto& fresh = emitter.last_native;
        if (!fresh) {
            report(std::string("fresh code is not relocatable: ") + (emitter.last_native_failure ? emitter.last_native_failure : "?"));
            return;
        }
        const auto* at = static_cast<const u8*>(entry);
        std::vector<u8> a, b;
        if (!emitter.RelocateNative(cached, at, a) || !emitter.RelocateNative(*fresh, at, b)) {
            report("relocation failed");
            return;
        }
        if (a.size() != b.size()) {
            report("size " + std::to_string(a.size()) + " vs fresh " + std::to_string(b.size()));
            return;
        }
        const auto in_patch_site = [&](size_t offset) {
            for (const auto& site : fresh->patch_sites) {
                if (offset >= site.offset && offset < site.offset + PatchSiteSize(site.kind)) {
                    return true;
                }
            }
            return false;
        };
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i] != b[i] && !in_patch_site(i)) {
                report("byte at offset " + std::to_string(i) + " differs");
                return;
            }
        }
        const auto same_relocs = cached.relocs.size() == fresh->relocs.size()
                              && std::equal(cached.relocs.begin(), cached.relocs.end(), fresh->relocs.begin(), [](const NativeReloc& x, const NativeReloc& y) {
                                     return x.offset == y.offset && x.kind == y.kind && x.value == y.value;
                                 });
        const auto same_sites = cached.patch_sites.size() == fresh->patch_sites.size()
                             && std::equal(cached.patch_sites.begin(), cached.patch_sites.end(), fresh->patch_sites.begin(), [](const NativePatchSite& x, const NativePatchSite& y) {
                                    return x.offset == y.offset && x.kind == y.kind && x.target == y.target;
                                });
        const auto same_fastmem = cached.fastmem.size() == fresh->fastmem.size()
                               && std::equal(cached.fastmem.begin(), cached.fastmem.end(), fresh->fastmem.begin(), [](const NativeFastmemEntry& x, const NativeFastmemEntry& y) {
                                      return x.rip == y.rip && x.resume_rip == y.resume_rip && x.callback == y.callback && x.inst_name == y.inst_name
                                          && x.recompile == y.recompile && x.location == y.location;
                                  });
        if (!same_relocs) {
            report("relocations differ");
        } else if (!same_sites) {
            report("patch sites differ");
        } else if (!same_fastmem) {
            report("fastmem entries differ");
        }
    }

    /// Everything besides the guest code that shapes the optimized IR of a block.
    u64 ComputeCacheFingerprint() const {
        u64 h = IR::SchemaHash();
        h = IR::HashCombine(h, 2);  // record format
        const A64::TranslationOptions options = TranslationOptions();
        h = IR::HashCombine(h, u64(options.define_unpredictable_behaviour) | u64(options.wall_clock_cntpct) << 1
                                   | u64(options.hook_hint_instructions) << 2 | u64(options.enable_pauth_returns) << 3);
        h = IR::HashCombine(h, options.address_space_bits);
        h = IR::HashCombine(h, static_cast<u64>(conf.optimizations));
        h = IR::HashCombine(h, u64(conf.unsafe_optimizations) | u64(conf.check_halt_on_memory_access) << 1
                                   | u64(conf.hook_data_cache_operations) << 2);
        h = IR::HashCombine(h, conf.dczid_el0);
        h = IR::HashCombine(h, u64(polyfill_options.sha256) | u64(polyfill_options.vector_multiply_widen) << 1);
        return h;
    }

    /// Everything besides the IR that shapes a block's host code: host CPU features, the prelude, and every
    /// configuration value the x64 emitter reads.
    u64 ComputeNativeFingerprint() const {
        u64 h = IR::HashCombine(cache_fingerprint, 1);  // native format
        h = IR::HashCombine(h, block_of_code.HostFeatureBits());
        h = IR::HashCombine(h, block_of_code.PreludeSize());
        h = IR::HashCombine(h, NativeModuleIdentity());  // recorded function offsets are per executable
        const auto flag = [&](bool b, unsigned bit) { return u64(b) << bit; };
        h = IR::HashCombine(h, flag(conf.page_table != nullptr, 0) | flag(conf.fastmem_pointer.has_value(), 1)
                                   | flag(conf.fastmem_exclusive_access, 2) | flag(conf.silently_mirror_page_table, 3)
                                   | flag(conf.silently_mirror_fastmem, 4) | flag(conf.absolute_offset_page_table, 5)
                                   | flag(conf.only_detect_misalignment_via_page_table_on_page_boundary, 6)
                                   | flag(conf.recompile_on_fastmem_failure, 7) | flag(conf.recompile_on_exclusive_fastmem_failure, 8)
                                   | flag(conf.enable_cycle_counting, 9) | flag(conf.wall_clock_cntpct, 10)
                                   | flag(conf.very_verbose_debugging_output, 11) | flag(conf.tpidr_el0 != nullptr, 12)
                                   | flag(conf.tpidrro_el0 != nullptr, 13) | flag(conf.hook_isb, 14)
                                   | flag(conf.global_monitor != nullptr, 15) | flag(conf.enable_pauth_returns, 16)
                                   | flag(conf.define_unpredictable_behaviour, 17) | flag(conf.check_halt_on_memory_access, 18)
                                   | flag(conf.hook_data_cache_operations, 19));
        h = IR::HashCombine(h, conf.page_table_pointer_mask);
        h = IR::HashCombine(h, u64(conf.page_table_address_space_bits) | u64(conf.fastmem_address_space_bits) << 8
                                   | u64(conf.page_table_log2_stride) << 16
                                   | u64(conf.page_table_marked_bit ? 1 + *conf.page_table_marked_bit : 0) << 24);
        h = IR::HashCombine(h, conf.page_table_sign_extension ? 1 + u64(*conf.page_table_sign_extension) : 0);
        h = IR::HashCombine(h, conf.detect_misaligned_access_via_page_table);
        h = IR::HashCombine(h, u64(conf.ctr_el0) | u64(conf.dczid_el0) << 32);
        h = IR::HashCombine(h, conf.cntfrq_el0);
        h = IR::HashCombine(h, conf.global_monitor ? GetExclusiveMonitorProcessorCount(conf.global_monitor) : 0);
        h = IR::HashCombine(h, static_cast<u64>(conf.optimizations));
        h = IR::HashCombine(h, u64(conf.unsafe_optimizations));
        return h;
    }

    void PerformRequestedCacheInvalidation(HaltReason hr) {
        if (Has(hr, HaltReason::CacheInvalidation)) {
            std::unique_lock lock{invalidation_mutex};

            ClearHalt(HaltReason::CacheInvalidation);

            if (!invalidate_entire_cache && invalid_cache_ranges.empty()) {
                return;
            }

            jit_state.ResetRSB();
            if (invalidate_entire_cache) {
                block_of_code.ClearCache();
                emitter.ClearCache();
            } else {
                emitter.InvalidateCacheRanges(invalid_cache_ranges);
            }
            invalid_cache_ranges.clear();
            invalidate_entire_cache = false;
        }
    }

    IR::Block ir_block = {LocationDescriptor(0, FP::FPCR(0), false)};
    IR::Block scan_block = {LocationDescriptor(0, FP::FPCR(0), false)};  // Jit::ScanBlock
    const UserConfig conf;
    BlockCacheStats cache_stats;
    A64JitState jit_state;
    BlockOfCode block_of_code;
    A64EmitX64 emitter;
    Optimization::PolyfillOptions polyfill_options;
    const u64 cache_fingerprint;
    const u64 native_fingerprint;
    bool is_executing = false;
    bool invalidate_entire_cache = false;
    boost::icl::interval_set<u64> invalid_cache_ranges;
    std::mutex invalidation_mutex;
};

Jit::Jit(UserConfig conf)
        : impl(std::make_unique<Jit::Impl>(this, conf)) {}

Jit::~Jit() = default;

HaltReason Jit::Run() {
    return impl->Run();
}

HaltReason Jit::Step() {
    return impl->Step();
}

void Jit::ClearCache() {
    impl->ClearCache();
}

BlockCacheStats Jit::GetBlockCacheStats() const {
    return impl->GetBlockCacheStats();
}

PrecompileResult Jit::Precompile(u64 descriptor, u64 end_pc, u64 code_hash) {
    return impl->Precompile(descriptor, end_pc, code_hash);
}

PrecompileResult Jit::PlaceCached(u64 descriptor, size_t keep_free) {
    return impl->PlaceCached(descriptor, keep_free);
}

size_t Jit::CodeSpaceRemaining() const {
    return impl->CodeSpaceRemaining();
}

std::optional<ScanResult> Jit::ScanBlock(u64 descriptor) {
    return impl->ScanBlock(descriptor);
}

std::optional<u64> Jit::GuestCodeHash(u64 begin, u64 end) const {
    return impl->GuestCodeHash(begin, end);
}

u64 Jit::CacheKey(u64 descriptor) const {
    return impl->CacheKey(descriptor);
}

void Jit::InvalidateCacheRange(u64 start_address, size_t length) {
    impl->InvalidateCacheRange(start_address, length);
}

void Jit::Reset() {
    impl->Reset();
}

void Jit::HaltExecution(HaltReason hr) {
    impl->HaltExecution(hr);
}

void Jit::ClearHalt(HaltReason hr) {
    impl->ClearHalt(hr);
}

u64 Jit::GetSP() const {
    return impl->GetSP();
}

void Jit::SetSP(u64 value) {
    impl->SetSP(value);
}

u64 Jit::GetPC() const {
    return impl->GetPC();
}

void Jit::SetPC(u64 value) {
    impl->SetPC(value);
}

u64 Jit::GetRegister(size_t index) const {
    return impl->GetRegister(index);
}

void Jit::SetRegister(size_t index, u64 value) {
    impl->SetRegister(index, value);
}

std::array<u64, 31> Jit::GetRegisters() const {
    return impl->GetRegisters();
}

void Jit::SetRegisters(const std::array<u64, 31>& value) {
    impl->SetRegisters(value);
}

Vector Jit::GetVector(size_t index) const {
    return impl->GetVector(index);
}

void Jit::SetVector(size_t index, Vector value) {
    impl->SetVector(index, value);
}

std::array<Vector, 32> Jit::GetVectors() const {
    return impl->GetVectors();
}

void Jit::SetVectors(const std::array<Vector, 32>& value) {
    impl->SetVectors(value);
}

u32 Jit::GetFpcr() const {
    return impl->GetFpcr();
}

void Jit::SetFpcr(u32 value) {
    impl->SetFpcr(value);
}

u32 Jit::GetFpsr() const {
    return impl->GetFpsr();
}

void Jit::SetFpsr(u32 value) {
    impl->SetFpsr(value);
}

u32 Jit::GetPstate() const {
    return impl->GetPstate();
}

void Jit::SetPstate(u32 value) {
    impl->SetPstate(value);
}

void Jit::ClearExclusiveState() {
    impl->ClearExclusiveState();
}

bool Jit::IsExecuting() const {
    return impl->IsExecuting();
}

std::string Jit::Disassemble() const {
    return impl->Disassemble();
}

}  // namespace Dynarmic::A64
