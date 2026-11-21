// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "dynarmic/interface/A64/config.h"
#include "dynarmic/interface/halt_reason.h"

namespace Dynarmic {
namespace A64 {

class Jit final {
public:
    explicit Jit(UserConfig conf);
    ~Jit();

    /**
     * Runs the emulated CPU.
     * Cannot be recursively called.
     */
    HaltReason Run();

    /**
     * Step the emulated CPU for one instruction.
     * Cannot be recursively called.
     */
    HaltReason Step();

    /**
     * Clears the code cache of all compiled code.
     * Can be called at any time. Halts execution if called within a callback.
     */
    void ClearCache();

    /**
     * Invalidate the code cache at a range of addresses.
     * @param start_address The starting address of the range to invalidate.
     * @param length The length (in bytes) of the range to invalidate.
     */
    void InvalidateCacheRange(std::uint64_t start_address, std::size_t length);

    /**
     * Reset CPU state to state at startup. Does not clear code cache.
     * Cannot be called from a callback.
     */
    void Reset();

    /**
     * Stops execution in Jit::Run.
     */
    void HaltExecution(HaltReason hr = HaltReason::UserDefined1);

    /**
     * Clears a halt reason from flags.
     * Warning: Only use this if you're sure this won't introduce races.
     */
    void ClearHalt(HaltReason hr = HaltReason::UserDefined1);

    /// Read Stack Pointer
    std::uint64_t GetSP() const;
    /// Modify Stack Pointer
    void SetSP(std::uint64_t value);

    /// Read Program Counter
    std::uint64_t GetPC() const;
    /// Modify Program Counter
    void SetPC(std::uint64_t value);

    /// Read general-purpose register.
    std::uint64_t GetRegister(std::size_t index) const;
    /// Modify general-purpose register.
    void SetRegister(size_t index, std::uint64_t value);

    /// Read all general-purpose registers.
    std::array<std::uint64_t, 31> GetRegisters() const;
    /// Modify all general-purpose registers.
    void SetRegisters(const std::array<std::uint64_t, 31>& value);

    /// Read floating point and SIMD register.
    Vector GetVector(std::size_t index) const;
    /// Modify floating point and SIMD register.
    void SetVector(std::size_t index, Vector value);

    /// Read all floating point and SIMD registers.
    std::array<Vector, 32> GetVectors() const;
    /// Modify all floating point and SIMD registers.
    void SetVectors(const std::array<Vector, 32>& value);

    /// View FPCR.
    std::uint32_t GetFpcr() const;
    /// Modify FPCR.
    void SetFpcr(std::uint32_t value);

    /// View FPSR.
    std::uint32_t GetFpsr() const;
    /// Modify FPSR.
    void SetFpsr(std::uint32_t value);

    /// View PSTATE
    std::uint32_t GetPstate() const;
    /// Modify PSTATE
    void SetPstate(std::uint32_t value);

    /// Clears exclusive state for this core.
    void ClearExclusiveState();

    /**
     * Returns true if Jit::Run was called but hasn't returned yet.
     * i.e.: We're in a callback.
     */
    bool IsExecuting() const;

    /// @brief Disassemble the instructions following the current pc and return
    /// the resulting instructions as a vector of their string representations.
    std::string Disassemble() const;

    /// Block cache and compile-time counters of this Jit.
    BlockCacheStats GetBlockCacheStats() const;

    /// Translates, emits and stores the block at `descriptor` (an IR::LocationDescriptor value) without running
    /// it, if the guest code in [PC, end_pc) still hashes to `code_hash` (the hash BlockCacheStorage::OnBlockPlaced
    /// reported) and the block is neither in this Jit nor in UserConfig::block_cache. The block stays in this
    /// Jit's code cache. A cache invalidation requested before the call (ClearCache, InvalidateCacheRange) is
    /// carried out first, as Run does. Must not be called while this Jit runs.
    PrecompileResult Precompile(std::uint64_t descriptor, std::uint64_t end_pc, std::uint64_t code_hash);

    /// Places the block at `descriptor` (an IR::LocationDescriptor value) in this Jit's code cache from the host
    /// code stored for it in UserConfig::block_cache, without running it, translating it or emitting code: the
    /// placement Run does on a cache hit, done ahead of time (for example while the game loads, so the first run
    /// of the block finds it built). Built: placed. Present: already in this Jit. GuestChanged / Unreadable: the
    /// guest code no longer matches the record or cannot be read. NotCacheable: no record with usable host code
    /// (or no storage, or host code is not kept). NoSpace: fewer than `keep_free` bytes (plus the JIT's own
    /// reserve) are left in the code cache; nothing was placed. Nothing is stored and no statistics or storage
    /// callbacks are reported. Must not be called while this Jit runs.
    PrecompileResult PlaceCached(std::uint64_t descriptor, std::size_t keep_free);

    /// Bytes left in this Jit's code cache before it has to be cleared.
    std::size_t CodeSpaceRemaining() const;

    /// Translates and optimizes the block at `descriptor` the way Run and Precompile do (same options, same block
    /// end, same guest hash) but emits nothing, stores nothing and leaves the code cache untouched: for finding a
    /// program's blocks before it runs. Nothing if the block's first instruction cannot be read. Must not be
    /// called while this Jit runs.
    std::optional<ScanResult> ScanBlock(std::uint64_t descriptor);

    /// Hash of the guest instructions in [begin, end) as Precompile and the block cache check it; nothing if any
    /// cannot be read or the range is empty or over 64 KiB.
    std::optional<std::uint64_t> GuestCodeHash(std::uint64_t begin, std::uint64_t end) const;

    /// The key the block at `descriptor` is stored under in UserConfig::block_cache (it mixes in this Jit's
    /// configuration, so only Jits configured alike share keys).
    std::uint64_t CacheKey(std::uint64_t descriptor) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};


/// A map from host code to the guest blocks it was emitted for, for profilers (the debug server's
/// jit_hot). Off unless the process starts with EDEN_JIT_HOT_MAP=1: then every A64 Jit (one code
/// space each) appends one record per block it emits or loads from the block cache, under a small
/// mutex, and drops its records when its whole cache is cleared. Records of blocks invalidated one
/// by one stay: their host code is not reused until the cache is cleared.
namespace HotMap {

enum class Kind : std::uint8_t {
    Outside,        ///< Not in any Jit's code space: emulator code, the OS, waiting.
    Prelude,        ///< In a code space before its first block: dispatcher, fast dispatch, thunks.
    BetweenBlocks,  ///< In a code space after the prelude but in no recorded block.
    Block,          ///< In a block's host code.
};

struct Hit {
    std::uint64_t guest_pc = 0;   ///< The block's first guest instruction (its location's PC).
    std::uint64_t guest_end = 0;  ///< One past the block's last guest instruction.
    std::uint32_t host_size = 0;  ///< Bytes of host code.
    std::int32_t space = -1;      ///< Which code space, in registration order; -1 when Outside.
    Kind kind = Kind::Outside;
};

struct Stats {
    std::size_t spaces = 0;
    std::size_t blocks = 0;
    std::uint64_t clears = 0;  ///< Whole-cache clears since start, over all code spaces.
};

bool Enabled();
/// Resolves host addresses (out.size() == rips.size()). Holds the map's mutex while it runs, so it
/// is safe while Jits compile on other threads; a clear between taking an address and resolving it
/// would attribute it to newer code (Stats::clears tells).
void Resolve(std::span<const std::uint64_t> rips, std::span<Hit> out);
Stats GetStats();

}  // namespace HotMap

}  // namespace A64
}  // namespace Dynarmic
