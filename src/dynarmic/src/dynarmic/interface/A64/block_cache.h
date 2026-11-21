// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace Dynarmic::A64 {

/// Persistent storage for translated blocks, supplied by the embedder (for example a file per game).
///
/// When UserConfig::block_cache is set, the JIT looks a block up here before translating it, and stores every
/// block it translates. A record holds the block's optimized IR and a hash of the guest instructions it was
/// translated from; a record whose instructions no longer match is ignored and replaced, so stale entries are
/// harmless. Keys already include a fingerprint of the JIT configuration and IR schema.
///
/// Implementations must be thread-safe: every core's JIT may call them concurrently.
class BlockCacheStorage {
public:
    virtual ~BlockCacheStorage() = default;

    /// Returns the record stored under key, if any.
    virtual std::optional<std::vector<std::uint8_t>> Load(std::uint64_t key) = 0;

    /// Stores (or replaces) the record under key.
    virtual void Store(std::uint64_t key, std::vector<std::uint8_t> record) = 0;

    /// How a block that was not in the code cache was produced.
    enum class CompileKind : std::uint8_t {
        Translated,  ///< translated, optimized and emitted (then stored)
        Native,      ///< placed from cached host code
        Ir,          ///< emitted from cached IR
    };

    /// Called on the compiling core's thread after each such block, with the whole time of the lookup (failed
    /// cache loads, translation, emission and the store included). For statistics; the default does nothing.
    virtual void OnBlockCompiled(CompileKind kind, std::uint64_t ns) {
        (void)kind;
        (void)ns;
    }

    /// Whether a record is stored under key: an index lookup, no decompression (Jit::Precompile skips such
    /// blocks). The default says no.
    virtual bool Contains(std::uint64_t key) {
        (void)key;
        return false;
    }

    /// Called on the compiling core's thread when a block that can be stored is placed for execution (from a
    /// record or freshly translated), with its location descriptor (IR::LocationDescriptor value: PC, FPCR mode
    /// bits, single-step), the end of the guest code it covers and that code's hash (the hash records carry).
    /// None of these depend on the emulator build, so the embedder can keep them as a durable list of the blocks
    /// a program runs and rebuild records from it with Jit::Precompile. The default does nothing.
    virtual void OnBlockPlaced(std::uint64_t descriptor, std::uint64_t end_pc, std::uint64_t code_hash) {
        (void)descriptor;
        (void)end_pc;
        (void)code_hash;
    }
};

/// Outcome of Jit::Precompile.
enum class PrecompileResult : std::uint8_t {
    Built,         ///< translated, emitted and stored
    Present,       ///< already in this Jit or in the storage
    GuestChanged,  ///< the guest code no longer hashes to the given value: skipped
    Unreadable,    ///< the guest code cannot be read (not mapped): skipped
    NotCacheable,  ///< the block cannot be stored (host calls in its IR), or there is no storage
    NoSpace,       ///< Jit::PlaceCached only: the code cache is too full
};

/// Outcome of Jit::ScanBlock: where the translator ends a block and where it can continue, without emitting it.
struct ScanResult {
    std::uint64_t end_pc = 0;     ///< end of the guest code the block covers (the JIT's own block end)
    std::uint64_t code_hash = 0;  ///< hash of that code, as Jit::Precompile checks it (valid if `hashed`)
    bool hashed = false;          ///< the code could be hashed (readable, at most 64 KiB)
    bool cacheable = false;       ///< the optimized IR can be stored (Precompile would build it)
    std::uint8_t successor_count = 0;
    /// Guest PCs the block names as continuations: linked targets, both arms of conditional terminals, the
    /// return address of a call (PushRSB) and immediate PC writes. Indirect branches add none.
    std::uint64_t successors[6] = {};
};

/// Counters for one Jit instance (Jit::GetBlockCacheStats).
struct BlockCacheStats {
    std::uint64_t hits = 0;           ///< blocks emitted from a cached record
    std::uint64_t misses = 0;         ///< blocks translated (and stored)
    std::uint64_t stale = 0;          ///< records rejected because the guest code changed or the record was bad
    std::uint64_t translate_ns = 0;   ///< time spent translating and optimizing on misses
    std::uint64_t load_ns = 0;        ///< time spent verifying and deserializing on hits
    std::uint64_t emit_ns = 0;        ///< time spent emitting host code (hits and misses)
    std::uint64_t native_hits = 0;    ///< hits placed from cached host code (no code generation)
    std::uint64_t native_stored = 0;  ///< blocks whose host code was stored
    std::uint64_t native_not_relocatable = 0;  ///< blocks whose host code could not be made relocatable
    std::uint64_t native_rejected = 0;  ///< cached host code not used (configuration changed, fastmem fallback, ...)
    std::uint64_t native_verified = 0;  ///< verification mode: cached host code compared with fresh code
    std::uint64_t native_verify_mismatches = 0;  ///< verification mode: comparisons that differed
    // Parts of load_ns on hits with host code:
    std::uint64_t load_storage_ns = 0;  ///< BlockCacheStorage::Load (lookup, decompression)
    std::uint64_t load_parse_ns = 0;    ///< parsing the record
    std::uint64_t load_hash_ns = 0;     ///< hashing the guest code
    std::uint64_t load_place_ns = 0;    ///< copying, relocating, linking and registering the code
};

}  // namespace Dynarmic::A64
