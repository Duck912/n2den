// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "core/arm/dynarmic/jit_block_cache.h"

namespace Kernel {
class KProcess;
}

namespace Core {

class ArmDynarmic64;
class JitPrebuildList;

/// Version of the emulator's own code list generator, written into the header of the lists it makes (the
/// header's first reserved word; lists made offline carry 0). A list made by another version is made again, so
/// a change to discovery or to where the translator ends blocks can invalidate old lists on purpose.
/// 2: with no manifest the common FPCR is FZ (jit_code_list_first_boot_fpcr), not 0; version 1 lists made on a
/// first boot hold the wrong mode.
/// 3: with no manifest the common FPCR comes from the code's own FPCR writes (JitStaticFpcrHint); the header's
/// last reserved word holds the number of manifest blocks the list was made with (0: a first boot), and a list
/// whose common FPCR the manifest contradicts is made again (JitPrebuildListMatches).
inline constexpr u32 jit_code_list_generator = 3;

/// FPCR guest threads start with (k_thread.cpp resets it to 0): the first-boot common FPCR when the code's own
/// FPCR writes say nothing (JitStaticFpcrHint).
inline constexpr u32 jit_code_list_first_boot_fpcr = 0;

/// The guest code a process will run, copied from its memory after the loader has mapped and patched every
/// module and before any guest thread ran: each executable Code/AliasCode block (a module's .text, load-time
/// patch trampolines included) and, with NVN-HLE, the thunk slots appended to the sdk module (they sit in its
/// data segment). Modules (blocks with a MOD0 header) are copied whole (.text, .rodata, .data), for their unwind
/// table, dynamic section and symbols.
class JitCodeImage {
public:
    struct Window {
        u64 base = 0;
        u64 size = 0;           ///< bytes of code
        const u8* data = nullptr;
        int module = -1;        ///< index into modules, or -1
    };
    struct Module {
        u64 base = 0;
        u64 text_size = 0;      ///< executable bytes from base
        std::vector<u8> bytes;  ///< the whole module
    };

    static std::unique_ptr<JitCodeImage> Capture(Kernel::KProcess& process);

    /// Hash of every window's address, size and bytes. Identifies the code a list was generated from: an update,
    /// another set of load-time patches or another NVN-HLE choice gives another hash.
    u64 Hash() const {
        return hash;
    }
    const std::vector<Window>& Windows() const {
        return windows;
    }
    const std::vector<Module>& Modules() const {
        return modules;
    }
    u64 ThunkBase() const {
        return thunk_base;
    }
    u64 CodeBytes() const;
    /// Index of the window holding `pc`, or -1.
    int Find(u64 pc) const;

private:
    std::vector<Module> modules;
    std::vector<std::vector<u8>> loose;  // bytes of windows that are not module .text
    std::vector<Window> windows;         // sorted by base
    u64 hash = 0;
    u64 thunk_base = 0;
};

/// What the code's own FPCR writes say about the FPCR mode most of it runs in, for a first boot (no manifest).
/// Every MSR FPCR, Xt of the image is found and the value written resolved backward within its block (at most
/// 24 instructions): MOVZ/MOVN/MOVK, logical immediates and registers, CSEL, bitfield moves, LDR literal, and MRS
/// FPCR (then the write is relative: some bits set or cleared, the rest kept). Library code saves and restores
/// FPCR or takes it from an argument (unresolved); a game that wants flush-to-zero sets it itself (Ocarina of
/// Time: one unconditional "FPCR |= FZ"; Fire Emblem: Fortune's Weave: no game write at all, and it runs in
/// FPCR 0). The hint is the thread-start FPCR changed by the most common write that sets or clears a bit the
/// block key keeps (a tie keeps the thread-start FPCR). Writes whose effect leaves the thread-start FPCR as it is
/// (msr fpcr, xzr from 0) are not counted: they cannot outvote a write that changes it.
struct JitFpcrHint {
    u32 fpcr = jit_code_list_first_boot_fpcr;
    std::size_t writes = 0;       ///< MSR FPCR instructions found
    std::size_t resolved = 0;     ///< writes that set or clear at least one key bit
    std::string detail;           ///< resolved effects and counts, for the log
};
JitFpcrHint JitStaticFpcrHint(const JitCodeImage& image);

/// The manifest's FPCR mix: block count per FPCR (key bits), most frequent first.
std::vector<std::pair<u32, std::size_t>> JitManifestFpcrMix(std::span<const JitBlockCache::ManifestEntry> manifest);

/// Whether `list` may be used for the code in `image`. A list the emulator generated must come from this
/// generator version and carry this image's hash. A list made by another tool (EDEN_JIT_CACHE_PREBUILD_IMPORT)
/// hashes the image its own way, so unless the hashes agree its entries are sampled instead: the guest hash of
/// up to 4096 entries spread over the list is recomputed from this process's code with `jit`, and the list is
/// used if at least 95% match. A generated list is also made again when the manifest contradicts its common
/// FPCR (`manifest_mix`, JitManifestFpcrMix): the manifest's most frequent FPCR holds more than twice the blocks
/// of the list's mode, or the list was made with no manifest and the manifest runs at least a quarter of its
/// blocks in other modes (the list then adds them for the played functions). Needs 4096 manifest blocks; the
/// override EDEN_JIT_CACHE_GENERATE_FPCR replaces the manifest's choice. `why` says what was found, for the log.
bool JitPrebuildListMatches(const JitPrebuildList& list, bool generated, const JitCodeImage& image,
                            ArmDynarmic64& jit, std::span<const std::pair<u32, std::size_t>> manifest_mix,
                            std::string& why);

/// Progress of the generation: seeds handled so far, seeds in total.
using JitCodeListProgress = std::function<void(std::size_t done, std::size_t total)>;

/// Finds every block of the image with the JIT's own translator (ArmDynarmic64::ScanBlock: the block ends,
/// guest hashes and keys Precompile expects) and writes them as a prebuild list (jit_prebuild_list.h format) to
/// `out` (a temp file, then renamed). Seeds: unwind table (FDE) starts, RELATIVE relocation targets in code,
/// ADR/ADRP+ADD code addresses, defined function symbols, module entries, the NVN-HLE thunk slots, the
/// manifest's blocks and a linear sweep (direct branch targets and the instruction after every terminator).
/// Descent follows every successor a block names. One thread per JIT, the JITs must be idle.
///
/// FPCR policy (as the offline exporter): the common mode is the manifest's most frequent FPCR, or
/// `default_fpcr` without a manifest (the caller passes the static hint, JitStaticFpcrHint; `default_from` names
/// it for the log) (EDEN_JIT_CACHE_GENERATE_FPCR=<hex> overrides it). Blocks in a function
/// (FDE range) that holds a manifest block get every FPCR the manifest has, the rest the common mode only.
/// Order: the manifest's keys, then the rest of the played functions, then everything else. Manifest keys that
/// discovery did not reach are added with their own end and hash. Returns the entry count, or nothing on failure.
std::optional<std::size_t> GenerateJitCodeList(const JitCodeImage& image, std::span<ArmDynarmic64* const> jits,
                                               std::span<const JitBlockCache::ManifestEntry> manifest,
                                               u32 default_fpcr, const std::string& default_from, u64 program_id,
                                               const std::filesystem::path& out,
                                               const JitCodeListProgress& progress);

} // namespace Core
