// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <functional>
#include <span>

namespace Kernel {
class KProcess;
}

namespace Core {

class ArmDynarmic64;
class JitBlockCache;

/// Progress of a build: blocks handled so far, blocks to handle.
using JitCacheBuildProgress = std::function<void(std::size_t done, std::size_t total)>;

/// Fills a program's per-executable JIT cache file before the guest runs: on the first boot after the emulator
/// executable changed (which discards the file), on a machine that has a prebuild list but no records yet, or when
/// the list or its scope changed (JitBlockCache::TakeRebuildRequest). The setting "Precompile CPU code"
/// (JitBlockCache::GetPrecompileMode) picks what is built: nothing, the manifest only, or the manifest and the list.
///
/// The manifest's blocks (seen in play, most placed first) and the prebuild list's executed entries are built
/// "resident": they stay in the code cache of the JIT that built them. The rest of the list within
/// EDEN_JIT_CACHE_PREBUILD_SCOPE (played: blocks of functions that ran; full: every entry) is built "file-only",
/// before the resident part: records go to the file and each JIT drops its code every few tens of thousands of
/// blocks, so millions of entries cost file space, not memory. Every entry whose guest code still hashes to the
/// recorded value is translated, emitted and stored; entries whose code changed or is not mapped are skipped, and
/// entries already in the file are not built again.
///
/// One host thread per JIT (EDEN_JIT_CACHE_REBUILD_THREADS caps the count). Blocks until done. Progress goes to
/// `progress` (if set) about ten times a second and to the log every two seconds. The JITs must not be running.
void RebuildJitCache(JitBlockCache& cache, std::span<ArmDynarmic64* const> jits,
                     const JitCacheBuildProgress& progress);

/// Places the manifest's blocks (seen in earlier sessions, most placed first) in every JIT from their cached x64
/// code (ArmDynarmic64::PlaceCachedBlock), so a launch that has nothing to build does not place them one by one on
/// guest frames as each core first reaches them. No translation, nothing stored, no placements counted. A core
/// stops when its code cache gets near full (the rest is left to the game). One host thread per JIT, as
/// RebuildJitCache. EDEN_JIT_CACHE_PRELOAD=0 turns it off. The JITs must not be running.
void PreloadJitCache(JitBlockCache& cache, std::span<ArmDynarmic64* const> jits,
                     const JitCacheBuildProgress& progress);

/// The boot step: asks the cache whether a build is due (JitBlockCache::TakeRebuildRequest, which checks the
/// prebuild lists against the code `process` loaded), generates the code list first when no list matches
/// (GenerateJitCodeList, jit_code_discovery.h), then runs RebuildJitCache. Only the first call per process does
/// anything. Then (or when nothing needs building) PreloadJitCache. The JITs must not be running.
void PrebuildJitCache(JitBlockCache& cache, Kernel::KProcess& process, std::span<ArmDynarmic64* const> jits,
                      const JitCacheBuildProgress& progress);

} // namespace Core
