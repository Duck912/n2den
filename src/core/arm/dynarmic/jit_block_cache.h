// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "common/settings_enums.h"
#include "dynarmic/interface/A64/block_cache.h"

namespace Core {

class JitPrebuildList;

/// Disk-backed store for Dynarmic NX2's translated A64 blocks (IR and, optionally, relocatable x64 code), one
/// file per program, plus a durable manifest of the blocks the program runs.
///
/// Records file: <cache dir>/jit/<program id>.bin (EDEN_JIT_CACHE_DIR=<dir> puts it in <dir> instead). A header
/// holds a magic, the format version and the identity of the emulator executable (a hash of its file, see
/// ExecutableIdentity); a file written by any other executable is discarded, since its translations and code may
/// differ (a translator or emitter edit changes the executable). Records follow, appended as blocks are
/// translated: key (8 bytes), stored size (4), checksum of the stored bytes (4), then the uncompressed size (4) and
/// the record compressed with LZ4. A later record for a key replaces an earlier one.
///
/// The file is memory-mapped; only an index (key, offset, size: 16 bytes per record, a flat table sorted by a hash
/// of the key) is kept in memory, and a record is checked and decompressed when it is looked up. The index is
/// saved next to the file (<program id>.bin.idx, see SaveIndex), so opening a large file reads the index instead
/// of every record header; records appended after the saved index are found by reading their headers. Loading
/// stops at a truncated record and drops it, so a torn write after a crash costs only that record. When replaced
/// records take more space than live ones, the file is compacted on open (streamed, not held in memory). The
/// file grows to at most 4 GiB. Dynarmic checks every record against the guest code before using it.
///
/// Manifest: <same dir>/<program id>.manifest. Independent of the emulator build, it survives executable changes:
/// one entry per block location descriptor (PC, FPCR mode bits, single-step, as the JIT keys blocks), with the end
/// of the guest code the block covers, a hash of that code, the number of placements over all sessions and the last
/// session that placed it. When the records file starts empty (new executable, first run), the blocks are rebuilt
/// from it before the guest runs (jit_cache_builder.h). See SaveManifest for the format and how it is written.
///
/// Prebuild list: <same dir>/<program id>.prebuild (jit_prebuild_list.h), blocks found offline in the title's
/// code; EDEN_JIT_CACHE_PREBUILD_IMPORT=<file> copies a list there at open. Its entries join the manifest's in the
/// boot build as static entries, after the blocks seen in play; EDEN_JIT_CACHE_PREBUILD_SCOPE picks how many
/// (PrebuildScope). A small marker (<program id>.prebuilt) records which list and scope the records file was last
/// built for, so a new list, scope or executable builds again (only what is missing). The list is only used when
/// the setting "Precompile CPU code" is "Entire game" (GetPrecompileMode). Without an imported list that matches
/// the loaded code, the emulator generates one itself at boot (<same dir>/<program id>.generated.prebuild, same
/// format, keyed by a hash of the loaded code; jit_code_discovery.h) and reuses it on later boots.
///
/// Rewrites (compaction, DropRecords, the empty file, the index, the manifest, the marker, the generated list) go
/// to a temp file named for this process, which is flushed to disk and then renamed over the old file with the
/// rename written through (ReplaceDurably): a power loss leaves the old file or the complete new one. Records
/// stored by a boot build are flushed to disk before the marker that vouches for them is written.
///
/// One process per cache: Open takes <program id>.lock in the cache directory (exclusive, held until the cache is
/// destroyed). A second process that finds it held gets no cache at all (Open returns nullptr) and touches none
/// of the files; see Open. The lock holder removes temp files of this program that a crash left behind.
///
/// Thread-safe: all cores' JITs share one instance.
class JitBlockCache final : public Dynarmic::A64::BlockCacheStorage {
public:
    /// Opens (or creates) the cache of a program. Returns nullptr when disabled (setting cpu_jit_cache Off, or
    /// EDEN_JIT_CACHE=0), when the cache directory cannot be used, or when another process holds the program's
    /// cache (its lock file): that process may rewrite the records file at any time, so this one neither reads
    /// nor writes it and runs without a cache (blocks are translated as the game reaches them).
    static std::shared_ptr<JitBlockCache> Open(u64 program_id);

    /// A temp file name next to `target`, unique to this process and call: <target>.<pid>-<n>.tmp.
    static std::filesystem::path TempPathFor(const std::filesystem::path& target);

    /// Flushes `temp` (closed, fully written) to disk, then renames it over `target` with the rename written
    /// through. On failure `temp` is left in place and `ec` says why.
    static bool ReplaceDurably(const std::filesystem::path& temp, const std::filesystem::path& target,
                               std::error_code& ec);

    /// Whether the cache is enabled (setting cpu_jit_cache not Off, and EDEN_JIT_CACHE not "0").
    static bool Enabled();

    /// Whether the cache also keeps generated x64 code (setting cpu_jit_cache = Translations + x64 code).
    static bool KeepsHostCode();

    /// Cached blocks hold absolute guest addresses, so they only match when the program loads at the same base.
    /// With the cache enabled and no RNG seed configured, the loader uses this per-program ASLR page offset
    /// instead of a random one: the same on every boot of a program, still different between programs.
    static u64 StableAslrOffset(u64 program_id);

    /// Identity of the executable that holds the JIT: a hash of its file's bytes, computed once. Any rebuild
    /// that changes the translator, optimizer, emitter, prelude or a host helper changes it.
    static u64 ExecutableIdentity();

    ~JitBlockCache() override;

    std::optional<std::vector<u8>> Load(u64 key) override;
    void Store(u64 key, std::vector<u8> record) override;
    bool Contains(u64 key) override;

    /// Adds the block's compile time to the frame statistics (EDEN_NX2_FRAME_STATS) and, while sched_trace
    /// records, files it as a host span. Otherwise two flag checks.
    void OnBlockCompiled(CompileKind kind, u64 ns) override;

    /// Counts a placement of the block in the manifest (adds or updates its entry). Only appends to a per-core
    /// buffer; the manifest itself is updated off the guest threads (DrainPlacements).
    void OnBlockPlaced(u64 descriptor, u64 end_pc, u64 code_hash) override;

    /// Names the guest core the calling host thread runs (its JIT spans carry it). Once per core thread.
    static void SetThreadCore(u32 core);

    /// Writes pending records to the file.
    void Flush();

    size_t RecordCount() const;

    /// One block of the manifest.
    struct ManifestEntry {
        u64 descriptor;    // IR::LocationDescriptor value
        u64 end_pc;        // end of the guest code the block covers
        u64 code_hash;     // Dynarmic's hash of that code
        u32 hits;          // placements over all sessions (all cores), saturating
        u32 last_session;  // last session that placed it
    };

    /// The manifest's entries (pending placements included), most placed first.
    std::vector<ManifestEntry> ManifestSnapshot();
    size_t ManifestSize() const;

    /// Adds blocks that the boot build compiled into a guest core's JIT and left there to the manifest, with no
    /// placement counted (hits 0), unless it lists them already. Such a block is never reported by
    /// OnBlockPlaced when that core runs it (the JIT finds it built and links to it), so without this the
    /// manifest only learns of the ones another core happens to run as well. Returns how many were added.
    size_t AddResidentBlocks(std::span<const ManifestEntry> blocks);

    /// How much of the prebuild list the boot build compiles (EDEN_JIT_CACHE_PREBUILD_SCOPE).
    enum class PrebuildScope : u32 {
        Executed = 1,  ///< "executed": entries seen running in a recorded session
        Played = 2,    ///< "played": plus every block of the functions that ran
        Full = 3,      ///< "full" (default): every entry
    };
    static PrebuildScope GetPrebuildScope();
    static const char* PrebuildScopeName(PrebuildScope scope);

    /// What the boot build compiles (setting cpu_jit_precompile, "Precompile CPU code"):
    ///   Off:         nothing; blocks are compiled when the game reaches them (and stored in the cache then).
    ///   PlayedAreas: the manifest's blocks (seen in this program's earlier sessions); the prebuild list is unused.
    ///   EntireGame:  the manifest's blocks plus the prebuild list within its scope (EDEN_JIT_CACHE_PREBUILD_SCOPE,
    ///                default full); without a list that matches the loaded code, one is generated at boot.
    /// Precedence, first match wins (test scripts set the variables; the GUI sets the setting):
    ///   1. EDEN_JIT_CACHE_REBUILD=0                                 -> Off
    ///   2. EDEN_JIT_CACHE_PREBUILD_IMPORT or _PREBUILD_SCOPE set    -> EntireGame
    ///   3. EDEN_JIT_CACHE_REBUILD set to anything else              -> the setting, but at least PlayedAreas
    ///   4. the setting cpu_jit_precompile (per game, default EntireGame)
    /// Only with cpu_jit_cache = Translations + x64 code: the build stores generated code, which the
    /// Translations mode does not keep.
    using PrecompileMode = Settings::JitPrecompile;
    static PrecompileMode GetPrecompileMode(const char** source = nullptr);

    /// Says whether a prebuild list may be used for the code the process loaded (`generated`: the list is the
    /// emulator's own, see jit_code_discovery.h). `why` goes to the log.
    using ListCheck = std::function<bool(const JitPrebuildList& list, bool generated, std::string& why)>;

    /// True once, when the boot build should run (x64 code kept, precompile mode not Off): the records file held
    /// no records when opened (new file or one from another executable) and the manifest (or, for EntireGame, the
    /// prebuild list) lists blocks; or, for EntireGame, a prebuild list exists that this records file was not yet
    /// built for (new list or scope); or, for EntireGame, no list matches the loaded code and one must be
    /// generated (GeneratesList).
    ///
    /// EntireGame picks the list in this order: the imported one (<id>.prebuild, EDEN_JIT_CACHE_PREBUILD_IMPORT)
    /// if `check` accepts it, then the generated one (<id>.generated.prebuild) if `check` accepts it, else a new
    /// generated one. A list `check` rejects is logged and ignored, never deleted.
    bool TakeRebuildRequest(const ListCheck& check);

    /// True once per process, when the boot preload (PreloadJitCache) should run: x64 code is kept and the
    /// precompile mode is not Off.
    bool TakePreloadRequest();

    /// Whether the boot build that TakeRebuildRequest asked for includes the prebuild list (EntireGame with a list).
    bool UsesPrebuildList() const {
        return use_prebuild_list;
    }

    /// Whether the boot build must first generate the list (no list matched the loaded code); it is written to
    /// PrebuildListPath().
    bool GeneratesList() const {
        return generate_list;
    }

    u64 ProgramId() const {
        return program_id;
    }
    /// The list the boot build uses (after TakeRebuildRequest).
    const std::filesystem::path& PrebuildListPath() const {
        return selected_list_path;
    }

    /// Notes that the records file now holds the build of the prebuild list with this header and scope.
    void MarkPrebuilt(std::span<const u8> list_header, PrebuildScope scope);

    /// While bulk storing, stored records go to the file only (not kept in memory, not visible to Contains);
    /// ending it writes them and adds them to the index (from where they were written: the file is not read
    /// again), so they are then loaded from it like any other record.
    void BeginBulkStore();
    void EndBulkStore();

    /// Writes the manifest if it changed since the last write (also done every two minutes and on close).
    void SaveManifest();

    /// Removes the records whose key `keep` rejects (records of blocks a regenerated list no longer names, such
    /// as the other FPCR mode's), by writing the live records that are kept to a new file and renaming it over
    /// the old one (streamed, as the compaction on open). Does nothing unless at least `min_bytes` would go.
    /// Only before the guest runs: it refuses while a bulk store is open or records stored this session are held
    /// in memory or not yet indexed, and on any write or rename failure the old file stays as it was. Returns
    /// the bytes removed, or nothing if it refused or failed; `what` says what happened, for the log.
    std::optional<u64> DropRecords(const std::function<bool(u64 key)>& keep, u64 min_bytes, std::string& what);

    /// Size of the records file and of the in-memory index, for logs.
    u64 FileSize() const;
    size_t IndexBytes() const;

private:
    struct Location {
        u64 offset;  // of the stored bytes in the mapped file (the checksum is the 4 bytes before)
        u32 size;
    };

    /// One record of the file in the index: Mix(key) (a bijection, so the key itself) and where the stored bytes
    /// are, offset << 24 | size.
    struct IndexEntry {
        u64 mixed;
        u64 where;
    };

    struct ManifestValue {
        u64 end_pc;
        u64 code_hash;
        u32 hits;
        u32 last_session;
    };

    struct ManifestShard {
        mutable std::mutex mutex;
        std::unordered_map<u64, ManifestValue> blocks;  // by descriptor
    };
    static constexpr size_t manifest_shard_count = 16;

    struct Placement {
        u64 descriptor;
        u64 end_pc;
        u64 code_hash;
    };
    struct PlacementBuffer {
        std::mutex mutex;  // uncontended but for the drain
        std::vector<Placement> items;
    };
    static constexpr size_t placement_buffer_count = 8;  // guest cores 0-5, then one shared by other threads

    JitBlockCache(u64 program_id, const std::filesystem::path& dir);

    /// Takes the cache lock (lock_path); false if another process holds it or it cannot be created.
    bool AcquireLock();
    void ReleaseLock();
    /// Removes <program id>.*.tmp files in the cache directory (left by a crash mid-rewrite). Lock held.
    void RemoveStaleTemps();

    void OpenFile();
    void WriteEmptyFile();
    bool CompactFile();
    bool MapFile(u64 size);
    void UnmapFile();
    void FlushLocked();
    void LoadManifest();
    void StartManifestSaver();
    void DrainPlacements();

    std::optional<Location> FindOnDisk(u64 key) const;
    /// Makes `entries` the index: sorts them by mixed key unless `sorted`, keeps the last record written per key
    /// (the highest offset), builds the buckets and clears index_tail. Returns the bytes of the records it dropped.
    u64 BuildIndex(std::vector<IndexEntry> entries, bool sorted);
    bool LoadIndexFile(u64 file_bytes, std::vector<IndexEntry>& entries, u64& covered, u64& dead);
    /// Writes the index file if every record of the file is indexed (map_mutex held).
    void SaveIndexLocked();
    std::vector<IndexEntry> AllIndexEntries() const;
    /// One entry per key (the last record written, as BuildIndex keeps), sorted by file offset: what a rewrite
    /// copies.
    std::vector<IndexEntry> LiveEntriesInFileOrder() const;

    u64 program_id;
    std::filesystem::path path;
    std::filesystem::path index_path;
    std::filesystem::path prebuild_path;   // imported list
    std::filesystem::path generated_path;  // the emulator's own list
    std::filesystem::path selected_list_path;
    std::filesystem::path marker_path;
    std::filesystem::path cache_dir;
    std::filesystem::path lock_path;
#ifdef _WIN32
    void* lock_handle = nullptr;
#else
    int lock_descriptor = -1;
#endif

    mutable std::shared_mutex map_mutex;
    std::vector<IndexEntry> index;                          // records in the mapped file, sorted by mixed
    std::vector<u32> index_buckets;                         // index[buckets[b], buckets[b + 1]) have mixed >> shift == b
    u32 index_shift = 63;
    std::unordered_map<u64, u64> index_tail;                // mixed -> where: records after the saved index's end
    u64 indexed_end = 0;                                    // every record before this file offset is indexed
    u64 dead_bytes = 0;                                     // replaced records in the file
    std::unordered_map<u64, std::vector<u8>> this_session;  // records stored since opening (uncompressed)
    std::vector<IndexEntry> bulk_written;                   // records written in bulk mode, not yet indexed
    std::vector<IndexEntry> bulk_pending;                   // records in `pending` while bulk storing (offset in file)
    bool pending_unindexed = false;                         // `pending` holds a record stored outside bulk mode

    const u8* view = nullptr;
    u64 view_size = 0;
#ifdef _WIN32
    void* file_handle = nullptr;
    void* mapping_handle = nullptr;
#else
    int file_descriptor = -1;
#endif

    std::mutex file_mutex;
    std::vector<u8> pending;  // encoded records not yet appended
    u64 file_size = 0;
    bool writable = true;
    std::atomic<bool> bulk_store{false};
    std::ofstream bulk_stream;  // kept open while bulk storing

    bool started_empty = false;
    bool prebuild_imported = false;
    std::atomic<bool> rebuild_taken{false};
    std::atomic<bool> preload_taken{false};
    bool use_prebuild_list = false;
    bool generate_list = false;

    std::filesystem::path manifest_path;
    std::array<ManifestShard, manifest_shard_count> manifest;
    u32 manifest_session = 1;  // this session's number: sessions in the loaded manifest + 1
    std::atomic<bool> manifest_dirty{false};
    std::array<PlacementBuffer, placement_buffer_count> placements;
    std::mutex manifest_save_mutex;
    std::mutex saver_mutex;
    std::condition_variable_any saver_cv;
    std::jthread manifest_saver;
};

} // namespace Core
