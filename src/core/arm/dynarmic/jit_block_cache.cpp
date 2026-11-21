// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/arm/dynarmic/jit_block_cache.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string_view>
#include <system_error>

#include <fmt/format.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "common/cityhash.h"
#include "common/frame_stats.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/scm_rev.h"
#include "common/lz4_compression.h"
#include "common/settings.h"
#include "core/arm/dynarmic/jit_prebuild_list.h"
#include "core/debugger/nx2_debug_tools.h"

namespace Core {

namespace {

constexpr u32 file_magic = 0x434A324E;  // "N2JC"
constexpr u32 file_version = 5;          // 4: LZ4 records with a checksum, memory-mapped; 5: executable identity
constexpr size_t header_size = 16;       // magic, version, build identity
constexpr size_t record_header_size = 16;  // key, stored size, checksum
constexpr size_t flush_threshold = 1u << 20;
constexpr size_t bulk_flush_threshold = 8u << 20;
// A full static prebuild of a large title is ~2.2 GiB; offsets in the index have 40 bits, sizes 24.
constexpr u64 max_file_size = u64{4} << 30;
constexpr u32 max_record_size = 1u << 20;
static_assert(max_record_size < (1u << 24) && max_file_size < (u64{1} << 40));

constexpr u32 index_magic = 0x494A324E;  // "N2JI"
constexpr u32 index_version = 1;
// magic, version, records file header (16), covered file bytes (8), count (8), dead bytes (8), checksum (8),
// reserved (8)
constexpr size_t index_header_size = 64;
// Index records after the saved index are kept in a small map until there are this many (then merged and saved).
constexpr size_t index_tail_merge = 65536;

/// splitmix64's finalizer: a bijection, so the index can keep it instead of the key; spreads hash-combined keys
/// evenly over the buckets.
u64 Mix(u64 x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

constexpr u64 PackWhere(u64 offset, u64 size) {
    return offset << 24 | size;
}
constexpr u64 WhereOffset(u64 where) {
    return where >> 24;
}
constexpr u32 WhereSize(u64 where) {
    return static_cast<u32>(where & 0xFFFFFF);
}

/// Checksum of a stored record, eight bytes at a time.
u32 Checksum(const u8* data, size_t size) {
    u64 h = 0x9E3779B97F4A7C15ULL ^ size;
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        u64 w;
        std::memcpy(&w, data + i, 8);
        h = (h ^ w) * 0xBF58476D1CE4E5B9ULL;
        h ^= h >> 31;
    }
    for (; i < size; ++i) {
        h = (h ^ data[i]) * 0x94D049BB133111EBULL;
    }
    h ^= h >> 29;
    return static_cast<u32>(h ^ (h >> 32));
}

u64 BuildHash() {
    const std::string_view rev{Common::g_scm_rev};
    const std::string_view name{Common::g_build_fullname};
    u64 h = 0xcbf29ce484222325ULL;
    for (const std::string_view s : {rev, std::string_view{"|"}, name}) {
        for (const char c : s) {
            h ^= static_cast<u8>(c);
            h *= 0x100000001b3ULL;
        }
    }
    return h;
}

template<typename T>
void Put(std::vector<u8>& out, T value) {
    for (size_t i = 0; i < sizeof(T); ++i) {
        out.push_back(static_cast<u8>(static_cast<u64>(value) >> (8 * i)));
    }
}

template<typename T>
T Get(const u8* p) {
    u64 v = 0;
    for (size_t i = 0; i < sizeof(T); ++i) {
        v |= u64{p[i]} << (8 * i);
    }
    return static_cast<T>(v);
}

std::vector<u8> Header() {
    std::vector<u8> h;
    Put<u32>(h, file_magic);
    Put<u32>(h, file_version);
    // The scm strings alone are constant across local rebuilds (they fall back to "stable" when git cannot
    // run), so they do not catch a translator or emitter edit; the executable's own bytes do.
    Put<u64>(h, BuildHash() ^ (JitBlockCache::ExecutableIdentity() * 0x9E3779B97F4A7C15ULL));
    return h;
}

constexpr u32 manifest_magic = 0x4D4A324E;   // "N2JM"
constexpr u32 manifest_version = 1;
constexpr size_t manifest_header_size = 32;  // magic, version, sessions, entry size, count (8), checksum (8)
constexpr size_t manifest_entry_size = 32;   // descriptor, end pc, code hash (8 each), hits, last session (4 each)
constexpr size_t max_manifest_entries = size_t{2} << 20;  // 64 MiB file
constexpr auto manifest_save_interval = std::chrono::seconds{120};
constexpr auto placement_drain_interval = std::chrono::seconds{1};

thread_local u32 t_core = 0xff;

size_t ShardOf(u64 descriptor) {
    return static_cast<size_t>((descriptor * 0x9E3779B97F4A7C15ULL) >> 60);
}

bool CacheEnabled() {
    const char* env = std::getenv("EDEN_JIT_CACHE");
    if (env && std::string_view{env} == "0") {
        return false;
    }
    return Settings::values.cpu_jit_cache.GetValue() != Settings::JitCache::Off;
}

/// Writes a closed file's cached data and size to disk.
bool FlushToDisk(const std::filesystem::path& file, std::error_code& ec) {
#ifdef _WIN32
    HANDLE h = CreateFileW(file.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        ec = std::error_code(static_cast<int>(GetLastError()), std::system_category());
        return false;
    }
    const bool ok = FlushFileBuffers(h) != 0;
    if (!ok) {
        ec = std::error_code(static_cast<int>(GetLastError()), std::system_category());
    }
    CloseHandle(h);
    return ok;
#else
    const int fd = open(file.c_str(), O_RDWR);
    if (fd < 0) {
        ec = std::error_code(errno, std::generic_category());
        return false;
    }
    const bool ok = fsync(fd) == 0;
    if (!ok) {
        ec = std::error_code(errno, std::generic_category());
    }
    close(fd);
    return ok;
#endif
}

}  // namespace

std::filesystem::path JitBlockCache::TempPathFor(const std::filesystem::path& target) {
    static std::atomic<u32> counter{0};
#ifdef _WIN32
    const u64 pid = GetCurrentProcessId();
#else
    const u64 pid = static_cast<u64>(getpid());
#endif
    return std::filesystem::path{target}.concat(fmt::format(".{}-{}.tmp", pid, counter.fetch_add(1)));
}

bool JitBlockCache::ReplaceDurably(const std::filesystem::path& temp, const std::filesystem::path& target,
                                   std::error_code& ec) {
    ec.clear();
    // The data first: NTFS journals the rename but not the data, so without this a power loss after the rename
    // can leave the new name on zero-filled or missing data.
    if (!FlushToDisk(temp, ec)) {
        return false;
    }
#ifdef _WIN32
    if (!MoveFileExW(temp.c_str(), target.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        ec = std::error_code(static_cast<int>(GetLastError()), std::system_category());
        return false;
    }
#else
    if (std::rename(temp.c_str(), target.c_str()) != 0) {
        ec = std::error_code(errno, std::generic_category());
        return false;
    }
    const std::filesystem::path parent = target.has_parent_path() ? target.parent_path() : std::filesystem::path{"."};
    if (const int d = open(parent.c_str(), O_RDONLY); d >= 0) {
        fsync(d);  // the rename itself
        close(d);
    }
#endif
    return true;
}

bool JitBlockCache::AcquireLock() {
#ifdef _WIN32
    // No sharing: a second process's open fails with a sharing violation until this handle is closed (also when
    // this process dies). Not inheritable.
    HANDLE h = CreateFileW(lock_path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        const DWORD error = GetLastError();
        LOG_WARNING(Core_ARM,
                    "JIT block cache {}: {}; this emulator runs without the JIT cache and does not read or write "
                    "its files",
                    path.string(),
                    error == ERROR_SHARING_VIOLATION || error == ERROR_ACCESS_DENIED
                        ? fmt::format("{} is held by another process (another emulator on this program and cache "
                                      "directory)", lock_path.string())
                        : fmt::format("cannot create the lock file {} (error {})", lock_path.string(), error));
        return false;
    }
    lock_handle = h;
#else
    const int fd = open(lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0 || flock(fd, LOCK_EX | LOCK_NB) != 0) {
        const int error = errno;
        if (fd >= 0) {
            close(fd);
        }
        LOG_WARNING(Core_ARM,
                    "JIT block cache {}: {}; this emulator runs without the JIT cache and does not read or write "
                    "its files",
                    path.string(),
                    error == EWOULDBLOCK ? fmt::format("{} is held by another process", lock_path.string())
                                         : fmt::format("cannot lock {} (errno {})", lock_path.string(), error));
        return false;
    }
    lock_descriptor = fd;
#endif
    return true;
}

void JitBlockCache::ReleaseLock() {
#ifdef _WIN32
    if (lock_handle) {
        CloseHandle(static_cast<HANDLE>(lock_handle));
        lock_handle = nullptr;
    }
#else
    if (lock_descriptor >= 0) {
        close(lock_descriptor);
        lock_descriptor = -1;
    }
#endif
}

void JitBlockCache::RemoveStaleTemps() {
    using Name = std::filesystem::path::string_type;
    const Name prefix = std::filesystem::path{fmt::format("{:016X}.", program_id)}.native();
    const Name suffix = std::filesystem::path{".tmp"}.native();
    std::vector<std::filesystem::path> stale;
    std::error_code ec;
    for (auto it = std::filesystem::directory_iterator(cache_dir, ec);
         !ec && it != std::filesystem::directory_iterator(); it.increment(ec)) {
        const Name name = it->path().filename().native();
        if (name.size() > prefix.size() + suffix.size() && name.starts_with(prefix) && name.ends_with(suffix)) {
            stale.push_back(it->path());
        }
    }
    u64 bytes = 0;
    size_t removed = 0;
    for (const auto& p : stale) {
        std::error_code e;
        const u64 size = std::filesystem::file_size(p, e);
        const bool size_known = !e;
        if (std::filesystem::remove(p, e)) {
            ++removed;
            bytes += size_known ? size : 0;
        }
    }
    if (removed != 0) {
        LOG_INFO(Core_ARM, "JIT block cache: removed {} temp files ({} MiB) left by an earlier run", removed,
                 bytes >> 20);
    }
}

bool JitBlockCache::Enabled() {
    return CacheEnabled();
}

bool JitBlockCache::KeepsHostCode() {
    return CacheEnabled() && Settings::values.cpu_jit_cache.GetValue() == Settings::JitCache::TranslationsAndCode;
}

u64 JitBlockCache::StableAslrOffset(u64 program_id) {
    u64 x = program_id + 0x9e3779b97f4a7c15ULL;  // splitmix64
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return (x << 12) & 0xfff000;  // same range as the random offset
}

std::shared_ptr<JitBlockCache> JitBlockCache::Open(u64 program_id) {
    if (!CacheEnabled() || program_id == 0) {
        return nullptr;
    }
    std::error_code ec;
    // EDEN_JIT_CACHE_DIR: measurements that must not share (or rewrite) the profile's cache file.
    const char* const own_dir = std::getenv("EDEN_JIT_CACHE_DIR");
    const auto dir = own_dir != nullptr && own_dir[0] != '\0'
                         ? std::filesystem::path{own_dir}
                         : Common::FS::GetEdenPath(Common::FS::EdenPath::CacheDir) / "jit";
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        LOG_WARNING(Core_ARM, "JIT block cache disabled: cannot create {}: {}", dir.string(), ec.message());
        return nullptr;
    }
    auto cache = std::shared_ptr<JitBlockCache>(new JitBlockCache(program_id, dir));
    // Only one process may use (and rewrite) a program's cache files; see the class comment.
    if (!cache->AcquireLock()) {
        return nullptr;
    }
    cache->RemoveStaleTemps();
    // EDEN_JIT_CACHE_PREBUILD_IMPORT=<file>: a prebuild list made offline becomes this program's list.
    if (const char* import = std::getenv("EDEN_JIT_CACHE_PREBUILD_IMPORT"); import && import[0] != '\0') {
        bool changed = false;
        if (JitPrebuildList::Import(std::filesystem::path{import}, cache->prebuild_path, program_id, changed)) {
            cache->prebuild_imported = changed;
            LOG_INFO(Core_ARM, "JIT prebuild list {}: {}", import,
                     changed ? fmt::format("imported to {}", cache->prebuild_path.string()) : std::string{"already imported"});
        } else {
            LOG_WARNING(Core_ARM, "JIT prebuild list {}: not imported (missing, unreadable, or not a list for program {:016X})",
                        import, program_id);
        }
    }
    using Clock = std::chrono::steady_clock;
    const auto open_start = Clock::now();
    cache->OpenFile();
    const auto open_ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - open_start).count();
    cache->started_empty = cache->RecordCount() == 0;
    cache->LoadManifest();
    cache->StartManifestSaver();
    LOG_INFO(Core_ARM, "JIT block cache {}: {} blocks ({} MiB){}; executable identity {:016X}; manifest {} blocks, session {}",
             cache->path.string(), cache->RecordCount(), cache->file_size >> 20, KeepsHostCode() ? ", with x64 code" : "",
             ExecutableIdentity(), cache->ManifestSize(), cache->manifest_session);
    LOG_INFO(Core_ARM, "JIT block cache index: {} records, {} after the saved index, {} MiB in memory; opened in {} ms",
             cache->index.size(), cache->index_tail.size(), cache->IndexBytes() >> 20, open_ms);
    return cache;
}

JitBlockCache::JitBlockCache(u64 program_id_, const std::filesystem::path& dir)
        : program_id(program_id_), path(dir / fmt::format("{:016X}.bin", program_id_)),
          index_path(dir / fmt::format("{:016X}.bin.idx", program_id_)),
          prebuild_path(dir / fmt::format("{:016X}.prebuild", program_id_)),
          generated_path(dir / fmt::format("{:016X}.generated.prebuild", program_id_)),
          selected_list_path(prebuild_path),
          marker_path(dir / fmt::format("{:016X}.prebuilt", program_id_)), cache_dir(dir),
          lock_path(dir / fmt::format("{:016X}.lock", program_id_)),
          manifest_path(dir / fmt::format("{:016X}.manifest", program_id_)) {}

JitBlockCache::~JitBlockCache() {
#ifdef _WIN32
    const bool locked = lock_handle != nullptr;
#else
    const bool locked = lock_descriptor >= 0;
#endif
    if (manifest_saver.joinable()) {
        manifest_saver.request_stop();
        manifest_saver.join();
    }
    if (locked) {
        SaveManifest();
        Flush();
    }
    UnmapFile();
    ReleaseLock();
}

void JitBlockCache::OpenFile() {
    std::error_code ec;
    const std::vector<u8> header = Header();
    u64 size = std::filesystem::exists(path, ec) ? std::filesystem::file_size(path, ec) : 0;
    if (ec) {
        size = 0;
    }
    index.clear();
    index_buckets.clear();
    index_tail.clear();
    dead_bytes = 0;

    const bool valid = size >= header_size && MapFile(size) && std::memcmp(view, header.data(), header_size) == 0;
    if (!valid) {
        if (size != 0) {
            LOG_INFO(Core_ARM, "JIT block cache {} is from another build or damaged; starting over", path.string());
        }
        WriteEmptyFile();
        return;
    }

    // The saved index covers the file up to `covered`; only record headers after it are read.
    std::vector<IndexEntry> saved;
    u64 covered = header_size;
    u64 saved_dead = 0;
    const bool have_saved = LoadIndexFile(size, saved, covered, saved_dead);
    if (!have_saved) {
        covered = header_size;
        saved.clear();
        saved_dead = 0;
    }
    std::vector<IndexEntry> scanned;
    u64 pos = covered;
    while (pos + record_header_size <= view_size) {
        const u64 key = Get<u64>(view + pos);
        const u32 stored = Get<u32>(view + pos + 8);
        if (stored == 0 || stored > max_record_size || pos + record_header_size + stored > view_size) {
            break;
        }
        scanned.push_back({Mix(key), PackWhere(pos + record_header_size, stored)});
        pos += record_header_size + stored;
    }
    const u64 good = pos;

    bool save_index = false;
    if (have_saved && scanned.size() <= std::max(index_tail_merge, saved.size() / 8)) {
        dead_bytes = saved_dead + BuildIndex(std::move(saved), true);
        for (const IndexEntry& e : scanned) {
            const auto it = index_tail.find(e.mixed);
            if (it != index_tail.end()) {
                dead_bytes += WhereSize(it->second);
                it->second = e.where;
                continue;
            }
            const u64 b = e.mixed >> index_shift;
            for (u32 i = index_buckets[b]; i < index_buckets[b + 1]; ++i) {
                if (index[i].mixed == e.mixed) {
                    dead_bytes += WhereSize(index[i].where);
                    break;
                }
            }
            index_tail.emplace(e.mixed, e.where);
        }
    } else {
        // No usable saved index, or many records after it: index everything and save the index.
        std::vector<IndexEntry> entries = std::move(saved);
        entries.insert(entries.end(), scanned.begin(), scanned.end());
        scanned = {};
        dead_bytes = saved_dead + BuildIndex(std::move(entries), false);
        save_index = true;
    }
    indexed_end = good;

    if (good != size) {
        // A torn record at the end (a crash while appending): cut it off. Records before it stay where they are.
        UnmapFile();
        std::filesystem::resize_file(path, good, ec);
        if (ec || !MapFile(good)) {
            LOG_WARNING(Core_ARM, "JIT block cache {}: cannot drop a damaged tail; disabled", path.string());
            index.clear();
            index_buckets.clear();
            index_tail.clear();
            writable = false;
            file_size = 0;
            return;
        }
        LOG_INFO(Core_ARM, "JIT block cache {}: dropped a damaged tail of {} bytes", path.string(), size - good);
        size = good;
    }

    const u64 live_bytes = good - header_size - std::min(dead_bytes, good - header_size);
    if (dead_bytes > live_bytes && dead_bytes > (u64{16} << 20)) {
        const u64 removed = dead_bytes;
        if (CompactFile()) {
            LOG_INFO(Core_ARM, "JIT block cache {}: compacted, {} MiB of replaced records removed", path.string(), removed >> 20);
            OpenFile();  // index the rewritten file
            return;
        }
        if (!view) {
            LOG_WARNING(Core_ARM, "JIT block cache {}: cannot map the file again after a failed compaction; disabled",
                        path.string());
            index.clear();
            index_buckets.clear();
            index_tail.clear();
            writable = false;
            file_size = 0;
            return;
        }
        save_index = true;  // the saved index may be gone (it is removed before the rename)
    }
    file_size = size;
    if (save_index && !index.empty()) {
        SaveIndexLocked();
    }
}

void JitBlockCache::WriteEmptyFile() {
    std::error_code ec;
    UnmapFile();
    std::filesystem::remove(index_path, ec);
    index.clear();
    index_buckets.clear();
    index_tail.clear();
    dead_bytes = 0;
    const std::vector<u8> header = Header();
    const auto temp = TempPathFor(path);
    {
        std::ofstream file(temp, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(header.size()));
        file.close();
        writable = !file.fail();
    }
    if (!writable || !ReplaceDurably(temp, path, ec)) {
        std::error_code ignored;
        std::filesystem::remove(temp, ignored);
        LOG_WARNING(Core_ARM, "JIT block cache {} could not be written{}; disabled", path.string(),
                    ec ? fmt::format(" ({})", ec.message()) : std::string{});
        writable = false;
        file_size = 0;
        return;
    }
    file_size = header.size();
    indexed_end = file_size;
}

bool JitBlockCache::CompactFile() {
    // Live records in file order, streamed to a new file (never the whole file in memory).
    const std::vector<IndexEntry> live = LiveEntriesInFileOrder();
    const auto temp = TempPathFor(path);
    bool ok;
    {
        std::ofstream file(temp, std::ios::binary | std::ios::trunc);
        const std::vector<u8> header = Header();
        file.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(header.size()));
        for (const IndexEntry& e : live) {
            const u64 offset = WhereOffset(e.where);
            // The record header (key, size, checksum) is the 16 bytes before the stored bytes.
            file.write(reinterpret_cast<const char*>(view + offset - record_header_size),
                       static_cast<std::streamsize>(record_header_size + WhereSize(e.where)));
        }
        file.close();
        ok = !file.fail();
    }
    std::error_code ec;
    if (!ok) {
        std::filesystem::remove(temp, ec);
        return false;
    }
    const u64 old_size = view_size;
    UnmapFile();
    std::filesystem::remove(index_path, ec);
    if (!ReplaceDurably(temp, path, ec)) {
        LOG_WARNING(Core_ARM, "JIT block cache {}: cannot replace the file with the compacted one ({}); old file kept",
                    path.string(), ec.message());
        std::error_code ignored;
        std::filesystem::remove(temp, ignored);
        MapFile(old_size);  // the caller checks `view`
        return false;
    }
    return true;
}

std::optional<u64> JitBlockCache::DropRecords(const std::function<bool(u64 key)>& keep, u64 min_bytes,
                                              std::string& what) {
    std::unique_lock map_lock{map_mutex};
    std::scoped_lock file_lock{file_mutex};
    if (!writable || !view || bulk_store.load()) {
        what = "records file not open for writing";
        return std::nullopt;
    }
    FlushLocked();
    if (!this_session.empty() || pending_unindexed || indexed_end != file_size || !bulk_written.empty()) {
        what = "records stored this session are not all indexed yet";
        return std::nullopt;
    }
    // Live records in file order; the kept ones are moved to the front in place (no second list).
    std::vector<IndexEntry> kept = LiveEntriesInFileOrder();
    u64 drop_bytes = 0;
    size_t dropped = 0;
    size_t out = 0;
    for (size_t i = 0; i < kept.size(); ++i) {
        const IndexEntry e = kept[i];
        const u64 offset = WhereOffset(e.where);
        if (offset < header_size + record_header_size || offset + WhereSize(e.where) > view_size) {
            what = "index points outside the file";
            return std::nullopt;
        }
        // The record header (key, size, checksum) is the 16 bytes before the stored bytes.
        const u64 key = Get<u64>(view + offset - record_header_size);
        if (Mix(key) != e.mixed || Get<u32>(view + offset - record_header_size + 8) != WhereSize(e.where)) {
            what = "index does not match the file";
            return std::nullopt;
        }
        if (keep(key)) {
            kept[out++] = e;
        } else {
            drop_bytes += record_header_size + WhereSize(e.where);
            ++dropped;
        }
    }
    kept.resize(out);
    if (drop_bytes < min_bytes) {
        what = fmt::format("{} records ({} MiB) would go, below the {} MiB threshold; file kept", dropped,
                           drop_bytes >> 20, min_bytes >> 20);
        return u64{0};
    }
    // Kept records in file order to a temp file; their new places become the index (rewritten in `kept`).
    const auto temp = TempPathFor(path);
    bool ok;
    u64 new_size = 0;
    {
        std::ofstream file(temp, std::ios::binary | std::ios::trunc);
        const std::vector<u8> header = Header();
        file.write(reinterpret_cast<const char*>(header.data()), static_cast<std::streamsize>(header.size()));
        u64 at = header.size();
        for (IndexEntry& e : kept) {
            const u64 offset = WhereOffset(e.where);
            const u64 bytes = record_header_size + WhereSize(e.where);
            file.write(reinterpret_cast<const char*>(view + offset - record_header_size),
                       static_cast<std::streamsize>(bytes));
            e.where = PackWhere(at + record_header_size, WhereSize(e.where));
            at += bytes;
        }
        file.close();
        ok = !file.fail();
        new_size = at;
    }
    std::error_code ec;
    if (!ok) {
        std::filesystem::remove(temp, ec);
        what = "cannot write the compacted file; old file kept";
        return std::nullopt;
    }
    const u64 old_size = file_size;
    UnmapFile();
    // The saved index goes first: a crash between the two steps then leaves a file that is fully scanned on
    // open, never an index that describes another file.
    std::filesystem::remove(index_path, ec);
    if (!ReplaceDurably(temp, path, ec)) {
        // The old file is still in place (another process may hold it): map it again and go on with it.
        std::error_code ignored;
        std::filesystem::remove(temp, ignored);
        if (!MapFile(old_size)) {
            index.clear();
            index_buckets.clear();
            index_tail.clear();
            writable = false;
            file_size = 0;
            what = fmt::format("cannot replace the file ({}) nor map it again; cache disabled", ec.message());
            return std::nullopt;
        }
        SaveIndexLocked();  // the index in memory still describes the old file
        what = fmt::format("cannot replace the file ({}); old file kept", ec.message());
        return std::nullopt;
    }
    index = {};
    const size_t kept_count = kept.size();
    dead_bytes = BuildIndex(std::move(kept), false);
    file_size = new_size;
    indexed_end = new_size;
    if (!MapFile(new_size)) {
        index.clear();
        index_buckets.clear();
        index_tail.clear();
        writable = false;
        file_size = 0;
        what = "cannot map the compacted file; cache disabled";
        return std::nullopt;
    }
    if (!index.empty()) {
        SaveIndexLocked();
    }
    what = fmt::format("{} records ({} MiB) removed, {} kept; {} MiB -> {} MiB", dropped, drop_bytes >> 20,
                       kept_count, old_size >> 20, new_size >> 20);
    return drop_bytes;
}

std::optional<JitBlockCache::Location> JitBlockCache::FindOnDisk(u64 key) const {
    const u64 m = Mix(key);
    if (!index_tail.empty()) {
        if (const auto it = index_tail.find(m); it != index_tail.end()) {
            return Location{WhereOffset(it->second), WhereSize(it->second)};
        }
    }
    if (index.empty()) {
        return std::nullopt;
    }
    const u64 b = m >> index_shift;
    for (u32 i = index_buckets[b], end = index_buckets[b + 1]; i < end; ++i) {
        if (index[i].mixed == m) {
            return Location{WhereOffset(index[i].where), WhereSize(index[i].where)};
        }
    }
    return std::nullopt;
}

u64 JitBlockCache::BuildIndex(std::vector<IndexEntry> entries, bool sorted) {
    if (!sorted) {
        std::sort(entries.begin(), entries.end(), [](const IndexEntry& a, const IndexEntry& b) {
            return a.mixed != b.mixed ? a.mixed < b.mixed : a.where < b.where;
        });
    }
    // One entry per key: a later record (higher offset) replaces an earlier one.
    u64 dropped = 0;
    size_t out = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        if (out != 0 && entries[out - 1].mixed == entries[i].mixed) {
            dropped += WhereSize(entries[out - 1].where);
            entries[out - 1] = entries[i];
        } else {
            entries[out++] = entries[i];
        }
    }
    entries.resize(out);
    if (entries.capacity() > out + out / 8 + 1024) {
        entries.shrink_to_fit();
    }
    // About four records per bucket.
    const u32 bits = std::clamp<u32>(static_cast<u32>(std::bit_width(out / 4)), 4, 24);
    index_shift = 64 - bits;
    index_buckets.assign((size_t{1} << bits) + 1, 0);
    for (const IndexEntry& e : entries) {
        ++index_buckets[(e.mixed >> index_shift) + 1];
    }
    for (size_t b = 1; b < index_buckets.size(); ++b) {
        index_buckets[b] += index_buckets[b - 1];
    }
    index = std::move(entries);
    index_tail.clear();
    return dropped;
}

std::vector<JitBlockCache::IndexEntry> JitBlockCache::LiveEntriesInFileOrder() const {
    std::vector<IndexEntry> all = AllIndexEntries();
    // A record after the saved index can replace one the index lists (OpenFile keeps both): keep the later one.
    std::sort(all.begin(), all.end(), [](const IndexEntry& a, const IndexEntry& b) {
        return a.mixed != b.mixed ? a.mixed < b.mixed : a.where < b.where;
    });
    size_t out = 0;
    for (size_t i = 0; i < all.size(); ++i) {
        if (out != 0 && all[out - 1].mixed == all[i].mixed) {
            all[out - 1] = all[i];
        } else {
            all[out++] = all[i];
        }
    }
    all.resize(out);
    std::sort(all.begin(), all.end(), [](const IndexEntry& a, const IndexEntry& b) { return a.where < b.where; });
    return all;
}

std::vector<JitBlockCache::IndexEntry> JitBlockCache::AllIndexEntries() const {
    std::vector<IndexEntry> all;
    all.reserve(index.size() + index_tail.size());
    all.insert(all.end(), index.begin(), index.end());
    for (const auto& [mixed, where] : index_tail) {
        all.push_back({mixed, where});
    }
    return all;
}

// Index file (<program id>.bin.idx), little-endian:
//   header (64 bytes): magic "N2JI", version (4), the records file's 16-byte header (so it only matches files of
//   this executable), covered (8): file bytes whose records it lists, count (8), dead bytes (8): replaced records
//   in that part, CityHash64 of the entries (8), reserved (8);
//   entries (16 bytes each), sorted: Mix(key) (8), offset << 24 | size of the stored bytes (8).
// Written to a temp file and renamed (ReplaceDurably). Removed whenever the records file is rewritten (new, compacted); a cut-off
// damaged tail lies after `covered`. Loading checks the header, size and checksum, and a sample of entries
// against the record headers in the file.
bool JitBlockCache::LoadIndexFile(u64 file_bytes, std::vector<IndexEntry>& entries, u64& covered, u64& dead) {
    static_assert(sizeof(IndexEntry) == 16);
    if constexpr (std::endian::native != std::endian::little) {
        return false;
    }
    std::error_code ec;
    const u64 size = std::filesystem::exists(index_path, ec) ? std::filesystem::file_size(index_path, ec) : 0;
    if (ec || size < index_header_size) {
        return false;
    }
    std::ifstream file(index_path, std::ios::binary);
    std::array<u8, index_header_size> h{};
    file.read(reinterpret_cast<char*>(h.data()), index_header_size);
    if (!file) {
        return false;
    }
    const std::vector<u8> header = Header();
    covered = Get<u64>(h.data() + 24);
    const u64 count = Get<u64>(h.data() + 32);
    dead = Get<u64>(h.data() + 40);
    const u64 checksum = Get<u64>(h.data() + 48);
    const bool header_ok = Get<u32>(h.data()) == index_magic && Get<u32>(h.data() + 4) == index_version &&
                           std::memcmp(h.data() + 8, header.data(), header_size) == 0 && covered >= header_size &&
                           covered <= file_bytes && count <= (size - index_header_size) / sizeof(IndexEntry) &&
                           size == index_header_size + count * sizeof(IndexEntry);
    if (!header_ok) {
        LOG_INFO(Core_ARM, "JIT block cache index {} does not match the records file; reading record headers",
                 index_path.string());
        return false;
    }
    entries.resize(static_cast<size_t>(count));
    file.read(reinterpret_cast<char*>(entries.data()), static_cast<std::streamsize>(count * sizeof(IndexEntry)));
    if (!file || Common::CityHash64(reinterpret_cast<const char*>(entries.data()), count * sizeof(IndexEntry)) != checksum) {
        LOG_INFO(Core_ARM, "JIT block cache index {} is damaged; reading record headers", index_path.string());
        return false;
    }
    // Sample: entries must point at record headers of the same key and size inside the covered part.
    const size_t step = std::max<size_t>(1, entries.size() / 64);
    for (size_t i = 0; i < entries.size(); i += step) {
        const u64 offset = WhereOffset(entries[i].where);
        const u32 stored = WhereSize(entries[i].where);
        if (offset < header_size + record_header_size || offset + stored > covered ||
            Mix(Get<u64>(view + offset - record_header_size)) != entries[i].mixed ||
            Get<u32>(view + offset - record_header_size + 8) != stored ||
            (i != 0 && entries[i - 1].mixed >= entries[i].mixed)) {
            LOG_INFO(Core_ARM, "JIT block cache index {} does not match the records file; reading record headers",
                     index_path.string());
            return false;
        }
    }
    return true;
}

void JitBlockCache::SaveIndexLocked() {
    if (indexed_end != file_size || !writable) {
        return;  // records written outside bulk mode are not indexed: the next open reads their headers
    }
    if (!index_tail.empty()) {
        dead_bytes += BuildIndex(AllIndexEntries(), false);
    }
    std::vector<u8> h;
    Put<u32>(h, index_magic);
    Put<u32>(h, index_version);
    const std::vector<u8> header = Header();
    h.insert(h.end(), header.begin(), header.end());
    Put<u64>(h, indexed_end);
    Put<u64>(h, index.size());
    Put<u64>(h, dead_bytes);
    Put<u64>(h, Common::CityHash64(reinterpret_cast<const char*>(index.data()), index.size() * sizeof(IndexEntry)));
    Put<u64>(h, 0);
    const auto temp = TempPathFor(index_path);
    bool ok;
    {
        std::ofstream file(temp, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(h.data()), static_cast<std::streamsize>(h.size()));
        file.write(reinterpret_cast<const char*>(index.data()), static_cast<std::streamsize>(index.size() * sizeof(IndexEntry)));
        file.close();
        ok = !file.fail();
    }
    std::error_code ec;
    if (!ok || !ReplaceDurably(temp, index_path, ec)) {
        std::filesystem::remove(temp, ec);
        LOG_WARNING(Core_ARM, "JIT block cache index {} could not be written", index_path.string());
        return;
    }
    LOG_INFO(Core_ARM, "JIT block cache index {}: {} records written", index_path.string(), index.size());
}

bool JitBlockCache::MapFile(u64 size) {
    UnmapFile();
#ifdef _WIN32
    HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                              nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return false;
    }
    HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping) {
        CloseHandle(file);
        return false;
    }
    void* p = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (!p) {
        CloseHandle(mapping);
        CloseHandle(file);
        return false;
    }
    file_handle = file;
    mapping_handle = mapping;
    view = static_cast<const u8*>(p);
#else
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;
    }
    void* p = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        close(fd);
        return false;
    }
    file_descriptor = fd;
    view = static_cast<const u8*>(p);
#endif
    view_size = size;
    return true;
}

void JitBlockCache::UnmapFile() {
    if (!view) {
        return;
    }
#ifdef _WIN32
    UnmapViewOfFile(view);
    CloseHandle(static_cast<HANDLE>(mapping_handle));
    CloseHandle(static_cast<HANDLE>(file_handle));
    mapping_handle = nullptr;
    file_handle = nullptr;
#else
    munmap(const_cast<u8*>(view), view_size);
    close(file_descriptor);
    file_descriptor = -1;
#endif
    view = nullptr;
    view_size = 0;
}

std::optional<std::vector<u8>> JitBlockCache::Load(u64 key) {
    std::shared_lock lock{map_mutex};
    if (const auto it = this_session.find(key); it != this_session.end()) {
        return it->second;
    }
    const auto location = FindOnDisk(key);
    if (!location || location->offset + location->size > view_size) {
        return std::nullopt;
    }
    // Stored: uncompressed size (4 bytes), then the LZ4 data; the checksum is the last field of the record header.
    const u8* stored = view + location->offset;
    if (location->size <= 4 || Checksum(stored, location->size) != Get<u32>(stored - 4)) {
        return std::nullopt;
    }
    const u32 size = Get<u32>(stored);
    if (size == 0 || size > max_record_size) {
        return std::nullopt;
    }
    std::vector<u8> record(size);
    const int result = Common::Compression::DecompressDataLZ4(record.data(), record.size(), stored + 4, location->size - 4);
    if (result != static_cast<int>(size)) {
        return std::nullopt;
    }
    return record;
}

void JitBlockCache::Store(u64 key, std::vector<u8> record) {
    if (record.empty() || record.size() > max_record_size) {
        return;
    }
    const std::vector<u8> lz4 = Common::Compression::CompressDataLZ4(record.data(), record.size());
    if (lz4.empty() || lz4.size() + 4 > max_record_size) {
        return;
    }
    std::vector<u8> compressed;
    compressed.reserve(4 + lz4.size());
    Put<u32>(compressed, static_cast<u32>(record.size()));
    compressed.insert(compressed.end(), lz4.begin(), lz4.end());
    const bool bulk = bulk_store.load(std::memory_order_relaxed);
    if (!bulk) {
        std::unique_lock lock{map_mutex};
        this_session[key] = std::move(record);
    }
    const u32 checksum = Checksum(compressed.data(), compressed.size());
    std::scoped_lock lock{file_mutex};
    if (!writable) {
        return;
    }
    if (bulk) {
        // Where it will land: pending is appended at file_size.
        bulk_pending.push_back({Mix(key), PackWhere(file_size + pending.size() + record_header_size, compressed.size())});
    } else {
        pending_unindexed = true;
    }
    Put<u64>(pending, key);
    Put<u32>(pending, static_cast<u32>(compressed.size()));
    Put<u32>(pending, checksum);
    pending.insert(pending.end(), compressed.begin(), compressed.end());
    if (pending.size() >= (bulk ? bulk_flush_threshold : flush_threshold)) {
        FlushLocked();
    }
}

void JitBlockCache::Flush() {
    std::scoped_lock lock{file_mutex};
    FlushLocked();
}

void JitBlockCache::FlushLocked() {
    if (pending.empty() || !writable) {
        return;
    }
    if (file_size + pending.size() > max_file_size) {
        LOG_INFO(Core_ARM, "JIT block cache {} reached {} MiB; no longer growing", path.string(), max_file_size >> 20);
        writable = false;
        pending.clear();
        bulk_pending.clear();
        return;
    }
    bool ok;
    if (bulk_stream.is_open()) {
        bulk_stream.write(reinterpret_cast<const char*>(pending.data()), static_cast<std::streamsize>(pending.size()));
        ok = static_cast<bool>(bulk_stream);
    } else {
        std::ofstream out(path, std::ios::binary | std::ios::app);
        out.write(reinterpret_cast<const char*>(pending.data()), static_cast<std::streamsize>(pending.size()));
        ok = static_cast<bool>(out);
    }
    if (!ok) {
        LOG_WARNING(Core_ARM, "JIT block cache {}: write failed; no longer writing", path.string());
        writable = false;
    } else {
        bulk_written.insert(bulk_written.end(), bulk_pending.begin(), bulk_pending.end());
        if (indexed_end == file_size && !pending_unindexed) {
            indexed_end = file_size + pending.size();
        }
        file_size += pending.size();
    }
    pending.clear();
    bulk_pending.clear();
    pending_unindexed = false;
}

void JitBlockCache::SetThreadCore(u32 core) {
    t_core = core;
}

void JitBlockCache::OnBlockCompiled(CompileKind kind, u64 ns) {
    using FS = Common::FrameStats::Cause;
    using FC = Common::FrameStats::Counter;
    if (Common::FrameStats::Enabled()) {
        switch (kind) {
        case CompileKind::Translated:
            Common::FrameStats::AddStall(FS::JitTranslate, ns);
            Common::FrameStats::AddCount(FC::JitTranslate);
            break;
        case CompileKind::Native:
            Common::FrameStats::AddStall(FS::JitNative, ns);
            Common::FrameStats::AddCount(FC::JitNative);
            break;
        case CompileKind::Ir:
            Common::FrameStats::AddStall(FS::JitIr, ns);
            Common::FrameStats::AddCount(FC::JitIr);
            break;
        }
    }
    if (Nx2Debug::SchedTrace::enabled.load(std::memory_order_relaxed)) {
        const auto end = std::chrono::steady_clock::now();
        // arg: compile kind, guest core << 8.
        Nx2Debug::SchedTrace::RecordHostSpan(Nx2Debug::SchedTrace::HostSpanKind::JitCompile,
                                             end - std::chrono::nanoseconds{ns}, end,
                                             static_cast<u64>(kind) | (u64{t_core} << 8));
    }
}

size_t JitBlockCache::RecordCount() const {
    std::shared_lock lock{map_mutex};
    return index.size() + index_tail.size() + this_session.size();
}

u64 JitBlockCache::FileSize() const {
    return file_size;
}

size_t JitBlockCache::IndexBytes() const {
    return index.capacity() * sizeof(IndexEntry) + index_buckets.capacity() * sizeof(u32) + index_tail.size() * 48;
}

u64 JitBlockCache::ExecutableIdentity() {
    static const u64 identity = [] {
        std::filesystem::path exe;
#ifdef _WIN32
        HMODULE module = nullptr;
        // The module that holds this code (the JIT is linked into the same executable).
        if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCWSTR>(&JitBlockCache::ExecutableIdentity), &module)) {
            std::vector<wchar_t> name(32768);
            const DWORD length = GetModuleFileNameW(module, name.data(), static_cast<DWORD>(name.size()));
            if (length != 0 && length < name.size()) {
                exe = std::filesystem::path{std::wstring{name.data(), length}};
            }
        }
#else
        std::error_code ec;
        exe = std::filesystem::read_symlink("/proc/self/exe", ec);
#endif
        u64 h = 0;
        bool ok = false;
        if (!exe.empty()) {
            std::ifstream file(exe, std::ios::binary);
            std::vector<char> chunk(4u << 20);
            u64 total = 0;
            while (file) {
                file.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
                const auto got = static_cast<size_t>(file.gcount());
                if (got == 0) {
                    break;
                }
                h = Common::CityHash64WithSeed(chunk.data(), got, h);
                total += got;
            }
            ok = file.eof() && total != 0;
            h ^= total;
        }
        if (!ok) {
            // Unknown executable: never reuse a cache (a value no other run produces).
            h = static_cast<u64>(std::chrono::steady_clock::now().time_since_epoch().count()) | 1;
            LOG_WARNING(Core_ARM, "JIT block cache: cannot read the executable; cached records will not be reused");
        }
        return h;
    }();
    return identity;
}

bool JitBlockCache::Contains(u64 key) {
    std::shared_lock lock{map_mutex};
    return this_session.contains(key) || FindOnDisk(key).has_value();
}

void JitBlockCache::OnBlockPlaced(u64 descriptor, u64 end_pc, u64 code_hash) {
    // On the guest core's thread at every placement: an append only (a map update there cost ~0.25 us per
    // placement, ~0.4 s over a warm OoT session).
    auto& buffer = placements[std::min<size_t>(t_core, placement_buffer_count - 1)];
    std::scoped_lock lock{buffer.mutex};
    buffer.items.push_back({descriptor, end_pc, code_hash});
}

void JitBlockCache::DrainPlacements() {
    std::vector<Placement> items;
    for (auto& buffer : placements) {
        {
            std::scoped_lock lock{buffer.mutex};
            items.swap(buffer.items);
            buffer.items.reserve(items.size());
        }
        for (const auto& p : items) {
            auto& shard = manifest[ShardOf(p.descriptor)];
            std::scoped_lock lock{shard.mutex};
            auto& v = shard.blocks.try_emplace(p.descriptor, ManifestValue{p.end_pc, p.code_hash, 0, manifest_session})
                          .first->second;
            if (v.end_pc != p.end_pc || v.code_hash != p.code_hash) {
                v = ManifestValue{p.end_pc, p.code_hash, 0, manifest_session};  // the guest code changed: a new block
            }
            if (v.hits != UINT32_MAX) {
                ++v.hits;
            }
            v.last_session = manifest_session;
        }
        if (!items.empty()) {
            manifest_dirty.store(true, std::memory_order_relaxed);
        }
        items.clear();
    }
}

std::vector<JitBlockCache::ManifestEntry> JitBlockCache::ManifestSnapshot() {
    DrainPlacements();
    std::vector<ManifestEntry> out;
    out.reserve(ManifestSize());
    for (const auto& shard : manifest) {
        std::scoped_lock lock{shard.mutex};  // one shard at a time: cores placing blocks wait briefly
        for (const auto& [descriptor, v] : shard.blocks) {
            out.push_back({descriptor, v.end_pc, v.code_hash, v.hits, v.last_session});
        }
    }
    std::sort(out.begin(), out.end(), [](const ManifestEntry& a, const ManifestEntry& b) {
        return a.hits != b.hits ? a.hits > b.hits : a.descriptor < b.descriptor;
    });
    return out;
}

size_t JitBlockCache::ManifestSize() const {
    size_t n = 0;
    for (const auto& shard : manifest) {
        std::scoped_lock lock{shard.mutex};
        n += shard.blocks.size();
    }
    return n;
}

size_t JitBlockCache::AddResidentBlocks(std::span<const ManifestEntry> blocks) {
    size_t added = 0;
    for (const auto& b : blocks) {
        auto& shard = manifest[ShardOf(b.descriptor)];
        std::scoped_lock lock{shard.mutex};
        // Hits 0: built, not seen placed. A real placement later counts from here (or, if the translator ended
        // the block elsewhere, replaces the entry with the placed end and hash).
        if (shard.blocks.try_emplace(b.descriptor, ManifestValue{b.end_pc, b.code_hash, 0, manifest_session}).second) {
            ++added;
        }
    }
    if (added != 0) {
        manifest_dirty.store(true, std::memory_order_relaxed);
    }
    return added;
}

JitBlockCache::PrebuildScope JitBlockCache::GetPrebuildScope() {
    const char* env = std::getenv("EDEN_JIT_CACHE_PREBUILD_SCOPE");
    const std::string_view v = env ? std::string_view{env} : std::string_view{};
    if (v == "executed") {
        return PrebuildScope::Executed;
    }
    if (v == "played") {
        return PrebuildScope::Played;
    }
    return PrebuildScope::Full;
}

const char* JitBlockCache::PrebuildScopeName(PrebuildScope scope) {
    switch (scope) {
    case PrebuildScope::Executed:
        return "executed";
    case PrebuildScope::Played:
        return "played";
    case PrebuildScope::Full:
        break;
    }
    return "full";
}

namespace {
// Marker <program id>.prebuilt: the prebuild list's 48-byte header, the scope (4) and the executable identity (8)
// the records file was last built for.
std::vector<u8> PrebuiltMarker(std::span<const u8> list_header, JitBlockCache::PrebuildScope scope) {
    std::vector<u8> m(list_header.begin(), list_header.end());
    Put<u32>(m, static_cast<u32>(scope));
    Put<u64>(m, JitBlockCache::ExecutableIdentity());
    return m;
}
}  // namespace

void JitBlockCache::MarkPrebuilt(std::span<const u8> list_header, PrebuildScope scope) {
    // The marker says the records file holds the build: it may only reach the disk after those records did, or a
    // power loss could leave a marker that stops the next boot from building records the file lost.
    std::error_code ec;
    {
        std::scoped_lock lock{file_mutex};
        FlushLocked();
        if (!writable || !FlushToDisk(path, ec)) {
            LOG_WARNING(Core_ARM, "JIT block cache {}: records not flushed to disk{}; build marker not written",
                        path.string(), ec ? fmt::format(" ({})", ec.message()) : std::string{});
            return;
        }
    }
    const std::vector<u8> m = PrebuiltMarker(list_header, scope);
    const auto temp = TempPathFor(marker_path);
    bool ok;
    {
        std::ofstream file(temp, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(m.data()), static_cast<std::streamsize>(m.size()));
        file.close();
        ok = !file.fail();
    }
    if (!ok || !ReplaceDurably(temp, marker_path, ec)) {
        std::error_code ignored;
        std::filesystem::remove(temp, ignored);
        LOG_WARNING(Core_ARM, "JIT block cache build marker {} could not be written", marker_path.string());
    }
}

JitBlockCache::PrecompileMode JitBlockCache::GetPrecompileMode(const char** source) {
    // See the header for the precedence: the environment variables (test scripts) win over the setting (GUI).
    const auto env_set = [](const char* name) {
        const char* v = std::getenv(name);
        return v != nullptr && v[0] != '\0';
    };
    const char* rebuild = std::getenv("EDEN_JIT_CACHE_REBUILD");
    const char* from = "setting";
    PrecompileMode mode = Settings::values.cpu_jit_precompile.GetValue();
    if (rebuild != nullptr && std::string_view{rebuild} == "0") {
        mode = PrecompileMode::Off;
        from = "EDEN_JIT_CACHE_REBUILD=0";
    } else if (env_set("EDEN_JIT_CACHE_PREBUILD_IMPORT") || env_set("EDEN_JIT_CACHE_PREBUILD_SCOPE")) {
        mode = PrecompileMode::EntireGame;
        from = "EDEN_JIT_CACHE_PREBUILD_IMPORT/_SCOPE";
    } else if (rebuild != nullptr && rebuild[0] != '\0' && mode == PrecompileMode::Off) {
        mode = PrecompileMode::PlayedAreas;
        from = "EDEN_JIT_CACHE_REBUILD";
    }
    if (source != nullptr) {
        *source = from;
    }
    return mode;
}

bool JitBlockCache::TakePreloadRequest() {
    if (preload_taken.exchange(true)) {
        return false;
    }
    return KeepsHostCode() && GetPrecompileMode() != PrecompileMode::Off;
}

bool JitBlockCache::TakeRebuildRequest(const ListCheck& check) {
    if (rebuild_taken.exchange(true)) {
        return false;
    }
    if (!KeepsHostCode() || !writable) {
        return false;
    }
    const char* source = nullptr;
    const PrecompileMode mode = GetPrecompileMode(&source);
    if (mode == PrecompileMode::Off) {
        LOG_INFO(Core_ARM, "Precompile CPU code: off ({})", source);
        return false;
    }
    LOG_INFO(Core_ARM, "Precompile CPU code: {} ({})",
             mode == PrecompileMode::EntireGame ? "entire game" : "previously played areas", source);
    std::array<u8, JitPrebuildList::header_size> list_header{};
    if (mode == PrecompileMode::EntireGame) {
        // Imported list first, then the generated one; the first that matches the loaded code is used.
        bool found = false;
        for (const auto* candidate : {&prebuild_path, &generated_path}) {
            const bool generated = candidate == &generated_path;
            std::string error;
            const std::unique_ptr<JitPrebuildList> list = JitPrebuildList::Open(*candidate, program_id, &error);
            if (!list) {
                if (!error.empty()) {
                    LOG_WARNING(Core_ARM, "JIT prebuild list {} not used: {}", candidate->string(), error);
                }
                continue;
            }
            std::string why;
            if (check && !check(*list, generated, why)) {
                LOG_WARNING(Core_ARM, "JIT prebuild list {} not used: not made for this code: {}",
                            candidate->string(), why);
                continue;
            }
            LOG_INFO(Core_ARM, "JIT prebuild list {} ({}, {} entries): {}", candidate->string(),
                     generated ? "generated" : "imported", list->Size(), why);
            list_header = list->Header();
            selected_list_path = *candidate;
            found = true;
            break;
        }
        use_prebuild_list = true;
        if (!found) {
            generate_list = true;
            selected_list_path = generated_path;
            LOG_INFO(Core_ARM, "JIT cache: no prebuild list for this code; generating one ({})",
                     generated_path.string());
            return true;
        }
    }
    if (started_empty && (ManifestSize() != 0 || use_prebuild_list)) {
        return true;
    }
    if (!use_prebuild_list) {
        return false;
    }
    // Built for this list, scope and executable already?
    const std::vector<u8> want = PrebuiltMarker(list_header, GetPrebuildScope());
    std::vector<u8> have(want.size() + 1);
    std::ifstream marker(marker_path, std::ios::binary);
    marker.read(reinterpret_cast<char*>(have.data()), static_cast<std::streamsize>(have.size()));
    have.resize(static_cast<size_t>(marker.gcount()));
    if (have != want) {
        LOG_INFO(Core_ARM, "JIT cache: the records file was not built for this prebuild list and scope ({}){}; "
                 "building what is missing", PrebuildScopeName(GetPrebuildScope()),
                 prebuild_imported && selected_list_path == prebuild_path ? " (list just imported)" : "");
        return true;
    }
    return false;
}

void JitBlockCache::BeginBulkStore() {
    std::scoped_lock file_lock{file_mutex};
    FlushLocked();
    bulk_stream.open(path, std::ios::binary | std::ios::app);
    bulk_store.store(true);
}

void JitBlockCache::EndBulkStore() {
    std::unique_lock map_lock{map_mutex};
    std::scoped_lock file_lock{file_mutex};
    FlushLocked();
    bulk_store.store(false);
    bulk_stream.close();
    // Index what was written from where it was written (the file is not read again); earlier records of the same
    // key are replaced (BuildIndex keeps the highest offset).
    std::vector<IndexEntry> entries = std::move(bulk_written);
    bulk_written = {};
    if (entries.empty()) {
        return;
    }
    entries.reserve(entries.size() + index.size() + index_tail.size());
    entries.insert(entries.end(), index.begin(), index.end());
    for (const auto& [mixed, where] : index_tail) {
        entries.push_back({mixed, where});
    }
    index = {};
    dead_bytes += BuildIndex(std::move(entries), false);
    UnmapFile();
    if (!MapFile(file_size)) {
        LOG_WARNING(Core_ARM, "JIT block cache {}: cannot map the file after a bulk store; disabled", path.string());
        index.clear();
        index_buckets.clear();
        index_tail.clear();
        writable = false;
        return;
    }
    SaveIndexLocked();
}

// Manifest file, little-endian:
//   header (32 bytes): magic "N2JM", version (4), sessions (4), entry size (4) = 32, entry count (8), CityHash64 of
//   the entries (8);
//   entries (32 bytes each), most placed first: descriptor (8), end pc (8), code hash (8), hits (4), last session (4).
// Written whole to a temp file, flushed to disk, then renamed over the old file (ReplaceDurably): a kill, crash or
// power loss at any point leaves either the old or the new manifest, never a partial one; a file that fails the checks is ignored (and replaced on the next
// write). At most max_manifest_entries entries: beyond that the ones placed longest ago (then least placed) go.
void JitBlockCache::SaveManifest() {
    std::scoped_lock save_lock{manifest_save_mutex};
    DrainPlacements();
    if (!manifest_dirty.exchange(false)) {
        return;
    }
    std::vector<ManifestEntry> entries = ManifestSnapshot();
    if (entries.size() > max_manifest_entries) {
        std::nth_element(entries.begin(), entries.begin() + max_manifest_entries, entries.end(),
                         [](const ManifestEntry& a, const ManifestEntry& b) {
                             return a.last_session != b.last_session ? a.last_session > b.last_session : a.hits > b.hits;
                         });
        entries.resize(max_manifest_entries);
        std::sort(entries.begin(), entries.end(), [](const ManifestEntry& a, const ManifestEntry& b) {
            return a.hits != b.hits ? a.hits > b.hits : a.descriptor < b.descriptor;
        });
    }
    std::vector<u8> out;
    out.reserve(manifest_header_size + entries.size() * manifest_entry_size);
    Put<u32>(out, manifest_magic);
    Put<u32>(out, manifest_version);
    Put<u32>(out, manifest_session);
    Put<u32>(out, static_cast<u32>(manifest_entry_size));
    Put<u64>(out, entries.size());
    Put<u64>(out, 0);  // checksum, below
    for (const auto& e : entries) {
        Put<u64>(out, e.descriptor);
        Put<u64>(out, e.end_pc);
        Put<u64>(out, e.code_hash);
        Put<u32>(out, e.hits);
        Put<u32>(out, e.last_session);
    }
    const u64 checksum = Common::CityHash64(reinterpret_cast<const char*>(out.data()) + manifest_header_size,
                                            out.size() - manifest_header_size);
    for (size_t i = 0; i < 8; ++i) {
        out[24 + i] = static_cast<u8>(checksum >> (8 * i));
    }
    const auto temp = TempPathFor(manifest_path);
    bool ok;
    {
        std::ofstream file(temp, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
        file.close();
        ok = !file.fail();
    }
    std::error_code ec;
    if (!ok || !ReplaceDurably(temp, manifest_path, ec)) {
        std::filesystem::remove(temp, ec);
        manifest_dirty.store(true);  // try again later
        LOG_WARNING(Core_ARM, "JIT block manifest {} could not be written", manifest_path.string());
        return;
    }
    LOG_INFO(Core_ARM, "JIT block manifest {}: {} blocks written (session {})", manifest_path.string(), entries.size(),
             manifest_session);
}

void JitBlockCache::LoadManifest() {
    std::error_code ec;
    const u64 size = std::filesystem::exists(manifest_path, ec) ? std::filesystem::file_size(manifest_path, ec) : 0;
    u32 sessions = 0;
    if (!ec && size != 0) {
        bool valid = false;
        std::vector<u8> data;
        if (size >= manifest_header_size && size <= manifest_header_size + max_manifest_entries * manifest_entry_size) {
            data.resize(static_cast<size_t>(size));
            std::ifstream file(manifest_path, std::ios::binary);
            file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
            const u8* d = data.data();
            valid = static_cast<bool>(file) && Get<u32>(d) == manifest_magic && Get<u32>(d + 4) == manifest_version
                 && Get<u32>(d + 12) == manifest_entry_size
                 && size == manifest_header_size + Get<u64>(d + 16) * manifest_entry_size
                 && Get<u64>(d + 24) == Common::CityHash64(reinterpret_cast<const char*>(d) + manifest_header_size,
                                                           static_cast<size_t>(size - manifest_header_size));
        }
        if (valid) {
            sessions = Get<u32>(data.data() + 8);
            const u64 count = Get<u64>(data.data() + 16);
            for (u64 i = 0; i < count; ++i) {
                const u8* p = data.data() + manifest_header_size + i * manifest_entry_size;
                const u64 descriptor = Get<u64>(p);
                manifest[ShardOf(descriptor)].blocks.insert_or_assign(
                    descriptor, ManifestValue{Get<u64>(p + 8), Get<u64>(p + 16), Get<u32>(p + 24), Get<u32>(p + 28)});
            }
        } else {
            LOG_WARNING(Core_ARM, "JIT block manifest {} is damaged or of another version; starting a new one",
                        manifest_path.string());
        }
    }
    manifest_session = sessions + 1;
}

void JitBlockCache::StartManifestSaver() {
    manifest_saver = std::jthread([this](std::stop_token stop) {
        auto last_save = std::chrono::steady_clock::now();
        while (!stop.stop_requested()) {
            {
                std::unique_lock lock{saver_mutex};
                saver_cv.wait_for(lock, stop, placement_drain_interval, [] { return false; });
            }
            if (stop.stop_requested()) {
                break;
            }
            DrainPlacements();
            if (std::chrono::steady_clock::now() - last_save >= manifest_save_interval) {
                SaveManifest();
                last_save = std::chrono::steady_clock::now();
            }
        }
    });
}

}  // namespace Core
