// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <fmt/format.h>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include "common/common_types.h"
#include "core/arm/dynarmic/jit_block_cache.h"

namespace {

using Core::JitBlockCache;
namespace fs = std::filesystem;

constexpr u64 program_id = 0x0100000000ABC000ULL;

void SetEnv(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

/// A cache directory of its own (EDEN_JIT_CACHE_DIR), removed afterwards.
struct ScratchDir {
    fs::path dir;
    ScratchDir() {
#ifdef _WIN32
        const int pid = _getpid();
#else
        const int pid = static_cast<int>(getpid());
#endif
        dir = fs::temp_directory_path() / fmt::format("eden_jit_cache_test_{}", pid);
        std::error_code ec;
        fs::remove_all(dir, ec);
        fs::create_directories(dir);
        SetEnv("EDEN_JIT_CACHE_DIR", dir.string());
        SetEnv("EDEN_JIT_CACHE", "1");
    }
    ~ScratchDir() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
    fs::path File(const std::string& suffix) const {
        return dir / (fmt::format("{:016X}", program_id) + suffix);
    }
    /// File name -> size, for "nothing was written" checks.
    std::map<std::string, u64> Listing() const {
        std::map<std::string, u64> out;
        for (const auto& e : fs::directory_iterator(dir)) {
            out[e.path().filename().string()] = e.file_size();
        }
        return out;
    }
};

void WriteText(const fs::path& p, const std::string& text) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f << text;
}

std::string ReadText(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), {});
}

std::vector<u8> Record(u64 key) {
    std::vector<u8> r(700 + key % 300);
    u64 x = key * 0x9E3779B97F4A7C15ULL + 1;
    for (auto& b : r) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        b = static_cast<u8>(x);
    }
    return r;
}

} // namespace

TEST_CASE("JitBlockCache: temp names are unique and the durable replace swaps the file", "[core][jit_cache]") {
    ScratchDir scratch;
    const fs::path target = scratch.dir / "target.bin";
    const fs::path a = JitBlockCache::TempPathFor(target);
    const fs::path b = JitBlockCache::TempPathFor(target);
    REQUIRE(a != b);
    REQUIRE(a.filename().string().starts_with("target.bin."));
    REQUIRE(a.extension() == ".tmp");

    WriteText(target, "old");
    WriteText(a, "new contents");
    std::error_code ec;
    REQUIRE(JitBlockCache::ReplaceDurably(a, target, ec));
    REQUIRE(!ec);
    REQUIRE(ReadText(target) == "new contents");
    REQUIRE(!fs::exists(a));

    // A missing temp fails and leaves the target alone.
    REQUIRE(!JitBlockCache::ReplaceDurably(b, target, ec));
    REQUIRE(ec);
    REQUIRE(ReadText(target) == "new contents");
}

TEST_CASE("JitBlockCache: a second instance gets no cache and writes nothing", "[core][jit_cache]") {
    ScratchDir scratch;
    // Temp files left by a crash: this program's (old and new names) and another program's.
    WriteText(scratch.File(".bin.tmp"), "stale");
    WriteText(scratch.File(".manifest.1234-0.tmp"), "stale");
    const fs::path other = scratch.dir / "0100000000DEF000.bin.tmp";
    WriteText(other, "not ours");

    auto first = JitBlockCache::Open(program_id);
    REQUIRE(first != nullptr);
    REQUIRE(fs::exists(scratch.File(".lock")));
    REQUIRE(!fs::exists(scratch.File(".bin.tmp")));
    REQUIRE(!fs::exists(scratch.File(".manifest.1234-0.tmp")));
    REQUIRE(fs::exists(other));

    // While the first holds the lock: a stale-looking temp (could be the first's own) stays, and a second Open
    // returns nothing without touching any file.
    const fs::path in_use = scratch.File(".bin.999-7.tmp");
    WriteText(in_use, "the first instance's temp");
    const auto before = scratch.Listing();
    {
        auto second = JitBlockCache::Open(program_id);
        REQUIRE(second == nullptr);
    }
    REQUIRE(scratch.Listing() == before);
    REQUIRE(fs::exists(in_use));

    first.reset();
    auto third = JitBlockCache::Open(program_id);  // the lock is free again
    REQUIRE(third != nullptr);
    REQUIRE(!fs::exists(in_use));
}

TEST_CASE("JitBlockCache: DropRecords keeps the chosen records across a reopen", "[core][jit_cache]") {
    ScratchDir scratch;
    constexpr u64 count = 4000;
    {
        auto cache = JitBlockCache::Open(program_id);
        REQUIRE(cache != nullptr);
        cache->BeginBulkStore();
        for (u64 k = 1; k <= count; ++k) {
            cache->Store(k, Record(k));
        }
        cache->Store(10, Record(10));  // a replaced record: the later one is the live one
        cache->EndBulkStore();
        REQUIRE(cache->RecordCount() == count);
        const u64 size_before = cache->FileSize();

        std::string what;
        // Below the threshold: nothing happens.
        const auto none = cache->DropRecords([](u64 key) { return key % 2 == 0; }, u64{1} << 40, what);
        REQUIRE(none.has_value());
        REQUIRE(*none == 0);
        REQUIRE(cache->FileSize() == size_before);

        const auto removed = cache->DropRecords([](u64 key) { return key % 2 == 0; }, 0, what);
        INFO(what);
        REQUIRE(removed.has_value());
        REQUIRE(*removed > 0);
        REQUIRE(cache->FileSize() < size_before);
        REQUIRE(cache->RecordCount() == count / 2);
        for (u64 k = 1; k <= count; ++k) {
            const auto r = cache->Load(k);
            if (k % 2 == 0) {
                REQUIRE(r.has_value());
                REQUIRE(*r == Record(k));
            } else {
                REQUIRE(!r.has_value());
            }
        }
        const std::array<u8, 48> list_header{};
        cache->MarkPrebuilt(list_header, JitBlockCache::PrebuildScope::Full);
        REQUIRE(fs::exists(scratch.File(".prebuilt")));
    }
    // No temp file is left behind, and the reopened file (through the saved index) holds the kept records.
    for (const auto& [name, size] : scratch.Listing()) {
        REQUIRE(!name.ends_with(".tmp"));
    }
    auto cache = JitBlockCache::Open(program_id);
    REQUIRE(cache != nullptr);
    REQUIRE(cache->RecordCount() == count / 2);
    REQUIRE(cache->Load(count) == std::optional{Record(count)});
    REQUIRE(!cache->Load(count - 1).has_value());
}
