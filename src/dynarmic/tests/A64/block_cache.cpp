// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// UserConfig::block_cache: blocks stored by one Jit are reused by the next, and never for changed guest code.

#include <map>

#include <catch2/catch_test_macros.hpp>

#include "dynarmic/interface/A64/block_cache.h"
#include "dynarmic/tests/A64/testenv.h"

namespace {

class MapCache final : public Dynarmic::A64::BlockCacheStorage {
public:
    std::map<u64, std::vector<u8>> records;
    std::optional<std::vector<u8>> Load(u64 key) override {
        const auto it = records.find(key);
        return it == records.end() ? std::nullopt : std::optional{it->second};
    }
    void Store(u64 key, std::vector<u8> record) override {
        records[key] = std::move(record);
    }
};

struct Result {
    u64 x0, x1;
    Dynarmic::A64::BlockCacheStats stats;
};

Result RunProgram(const std::vector<u32>& code, MapCache* cache) {
    A64TestEnv env;
    env.code_mem = code;
    Dynarmic::A64::UserConfig config;
    config.callbacks = &env;
    config.block_cache = cache;
    config.code_cache_size = 16 * 1024 * 1024;
    Dynarmic::A64::Jit jit{config};
    jit.SetRegister(0, 3);
    jit.SetRegister(1, 0);
    env.ticks_left = 200;
    jit.Run();
    return {jit.GetRegister(0), jit.GetRegister(1), jit.GetBlockCacheStats()};
}

// loop: add x1, x1, x0 ; subs x0, x0, #1 ; b.ne loop ; then a tail block: add x1, x1, #100 ; b .
const std::vector<u32> program = {
    0x8b000021,  // add x1, x1, x0
    0xf1000400,  // subs x0, x0, #1
    0x54ffffc1,  // b.ne #-8
    0x91019021,  // add x1, x1, #100
    0x14000000,  // b .
};

}  // namespace

TEST_CASE("A64: block cache reuses blocks across Jit instances", "[a64][block_cache]") {
    MapCache cache;
    const Result cold = RunProgram(program, &cache);
    REQUIRE(cold.x1 == 3 + 2 + 1 + 100);
    REQUIRE(cold.stats.misses > 0);
    REQUIRE(cold.stats.hits == 0);
    REQUIRE(!cache.records.empty());

    const Result warm = RunProgram(program, &cache);
    REQUIRE(warm.x0 == cold.x0);
    REQUIRE(warm.x1 == cold.x1);
    REQUIRE(warm.stats.misses == 0);
    REQUIRE(warm.stats.hits == cold.stats.misses);
}

TEST_CASE("A64: block cache ignores records for changed guest code", "[a64][block_cache]") {
    MapCache cache;
    RunProgram(program, &cache);

    std::vector<u32> patched = program;
    patched[3] = 0x91032021;  // add x1, x1, #200 instead of #100
    const Result result = RunProgram(patched, &cache);
    REQUIRE(result.x1 == 3 + 2 + 1 + 200);
    REQUIRE(result.stats.stale > 0);  // the old tail block was found and rejected
}

TEST_CASE("A64: block cache rejects damaged records", "[a64][block_cache]") {
    MapCache cache;
    const Result cold = RunProgram(program, &cache);
    for (auto& [key, record] : cache.records) {
        record.resize(record.size() / 2);  // truncate every record
    }
    const Result result = RunProgram(program, &cache);
    REQUIRE(result.x1 == cold.x1);
    REQUIRE(result.stats.hits == 0);
    REQUIRE(result.stats.stale == cold.stats.misses);
}
