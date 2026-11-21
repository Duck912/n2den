// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Cortex-A78C (ARMv8.2 + LRCPC, FlagM, PAuth) instructions added for Switch 2 guests, checked against values
// worked out from the Arm ARM pseudocode.

#include <catch2/catch_test_macros.hpp>

#include "dynarmic/interface/exclusive_monitor.h"
#include "dynarmic/tests/A64/testenv.h"

namespace {

struct Harness {
    A64TestEnv env;
    Dynarmic::ExclusiveMonitor monitor{1};
    Dynarmic::A64::UserConfig config;
    std::unique_ptr<Dynarmic::A64::Jit> jit;

    explicit Harness(std::vector<u32> code) {
        env.code_mem = std::move(code);
        env.code_mem.push_back(0x14000000);  // B .
        config.callbacks = &env;
        config.global_monitor = &monitor;
        config.enable_pauth_returns = true;
        config.code_cache_size = 16 * 1024 * 1024;
        jit = std::make_unique<Dynarmic::A64::Jit>(config);
    }

    void Run(u64 instructions = 1) {
        env.ticks_left = instructions;
        jit->Run();
    }
};

}  // namespace

TEST_CASE("A64: LDSMAX/LDSMIN compare signed, LDUMAX unsigned", "[a64][lse][armv82]") {
    {
        Harness h{{0xb8204022}};  // ldsmax w0, w2, [x1]
        h.env.MemoryWrite(0x1000, 0xb3b2b1b0, 4);
        h.jit->SetRegister(0, 0x3d8b068d);
        h.jit->SetRegister(1, 0x1000);
        h.Run();
        REQUIRE(h.env.MemoryRead(0x1000, 4) == 0x3d8b068d);
        REQUIRE(h.jit->GetRegister(2) == 0xb3b2b1b0);
    }
    {
        Harness h{{0xb8206022}};  // ldumax w0, w2, [x1]
        h.env.MemoryWrite(0x1000, 0xb3b2b1b0, 4);
        h.jit->SetRegister(0, 0x3d8b068d);
        h.jit->SetRegister(1, 0x1000);
        h.Run();
        REQUIRE(h.env.MemoryRead(0x1000, 4) == 0xb3b2b1b0);
        REQUIRE(h.jit->GetRegister(2) == 0xb3b2b1b0);
    }
    {
        Harness h{{0x38204022}};  // ldsmaxb w0, w2, [x1]
        h.env.MemoryWrite(0x1000, 0xd0, 1);
        h.jit->SetRegister(0, 0xfa2b);
        h.jit->SetRegister(1, 0x1000);
        h.Run();
        REQUIRE(h.env.MemoryRead(0x1000, 1) == 0x2b);
        REQUIRE(h.jit->GetRegister(2) == 0xd0);
    }
    {
        Harness h{{0x78205022}};  // ldsminh w0, w2, [x1]
        h.env.MemoryWrite(0x1000, 0x1110, 2);
        h.jit->SetRegister(0, 0xffff);
        h.jit->SetRegister(1, 0x1000);
        h.Run();
        REQUIRE(h.env.MemoryRead(0x1000, 2) == 0xffff);
        REQUIRE(h.jit->GetRegister(2) == 0x1110);
    }
    {
        Harness h{{0xf8204022}};  // ldsmax x0, x2, [x1]
        h.env.MemoryWrite(0x1000, 0x8000000000000000, 8);
        h.jit->SetRegister(0, 5);
        h.jit->SetRegister(1, 0x1000);
        h.Run();
        REQUIRE(h.env.MemoryRead(0x1000, 8) == 5);
        REQUIRE(h.jit->GetRegister(2) == 0x8000000000000000);
    }
}

TEST_CASE("A64: CASP compares and swaps register pairs", "[a64][lse][armv82]") {
    for (const bool match : {true, false}) {
        Harness h{{0x48227c24}};  // casp x2, x3, x4, x5, [x1]
        h.env.MemoryWrite(0x1000, 0x1111111111111111, 8);
        h.env.MemoryWrite(0x1008, 0x2222222222222222, 8);
        h.jit->SetRegister(1, 0x1000);
        h.jit->SetRegister(2, match ? 0x1111111111111111 : 0x1111111111111110);
        h.jit->SetRegister(3, 0x2222222222222222);
        h.jit->SetRegister(4, 0x3333333333333333);
        h.jit->SetRegister(5, 0x4444444444444444);
        h.jit->SetPstate(0x60000000);
        h.Run();
        REQUIRE(h.env.MemoryRead(0x1000, 8) == (match ? 0x3333333333333333 : 0x1111111111111111));
        REQUIRE(h.env.MemoryRead(0x1008, 8) == (match ? 0x4444444444444444 : 0x2222222222222222));
        REQUIRE(h.jit->GetRegister(2) == 0x1111111111111111);
        REQUIRE(h.jit->GetRegister(3) == 0x2222222222222222);
        REQUIRE(h.jit->GetPstate() == 0x60000000);  // flags untouched
    }
    {
        Harness h{{0x08207c82}};  // casp w0, w1, w2, w3, [x4]
        h.env.MemoryWrite(0x1000, 0x11111111, 4);
        h.env.MemoryWrite(0x1004, 0x22222222, 4);
        h.jit->SetRegister(4, 0x1000);
        h.jit->SetRegister(0, 0x11111111);
        h.jit->SetRegister(1, 0x22222222);
        h.jit->SetRegister(2, 0x33);
        h.jit->SetRegister(3, 0x44);
        h.Run();
        REQUIRE(h.env.MemoryRead(0x1000, 4) == 0x33);
        REQUIRE(h.env.MemoryRead(0x1004, 4) == 0x44);
        REQUIRE(h.jit->GetRegister(0) == 0x11111111);
        REQUIRE(h.jit->GetRegister(1) == 0x22222222);
    }
}

TEST_CASE("A64: PAuth branches and loads with keys disabled", "[a64][pauth][armv82]") {
    {
        Harness h{{0xd71f0822}};  // braa x1, x2
        h.jit->SetRegister(1, 0x100);
        h.jit->SetRegister(2, 0xdead);
        h.Run();
        REQUIRE(h.jit->GetPC() == 0x100);
    }
    {
        Harness h{{0xd73f0822}};  // blraa x1, x2
        h.jit->SetRegister(1, 0x100);
        h.Run();
        REQUIRE(h.jit->GetPC() == 0x100);
        REQUIRE(h.jit->GetRegister(30) == 4);
    }
    {
        Harness h{{0xf8200420, 0xf8fff423}};  // ldraa x0, [x1] ; ldrab x3, [x1, #-8]
        h.jit->SetRegister(1, 0x2008);
        h.Run(2);
        REQUIRE(h.jit->GetRegister(0) == 0x0f0e0d0c0b0a0908);  // unwritten memory reads as its address' low byte
        REQUIRE(h.jit->GetRegister(3) == 0x0706050403020100);
        REQUIRE(h.jit->GetRegister(1) == 0x2008);
    }
    {
        Harness h{{0xf83ffc20}};  // ldraa x0, [x1, #4088]!
        h.jit->SetRegister(1, 0x1008);
        h.Run();
        REQUIRE(h.jit->GetRegister(0) == 0x0706050403020100);
        REQUIRE(h.jit->GetRegister(1) == 0x2000);
    }
}

TEST_CASE("A64: SETF8 and SETF16", "[a64][flagm][armv82]") {
    {
        Harness h{{0x3a00082d}};  // setf8 w1
        h.jit->SetRegister(1, 0x180);  // N = bit 7, Z = 0, V = bit 8 ^ bit 7 = 0
        h.jit->SetPstate(0x20000000);  // C is preserved
        h.Run();
        REQUIRE(h.jit->GetPstate() == 0xA0000000);
    }
    {
        Harness h{{0x3a00484d}};  // setf16 w2
        h.jit->SetRegister(2, 0x8000);  // N = 1, Z = 0, V = bit 16 ^ bit 15 = 1
        h.Run();
        REQUIRE(h.jit->GetPstate() == 0x90000000);
    }
    {
        Harness h{{0x3a00082d}};  // setf8 w1 with a zero low byte
        h.jit->SetRegister(1, 0xff00);
        h.Run();
        REQUIRE(h.jit->GetPstate() == 0x50000000);  // Z, and V = bit 8 ^ bit 7 = 1
    }
}

TEST_CASE("A64: SQRDMLAH saturates once on the full-precision sum", "[a64][rdm][armv82]") {
    Harness h{{0x2e408422}};  // sqrdmlah v2.4h, v1.4h, v0.4h
    h.jit->SetVector(0, {0x0001800040007fff, 0});
    h.jit->SetVector(1, {0x0001800040007fff, 0});
    h.jit->SetVector(2, {0xffff010000007fff, 0});
    h.Run();
    // lanes: sat(0x7fff + 0x7ffe) = 0x7fff; 0x2000; sat(0x100 + 0x8000) = 0x7fff; (-65536 + 2 + 32768) >> 16 = -1
    REQUIRE(h.jit->GetVector(2) == Vector{0xffff7fff20007fff, 0});
    REQUIRE((h.jit->GetFpsr() & (1 << 27)) != 0);  // QC
}

TEST_CASE("A64: FP16 scalar arithmetic rounds and raises flags", "[a64][fp16][armv82]") {
    Harness h{{0x1ee22820, 0x1ee22883}};  // fadd h0, h1, h2 ; fadd h3, h4, h2
    h.jit->SetVector(1, {0x3c00, 0});  // 1.0
    h.jit->SetVector(2, {0x7bff, 0});  // 65504
    h.jit->SetVector(4, {0x7bff, 0});
    h.Run(2);
    REQUIRE(h.jit->GetVector(0) == Vector{0x7bff, 0});  // 65504 + 1 rounds back to 65504
    REQUIRE(h.jit->GetVector(3) == Vector{0x7c00, 0});  // overflow to +inf
    REQUIRE((h.jit->GetFpsr() & 0x1f) == 0x14);         // OFC | IXC
}
