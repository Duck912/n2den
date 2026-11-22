// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// dynarmic_a64_probe: tooling for checking A64 instruction coverage and behaviour.
//
//   dynarmic_a64_probe coverage <words.bin> <result.bin> [start-index]
//       Translates every little-endian 32-bit word of words.bin as a one-instruction block and writes one byte per
//       word: 0 translated, 1 unallocated encoding, 2 reserved value, 3 unpredictable, 4 another exception.
//
//   dynarmic_a64_probe run <cases.txt>
//       Runs each case line and prints the final state on one line. A case line is hex tokens:
//         N insn*N X0..X30 SP NZCV FPCR FPSR V0lo V0hi .. V31lo V31hi
//       The output line is: X0..X30 SP PC NZCV FPSR V0lo V0hi .. V31hi [exc=<n>] [m:<addr>=<byte> ...]
//       Unwritten memory reads as the low byte of its address, like A64TestEnv.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <span>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cstdlib>
#endif

#include "./A64/testenv.h"
#include "common/logging.h"
#include "common/lz4_compression.h"
#include "common/zbic_compression.h"
#include "dynarmic/frontend/A64/a64_location_descriptor.h"
#include "dynarmic/frontend/A64/decoder/a64.h"
#include "dynarmic/frontend/A64/translate/impl/impl.h"
#include "dynarmic/frontend/A64/translate/a64_translate.h"
#include "dynarmic/interface/A64/a64.h"
#include "dynarmic/interface/exclusive_monitor.h"
#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"

using namespace Dynarmic;

namespace {

int Coverage(const char* in_path, const char* out_path, size_t start) {
    std::ifstream in(in_path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", in_path);
        return 1;
    }
    std::vector<u32> words;
    u32 w;
    while (in.read(reinterpret_cast<char*>(&w), 4)) {
        words.push_back(w);
    }
    // Results are appended and flushed per word, so after a crash the caller knows which word crashed
    // (the one at the result file's length) and can resume after it.
    std::FILE* out = std::fopen(out_path, start == 0 ? "wb" : "ab");
    if (!out) {
        std::fprintf(stderr, "cannot open %s\n", out_path);
        return 1;
    }

    A64::TranslationOptions options{};
    options.define_unpredictable_behaviour = true;
    options.hook_hint_instructions = false;
    options.enable_pauth_returns = true;
    options.address_space_bits = 39;

    for (size_t i = start; i < words.size(); ++i) {
        const u32 insn = words[i];
        const A64::LocationDescriptor loc{0x1000, FP::FPCR{0}, false};
        IR::Block block{loc};
        // Only the probed address holds code: the next fetch faults, so every block ends after one instruction.
        A64::Translate(block, loc, [insn](u64 vaddr) -> std::optional<u32> {
            if (vaddr == 0x1000) {
                return insn;
            }
            return std::nullopt;
        }, options);
        u8 r = 0;
        for (const auto& inst : block.instructions) {
            if (inst.GetOpcode() == IR::Opcode::A64ExceptionRaised) {
                const auto exc = static_cast<A64::Exception>(inst.GetArg(1).GetU64());
                if (exc == A64::Exception::NoExecuteFault) {
                    continue;  // the fetch past the probed instruction
                }
                switch (exc) {
                case A64::Exception::UnallocatedEncoding:
                    r = 1;
                    break;
                case A64::Exception::ReservedValue:
                    r = 2;
                    break;
                case A64::Exception::UnpredictableInstruction:
                    r = 3;
                    break;
                case A64::Exception::Breakpoint:
                    r = 0;
                    break;
                default:
                    r = 4;
                    break;
                }
                break;
            }
        }
        std::fputc(r, out);
        std::fflush(out);
    }
    std::fclose(out);
    return 0;
}

class ProbeEnv final : public A64TestEnv {
public:
    std::optional<int> exception;
    Dynarmic::A64::Jit* jit = nullptr;

    void ExceptionRaised(u64, Dynarmic::A64::Exception e) override {
        if (!exception) {
            exception = static_cast<int>(e);
        }
        ticks_left = 0;
        if (jit) {
            jit->HaltExecution();
        }
    }
    void CallSVC(std::uint32_t) override {
        if (!exception) {
            exception = 100;
        }
        ticks_left = 0;
        if (jit) {
            jit->HaltExecution();
        }
    }

    // Memory callbacks run with the guest's MXCSR, so host floating-point work in them (an unordered_map's load
    // factor) would leak into the guest FPSR. Keep data memory in a std::map, which uses no floating point.
    std::map<u64, u8> memory;

    u8 Read8(u64 vaddr) const {
        if (IsInCodeMem(vaddr)) {
            return reinterpret_cast<const u8*>(code_mem.data())[vaddr - code_mem_start_address];
        }
        const auto it = memory.find(vaddr);
        return it != memory.end() ? it->second : static_cast<u8>(vaddr);
    }
    u64 MemoryRead(u64 vaddr, size_t size) override {
        u64 value = 0;
        for (size_t i = 0; i < size; ++i) {
            value |= u64{Read8(vaddr + i)} << (8 * i);
        }
        return value;
    }
    Vector MemoryRead128(u64 vaddr) override {
        return {MemoryRead(vaddr, 8), MemoryRead(vaddr + 8, 8)};
    }
    void MemoryWrite(Dynarmic::A64::VAddr vaddr, u64 value, size_t size) override {
        for (size_t i = 0; i < size; ++i) {
            memory[vaddr + i] = static_cast<u8>(value >> (8 * i));
        }
    }
    void MemoryWrite128(u64 vaddr, Vector value) override {
        MemoryWrite(vaddr, value[0], 8);
        MemoryWrite(vaddr + 8, value[1], 8);
    }
    bool MemoryWriteExclusive8(u64 vaddr, std::uint8_t value, std::uint8_t) override {
        MemoryWrite(vaddr, value, 1);
        return true;
    }
    bool MemoryWriteExclusive16(u64 vaddr, std::uint16_t value, std::uint16_t) override {
        MemoryWrite(vaddr, value, 2);
        return true;
    }
    bool MemoryWriteExclusive32(u64 vaddr, std::uint32_t value, std::uint32_t) override {
        MemoryWrite(vaddr, value, 4);
        return true;
    }
    bool MemoryWriteExclusive64(u64 vaddr, std::uint64_t value, std::uint64_t) override {
        MemoryWrite(vaddr, value, 8);
        return true;
    }
    bool MemoryWriteExclusive128(u64 vaddr, Vector value, Vector) override {
        MemoryWrite128(vaddr, value);
        return true;
    }
};

// nsotext <in.nso> <out.bin>: writes the decompressed .text segment, decompressed as the NSO loader does it
// (src/core/loader/nso.cpp: flag bit 0 = text compressed, bit 7 = ZBIC instead of LZ4).
int NsoText(const char* in_path, const char* out_path) {
    std::ifstream in(in_path, std::ios::binary);
    std::vector<u8> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (data.size() < 0x100 || std::memcmp(data.data(), "NSO0", 4) != 0) {
        std::fprintf(stderr, "%s: not an NSO\n", in_path);
        return 1;
    }
    const auto rd = [&](size_t off) {
        u32 v;
        std::memcpy(&v, data.data() + off, 4);
        return v;
    };
    const u32 flags = rd(0xC), file_off = rd(0x10), size = rd(0x18), csize = rd(0x60);
    const bool compressed = (flags & 1) != 0;
    const size_t stored = compressed ? csize : size;
    if (size_t{file_off} + stored > data.size()) {
        std::fprintf(stderr, "%s: text segment past the end of the file\n", in_path);
        return 1;
    }
    std::vector<u8> text(size);
    const u8* src = data.data() + file_off;
    if (!compressed) {
        std::memcpy(text.data(), src, size);
    } else if ((flags >> 7) & 1) {
        if (::Common::Compression::DecompressDataZBIC(text, std::span<const u8>{src, stored}) != int(size)) {
            std::fprintf(stderr, "%s: ZBIC decompression failed\n", in_path);
            return 1;
        }
    } else if (::Common::Compression::DecompressDataLZ4(text.data(), size, src, stored) != int(size)) {
        std::fprintf(stderr, "%s: LZ4 decompression failed\n", in_path);
        return 1;
    }
    std::ofstream out(out_path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(text.data()), static_cast<std::streamsize>(text.size()));
    return out ? 0 : 1;
}

// Prints, for each hex word on the command line, the name of the first decoder entry that matches it.
int Names(int argc, char** argv) {
    for (int i = 0; i < argc; ++i) {
        const u32 insn = static_cast<u32>(std::stoul(argv[i], nullptr, 16));
        const auto name = A64::GetName<A64::TranslatorVisitor>(insn);
        std::printf("%08x %s\n", insn, name ? std::string(*name).c_str() : "<none>");
    }
    return 0;
}

// Prints the translated (unoptimized) IR of each hex word on the command line.
int IrDump(int argc, char** argv) {
    A64::TranslationOptions options{};
    options.define_unpredictable_behaviour = true;
    options.hook_hint_instructions = false;
    options.enable_pauth_returns = true;
    options.address_space_bits = 39;
    for (int i = 0; i < argc; ++i) {
        const u32 insn = static_cast<u32>(std::stoul(argv[i], nullptr, 16));
        const A64::LocationDescriptor loc{0x1000, FP::FPCR{0}, false};
        IR::Block block{loc};
        A64::Translate(block, loc, [insn](u64 vaddr) -> std::optional<u32> {
            return vaddr == 0x1000 ? std::optional<u32>{insn} : std::nullopt;
        }, options);
        std::printf("%08x:\n%s\n", insn, IR::DumpBlock(block).c_str());
    }
    return 0;
}

struct CacheOptions {
    bool native = false;
    bool verify = false;
};
CacheOptions cache_options;

/// In-memory BlockCacheStorage for tests and benchmarks.
class MemoryBlockCache final : public A64::BlockCacheStorage {
public:
    std::map<u64, std::vector<u8>> records;

    bool SaveTo(const char* path) const {
        std::ofstream out(path, std::ios::binary);
        for (const auto& [key, rec] : records) {
            const u64 size = rec.size();
            out.write(reinterpret_cast<const char*>(&key), 8);
            out.write(reinterpret_cast<const char*>(&size), 8);
            out.write(reinterpret_cast<const char*>(rec.data()), static_cast<std::streamsize>(size));
        }
        return static_cast<bool>(out);
    }
    bool LoadFrom(const char* path) {
        std::ifstream in(path, std::ios::binary);
        u64 key, size;
        while (in.read(reinterpret_cast<char*>(&key), 8) && in.read(reinterpret_cast<char*>(&size), 8)) {
            std::vector<u8> rec(size);
            in.read(reinterpret_cast<char*>(rec.data()), static_cast<std::streamsize>(size));
            records[key] = std::move(rec);
        }
        return !records.empty();
    }
    std::optional<std::vector<u8>> Load(u64 key) override {
        const auto it = records.find(key);
        if (it == records.end()) {
            return std::nullopt;
        }
        return it->second;
    }
    void Store(u64 key, std::vector<u8> record) override {
        records[key] = std::move(record);
    }
};

/// Runs one case line (see the header comment) and returns the final state line.
std::string RunCase(const std::string& line, Dynarmic::ExclusiveMonitor& monitor, A64::BlockCacheStorage* cache,
                    A64::BlockCacheStats* stats = nullptr) {
    std::istringstream ss(line);
    ss >> std::hex;
    u64 n = 0;
    ss >> n;
    ProbeEnv env;
    for (u64 i = 0; i < n; ++i) {
        u64 insn;
        ss >> insn;
        env.code_mem.push_back(static_cast<u32>(insn));
    }
    env.code_mem.push_back(0x14000000);  // B .

    std::array<u64, 31> x{};
    for (auto& r : x) {
        ss >> r;
    }
    u64 sp, nzcv, fpcr, fpsr;
    ss >> sp >> nzcv >> fpcr >> fpsr;
    std::array<Vector, 32> v{};
    for (auto& r : v) {
        ss >> r[0] >> r[1];
    }

    A64::UserConfig conf{};
    conf.callbacks = &env;
    conf.global_monitor = &monitor;
    conf.define_unpredictable_behaviour = true;
    conf.enable_pauth_returns = true;
    conf.hook_hint_instructions = false;
    conf.block_cache = cache;
    conf.block_cache_native = cache_options.native;
    conf.block_cache_verify_native = cache_options.verify;
    A64::Jit jit{conf};
    env.jit = &jit;
    jit.SetRegisters(x);
    jit.SetSP(sp);
    jit.SetVectors(v);
    jit.SetPstate(static_cast<u32>(nzcv));
    jit.SetFpcr(static_cast<u32>(fpcr));
    jit.SetFpsr(static_cast<u32>(fpsr));
    jit.SetPC(0);
    env.ticks_left = n;
    jit.Run();
    if (stats) {
        *stats = jit.GetBlockCacheStats();
    }

    std::ostringstream o;
    o << std::hex;
    for (const u64 r : jit.GetRegisters()) {
        o << r << ' ';
    }
    o << jit.GetSP() << ' ' << jit.GetPC() << ' ' << (jit.GetPstate() & 0xF0000000) << ' ' << jit.GetFpsr();
    for (const auto& r : jit.GetVectors()) {
        o << ' ' << r[0] << ' ' << r[1];
    }
    if (env.exception) {
        o << " exc=" << *env.exception;
    }
    for (const auto& [addr, byte] : env.memory) {
        o << " m:" << addr << '=' << static_cast<unsigned>(byte);
    }
    return o.str();
}

int Run(const char* path) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    Dynarmic::ExclusiveMonitor monitor{1};
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            std::cout << RunCase(line, monitor, nullptr) << '\n';
        }
    }
    return 0;
}

// Runs every case twice in fresh JITs sharing one block cache: the second run must load every block from the
// cache and end in exactly the same state.
int RunCache(const char* path) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    Dynarmic::ExclusiveMonitor monitor{1};
    std::string line;
    u64 cases = 0, mismatches = 0, misses_on_second = 0, hits = 0, bytes = 0, records = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        MemoryBlockCache cache;
        A64::BlockCacheStats first{}, second{};
        const std::string a = RunCase(line, monitor, &cache, &first);
        const std::string b = RunCase(line, monitor, &cache, &second);
        ++cases;
        hits += second.hits;
        misses_on_second += second.misses;
        records += cache.records.size();
        for (const auto& [key, rec] : cache.records) {
            bytes += rec.size();
        }
        if (a != b) {
            if (++mismatches <= 5) {
                std::printf("MISMATCH case %llu\n  first : %s\n  second: %s\n", (unsigned long long)cases, a.c_str(), b.c_str());
            }
        }
    }
    std::printf("cases=%llu mismatches=%llu second-run hits=%llu second-run misses=%llu records=%llu avg-record-bytes=%llu\n",
                (unsigned long long)cases, (unsigned long long)mismatches, (unsigned long long)hits,
                (unsigned long long)misses_on_second, (unsigned long long)records,
                (unsigned long long)(records ? bytes / records : 0));
    return mismatches == 0 && misses_on_second == 0 ? 0 : 1;
}

/// Reads code from a raw image (address = file offset) for the benchmark.
class ImageEnv final : public A64TestEnv {
public:
    const std::vector<u32>* image = nullptr;
    std::optional<std::uint32_t> MemoryReadCode(u64 vaddr) override {
        const u64 index = vaddr / 4;
        if (vaddr % 4 != 0 || index >= image->size()) {
            return std::nullopt;
        }
        return (*image)[index];
    }
    u64 MemoryRead(u64 vaddr, size_t size) override {
        u64 value = 0;
        for (size_t i = 0; i < size; ++i) {
            value |= u64{static_cast<u8>(vaddr + i)} << (8 * i);
        }
        return value;
    }
    void MemoryWrite(Dynarmic::A64::VAddr, u64, size_t) override {}
    void ExceptionRaised(u64, Dynarmic::A64::Exception) override { ticks_left = 0; }
    void CallSVC(std::uint32_t) override { ticks_left = 0; }
};

// bench <image> <text-size> <count>: compiles the first block of `count` functions (PACIBSP entry points)
// without a cache, then again in a fresh JIT from the filled cache.
int Bench(const char* image_path, u64 text_size, u64 count, const char* load_path, const char* save_path) {
    std::ifstream in(image_path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", image_path);
        return 1;
    }
    std::vector<u32> image(text_size / 4);
    in.read(reinterpret_cast<char*>(image.data()), static_cast<std::streamsize>(image.size() * 4));
    std::vector<u64> entries;
    for (u64 i = 0; i < image.size() && entries.size() < count; ++i) {
        if (image[i] == 0xd503237f) {  // PACIBSP
            entries.push_back(i * 4);
        }
    }
    MemoryBlockCache cache;
    if (load_path && !cache.LoadFrom(load_path)) {
        std::fprintf(stderr, "cannot load %s\n", load_path);
        return 1;
    }
    Dynarmic::ExclusiveMonitor monitor{1};
    // With a loaded cache, only the warm pass runs: its host code comes from another process.
    for (int pass = load_path ? 1 : 0; pass < 2; ++pass) {
        ImageEnv env;
        env.image = &image;
        A64::UserConfig conf{};
        conf.callbacks = &env;
        conf.global_monitor = &monitor;
        conf.define_unpredictable_behaviour = true;
        conf.enable_pauth_returns = true;
        conf.block_cache = &cache;
        conf.block_cache_native = cache_options.native;
        conf.block_cache_verify_native = cache_options.verify && pass == 1;
        conf.code_cache_size = 512 * 1024 * 1024;
        A64::Jit jit{conf};
        const auto start = std::chrono::steady_clock::now();
        for (const u64 entry : entries) {
            jit.SetPC(entry);
            jit.SetSP(0x10000000);
            env.ticks_left = 1;
            jit.Run();
        }
        const double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        const auto s = jit.GetBlockCacheStats();
        u64 bytes = 0;
        for (const auto& [k, r] : cache.records) {
            bytes += r.size();
        }
        std::printf("%s: blocks hit=%llu (native %llu) miss=%llu stale=%llu | translate %.1f ms, cache load %.1f ms, emit %.1f ms | wall %.1f ms | records=%zu avg %llu bytes\n"
                    "  native: stored=%llu not-relocatable=%llu rejected=%llu verified=%llu mismatches=%llu\n",
                    pass == 0 ? "cold (no cache)" : "warm (from cache)",
                    (unsigned long long)s.hits, (unsigned long long)s.native_hits, (unsigned long long)s.misses, (unsigned long long)s.stale,
                    s.translate_ns / 1e6, s.load_ns / 1e6, s.emit_ns / 1e6, total_ms, cache.records.size(),
                    (unsigned long long)(cache.records.empty() ? 0 : bytes / cache.records.size()),
                    (unsigned long long)s.native_stored, (unsigned long long)s.native_not_relocatable, (unsigned long long)s.native_rejected,
                    (unsigned long long)s.native_verified, (unsigned long long)s.native_verify_mismatches);
    }
    if (save_path) {
        cache.SaveTo(save_path);
    }
    return 0;
}

}  // namespace

#ifdef _WIN32
// A runaway translation must fail this process, not exhaust the machine's memory.
static void LimitProcessMemory(size_t bytes) {
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (!job) {
        return;
    }
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    info.ProcessMemoryLimit = bytes;
    SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info, sizeof(info));
    AssignProcessToJobObject(job, GetCurrentProcess());
}
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    LimitProcessMemory(size_t{2} << 30);
    // A translator abort ends the process quietly: no error-reporting dialog to block the audit.
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
    // Assertion messages go through the common logger; send them to the console.
    ::Common::Log::Initialize();
    ::Common::Log::SetColorConsoleBackendEnabled(true);
    ::Common::Log::Start();
    if (argc >= 4 && std::string(argv[1]) == "coverage") {
        return Coverage(argv[2], argv[3], argc >= 5 ? std::stoull(argv[4]) : 0);
    }
    if (argc >= 2 && std::string(argv[1]) == "irdump") {
        return IrDump(argc - 2, argv + 2);
    }
    if (argc >= 4 && std::string(argv[1]) == "nsotext") {
        return NsoText(argv[2], argv[3]);
    }
    if (argc >= 2 && std::string(argv[1]) == "names") {
        return Names(argc - 2, argv + 2);
    }
    // Cache mode flags: --native (keep host code), --verify (check cached host code), --load/--save <file> (bench).
    const char* load_path = nullptr;
    const char* save_path = nullptr;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--native") {
            cache_options.native = true;
        } else if (a == "--verify") {
            cache_options.verify = true;
        } else if (a == "--load" && i + 1 < argc) {
            load_path = argv[++i];
        } else if (a == "--save" && i + 1 < argc) {
            save_path = argv[++i];
        }
    }
    if (argc >= 3 && std::string(argv[1]) == "runcache") {
        return RunCache(argv[2]);
    }
    if (argc >= 5 && std::string(argv[1]) == "bench") {
        return Bench(argv[2], std::stoull(argv[3], nullptr, 0), std::stoull(argv[4], nullptr, 0), load_path, save_path);
    }
    if (argc >= 3 && std::string(argv[1]) == "run") {
        return Run(argv[2]);
    }
    std::fprintf(stderr, "usage: %s coverage <words.bin> <out.bin> | run <cases.txt>\n", argv[0]);
    return 2;
}
