// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Relocatable x64 code for the persistent block cache.
//
// A block emitted while recording keeps everything that depends on where things are in memory out of its bytes,
// or describes it with a relocation:
// - constants live in the block itself (rip-relative to labels in the block), not in the shared constant pool;
// - calls to host functions use `mov rax, imm64; call rax`, and branches to absolute targets use rel32 forms, so
//   instruction lengths do not depend on distances;
// - every embedded host address is recorded: a function or static in this module (relative to the module base),
//   a prelude routine (relative to the start of the code space), or a JIT object such as the callbacks or a
//   monitor slot (a symbol, resolved for the core that loads the block);
// - links to other blocks are patch sites, rewritten when the block is loaded;
// - fastmem fault entries are kept relative to the block.
// Anything else that looks like a host address marks the block as not relocatable; it is then cached as IR only.

#pragma once

#include <deque>
#include <optional>
#include <vector>

#include "common/common_types.h"
#include "dynarmic/backend/x64/xbyak.h"

namespace Dynarmic::Backend::X64 {

enum class NativeRelocKind : u8 {
    CodeRel32,    // rel32 to code space start + value (a prelude routine)
    ModuleRel32,  // rel32 to module base + value
    CodeAbs64,    // imm64 = code space start + value
    ModuleAbs64,  // imm64 = module base + value
    SymbolAbs64,  // imm64 = symbol `value` of the loading JIT (see NativeSymbol)
};

struct NativeReloc {
    u32 offset;  // of the 4- or 8-byte field, from the block start
    NativeRelocKind kind;
    u64 value;
};

/// JIT objects referenced by emitted code. The low byte of a symbol id is an argument.
enum class NativeSymbol : u32 {
    Callbacks = 1 << 8,           // UserConfig::callbacks
    Conf = 2 << 8,                // the emitter's UserConfig
    TpidrEl0 = 3 << 8,            // UserConfig::tpidr_el0
    TpidrroEl0 = 4 << 8,          // UserConfig::tpidrro_el0
    MonitorLock = 5 << 8,         // exclusive monitor lock
    MonitorAddressSelf = 6 << 8,  // this core's exclusive address slot
    MonitorValueSelf = 7 << 8,    // this core's exclusive value slot
    MonitorAddressOther = 8 << 8, // + n: the address slot of the n-th other core, in index order
};

inline u32 MakeSymbol(NativeSymbol symbol, u32 argument = 0) {
    return static_cast<u32>(symbol) | (argument & 0xFF);
}

struct NativePatchSite {
    u32 offset;
    u32 kind;  // EmitX64::PatchKind
    u64 target;  // IR::LocationDescriptor value
};

struct NativeFastmemEntry {
    u32 rip;          // from the block start
    u32 resume_rip;   // from the block start
    u32 callback;     // from the code space start (a prelude routine)
    u32 inst_name;
    bool recompile;
    u64 location;     // IR::LocationDescriptor value of the marker
};

/// What a recorded block needs to be placed elsewhere.
struct NativeBlock {
    std::vector<u8> code;
    std::vector<NativeReloc> relocs;
    std::vector<NativePatchSite> patch_sites;
    std::vector<NativeFastmemEntry> fastmem;
};

/// Recording state while a block is emitted (owned by BlockOfCode).
struct NativeRecording {
    const u8* begin = nullptr;
    std::vector<NativeReloc> relocs;
    std::vector<NativePatchSite> patch_sites;
    std::vector<NativeFastmemEntry> fastmem;
    std::deque<Xbyak::Label> constant_labels;
    std::vector<std::pair<u64, u64>> constant_values;
    const char* not_relocatable = nullptr;  // reason, when the block cannot be cached as code
};

/// Size in bytes of each patch site kind (see A64EmitX64::EmitPatch*).
inline u32 PatchSiteSize(u32 kind) {
    constexpr u32 sizes[] = {23, 23, 22, 10};  // Jg, Jz, Jmp, MovRcx
    return kind < 4 ? sizes[kind] : 0;
}

/// [base, end) of the module that holds this code, empty if it cannot be determined.
std::pair<u64, u64> NativeModuleRange();

/// Identifies the module's layout (PE timestamp, image size, entry point): offsets of functions recorded by one
/// executable are meaningless in another, such as eden.exe and eden-cli.exe of the same build.
u64 NativeModuleIdentity();

}  // namespace Dynarmic::Backend::X64
