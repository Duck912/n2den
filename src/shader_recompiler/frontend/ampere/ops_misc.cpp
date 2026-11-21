// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Encoding reference: Mesa NAK sm70_encode.rs (MIT), Copyright 2022 Collabora, Ltd.

#include <fmt/format.h>

#include "shader_recompiler/frontend/ampere/visitor.h"
#include "shader_recompiler/frontend/ir/attribute.h"
#include "shader_recompiler/frontend/ir/patch.h"

namespace Shader::Ampere {

namespace {
// Patch attribute words IR::Patch models: the tessellation levels (words 0..5) and the generic
// components after them, mapped word for word as the Maxwell front end maps ALD.P / AST.P.
constexpr u32 PATCH_WORDS = static_cast<u32>(IR::Patch::Component119) + 1;
} // Anonymous namespace

// ALD: attribute load (vertex / geometry inputs, tessellation inputs).
// Fields: Rd 16..23, Ra (index) 24..31, vertex 32..39, address 40..49, components - 1 at 74..75,
// .P (patch) 76, .PHYS 77, .O (output) 79.
void Visitor::ALD() {
    const u32 dest = Take(16, 8);
    const u32 offset_reg = Take(24, 8);
    const u32 vertex_reg = Take(32, 8);
    const u32 address = Take(40, 10);
    const u32 components = Take(74, 2) + 1;
    const bool patch = Take(76, 1) != 0;
    const bool physical = Take(77, 1) != 0;
    const bool output = Take(79, 1) != 0;
    const bool extended = NativeExtensions();
    Require(offset_reg == 255 || extended, "native indexed attribute access");
    Require(!physical && (!patch || extended), "native patch or physical attribute access");
    if (patch) {
        // Tessellation control programs read their own patch outputs back (.O.P), evaluation
        // programs their patch inputs (.P), as GetPatch reads them in each stage.
        const Stage stage = env.ShaderStage();
        Require(offset_reg == 255, "native indexed patch attribute access");
        Require(stage == Stage::TessellationControl || stage == Stage::TessellationEval,
                "native patch attribute stage");
        Require(output == (stage == Stage::TessellationControl), "native patch attribute direction");
        Require(address % 4 == 0 && address / 4 + components <= PATCH_WORDS,
                "native patch attribute address");
        // The tessellation levels are only written (the backend has no reads of them).
        Require(address / 4 >= static_cast<u32>(IR::Patch::Component0), "native tessellation level read");
        Require(dest + components <= 255, "native attribute register range");
        for (u32 element = 0; element < components; ++element) {
            F(dest + element, ir.GetPatch(IR::Patch{address / 4 + element}));
        }
        return;
    }
    Require(!output, "native output attribute read");
    if (offset_reg != 255) {
        // a[Ra + address]: a per-vertex attribute selected at run time (indexed geometry or
        // tessellation inputs), read as the Maxwell front end reads an indexed ALD.
        Require(address % 4 == 0, "native attribute address");
        Require(dest + components <= 255, "native attribute register range");
        const IR::U32 vertex{X(vertex_reg)};
        const IR::U32 base{ir.IAdd(X(offset_reg), ir.Imm32(address))};
        for (u32 element = 0; element < components; ++element) {
            const IR::U32 element_address{element == 0 ? base : IR::U32{ir.IAdd(base, ir.Imm32(element * 4))}};
            F(dest + element, ir.GetAttributeIndexed(element_address, vertex));
        }
        return;
    }
    Require(address % 4 == 0 && address + components * 4 <= 0x400, "native attribute address");
    Require(dest + components <= 255, "native attribute register range");
    const IR::U32 vertex{X(vertex_reg)};
    for (u32 element = 0; element < components; ++element) {
        const IR::Attribute attribute{address / 4 + element};
        F(dest + element, ir.GetAttribute(attribute, vertex));
    }
}

// AST: attribute store (vertex outputs; tessellation control patch outputs with .P).
// Fields: Ra (index) 24..31, data 32..39, address 40..49, vertex 64..71, components - 1 at 74..75,
// .P (patch) 76, .PHYS 77.
void Visitor::AST() {
    Require(Take(24, 8) == 255, "native indexed attribute access");
    const u32 data_reg = Take(32, 8);
    const u32 address = Take(40, 10);
    const u32 vertex_reg = Take(64, 8);
    const u32 components = Take(74, 2) + 1;
    const bool patch = Take(76, 1) != 0;
    const bool physical = Take(77, 1) != 0;
    Require(!physical && (!patch || NativeExtensions()), "native patch or physical attribute access");
    if (patch) {
        // Patch outputs (tessellation levels and generic patch components) of a control program.
        Require(env.ShaderStage() == Stage::TessellationControl, "native patch attribute stage");
        Require(address % 4 == 0 && address / 4 + components <= PATCH_WORDS,
                "native patch attribute address");
        Require(data_reg == 255 || data_reg + components <= 255, "native attribute register range");
        for (u32 element = 0; element < components; ++element) {
            ir.SetPatch(IR::Patch{address / 4 + element}, F(data_reg == 255 ? 255 : data_reg + element));
        }
        return;
    }
    Require(address % 4 == 0 && address + components * 4 <= 0x400, "native attribute address");
    // RZ as the data source stores zero for every component.
    Require(data_reg == 255 || data_reg + components <= 255, "native attribute register range");
    const IR::U32 vertex{X(vertex_reg)};
    for (u32 element = 0; element < components; ++element) {
        const IR::Attribute attribute{address / 4 + element};
        ir.SetAttribute(attribute, F(data_reg == 255 ? 255 : data_reg + element), vertex);
    }
}

// OUT (0x324 stream in Rb, 0x924 stream immediate): geometry output. Rd 16..23 receives the new
// output handle, Ra 24..31 holds the old one, the stream is Rb 32..39 or the immediate 32..63, and
// 78..79 select emit (1), cut (2) or both (3); 0 is OUTFINAL. The host keeps no output handles: as
// the Maxwell OUT translation does, emit / cut become EmitVertex / EndPrimitive on stream & 3 and
// Rd reads zero afterwards; OUTFINAL copies Ra to Rd, as the lowering does (lower_to_maxwell.h).
void Visitor::OUT() {
    if (env.ShaderStage() != Stage::Geometry || !NativeExtensions()) {
        Unsupported(fmt::format("unsupported native opcode {:#x}", op)); // as before these existed
    }
    const u32 dest = Take(16, 8);
    const u32 handle = Take(24, 8);
    const IR::U32 stream_value{op == 0x324 ? X(Take(32, 8)) : ir.Imm32(Take(32, 32))};
    const u32 type = Take(78, 2);
    if (type == 0) {
        X(dest, X(handle));
        return;
    }
    const IR::U32 stream{ir.BitwiseAnd(stream_value, ir.Imm32(3U))};
    if ((type & 1) != 0) {
        ir.EmitVertex(stream);
    }
    if ((type & 2) != 0) {
        ir.EndPrimitive(stream);
    }
    X(dest, ir.Imm32(0U));
}

// LEPC (geometry programs): this instruction's address as a register pair. The host has no guest
// GPU address for the program, so the pair holds the instruction's byte offset from the first native
// instruction (index * 16, high word zero). That depends on the instructions alone, so identical
// programs at different addresses share their verdict, layout and IR safely. The only consumer seen
// is the geometry exit stub setup (0x357), which the host drops.
void Visitor::LEPC() {
    if (env.ShaderStage() != Stage::Geometry || !NativeExtensions()) {
        Unsupported(fmt::format("unsupported native opcode {:#x}", op)); // as before these existed
    }
    const u32 dest = Take(16, 8);
    Require(GuardOf(insn).Always(), "native program counter load mode");
    Require(dest == 255 || dest < 254, "native program counter load mode");
    if (dest == 255) {
        return;
    }
    X(dest, ir.Imm32(index * static_cast<u32>(sizeof(Instruction))));
    X(dest + 1, ir.Imm32(0U));
}

// 0x357 sets the native geometry exit stub (the address register at 24..31); 0x957 finishes it
// before the program's own EXIT and has no fields (ISA rows: ampere_isa_table.h). Host geometry
// invocations need no stub, so both have no effect, as in the lowering (lower_to_maxwell.h).
void Visitor::GeometryExitStub() {
    if (env.ShaderStage() != Stage::Geometry || !NativeExtensions()) {
        Unsupported(fmt::format("unsupported native opcode {:#x}", op)); // as before these existed
    }
    if (op == 0x357) {
        Require(GuardOf(insn).Always(), "native geometry exit setup mode");
        Ignore(24, 8);
    }
}

// LDC: indexed constant buffer load, 32 or 64 bits.
void Visitor::LDC() {
    const u32 dest = Take(16, 8);
    const u32 offset_reg = Take(24, 8);
    const s32 offset = static_cast<s16>(Take(38, 16));
    const u32 cbuf_index = Take(54, 5);
    const u32 width = Take(73, 3);
    Require(Take(78, 2) == 0, "native indexed constant load mode");
    Require(Take(91, 1) == 0, "native bindless constant load");
    Require(width <= 6, "native constant load width");
    Require(cbuf_index < 18, "native constant buffer index");
    const IR::U32 binding{ir.Imm32(cbuf_index)};
    const IR::U32 byte_offset{ir.IAdd(X(offset_reg), ir.Imm32(static_cast<u32>(offset)))};
    if (width < 4) {
        // 8 or 16 bit element, zero or sign extended (types U8, S8, U16, S16).
        X(dest, IR::U32{ir.GetCbuf(binding, byte_offset, width < 2 ? 8 : 16, (width & 1) != 0)});
    } else if (width == 4) {
        X(dest, IR::U32{ir.GetCbuf(binding, byte_offset, 32, false)});
    } else if (width == 5) {
        Require(dest % 2 == 0 && dest < 254, "native constant load destination");
        const IR::Value vector{ir.GetCbuf(binding, byte_offset, 64, false)};
        for (u32 i = 0; i < 2; ++i) {
            X(dest + i, IR::U32{ir.CompositeExtract(vector, i)});
        }
    } else {
        Require(dest % 4 == 0 && dest < 252, "native constant load destination");
        for (u32 i = 0; i < 4; ++i) {
            const IR::U32 word_offset{ir.IAdd(byte_offset, ir.Imm32(i * 4))};
            X(dest + i, IR::U32{ir.GetCbuf(binding, word_offset, 32, false)});
        }
    }
}

// CS2R: special register pair read; only SRZ (a constant zero) is modelled.
void Visitor::CS2R() {
    const u32 dest = Take(16, 8);
    Require(Take(72, 8) == 255, "native special register pair");
    const bool wide = Take(80, 1) != 0;
    X(dest, ir.Imm32(0U));
    if (wide) {
        Require(dest + 1 < 255, "native special register pair destination");
        X(dest + 1, ir.Imm32(0U));
    }
}

// S2R: thread and workgroup identities and the other special registers the Maxwell S2R reads.
void Visitor::S2R() {
    const u32 dest = Dst();
    const u32 special = Take(72, 8);
    switch (special) {
    case 0:
        WriteDst(dest, ir.LaneId());
        break;
    case 17: // SR_INVOCATION_ID
        WriteDst(dest, ir.InvocationId());
        break;
    case 18: // SR_Y_DIRECTION: the bits of a float
        WriteDst(dest, ir.BitCast<IR::U32>(ir.YDirection()));
        break;
    case 19: // SR_THREAD_KILL: all ones in a helper invocation
        WriteDst(dest, IR::U32{ir.Select(ir.IsHelperInvocation(), ir.Imm32(0xffffffffU), ir.Imm32(0U))});
        break;
    case 29: // SR_INVOCATION_INFO
        WriteDst(dest, ir.InvocationInfo());
        break;
    case 32: { // SR_TID: x in bits 0..15, y in 16..23, z in 26..31
        const IR::Value tid{ir.LocalInvocationId()};
        WriteDst(dest, ir.BitFieldInsert(ir.BitFieldInsert(IR::U32{ir.CompositeExtract(tid, 0)},
                                                           IR::U32{ir.CompositeExtract(tid, 1)},
                                                           ir.Imm32(16), ir.Imm32(8)),
                                         IR::U32{ir.CompositeExtract(tid, 2)}, ir.Imm32(26),
                                         ir.Imm32(6)));
        break;
    }
    case 33:
        WriteDst(dest, ir.LocalInvocationIdX());
        break;
    case 34:
        WriteDst(dest, ir.LocalInvocationIdY());
        break;
    case 35:
        WriteDst(dest, ir.LocalInvocationIdZ());
        break;
    case 37:
        WriteDst(dest, ir.WorkgroupIdX());
        break;
    case 38:
        WriteDst(dest, ir.WorkgroupIdY());
        break;
    case 39:
        WriteDst(dest, ir.WorkgroupIdZ());
        break;
    case 30:
    case 31: // SR_WSCALEFACTOR_Z: read next to XY and multiplied into a float (FMUL) in the Kirby shaders
        // SR_WSCALEFACTOR_XY (assumed from the Maxwell/Volta register numbering, not in NAK). Without
        // viewport W scaling the factor is the identity; the 32-bit float 1.0 is an unverified assumption.
        WriteDst(dest, ir.Imm32(0x3f800000U));
        break;
    case 56:
        WriteDst(dest, ir.SubgroupEqMask());
        break;
    case 57:
        WriteDst(dest, ir.SubgroupLtMask());
        break;
    case 58:
        WriteDst(dest, ir.SubgroupLeMask());
        break;
    case 59:
        WriteDst(dest, ir.SubgroupGtMask());
        break;
    case 60:
        WriteDst(dest, ir.SubgroupGeMask());
        break;
    case 132:
        // SR_VARIABLE_RATE: without variable-rate shading every fragment is 1x1, its width and height
        // in bytes 1 and 2 (the game multiplies them). Same constant as the lowering.
        WriteDst(dest, ir.Imm32(0x00010100U));
        break;
    default:
        Unsupported("native special register");
    }
}

// UMOV: move an immediate or a uniform register into a uniform register.
void Visitor::UMOV() {
    const u32 dest = Dst();
    if (op == 0x882) {
        UR(dest, ir.Imm32(Take(32, 32)));
    } else {
        const u32 source = Take(32, 6);
        Require(Take(91, 1) == 1, "native uniform move flag");
        UR(dest, UR(source));
    }
}

// ULDC: constant buffer load into uniform registers; 0xabb indexes the offset with a uniform register.
void Visitor::ULDC() {
    const u32 dest = Dst();
    const bool indexed = op == 0xabb;
    const u32 index_reg = indexed ? Take(24, 6) : 63;
    if (!indexed) {
        Ignore(24, 6);
    }
    const u32 offset = Take(38, 16);
    const u32 cbuf_index = Take(54, 5);
    const u32 width = Take(73, 3);
    Require(Take(91, 1) == (indexed ? 1U : 0U), "native uniform constant load flag");
    Require(width <= 6, "native uniform constant load width");
    // The offset is aligned to the element: ULDC.U8 c[0x4][UR6+0x82] is a legal byte load
    // (Ocarina of Time's foliage computes).
    const u32 element_bytes{width < 2 ? 1U : width < 4 ? 2U : 4U};
    // Indexed loads only need the sum aligned (ULDC UR7, c[0x4][UR15+0x29]): not checked here.
    Require(cbuf_index < 18 && (indexed || offset % element_bytes == 0),
            "native uniform constant buffer address");
    const IR::U32 binding{ir.Imm32(cbuf_index)};
    const IR::U32 byte_offset{indexed ? IR::U32{ir.IAdd(UR(index_reg), ir.Imm32(offset))} : ir.Imm32(offset)};
    if (width < 4) {
        // 8 or 16 bit element, zero or sign extended (types U8, S8, U16, S16).
        UR(dest, IR::U32{ir.GetCbuf(binding, byte_offset, width < 2 ? 8 : 16, (width & 1) != 0)});
    } else if (width == 4) {
        UR(dest, IR::U32{ir.GetCbuf(binding, byte_offset, 32, false)});
    } else if (width == 5) {
        Require(dest % 2 == 0 && dest < 62, "native uniform constant load destination");
        const IR::Value vector{ir.GetCbuf(binding, byte_offset, 64, false)};
        for (u32 i = 0; i < 2; ++i) {
            UR(dest + i, IR::U32{ir.CompositeExtract(vector, i)});
        }
    } else {
        Require(dest % 4 == 0 && dest < 60, "native uniform constant load destination");
        for (u32 i = 0; i < 4; ++i) {
            const IR::U32 word_offset{ir.IAdd(byte_offset, ir.Imm32(i * 4))};
            UR(dest + i, IR::U32{ir.GetCbuf(binding, word_offset, 32, false)});
        }
    }
}

// MATCH.ALL on a 32 or 64 bit value with only the all-equal predicate kept (no result register): the
// predicate says whether every active lane holds the same value as the first active lane.
void Visitor::MATCH() {
    const u32 dest = Take(16, 8);
    const u32 source = Take(24, 8);
    const bool wide = Take(73, 1) != 0;
    const bool any = Take(79, 1) != 0;
    const u32 predicate = Take(81, 3);
    Require(dest == 255 && !any, "native warp match mode");
    Require(source < 255 && (!wide || (source % 2 == 0 && source + 1 < 255)), "native warp match source");
    const IR::U32 active{ir.SubgroupBallot(ir.Imm1(true))};
    const IR::U32 first_lane{ir.FindUMsb(ir.BitwiseAnd(active, ir.INeg(active)))};
    IR::U1 equal{ir.IEqual(X(source), ShuffleLane(X(source), first_lane))};
    if (wide) {
        equal = ir.LogicalAnd(equal, ir.IEqual(X(source + 1), ShuffleLane(X(source + 1), first_lane)));
    }
    PredOut(predicate, ir.VoteAll(equal));
}

} // namespace Shader::Ampere
