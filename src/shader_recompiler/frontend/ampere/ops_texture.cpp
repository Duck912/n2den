// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Encoding reference: Mesa NAK sm70_encode.rs (MIT), Copyright 2022 Collabora, Ltd.

#include <array>
#include <bit>
#include <functional>

#include "shader_recompiler/frontend/ampere/visitor.h"
#include "shader_recompiler/frontend/ir/modifiers.h"

namespace Shader::Ampere {
namespace {
struct Dimension {
    Shader::TextureType type;
    u32 coordinates; // including the array index
};

Dimension DimensionOf(u32 value) {
    switch (value) {
    case 0:
        return {Shader::TextureType::Color1D, 1};
    case 4:
        return {Shader::TextureType::ColorArray1D, 2};
    case 1:
        return {Shader::TextureType::Color2D, 2};
    case 5:
        return {Shader::TextureType::ColorArray2D, 3};
    case 2:
        return {Shader::TextureType::Color3D, 3};
    case 3:
        return {Shader::TextureType::ColorCube, 3};
    case 7:
        return {Shader::TextureType::ColorArrayCube, 4};
    default:
        return {Shader::TextureType::Color2D, 0};
    }
}
} // Anonymous namespace

void Visitor::TEX() {
    const bool bindless = op == 0xf60; // handle in a uniform register instead of a constant buffer slot
    const bool register_handle = op == 0x361; // handle in a general register after the operands
    const u32 dest = Take(16, 8);
    const u32 source_a = Take(24, 8);
    const u32 source_b = Take(32, 8);
    u32 cbuf_offset = 0;
    u32 cbuf_index = 0;
    u32 handle_register = 0;
    if (register_handle) {
        Require(Take(59, 1) == 1, "native bindless texture handle");
    } else if (bindless) {
        handle_register = Take(40, 6);
        Require(Take(91, 1) == 1 && handle_register < 63, "native bindless texture handle");
    } else {
        cbuf_offset = Take(40, 14) * 4;
        cbuf_index = Take(54, 5);
        Require(Take(59, 1) == 0, "native bindless texture");
    }
    // Without the scalar flag the coordinates are consecutive registers at A and the lod, bias or depth
    // reference sits at B (the vector layout); bit 79 selects half precision results.
    const bool scalar = Take(60, 1) != 0;
    const Dimension dim = DimensionOf(Take(61, 3));
    Require(dim.coordinates != 0, "native texture dimension");
    const u32 dest_b = Take(64, 8);
    const u32 mask = Take(72, 4);
    // Bit 76 adds a register of packed texel offsets after the lod register (or at B when there is
    // none): signed 4-bit fields, x in bits 0..3, y in 4..7 and (3D) z in 8..11, exactly as the
    // Maxwell translation reads them.
    const bool texel_offset = Take(76, 1) != 0;
    Require(Take(77, 1) == 0, "native non-divergent texture derivative");
    const bool depth_compare = Take(78, 1) != 0;
    const bool half = Take(79, 1) != 0;
    Require(Take(81, 3) == 7, "native sparse texture residency");
    Ignore(84, 3); // cache eviction priority
    const u32 lod_mode = Take(87, 3);
    Ignore(90, 1); // nodep scoreboard hint
    Require(lod_mode <= 3, "native texture clamp mode");
    Require(mask != 0, "native empty texture channel mask");
    if (!bindless && !register_handle) {
        // The descriptor table is the constant buffer named by the binding field.
        UseDescriptorBuffer(cbuf_index);
    }

    // Operands are one list (array index first, then coordinates, lod or bias, depth reference)
    // spread over two register groups: a single register each for two operands, otherwise a pair
    // at A followed by the remainder at B.
    const bool has_lod = lod_mode == 2 || lod_mode == 3;
    const u32 count = dim.coordinates + (has_lod ? 1 : 0) + (depth_compare ? 1 : 0);
    Require(count <= 6, "native texture operand count");
    std::array<u32, 8> regs{};
    u32 gpr_handle = 0;
    u32 offset_register = source_b;
    if (register_handle) {
        // The handle is one more value right after the coordinates: the operand list is
        // coordinates, handle, lod or bias, texel offsets, depth reference, as LowerToMaxwell reads
        // it. The scalar layout splits up to four values in halves over A and B: OoT's shadow
        // lookups, TEX.B.DC 2D, are Ra = x, y and Rb = handle (low word of a 64-bit handle loaded
        // with LDG), reference; Fortune's Weave's TEX.B.LZ CUBE is Ra = x, y, Rb = z, handle, and its
        // TEX.B.AOFFI 2D is Ra = x, y, Rb = handle, offsets. With five or more the coordinates keep
        // A, as they always do in the vector layout (TEX.B.LL CUBE: Ra = x, y, z, Rb = handle, lod).
        Require(!half, "native bindless texture operands");
        Require(!(texel_offset && depth_compare), "native texture offset");
        const u32 values = count + 1 + (texel_offset ? 1 : 0);
        const u32 in_a = !scalar || values > 4 ? dim.coordinates : (values + 1) / 2;
        const auto value_register = [&](u32 index) {
            const u32 reg = index < in_a ? source_a + index : source_b + (index - in_a);
            Require(reg < 255, "native texture operand register");
            return reg;
        };
        u32 position = 0;
        u32 next_operand = 0;
        for (; next_operand < dim.coordinates; ++next_operand) {
            regs[next_operand] = value_register(position++);
        }
        gpr_handle = value_register(position++);
        if (has_lod) {
            regs[next_operand++] = value_register(position++);
        }
        if (texel_offset) {
            offset_register = value_register(position++);
        }
        if (depth_compare) {
            regs[next_operand++] = value_register(position++);
        }
    } else if (texel_offset) {
        // Coordinates start at A. A lod register, when there is one (explicit lod), is B and the
        // offset register follows it (B + 1); otherwise the offset register is B.
        Require(!half && !depth_compare, "native texture offset");
        if (scalar) {
            Require(dim.type == Shader::TextureType::Color2D && (lod_mode <= 1 || lod_mode == 3) &&
                        source_a % 2 == 0 && source_a + 1 < 255 && source_b + 1 < 255 &&
                        (lod_mode != 3 || source_b % 2 == 0),
                    "native texture offset");
            regs[0] = source_a;
            regs[1] = source_a + 1;
            if (lod_mode == 3) {
                regs[2] = source_b;
                offset_register = source_b + 1;
            }
        } else {
            // Vector layout, seen as TEX.LL.AOFFI 3D: x, y, z in the four-register group at A, the
            // explicit lod at B and the packed offsets at B + 1 (the programs load a float lod into
            // B and a small integer into B + 1, and never write A + 3).
            Require(dim.type == Shader::TextureType::Color3D && lod_mode == 3 && source_a % 4 == 0 &&
                        source_a + 2 < 255 && source_b % 2 == 0 && source_b + 1 < 255,
                    "native texture offset");
            for (u32 k = 0; k < 3; ++k) {
                regs[k] = source_a + k;
            }
            regs[3] = source_b;
            offset_register = source_b + 1;
        }
    } else if (scalar) {
        for (u32 k = 0; k < count; ++k) {
            u32 reg;
            if (count <= 2) {
                reg = k == 0 ? source_a : source_b;
            } else {
                reg = k < 2 ? source_a + k : source_b + (k - 2);
            }
            Require(reg < 255, "native texture operand register");
            regs[k] = reg;
        }
    } else {
        // Vector layout: every coordinate (array index first) follows from A, then the one extra
        // operand, if any, is B.
        Require(count <= dim.coordinates + 1, "native vector texture operand count");
        for (u32 k = 0; k < count; ++k) {
            regs[k] = k < dim.coordinates ? source_a + k : source_b;
            Require(regs[k] < 255, "native texture operand register");
        }
    }
    u32 next = 0;
    const auto array_index = [&]() -> IR::F32 {
        const IR::U32 raw = X(regs[next++]);
        return IR::F32{
            ir.ConvertUToF(32, 16, ir.BitFieldExtract(raw, ir.Imm32(0), ir.Imm32(16)))};
    };
    IR::Value coords;
    switch (dim.type) {
    case Shader::TextureType::Color1D:
        coords = F(regs[next++]);
        break;
    case Shader::TextureType::ColorArray1D: {
        const IR::F32 layer = array_index();
        coords = ir.CompositeConstruct(F(regs[next++]), layer);
        break;
    }
    case Shader::TextureType::Color2D: {
        const IR::F32 x = F(regs[next++]);
        coords = ir.CompositeConstruct(x, F(regs[next++]));
        break;
    }
    case Shader::TextureType::ColorArray2D: {
        const IR::F32 layer = array_index();
        const IR::F32 x = F(regs[next++]);
        coords = ir.CompositeConstruct(x, F(regs[next++]), layer);
        break;
    }
    case Shader::TextureType::Color3D:
    case Shader::TextureType::ColorCube: {
        const IR::F32 x = F(regs[next++]);
        const IR::F32 y = F(regs[next++]);
        coords = ir.CompositeConstruct(x, y, F(regs[next++]));
        break;
    }
    case Shader::TextureType::ColorArrayCube: {
        const IR::F32 layer = array_index();
        const IR::F32 x = F(regs[next++]);
        const IR::F32 y = F(regs[next++]);
        coords = ir.CompositeConstruct(x, y, F(regs[next++]), layer);
        break;
    }
    default:
        Unsupported("native texture dimension");
    }
    IR::F32 lod = ir.Imm32(0.0f);
    if (has_lod) {
        lod = F(regs[next++]);
    }
    IR::F32 dref;
    if (depth_compare) {
        dref = F(regs[next++]);
    }

    IR::TextureInstInfo info{};
    info.type.Assign(dim.type);
    info.is_depth.Assign(depth_compare ? 1 : 0);
    info.has_bias.Assign(lod_mode == 2 ? 1 : 0);
    info.relaxed_precision.Assign(half ? 1 : 0);
    const IR::U32 handle = register_handle ? X(gpr_handle)
                                           : bindless ? UR(handle_register) : DescriptorHandle(cbuf_offset);
    const bool explicit_lod = lod_mode == 1 || lod_mode == 3;
    const IR::F32 bias = lod_mode == 2 ? lod : IR::F32{};
    IR::Value offset;
    if (texel_offset) {
        const IR::U32 packed{X(offset_register)};
        const auto field = [&](u32 k) {
            return IR::U32{ir.BitFieldExtract(packed, ir.Imm32(4 * k), ir.Imm32(4), true)};
        };
        const IR::U32 offset_x{field(0)};
        const IR::U32 offset_y{field(1)};
        offset = dim.type == Shader::TextureType::Color3D
                     ? ir.CompositeConstruct(offset_x, offset_y, field(2))
                     : ir.CompositeConstruct(offset_x, offset_y);
    }
    const IR::Value sample{[&]() -> IR::Value {
        if (!depth_compare) {
            if (explicit_lod) {
                return ir.ImageSampleExplicitLod(handle, coords, lod, offset, info);
            }
            return ir.ImageSampleImplicitLod(handle, coords, bias, offset, {}, info);
        }
        if (explicit_lod) {
            return ir.ImageSampleDrefExplicitLod(handle, coords, dref, lod, {}, info);
        }
        return ir.ImageSampleDrefImplicitLod(handle, coords, dref, bias, {}, {}, info);
    }()};

    // Channels in R,G,B,A order are packed into dest, dest + 1, then dest_b, dest_b + 1.
    const u32 outputs = static_cast<u32>(std::popcount(mask));
    Require(outputs <= 2 ? dest_b == 255 : dest_b != 255, "native texture result layout");
    u32 stored = 0;
    std::array<IR::F32, 4> values;
    for (u32 component = 0; component < 4; ++component) {
        if (((mask >> component) & 1) == 0) {
            continue;
        }
        IR::F32 value;
        if (depth_compare) {
            value = component < 3 ? IR::F32{sample} : ir.Imm32(1.0f);
        } else {
            value = IR::F32{ir.CompositeExtract(sample, component)};
        }
        if (!half) {
            const u32 target = stored < 2 ? dest + stored : dest_b + (stored - 2);
            Require(target < 255, "native texture result register");
            F(target, value);
        } else {
            values[stored] = value;
        }
        ++stored;
    }
    if (half) {
        // Half precision results: pairs of channels, in order, share one register (low half first);
        // a missing last channel reads as zero, as for the Maxwell TEXS.F16 encoding.
        const auto channel = [&](u32 k) -> IR::F32 {
            return k < stored ? values[k] : IR::F32{ir.Imm32(0.0f)};
        };
        Require(dest < 255, "native texture result register");
        X(dest, ir.PackHalf2x16(ir.CompositeConstruct(channel(0), channel(1))));
        if (stored > 2) {
            Require(dest_b < 255, "native texture result register");
            X(dest_b, ir.PackHalf2x16(ir.CompositeConstruct(channel(2), channel(3))));
        }
    }
}

void Visitor::TLD() {
    const bool bindless = op == 0xf66; // handle in a uniform register instead of a constant buffer slot
    const u32 dest = Take(16, 8);
    const u32 source_a = Take(24, 8);
    const u32 source_b = Take(32, 8);
    u32 cbuf_offset = 0;
    u32 cbuf_index = 0;
    u32 handle_register = 0;
    if (bindless) {
        handle_register = Take(40, 6);
        Require(Take(91, 1) == 1 && handle_register < 63, "native bindless texture handle");
    } else {
        cbuf_offset = Take(40, 14) * 4;
        cbuf_index = Take(54, 5);
        Require(Take(59, 1) == 0, "native bindless texture");
    }
    Require(Take(60, 1) != 0, "native packed half-precision texel fetch result");
    const Dimension dim = DimensionOf(Take(61, 3));
    Require(dim.coordinates != 0, "native texture dimension");
    const u32 dest_b = Take(64, 8);
    const u32 mask = Take(72, 4);
    // Bit 76 (.AOFFI): a register of packed texel offsets at B, two signed 4-bit fields (x in bits
    // 0..3, y in 4..7), with both coordinates at A. Only the 2D level-zero form the lowering maps to
    // the Maxwell TLDS offset encoding is modelled.
    const bool texel_offset = Take(76, 1) != 0;
    Require(Take(77, 2) == 0, "native texel fetch multisample or clamp");
    const bool half = Take(79, 1) != 0; // bit 79: half precision results (.F16)
    Require(Take(81, 3) == 7, "native sparse texture residency");
    Ignore(84, 3);
    const u32 lod_mode = Take(87, 3);
    Ignore(90, 1);
    Require(lod_mode == 1 || lod_mode == 3, "native texel fetch lod mode");
    Require(mask != 0, "native empty texture channel mask");
    if (!bindless) {
        UseDescriptorBuffer(cbuf_index);
    }
    const bool has_lod = lod_mode == 3;
    const u32 count = dim.coordinates + (has_lod ? 1 : 0);
    std::array<u32, 8> regs{};
    if (texel_offset) {
        Require(dim.type == Shader::TextureType::Color2D && lod_mode == 1 && source_a % 2 == 0 &&
                    source_a + 1 < 255 && source_b < 255,
                "native texel fetch offset");
        regs[0] = source_a;
        regs[1] = source_a + 1;
    } else {
        for (u32 k = 0; k < count; ++k) {
            u32 reg;
            if (count <= 2) {
                reg = k == 0 ? source_a : source_b;
            } else {
                reg = k < 2 ? source_a + k : source_b + (k - 2);
            }
            Require(reg < 255, "native texture operand register");
            regs[k] = reg;
        }
    }
    u32 next = 0;
    const auto array_index = [&]() -> IR::U32 {
        return ir.BitFieldExtract(X(regs[next++]), ir.Imm32(0), ir.Imm32(16));
    };
    IR::Value coords;
    switch (dim.type) {
    case Shader::TextureType::Color1D:
        coords = X(regs[next++]);
        break;
    case Shader::TextureType::ColorArray1D: {
        const IR::U32 layer = array_index();
        coords = ir.CompositeConstruct(X(regs[next++]), layer);
        break;
    }
    case Shader::TextureType::Color2D: {
        const IR::U32 x = X(regs[next++]);
        coords = ir.CompositeConstruct(x, X(regs[next++]));
        break;
    }
    case Shader::TextureType::ColorArray2D: {
        const IR::U32 layer = array_index();
        const IR::U32 x = X(regs[next++]);
        coords = ir.CompositeConstruct(x, X(regs[next++]), layer);
        break;
    }
    case Shader::TextureType::Color3D: {
        const IR::U32 x = X(regs[next++]);
        const IR::U32 y = X(regs[next++]);
        coords = ir.CompositeConstruct(x, y, X(regs[next++]));
        break;
    }
    default:
        Unsupported("native texel fetch dimension");
    }
    const IR::U32 lod = has_lod ? X(regs[next++]) : ir.Imm32(0U);
    IR::Value offset;
    if (texel_offset) {
        const IR::U32 packed{X(source_b)};
        const IR::U32 offset_x{ir.BitFieldExtract(packed, ir.Imm32(0), ir.Imm32(4), true)};
        offset = ir.CompositeConstruct(offset_x,
                                       ir.BitFieldExtract(packed, ir.Imm32(4), ir.Imm32(4), true));
    }
    IR::TextureInstInfo info{};
    info.type.Assign(dim.type);
    const IR::U32 handle = bindless ? UR(handle_register) : DescriptorHandle(cbuf_offset);
    const IR::Value sample{ir.ImageFetch(handle, coords, offset, lod, {}, info)};
    const u32 outputs = static_cast<u32>(std::popcount(mask));
    Require(outputs <= 2 ? dest_b == 255 : dest_b != 255, "native texture result layout");
    u32 stored = 0;
    std::array<IR::F32, 4> values;
    for (u32 component = 0; component < 4; ++component) {
        if (((mask >> component) & 1) == 0) {
            continue;
        }
        const IR::F32 value{ir.CompositeExtract(sample, component)};
        if (half) {
            values[stored] = value;
        } else {
            const u32 target = stored < 2 ? dest + stored : dest_b + (stored - 2);
            Require(target < 255, "native texture result register");
            F(target, value);
        }
        ++stored;
    }
    if (half) {
        // Pairs of channels share one register as half precision; a missing last channel is zero.
        const auto channel = [&](u32 k) -> IR::F32 {
            return k < stored ? values[k] : IR::F32{ir.Imm32(0.0f)};
        };
        Require(dest < 255, "native texture result register");
        X(dest, ir.PackHalf2x16(ir.CompositeConstruct(channel(0), channel(1))));
        if (stored > 2) {
            Require(dest_b < 255, "native texture result register");
            X(dest_b, ir.PackHalf2x16(ir.CompositeConstruct(channel(2), channel(3))));
        }
    }
}


namespace {
// Image dimension codes of surface instructions (NAK ImageDim).
Dimension ImageDimensionOf(u32 value) {
    switch (value) {
    case 0:
        return {Shader::TextureType::Color1D, 1};
    case 1:
        return {Shader::TextureType::Buffer, 1};
    case 2:
        return {Shader::TextureType::ColorArray1D, 2};
    case 3:
        return {Shader::TextureType::Color2D, 2};
    case 4:
        return {Shader::TextureType::ColorArray2D, 3};
    case 5:
        return {Shader::TextureType::Color3D, 3};
    default:
        return {Shader::TextureType::Color2D, 0};
    }
}
} // Anonymous namespace

// Registers the descriptor buffer a surface or texture instruction names.
// The first descriptor buffer a program names is its bound one: handles are immediate offsets into
// it. A texture or surface in another one (61 of Fortune's Weave's programs sample from two) is read
// like a bindless handle, from its constant buffer, and the texture pass makes a descriptor in that
// buffer of it.
void Visitor::UseDescriptorBuffer(u32 cbuf_index) {
    Require(cbuf_index < 18, "native descriptor buffer");
    descriptor_bank = cbuf_index;
    if (program_info && program_info->texture_buffer == ~0U) {
        program_info->texture_buffer = cbuf_index;
    }
}

IR::U32 Visitor::DescriptorHandle(u32 cbuf_offset) {
    const u32 bound = program_info ? program_info->texture_buffer : env.TextureBoundBuffer();
    if (descriptor_bank == bound) {
        return ir.Imm32(cbuf_offset);
    }
    return ir.GetCbuf(ir.Imm32(descriptor_bank), ir.Imm32(cbuf_offset));
}

namespace {
// Integer coordinates of a surface access starting at `reg`; an array index is the low 16 bits.
IR::Value SurfaceCoordinates(IR::IREmitter& ir, const Dimension& dim, u32 reg,
                             const std::function<IR::U32(u32)>& read) {
    const auto x = [&](u32 k) { return read(reg + k); };
    switch (dim.type) {
    case Shader::TextureType::Color1D:
    case Shader::TextureType::Buffer:
        return x(0);
    case Shader::TextureType::ColorArray1D:
        return ir.CompositeConstruct(x(0), ir.BitFieldExtract(x(1), ir.Imm32(0), ir.Imm32(16)));
    case Shader::TextureType::Color2D:
        return ir.CompositeConstruct(x(0), x(1));
    case Shader::TextureType::ColorArray2D:
        return ir.CompositeConstruct(x(0), x(1), ir.BitFieldExtract(x(2), ir.Imm32(0), ir.Imm32(16)));
    default:
        return ir.CompositeConstruct(x(0), x(1), x(2));
    }
}
} // Anonymous namespace

// SUST: formatted surface store. The data registers hold the channels selected by the mask in order;
// only masks covering the leading channels are modelled, since the store writes whole texels.
void Visitor::SUST() {
    const u32 coord_reg = Take(24, 8);
    const u32 data_reg = Take(32, 8);
    const u32 cbuf_offset = Take(40, 14) * 4;
    const u32 cbuf_index = Take(54, 5);
    Require(Take(59, 2) == 0, "native surface handle mode");
    const Dimension dim = ImageDimensionOf(Take(61, 3));
    Require(dim.coordinates != 0, "native surface dimension");
    const u32 mask = Take(72, 4);
    Ignore(76, 4); // memory ordering
    Ignore(84, 3); // cache eviction priority
    Require(mask == 1 || mask == 3 || mask == 7 || mask == 15, "native surface channel mask");
    UseDescriptorBuffer(cbuf_index);
    const u32 count = static_cast<u32>(std::popcount(mask));
    Require(data_reg + count < 255 && coord_reg + dim.coordinates < 255, "native surface registers");
    const IR::Value coords{SurfaceCoordinates(ir, dim, coord_reg, [this](u32 r) { return X(r); })};
    std::array<IR::U32, 4> colors;
    for (u32 k = 0; k < 4; ++k) {
        colors[k] = k < count ? X(data_reg + k) : ir.Imm32(0U);
    }
    IR::TextureInstInfo info{};
    info.type.Assign(dim.type);
    info.image_format.Assign(ImageFormat::Typeless);
    ir.ImageWrite(DescriptorHandle(cbuf_offset), coords,
                  ir.CompositeConstruct(colors[0], colors[1], colors[2], colors[3]), info);
}

// SULD: formatted surface load into consecutive registers, one per selected channel.
void Visitor::SULD() {
    const u32 dest = Take(16, 8);
    const u32 coord_reg = Take(24, 8);
    const u32 cbuf_offset = Take(40, 14) * 4;
    const u32 cbuf_index = Take(54, 5);
    Require(Take(59, 2) == 0, "native surface handle mode");
    const Dimension dim = ImageDimensionOf(Take(61, 3));
    Require(dim.coordinates != 0, "native surface dimension");
    const u32 mask = Take(72, 4);
    Ignore(76, 4); // memory ordering
    Require(Take(81, 3) == 7, "native sparse surface residency");
    Ignore(84, 3);
    Require(mask != 0, "native empty surface channel mask");
    UseDescriptorBuffer(cbuf_index);
    Require(coord_reg + dim.coordinates < 255, "native surface registers");
    const IR::Value coords{SurfaceCoordinates(ir, dim, coord_reg, [this](u32 r) { return X(r); })};
    IR::TextureInstInfo info{};
    info.type.Assign(dim.type);
    info.image_format.Assign(ImageFormat::Typeless);
    const IR::Value result{ir.ImageRead(DescriptorHandle(cbuf_offset), coords, info)};
    u32 stored = 0;
    for (u32 component = 0; component < 4; ++component) {
        if (((mask >> component) & 1) == 0) {
            continue;
        }
        Require(dest + stored < 255, "native surface result register");
        X(dest + stored, IR::U32{ir.CompositeExtract(result, component)});
        ++stored;
    }
}

// TLD4: gather of one component, descriptor in a constant buffer (0xb63) or a uniform register (0xf63).
void Visitor::TLD4() {
    const bool bindless = op == 0xf63;
    const u32 dest = Take(16, 8);
    const u32 source_a = Take(24, 8);
    const u32 source_b = Take(32, 8);
    u32 cbuf_offset = 0;
    u32 handle_register = 0;
    if (bindless) {
        handle_register = Take(40, 6);
        Require(Take(91, 1) == 1 && handle_register < 63, "native bindless texture handle");
    } else {
        cbuf_offset = Take(40, 14) * 4;
        UseDescriptorBuffer(Take(54, 5));
        Require(Take(59, 1) == 0, "native bindless texture");
    }
    Require(Take(60, 1) != 0, "native packed half-precision gather result");
    const Dimension dim = DimensionOf(Take(61, 3));
    Require(dim.coordinates != 0, "native texture dimension");
    const u32 dest_b = Take(64, 8);
    const u32 mask = Take(72, 4);
    // 1: one packed register of signed 6-bit offsets (x at bit 0, y at bit 8), 2: per-texel offsets in two
    // registers (four x/y pairs). The registers follow the coordinates and precede the depth reference
    // (the order of the Maxwell instruction; the combination with a reference is not seen in logs).
    const u32 offset_mode = Take(76, 2);
    Require(offset_mode <= 2, "native gather offset");
    const bool depth_compare = Take(78, 1) != 0;
    // Bit 79 (.F16): the four gathered values are packed as half pairs, (0, 1) in the first
    // destination and (2, 3) in the second. As in the lowering, only the full mask is modelled.
    const bool half = Take(79, 1) != 0;
    Require(Take(81, 3) == 7, "native sparse texture residency");
    Ignore(84, 3);
    const u32 component = Take(87, 2);
    Ignore(90, 1);
    Require(mask != 0, "native empty texture channel mask");
    const u32 offset_registers = offset_mode == 0 ? 0 : offset_mode;
    const u32 count = dim.coordinates + offset_registers + (depth_compare ? 1 : 0);
    Require(offset_mode == 0 || dim.type == Shader::TextureType::Color2D ||
                dim.type == Shader::TextureType::ColorArray2D, "native gather offset dimension");
    std::array<u32, 8> regs{};
    for (u32 k = 0; k < count; ++k) {
        const u32 reg = count <= 2 ? (k == 0 ? source_a : source_b)
                                   : (k < 2 ? source_a + k : source_b + (k - 2));
        Require(reg < 255, "native texture operand register");
        regs[k] = reg;
    }
    u32 next = 0;
    IR::Value coords;
    switch (dim.type) {
    case Shader::TextureType::Color2D: {
        const IR::F32 x = F(regs[next++]);
        coords = ir.CompositeConstruct(x, F(regs[next++]));
        break;
    }
    case Shader::TextureType::ColorArray2D: {
        const IR::F32 layer{
            ir.ConvertUToF(32, 16, ir.BitFieldExtract(X(regs[next++]), ir.Imm32(0), ir.Imm32(16)))};
        const IR::F32 x = F(regs[next++]);
        coords = ir.CompositeConstruct(x, F(regs[next++]), layer);
        break;
    }
    case Shader::TextureType::ColorCube: {
        const IR::F32 x = F(regs[next++]);
        const IR::F32 y = F(regs[next++]);
        coords = ir.CompositeConstruct(x, y, F(regs[next++]));
        break;
    }
    default:
        Unsupported("native gather dimension");
    }
    IR::Value offset;
    IR::Value offset2;
    const auto pair = [&](const IR::U32& value) {
        return ir.CompositeConstruct(ir.BitFieldExtract(value, ir.Imm32(0), ir.Imm32(6), true),
                                     ir.BitFieldExtract(value, ir.Imm32(8), ir.Imm32(6), true));
    };
    const auto quad = [&](const IR::U32& value) {
        return ir.CompositeConstruct(ir.BitFieldExtract(value, ir.Imm32(0), ir.Imm32(6), true),
                                     ir.BitFieldExtract(value, ir.Imm32(8), ir.Imm32(6), true),
                                     ir.BitFieldExtract(value, ir.Imm32(16), ir.Imm32(6), true),
                                     ir.BitFieldExtract(value, ir.Imm32(24), ir.Imm32(6), true));
    };
    if (offset_mode == 1) {
        offset = pair(X(regs[next++]));
    } else if (offset_mode == 2) {
        const IR::U32 first{X(regs[next++])};
        const IR::U32 second{X(regs[next++])};
        offset = quad(first);
        offset2 = quad(second);
    }
    IR::F32 dref;
    if (depth_compare) {
        dref = F(regs[next++]);
    }
    IR::TextureInstInfo info{};
    info.type.Assign(dim.type);
    info.is_depth.Assign(depth_compare ? 1 : 0);
    info.gather_component.Assign(component);
    const IR::U32 handle = bindless ? UR(handle_register) : DescriptorHandle(cbuf_offset);
    const IR::Value sample = depth_compare ? ir.ImageGatherDref(handle, coords, offset, offset2, dref, info)
                                           : ir.ImageGather(handle, coords, offset, offset2, info);
    const u32 outputs = static_cast<u32>(std::popcount(mask));
    Require(outputs <= 2 ? dest_b == 255 : dest_b != 255, "native texture result layout");
    if (half) {
        Require(mask == 15 && dest < 255 && dest_b < 255, "native half-precision gather result");
        std::array<IR::F32, 4> values;
        for (u32 c = 0; c < 4; ++c) {
            values[c] = IR::F32{ir.CompositeExtract(sample, c)};
        }
        X(dest, ir.PackHalf2x16(ir.CompositeConstruct(values[0], values[1])));
        X(dest_b, ir.PackHalf2x16(ir.CompositeConstruct(values[2], values[3])));
        return;
    }
    u32 stored = 0;
    for (u32 c = 0; c < 4; ++c) {
        if (((mask >> c) & 1) == 0) {
            continue;
        }
        const u32 target = stored < 2 ? dest + stored : dest_b + (stored - 2);
        Require(target < 255, "native texture result register");
        F(target, IR::F32{ir.CompositeExtract(sample, c)});
        ++stored;
    }
}

// TXQ: texture dimension query through a bound descriptor; the selected channels of
// (width, height, depth, mip count) go to consecutive registers.
void Visitor::TXQ() {
    // 0x370 is the register-handle form (TXQ.B): A holds the handle and A + 1 the mip level, as
    // Maxwell's bindless TXQ reads them (Fortune's Weave, TXQ.B RZ, R29, R28, TEX_HEADER_DIMENSION).
    const bool register_handle = op == 0x370;
    const u32 dest = Take(16, 8);
    const u32 lod_reg = Take(24, 8);
    u32 cbuf_offset = 0;
    if (register_handle) {
        Require(Take(32, 27) == 0 && Take(59, 3) == 1, "native bindless texture query mode");
        Require(lod_reg + 1 < 255, "native bindless texture query registers");
    } else {
        cbuf_offset = Take(40, 14) * 4;
        UseDescriptorBuffer(Take(54, 5));
        Require(Take(59, 3) == 0, "native texture query handle mode");
    }
    Require(Take(62, 2) == 0, "native texture query kind");
    Require(Take(64, 8) == 255, "native texture query extra register");
    const u32 mask = Take(72, 4);
    Ignore(90, 1);
    Require(mask != 0, "native empty texture query mask");
    const IR::U32 handle = register_handle ? X(lod_reg) : DescriptorHandle(cbuf_offset);
    const IR::U32 lod = register_handle ? X(lod_reg + 1) : X(lod_reg);
    const IR::Value query{ir.ImageQueryDimension(handle, lod, ir.Imm1(((mask >> 3) & 1) == 0))};
    u32 stored = 0;
    for (u32 element = 0; element < 4; ++element) {
        if (((mask >> element) & 1) == 0) {
            continue;
        }
        Require(dest + stored < 255, "native texture query result register");
        X(dest + stored, IR::U32{ir.CompositeExtract(query, element)});
        ++stored;
    }
}

// TXD: texture sample with explicit gradients through a bound descriptor. The coordinates and the
// four derivatives are consecutive registers (an array layer follows the coordinates); the selected
// channels go to the first destination (two channels) and the second one (the rest).
void Visitor::TXD() {
    const u32 dest = Take(16, 8);
    const u32 coord_reg = Take(24, 8);
    const u32 derivative_reg = Take(32, 8);
    const u32 cbuf_offset = Take(40, 14) * 4;
    UseDescriptorBuffer(Take(54, 5));
    Require(Take(59, 2) == 0, "native gradient texture handle mode");
    const Dimension dim = DimensionOf(Take(61, 3));
    const u32 dest_b = Take(64, 8);
    const u32 mask = Take(72, 4);
    Require(Take(76, 1) == 0, "native gradient texture offset");
    Require(Take(81, 3) == 7, "native sparse texture residency");
    Ignore(84, 3);
    Ignore(90, 1);
    Require(mask != 0, "native empty texture channel mask");
    Require(dim.type == Shader::TextureType::Color2D || dim.type == Shader::TextureType::ColorArray2D,
            "native gradient texture dimension");
    const bool array = dim.type == Shader::TextureType::ColorArray2D;
    Require(coord_reg + (array ? 2 : 1) < 255 && derivative_reg + 3 < 255, "native gradient operand registers");
    IR::Value coords;
    if (array) {
        const IR::F32 layer{
            ir.ConvertUToF(32, 16, ir.BitFieldExtract(X(coord_reg + 2), ir.Imm32(0), ir.Imm32(16)))};
        coords = ir.CompositeConstruct(F(coord_reg), F(coord_reg + 1), layer);
    } else {
        coords = ir.CompositeConstruct(F(coord_reg), F(coord_reg + 1));
    }
    const IR::Value derivatives{ir.CompositeConstruct(F(derivative_reg), F(derivative_reg + 1),
                                                      F(derivative_reg + 2), F(derivative_reg + 3))};
    IR::TextureInstInfo info{};
    info.type.Assign(dim.type);
    info.num_derivatives.Assign(2);
    info.has_lod_clamp.Assign(0);
    const IR::Value sample{ir.ImageGradient(DescriptorHandle(cbuf_offset), coords, derivatives, {}, {}, info)};
    const u32 outputs = static_cast<u32>(std::popcount(mask));
    Require(outputs <= 2 ? dest_b == 255 : dest_b != 255, "native texture result layout");
    u32 stored = 0;
    for (u32 c = 0; c < 4; ++c) {
        if (((mask >> c) & 1) == 0) {
            continue;
        }
        const u32 target = stored < 2 ? dest + stored : dest_b + (stored - 2);
        Require(target < 255, "native texture result register");
        F(target, IR::F32{ir.CompositeExtract(sample, c)});
        ++stored;
    }
}

// TMML: mipmap level query (clamped and unclamped LOD, as 8.8 fixed point) through a bound descriptor. The coordinates
// are consecutive registers starting at source A; an array index register is skipped, as the
// Maxwell translation does.
void Visitor::TMML() {
    const u32 dest = Take(16, 8);
    const u32 coord_reg = Take(24, 8);
    Require(Take(32, 8) == 255, "native mipmap query extra register");
    const u32 cbuf_offset = Take(40, 14) * 4;
    UseDescriptorBuffer(Take(54, 5));
    Require(Take(59, 2) == 0, "native mipmap query handle mode");
    const Dimension dim = DimensionOf(Take(61, 3));
    Require(dim.coordinates != 0, "native texture dimension");
    Require(Take(64, 8) == 255, "native mipmap query extra register");
    const u32 mask = Take(72, 4);
    Require((mask & 0xc) == 0 && mask != 0, "native mipmap query channels");
    Ignore(77, 1);
    Ignore(90, 1);
    Require(coord_reg + dim.coordinates < 255, "native mipmap query registers");
    IR::Value coords;
    switch (dim.type) {
    case Shader::TextureType::Color1D:
        coords = F(coord_reg);
        break;
    case Shader::TextureType::ColorArray1D:
        coords = F(coord_reg + 1);
        break;
    case Shader::TextureType::Color2D:
        coords = ir.CompositeConstruct(F(coord_reg), F(coord_reg + 1));
        break;
    case Shader::TextureType::ColorArray2D:
        coords = ir.CompositeConstruct(F(coord_reg + 1), F(coord_reg + 2));
        break;
    case Shader::TextureType::Color3D:
    case Shader::TextureType::ColorCube:
        coords = ir.CompositeConstruct(F(coord_reg), F(coord_reg + 1), F(coord_reg + 2));
        break;
    default:
        coords = ir.CompositeConstruct(F(coord_reg + 1), F(coord_reg + 2), F(coord_reg + 3));
        break;
    }
    IR::TextureInstInfo info{};
    info.type.Assign(dim.type);
    const IR::Value lod{ir.ImageQueryLod(DescriptorHandle(cbuf_offset), coords, info)};
    u32 stored = 0;
    for (u32 element = 0; element < 2; ++element) {
        if (((mask >> element) & 1) == 0) {
            continue;
        }
        Require(dest + stored < 255, "native mipmap query result register");
        // The hardware returns the LODs as 8.8 fixed-point 16-bit integers (shaders convert them back
        // with I2F.S16 and a scale). The host query returns absolute then relative LOD, and TMML the
        // reverse; the first result is signed and the second unsigned.
        const IR::F32 level{ir.CompositeExtract(lod, element ^ 1)};
        const IR::F32 scaled{ir.FPMul(level, ir.Imm32(256.0f))};
        const IR::U32 fixed{element == 0 ? IR::U32{ir.ConvertFToS(32, scaled)}
                                         : IR::U32{ir.ConvertFToU(32, scaled)}};
        X(dest + stored, ir.BitwiseAnd(fixed, ir.Imm32(0xffffU)));
        ++stored;
    }
}

// SURED.D.<dim>.<op>.STRONG.GPU.IGN: atomically combine a U32 into a surface texel, discarding the
// old value. Coordinates start at Ra and are laid out as SUATOM's; the descriptor offset is in
// words. Every dimension, as Eglantine's lowering (59c4bde459) takes 1D buffers as well as 2D.
void Visitor::SURED() {
    Require(Take(16, 8) == 0, "native surface reduction reserved destination");
    const u32 coord_reg = Take(24, 8);
    const u32 data_reg = Take(32, 8);
    const u32 cbuf_offset = Take(40, 14) * 4;
    const u32 cbuf_index = Take(54, 5);
    Require(Take(59, 2) == 0, "native surface reduction clamp mode"); // .IGN
    const Dimension dim = ImageDimensionOf(Take(61, 3));
    Require(dim.coordinates != 0, "native surface reduction dimension");
    Require(Take(64, 8) == 0, "native surface reduction reserved source");
    Require(Take(72, 1) == 0, "native surface reduction byte addressing");
    const u32 type = Take(73, 3);
    Require(type <= 1, "native surface reduction type"); // U32, S32
    Require(Take(76, 1) == 0, "native surface reduction modifier");
    Require(Take(77, 4) == 7, "native surface reduction memory ordering"); // STRONG.GPU
    Require(Take(81, 3) == 0, "native surface reduction predicate result");
    Require(Take(84, 3) == 1, "native surface reduction eviction priority");
    // ADD, MIN, MAX, INC, DEC, AND, OR, XOR, numbered as SUATOM's (Mario Kart World adds).
    const u32 operation = Take(87, 3);
    Require(Take(90, 15) == 0, "native surface reduction modifiers");
    Require(coord_reg + dim.coordinates < 255, "native surface reduction coordinates");
    UseDescriptorBuffer(cbuf_index);
    const IR::Value coords{SurfaceCoordinates(ir, dim, coord_reg, [this](u32 r) { return X(r); })};
    IR::TextureInstInfo info{};
    info.type.Assign(dim.type);
    info.image_format.Assign(ImageFormat::Typeless);
    const IR::Value handle{DescriptorHandle(cbuf_offset)};
    const IR::Value value{X(data_reg)};
    const bool is_signed = type == 1;
    switch (operation) {
    case 0:
        (void)ir.ImageAtomicIAdd(handle, coords, value, info);
        break;
    case 1:
        (void)ir.ImageAtomicIMin(handle, coords, value, is_signed, info);
        break;
    case 2:
        (void)ir.ImageAtomicIMax(handle, coords, value, is_signed, info);
        break;
    case 3:
        (void)ir.ImageAtomicInc(handle, coords, value, info);
        break;
    case 4:
        (void)ir.ImageAtomicDec(handle, coords, value, info);
        break;
    case 5:
        (void)ir.ImageAtomicAnd(handle, coords, value, info);
        break;
    case 6:
        (void)ir.ImageAtomicOr(handle, coords, value, info);
        break;
    default:
        (void)ir.ImageAtomicXor(handle, coords, value, info);
        break;
    }
}

// SUATOM (bound descriptor): atomic operation on one texel of a surface, returning the old value.
// Laid out like SUST/SULD: destination 16, coordinates 24, data 32, descriptor 40..58, dimension
// 61..63, type 73..75, fault predicate 81..83 and the operation at 87..90 (NAK's SuAtom fields).
void Visitor::SUATOM() {
    const u32 dest = Take(16, 8);
    const u32 coord_reg = Take(24, 8);
    const u32 data_reg = Take(32, 8);
    const u32 cbuf_offset = Take(40, 14) * 4;
    UseDescriptorBuffer(Take(54, 5));
    Require(Take(59, 2) == 0, "native surface handle mode");
    const Dimension dim = ImageDimensionOf(Take(61, 3));
    Require(dim.coordinates != 0, "native surface dimension");
    Require(Take(72, 1) == 0, "native surface atomic mode");
    const u32 type = Take(73, 3);
    Ignore(77, 4); // memory ordering
    Require(Take(81, 3) == 7, "native sparse surface residency");
    Ignore(84, 3);
    const u32 atom_op = Take(87, 4);
    Require(type <= 1 && atom_op <= 8, "native surface atomic operation");
    Require(coord_reg + dim.coordinates < 255, "native surface registers");
    const IR::Value coords{SurfaceCoordinates(ir, dim, coord_reg, [this](u32 r) { return X(r); })};
    IR::TextureInstInfo info{};
    info.type.Assign(dim.type);
    info.image_format.Assign(ImageFormat::Typeless);
    const IR::Value handle{DescriptorHandle(cbuf_offset)};
    const IR::Value value{X(data_reg)};
    const bool is_signed = type == 1;
    IR::Value old;
    switch (atom_op) {
    case 0:
        old = ir.ImageAtomicIAdd(handle, coords, value, info);
        break;
    case 1:
        old = ir.ImageAtomicIMin(handle, coords, value, is_signed, info);
        break;
    case 2:
        old = ir.ImageAtomicIMax(handle, coords, value, is_signed, info);
        break;
    case 3:
        old = ir.ImageAtomicInc(handle, coords, value, info);
        break;
    case 4:
        old = ir.ImageAtomicDec(handle, coords, value, info);
        break;
    case 5:
        old = ir.ImageAtomicAnd(handle, coords, value, info);
        break;
    case 6:
        old = ir.ImageAtomicOr(handle, coords, value, info);
        break;
    case 7:
        old = ir.ImageAtomicXor(handle, coords, value, info);
        break;
    default:
        old = ir.ImageAtomicExchange(handle, coords, value, info);
        break;
    }
    if (dest != 255) {
        X(dest, IR::U32{old});
    }
}

} // namespace Shader::Ampere
