// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Encoding reference: Mesa NAK sm70_encode.rs (MIT), Copyright 2022 Collabora, Ltd.

#include <bit>
#include <limits>

#include "shader_recompiler/frontend/ampere/visitor.h"
#include "shader_recompiler/frontend/ir/modifiers.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/common_funcs.h"
#include "shader_recompiler/frontend/maxwell/translate/impl/half_floating_point_helper.h"

namespace Shader::Ampere {
namespace {
IR::FpRounding RoundingOf(u32 mode) {
    switch (mode) {
    case 0:
        return IR::FpRounding::RN;
    case 1:
        return IR::FpRounding::RM;
    case 2:
        return IR::FpRounding::RP;
    default:
        return IR::FpRounding::RZ;
    }
}
} // Anonymous namespace

// F2IP.U8.F32 Rd, Ra, Rb, Rc: two floats rounded and saturated to unsigned bytes, packed as PTX
// cvt.pack.sat.u8 packs: Rd = Rc[15:0] << 16 | byte(Ra) << 8 | byte(Rb); NaN is 0.
// Only the .U8.F32.NTZ form (bit 74) Ocarina of Time uses is accepted. Bits 78-79 are the rounding
// (nvdisasm: 0 nearest, 3 .TRUNC, 1 and 2 INVALID); the task/mesh programs use .TRUNC.
void Visitor::F2IP() {
    const u32 dest = Dst();
    const Operand a = Source(0, false);
    const Operand b = Source(1, false);
    const Operand c = Source(2, false);
    Require(Take(72, 2) == 0 && Take(74, 1) == 1 && Take(75, 1) == 0, "native float pack mode");
    const u32 rounding = Take(78, 2);
    Require(rounding == 0 || rounding == 3, "native float pack rounding");
    const IR::FpControl control{
        .no_contraction = true,
        .rounding = IR::FpRounding::DontCare,
        .fmz_mode = IR::FmzMode::FTZ,
    };
    const auto to_byte = [&](const Operand& operand) {
        const IR::F32 value{ir.BitCast<IR::F32>(Value(operand))};
        const IR::F32 rounded{rounding == 3 ? ir.FPTrunc(value, control) : ir.FPRoundEven(value, control)};
        // FPClamp's NaN result is not defined here: pick 0 for NaN explicitly.
        const IR::F32 clamped{ir.FPClamp(rounded, ir.Imm32(0.0f), ir.Imm32(255.0f))};
        const IR::U32 byte{ir.ConvertFToU(32, clamped)};
        return IR::U32{ir.Select(ir.FPIsNan(value), ir.Imm32(0U), byte)};
    };
    IR::U32 result{ir.ShiftLeftLogical(Value(c), ir.Imm32(16U))};
    result = ir.BitFieldInsert(result, to_byte(a), ir.Imm32(8U), ir.Imm32(8U));
    result = ir.BitFieldInsert(result, to_byte(b), ir.Imm32(0U), ir.Imm32(8U));
    WriteDst(dest, result);
}

// F2I: float to integer, rounding by the rounding field then clamping to the destination range.
void Visitor::F2I() {
    const u32 dest = Take(16, 8);
    const Operand source = Source(1, true);
    const bool dst_signed = Take(72, 1) != 0;
    const u32 dst_size = Take(75, 2);
    Ignore(77, 1); // NTZ
    const u32 rounding = Take(78, 2);
    const bool ftz = Take(80, 1) != 0;
    const u32 src_size = Take(84, 2);
    // The 64-bit form (base 0x111) writes a register pair: Fortune's Weave's F2I.FTZ.U64.TRUNC.
    const bool wide = (op & 0x1ff) == 0x111;
    Require(wide ? dst_size == 3 : dst_size <= 2, "native float-to-integer destination size");
    Require(src_size == 2 || src_size == 1, "native float-to-integer source size");
    Require(!wide || dest % 2 == 0 || dest == 255, "native unaligned 64-bit destination pair");
    const bool half_source = src_size == 1;
    const u32 half_select = half_source ? Take(60, 1) : 0;

    IR::F16F32F64 value;
    if (half_source) {
        Require(source.kind == Operand::Kind::Reg, "native half-precision float-to-integer source");
        auto [lhs, rhs] = Maxwell::Extract(ir, X(source.reg), Maxwell::Swizzle::H1_H0);
        value = half_select ? rhs : lhs;
    } else {
        value = ir.BitCast<IR::F32>(Value(source));
    }
    const bool denorm_cares = !half_source;
    const IR::FpControl control{
        .no_contraction = true,
        .rounding = IR::FpRounding::DontCare,
        .fmz_mode = denorm_cares ? (ftz ? IR::FmzMode::FTZ : IR::FmzMode::None) : IR::FmzMode::DontCare,
    };
    const IR::F16F32F64 op_a{ir.FPAbsNeg(value, source.abs, source.negate)};
    IR::F16F32F64 rounded;
    switch (rounding) {
    case 0:
        rounded = ir.FPRoundEven(op_a, control);
        break;
    case 1:
        rounded = ir.FPFloor(op_a, control);
        break;
    case 2:
        rounded = ir.FPCeil(op_a, control);
        break;
    default:
        rounded = ir.FPTrunc(op_a, control);
        break;
    }
    // Destination widths: 8, 16, 32 or 64 bits, saturating to the destination range. For 64 bits
    // the bounds are the floats closest to the limits from inside (2^64 and 2^63 are not in range).
    const u32 dst_bits = 8U << dst_size;
    f64 max_bound, min_bound;
    if (dst_bits == 64) {
        max_bound = dst_signed ? static_cast<f64>(std::bit_cast<f32>(0x5effffffU))
                               : static_cast<f64>(std::bit_cast<f32>(0x5f7fffffU));
        min_bound = dst_signed ? -9223372036854775808.0 : 0.0;
    } else if (dst_signed) {
        max_bound = static_cast<f64>((s64{1} << (dst_bits - 1)) - 1);
        min_bound = -static_cast<f64>(s64{1} << (dst_bits - 1));
    } else {
        max_bound = static_cast<f64>((s64{1} << dst_bits) - 1);
        min_bound = 0.0;
    }
    IR::F16F32F64 intermediate;
    if (half_source) {
        const IR::F16 max_val{ir.FPConvert(16, ir.Imm32(static_cast<f32>(max_bound)))};
        const IR::F16 min_val{ir.FPConvert(16, ir.Imm32(static_cast<f32>(min_bound)))};
        intermediate = ir.FPClamp(rounded, min_val, max_val);
    } else {
        const IR::F32 max_val{ir.Imm32(static_cast<f32>(max_bound))};
        const IR::F32 min_val{ir.Imm32(static_cast<f32>(min_bound))};
        intermediate = ir.FPClamp(rounded, min_val, max_val);
    }
    if (wide) {
        IR::U64 result{ir.ConvertFToI(64, dst_signed, intermediate)};
        if (dst_signed) {
            result = IR::U64{ir.Select(ir.FPIsNan(op_a), ir.Imm64(u64{0}), result)};
        }
        const IR::Value words{ir.UnpackUint2x32(result)};
        X(dest, IR::U32{ir.CompositeExtract(words, 0)});
        X(dest == 255 ? 255 : dest + 1, IR::U32{ir.CompositeExtract(words, 1)});
        return;
    }
    if (dst_bits == 32 && !half_source) {
        // 2^31 - 1 and 2^32 - 1 are not floats: as clamp bounds they round up to 2^31 and 2^32,
        // outside the integer range, and OpConvertFToS/FToU of an out-of-range value is undefined.
        // Driver compilers may turn that into anything (Fire Emblem vertex shaders lost the GPU
        // device: "Shader Instruction Error"). Clamp to the largest float below the limit and pick
        // the saturated value for inputs at or above it, as the hardware F2I does.
        const f32 limit{dst_signed ? 2147483648.0f : 4294967296.0f};
        const IR::F32 below_limit{ir.Imm32(std::bit_cast<f32>(dst_signed ? 0x4effffffU : 0x4f7fffffU))};
        // NaN is replaced before the clamp (a clamped NaN is undefined too).
        const IR::F32 not_nan{ir.Select(ir.FPIsNan(op_a), ir.Imm32(0.0f), IR::F32{rounded})};
        const IR::F32 safe{ir.FPClamp(not_nan, ir.Imm32(static_cast<f32>(min_bound)),
                                      below_limit)};
        IR::U32 result{ir.ConvertFToI(32, dst_signed, safe)};
        const IR::U1 at_limit{ir.FPGreaterThanEqual(IR::F32{rounded}, ir.Imm32(limit))};
        result = IR::U32{ir.Select(at_limit, ir.Imm32(dst_signed ? 0x7fffffffU : 0xffffffffU),
                                   result)};
        if (dst_signed) {
            result = IR::U32{ir.Select(ir.FPIsNan(op_a), ir.Imm32(0U), result)};
        }
        X(dest, result);
        return;
    }
    IR::U32 result{ir.ConvertFToI(32, dst_signed, intermediate)};
    if (dst_signed) {
        result = IR::U32{ir.Select(ir.FPIsNan(op_a), ir.Imm32(0U), result)};
    }
    X(dest, result);
}

// I2FP / I2F: 32-bit (or small) integer to F32, or to F16 (destination size 1, I2F.F16: the half
// in the low 16 bits of the destination, the high half zero, as F2F.F16 writes it). The F16 form is
// in the output pass of the newer upscaling library (Fast Fusion's hiluma_engine_output_*), whose
// consumers read only .H0_H0.
void Visitor::I2F() {
    const u32 dest = Take(16, 8);
    const Operand source = Source(1, false);
    const bool is_signed = Take(74, 1) != 0;
    const u32 dst_size = Take(75, 2);
    const u32 rounding = Take(78, 2);
    const u32 src_size = Take(84, 2);
    // The 64-bit form (base 0x112) reads a register pair: Fortune's Weave's I2F.U64 R2, R2.
    const bool wide = (op & 0x1ff) == 0x112;
    const bool half_dest = dst_size == 1;
    Require(dst_size == 2 || (half_dest && !wide && (op & 0x1ff) == 0x106),
            "native integer-to-float destination size");
    Require(wide ? src_size == 3 : src_size <= 2, "native integer-to-float source size");
    // Bits 60..61 pick which byte or half of the register is converted (zero for a full word). In
    // the immediate form (0x906) they are bits 28..29 of the 32-bit immediate instead.
    const u32 select = source.kind == Operand::Kind::Imm ? 0 : Take(60, 2);
    Require(src_size < 2 || select == 0, "native integer-to-float source select");
    Require(src_size >= 2 || select < (32U >> (3 + src_size)), "native integer-to-float source select");
    if (wide) {
        Require(source.kind == Operand::Kind::Reg && !source.abs && !source.negate &&
                    (source.reg == 255 || (source.reg % 2 == 0 && source.reg + 1 < 255)),
                "native 64-bit integer-to-float source");
        const IR::U64 pair{ir.PackUint2x32(ir.CompositeConstruct(X(source.reg),
                                                                  X(source.reg == 255 ? 255 : source.reg + 1)))};
        const IR::FpControl wide_control{
            .no_contraction = false,
            .rounding = RoundingOf(rounding),
            .fmz_mode = IR::FmzMode::DontCare,
        };
        F(dest, IR::F32{ir.ConvertIToF(32, 64, is_signed, pair, wide_control)});
        return;
    }
    IR::U32 value{Value(source)};
    if (src_size < 2) {
        const u32 width = 8U << src_size;
        value = ir.BitFieldExtract(value, ir.Imm32(select * width), ir.Imm32(width), is_signed);
    }
    const IR::FpControl control{
        .no_contraction = false,
        .rounding = RoundingOf(rounding),
        .fmz_mode = IR::FmzMode::DontCare,
    };
    if (half_dest) {
        const IR::F16F32F64 half{ir.ConvertIToF(16, 32, is_signed, value, control)};
        const IR::F16 zero{ir.FPConvert(16, ir.Imm32(0.0f))};
        X(dest, ir.PackFloat2x16(ir.CompositeConstruct(half, zero)));
        return;
    }
    F(dest, IR::F32{ir.ConvertIToF(32, 32, is_signed, value, control)});
}

// FRND: round a float to an integral float. F32 to F32, or F16 to F16 (sizes 1, FRND.F16: bit 60
// selects the source's high half, .H1; the result is the low half of the destination with the high
// half zero, as F2F.F16 writes it). Fast Fusion's upscaling output pass uses FRND.F16(.TRUNC).
void Visitor::FRND() {
    const u32 dest = Take(16, 8);
    const Operand source = Source(1, true);
    const u32 dst_size = Take(75, 2);
    const u32 rounding = Take(78, 2);
    const bool ftz = Take(80, 1) != 0;
    const u32 src_size = Take(84, 2);
    const bool half = dst_size == 1;
    Require(dst_size == 2 || half, "native float round destination size");
    Require(src_size == dst_size, "native float round source size");
    const auto round = [&](const IR::F16F32F64& input, const IR::FpControl& control) {
        switch (rounding) {
        case 0:
            return ir.FPRoundEven(input, control);
        case 1:
            return ir.FPFloor(input, control);
        case 2:
            return ir.FPCeil(input, control);
        default:
            return ir.FPTrunc(input, control);
        }
    };
    if (half) {
        Require(source.kind == Operand::Kind::Reg, "native half-precision float round source");
        const bool high = Take(60, 1) != 0;
        auto [lhs, rhs] = Maxwell::Extract(ir, X(source.reg), Maxwell::Swizzle::H1_H0);
        const IR::F16F32F64 input{ir.FPAbsNeg(high ? rhs : lhs, source.abs, source.negate)};
        const IR::FpControl control{
            .no_contraction = false,
            .rounding = IR::FpRounding::DontCare,
            .fmz_mode = IR::FmzMode::DontCare, // half precision keeps denormals
        };
        const IR::F16 zero{ir.FPConvert(16, ir.Imm32(0.0f))};
        X(dest, ir.PackFloat2x16(ir.CompositeConstruct(round(input, control), zero)));
        return;
    }
    const IR::FpControl control{
        .no_contraction = false,
        .rounding = IR::FpRounding::DontCare,
        .fmz_mode = ftz ? IR::FmzMode::FTZ : IR::FmzMode::None,
    };
    F(dest, IR::F32{round(FloatValue(source), control)});
}

// F2F: F32 to F16 conversion into the low half of the destination.
void Visitor::F2F() {
    const u32 dest = Take(16, 8);
    const Operand source = Source(1, true);
    Require(Take(75, 2) == 1, "native float conversion destination size");
    const u32 rounding = Take(78, 2);
    const bool ftz = Take(80, 1) != 0;
    Require(Take(84, 2) == 2, "native float conversion source size");
    const IR::FpControl control{
        .no_contraction = false,
        .rounding = RoundingOf(rounding),
        .fmz_mode = ftz ? IR::FmzMode::FTZ : IR::FmzMode::None,
    };
    const IR::F32 input{FloatValue(source)};
    const IR::F16F32F64 converted{ir.FPConvert(16, input, control)};
    const IR::F16 zero{ir.FPConvert(16, ir.Imm32(0.0f))};
    X(dest, ir.PackFloat2x16(ir.CompositeConstruct(converted, zero)));
}

// F2FP.PACK_AB: two F32 sources converted to F16 and packed (first source in the high half).
// MERGE_C keeps the low half of the third source raw in the high half instead.
void Visitor::F2FP() {
    const u32 dest = Take(16, 8);
    const Operand a = Source(0, false);
    const Operand b = Source(1, false);
    Ignore(64, 8);
    Require(Take(72, 1) == 0, "native float packing selector");
    const bool merge = Take(78, 1) != 0;
    Require(Take(79, 2) == 0, "native float packing rounding");
    const IR::FpControl control{
        .no_contraction = false,
        .rounding = IR::FpRounding::RN,
        .fmz_mode = IR::FmzMode::None,
    };
    const IR::F16 zero{ir.FPConvert(16, ir.Imm32(0.0f))};
    const auto to_half = [&](const IR::F32& value) {
        const IR::F16F32F64 converted{ir.FPConvert(16, value, control)};
        return IR::U32{ir.PackFloat2x16(ir.CompositeConstruct(converted, zero))};
    };
    IR::U32 high;
    if (merge) {
        const Operand c = Source(2, false);
        high = Value(c);
    } else {
        high = to_half(ir.BitCast<IR::F32>(Value(a)));
    }
    const IR::U32 low{to_half(ir.BitCast<IR::F32>(Value(b)))};
    X(dest, ir.IAdd(ir.ShiftLeftLogical(high, ir.Imm32(16U)), low));
}

} // namespace Shader::Ampere
