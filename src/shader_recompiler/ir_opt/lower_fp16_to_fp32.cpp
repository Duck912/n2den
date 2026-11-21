// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace Shader::Optimization {
namespace {
// The exact rounding of `value` to half precision for RZ, RM and RP (see RoundToHalf).
IR::F32 RoundToHalfDirected(IR::IREmitter& ir, const IR::F32& value, IR::FpRounding rounding) {
    const IR::FpControl exact{.no_contraction = true};
    const IR::U32 bits{ir.BitCast<IR::U32>(value)};
    const IR::U32 biased{ir.BitFieldExtract(bits, ir.Imm32(23), ir.Imm32(8))};
    const IR::U32 ulp_log{
        ir.SMax(IR::U32{ir.ISub(biased, ir.Imm32(127 + 10))}, ir.Imm32(static_cast<u32>(-24)))};
    const auto power_of_two = [&](const IR::U32& biased_exponent) {
        return ir.BitCast<IR::F32>(IR::U32{ir.ShiftLeftLogical(biased_exponent, ir.Imm32(23))});
    };
    const IR::F32 scale_up{power_of_two(IR::U32{ir.ISub(ir.Imm32(127), ulp_log)})};
    const IR::F32 scale_down{power_of_two(IR::U32{ir.IAdd(ir.Imm32(127), ulp_log)})};
    const IR::F32 scaled{ir.FPMul(value, scale_up, exact)};
    IR::F32 integral;
    switch (rounding) {
    case IR::FpRounding::RZ:
        integral = IR::F32{ir.FPTrunc(scaled)};
        break;
    case IR::FpRounding::RM:
        integral = IR::F32{ir.FPFloor(scaled)};
        break;
    case IR::FpRounding::RP:
        integral = IR::F32{ir.FPCeil(scaled)};
        break;
    default:
        integral = IR::F32{ir.FPRoundEven(scaled)};
        break;
    }
    const IR::F32 rounded{ir.FPMul(integral, scale_down, exact)};
    const IR::U32 sign{ir.BitwiseAnd(bits, ir.Imm32(0x80000000U))};
    const IR::U32 infinity{ir.BitwiseOr(sign, ir.Imm32(0x7f800000U))};
    const IR::U32 largest{ir.BitwiseOr(sign, ir.Imm32(0x477fe000U))}; // 65504
    const IR::U1 positive{ir.IEqual(sign, ir.Imm32(0U))};
    IR::U32 limit;
    switch (rounding) {
    case IR::FpRounding::RZ:
        limit = largest;
        break;
    case IR::FpRounding::RM:
        limit = IR::U32{ir.Select(positive, largest, infinity)};
        break;
    case IR::FpRounding::RP:
        limit = IR::U32{ir.Select(positive, infinity, largest)};
        break;
    default:
        limit = infinity;
        break;
    }
    const IR::U1 overflow{ir.FPGreaterThan(ir.FPAbs(rounded), ir.Imm32(65504.0f), {}, true)};
    const IR::U32 result{ir.Select(overflow, limit, ir.BitCast<IR::U32>(rounded))};
    const IR::U1 special{ir.IEqual(biased, ir.Imm32(0xffU))};
    return ir.BitCast<IR::F32>(IR::U32{ir.Select(special, bits, result)});
}

// The f32 value of a conversion to half, kept in f32: `value` rounded to half precision in the
// conversion's rounding mode. Without it the conversion was an identity and the rounding steps of
// F2F, F2FP and the shallow HMMA accumulator were lost.
//
// Round to nearest (and no stated mode) goes through PackHalf2x16 and back: the rounding every
// half register write already gets on these hosts, in a few instructions. The directed modes,
// which the driver cannot be asked for, are exact: the half's ulp at the value's magnitude is
// scaled to 1 (2^(exponent - 10) for normal halves, 2^-24 below them), the value is rounded to an
// integer and scaled back (both scalings are exact powers of two). Results beyond the largest half
// become infinity or the largest half as the mode says; infinities and NaNs pass through.
IR::F32 RoundToHalf(IR::IREmitter& ir, const IR::F32& value, IR::FpRounding rounding) {
    if (rounding != IR::FpRounding::RZ && rounding != IR::FpRounding::RM &&
        rounding != IR::FpRounding::RP) {
        const IR::U32 packed{ir.PackHalf2x16(ir.CompositeConstruct(value, ir.Imm32(0.0f)))};
        return IR::F32{ir.CompositeExtract(ir.UnpackHalf2x16(packed), 0)};
    }
    return RoundToHalfDirected(ir, value, rounding);
}

IR::Opcode Replace(IR::Opcode op) {
    switch (op) {
    case IR::Opcode::FPAbs16:
        return IR::Opcode::FPAbs32;
    case IR::Opcode::FPAdd16:
        return IR::Opcode::FPAdd32;
    case IR::Opcode::FPCeil16:
        return IR::Opcode::FPCeil32;
    case IR::Opcode::FPFloor16:
        return IR::Opcode::FPFloor32;
    case IR::Opcode::FPFma16:
        return IR::Opcode::FPFma32;
    case IR::Opcode::FPMul16:
        return IR::Opcode::FPMul32;
    case IR::Opcode::FPNeg16:
        return IR::Opcode::FPNeg32;
    case IR::Opcode::FPRoundEven16:
        return IR::Opcode::FPRoundEven32;
    case IR::Opcode::FPSaturate16:
        return IR::Opcode::FPSaturate32;
    case IR::Opcode::FPClamp16:
        return IR::Opcode::FPClamp32;
    case IR::Opcode::FPTrunc16:
        return IR::Opcode::FPTrunc32;
    case IR::Opcode::CompositeConstructF16x2:
        return IR::Opcode::CompositeConstructF32x2;
    case IR::Opcode::CompositeConstructF16x3:
        return IR::Opcode::CompositeConstructF32x3;
    case IR::Opcode::CompositeConstructF16x4:
        return IR::Opcode::CompositeConstructF32x4;
    case IR::Opcode::CompositeExtractF16x2:
        return IR::Opcode::CompositeExtractF32x2;
    case IR::Opcode::CompositeExtractF16x3:
        return IR::Opcode::CompositeExtractF32x3;
    case IR::Opcode::CompositeExtractF16x4:
        return IR::Opcode::CompositeExtractF32x4;
    case IR::Opcode::CompositeInsertF16x2:
        return IR::Opcode::CompositeInsertF32x2;
    case IR::Opcode::CompositeInsertF16x3:
        return IR::Opcode::CompositeInsertF32x3;
    case IR::Opcode::CompositeInsertF16x4:
        return IR::Opcode::CompositeInsertF32x4;
    case IR::Opcode::FPOrdEqual16:
        return IR::Opcode::FPOrdEqual32;
    case IR::Opcode::FPUnordEqual16:
        return IR::Opcode::FPUnordEqual32;
    case IR::Opcode::FPOrdNotEqual16:
        return IR::Opcode::FPOrdNotEqual32;
    case IR::Opcode::FPUnordNotEqual16:
        return IR::Opcode::FPUnordNotEqual32;
    case IR::Opcode::FPOrdLessThan16:
        return IR::Opcode::FPOrdLessThan32;
    case IR::Opcode::FPUnordLessThan16:
        return IR::Opcode::FPUnordLessThan32;
    case IR::Opcode::FPOrdGreaterThan16:
        return IR::Opcode::FPOrdGreaterThan32;
    case IR::Opcode::FPUnordGreaterThan16:
        return IR::Opcode::FPUnordGreaterThan32;
    case IR::Opcode::FPOrdLessThanEqual16:
        return IR::Opcode::FPOrdLessThanEqual32;
    case IR::Opcode::FPUnordLessThanEqual16:
        return IR::Opcode::FPUnordLessThanEqual32;
    case IR::Opcode::FPOrdGreaterThanEqual16:
        return IR::Opcode::FPOrdGreaterThanEqual32;
    case IR::Opcode::FPUnordGreaterThanEqual16:
        return IR::Opcode::FPUnordGreaterThanEqual32;
    case IR::Opcode::FPIsNan16:
        return IR::Opcode::FPIsNan32;
    case IR::Opcode::ConvertS16F16:
        return IR::Opcode::ConvertS16F32;
    case IR::Opcode::ConvertS32F16:
        return IR::Opcode::ConvertS32F32;
    case IR::Opcode::ConvertS64F16:
        return IR::Opcode::ConvertS64F32;
    case IR::Opcode::ConvertU16F16:
        return IR::Opcode::ConvertU16F32;
    case IR::Opcode::ConvertU32F16:
        return IR::Opcode::ConvertU32F32;
    case IR::Opcode::ConvertU64F16:
        return IR::Opcode::ConvertU64F32;
    case IR::Opcode::PackFloat2x16:
        return IR::Opcode::PackHalf2x16;
    case IR::Opcode::UnpackFloat2x16:
        return IR::Opcode::UnpackHalf2x16;
    case IR::Opcode::ConvertF32F16:
        return IR::Opcode::Identity;
    case IR::Opcode::ConvertF16F32:
        return IR::Opcode::Identity;
    case IR::Opcode::ConvertF16S8:
        return IR::Opcode::ConvertF32S8;
    case IR::Opcode::ConvertF16S16:
        return IR::Opcode::ConvertF32S16;
    case IR::Opcode::ConvertF16S32:
        return IR::Opcode::ConvertF32S32;
    case IR::Opcode::ConvertF16S64:
        return IR::Opcode::ConvertF32S64;
    case IR::Opcode::ConvertF16U8:
        return IR::Opcode::ConvertF32U8;
    case IR::Opcode::ConvertF16U16:
        return IR::Opcode::ConvertF32U16;
    case IR::Opcode::ConvertF16U32:
        return IR::Opcode::ConvertF32U32;
    case IR::Opcode::ConvertF16U64:
        return IR::Opcode::ConvertF32U64;
    case IR::Opcode::GlobalAtomicAddF16x2:
        return IR::Opcode::GlobalAtomicAddF32x2;
    case IR::Opcode::StorageAtomicAddF16x2:
        return IR::Opcode::StorageAtomicAddF32x2;
    case IR::Opcode::GlobalAtomicMinF16x2:
        return IR::Opcode::GlobalAtomicMinF32x2;
    case IR::Opcode::StorageAtomicMinF16x2:
        return IR::Opcode::StorageAtomicMinF32x2;
    case IR::Opcode::GlobalAtomicMaxF16x2:
        return IR::Opcode::GlobalAtomicMaxF32x2;
    case IR::Opcode::StorageAtomicMaxF16x2:
        return IR::Opcode::StorageAtomicMaxF32x2;
    default:
        return op;
    }
}
} // Anonymous namespace

void LowerFp16ToFp32(IR::Program& program) {
    for (IR::Block* const block : program.blocks) {
        LowerFp16ToFp32(*block);
    }
}

void LowerFp16ToFp32(IR::Block& block) {
    for (IR::Inst& inst : block.Instructions()) {
        if (inst.GetOpcode() == IR::Opcode::ConvertF16F32) {
            IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
            const IR::F32 rounded{
                RoundToHalf(ir, IR::F32{inst.Arg(0)}, inst.Flags<IR::FpControl>().rounding)};
            inst.ReplaceUsesWith(rounded);
            continue;
        }
        inst.ReplaceOpcode(Replace(inst.GetOpcode()));
    }
}

} // namespace Shader::Optimization
