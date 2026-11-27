// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "shader_recompiler/frontend/ir/modifiers.h"

namespace Shader::Backend::SPIRV {
namespace {
Id ExtractU16(EmitContext& ctx, Id value) {
    if (ctx.profile.support_int16) {
        return ctx.OpUConvert(ctx.U16, value);
    } else {
        return ctx.OpBitFieldUExtract(ctx.U32[1], value, ctx.u32_zero_value, ctx.Const(16u));
    }
}

Id ExtractS16(EmitContext& ctx, Id value) {
    if (ctx.profile.support_int16) {
        return ctx.OpSConvert(ctx.S16, value);
    } else {
        return ctx.OpBitFieldSExtract(ctx.U32[1], value, ctx.u32_zero_value, ctx.Const(16u));
    }
}

Id ExtractU8(EmitContext& ctx, Id value) {
    if (ctx.profile.support_int8) {
        return ctx.OpUConvert(ctx.U8, value);
    } else {
        return ctx.OpBitFieldUExtract(ctx.U32[1], value, ctx.u32_zero_value, ctx.Const(8u));
    }
}

Id ExtractS8(EmitContext& ctx, Id value) {
    if (ctx.profile.support_int8) {
        return ctx.OpSConvert(ctx.S8, value);
    } else {
        return ctx.OpBitFieldSExtract(ctx.U32[1], value, ctx.u32_zero_value, ctx.Const(8u));
    }
}
} // Anonymous namespace

Id EmitConvertS16F16(EmitContext& ctx, Id value) {
    if (ctx.profile.support_int16) {
        return ctx.OpSConvert(ctx.U32[1], ctx.OpConvertFToS(ctx.U16, value));
    } else {
        return ExtractS16(ctx, ctx.OpConvertFToS(ctx.U32[1], value));
    }
}

Id EmitConvertS16F32(EmitContext& ctx, Id value) {
    if (ctx.profile.support_int16) {
        return ctx.OpSConvert(ctx.U32[1], ctx.OpConvertFToS(ctx.U16, value));
    } else {
        return ExtractS16(ctx, ctx.OpConvertFToS(ctx.U32[1], value));
    }
}

Id EmitConvertS16F64(EmitContext& ctx, Id value) {
    if (ctx.profile.support_int16) {
        return ctx.OpSConvert(ctx.U32[1], ctx.OpConvertFToS(ctx.U16, value));
    } else {
        return ExtractS16(ctx, ctx.OpConvertFToS(ctx.U32[1], value));
    }
}

Id EmitConvertS32F16(EmitContext& ctx, Id value) {
    return ctx.OpConvertFToS(ctx.U32[1], value);
}

Id EmitConvertS32F32(EmitContext& ctx, Id value) {
    if (ctx.profile.has_broken_signed_operations) {
        return ctx.OpBitcast(ctx.U32[1], ctx.OpConvertFToS(ctx.S32[1], value));
    } else {
        return ctx.OpConvertFToS(ctx.U32[1], value);
    }
}

Id EmitConvertS32F64(EmitContext& ctx, Id value) {
    return ctx.OpConvertFToS(ctx.U32[1], value);
}

Id EmitConvertS64F16(EmitContext& ctx, Id value) {
    return ctx.OpConvertFToS(ctx.U64, value);
}

Id EmitConvertS64F32(EmitContext& ctx, Id value) {
    return ctx.OpConvertFToS(ctx.U64, value);
}

Id EmitConvertS64F64(EmitContext& ctx, Id value) {
    return ctx.OpConvertFToS(ctx.U64, value);
}

Id EmitConvertU16F16(EmitContext& ctx, Id value) {
    if (ctx.profile.support_int16) {
        return ctx.OpUConvert(ctx.U32[1], ctx.OpConvertFToU(ctx.U16, value));
    } else {
        return ExtractU16(ctx, ctx.OpConvertFToU(ctx.U32[1], value));
    }
}

Id EmitConvertU16F32(EmitContext& ctx, Id value) {
    if (ctx.profile.support_int16) {
        return ctx.OpUConvert(ctx.U32[1], ctx.OpConvertFToU(ctx.U16, value));
    } else {
        return ExtractU16(ctx, ctx.OpConvertFToU(ctx.U32[1], value));
    }
}

Id EmitConvertU16F64(EmitContext& ctx, Id value) {
    if (ctx.profile.support_int16) {
        return ctx.OpUConvert(ctx.U32[1], ctx.OpConvertFToU(ctx.U16, value));
    } else {
        return ExtractU16(ctx, ctx.OpConvertFToU(ctx.U32[1], value));
    }
}

Id EmitConvertU32F16(EmitContext& ctx, Id value) {
    return ctx.OpConvertFToU(ctx.U32[1], value);
}

Id EmitConvertU32F32(EmitContext& ctx, Id value) {
    return ctx.OpConvertFToU(ctx.U32[1], value);
}

Id EmitConvertU32F64(EmitContext& ctx, Id value) {
    return ctx.OpConvertFToU(ctx.U32[1], value);
}

Id EmitConvertU64F16(EmitContext& ctx, Id value) {
    return ctx.OpConvertFToU(ctx.U64, value);
}

Id EmitConvertU64F32(EmitContext& ctx, Id value) {
    return ctx.OpConvertFToU(ctx.U64, value);
}

Id EmitConvertU64F64(EmitContext& ctx, Id value) {
    return ctx.OpConvertFToU(ctx.U64, value);
}

Id EmitConvertU64U32(EmitContext& ctx, Id value) {
    return ctx.OpUConvert(ctx.U64, value);
}

Id EmitConvertU32U64(EmitContext& ctx, Id value) {
    return ctx.OpUConvert(ctx.U32[1], value);
}

Id EmitConvertF16F32(EmitContext& ctx, Id value) {
#ifdef __ANDROID__
    return ctx.OpFConvert(ctx.F16[1], value);
#else
    const auto result = ctx.OpFConvert(ctx.F16[1], value);
    const auto isOverflowing = ctx.OpIsNan(ctx.U1, result);
    return ctx.OpSelect(ctx.F16[1], isOverflowing, ctx.Constant(ctx.F16[1], 0), result);
#endif
}

Id EmitConvertF32F16(EmitContext& ctx, Id value) {
    return ctx.OpFConvert(ctx.F32[1], value);
}

Id EmitConvertF32F64(EmitContext& ctx, Id value) {
    return ctx.OpFConvert(ctx.F32[1], value);
}

Id EmitConvertF64F32(EmitContext& ctx, Id value) {
    return ctx.OpFConvert(ctx.F64[1], value);
}

Id EmitConvertF16S8(EmitContext& ctx, Id value) {
    return ctx.OpConvertSToF(ctx.F16[1], ExtractS8(ctx, value));
}

Id EmitConvertF16S16(EmitContext& ctx, Id value) {
    return ctx.OpConvertSToF(ctx.F16[1], ExtractS16(ctx, value));
}

Id EmitConvertF16S32(EmitContext& ctx, Id value) {
    return ctx.OpConvertSToF(ctx.F16[1], value);
}

Id EmitConvertF16S64(EmitContext& ctx, Id value) {
    return ctx.OpConvertSToF(ctx.F16[1], value);
}

Id EmitConvertF16U8(EmitContext& ctx, Id value) {
    return ctx.OpConvertUToF(ctx.F16[1], ExtractU8(ctx, value));
}

Id EmitConvertF16U16(EmitContext& ctx, Id value) {
    return ctx.OpConvertUToF(ctx.F16[1], ExtractU16(ctx, value));
}

Id EmitConvertF16U32(EmitContext& ctx, Id value) {
    return ctx.OpConvertUToF(ctx.F16[1], value);
}

Id EmitConvertF16U64(EmitContext& ctx, Id value) {
    return ctx.OpConvertUToF(ctx.F16[1], value);
}

Id EmitConvertF32S8(EmitContext& ctx, Id value) {
    return ctx.OpConvertSToF(ctx.F32[1], ExtractS8(ctx, value));
}

Id EmitConvertF32S16(EmitContext& ctx, Id value) {
    return ctx.OpConvertSToF(ctx.F32[1], ExtractS16(ctx, value));
}

// I2F.RZ: the conversion rounds to nearest; above 2^24 the nearest float may be larger in
// magnitude than the integer, and toward zero takes it one step back. The float is integral, so
// converting it back compares exactly (2^31 and 2^32 themselves are out of the integer range).
Id EmitConvertF32S32(EmitContext& ctx, IR::Inst* inst, Id value) {
    if (ctx.profile.has_broken_signed_operations) {
        value = ctx.OpBitcast(ctx.S32[1], value);
    }
    const Id r{ctx.OpConvertSToF(ctx.F32[1], value)};
    if (inst->Flags<IR::FpControl>().rounding != IR::FpRounding::RZ ||
        std::getenv("EDEN_NX2_NO_RZ_EMULATION") != nullptr) {
        return r;
    }
    const Id s{ctx.OpBitcast(ctx.U32[1], value)};
    const Id two31{ctx.Const(2147483648.0f)};
    const Id at_limit{ctx.OpFOrdGreaterThanEqual(ctx.U1, ctx.OpFAbs(ctx.F32[1], r), two31)};
    const Id back{ctx.OpConvertFToS(ctx.U32[1], ctx.OpFMin(ctx.F32[1], ctx.OpFMax(ctx.F32[1], r,
                      ctx.Const(-2147483520.0f)), ctx.Const(2147483520.0f)))};
    const Id negative{ctx.OpSLessThan(ctx.U1, s, ctx.u32_zero_value)};
    const Id beyond{ctx.OpSelect(ctx.U1, negative, ctx.OpSLessThan(ctx.U1, back, s),
                                 ctx.OpSGreaterThan(ctx.U1, back, s))};
    // -2^31 is exactly a float: only +2^31 (from 2^31 - 63 and up) needs the step at the limit.
    const Id step{ctx.OpLogicalOr(ctx.U1, ctx.OpLogicalAnd(ctx.U1, at_limit,
                                                           ctx.OpLogicalNot(ctx.U1, negative)),
                                  ctx.OpLogicalAnd(ctx.U1, ctx.OpLogicalNot(ctx.U1, at_limit), beyond))};
    const Id stepped{ctx.OpBitcast(ctx.F32[1],
                                   ctx.OpISub(ctx.U32[1], ctx.OpBitcast(ctx.U32[1], r), ctx.Const(1u)))};
    return ctx.OpSelect(ctx.F32[1], step, stepped, r);
}

Id EmitConvertF32S64(EmitContext& ctx, Id value) {
    return ctx.OpConvertSToF(ctx.F32[1], value);
}

Id EmitConvertF32U8(EmitContext& ctx, Id value) {
    return ctx.OpConvertUToF(ctx.F32[1], ExtractU8(ctx, value));
}

Id EmitConvertF32U16(EmitContext& ctx, Id value) {
    return ctx.OpConvertUToF(ctx.F32[1], ExtractU16(ctx, value));
}

Id EmitConvertF32U32(EmitContext& ctx, IR::Inst* inst, Id value) {
    const Id r{ctx.OpConvertUToF(ctx.F32[1], value)};
    if (inst->Flags<IR::FpControl>().rounding != IR::FpRounding::RZ ||
        std::getenv("EDEN_NX2_NO_RZ_EMULATION") != nullptr) {
        return r;
    }
    const Id at_limit{ctx.OpFOrdGreaterThanEqual(ctx.U1, r, ctx.Const(4294967296.0f))};
    const Id back{ctx.OpConvertFToU(ctx.U32[1], ctx.OpFMin(ctx.F32[1], r, ctx.Const(4294967040.0f)))};
    const Id beyond{ctx.OpUGreaterThan(ctx.U1, back, value)};
    const Id step{ctx.OpLogicalOr(ctx.U1, at_limit, beyond)};
    const Id stepped{ctx.OpBitcast(ctx.F32[1],
                                   ctx.OpISub(ctx.U32[1], ctx.OpBitcast(ctx.U32[1], r), ctx.Const(1u)))};
    return ctx.OpSelect(ctx.F32[1], step, stepped, r);
}

Id EmitConvertF32U64(EmitContext& ctx, Id value) {
    return ctx.OpConvertUToF(ctx.F32[1], value);
}

Id EmitConvertF64S8(EmitContext& ctx, Id value) {
    return ctx.OpConvertSToF(ctx.F64[1], ExtractS8(ctx, value));
}

Id EmitConvertF64S16(EmitContext& ctx, Id value) {
    return ctx.OpConvertSToF(ctx.F64[1], ExtractS16(ctx, value));
}

Id EmitConvertF64S32(EmitContext& ctx, Id value) {
    if (ctx.profile.has_broken_signed_operations) {
        value = ctx.OpBitcast(ctx.S32[1], value);
    }
    return ctx.OpConvertSToF(ctx.F64[1], value);
}

Id EmitConvertF64S64(EmitContext& ctx, Id value) {
    return ctx.OpConvertSToF(ctx.F64[1], value);
}

Id EmitConvertF64U8(EmitContext& ctx, Id value) {
    return ctx.OpConvertUToF(ctx.F64[1], ExtractU8(ctx, value));
}

Id EmitConvertF64U16(EmitContext& ctx, Id value) {
    return ctx.OpConvertUToF(ctx.F64[1], ExtractU16(ctx, value));
}

Id EmitConvertF64U32(EmitContext& ctx, Id value) {
    return ctx.OpConvertUToF(ctx.F64[1], value);
}

Id EmitConvertF64U64(EmitContext& ctx, Id value) {
    return ctx.OpConvertUToF(ctx.F64[1], value);
}

Id EmitConvertU16U32(EmitContext& ctx, Id value) {
    if (ctx.profile.support_int16) {
        return ctx.OpUConvert(ctx.U16, value);
    } else {
        return ctx.OpBitFieldUExtract(ctx.U32[1], value, ctx.u32_zero_value, ctx.Const(16u));
    }
}

Id EmitConvertU32U16(EmitContext& ctx, Id value) {
    if (ctx.profile.support_int16) {
        return ctx.OpUConvert(ctx.U32[1], value);
    } else {
        return ExtractU16(ctx, value);
    }
}

Id EmitConvertU8U32(EmitContext& ctx, Id value) {
    if (ctx.profile.support_int8) {
        return ctx.OpUConvert(ctx.U8, value);
    } else {
        return ExtractU8(ctx, value);
    }
}

Id EmitConvertU32U8(EmitContext& ctx, Id value) {
    if (ctx.profile.support_int8) {
        return ctx.OpUConvert(ctx.U32[1], value);
    } else {
        return ctx.OpBitFieldUExtract(ctx.U32[1], value, ctx.u32_zero_value, ctx.Const(8u));
    }
}

// in signed
Id EmitConvertS32S8(EmitContext& ctx, Id value) {
    if (ctx.profile.support_int8) {
        return ctx.OpSConvert(ctx.U32[1], value);
    } else {
        return ctx.OpBitFieldSExtract(ctx.U32[1], value, ctx.u32_zero_value, ctx.Const(8u));
    }
}

Id EmitConvertS32S16(EmitContext& ctx, Id value) {
    if (ctx.profile.support_int16) {
        return ctx.OpSConvert(ctx.U32[1], value);
    } else {
        return ctx.OpBitFieldSExtract(ctx.U32[1], value, ctx.u32_zero_value, ctx.Const(16u));
    }
}

} // namespace Shader::Backend::SPIRV
