/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/frontend/A64/translate/impl/impl.h"

namespace Dynarmic::A64 {

bool TranslatorVisitor::SCVTF_float_fix(bool sf, Imm<2> type, Imm<6> scale, Reg Rn, Vec Vd) {
    const size_t intsize = sf ? 64 : 32;
    const auto fltsize = FPGetDataSize(type);
    if (!fltsize) {
        return UnallocatedEncoding();
    }
    if (!sf && !scale.Bit<5>()) {
        return UnallocatedEncoding();
    }
    const u8 fracbits = 64 - scale.ZeroExtend<u8>();
    if (*fltsize == 16) {
        const IR::U64 value = sf ? IR::U64{X(64, Rn)} : (true ? ir.SignExtendWordToLong(X(32, Rn)) : ir.ZeroExtendWordToLong(X(32, Rn)));
        V_scalar(16, Vd, ir.FPFixedToHalf(value, true, fracbits, ir.current_location->FPCR().RMode()));
        return true;
    }
    const FP::RoundingMode rounding_mode = ir.current_location->FPCR().RMode();

    const IR::U32U64 intval = X(intsize, Rn);
    const IR::U32U64 fltval = [&]() -> IR::U32U64 {
        switch (*fltsize) {
        case 32:
            return ir.FPSignedFixedToSingle(intval, fracbits, rounding_mode);
        case 64:
            return ir.FPSignedFixedToDouble(intval, fracbits, rounding_mode);
        }
        UNREACHABLE();
    }();

    V_scalar(*fltsize, Vd, fltval);
    return true;
}

bool TranslatorVisitor::UCVTF_float_fix(bool sf, Imm<2> type, Imm<6> scale, Reg Rn, Vec Vd) {
    const size_t intsize = sf ? 64 : 32;
    const auto fltsize = FPGetDataSize(type);
    if (!fltsize) {
        return UnallocatedEncoding();
    }
    if (!sf && !scale.Bit<5>()) {
        return UnallocatedEncoding();
    }
    const u8 fracbits = 64 - scale.ZeroExtend<u8>();
    if (*fltsize == 16) {
        const IR::U64 value = sf ? IR::U64{X(64, Rn)} : (false ? ir.SignExtendWordToLong(X(32, Rn)) : ir.ZeroExtendWordToLong(X(32, Rn)));
        V_scalar(16, Vd, ir.FPFixedToHalf(value, false, fracbits, ir.current_location->FPCR().RMode()));
        return true;
    }
    const FP::RoundingMode rounding_mode = ir.current_location->FPCR().RMode();

    const IR::U32U64 intval = X(intsize, Rn);
    const IR::U32U64 fltval = [&]() -> IR::U32U64 {
        switch (*fltsize) {
        case 32:
            return ir.FPUnsignedFixedToSingle(intval, fracbits, rounding_mode);
        case 64:
            return ir.FPUnsignedFixedToDouble(intval, fracbits, rounding_mode);
        }
        UNREACHABLE();
    }();

    V_scalar(*fltsize, Vd, fltval);
    return true;
}

bool TranslatorVisitor::FCVTZS_float_fix(bool sf, Imm<2> type, Imm<6> scale, Vec Vn, Reg Rd) {
    const size_t intsize = sf ? 64 : 32;
    const auto fltsize = FPGetDataSize(type);
    if (!fltsize) {
        return UnallocatedEncoding();
    }
    if (!sf && !scale.Bit<5>()) {
        return UnallocatedEncoding();
    }
    const u8 fracbits = 64 - scale.ZeroExtend<u8>();

    const IR::U16U32U64 fltval = V_scalar(*fltsize, Vn);
    IR::U32U64 intval;
    if (intsize == 32) {
        intval = ir.FPToFixedS32(fltval, fracbits, FP::RoundingMode::TowardsZero);
    } else if (intsize == 64) {
        intval = ir.FPToFixedS64(fltval, fracbits, FP::RoundingMode::TowardsZero);
    } else {
        UNREACHABLE();
    }

    X(intsize, Rd, intval);
    return true;
}

bool TranslatorVisitor::FCVTZU_float_fix(bool sf, Imm<2> type, Imm<6> scale, Vec Vn, Reg Rd) {
    const size_t intsize = sf ? 64 : 32;
    const auto fltsize = FPGetDataSize(type);
    if (!fltsize) {
        return UnallocatedEncoding();
    }
    if (!sf && !scale.Bit<5>()) {
        return UnallocatedEncoding();
    }
    const u8 fracbits = 64 - scale.ZeroExtend<u8>();

    const IR::U16U32U64 fltval = V_scalar(*fltsize, Vn);
    IR::U32U64 intval;
    if (intsize == 32) {
        intval = ir.FPToFixedU32(fltval, fracbits, FP::RoundingMode::TowardsZero);
    } else if (intsize == 64) {
        intval = ir.FPToFixedU64(fltval, fracbits, FP::RoundingMode::TowardsZero);
    } else {
        UNREACHABLE();
    }

    X(intsize, Rd, intval);
    return true;
}

}  // namespace Dynarmic::A64
