/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/frontend/A64/translate/impl/impl.h"

namespace Dynarmic::A64 {
namespace {
bool FPCompare(TranslatorVisitor& v, Imm<2> type, Vec Vm, Cond cond, Vec Vn, Imm<4> nzcv, bool exc_on_qnan) {
    const auto datasize = FPGetDataSize(type);
    if (!datasize) {
        return v.UnallocatedEncoding();
    }
    const u32 flags = nzcv.ZeroExtend<u32>() << 28;
    if (*datasize == 16) {
        // Compare only when the condition holds: a failed condition compares zeros, which raises no exception.
        const auto operand = [&](Vec vec) {
            const IR::U32 value = v.ir.ZeroExtendToWord(v.V_scalar(16, vec));
            return v.ir.LeastSignificantHalf(v.ir.ConditionalSelect(cond, value, v.ir.Imm32(0)));
        };
        const IR::NZCV then_flags = v.ir.FPCompare16(operand(Vn), operand(Vm), exc_on_qnan);
        const IR::NZCV else_flags = v.ir.NZCVFromPackedFlags(v.ir.Imm32(flags));
        v.ir.SetNZCV(v.ir.ConditionalSelect(cond, then_flags, else_flags));
        return true;
    }

    const IR::U32U64 operand1 = v.V_scalar(*datasize, Vn);
    const IR::U32U64 operand2 = v.V_scalar(*datasize, Vm);

    const IR::NZCV then_flags = v.ir.FPCompare(operand1, operand2, exc_on_qnan);
    const IR::NZCV else_flags = v.ir.NZCVFromPackedFlags(v.ir.Imm32(flags));
    v.ir.SetNZCV(v.ir.ConditionalSelect(cond, then_flags, else_flags));
    return true;
}
}  // Anonymous namespace

bool TranslatorVisitor::FCCMP_float(Imm<2> type, Vec Vm, Cond cond, Vec Vn, Imm<4> nzcv) {
    return FPCompare(*this, type, Vm, cond, Vn, nzcv, false);
}

bool TranslatorVisitor::FCCMPE_float(Imm<2> type, Vec Vm, Cond cond, Vec Vn, Imm<4> nzcv) {
    return FPCompare(*this, type, Vm, cond, Vn, nzcv, true);
}

}  // namespace Dynarmic::A64
