/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/frontend/A64/translate/impl/impl.h"

namespace Dynarmic::A64 {

bool TranslatorVisitor::FMUL_float(Imm<2> type, Vec Vm, Vec Vn, Vec Vd) {
    const auto datasize = FPGetDataSize(type);
    if (!datasize) {
        return UnallocatedEncoding();
    }
    if (*datasize == 16) {
        V_scalar(16, Vd, HalfBinary(HalfOp::Mul, IR::U16{V_scalar(16, Vn)}, IR::U16{V_scalar(16, Vm)}));
        return true;
    }

    const IR::U32U64 operand1 = V_scalar(*datasize, Vn);
    const IR::U32U64 operand2 = V_scalar(*datasize, Vm);

    const IR::U32U64 result = ir.FPMul(operand1, operand2);

    V_scalar(*datasize, Vd, result);
    return true;
}

bool TranslatorVisitor::FDIV_float(Imm<2> type, Vec Vm, Vec Vn, Vec Vd) {
    const auto datasize = FPGetDataSize(type);
    if (!datasize) {
        return UnallocatedEncoding();
    }
    if (*datasize == 16) {
        V_scalar(16, Vd, HalfBinary(HalfOp::Div, IR::U16{V_scalar(16, Vn)}, IR::U16{V_scalar(16, Vm)}));
        return true;
    }

    const IR::U32U64 operand1 = V_scalar(*datasize, Vn);
    const IR::U32U64 operand2 = V_scalar(*datasize, Vm);

    const IR::U32U64 result = ir.FPDiv(operand1, operand2);

    V_scalar(*datasize, Vd, result);
    return true;
}

bool TranslatorVisitor::FADD_float(Imm<2> type, Vec Vm, Vec Vn, Vec Vd) {
    const auto datasize = FPGetDataSize(type);
    if (!datasize) {
        return UnallocatedEncoding();
    }
    if (*datasize == 16) {
        V_scalar(16, Vd, HalfBinary(HalfOp::Add, IR::U16{V_scalar(16, Vn)}, IR::U16{V_scalar(16, Vm)}));
        return true;
    }

    const IR::U32U64 operand1 = V_scalar(*datasize, Vn);
    const IR::U32U64 operand2 = V_scalar(*datasize, Vm);

    const IR::U32U64 result = ir.FPAdd(operand1, operand2);

    V_scalar(*datasize, Vd, result);
    return true;
}

bool TranslatorVisitor::FSUB_float(Imm<2> type, Vec Vm, Vec Vn, Vec Vd) {
    const auto datasize = FPGetDataSize(type);
    if (!datasize) {
        return UnallocatedEncoding();
    }
    if (*datasize == 16) {
        V_scalar(16, Vd, HalfBinary(HalfOp::Sub, IR::U16{V_scalar(16, Vn)}, IR::U16{V_scalar(16, Vm)}));
        return true;
    }

    const IR::U32U64 operand1 = V_scalar(*datasize, Vn);
    const IR::U32U64 operand2 = V_scalar(*datasize, Vm);

    const IR::U32U64 result = ir.FPSub(operand1, operand2);

    V_scalar(*datasize, Vd, result);
    return true;
}

bool TranslatorVisitor::FMAX_float(Imm<2> type, Vec Vm, Vec Vn, Vec Vd) {
    const auto datasize = FPGetDataSize(type);
    if (!datasize) {
        return UnallocatedEncoding();
    }
    if (*datasize == 16) {
        V_scalar(16, Vd, HalfBinary(HalfOp::Max, IR::U16{V_scalar(16, Vn)}, IR::U16{V_scalar(16, Vm)}));
        return true;
    }

    const IR::U32U64 operand1 = V_scalar(*datasize, Vn);
    const IR::U32U64 operand2 = V_scalar(*datasize, Vm);

    const IR::U32U64 result = ir.FPMax(operand1, operand2);

    V_scalar(*datasize, Vd, result);
    return true;
}

bool TranslatorVisitor::FMIN_float(Imm<2> type, Vec Vm, Vec Vn, Vec Vd) {
    const auto datasize = FPGetDataSize(type);
    if (!datasize) {
        return UnallocatedEncoding();
    }
    if (*datasize == 16) {
        V_scalar(16, Vd, HalfBinary(HalfOp::Min, IR::U16{V_scalar(16, Vn)}, IR::U16{V_scalar(16, Vm)}));
        return true;
    }

    const IR::U32U64 operand1 = V_scalar(*datasize, Vn);
    const IR::U32U64 operand2 = V_scalar(*datasize, Vm);

    const IR::U32U64 result = ir.FPMin(operand1, operand2);

    V_scalar(*datasize, Vd, result);
    return true;
}

bool TranslatorVisitor::FMAXNM_float(Imm<2> type, Vec Vm, Vec Vn, Vec Vd) {
    const auto datasize = FPGetDataSize(type);
    if (!datasize) {
        return UnallocatedEncoding();
    }
    if (*datasize == 16) {
        V_scalar(16, Vd, HalfBinary(HalfOp::MaxNum, IR::U16{V_scalar(16, Vn)}, IR::U16{V_scalar(16, Vm)}));
        return true;
    }

    const IR::U32U64 operand1 = V_scalar(*datasize, Vn);
    const IR::U32U64 operand2 = V_scalar(*datasize, Vm);

    const IR::U32U64 result = ir.FPMaxNumeric(operand1, operand2);

    V_scalar(*datasize, Vd, result);
    return true;
}

bool TranslatorVisitor::FMINNM_float(Imm<2> type, Vec Vm, Vec Vn, Vec Vd) {
    const auto datasize = FPGetDataSize(type);
    if (!datasize) {
        return UnallocatedEncoding();
    }
    if (*datasize == 16) {
        V_scalar(16, Vd, HalfBinary(HalfOp::MinNum, IR::U16{V_scalar(16, Vn)}, IR::U16{V_scalar(16, Vm)}));
        return true;
    }

    const IR::U32U64 operand1 = V_scalar(*datasize, Vn);
    const IR::U32U64 operand2 = V_scalar(*datasize, Vm);

    const IR::U32U64 result = ir.FPMinNumeric(operand1, operand2);

    V_scalar(*datasize, Vd, result);
    return true;
}

bool TranslatorVisitor::FNMUL_float(Imm<2> type, Vec Vm, Vec Vn, Vec Vd) {
    const auto datasize = FPGetDataSize(type);
    if (!datasize) {
        return UnallocatedEncoding();
    }
    if (*datasize == 16) {
        V_scalar(16, Vd, HalfNeg(HalfBinary(HalfOp::Mul, IR::U16{V_scalar(16, Vn)}, IR::U16{V_scalar(16, Vm)})));
        return true;
    }

    const IR::U32U64 operand1 = V_scalar(*datasize, Vn);
    const IR::U32U64 operand2 = V_scalar(*datasize, Vm);

    const IR::U32U64 result = ir.FPNeg(ir.FPMul(operand1, operand2));

    V_scalar(*datasize, Vd, result);
    return true;
}

}  // namespace Dynarmic::A64
