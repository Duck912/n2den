// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// FEAT_FP16 Advanced SIMD instructions (half-precision lanes): three-same, two-register miscellaneous, pairwise,
// across-lanes and by-element forms, scalar and vector.
//
// A 64-bit vector leaves the upper four lanes zero, and an operation such as 0/0 there would raise a spurious
// FPSR flag. 64-bit operands therefore get their low half duplicated into the upper half first: the extra lanes
// raise exactly the flags of the real ones and are discarded. Scalar forms broadcast the element for the same reason.

#include "dynarmic/frontend/A64/translate/impl/impl.h"

namespace Dynarmic::A64 {

namespace {

constexpr size_t half = 16;

IR::U128 Fp16Operand(TranslatorVisitor& v, bool Q, Vec vec) {
    const IR::U128 x = v.V(Q ? 128 : 64, vec);
    return Q ? x : v.ir.VectorInterleaveLower(64, x, x);
}

IR::U128 Fp16ScalarOperand(TranslatorVisitor& v, Vec vec, size_t index = 0) {
    return v.ir.VectorBroadcastElement(half, v.V(128, vec), index);
}

void Fp16WriteScalar(TranslatorVisitor& v, Vec Vd, const IR::U128& result) {
    v.V_scalar(half, Vd, v.ir.VectorGetElement(half, result, 0));
}

enum class Fp16Binary { Add, Sub, Mul, MulX, Div, Max, Min, MaxNum, MinNum, Abd, CmGE, CmGT, AcGE, AcGT };

IR::U128 Fp16Apply(TranslatorVisitor& v, Fp16Binary op, const IR::U128& a, const IR::U128& b) {
    auto& ir = v.ir;
    switch (op) {
    case Fp16Binary::Add:
        return ir.FPVectorAdd(half, a, b);
    case Fp16Binary::Sub:
        return ir.FPVectorSub(half, a, b);
    case Fp16Binary::Mul:
        return ir.FPVectorMul(half, a, b);
    case Fp16Binary::MulX:
        return ir.FPVectorMulX(half, a, b);
    case Fp16Binary::Div:
        return ir.FPVectorDiv(half, a, b);
    case Fp16Binary::Max:
        return ir.FPVectorMax(half, a, b);
    case Fp16Binary::Min:
        return ir.FPVectorMin(half, a, b);
    case Fp16Binary::MaxNum:
        return ir.FPVectorMaxNumeric(half, a, b);
    case Fp16Binary::MinNum:
        return ir.FPVectorMinNumeric(half, a, b);
    case Fp16Binary::Abd:
        return ir.FPVectorAbs(half, ir.FPVectorSub(half, a, b));
    case Fp16Binary::CmGE:
        return ir.FPVectorGreaterEqual(half, a, b);
    case Fp16Binary::CmGT:
        return ir.FPVectorGreater(half, a, b);
    case Fp16Binary::AcGE:
        return ir.FPVectorGreaterEqual(half, ir.FPVectorAbs(half, a), ir.FPVectorAbs(half, b));
    case Fp16Binary::AcGT:
        return ir.FPVectorGreater(half, ir.FPVectorAbs(half, a), ir.FPVectorAbs(half, b));
    }
    UNREACHABLE();
}

bool Fp16VectorBinary(TranslatorVisitor& v, bool Q, Vec Vm, Vec Vn, Vec Vd, Fp16Binary op) {
    const IR::U128 result = Fp16Apply(v, op, Fp16Operand(v, Q, Vn), Fp16Operand(v, Q, Vm));
    v.V(Q ? 128 : 64, Vd, result);
    return true;
}

bool Fp16ScalarBinary(TranslatorVisitor& v, Vec Vm, Vec Vn, Vec Vd, Fp16Binary op) {
    Fp16WriteScalar(v, Vd, Fp16Apply(v, op, Fp16ScalarOperand(v, Vn), Fp16ScalarOperand(v, Vm)));
    return true;
}

// Pairwise: lane e of the result is op(concat[2e], concat[2e+1]), concat = Vm:Vn.
bool Fp16VectorPairwise(TranslatorVisitor& v, bool Q, Vec Vm, Vec Vn, Vec Vd, Fp16Binary op) {
    IR::U128 even, odd;
    if (Q) {
        even = v.ir.VectorDeinterleaveEven(half, v.V(128, Vn), v.V(128, Vm));
        odd = v.ir.VectorDeinterleaveOdd(half, v.V(128, Vn), v.V(128, Vm));
    } else {
        const IR::U128 concat = v.ir.VectorInterleaveLower(64, v.V(64, Vn), v.V(64, Vm));
        even = v.ir.VectorDeinterleaveEven(half, concat, concat);
        odd = v.ir.VectorDeinterleaveOdd(half, concat, concat);
    }
    v.V(Q ? 128 : 64, Vd, Fp16Apply(v, op, even, odd));
    return true;
}

bool Fp16ScalarPairwise(TranslatorVisitor& v, Vec Vn, Vec Vd, Fp16Binary op) {
    Fp16WriteScalar(v, Vd, Fp16Apply(v, op, Fp16ScalarOperand(v, Vn, 0), Fp16ScalarOperand(v, Vn, 1)));
    return true;
}

// Across lanes, reduced as the Arm ARM's Reduce(): adjacent pairs first, then pairs of those results.
bool Fp16AcrossLanes(TranslatorVisitor& v, bool Q, Vec Vn, Vec Vd, Fp16Binary op) {
    IR::U128 x = Fp16Operand(v, Q, Vn);
    for (size_t lanes = Q ? 8 : 4; lanes > 1; lanes /= 2) {
        x = Fp16Apply(v, op, v.ir.VectorDeinterleaveEven(half, x, x), v.ir.VectorDeinterleaveOdd(half, x, x));
    }
    Fp16WriteScalar(v, Vd, x);
    return true;
}

enum class Fp16CompareZero { GE, GT, LE, LT };

IR::U128 Fp16ApplyCompareZero(TranslatorVisitor& v, Fp16CompareZero cmp, const IR::U128& x) {
    const IR::U128 zero = v.ir.ZeroVector();
    switch (cmp) {
    case Fp16CompareZero::GE:
        return v.ir.FPVectorGreaterEqual(half, x, zero);
    case Fp16CompareZero::GT:
        return v.ir.FPVectorGreater(half, x, zero);
    case Fp16CompareZero::LE:
        return v.ir.FPVectorGreaterEqual(half, zero, x);
    case Fp16CompareZero::LT:
        return v.ir.FPVectorGreater(half, zero, x);
    }
    UNREACHABLE();
}

enum class Fp16Convert { ToSigned, ToUnsigned, FromSigned, FromUnsigned };

IR::U128 Fp16ApplyConvert(TranslatorVisitor& v, Fp16Convert kind, const IR::U128& x, size_t fbits, FP::RoundingMode rounding) {
    switch (kind) {
    case Fp16Convert::ToSigned:
        return v.ir.FPVectorToSignedFixed(half, x, fbits, rounding);
    case Fp16Convert::ToUnsigned:
        return v.ir.FPVectorToUnsignedFixed(half, x, fbits, rounding);
    case Fp16Convert::FromSigned:
        return v.ir.FPVectorFromSignedFixed(half, x, fbits, rounding);
    case Fp16Convert::FromUnsigned:
        return v.ir.FPVectorFromUnsignedFixed(half, x, fbits, rounding);
    }
    UNREACHABLE();
}

FP::RoundingMode CurrentRounding(TranslatorVisitor& v) {
    return v.ir.current_location->FPCR().RMode();
}

}  // Anonymous namespace

// Shared with the fixed-point (shift by immediate) forms.
bool Fp16VectorConvert(TranslatorVisitor& v, bool Q, Vec Vn, Vec Vd, bool to_fixed, bool is_signed, size_t fbits, std::optional<FP::RoundingMode> rounding) {
    const Fp16Convert kind = to_fixed ? (is_signed ? Fp16Convert::ToSigned : Fp16Convert::ToUnsigned)
                                      : (is_signed ? Fp16Convert::FromSigned : Fp16Convert::FromUnsigned);
    v.V(Q ? 128 : 64, Vd, Fp16ApplyConvert(v, kind, Fp16Operand(v, Q, Vn), fbits, rounding.value_or(CurrentRounding(v))));
    return true;
}

bool Fp16ScalarConvert(TranslatorVisitor& v, Vec Vn, Vec Vd, bool to_fixed, bool is_signed, size_t fbits, std::optional<FP::RoundingMode> rounding) {
    const Fp16Convert kind = to_fixed ? (is_signed ? Fp16Convert::ToSigned : Fp16Convert::ToUnsigned)
                                      : (is_signed ? Fp16Convert::FromSigned : Fp16Convert::FromUnsigned);
    Fp16WriteScalar(v, Vd, Fp16ApplyConvert(v, kind, Fp16ScalarOperand(v, Vn), fbits, rounding.value_or(CurrentRounding(v))));
    return true;
}

// ---- Three same (FP16)

bool TranslatorVisitor::FMAXNM_1(bool Q, Vec Vm, Vec Vn, Vec Vd) { return Fp16VectorBinary(*this, Q, Vm, Vn, Vd, Fp16Binary::MaxNum); }
bool TranslatorVisitor::FADD_1(bool Q, Vec Vm, Vec Vn, Vec Vd) { return Fp16VectorBinary(*this, Q, Vm, Vn, Vd, Fp16Binary::Add); }
bool TranslatorVisitor::FMULX_vec_3(bool Q, Vec Vm, Vec Vn, Vec Vd) { return Fp16VectorBinary(*this, Q, Vm, Vn, Vd, Fp16Binary::MulX); }
bool TranslatorVisitor::FMAX_1(bool Q, Vec Vm, Vec Vn, Vec Vd) { return Fp16VectorBinary(*this, Q, Vm, Vn, Vd, Fp16Binary::Max); }
bool TranslatorVisitor::FMINNM_1(bool Q, Vec Vm, Vec Vn, Vec Vd) { return Fp16VectorBinary(*this, Q, Vm, Vn, Vd, Fp16Binary::MinNum); }
bool TranslatorVisitor::FSUB_1(bool Q, Vec Vm, Vec Vn, Vec Vd) { return Fp16VectorBinary(*this, Q, Vm, Vn, Vd, Fp16Binary::Sub); }
bool TranslatorVisitor::FMIN_1(bool Q, Vec Vm, Vec Vn, Vec Vd) { return Fp16VectorBinary(*this, Q, Vm, Vn, Vd, Fp16Binary::Min); }
bool TranslatorVisitor::FMUL_vec_1(bool Q, Vec Vm, Vec Vn, Vec Vd) { return Fp16VectorBinary(*this, Q, Vm, Vn, Vd, Fp16Binary::Mul); }
bool TranslatorVisitor::FCMGE_reg_3(bool Q, Vec Vm, Vec Vn, Vec Vd) { return Fp16VectorBinary(*this, Q, Vm, Vn, Vd, Fp16Binary::CmGE); }
bool TranslatorVisitor::FACGE_3(bool Q, Vec Vm, Vec Vn, Vec Vd) { return Fp16VectorBinary(*this, Q, Vm, Vn, Vd, Fp16Binary::AcGE); }
bool TranslatorVisitor::FDIV_1(bool Q, Vec Vm, Vec Vn, Vec Vd) { return Fp16VectorBinary(*this, Q, Vm, Vn, Vd, Fp16Binary::Div); }
bool TranslatorVisitor::FABD_3(bool Q, Vec Vm, Vec Vn, Vec Vd) { return Fp16VectorBinary(*this, Q, Vm, Vn, Vd, Fp16Binary::Abd); }
bool TranslatorVisitor::FCMGT_reg_3(bool Q, Vec Vm, Vec Vn, Vec Vd) { return Fp16VectorBinary(*this, Q, Vm, Vn, Vd, Fp16Binary::CmGT); }
bool TranslatorVisitor::FACGT_3(bool Q, Vec Vm, Vec Vn, Vec Vd) { return Fp16VectorBinary(*this, Q, Vm, Vn, Vd, Fp16Binary::AcGT); }

bool TranslatorVisitor::FMAXNMP_vec_1(bool Q, Vec Vm, Vec Vn, Vec Vd) { return Fp16VectorPairwise(*this, Q, Vm, Vn, Vd, Fp16Binary::MaxNum); }
bool TranslatorVisitor::FADDP_vec_1(bool Q, Vec Vm, Vec Vn, Vec Vd) { return Fp16VectorPairwise(*this, Q, Vm, Vn, Vd, Fp16Binary::Add); }
bool TranslatorVisitor::FMAXP_vec_1(bool Q, Vec Vm, Vec Vn, Vec Vd) { return Fp16VectorPairwise(*this, Q, Vm, Vn, Vd, Fp16Binary::Max); }
bool TranslatorVisitor::FMINNMP_vec_1(bool Q, Vec Vm, Vec Vn, Vec Vd) { return Fp16VectorPairwise(*this, Q, Vm, Vn, Vd, Fp16Binary::MinNum); }
bool TranslatorVisitor::FMINP_vec_1(bool Q, Vec Vm, Vec Vn, Vec Vd) { return Fp16VectorPairwise(*this, Q, Vm, Vn, Vd, Fp16Binary::Min); }

// ---- Scalar three same (FP16)

bool TranslatorVisitor::FMULX_vec_1(Vec Vm, Vec Vn, Vec Vd) { return Fp16ScalarBinary(*this, Vm, Vn, Vd, Fp16Binary::MulX); }
bool TranslatorVisitor::FCMGE_reg_1(Vec Vm, Vec Vn, Vec Vd) { return Fp16ScalarBinary(*this, Vm, Vn, Vd, Fp16Binary::CmGE); }
bool TranslatorVisitor::FACGE_1(Vec Vm, Vec Vn, Vec Vd) { return Fp16ScalarBinary(*this, Vm, Vn, Vd, Fp16Binary::AcGE); }
bool TranslatorVisitor::FABD_1(Vec Vm, Vec Vn, Vec Vd) { return Fp16ScalarBinary(*this, Vm, Vn, Vd, Fp16Binary::Abd); }
bool TranslatorVisitor::FCMGT_reg_1(Vec Vm, Vec Vn, Vec Vd) { return Fp16ScalarBinary(*this, Vm, Vn, Vd, Fp16Binary::CmGT); }
bool TranslatorVisitor::FACGT_1(Vec Vm, Vec Vn, Vec Vd) { return Fp16ScalarBinary(*this, Vm, Vn, Vd, Fp16Binary::AcGT); }

// ---- Scalar pairwise and across lanes (FP16)

bool TranslatorVisitor::FADDP_pair_1(Vec Vn, Vec Vd) { return Fp16ScalarPairwise(*this, Vn, Vd, Fp16Binary::Add); }
bool TranslatorVisitor::FMAXNMP_pair_1(Vec Vn, Vec Vd) { return Fp16ScalarPairwise(*this, Vn, Vd, Fp16Binary::MaxNum); }
bool TranslatorVisitor::FMAXP_pair_1(Vec Vn, Vec Vd) { return Fp16ScalarPairwise(*this, Vn, Vd, Fp16Binary::Max); }
bool TranslatorVisitor::FMINNMP_pair_1(Vec Vn, Vec Vd) { return Fp16ScalarPairwise(*this, Vn, Vd, Fp16Binary::MinNum); }
bool TranslatorVisitor::FMINP_pair_1(Vec Vn, Vec Vd) { return Fp16ScalarPairwise(*this, Vn, Vd, Fp16Binary::Min); }

bool TranslatorVisitor::FMAXNMV_1(bool Q, Vec Vn, Vec Vd) { return Fp16AcrossLanes(*this, Q, Vn, Vd, Fp16Binary::MaxNum); }
bool TranslatorVisitor::FMAXV_1(bool Q, Vec Vn, Vec Vd) { return Fp16AcrossLanes(*this, Q, Vn, Vd, Fp16Binary::Max); }
bool TranslatorVisitor::FMINNMV_1(bool Q, Vec Vn, Vec Vd) { return Fp16AcrossLanes(*this, Q, Vn, Vd, Fp16Binary::MinNum); }
bool TranslatorVisitor::FMINV_1(bool Q, Vec Vn, Vec Vd) { return Fp16AcrossLanes(*this, Q, Vn, Vd, Fp16Binary::Min); }

// ---- Two-register miscellaneous (FP16)

bool TranslatorVisitor::FSQRT_1(bool Q, Vec Vn, Vec Vd) {
    V(Q ? 128 : 64, Vd, ir.FPVectorSqrt(half, Fp16Operand(*this, Q, Vn)));
    return true;
}

#define FP16_COMPARE_ZERO(name_vec, name_scalar, cmp)                                                  \
    bool TranslatorVisitor::name_vec(bool Q, Vec Vn, Vec Vd) {                                         \
        V(Q ? 128 : 64, Vd, Fp16ApplyCompareZero(*this, Fp16CompareZero::cmp, Fp16Operand(*this, Q, Vn))); \
        return true;                                                                                   \
    }                                                                                                  \
    bool TranslatorVisitor::name_scalar(Vec Vn, Vec Vd) {                                              \
        Fp16WriteScalar(*this, Vd, Fp16ApplyCompareZero(*this, Fp16CompareZero::cmp, Fp16ScalarOperand(*this, Vn))); \
        return true;                                                                                   \
    }

FP16_COMPARE_ZERO(FCMGE_zero_3, FCMGE_zero_1, GE)
FP16_COMPARE_ZERO(FCMGT_zero_3, FCMGT_zero_1, GT)
FP16_COMPARE_ZERO(FCMLE_3, FCMLE_1, LE)
FP16_COMPARE_ZERO(FCMLT_3, FCMLT_1, LT)
#undef FP16_COMPARE_ZERO

#define FP16_CONVERT(name_vec, name_scalar, to_fixed, is_signed, rounding)               \
    bool TranslatorVisitor::name_vec(bool Q, Vec Vn, Vec Vd) {                           \
        return Fp16VectorConvert(*this, Q, Vn, Vd, to_fixed, is_signed, 0, rounding);   \
    }                                                                                    \
    bool TranslatorVisitor::name_scalar(Vec Vn, Vec Vd) {                                \
        return Fp16ScalarConvert(*this, Vn, Vd, to_fixed, is_signed, 0, rounding);      \
    }

FP16_CONVERT(FCVTNS_3, FCVTNS_1, true, true, FP::RoundingMode::ToNearest_TieEven)
FP16_CONVERT(FCVTMS_3, FCVTMS_1, true, true, FP::RoundingMode::TowardsMinusInfinity)
FP16_CONVERT(FCVTAS_3, FCVTAS_1, true, true, FP::RoundingMode::ToNearest_TieAwayFromZero)
FP16_CONVERT(FCVTPS_3, FCVTPS_1, true, true, FP::RoundingMode::TowardsPlusInfinity)
FP16_CONVERT(FCVTZS_int_3, FCVTZS_int_1, true, true, FP::RoundingMode::TowardsZero)
FP16_CONVERT(FCVTNU_3, FCVTNU_1, true, false, FP::RoundingMode::ToNearest_TieEven)
FP16_CONVERT(FCVTMU_3, FCVTMU_1, true, false, FP::RoundingMode::TowardsMinusInfinity)
FP16_CONVERT(FCVTAU_3, FCVTAU_1, true, false, FP::RoundingMode::ToNearest_TieAwayFromZero)
FP16_CONVERT(FCVTPU_3, FCVTPU_1, true, false, FP::RoundingMode::TowardsPlusInfinity)
FP16_CONVERT(FCVTZU_int_3, FCVTZU_int_1, true, false, FP::RoundingMode::TowardsZero)
FP16_CONVERT(SCVTF_int_3, SCVTF_int_1, false, true, std::nullopt)
FP16_CONVERT(UCVTF_int_3, UCVTF_int_1, false, false, std::nullopt)
#undef FP16_CONVERT

// ---- By element (FP16): index = H:L:M, Vm = Vmlo (V0-V15)

bool TranslatorVisitor::FMUL_elt_3(bool Q, Imm<1> L, Imm<1> M, Imm<4> Vmlo, Imm<1> H, Vec Vn, Vec Vd) {
    const size_t index = concatenate(H, L, M).ZeroExtend();
    const IR::U128 element = ir.VectorBroadcastElement(half, V(128, static_cast<Vec>(Vmlo.ZeroExtend())), index);
    V(Q ? 128 : 64, Vd, ir.FPVectorMul(half, Fp16Operand(*this, Q, Vn), element));
    return true;
}

bool TranslatorVisitor::FMULX_elt_3(bool Q, Imm<1> L, Imm<1> M, Imm<4> Vmlo, Imm<1> H, Vec Vn, Vec Vd) {
    const size_t index = concatenate(H, L, M).ZeroExtend();
    const IR::U128 element = ir.VectorBroadcastElement(half, V(128, static_cast<Vec>(Vmlo.ZeroExtend())), index);
    V(Q ? 128 : 64, Vd, ir.FPVectorMulX(half, Fp16Operand(*this, Q, Vn), element));
    return true;
}

bool TranslatorVisitor::FMUL_elt_1(Imm<1> L, Imm<1> M, Imm<4> Vmlo, Imm<1> H, Vec Vn, Vec Vd) {
    const size_t index = concatenate(H, L, M).ZeroExtend();
    Fp16WriteScalar(*this, Vd, ir.FPVectorMul(half, Fp16ScalarOperand(*this, Vn), Fp16ScalarOperand(*this, static_cast<Vec>(Vmlo.ZeroExtend()), index)));
    return true;
}

bool TranslatorVisitor::FMULX_elt_1(Imm<1> L, Imm<1> M, Imm<4> Vmlo, Imm<1> H, Vec Vn, Vec Vd) {
    const size_t index = concatenate(H, L, M).ZeroExtend();
    Fp16WriteScalar(*this, Vd, ir.FPVectorMulX(half, Fp16ScalarOperand(*this, Vn), Fp16ScalarOperand(*this, static_cast<Vec>(Vmlo.ZeroExtend()), index)));
    return true;
}

}  // namespace Dynarmic::A64
