// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "dynarmic/common/fp/op/FPHalfArith.h"

#include <cmath>

#include "dynarmic/common/fp/info.h"
#include "dynarmic/common/fp/process_exception.h"
#include "dynarmic/common/fp/process_nan.h"
#include "dynarmic/common/fp/rounding_mode.h"
#include "dynarmic/common/fp/unpacked.h"

namespace Dynarmic::FP {

namespace {

using Info = FPInfo<u16>;

struct Operand {
    FPType type;
    bool sign;
    u64 m;  // value = m * 2^e for finite non-zero operands, m < 2^11
    int e;
};

Operand Unpack(u16 op, FPCR fpcr, FPSR& fpsr) {
    const auto [type, sign, value] = FPUnpack<u16>(op, fpcr, fpsr);
    Operand r{type, sign, 0, 0};
    if (type == FPType::Nonzero) {
        // Normalized mantissa has its leading bit at bit 62; a half has at most 11 significant bits.
        r.m = value.mantissa >> (normalized_point_position - Info::explicit_mantissa_width);
        r.e = value.exponent - static_cast<int>(Info::explicit_mantissa_width);
    }
    return r;
}

bool IsNaN(const Operand& o) {
    return o.type == FPType::QNaN || o.type == FPType::SNaN;
}

u16 Round(bool sign, int exponent, u64 magnitude, FPCR fpcr, FPSR& fpsr) {
    return FPRound<u16>(ToNormalized(sign, exponent, magnitude), fpcr, fpsr);
}

/// Exact comparison of two non-NaN operands: -1, 0 or 1.
int Compare(const Operand& a, const Operand& b) {
    const auto rank = [](const Operand& o) -> int {
        // Infinities and zeros are ordered without looking at the magnitude.
        if (o.type == FPType::Infinity) {
            return o.sign ? -2 : 2;
        }
        return 0;
    };
    const int ra = rank(a), rb = rank(b);
    if (ra != 0 || rb != 0) {
        return ra == rb ? 0 : (ra < rb ? -1 : 1);
    }
    const bool za = a.type == FPType::Zero, zb = b.type == FPType::Zero;
    if (za && zb) {
        return 0;
    }
    // Align both to the smaller exponent: shifts stay below 2^52 for half-precision operands.
    const int emin = za ? b.e : (zb ? a.e : std::min(a.e, b.e));
    const s64 xa = za ? 0 : static_cast<s64>(a.m << (a.e - emin)) * (a.sign ? -1 : 1);
    const s64 xb = zb ? 0 : static_cast<s64>(b.m << (b.e - emin)) * (b.sign ? -1 : 1);
    return xa == xb ? 0 : (xa < xb ? -1 : 1);
}

u16 AddImpl(u16 op1, u16 op2, bool subtract, FPCR fpcr, FPSR& fpsr) {
    Operand a = Unpack(op1, fpcr, fpsr);
    Operand b = Unpack(op2, fpcr, fpsr);
    if (const auto nan = FPProcessNaNs<u16>(a.type, b.type, op1, op2, fpcr, fpsr)) {
        return *nan;
    }
    if (subtract) {
        b.sign = !b.sign;
    }
    const bool inf1 = a.type == FPType::Infinity, inf2 = b.type == FPType::Infinity;
    const bool zero1 = a.type == FPType::Zero, zero2 = b.type == FPType::Zero;
    if (inf1 && inf2 && a.sign != b.sign) {
        FPProcessException(FPExc::InvalidOp, fpcr, fpsr);
        return Info::DefaultNaN();
    }
    if ((inf1 && !a.sign) || (inf2 && !b.sign)) {
        return Info::Infinity(false);
    }
    if ((inf1 && a.sign) || (inf2 && b.sign)) {
        return Info::Infinity(true);
    }
    if (zero1 && zero2 && a.sign == b.sign) {
        return Info::Zero(a.sign);
    }
    const int emin = zero1 ? b.e : (zero2 ? a.e : std::min(a.e, b.e));
    const s64 xa = zero1 ? 0 : static_cast<s64>(a.m << (a.e - emin)) * (a.sign ? -1 : 1);
    const s64 xb = zero2 ? 0 : static_cast<s64>(b.m << (b.e - emin)) * (b.sign ? -1 : 1);
    const s64 sum = xa + xb;
    if (sum == 0) {
        return Info::Zero(fpcr.RMode() == RoundingMode::TowardsMinusInfinity);
    }
    return Round(sum < 0, emin, static_cast<u64>(sum < 0 ? -sum : sum), fpcr, fpsr);
}

u16 MulImpl(u16 op1, u16 op2, bool mulx, FPCR fpcr, FPSR& fpsr) {
    const Operand a = Unpack(op1, fpcr, fpsr);
    const Operand b = Unpack(op2, fpcr, fpsr);
    if (const auto nan = FPProcessNaNs<u16>(a.type, b.type, op1, op2, fpcr, fpsr)) {
        return *nan;
    }
    const bool inf1 = a.type == FPType::Infinity, inf2 = b.type == FPType::Infinity;
    const bool zero1 = a.type == FPType::Zero, zero2 = b.type == FPType::Zero;
    const bool sign = a.sign != b.sign;
    if ((inf1 && zero2) || (zero1 && inf2)) {
        if (mulx) {
            return static_cast<u16>(0x4000 | Info::Zero(sign));  // FPTwo(sign)
        }
        FPProcessException(FPExc::InvalidOp, fpcr, fpsr);
        return Info::DefaultNaN();
    }
    if (inf1 || inf2) {
        return Info::Infinity(sign);
    }
    if (zero1 || zero2) {
        return Info::Zero(sign);
    }
    return Round(sign, a.e + b.e, a.m * b.m, fpcr, fpsr);
}

u16 MaxMinImpl(u16 op1, u16 op2, bool is_max, FPCR fpcr, FPSR& fpsr) {
    const Operand a = Unpack(op1, fpcr, fpsr);
    const Operand b = Unpack(op2, fpcr, fpsr);
    if (const auto nan = FPProcessNaNs<u16>(a.type, b.type, op1, op2, fpcr, fpsr)) {
        return *nan;
    }
    const int cmp = Compare(a, b);
    const Operand& r = (is_max ? cmp > 0 : cmp < 0) ? a : b;
    if (r.type == FPType::Infinity) {
        return Info::Infinity(r.sign);
    }
    if (r.type == FPType::Zero) {
        return Info::Zero(is_max ? (a.sign && b.sign) : (a.sign || b.sign));
    }
    return Round(r.sign, r.e, r.m, fpcr, fpsr);
}

u16 MaxMinNumImpl(u16 op1, u16 op2, bool is_max, FPCR fpcr, FPSR& fpsr) {
    // A single quiet NaN is replaced by the infinity that loses the comparison.
    FPSR scratch = fpsr;
    const auto t1 = std::get<0>(FPUnpack<u16>(op1, fpcr, scratch));
    const auto t2 = std::get<0>(FPUnpack<u16>(op2, fpcr, scratch));
    const u16 loser = Info::Infinity(is_max);
    if (t1 == FPType::QNaN && t2 != FPType::QNaN) {
        op1 = loser;
    } else if (t1 != FPType::QNaN && t2 == FPType::QNaN) {
        op2 = loser;
    }
    return MaxMinImpl(op1, op2, is_max, fpcr, fpsr);
}

}  // namespace

u16 FPAdd16(u16 op1, u16 op2, FPCR fpcr, FPSR& fpsr) {
    return AddImpl(op1, op2, false, fpcr, fpsr);
}

u16 FPSub16(u16 op1, u16 op2, FPCR fpcr, FPSR& fpsr) {
    return AddImpl(op1, op2, true, fpcr, fpsr);
}

u16 FPMul16(u16 op1, u16 op2, FPCR fpcr, FPSR& fpsr) {
    return MulImpl(op1, op2, false, fpcr, fpsr);
}

u16 FPMulX16(u16 op1, u16 op2, FPCR fpcr, FPSR& fpsr) {
    return MulImpl(op1, op2, true, fpcr, fpsr);
}

u16 FPDiv16(u16 op1, u16 op2, FPCR fpcr, FPSR& fpsr) {
    const Operand a = Unpack(op1, fpcr, fpsr);
    const Operand b = Unpack(op2, fpcr, fpsr);
    if (const auto nan = FPProcessNaNs<u16>(a.type, b.type, op1, op2, fpcr, fpsr)) {
        return *nan;
    }
    const bool inf1 = a.type == FPType::Infinity, inf2 = b.type == FPType::Infinity;
    const bool zero1 = a.type == FPType::Zero, zero2 = b.type == FPType::Zero;
    const bool sign = a.sign != b.sign;
    if ((inf1 && inf2) || (zero1 && zero2)) {
        FPProcessException(FPExc::InvalidOp, fpcr, fpsr);
        return Info::DefaultNaN();
    }
    if (inf1 || zero2) {
        if (!inf1) {
            FPProcessException(FPExc::DivideByZero, fpcr, fpsr);
        }
        return Info::Infinity(sign);
    }
    if (zero1 || inf2) {
        return Info::Zero(sign);
    }
    // a.m < 2^11, so the dividend fits; the quotient keeps at least 41 bits, and the remainder becomes a sticky bit.
    const u64 dividend = a.m << 52;
    u64 quotient = dividend / b.m;
    if (dividend % b.m != 0) {
        quotient |= 1;
    }
    return Round(sign, a.e - b.e - 52, quotient, fpcr, fpsr);
}

u16 FPSqrt16(u16 op, FPCR fpcr, FPSR& fpsr) {
    Operand a = Unpack(op, fpcr, fpsr);
    if (IsNaN(a)) {
        return FPProcessNaN<u16>(a.type, op, fpcr, fpsr);
    }
    if (a.type == FPType::Zero) {
        return Info::Zero(a.sign);
    }
    if (a.type == FPType::Infinity && !a.sign) {
        return Info::Infinity(false);
    }
    if (a.sign) {
        FPProcessException(FPExc::InvalidOp, fpcr, fpsr);
        return Info::DefaultNaN();
    }
    if ((a.e & 1) != 0) {
        a.m <<= 1;
        a.e -= 1;
    }
    // sqrt(m * 2^e) = sqrt(m * 2^50) * 2^(e/2 - 25); m < 2^12 keeps the radicand below 2^62.
    const u64 radicand = a.m << 50;
    u64 root = static_cast<u64>(std::sqrt(static_cast<double>(radicand)));
    while (root * root > radicand) {
        --root;
    }
    while ((root + 1) * (root + 1) <= radicand) {
        ++root;
    }
    if (root * root != radicand) {
        root |= 1;
    }
    return Round(false, a.e / 2 - 25, root, fpcr, fpsr);
}

u16 FPMax16(u16 op1, u16 op2, FPCR fpcr, FPSR& fpsr) {
    return MaxMinImpl(op1, op2, true, fpcr, fpsr);
}

u16 FPMin16(u16 op1, u16 op2, FPCR fpcr, FPSR& fpsr) {
    return MaxMinImpl(op1, op2, false, fpcr, fpsr);
}

u16 FPMaxNum16(u16 op1, u16 op2, FPCR fpcr, FPSR& fpsr) {
    return MaxMinNumImpl(op1, op2, true, fpcr, fpsr);
}

u16 FPMinNum16(u16 op1, u16 op2, FPCR fpcr, FPSR& fpsr) {
    return MaxMinNumImpl(op1, op2, false, fpcr, fpsr);
}

bool FPCompareGE16(u16 op1, u16 op2, FPCR fpcr, FPSR& fpsr) {
    const Operand a = Unpack(op1, fpcr, fpsr);
    const Operand b = Unpack(op2, fpcr, fpsr);
    if (IsNaN(a) || IsNaN(b)) {
        FPProcessException(FPExc::InvalidOp, fpcr, fpsr);
        return false;
    }
    return Compare(a, b) >= 0;
}

bool FPCompareGT16(u16 op1, u16 op2, FPCR fpcr, FPSR& fpsr) {
    const Operand a = Unpack(op1, fpcr, fpsr);
    const Operand b = Unpack(op2, fpcr, fpsr);
    if (IsNaN(a) || IsNaN(b)) {
        FPProcessException(FPExc::InvalidOp, fpcr, fpsr);
        return false;
    }
    return Compare(a, b) > 0;
}

u32 FPCompare16(u16 op1, u16 op2, bool signal_nans, FPCR fpcr, FPSR& fpsr) {
    const Operand a = Unpack(op1, fpcr, fpsr);
    const Operand b = Unpack(op2, fpcr, fpsr);
    if (IsNaN(a) || IsNaN(b)) {
        if (a.type == FPType::SNaN || b.type == FPType::SNaN || signal_nans) {
            FPProcessException(FPExc::InvalidOp, fpcr, fpsr);
        }
        return 0b0011u << 28;
    }
    const int cmp = Compare(a, b);
    if (cmp == 0) {
        return 0b0110u << 28;
    }
    return (cmp < 0 ? 0b1000u : 0b0010u) << 28;
}

u16 FPFixedToHalf(u64 value, bool is_signed, unsigned fbits, FPCR fpcr, RoundingMode rounding, FPSR& fpsr) {
    const bool sign = is_signed && static_cast<s64>(value) < 0;
    const u64 magnitude = sign ? ~value + 1 : value;
    if (magnitude == 0) {
        return Info::Zero(false);
    }
    return FPRound<u16>(ToNormalized(sign, -static_cast<int>(fbits), magnitude), fpcr, rounding, fpsr);
}

}  // namespace Dynarmic::FP
