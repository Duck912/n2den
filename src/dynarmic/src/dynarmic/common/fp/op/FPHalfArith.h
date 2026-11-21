// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "dynarmic/common/fp/fpcr.h"
#include "dynarmic/common/fp/fpsr.h"
#include "dynarmic/common/fp/rounding_mode.h"

namespace Dynarmic::FP {

// Half-precision (FEAT_FP16) arithmetic, following the Arm ARM pseudocode: NaN propagation, FPCR.DN/FZ16/RMode,
// and FPSR cumulative exception flags. Results are correctly rounded: the exact result is formed with integer
// arithmetic (or with a sticky bit for division and square root) and rounded once.

u16 FPAdd16(u16 op1, u16 op2, FPCR fpcr, FPSR& fpsr);
u16 FPSub16(u16 op1, u16 op2, FPCR fpcr, FPSR& fpsr);
u16 FPMul16(u16 op1, u16 op2, FPCR fpcr, FPSR& fpsr);
u16 FPMulX16(u16 op1, u16 op2, FPCR fpcr, FPSR& fpsr);
u16 FPDiv16(u16 op1, u16 op2, FPCR fpcr, FPSR& fpsr);
u16 FPSqrt16(u16 op, FPCR fpcr, FPSR& fpsr);
u16 FPMax16(u16 op1, u16 op2, FPCR fpcr, FPSR& fpsr);
u16 FPMin16(u16 op1, u16 op2, FPCR fpcr, FPSR& fpsr);
u16 FPMaxNum16(u16 op1, u16 op2, FPCR fpcr, FPSR& fpsr);
u16 FPMinNum16(u16 op1, u16 op2, FPCR fpcr, FPSR& fpsr);

/// FCMGE / FCMGT: any NaN operand raises Invalid Operation and the comparison is false.
bool FPCompareGE16(u16 op1, u16 op2, FPCR fpcr, FPSR& fpsr);
bool FPCompareGT16(u16 op1, u16 op2, FPCR fpcr, FPSR& fpsr);

/// FCMP / FCMPE: returns NZCV in bits 31:28.
u32 FPCompare16(u16 op1, u16 op2, bool signal_nans, FPCR fpcr, FPSR& fpsr);

/// SCVTF / UCVTF to half: value / 2^fbits, rounded once with the given rounding mode.
u16 FPFixedToHalf(u64 value, bool is_signed, unsigned fbits, FPCR fpcr, RoundingMode rounding, FPSR& fpsr);

}  // namespace Dynarmic::FP
