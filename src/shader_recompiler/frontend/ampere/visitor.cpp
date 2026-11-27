// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Encoding reference: Mesa NAK sm70_encode.rs (MIT), Copyright 2022 Collabora, Ltd.

#include <bit>
#include <string>

#include "shader_recompiler/frontend/ampere/isa.h"
#include "shader_recompiler/frontend/ampere/visitor.h"

namespace Shader::Ampere {

Visitor::Visitor(Environment& env_, IR::Block& block, const Instruction& instruction, u32 index_,
                 ProgramInfo* program_info_, const UniformLayout& layout_, bool probe_)
    : env{env_}, ir{block}, insn{instruction}, index{index_}, program_info{program_info_},
      layout{layout_}, probe{probe_}, op{instruction.Bits(0, 12)} {
    // Opcode, guard predicate and the scheduling word (105..127) are never instruction semantics.
    consumed = {0x0000ffffU, 0, 0, 0xfffffe00U};
}

u32 Visitor::Take(u32 lo, u32 count) {
    for (u32 bit = lo; bit < lo + count; ++bit) {
        consumed[bit / 32] |= 1U << (bit % 32);
    }
    return insn.Bits(lo, count);
}

void Visitor::Ignore(u32 lo, u32 count) {
    for (u32 bit = lo; bit < lo + count; ++bit) {
        consumed[bit / 32] |= 1U << (bit % 32);
    }
}

void Visitor::Unsupported(std::string_view reason) const {
    throw UnsupportedError(index, reason);
}

IR::U32 Visitor::X(u32 reg) {
    if (reg >= 255) {
        return ir.Imm32(0U);
    }
    NoteGpr(reg);
    return ir.GetReg(static_cast<IR::Reg>(reg));
}

IR::F32 Visitor::F(u32 reg) {
    return ir.BitCast<IR::F32>(X(reg));
}

void Visitor::X(u32 reg, const IR::U32& value) {
    if (reg >= 255) {
        return; // RZ discards
    }
    NoteGpr(reg);
    const IR::Reg dest = static_cast<IR::Reg>(reg);
    if (guard_select) {
        ir.SetReg(dest, IR::U32{ir.Select(*guard_select, value, ir.GetReg(dest))});
    } else {
        ir.SetReg(dest, value);
    }
}

void Visitor::F(u32 reg, const IR::F32& value) {
    X(reg, ir.BitCast<IR::U32>(value));
}

IR::U1 Visitor::Pred(u32 pred_index, bool negated) {
    if (pred_index == 7) {
        return ir.Imm1(!negated);
    }
    return ir.GetPred(static_cast<IR::Pred>(pred_index), negated);
}

void Visitor::SetPred(u32 pred_index, const IR::U1& value) {
    if (pred_index == 7) {
        return; // PT discards
    }
    const IR::Pred dest = static_cast<IR::Pred>(pred_index);
    if (guard_select) {
        ir.SetPred(dest, IR::U1{ir.Select(*guard_select, value, ir.GetPred(dest))});
    } else {
        ir.SetPred(dest, value);
    }
}

void Visitor::NoteGpr(u32 reg) {
    if (probe) {
        if (program_info) {
            program_info->max_gpr = (std::max)(program_info->max_gpr, reg);
            program_info->used_gprs[reg] = true;
        }
        return;
    }
    Require(layout.scattered ? !layout.reserved[reg] : reg < layout.gpr_limit,
            "native program uses registers reserved for the uniform file");
}

IR::Reg Visitor::UniformSlot(u32 reg) {
    if (probe) {
        if (program_info) {
            program_info->uses_any_ur = true;
            program_info->max_ur = (std::max)(program_info->max_ur, reg);
        }
        return static_cast<IR::Reg>(UNIFORM_REGISTER_BASE + reg); // the probe result is discarded
    }
    return static_cast<IR::Reg>(layout.scattered ? layout.slots[reg] : layout.register_base + reg);
}

IR::Reg Visitor::PredicateSlot() {
    if (probe) {
        if (program_info) program_info->uses_up = true;
        return static_cast<IR::Reg>(UNIFORM_PREDICATE_SLOT);
    }
    return static_cast<IR::Reg>(layout.predicate_slot);
}

IR::U32 Visitor::UR(u32 reg) {
    if (reg >= 63) {
        return ir.Imm32(0U); // URZ
    }
    return ir.GetReg(UniformSlot(reg));
}

void Visitor::UR(u32 reg, const IR::U32& value) {
    if (reg >= 63) {
        return;
    }
    const IR::Reg dest = UniformSlot(reg);
    if (guard_select) {
        ir.SetReg(dest, IR::U32{ir.Select(*guard_select, value, ir.GetReg(dest))});
    } else {
        ir.SetReg(dest, value);
    }
}

IR::U1 Visitor::UPred(u32 pred_index, bool negated) {
    if (pred_index >= 7) {
        return ir.Imm1(!negated);
    }
    const IR::U32 bits{ir.GetReg(PredicateSlot())};
    const IR::U1 bit{ir.INotEqual(ir.BitFieldExtract(bits, ir.Imm32(pred_index), ir.Imm32(1U)), ir.Imm32(0U))};
    return negated ? IR::U1{ir.LogicalNot(bit)} : bit;
}

void Visitor::SetUPred(u32 pred_index, const IR::U1& value) {
    if (pred_index >= 7) {
        return; // UPT discards
    }
    const IR::Reg slot = PredicateSlot();
    const IR::U32 old{ir.GetReg(slot)};
    IR::U1 final_value{value};
    if (guard_select) {
        final_value = IR::U1{ir.Select(*guard_select, value, UPred(pred_index, false))};
    }
    const IR::U32 bit{ir.Select(final_value, ir.Imm32(1U), ir.Imm32(0U))};
    ir.SetReg(slot, ir.BitFieldInsert(old, bit, ir.Imm32(pred_index), ir.Imm32(1U)));
}

u32 Visitor::Dst() {
    return Take(16, uniform_op ? 6 : 8);
}

void Visitor::WriteDst(u32 reg, const IR::U32& value) {
    if (uniform_op) {
        UR(reg, value);
    } else {
        X(reg, value);
    }
}

IR::U1 Visitor::PredIn(u32 pred_index, bool negated) {
    return uniform_op ? UPred(pred_index, negated) : Pred(pred_index, negated);
}

void Visitor::PredOut(u32 pred_index, const IR::U1& value) {
    if (uniform_op) {
        SetUPred(pred_index, value);
    } else {
        SetPred(pred_index, value);
    }
}

IR::U32 Visitor::Cbuf(u32 cbuf_index, u32 byte_offset) {
    // Bank 24 is read like the others (the 3D engine and the buffer cache keep 32 banks), as the
    // lowering reads it for operands. Fortune's Weave's light shaders bind a camera matrix there
    // (c[0x18][0x0] to [0x4c]); reading zeros instead, as this once did, left the town square unlit
    // and its main character black (d0b4eb7fcc162208, 06db480b69cbfcf3).
    if (cbuf_index >= 18 && cbuf_index != 24) {
        Unsupported("native constant buffer address");
    }
    const u32 skew = byte_offset % 4;
    if (skew == 0) {
        return ir.GetCbuf(ir.Imm32(cbuf_index), ir.Imm32(byte_offset));
    }
    // A byte address inside a word reads 32 bits starting there: the tail of the word it falls in
    // and the head of the next one (little endian).
    const IR::U32 low{ir.GetCbuf(ir.Imm32(cbuf_index), ir.Imm32(byte_offset - skew))};
    const IR::U32 high{ir.GetCbuf(ir.Imm32(cbuf_index), ir.Imm32(byte_offset - skew + 4))};
    return ir.BitwiseOr(ir.ShiftRightLogical(low, ir.Imm32(8 * skew)),
                        ir.ShiftLeftLogical(high, ir.Imm32(32 - 8 * skew)));
}

Operand Visitor::WideSlot(bool modifiers) {
    Operand operand;
    switch (Form()) {
    case 2:
    case 4:
        operand.kind = Operand::Kind::Imm;
        operand.imm = Take(32, 32);
        break;
    case 3:
    case 5:
        operand.kind = Operand::Kind::Cbuf;
        Require(Take(91, 1) == 0, "native bindless constant buffer");
        operand.cbuf_index = Take(54, 5);
        operand.cbuf_offset = Take(38, 16);
        if (modifiers) {
            operand.abs = Take(62, 1) != 0;
            operand.negate = Take(63, 1) != 0;
        }
        break;
    case 6:
    case 7:
        operand.kind = Operand::Kind::Ureg;
        Require(Take(91, 1) == 1, "native uniform source flag");
        operand.reg = Take(32, 6);
        if (modifiers) {
            operand.abs = Take(62, 1) != 0;
            operand.negate = Take(63, 1) != 0;
        }
        break;
    default:
        Unsupported("native source form");
    }
    return operand;
}

Operand Visitor::Source(u32 which, bool modifiers) {
    Operand operand;
    const u32 form = Form();
    Require(form >= 1 && form <= 7, "native ALU source form");
    const u32 reg_bits = uniform_op ? 6 : 8;
    if (uniform_op) {
        operand.kind = Operand::Kind::Ureg;
    }
    if (which == 0) {
        operand.reg = Take(24, reg_bits);
        if (modifiers) {
            operand.negate = Take(72, 1) != 0;
            operand.abs = Take(73, 1) != 0;
        }
        return operand;
    }
    // Sources 1 and 2 share two physical slots: the wide slot (32..63) and a register in 64..71.
    const bool wide = which == 1 ? (form == 4 || form == 5 || form == 6)
                                 : (form == 2 || form == 3 || form == 7);
    if (wide) {
        return WideSlot(modifiers);
    }
    if (which == 1 && form == 1) {
        operand.reg = Take(32, reg_bits);
        if (modifiers) {
            operand.abs = Take(62, 1) != 0;
            operand.negate = Take(63, 1) != 0;
        }
    } else {
        operand.reg = Take(64, reg_bits);
        if (modifiers) {
            operand.abs = Take(74, 1) != 0;
            operand.negate = Take(75, 1) != 0;
        }
    }
    return operand;
}

IR::U32 Visitor::Value(const Operand& operand) {
    switch (operand.kind) {
    case Operand::Kind::Reg:
        return X(operand.reg);
    case Operand::Kind::Ureg:
        return UR(operand.reg);
    case Operand::Kind::Imm:
        return ir.Imm32(operand.imm);
    case Operand::Kind::Cbuf:
        return Cbuf(operand.cbuf_index, operand.cbuf_offset);
    }
    Unsupported("native operand kind");
}

IR::F32 Visitor::FloatValue(const Operand& operand, bool apply_modifiers) {
    IR::F32 value{ir.BitCast<IR::F32>(Value(operand))};
    if (apply_modifiers && (operand.abs || operand.negate)) {
        value = ir.FPAbsNeg(value, operand.abs, operand.negate);
    }
    return value;
}

IR::F64 Visitor::DoubleValue(const Operand& operand) {
    IR::U32 low;
    IR::U32 high;
    switch (operand.kind) {
    case Operand::Kind::Reg:
        Require(operand.reg >= 255 || operand.reg % 2 == 0, "native unaligned 64-bit register pair");
        low = X(operand.reg);
        high = X(operand.reg >= 255 ? 255 : operand.reg + 1);
        break;
    case Operand::Kind::Ureg:
        Require(operand.reg >= 63 || operand.reg % 2 == 0, "native unaligned 64-bit uniform pair");
        low = UR(operand.reg);
        high = UR(operand.reg >= 63 ? 63 : operand.reg + 1);
        break;
    case Operand::Kind::Imm:
        low = ir.Imm32(0U);
        high = ir.Imm32(operand.imm);
        break;
    case Operand::Kind::Cbuf:
        Require(operand.cbuf_offset % 8 == 0, "native unaligned 64-bit constant");
        low = Cbuf(operand.cbuf_index, operand.cbuf_offset);
        high = Cbuf(operand.cbuf_index, operand.cbuf_offset + 4);
        break;
    }
    IR::F64 value{ir.PackDouble2x32(ir.CompositeConstruct(low, high))};
    if (operand.abs || operand.negate) {
        value = IR::F64{ir.FPAbsNeg(value, operand.abs, operand.negate)};
    }
    return value;
}

void Visitor::D(u32 reg, const IR::F64& value) {
    Require(reg >= 255 || reg % 2 == 0, "native unaligned 64-bit destination pair");
    const IR::Value words{ir.UnpackDouble2x32(value)};
    X(reg, IR::U32{ir.CompositeExtract(words, 0)});
    X(reg >= 255 ? 255 : reg + 1, IR::U32{ir.CompositeExtract(words, 1)});
}

void Visitor::Run() {
    const Guard guard = GuardOf(insn);
    // LEPC and the geometry exit stub (0x34e, 0x357, 0x957) used to have no ISA row, so the flow
    // analysis rejected them even behind @!PT. Their rows exist now for nvn-hle geometry programs;
    // reject them here before the never-taken guard is skipped, so every other program still is.
    // LEPC is also taken in compute programs: CUDA kernels compute the return address of a call to
    // an external function from it (LEPC, IADD3, CALL.ABS.NOINC; see CallAbsolute).
    const bool lepc_allowed = op == 0x34e && env.ShaderStage() == Stage::Compute;
    if ((op == 0x34e || op == 0x357 || op == 0x957) && !lepc_allowed &&
        (env.ShaderStage() != Stage::Geometry || !NativeExtensions())) {
        Unsupported(fmt::format("unsupported native opcode {:#x}", op));
    }
    if (guard.Never()) {
        return;
    }
    if (WarpStateBranchOf(insn) == WarpStateBranch::Diverged) {
        return; // BRA.DIV ~URZ: never taken (the warp counts as converged), so it stays in its block
    }
    const OpInfo info = Classify(op);
    uniform_op = info.uniform;
    if (info.pure && !guard.Always()) {
        guard_select = info.uniform ? UPred(guard.index, guard.negated)
                                    : Pred(guard.index, guard.negated);
    }
    // ALU-style encodings are dispatched on their 9 bit base operation.
    if (info.pure && Form() >= 1 && Form() <= 7 && op != 0x202 && op != 0x802 && op != 0xa02 &&
        op != 0xc02 && !(info.uniform && (op == 0x882 || op == 0xc82 || op == 0xab9 || op == 0xabb ||
                                          op == 0x9c3 || op == 0x89c || op == 0x886))) {
        if (info.uniform) {
            Require(Take(91, 1) == 1, "native uniform datapath flag");
        }
        bool handled = true;
        switch (BaseOf(op)) {
        case 0x090: IADD3(); break;
        case 0x091: LEA(); break;
        case 0x092: LOP3(); break;
        case 0x099: SHF(); break;
        case 0x0a4: IMAD(); break;
        case 0x08c: ISETP(); break;
        case 0x087: SEL(false); break;
        case 0x021: FADD(); break;
        case 0x023: FFMA(); break;
        case 0x028: DMUL(); break;
        case 0x029: DADD(); break;
        case 0x02a: DSETP(); break;
        case 0x02b: DFMA(); break;
        case 0x009: FMNMX(); break;
        case 0x007: SEL(false); break;
        case 0x008: SEL(true); break;
        case 0x00b: FSETP(); break;
        case 0x00a: FSET(); break;
        case 0x040: HMNMX2(); break;
        case 0x013: IABS(); break;
        case 0x100: FLO(); break;
        case 0x0bd: FLO(); break;
        case 0x109: POPC(); break;
        case 0x0bf: POPC(); break;
        case 0x01a: SGXT(); break;
        case 0x09a: SGXT(); break;
        case 0x0be: BREV(); break;
        case 0x00c: ISETP(); break;
        case 0x010: IADD3(); break;
        case 0x012: LOP3(); break;
        case 0x019: SHF(); break;
        case 0x011: LEA(); break;
        case 0x017: IMNMX(); break;
        case 0x016: PRMT(); break;
        case 0x096: PRMT(); break;
        case 0x043: F2IP(); break;
        case 0x030: HADD2(); break;
        case 0x031: HFMA2(); break;
        case 0x032: HMUL2(); break;
        case 0x024: IMAD(); break;
        case 0x027: IMAD_HI(); break;
        case 0x025: IMAD_WIDE(); break;
        case 0x0a5: IMAD_WIDE(); break;
        case 0x033: HSET2(); break;
        case 0x034: HSETP2(); break;
        case 0x03e: F2FP(); break;
        case 0x105: F2I(); break;
        case 0x111: F2I(); break; // 64-bit destination
        case 0x106: case 0x045: I2F(); break;
        case 0x112: I2F(); break; // 64-bit source
        case 0x107: FRND(); break;
        case 0x104: F2F(); break;
        default: handled = false; break;
        }
        if (handled) {
            CheckConsumed();
            return;
        }
    }
    switch (op) {
    case 0x202:
    case 0x802:
    case 0xa02:
    case 0xc02:
        MOV();
        break;
    case 0x882:
    case 0xc82:
        UMOV();
        break;
    case 0xab9:
    case 0xabb:
        ULDC();
        break;
    case 0x89c:
        PLOP3();
        break;
    case 0x220:
    case 0x820:
    case 0xa20:
    case 0xc20:
        FMUL();
        break;
    case 0x308:
    case 0xb08:
    case 0xd08:
        MUFU();
        break;
    case 0x326:
    case 0x526:
        IPA();
        break;
    case 0xb60:
    case 0xf60:
    case 0x361:
        TEX();
        break;
    case 0xb66:
    case 0xf66:
    case 0x367:
        TLD();
        break;
    case 0xb6c:
    case 0xf6c:
        TXD();
        break;
    case 0xb69:
    case 0xf69:
    case 0x36a:
        TMML();
        break;
    case 0x822:
        FSWZADD();
        break;
    case 0x301:
        BREV();
        break;
    case 0xfae:
        LDGSTS();
        break;
    case 0x9af:
        ir.WorkgroupMemoryBarrier();
        break;
    case 0x806:
        VOTE();
        break;
    case 0x886:
        VOTEU();
        break;
    case 0xb1d:
        BAR();
        break;
    case 0x9ab: // ERRBAR: kept as a device memory barrier, like the lowering
        ir.DeviceMemoryBarrier();
        break;
    case 0x992:
        MEMBAR();
        break;
    case 0xb6f:
    case 0x370:
        TXQ();
        break;
    case 0x95c: // BPT.TRAP: the thread stops (an assertion path); the flow builder ends it like EXIT
        Trap();
        break;
    case 0x343: // CALL.ABS.NOINC to an external function: the flow builder ends the thread here
        CallAbsolute();
        break;
    case 0xf8c:
        ATOMS_POPC();
        break;
    case 0xf89:
    case 0x389:
    case 0x589:
    case 0x989:
        SHFL();
        break;
    case 0x983:
        LDL();
        break;
    case 0x387:
    case 0x987:
        STL();
        break;
    case 0x226:
        IDP();
        break;
    case 0x38c:
        ATOMS();
        break;
    case 0x38d:
        ATOMS_CAS();
        break;
    case 0x3c4:
        REDUX();
        break;
    case 0x3a1:
        MATCH();
        break;
    case 0x923:
        ISBERD();
        break;
    case 0x927:
        ISBEWR();
        break;
    case 0x3a8:
    case 0x9a8:
    case 0x98e:
        ATOMG();
        break;
    case 0x3a9:
        ATOMG_CAS();
        break;
    case 0x83b:
        LDSM();
        break;
    case 0x23a:
        MOVM();
        break;
    case 0x23c:
        HMMA();
        break;
    case 0x984:
        LDS();
        break;
    case 0x388:
    case 0x988:
        STS();
        break;
    case 0x3c2:
        R2UR();
        break;
    case 0x381:
    case 0x981:
        LDG();
        break;
    case 0x386:
    case 0x986:
        STG();
        break;
    case 0xb9b:
        SUST();
        break;
    case 0xb9d:
        SUST_D();
        break;
    case 0x593:
        SUATOM();
        break;
    case 0x59f:
        SURED();
        break;
    case 0xb97:
        SULD();
        break;
    case 0xb63:
    case 0xf63:
        TLD4();
        break;
    case 0x321:
        ALD();
        break;
    case 0x322:
        AST();
        break;
    case 0x324:
    case 0x924:
        OUT();
        break;
    case 0x34e:
        LEPC();
        break;
    case 0x357:
    case 0x957:
        GeometryExitStub();
        break;
    case 0xb82:
        LDC();
        break;
    case 0x805:
        CS2R();
        break;
    case 0x919:
    case 0x9c3:
        S2R();
        break;
    case 0x94d:
        EXIT();
        break;
    case 0x803:
    case 0x883: // UP2UR
        P2R();
        break;
    case 0x804:
        R2P();
        break;
    case 0x81c: // PLOP3
        PLOP3();
        break;
    case 0x918: // NOP
        break;
    case 0x80d: { // CSMTEST: a test of shading-mode state the IR does not model.
        // Fields (nvdisasm 13.0): immediate 32..59, .R/.A 62..63, .AND/.OR/.XOR 74..75, test
        // 76..78, results 81..83 and 84..86, source predicate 87..90. Kirby and the Forgotten Land
        // S2 has this pair in about 170 vertex programs (title scene, opening movie):
        //   CSMTEST.RA.F.AND PT, PT, 0x4321111, PT    results discarded: no effect
        //   CSMTEST.A.FCSM_TR.AND P0, PT, 0x0, PT     then @P0 EXIT before the outputs are stored
        // The test is taken as false, so the vertex takes the full path and stores its outputs
        // (true would leave every vertex without outputs). Inferred from that code, not from a
        // reference; Eglantine's lowering does the same for Kirby Air Riders.
        Take(32, 28);
        Take(62, 2);
        const u32 combine = Take(74, 2);
        Take(76, 3);
        const u32 first = Take(81, 3);
        const u32 second = Take(84, 3);
        const u32 source = Take(87, 4);
        if (first != 7 || second != 7) {
            // Only the seen form: one result, AND with PT, second result discarded.
            Require(second == 7 && combine == 0 && source == 7, "native CSM test mode");
            SetPred(first, ir.Imm1(false));
        }
        break;
    }
    case 0x942: // BREAK Bx: only drops threads from the barrier; the following BRA leaves the loop
        Take(16, 4);
        Take(87, 4); // the predicate only chooses which threads leave; threads run independently here
        break;
    case 0x948: // WARPSYNC with the full-warp mask: lanes are executed independently
        Require(Take(32, 32) == 0xffffffffU, "native partial warp sync");
        Require(Take(87, 4) == 7, "native predicated warp sync");
        break;
    case 0x348: // WARPSYNC Rn: the mask is a register. Lanes are executed independently and a
                // subgroup operation reads every active lane, so any mask has no effect here, as
                // for the immediate form. CUDA's __cuda_sm70_shflsync_* helpers load -1 into it.
        Take(32, 8);
        Require(Take(87, 4) == 7, "native predicated warp sync");
        break;
    case 0x91a: { // DEPBAR.LE SB5/SB0, n: scoreboards are replaced by IR data dependencies
        // SB0 (field 8) waits for LDGSTS copy groups, which complete synchronously here.
        Take(38, 6);
        const u32 scoreboard = Take(44, 4);
        Require(scoreboard == 0xd || scoreboard == 0x8, "native dependency barrier mode");
        break;
    }
    case 0x95b: // KILL: the discard itself is a block end class
        Require(Take(87, 4) == 7, "native predicated KILL");
        break;
    case 0x944: // CALL: resolved by the control flow builder
        Take(34, 48);
        Require(Take(86, 1) == 1, "native call increment mode");
        Take(87, 4);
        break;
    case 0x950: // RET.REL: the return address register was validated by the control flow analysis
        Take(24, 8);
        Take(34, 48);
        Require(Take(86, 1) == 1, "native return decrement mode");
        Take(87, 4);
        break;
    case 0x958: // BRXU: resolved by the control flow builder
        Take(24, 6);
        Take(34, 48);
        Take(87, 4);
        Take(91, 1);
        break;
    case 0x949: // BRX: resolved by the control flow builder
        Take(24, 8);
        Take(34, 48);
        Take(87, 4);
        break;
    case 0x945: // BSSY / BSYNC are control flow only
    case 0x941:
        Take(16, 4);
        Take(34, 30);
        Take(87, 4);
        break;
    default:
        Unsupported(fmt::format("unsupported native opcode {:#x}", op));
    }
    CheckConsumed();
}

// BPT.TRAP. On hardware a trap faults the channel; nothing after it runs. The flow builder ends the
// thread here like EXIT (Classify: OpKind::Exit), which keeps every side effect past the trap from
// happening. The emulated GPU only lets a trap through for compute programs whose uniform guard it
// proved never taken (VerifyTrapGuards, video_core/shader_environment.cpp); nvn-hle programs get no
// such check, so a reachable trap simply ends the thread.
// A fragment that ends here wrote no outputs (only Visitor::EXIT stores them), so its colours, depth
// and sample mask would be undefined: it is demoted first, which drops the fragment like KILL. That
// path is nvn-hle only; the emulated GPU rejects non-compute trap programs before this front end.
void Visitor::Trap() {
    Require(Take(84, 3) == 3, "native breakpoint mode");
    Ignore(32, 32); // trap code
    Take(87, 4);
    if (env.ShaderStage() != Stage::Fragment) {
        return;
    }
    Require(NativeExtensions(), "native fragment trap");
    ir.DemoteToHelperInvocation();
}

// CALL.ABS.NOINC Rn: a call to an absolute address in Rn:Rn+1, with the return address set up by
// the caller (LEPC + IADD3 into R20:R21 in CUDA's ABI). Every such call in the CUDA kernels of the
// games' upscaling libraries (85 in Fast Fusion's) loads its target from c[0x4][...], the table of
// external functions the CUDA loader fills, and passes a line number and string pointers: it is
// __assertfail, behind a branch the kernel takes only when an assertion fails. The external
// function is not in the program and does not return (the kernel traps), so the thread ends here,
// as for BPT.TRAP (Trap): the flow builder classifies 0x343 as an exit (Classify).
// It is accepted in compute programs only; the uniform trap-guard proof the emulated path runs for
// BPT.TRAP (VerifyTrapGuards) does not apply, since an assertion's guard depends on data.
void Visitor::CallAbsolute() {
    Require(env.ShaderStage() == Stage::Compute, "native absolute call outside compute");
    Take(24, 8); // target register pair
    Require(Take(84, 2) == 0, "native absolute call mode");
    Require(Take(86, 1) == 1, "native absolute call increment mode"); // .NOINC
    Take(87, 4);
}

void Visitor::CheckConsumed() {
    for (u32 word = 0; word < 4; ++word) {
        const u32 stray = insn.words[word] & ~consumed[word];
        if (stray) {
            Unsupported(fmt::format("op {:#x}: bit {} not modelled", op, word * 32 + std::countr_zero(stray)));
        }
    }
}

} // namespace Shader::Ampere
