/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/frontend/A64/translate/impl/impl.h"

namespace Dynarmic::A64 {

static bool LoadStoreRegisterImmediate(TranslatorVisitor& v, bool wback, bool postindex, size_t scale, u64 offset, Imm<2> size, Imm<2> opc, Reg Rn, Reg Rt) {
    IR::MemOp memop;
    bool signed_ = false;
    size_t regsize = 0;

    if (opc.Bit<1>() == 0) {
        memop = opc.Bit<0>() ? IR::MemOp::LOAD : IR::MemOp::STORE;
        regsize = size == 0b11 ? 64 : 32;
        signed_ = false;
    } else if (size == 0b11) {
        memop = IR::MemOp::PREFETCH;
        ASSERT(!opc.Bit<0>());
    } else {
        memop = IR::MemOp::LOAD;
        ASSERT(!(size == 0b10 && opc.Bit<0>() == 1));
        regsize = opc.Bit<0>() ? 32 : 64;
        signed_ = true;
    }

    if (memop == IR::MemOp::LOAD && wback && Rn == Rt && Rn != Reg::R31) {
        return v.UnpredictableInstruction();
    }
    if (memop == IR::MemOp::STORE && wback && Rn == Rt && Rn != Reg::R31) {
        return v.UnpredictableInstruction();
    }

    // TODO: Check SP alignment
    IR::U64 address = Rn == Reg::SP ? IR::U64(v.SP(64)) : IR::U64(v.X(64, Rn));
    if (!postindex) {
        address = v.ir.Add(address, v.ir.Imm64(offset));
    }

    const size_t datasize = 8 << scale;
    switch (memop) {
    case IR::MemOp::STORE: {
        const auto data = v.X(datasize, Rt);
        v.Mem(address, datasize / 8, IR::AccType::NORMAL, data);
        break;
    }
    case IR::MemOp::LOAD: {
        const auto data = v.Mem(address, datasize / 8, IR::AccType::NORMAL);
        if (signed_) {
            v.X(regsize, Rt, v.SignExtend(data, regsize));
        } else {
            v.X(regsize, Rt, v.ZeroExtend(data, regsize));
        }
        break;
    }
    case IR::MemOp::PREFETCH:
        // Prefetch(address, Rt)
        break;
    }

    if (wback) {
        if (postindex) {
            address = v.ir.Add(address, v.ir.Imm64(offset));
        }

        if (Rn == Reg::SP) {
            v.SP(64, address);
        } else {
            v.X(64, Rn, address);
        }
    }

    return true;
}

bool TranslatorVisitor::STRx_LDRx_imm_1(Imm<2> size, Imm<2> opc, Imm<9> imm9, bool not_postindex, Reg Rn, Reg Rt) {
    const bool wback = true;
    const bool postindex = !not_postindex;
    const size_t scale = size.ZeroExtend<size_t>();
    const u64 offset = imm9.SignExtend<u64>();

    return LoadStoreRegisterImmediate(*this, wback, postindex, scale, offset, size, opc, Rn, Rt);
}

bool TranslatorVisitor::STRx_LDRx_imm_2(Imm<2> size, Imm<2> opc, Imm<12> imm12, Reg Rn, Reg Rt) {
    const bool wback = false;
    const bool postindex = false;
    const size_t scale = size.ZeroExtend<size_t>();
    const u64 offset = imm12.ZeroExtend<u64>() << scale;

    return LoadStoreRegisterImmediate(*this, wback, postindex, scale, offset, size, opc, Rn, Rt);
}

bool TranslatorVisitor::STURx_LDURx(Imm<2> size, Imm<2> opc, Imm<9> imm9, Reg Rn, Reg Rt) {
    const bool wback = false;
    const bool postindex = false;
    const size_t scale = size.ZeroExtend<size_t>();
    const u64 offset = imm9.SignExtend<u64>();

    return LoadStoreRegisterImmediate(*this, wback, postindex, scale, offset, size, opc, Rn, Rt);
}

bool TranslatorVisitor::PRFM_imm([[maybe_unused]] Imm<12> imm12, [[maybe_unused]] Reg Rn, [[maybe_unused]] Reg Rt) {
    // Currently a NOP (which is valid behavior, as indicated by
    // the ARMv8 architecture reference manual)
    return true;
}

bool TranslatorVisitor::PRFM_unscaled_imm([[maybe_unused]] Imm<9> imm9, [[maybe_unused]] Reg Rn, [[maybe_unused]] Reg Rt) {
    // Currently a NOP (which is valid behavior, as indicated by
    // the ARMv8 architecture reference manual)
    return true;
}

static bool LoadStoreSIMD(TranslatorVisitor& v, bool wback, bool postindex, size_t scale, u64 offset, IR::MemOp memop, Reg Rn, Vec Vt) {
    const auto acctype = IR::AccType::VEC;
    const size_t datasize = 8 << scale;

    IR::U64 address;
    if (Rn == Reg::SP) {
        // TODO: Check SP Alignment
        address = v.SP(64);
    } else {
        address = v.X(64, Rn);
    }

    if (!postindex) {
        address = v.ir.Add(address, v.ir.Imm64(offset));
    }

    switch (memop) {
    case IR::MemOp::STORE:
        if (datasize == 128) {
            const IR::U128 data = v.V(128, Vt);
            v.Mem(address, 16, acctype, data);
        } else {
            const IR::UAny data = v.ir.VectorGetElement(datasize, v.V(128, Vt), 0);
            v.Mem(address, datasize / 8, acctype, data);
        }
        break;
    case IR::MemOp::LOAD:
        if (datasize == 128) {
            const IR::U128 data = v.Mem(address, 16, acctype);
            v.V(128, Vt, data);
        } else {
            const IR::UAny data = v.Mem(address, datasize / 8, acctype);
            v.V(128, Vt, v.ir.ZeroExtendToQuad(data));
        }
        break;
    default:
        UNREACHABLE();
    }

    if (wback) {
        if (postindex) {
            address = v.ir.Add(address, v.ir.Imm64(offset));
        }

        if (Rn == Reg::SP) {
            v.SP(64, address);
        } else {
            v.X(64, Rn, address);
        }
    }

    return true;
}

bool TranslatorVisitor::STR_imm_fpsimd_1(Imm<2> size, Imm<1> opc_1, Imm<9> imm9, bool not_postindex, Reg Rn, Vec Vt) {
    const size_t scale = concatenate(opc_1, size).ZeroExtend<size_t>();
    if (scale > 4) {
        return UnallocatedEncoding();
    }

    const bool wback = true;
    const bool postindex = !not_postindex;
    const u64 offset = imm9.SignExtend<u64>();

    return LoadStoreSIMD(*this, wback, postindex, scale, offset, IR::MemOp::STORE, Rn, Vt);
}

bool TranslatorVisitor::STR_imm_fpsimd_2(Imm<2> size, Imm<1> opc_1, Imm<12> imm12, Reg Rn, Vec Vt) {
    const size_t scale = concatenate(opc_1, size).ZeroExtend<size_t>();
    if (scale > 4) {
        return UnallocatedEncoding();
    }

    const bool wback = false;
    const bool postindex = false;
    const u64 offset = imm12.ZeroExtend<u64>() << scale;

    return LoadStoreSIMD(*this, wback, postindex, scale, offset, IR::MemOp::STORE, Rn, Vt);
}

bool TranslatorVisitor::LDR_imm_fpsimd_1(Imm<2> size, Imm<1> opc_1, Imm<9> imm9, bool not_postindex, Reg Rn, Vec Vt) {
    const size_t scale = concatenate(opc_1, size).ZeroExtend<size_t>();
    if (scale > 4) {
        return UnallocatedEncoding();
    }

    const bool wback = true;
    const bool postindex = !not_postindex;
    const u64 offset = imm9.SignExtend<u64>();

    return LoadStoreSIMD(*this, wback, postindex, scale, offset, IR::MemOp::LOAD, Rn, Vt);
}

bool TranslatorVisitor::LDR_imm_fpsimd_2(Imm<2> size, Imm<1> opc_1, Imm<12> imm12, Reg Rn, Vec Vt) {
    const size_t scale = concatenate(opc_1, size).ZeroExtend<size_t>();
    if (scale > 4) {
        return UnallocatedEncoding();
    }

    const bool wback = false;
    const bool postindex = false;
    const u64 offset = imm12.ZeroExtend<u64>() << scale;

    return LoadStoreSIMD(*this, wback, postindex, scale, offset, IR::MemOp::LOAD, Rn, Vt);
}

bool TranslatorVisitor::STUR_fpsimd(Imm<2> size, Imm<1> opc_1, Imm<9> imm9, Reg Rn, Vec Vt) {
    const size_t scale = concatenate(opc_1, size).ZeroExtend<size_t>();
    if (scale > 4) {
        return UnallocatedEncoding();
    }

    const bool wback = false;
    const bool postindex = false;
    const u64 offset = imm9.SignExtend<u64>();

    return LoadStoreSIMD(*this, wback, postindex, scale, offset, IR::MemOp::STORE, Rn, Vt);
}

bool TranslatorVisitor::LDUR_fpsimd(Imm<2> size, Imm<1> opc_1, Imm<9> imm9, Reg Rn, Vec Vt) {
    const size_t scale = concatenate(opc_1, size).ZeroExtend<size_t>();
    if (scale > 4) {
        return UnallocatedEncoding();
    }

    const bool wback = false;
    const bool postindex = false;
    const u64 offset = imm9.SignExtend<u64>();

    return LoadStoreSIMD(*this, wback, postindex, scale, offset, IR::MemOp::LOAD, Rn, Vt);
}

// LDRAA, LDRAB: with SCTLR.EnDA/EnDB clear, AUTDA/AUTDB leave the base unchanged, so this is a 64-bit load
// at base + (S:imm9 << 3), with optional pre-index writeback.
bool TranslatorVisitor::LDRA([[maybe_unused]] bool M, bool S, Imm<9> imm9, bool W, Reg Rn, Reg Rt) {
    if (!options.enable_pauth_returns) {
        return UnallocatedEncoding();
    }
    if (W && Rn == Rt && Rn != Reg::SP && !options.define_unpredictable_behaviour) {
        return UnpredictableInstruction();
    }
    const s64 offset = concatenate(Imm<1>{S}, imm9).SignExtend<s64>() << 3;
    const IR::U64 base = Rn == Reg::SP ? IR::U64{SP(64)} : IR::U64{X(64, Rn)};
    const IR::U64 address = ir.Add(base, ir.Imm64(static_cast<u64>(offset)));
    const IR::U64 data{Mem(address, 8, IR::AccType::NORMAL)};
    if (W) {
        if (Rn == Reg::SP) {
            SP(64, address);
        } else {
            X(64, Rn, address);
        }
    }
    X(64, Rt, data);
    return true;
}

}  // namespace Dynarmic::A64
