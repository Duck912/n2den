// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2016 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>

#include "common/common_types.h"
#include "common/x64/xbyak.h"
#include "dynarmic/mcl/bit.hpp"
#include "dynarmic/backend/x64/xbyak.h"
#include "dynarmic/backend/x64/abi.h"
#include "dynarmic/backend/x64/callback.h"
#include "dynarmic/backend/x64/constant_pool.h"
#include "dynarmic/backend/x64/host_feature.h"
#include "dynarmic/backend/x64/jitstate_info.h"
#include "dynarmic/backend/x64/native_cache.h"
#include "dynarmic/common/cast_util.h"
#include "dynarmic/interface/halt_reason.h"
#include "dynarmic/ir/cond.h"

namespace Dynarmic::Backend::X64 {

using CodePtr = const void*;

struct RunCodeCallbacks {
    ArgCallback LookupBlock;
    ArgCallback AddTicks;
    ArgCallback GetTicksRemaining;
    bool enable_cycle_counting;
};

class BlockOfCode final : public Xbyak::CodeGenerator {
public:
    BlockOfCode(RunCodeCallbacks cb, JitStateInfo jsi, size_t total_code_size, std::function<void(BlockOfCode&)> rcp);
    BlockOfCode(const BlockOfCode&) = delete;

    /// Call when external emitters have finished emitting their preludes.
    void PreludeComplete();

    /// Change permissions to RW. This is required to support systems with W^X enforced.
    void EnableWriting();
    /// Change permissions to RX. This is required to support systems with W^X enforced.
    void DisableWriting();

    /// Clears this block of code and resets code pointer to beginning.
    void ClearCache();
    /// Calculates how much space is remaining to use.
    size_t SpaceRemaining() const;
    /// Ensure at least codesize bytes of code cache memory are committed at the current code_ptr.
    void EnsureMemoryCommitted(size_t codesize);

    /// Runs emulated code from code_ptr.
    HaltReason RunCode(void* jit_state, CodePtr code_ptr) const;
    /// Runs emulated code from code_ptr for a single cycle.
    HaltReason StepCode(void* jit_state, CodePtr code_ptr) const;
    /// Code emitter: Returns to dispatcher
    void ReturnFromRunCode(bool mxcsr_already_exited = false);
    /// Code emitter: Returns to dispatcher, forces return to host
    void ForceReturnFromRunCode(bool mxcsr_already_exited = false);
    /// Code emitter: Makes guest MXCSR the current MXCSR
    void SwitchMxcsrOnEntry();
    /// Code emitter: Makes saved host MXCSR the current MXCSR
    void SwitchMxcsrOnExit();
    /// Code emitter: Enter standard ASIMD MXCSR region
    void EnterStandardASIMD();
    /// Code emitter: Leave standard ASIMD MXCSR region
    void LeaveStandardASIMD();
    /// Code emitter: Updates cycles remaining my calling cb.AddTicks and cb.GetTicksRemaining
    /// @note this clobbers ABI caller-save registers
    void UpdateTicks();
    /// Code emitter: Performs a block lookup based on current state
    /// @note this clobbers ABI caller-save registers
    void LookupBlock();

    /// Code emitter: Load required flags for conditional cond from rax into host rflags
    void LoadRequiredFlagsForCondFromRax(IR::Cond cond);

    /// @brief Code emitter: Calls the function
    template<typename F>
        requires std::is_pointer_v<F> && std::is_function_v<std::remove_pointer_t<F>>
    void CallFunction(F fn) {
        if (native_recording) {
            // Fixed length, independent of the distance to the function.
            MovHostPointer(ABI_RETURN, reinterpret_cast<const void*>(fn));
            call(ABI_RETURN);
            return;
        }
        ::Common::X64::CallFarFunction(*this, fn);
    }

    /// @brief Code emitter: Calls the lambda. Lambda must not have any captures.
    template<typename Lambda>
    void CallLambda(Lambda l) {
        CallFunction(Common::FptrCast(l));
    }

    // ---- Relocatable code for the persistent block cache (native_cache.h) ----

    /// Starts recording the block that begins at the current position.
    void BeginNativeRecording();
    /// Emits the block's constants; call after the last instruction of the block.
    void EmitNativeConstants();
    /// Ends recording and returns what was recorded (bytes are taken by the caller).
    NativeRecording EndNativeRecording();
    bool IsRecordingNative() const noexcept {
        return native_recording.has_value();
    }
    void MarkNotRelocatable(const char* reason);
    /// Loads a host address into reg, recorded as a relocation while recording.
    void MovHostPointer(const Xbyak::Reg64& reg, const void* pointer);
    /// Loads a JIT object's address into reg, recorded as symbol `symbol` while recording.
    void MovSymbol(const Xbyak::Reg64& reg, u32 symbol, const volatile void* pointer);
    void RecordNativePatchSite(u32 kind, u64 target);
    void RecordNativeFastmem(const void* rip, const void* resume_rip, const void* callback, u32 inst_name, bool recompile, u64 location);
    /// Maps a value embedded by an ArgCallback (an object pointer) to a symbol, if it is one.
    void SetNativeSymbolLookup(std::function<std::optional<u32>(u64)> lookup) {
        native_symbol_lookup = std::move(lookup);
    }
    /// Start of the code space: prelude routines are addressed from here.
    const u8* CodeSpaceStart() const {
        return static_cast<const u8*>(getCode());
    }
    size_t PreludeSize() const {
        return static_cast<const u8*>(code_begin) - CodeSpaceStart();
    }

    // Branches to absolute addresses and immediate moves, recorded while recording.
    using Xbyak::CodeGenerator::call;
    using Xbyak::CodeGenerator::jmp;
    using Xbyak::CodeGenerator::mov;
    void jmp(const void* addr, LabelType type = T_AUTO) {
        Xbyak::CodeGenerator::jmp(addr, native_recording ? T_NEAR : type);
        RecordNativeBranch(addr);
    }
    void call(const void* addr) {
        Xbyak::CodeGenerator::call(addr);
        RecordNativeBranch(addr);
    }
    void mov(const Xbyak::Operand& op, uint64_t imm) {
        if (native_recording && op.isREG()) {
            CheckNativeImmediate(imm);
        }
        Xbyak::CodeGenerator::mov(op, imm);
    }
#define DYNARMIC_NATIVE_JCC(name)                    using Xbyak::CodeGenerator::name;                void name(const void* addr) {                        Xbyak::CodeGenerator::name(addr);                RecordNativeBranch(addr);                    }
    DYNARMIC_NATIVE_JCC(ja) DYNARMIC_NATIVE_JCC(jae) DYNARMIC_NATIVE_JCC(jb) DYNARMIC_NATIVE_JCC(jbe)
    DYNARMIC_NATIVE_JCC(jc) DYNARMIC_NATIVE_JCC(je) DYNARMIC_NATIVE_JCC(jg) DYNARMIC_NATIVE_JCC(jge)
    DYNARMIC_NATIVE_JCC(jl) DYNARMIC_NATIVE_JCC(jle) DYNARMIC_NATIVE_JCC(jna) DYNARMIC_NATIVE_JCC(jnae)
    DYNARMIC_NATIVE_JCC(jnb) DYNARMIC_NATIVE_JCC(jnbe) DYNARMIC_NATIVE_JCC(jnc) DYNARMIC_NATIVE_JCC(jne)
    DYNARMIC_NATIVE_JCC(jng) DYNARMIC_NATIVE_JCC(jnge) DYNARMIC_NATIVE_JCC(jnl) DYNARMIC_NATIVE_JCC(jnle)
    DYNARMIC_NATIVE_JCC(jno) DYNARMIC_NATIVE_JCC(jnp) DYNARMIC_NATIVE_JCC(jns) DYNARMIC_NATIVE_JCC(jnz)
    DYNARMIC_NATIVE_JCC(jo) DYNARMIC_NATIVE_JCC(jp) DYNARMIC_NATIVE_JCC(jpe) DYNARMIC_NATIVE_JCC(jpo)
    DYNARMIC_NATIVE_JCC(js) DYNARMIC_NATIVE_JCC(jz)
#undef DYNARMIC_NATIVE_JCC

    void ZeroExtendFrom(size_t bitsize, Xbyak::Reg64 reg) {
        switch (bitsize) {
        case 8:
            movzx(reg.cvt32(), reg.cvt8());
            return;
        case 16:
            movzx(reg.cvt32(), reg.cvt16());
            return;
        case 32:
            mov(reg.cvt32(), reg.cvt32());
            return;
        case 64:
            return;
        default:
            UNREACHABLE();
        }
    }

    Xbyak::Address Const(const Xbyak::AddressFrame& frame, u64 lower, u64 upper = 0);

    template<size_t esize>
    Xbyak::Address BConst(const Xbyak::AddressFrame& frame, u64 value) {
        return Const(frame, mcl::bit::replicate_element<u64>(esize, value),
                     mcl::bit::replicate_element<u64>(esize, value));
    }

    CodePtr GetCodeBegin() const;
    size_t GetTotalCodeSize() const;

    const void* GetReturnFromRunCodeAddress() const {
        return return_from_run_code[0];
    }

    const void* GetForceReturnFromRunCodeAddress() const {
        return return_from_run_code[FORCE_RETURN];
    }

    void int3() { db(0xCC); }

    /// Allocate memory of `size` bytes from the same block of memory the code is in.
    /// This is useful for objects that need to be placed close to or within code.
    /// The lifetime of this memory is the same as the code around it.
    void* AllocateFromCodeSpace(size_t size);

    void SetCodePtr(CodePtr code_ptr);
    void EnsurePatchLocationSize(CodePtr begin, size_t size);

    static const Xbyak::Reg64 ABI_JIT_PTR;
    // ABI registers
#ifdef _WIN32
    static const Xbyak::Reg64 ABI_RETURN;
    static const Xbyak::Reg64 ABI_PARAM1;
    static const Xbyak::Reg64 ABI_PARAM2;
    static const Xbyak::Reg64 ABI_PARAM3;
    static const Xbyak::Reg64 ABI_PARAM4;
    static const std::array<Xbyak::Reg64, ABI_PARAM_COUNT> ABI_PARAMS;
#else
    static const Xbyak::Reg64 ABI_RETURN;
    static const Xbyak::Reg64 ABI_RETURN2;
    static const Xbyak::Reg64 ABI_PARAM1;
    static const Xbyak::Reg64 ABI_PARAM2;
    static const Xbyak::Reg64 ABI_PARAM3;
    static const Xbyak::Reg64 ABI_PARAM4;
    static const Xbyak::Reg64 ABI_PARAM5;
    static const Xbyak::Reg64 ABI_PARAM6;
    static const std::array<Xbyak::Reg64, ABI_PARAM_COUNT> ABI_PARAMS;
#endif

    JitStateInfo GetJitStateInfo() const { return jsi; }

    bool HasHostFeature(HostFeature feature) const noexcept;
    /// All host CPU features in use, as bits.
    u64 HostFeatureBits() const noexcept;

private:
    using RunCodeFuncType = HaltReason (*)(void*, CodePtr);
    static constexpr size_t MXCSR_ALREADY_EXITED = 1 << 0;
    static constexpr size_t FORCE_RETURN = 1 << 1;

    ConstantPool constant_pool;
    JitStateInfo jsi;
    std::array<const void*, 4> return_from_run_code;
    RunCodeFuncType run_code = nullptr;
    RunCodeFuncType step_code = nullptr;
    RunCodeCallbacks cb;
    CodePtr code_begin = nullptr;
#ifdef _WIN32
    size_t committed_size = 0;
#endif
    bool prelude_complete = false;

    std::optional<NativeRecording> native_recording;
    std::function<std::optional<u32>(u64)> native_symbol_lookup;
    void RecordNativeBranch(const void* target);
    void CheckNativeImmediate(u64 imm);
    void EmitMovImm64(const Xbyak::Reg64& reg, u64 value);
    bool InNativePatchSite(u32 offset) const;
    u32 NativeOffset(const u8* p) const {
        return static_cast<u32>(p - native_recording->begin);
    }

    void GenRunCode(std::function<void(BlockOfCode&)> rcp);
};

}  // namespace Dynarmic::Backend::X64
