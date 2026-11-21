// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <optional>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/frontend/ampere/classify.h"
#include "shader_recompiler/frontend/ampere/instruction.h"

namespace Shader::Ampere {
class ProgramCaptureState {
public:
    enum class Result { Continue, Complete, InvalidTarget };
    static constexpr u32 MaximumInstructions = 65536;

    Result Observe(const Instruction& instruction, u32 index) {
        const u32 opcode = instruction.Bits(0, 12);
        if (opcode == 0x944 || opcode == 0x947 || opcode == 0x945) {
            // BRA.DIV / BRA.CONV (bit 91 and a uniform register) keep their mode in the low two bits of
            // the word offset (2 = DIV, 3 = CONV); they are not part of the displacement. Only CONV was
            // masked before, so every BRA.DIV (the DLSS library's attention layers) read as a misaligned
            // target and failed the capture. Other branches keep the check on those bits.
            const bool warp_state = opcode == 0x947 && (instruction.words[2] & ~0x3ffffU) == 0x0b800000U &&
                                    (instruction.words[1] & 3) >= 2;
            const u32 low = warp_state ? instruction.words[1] & ~3U : instruction.words[1];
            const u64 raw = low | (u64{instruction.words[2] & 0x3ffff} << 32);
            const s64 relative = opcode == 0x945 ? static_cast<s32>(instruction.words[1])
                                                : static_cast<s64>(raw << 14) >> 14;
            const s64 target = s64{index} * 16 + 16 + relative;
            if (target < 0 || target % 16 || target >= MaximumInstructions * 16)
                return Result::InvalidTarget;
            if (opcode == 0x944 && instruction.Bits(12, 4) == 7) {
                last_call_target = std::max(last_call_target, static_cast<u32>(target / 16));
            } else {
                last_forward_target = std::max(last_forward_target, static_cast<u32>(target / 16));
            }
        }
        if (opcode == 0x94d && instruction.Bits(12, 4) == 7 && instruction.Bits(87, 4) == 7 &&
            !instruction.Bits(84, 2) && index >= last_forward_target) {
            if (!last_call_target) return Result::Complete;
            main_exited = true;
        }
        if (opcode == 0x950 && instruction.Bits(12, 4) == 7 && main_exited &&
            index >= last_call_target && index >= last_forward_target)
            return Result::Complete;
        return Result::Continue;
    }

private:
    u32 last_call_target{};
    u32 last_forward_target{};
    bool main_exited{};
};

// A BRX selects among its blocks through a constant-buffer table, so the furthest block can lie past the first
// EXIT that ends the linear capture. The capture is extended through it and the per-BRX target lists are
// reported to the environment. The tables are neither validated nor rejected (entries that are not source
// addresses are ignored) and the translators still decide what is acceptable.
struct CaptureIndirectBranch {
    u32 index;       // instruction index of the BRX / BRXU
    u32 cbuf_index;  // constant bank holding the table
    u32 cbuf_offset; // byte offset of the first entry
    u32 entries;
};

struct CaptureIndirectTable {
    u32 branch{};
    std::vector<u32> values;  // raw table words
    std::vector<u32> targets; // sorted distinct instruction indices inside the capture
};

struct CaptureIndirectResult {
    bool extended{};
    std::vector<CaptureIndirectTable> tables;
};

inline s64 CaptureRelativeOffset48(const Instruction& instruction) {
    const u64 raw = u64{instruction.words[1] & ~3U} | (u64{instruction.words[2] & 0x3ffffU} << 32);
    return static_cast<s64>(raw << 14) >> 14;
}

// True when `instruction` can write uniform register `reg`: the uniform datapath instructions with a register
// destination and the ordinary ones that write the uniform file. It is only used to give up on a chain, so an
// instruction wrongly taken for a writer just leaves a branch untracked.
inline bool CaptureMayWriteUniformRegister(const Instruction& instruction, u32 reg) {
    const u32 opcode = instruction.Bits(0, 12);
    const u32 base = BaseOf(opcode);
    if (Classify(opcode).uniform) {
        if (base == 0x08c || opcode == 0x89c) // UISETP and UPLOP3 write uniform predicates only
            return false;
    } else if (opcode != 0x3c2 && opcode != 0x3c4 && opcode != 0x883 && opcode != 0x896) {
        return false; // not R2UR, REDUX, UP2UR or UPRMT
    }
    const u32 destination = instruction.Bits(16, 6);
    u32 count = 1;
    if (opcode == 0xab9 || opcode == 0xabb) { // ULDC: 64 and 128 bit loads fill 2 and 4 registers
        const u32 width = instruction.Bits(73, 3);
        count = width == 5 ? 2 : width >= 6 ? 4 : 1;
    } else if (base == 0x0a5) { // UIMAD.WIDE writes a pair
        count = 2;
    }
    return reg >= destination && reg - destination < count;
}

// Recognises, for every self-relative computed branch, the selector sequence of its jump table: a value
// clamped to [0, N] by an unsigned minimum with an immediate, scaled by 4, loaded from a constant bank and
// branched on. Three spellings are known, each matched on its exact encoding (unguarded, no modifiers):
//
//            BRX Rb (0x949)                          BRXU URb (0x958)
//   load     LDC Rb, c[bank][Rs + offset]            ULDC URb, c[bank][URs + offset]
//   scale    SHF.L.U32 Rs, Rc, 0x2, RZ               USHF.L.U32 URs, URc, 0x2, URZ
//         or IMAD[.SHL.U32] Rs, Rc, 0x4, RZ
//   clamp    IMNMX.U32 Rc, Rx, N, PT                 UISETP.LT.U32.AND UPn, UPT, URx, N, UPT
//                                                    USEL URc, URx, N, UPn (the next instruction)
//
// The table has N + 1 entries starting at c[bank][offset]. Each step is the nearest earlier instruction of
// that spelling writing the register. On the uniform side the nearest earlier writer of the register of any
// kind must be that instruction, so a selector that is modified in between is not tracked. A branch that does
// not match is simply not tracked (the direct front end then refuses the program).
inline std::vector<CaptureIndirectBranch> TrackCaptureIndirectBranches(std::span<const Instruction> code) {
    std::vector<CaptureIndirectBranch> result;
    const auto unguarded = [](const Instruction& instruction, u32 opcode) {
        return instruction.Bits(0, 12) == opcode && instruction.Bits(12, 4) == 7;
    };
    // Bank and byte offset of an indexed 32-bit load: only the bank and a word-aligned offset may be set.
    const auto table_address = [](const Instruction& load) {
        return !(load.words[1] & ~((0x1fU << 22) | (0xffffU << 6))) && !((load.words[1] >> 6) & 3);
    };
    // Nearest instruction before `end` accepted by `match`.
    const auto nearest = [&](u32 end, auto&& match) -> std::optional<u32> {
        for (u32 index = end; index-- > 0;)
            if (match(code[index]))
                return index;
        return std::nullopt;
    };
    // Nearest instruction before `end` that can write uniform register `reg`.
    const auto uniform_writer = [&](u32 end, u32 reg) {
        return nearest(end, [&](const Instruction& instruction) {
            return CaptureMayWriteUniformRegister(instruction, reg);
        });
    };
    for (u32 branch_index = 0; branch_index < code.size(); ++branch_index) {
        const auto& branch = code[branch_index];
        const u32 branch_opcode = branch.Bits(0, 12);
        if (branch_opcode != 0x949 && branch_opcode != 0x958)
            continue;
        const bool uniform = branch_opcode == 0x958;
        if (branch.Bits(12, 4) != 7 || (branch.words[2] & ~0x3ffffU) != (uniform ? 0x0b800000U : 0x03800000U) ||
            s64{branch_index} * sizeof(Instruction) + sizeof(Instruction) + CaptureRelativeOffset48(branch) != 0)
            continue;
        const u32 branch_source = branch.Bits(24, 8);
        if (uniform) {
            if (branch_source >= 63)
                continue;
            // ULDC URb, c[bank][URs + offset], 32 bit
            const auto load_index = uniform_writer(branch_index, branch_source);
            if (!load_index)
                continue;
            const auto& load = code[*load_index];
            const u32 selector = load.Bits(24, 8);
            if (!unguarded(load, 0xabb) || load.Bits(16, 8) != branch_source || load.words[2] != 0x08000800 ||
                !table_address(load) || selector >= 63)
                continue;
            // USHF.L.U32 URs, URc, 0x2, URZ
            const auto shift_index = uniform_writer(*load_index, selector);
            if (!shift_index)
                continue;
            const auto& shift = code[*shift_index];
            const u32 bounded_selector = shift.Bits(24, 8);
            if (!unguarded(shift, 0x899) || shift.Bits(16, 8) != selector || shift.words[1] != 2 ||
                shift.words[2] != 0x0800063f || bounded_selector >= 63)
                continue;
            // UISETP.LT.U32.AND UPn, UPT, URx, N, UPT ; USEL URc, URx, N, UPn: the unsigned minimum of URx and N
            const auto select_index = uniform_writer(*shift_index, bounded_selector);
            if (!select_index || *select_index == 0)
                continue;
            const auto& select = code[*select_index];
            const auto& compare = code[*select_index - 1];
            const u32 condition = select.Bits(87, 3);
            const u32 maximum = select.words[1];
            if (!unguarded(select, 0x887) || select.Bits(16, 8) != bounded_selector || select.Bits(24, 8) >= 63 ||
                condition == 7 || maximum < 1 || maximum > 15 || select.words[2] != (0x08000000U | (condition << 23)) ||
                compare.words[0] != (0x0000788cU | (select.Bits(24, 8) << 24)) || compare.words[1] != maximum ||
                compare.words[2] != (0x0bf01070U | (condition << 17)))
                continue;
            result.push_back({.index = branch_index,
                              .cbuf_index = (load.words[1] >> 22) & 0x1f,
                              .cbuf_offset = (load.words[1] >> 6) & 0xffff,
                              .entries = maximum + 1});
            continue;
        }
        // LDC Rb, c[bank][Rs + offset], 32 bit
        const auto load_index = nearest(branch_index, [&](const Instruction& instruction) {
            return unguarded(instruction, 0xb82) && instruction.Bits(16, 8) == branch_source &&
                   instruction.words[2] == 0x00000800 && table_address(instruction);
        });
        if (!load_index)
            continue;
        const auto& load = code[*load_index];
        const u32 selector = load.Bits(24, 8);
        // SHF.L.U32 Rs, Rc, 0x2, RZ, or the same value as a multiply: IMAD Rs, Rc, 0x4, RZ (no carry in or
        // out; bit 73, the signedness, only matters to the high and wide products).
        const auto shift_index = nearest(*load_index, [&](const Instruction& instruction) {
            if (instruction.Bits(16, 8) != selector)
                return false;
            if (unguarded(instruction, 0x819))
                return instruction.words[1] == 2 && instruction.words[2] == 0x000006ff;
            return unguarded(instruction, 0x824) && instruction.words[1] == 4 &&
                   (instruction.words[2] & ~0x200U) == 0x078e00ff;
        });
        if (!shift_index)
            continue;
        const u32 bounded_selector = code[*shift_index].Bits(24, 8);
        // IMNMX.U32 Rc, Rx, N, PT
        const auto clamp_index = nearest(*shift_index, [&](const Instruction& instruction) {
            return unguarded(instruction, 0x817) && instruction.Bits(16, 8) == bounded_selector &&
                   instruction.words[1] >= 1 && instruction.words[1] <= 15 && instruction.words[2] == 0x03800000;
        });
        if (!clamp_index)
            continue;
        result.push_back({.index = branch_index,
                          .cbuf_index = (load.words[1] >> 22) & 0x1f,
                          .cbuf_offset = (load.words[1] >> 6) & 0xffff,
                          .entries = code[*clamp_index].words[1] + 1});
    }
    return result;
}

// Reads the table of every tracked BRX in `native`, extends `native` (already captured through its first
// terminating EXIT) through the furthest in-range target and returns the per-BRX target indices.
// Table values are byte offsets from instruction 0, with no program header in front: a tracked BRX is
// self-relative (its displacement cancels the address of the next instruction), so it jumps to exactly the
// word it loaded. The lowering maps the words the same way (NativeProgramAddress). When there is no table,
// every target is already inside the capture, or the tail cannot be read to a terminating EXIT, `native` and
// `capture` are left untouched, so shaders that capture correctly today are not affected.
template <typename ReadInstruction, typename ReadCbuf>
CaptureIndirectResult ResolveCaptureIndirectBranches(std::vector<Instruction>& native, ProgramCaptureState& capture,
                                                     ReadInstruction&& read_instruction, ReadCbuf&& read_cbuf) {
    const auto read_tables = [&] {
        std::vector<CaptureIndirectTable> tables;
        for (const auto& branch : TrackCaptureIndirectBranches(native)) {
            auto& table = tables.emplace_back();
            table.branch = branch.index;
            for (u32 entry = 0; entry < branch.entries; ++entry)
                table.values.push_back(read_cbuf(branch.cbuf_index, branch.cbuf_offset + entry * 4));
        }
        return tables;
    };
    CaptureIndirectResult result;
    result.tables = read_tables();
    if (result.tables.empty())
        return result;
    const auto entry_index = [&](u32 value) -> std::optional<u32> {
        if (!value || value % sizeof(Instruction))
            return {};
        const u32 index = value / sizeof(Instruction);
        if (index >= ProgramCaptureState::MaximumInstructions)
            return {};
        return index;
    };
    u32 furthest = 0;
    for (const auto& table : result.tables)
        for (const u32 value : table.values)
            if (const auto index = entry_index(value))
                furthest = std::max(furthest, *index);
    if (furthest >= native.size()) {
        const auto saved_native = native;
        const auto saved_capture = capture;
        bool complete = false;
        for (u32 index = static_cast<u32>(native.size()); index < ProgramCaptureState::MaximumInstructions;
             ++index) {
            const std::optional<Instruction> instruction = read_instruction(index);
            if (!instruction)
                break;
            native.push_back(*instruction);
            const auto status = capture.Observe(*instruction, index);
            if (status == ProgramCaptureState::Result::InvalidTarget)
                break;
            if (status == ProgramCaptureState::Result::Complete && native.size() > furthest) {
                complete = true;
                break;
            }
        }
        if (complete) {
            result.extended = true;
            result.tables = read_tables();
        } else {
            native = saved_native;
            capture = saved_capture;
        }
    }
    for (auto& table : result.tables) {
        for (const u32 value : table.values)
            if (const auto index = entry_index(value); index && *index < native.size())
                table.targets.push_back(*index);
        std::ranges::sort(table.targets);
        table.targets.erase(std::unique(table.targets.begin(), table.targets.end()), table.targets.end());
    }
    return result;
}
} // namespace Shader::Ampere
