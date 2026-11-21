// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "dynarmic/ir/block_serialization.h"

#include <optional>
#include <string_view>
#include <unordered_map>

#include "dynarmic/ir/basic_block.h"
#include "dynarmic/ir/cond.h"
#include "dynarmic/ir/location_descriptor.h"
#include "dynarmic/ir/microinstruction.h"
#include "dynarmic/ir/opcodes.h"
#include "dynarmic/ir/terminal.h"
#include "dynarmic/ir/type.h"
#include "dynarmic/ir/value.h"

namespace Dynarmic::IR {

namespace {

// Bump when the record layout changes.
constexpr u64 format_version = 1;

enum class ArgTag : u8 {
    Immediate = 0,  // includes Void (an absent argument)
    Instruction = 1,
};

enum class TermTag : u8 {
    None = 0,
    ReturnToDispatch,
    LinkBlock,
    LinkBlockFast,
    PopRSBHint,
    FastDispatchHint,
    If,
    CheckBit,
    CheckHalt,
};

class Writer {
public:
    std::vector<u8> out;

    void U8(u8 v) {
        out.push_back(v);
    }
    void Var(u64 v) {
        while (v >= 0x80) {
            out.push_back(static_cast<u8>(v | 0x80));
            v >>= 7;
        }
        out.push_back(static_cast<u8>(v));
    }
};

class Reader {
public:
    explicit Reader(std::span<const u8> data)
            : data(data) {}

    std::optional<u8> U8() {
        if (pos >= data.size()) {
            return std::nullopt;
        }
        return data[pos++];
    }
    std::optional<u64> Var() {
        u64 v = 0;
        for (unsigned shift = 0; shift < 64; shift += 7) {
            const auto byte = U8();
            if (!byte) {
                return std::nullopt;
            }
            v |= u64(*byte & 0x7F) << shift;
            if ((*byte & 0x80) == 0) {
                return v;
            }
        }
        return std::nullopt;
    }
    bool AtEnd() const {
        return pos == data.size();
    }

private:
    std::span<const u8> data;
    size_t pos = 0;
};

void WriteLeaf(Writer& w, const Term::LeafTerminal& leaf) {
    std::visit([&](const auto& t) {
        using T = std::decay_t<decltype(t)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            w.U8(u8(TermTag::None));
        } else if constexpr (std::is_same_v<T, Term::ReturnToDispatch>) {
            w.U8(u8(TermTag::ReturnToDispatch));
        } else if constexpr (std::is_same_v<T, Term::LinkBlock>) {
            w.U8(u8(TermTag::LinkBlock));
            w.Var(t.next.Value());
        } else if constexpr (std::is_same_v<T, Term::LinkBlockFast>) {
            w.U8(u8(TermTag::LinkBlockFast));
            w.Var(t.next.Value());
        } else if constexpr (std::is_same_v<T, Term::PopRSBHint>) {
            w.U8(u8(TermTag::PopRSBHint));
        } else if constexpr (std::is_same_v<T, Term::FastDispatchHint>) {
            w.U8(u8(TermTag::FastDispatchHint));
        } else {
            static_assert(!sizeof(T), "unhandled leaf terminal");
        }
    }, leaf);
}

std::optional<Term::LeafTerminal> ReadLeaf(Reader& r) {
    const auto tag = r.U8();
    if (!tag) {
        return std::nullopt;
    }
    switch (TermTag(*tag)) {
    case TermTag::None:
        return Term::LeafTerminal{std::monostate{}};
    case TermTag::ReturnToDispatch:
        return Term::LeafTerminal{Term::ReturnToDispatch{}};
    case TermTag::LinkBlock: {
        const auto next = r.Var();
        if (!next) {
            return std::nullopt;
        }
        return Term::LeafTerminal{Term::LinkBlock{LocationDescriptor{*next}}};
    }
    case TermTag::LinkBlockFast: {
        const auto next = r.Var();
        if (!next) {
            return std::nullopt;
        }
        return Term::LeafTerminal{Term::LinkBlockFast{LocationDescriptor{*next}}};
    }
    case TermTag::PopRSBHint:
        return Term::LeafTerminal{Term::PopRSBHint{}};
    case TermTag::FastDispatchHint:
        return Term::LeafTerminal{Term::FastDispatchHint{}};
    default:
        return std::nullopt;
    }
}

void WriteTerminal(Writer& w, const Term::Terminal& term) {
    std::visit([&](const auto& t) {
        using T = std::decay_t<decltype(t)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
            w.U8(u8(TermTag::None));
        } else if constexpr (std::is_same_v<T, Term::LeafTerminal>) {
            WriteLeaf(w, t);
        } else if constexpr (std::is_same_v<T, Term::If>) {
            w.U8(u8(TermTag::If));
            w.U8(u8(t.if_));
            WriteLeaf(w, t.then_);
            WriteLeaf(w, t.else_);
        } else if constexpr (std::is_same_v<T, Term::CheckBit>) {
            w.U8(u8(TermTag::CheckBit));
            WriteLeaf(w, t.then_);
            WriteLeaf(w, t.else_);
        } else if constexpr (std::is_same_v<T, Term::CheckHalt>) {
            w.U8(u8(TermTag::CheckHalt));
            WriteLeaf(w, t.else_);
        } else {
            static_assert(!sizeof(T), "unhandled terminal");
        }
    }, term);
}

std::optional<Term::Terminal> ReadTerminal(Reader& r) {
    // Peek the tag: leaf tags are read back by ReadLeaf.
    Reader peek = r;
    const auto tag = peek.U8();
    if (!tag) {
        return std::nullopt;
    }
    switch (TermTag(*tag)) {
    case TermTag::None:
        r.U8();
        return Term::Terminal{std::monostate{}};
    case TermTag::If: {
        r.U8();
        const auto cond = r.U8();
        auto then_ = ReadLeaf(r);
        auto else_ = ReadLeaf(r);
        if (!cond || !then_ || !else_) {
            return std::nullopt;
        }
        return Term::Terminal{Term::If{Cond(*cond), std::move(*then_), std::move(*else_)}};
    }
    case TermTag::CheckBit: {
        r.U8();
        auto then_ = ReadLeaf(r);
        auto else_ = ReadLeaf(r);
        if (!then_ || !else_) {
            return std::nullopt;
        }
        return Term::Terminal{Term::CheckBit{std::move(*then_), std::move(*else_)}};
    }
    case TermTag::CheckHalt: {
        r.U8();
        auto else_ = ReadLeaf(r);
        if (!else_) {
            return std::nullopt;
        }
        return Term::Terminal{Term::CheckHalt{std::move(*else_)}};
    }
    default: {
        auto leaf = ReadLeaf(r);
        if (!leaf) {
            return std::nullopt;
        }
        return Term::Terminal{std::move(*leaf)};
    }
    }
}

}  // namespace

std::vector<u8> SerializeBlock(const Block& block) {
    Writer w;
    w.out.reserve(64 + block.instructions.size() * 12);
    w.Var(format_version);
    w.Var(block.Location().Value());
    w.Var(block.EndLocation().Value());
    w.U8(u8(block.GetCondition()));
    w.U8(block.HasConditionFailedLocation() ? 1 : 0);
    if (block.HasConditionFailedLocation()) {
        w.Var(block.ConditionFailedLocation().Value());
    }
    w.Var(block.ConditionFailedCycleCount());
    w.Var(block.CycleCount());
    WriteTerminal(w, block.GetTerminal());

    std::unordered_map<const Inst*, u64> index;
    index.reserve(block.instructions.size());
    w.Var(block.instructions.size());
    for (const Inst& inst : block.instructions) {
        index.emplace(&inst, index.size());
        w.Var(u64(inst.GetOpcode()));
        w.Var(inst.GetName());
        const size_t num_args = inst.NumArgs();
        for (size_t i = 0; i < num_args; ++i) {
            const Value arg = inst.GetArg(i);
            if (arg.GetRawType() == Type::Opaque) {
                // Arguments always refer to earlier instructions of the same block.
                w.U8(u8(ArgTag::Instruction));
                w.Var(index.at(arg.GetInst()));
            } else {
                w.U8(u8(ArgTag::Immediate));
                w.Var(u64(arg.GetRawType()));
                w.Var(arg.GetRawImmediateBits());
            }
        }
    }
    return std::move(w.out);
}

std::optional<u64> PeekBlockLocation(std::span<const u8> data) {
    Reader r{data};
    const auto version = r.Var();
    if (!version || *version != format_version) {
        return std::nullopt;
    }
    return r.Var();
}

bool DeserializeBlock(std::span<const u8> data, Block& block) {
    Reader r{data};
    const auto version = r.Var();
    const auto location = r.Var();
    const auto end_location = r.Var();
    const auto cond = r.U8();
    const auto has_cond_failed = r.U8();
    if (!version || *version != format_version || !location || !end_location || !cond || !has_cond_failed) {
        return false;
    }
    if (LocationDescriptor{*location} != block.Location()) {
        return false;
    }
    block.SetEndLocation(LocationDescriptor{*end_location});
    block.SetCondition(Cond(*cond));
    if (*has_cond_failed) {
        const auto cond_failed = r.Var();
        if (!cond_failed) {
            return false;
        }
        block.SetConditionFailedLocation(LocationDescriptor{*cond_failed});
    }
    const auto cond_failed_cycles = r.Var();
    const auto cycles = r.Var();
    auto terminal = ReadTerminal(r);
    const auto count = r.Var();
    if (!cond_failed_cycles || !cycles || !terminal || !count) {
        return false;
    }
    block.ConditionFailedCycleCount() = static_cast<size_t>(*cond_failed_cycles);
    block.CycleCount() = static_cast<size_t>(*cycles);

    std::vector<Inst*> insts;
    insts.reserve(static_cast<size_t>(*count));
    for (u64 n = 0; n < *count; ++n) {
        const auto opcode = r.Var();
        const auto name = r.Var();
        if (!opcode || !name || *opcode >= OpcodeCount) {
            return false;
        }
        const Opcode op = Opcode(*opcode);
        Inst& inst = *block.AppendNewInst(op, {});
        // AppendNewInst with no arguments leaves them empty; set them now (SetArg records uses and pseudo-ops).
        const size_t num_args = GetNumArgsOf(op);
        for (size_t i = 0; i < num_args; ++i) {
            const auto tag = r.U8();
            if (!tag) {
                return false;
            }
            if (ArgTag(*tag) == ArgTag::Instruction) {
                const auto ref = r.Var();
                if (!ref || *ref >= insts.size()) {
                    return false;
                }
                inst.SetArg(i, Value(insts[static_cast<size_t>(*ref)]));
            } else if (ArgTag(*tag) == ArgTag::Immediate) {
                const auto type = r.Var();
                const auto bits = r.Var();
                if (!type || !bits || Type(*type) == Type::Opaque) {
                    return false;
                }
                inst.SetArg(i, Value::FromRawImmediate(Type(*type), *bits));
            } else {
                return false;
            }
        }
        inst.SetName(static_cast<unsigned>(*name));
        insts.push_back(&inst);
    }
    if (!r.AtEnd()) {
        return false;
    }
    block.SetTerminal(std::move(*terminal));
    return true;
}

u64 HashBytes(const void* data, size_t size, u64 seed) {
    // FNV-1a over the bytes, finished with a splitmix64 avalanche.
    u64 h = 0xcbf29ce484222325ULL ^ seed;
    const auto* p = static_cast<const u8*>(data);
    for (size_t i = 0; i < size; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return HashCombine(h, size);
}

u64 HashCombine(u64 a, u64 b) {
    u64 x = a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

u64 SchemaHash() {
    static const u64 hash = [] {
        u64 h = HashCombine(0, format_version);
        for (size_t i = 0; i < OpcodeCount; ++i) {
            const Opcode op = Opcode(i);
            const std::string_view name = GetNameOf(op);
            h = HashCombine(h, HashBytes(name.data(), name.size()));
            h = HashCombine(h, u64(GetTypeOf(op)));
            const size_t num_args = GetNumArgsOf(op);
            for (size_t a = 0; a < num_args; ++a) {
                h = HashCombine(h, u64(GetArgTypeOf(op, a)));
            }
        }
        return h;
    }();
    return hash;
}

}  // namespace Dynarmic::IR
