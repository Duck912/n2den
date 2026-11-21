// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/arm/dynarmic/jit_code_discovery.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <dynarmic/frontend/A64/a64_location_descriptor.h>
#include <fmt/format.h>

#include "common/cityhash.h"
#include "common/logging.h"
#include "common/thread.h"
#include "core/arm/debug.h"
#include "core/arm/dynarmic/arm_dynarmic_64.h"
#include "core/arm/dynarmic/jit_prebuild_list.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/nvn/bridge.h"
#include "core/hle/nvn/thunk_encoding.h"
#include "core/memory.h"

namespace Core {

namespace {

using Clock = std::chrono::steady_clock;

u64 Descriptor(u64 pc, u32 fpcr) {
    return Dynarmic::A64::LocationDescriptor{pc, Dynarmic::FP::FPCR{fpcr}, false}.UniqueHash();
}
u64 DescriptorPc(u64 descriptor) {
    return Dynarmic::A64::LocationDescriptor{Dynarmic::IR::LocationDescriptor{descriptor}}.PC();
}
u32 DescriptorFpcr(u64 descriptor) {
    return Dynarmic::A64::LocationDescriptor{Dynarmic::IR::LocationDescriptor{descriptor}}.FPCR().Value();
}

template <typename T>
T GetLe(const u8* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

/// Bounds-checked reads from a module's bytes (module offsets).
struct ModuleView {
    const JitCodeImage::Module& m;
    bool ok = true;

    bool In(s64 off, size_t n) const {
        return off >= 0 && static_cast<u64>(off) + n <= m.bytes.size();
    }
    template <typename T>
    T Get(s64 off) {
        if (!In(off, sizeof(T))) {
            ok = false;
            return T{};
        }
        return GetLe<T>(m.bytes.data() + off);
    }
    u8 Byte(s64 off) {
        return Get<u8>(off);
    }
    u64 Uleb(s64& p) {
        u64 v = 0;
        for (u32 shift = 0; shift < 64 && ok; shift += 7) {
            const u8 b = Byte(p++);
            v |= u64{b & 0x7Fu} << shift;
            if ((b & 0x80) == 0) {
                break;
            }
        }
        return v;
    }
    /// DWARF EH pointer at p, as a module offset; std::nullopt for DW_EH_PE_omit or an unknown encoding.
    std::optional<s64> Encoded(s64& p, u8 enc, s64 datarel) {
        if (enc == 0xFF) {
            return std::nullopt;
        }
        const s64 at = p;
        s64 v = 0;
        switch (enc & 0x0F) {
        case 0x0B:  // sdata4
            v = Get<s32>(p);
            p += 4;
            break;
        case 0x03:  // udata4
            v = Get<u32>(p);
            p += 4;
            break;
        case 0x00:  // absptr
        case 0x04:  // udata8
        case 0x0C:  // sdata8
            v = static_cast<s64>(Get<u64>(p));
            p += 8;
            break;
        case 0x0A:  // sdata2
            v = Get<s16>(p);
            p += 2;
            break;
        case 0x02:  // udata2
            v = Get<u16>(p);
            p += 2;
            break;
        default:
            ok = false;
            return std::nullopt;
        }
        switch (enc & 0x70) {
        case 0x00:
            break;
        case 0x10:  // pcrel
            v += at;
            break;
        case 0x30:  // datarel
            v += datarel;
            break;
        default:
            ok = false;
            return std::nullopt;
        }
        return v;
    }
};

constexpr u32 mod0_magic = 0x30444F4D;  // "MOD0"

struct SeedCounts {
    size_t fde = 0, reloc = 0, adr = 0, sym = 0, entry = 0, sweep = 0, thunk = 0, manifest = 0;
};

s64 SignExtend(u64 v, u32 bits) {
    const u64 m = u64{1} << (bits - 1);
    return static_cast<s64>((v ^ m) - m);
}

/// Seeds from one module's metadata: FDE starts (and ranges), relocation targets, function symbols, ADR/ADRP+ADD
/// code addresses, the entry.
void ModuleSeeds(const JitCodeImage::Module& mod, const JitCodeImage& image, std::vector<u64>& seeds,
                 std::vector<std::pair<u64, u64>>& fde_ranges, SeedCounts& counts) {
    ModuleView m{mod};
    const auto in_text = [&](s64 off) {
        return off >= 0 && static_cast<u64>(off) < mod.text_size && (off & 3) == 0;
    };
    const auto add = [&](s64 off, size_t& counter) {
        if (in_text(off)) {
            seeds.push_back(mod.base + static_cast<u64>(off));
            ++counter;
        }
    };
    const s64 mod0 = m.Get<u32>(4);
    if (!m.ok || m.Get<u32>(mod0) != mod0_magic) {
        return;
    }
    // Entry (rtld's _start is a branch at the module base).
    if ((m.Get<u32>(0) & 0xFC000000) == 0x14000000) {
        add(0, counts.entry);
    }

    // FDE starts and ranges: MOD0 -> .eh_frame_hdr -> binary search table -> each FDE's pc_begin / pc_range.
    {
        ModuleView v{mod};
        const s64 hdr = mod0 + v.Get<s32>(mod0 + 16);
        const s64 hdr_end = mod0 + v.Get<s32>(mod0 + 20);
        if (v.ok && hdr_end > hdr && v.In(hdr, 4) && v.Byte(hdr) == 1) {
            const u8 ptr_enc = v.Byte(hdr + 1);
            const u8 cnt_enc = v.Byte(hdr + 2);
            const u8 tab_enc = v.Byte(hdr + 3);
            s64 p = hdr + 4;
            v.Encoded(p, ptr_enc, hdr);
            const auto count = v.Encoded(p, cnt_enc, hdr);
            std::unordered_map<s64, u8> cie_enc;
            const s64 n = count && *count > 0 && *count < (s64{1} << 26) ? *count : 0;
            for (s64 i = 0; i < n && v.ok; ++i) {
                v.Encoded(p, tab_enc, hdr);  // initial location (also in the FDE)
                const auto fde = v.Encoded(p, tab_enc, hdr);
                if (!fde) {
                    break;
                }
                const s64 cie = *fde + 4 - static_cast<s64>(v.Get<u32>(*fde + 4));
                auto it = cie_enc.find(cie);
                if (it == cie_enc.end()) {
                    u8 enc = 0;
                    const u8 version = v.Byte(cie + 8);
                    s64 q = cie + 9;
                    std::string aug;
                    while (v.ok && aug.size() < 16) {
                        const char c = static_cast<char>(v.Byte(q++));
                        if (c == '\0') {
                            break;
                        }
                        aug += c;
                    }
                    v.Uleb(q);  // code alignment
                    v.Uleb(q);  // data alignment (sleb: same length)
                    if (version == 1) {
                        ++q;  // return address register
                    } else {
                        v.Uleb(q);
                    }
                    if (!aug.empty() && aug[0] == 'z') {
                        v.Uleb(q);  // augmentation data length
                        for (size_t k = 1; k < aug.size() && v.ok; ++k) {
                            if (aug[k] == 'R') {
                                enc = v.Byte(q++);
                            } else if (aug[k] == 'P') {
                                const u8 penc = v.Byte(q++);
                                v.Encoded(q, static_cast<u8>(penc & 0x7F), 0);
                            } else if (aug[k] == 'L') {
                                ++q;
                            } else {
                                break;
                            }
                        }
                    }
                    it = cie_enc.emplace(cie, enc).first;
                }
                s64 q = *fde + 8;
                const auto begin = v.Encoded(q, it->second, 0);
                const auto range = v.Encoded(q, static_cast<u8>(it->second & 0x0F), 0);
                if (!begin || !range || !v.ok) {
                    break;
                }
                if (in_text(*begin)) {
                    fde_ranges.emplace_back(mod.base + *begin, mod.base + *begin + *range);
                    add(*begin, counts.fde);
                }
            }
        }
    }

    // Dynamic section: relocations and symbols.
    s64 rela = -1, relasz = 0, relr = -1, relrsz = 0, symtab = -1, strtab = -1, syment = 24;
    {
        s64 p = mod0 + m.Get<s32>(mod0 + 4);
        for (int i = 0; i < 4096 && m.ok; ++i, p += 16) {
            const s64 tag = m.Get<s64>(p);
            const s64 val = static_cast<s64>(m.Get<u64>(p + 8));
            if (tag == 0) {
                break;
            }
            switch (tag) {
            case 7: rela = val; break;
            case 8: relasz = val; break;
            case 36: relr = val; break;
            case 35: relrsz = val; break;
            case 6: symtab = val; break;
            case 5: strtab = val; break;
            case 11: syment = val; break;
            default: break;
            }
        }
    }
    if (rela >= 0 && m.In(rela, static_cast<size_t>(std::max<s64>(relasz, 0)))) {
        for (s64 p = rela; p + 24 <= rela + relasz; p += 24) {
            if ((m.Get<u64>(p + 8) & 0xFFFFFFFF) == 1027) {  // R_AARCH64_RELATIVE
                add(m.Get<s64>(p + 16), counts.reloc);
            }
        }
    }
    if (relr >= 0 && m.In(relr, static_cast<size_t>(std::max<s64>(relrsz, 0)))) {
        // RELR: the addend is the word at the location (not yet relocated: no guest code ran).
        s64 where = 0;
        for (s64 p = relr; p + 8 <= relr + relrsz; p += 8) {
            const u64 e = m.Get<u64>(p);
            if ((e & 1) == 0) {
                add(m.Get<s64>(static_cast<s64>(e)), counts.reloc);
                where = static_cast<s64>(e) + 8;
            } else {
                u64 bits = e >> 1;
                for (s64 k = 0; bits != 0; ++k, bits >>= 1) {
                    if (bits & 1) {
                        add(m.Get<s64>(where + 8 * k), counts.reloc);
                    }
                }
                where += 63 * 8;
            }
        }
    }
    if (symtab >= 0 && strtab > symtab && syment >= 24) {
        for (s64 p = symtab; p + 24 <= strtab && m.In(p, 24); p += syment) {
            const u8 info = m.Byte(p + 4);
            const u16 shndx = m.Get<u16>(p + 6);
            const u64 value = m.Get<u64>(p + 8);
            if (shndx != 0 && (info & 0xF) == 2 && value != 0) {  // defined STT_FUNC
                add(static_cast<s64>(value), counts.sym);
            }
        }
    }

    // ADR, and ADRP + ADD (64-bit, same register, within the next four instructions).
    const u8* text = mod.bytes.data();
    const u64 words = mod.text_size / 4;
    for (u64 i = 0; i < words; ++i) {
        const u32 w = GetLe<u32>(text + 4 * i);
        const u64 pc = 4 * i;
        if ((w & 0x9F000000) == 0x10000000) {
            const s64 imm = SignExtend(((w >> 29) & 3) | (((w >> 5) & 0x7FFFF) << 2), 21);
            add(static_cast<s64>(pc) + imm, counts.adr);
        } else if ((w & 0x9F000000) == 0x90000000) {
            const s64 imm = SignExtend(((w >> 29) & 3) | (((w >> 5) & 0x7FFFF) << 2), 21) * 4096;
            const s64 page = static_cast<s64>(pc & ~u64{0xFFF}) + imm;
            const u32 rd = w & 31;
            for (u64 k = 1; k <= 4 && i + k < words; ++k) {
                const u32 n = GetLe<u32>(text + 4 * (i + k));
                if ((n & 0xFFC00000) == 0x91000000 && ((n >> 5) & 31) == rd) {
                    add(page + ((n >> 10) & 0xFFF), counts.adr);
                }
            }
        }
    }
    (void)image;
}

/// Linear sweep over a window (word rules, no reachability): direct branch targets, the instruction after every
/// non-final terminator, and the instruction after every final one (B, RET, BR, padding). Finds the case labels
/// of jump tables that no recogniser resolves.
void SweepSeeds(const JitCodeImage::Window& w, std::vector<u64>& seeds, size_t& counter) {
    const u64 words = w.size / 4;
    const auto add = [&](s64 pc) {
        if (pc >= static_cast<s64>(w.base) && static_cast<u64>(pc) < w.base + w.size && (pc & 3) == 0) {
            seeds.push_back(static_cast<u64>(pc));
            ++counter;
        }
    };
    for (u64 i = 0; i < words; ++i) {
        const u32 x = GetLe<u32>(w.data + 4 * i);
        const s64 pc = static_cast<s64>(w.base + 4 * i);
        if ((x & 0x7C000000) == 0x14000000) {  // B, BL
            add(pc + SignExtend(x & 0x3FFFFFF, 26) * 4);
        }
        const bool bcond = (x & 0xFF000010) == 0x54000000;
        const bool cb = (x & 0x7E000000) == 0x34000000;
        const bool tb = (x & 0x7E000000) == 0x36000000;
        if (bcond || cb) {
            add(pc + SignExtend((x >> 5) & 0x7FFFF, 19) * 4);
        } else if (tb) {
            add(pc + SignExtend((x >> 5) & 0x3FFF, 14) * 4);
        }
        const bool bl = (x & 0xFC000000) == 0x94000000;
        const bool blr = (x & 0xFFFFFC1F) == 0xD63F0000 || (x & 0xFEFFF800) == 0xD63F0800;
        const bool exc = (x & 0xFF000000) == 0xD4000000;
        const bool atom = (x & 0x3F200C00) == 0x38200000 && (x & 0x0000F000) != 0xC000 && (x & 0x04000000) == 0;
        const bool cas = (x & 0x3FA07C00) == 0x08A07C00 || (x & 0xBFA07C00) == 0x08207C00;
        const bool sysreg = (x & 0xFFFFFFE0) == 0xD51B4400 || (x & 0xFFFFF0FF) == 0xD50330DF ||
                            (x & 0xFFF80000) == 0xD5080000;
        const bool final_ = (x & 0xFC000000) == 0x14000000 || (x & 0xFFFFFC1F) == 0xD65F0000 ||
                            (x & 0xFFFFFC1F) == 0xD61F0000 || (x & 0xFFFFFBFF) == 0xD65F0BFF ||
                            (x & 0xFEFFF800) == 0xD61F0800 || x == 0 || x == 0xD503201F;
        if (bcond || cb || tb || bl || blr || exc || atom || cas || sysreg || final_) {
            add(pc + 4);
        }
    }
}

struct Found {
    u64 pc;
    u64 end_pc;
    u64 code_hash;
};

/// One visited bit per instruction of each window, set atomically.
class VisitMap {
public:
    explicit VisitMap(const JitCodeImage& image_) : image(image_) {
        bits.reserve(image.Windows().size());
        for (const auto& w : image.Windows()) {
            bits.emplace_back(new std::atomic<u64>[(w.size / 4 + 63) / 64]());
        }
    }
    /// True if `pc` is an instruction of the image not visited before (and marks it).
    bool Mark(u64 pc) {
        if ((pc & 3) != 0) {
            return false;
        }
        const int i = image.Find(pc);
        if (i < 0) {
            return false;
        }
        const u64 index = (pc - image.Windows()[i].base) / 4;
        const u64 bit = u64{1} << (index & 63);
        return (bits[i][index / 64].fetch_or(bit, std::memory_order_relaxed) & bit) == 0;
    }

private:
    const JitCodeImage& image;
    std::vector<std::unique_ptr<std::atomic<u64>[]>> bits;
};

class ListWriter {
public:
    ListWriter(const std::filesystem::path& path_) : path(path_), file(path_, std::ios::binary | std::ios::trunc) {
        std::array<u8, JitPrebuildList::header_size> zero{};
        file.write(reinterpret_cast<const char*>(zero.data()), zero.size());
    }
    void Add(u64 descriptor, u64 end_pc, u64 hash, u16 flags) {
        u8 e[JitPrebuildList::entry_size]{};
        const u64 pc = DescriptorPc(descriptor);
        const u16 insts = static_cast<u16>(std::min<u64>(end_pc > pc ? (end_pc - pc) / 4 : 0, 0xFFFF));
        std::memcpy(e, &descriptor, 8);
        std::memcpy(e + 8, &end_pc, 8);
        std::memcpy(e + 16, &hash, 8);
        std::memcpy(e + 24, &flags, 2);
        std::memcpy(e + 26, &insts, 2);
        buffer.insert(buffer.end(), e, e + sizeof(e));
        if (buffer.size() >= (4u << 20)) {
            FlushBuffer();
        }
        ++count;
    }
    bool Finish(u64 program_id, u64 image_hash, u32 common_fpcr, u32 manifest_blocks) {
        FlushBuffer();
        std::array<u8, JitPrebuildList::header_size> h{};
        const auto put32 = [&](size_t at, u32 v) { std::memcpy(h.data() + at, &v, 4); };
        const auto put64 = [&](size_t at, u64 v) { std::memcpy(h.data() + at, &v, 8); };
        put32(0, 0x4C50324E);  // "N2PL"
        put32(4, 1);
        put32(8, static_cast<u32>(JitPrebuildList::entry_size));
        put32(12, jit_code_list_generator);
        put64(16, program_id);
        put64(24, image_hash);
        put64(32, count);
        put32(40, common_fpcr);
        put32(44, manifest_blocks);
        file.seekp(0);
        file.write(reinterpret_cast<const char*>(h.data()), h.size());
        file.close();
        return static_cast<bool>(file);
    }
    size_t Count() const {
        return count;
    }

private:
    void FlushBuffer() {
        file.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        buffer.clear();
    }
    std::filesystem::path path;
    std::ofstream file;
    std::vector<u8> buffer;
    size_t count = 0;
};

/// FPCR value as the static hint resolves it: per bit, set (s), clear (c), the FPCR's old bit (o: the write keeps
/// it) or unknown (the rest).
struct FpcrValue {
    u32 s = 0, c = 0, o = 0;

    static FpcrValue Const(u64 v) {
        return {static_cast<u32>(v), static_cast<u32>(~v), 0};
    }
    static FpcrValue Old() {
        return {0, 0, 0xFFFFFFFF};
    }
    static FpcrValue Unknown() {
        return {};
    }
    FpcrValue Normal() const {
        return {s, c & ~s, o & ~s & ~c};
    }
};
FpcrValue FpcrOr(FpcrValue a, FpcrValue b) {
    return FpcrValue{a.s | b.s, a.c & b.c, (a.o & b.o) | (a.o & b.c) | (a.c & b.o)}.Normal();
}
FpcrValue FpcrAnd(FpcrValue a, FpcrValue b) {
    return FpcrValue{a.s & b.s, a.c | b.c, (a.o & b.o) | (a.o & b.s) | (a.s & b.o)}.Normal();
}
FpcrValue FpcrXor(FpcrValue a, FpcrValue b) {
    return FpcrValue{(a.s & b.c) | (a.c & b.s), (a.s & b.s) | (a.c & b.c), (a.o & b.c) | (a.c & b.o)}.Normal();
}
FpcrValue FpcrNot(FpcrValue a) {
    return FpcrValue{a.c, a.s, 0}.Normal();
}
FpcrValue FpcrSelect(FpcrValue a, FpcrValue b) {
    return FpcrValue{a.s & b.s, a.c & b.c, a.o & b.o}.Normal();
}
FpcrValue FpcrInsert(FpcrValue prev, FpcrValue field, u32 mask) {
    return FpcrValue{(prev.s & ~mask) | (field.s & mask), (prev.c & ~mask) | (field.c & mask),
                     (prev.o & ~mask) | (field.o & mask)}.Normal();
}

std::optional<u64> DecodeBitMask(u32 n, u32 imms, u32 immr, u32 width) {
    const u32 len_bits = (n << 6) | (~imms & 0x3F);
    if (len_bits == 0) {
        return std::nullopt;
    }
    const u32 len = static_cast<u32>(std::bit_width(len_bits)) - 1;
    if (len < 1) {
        return std::nullopt;
    }
    const u32 esize = 1u << len;
    const u32 levels = esize - 1;
    const u32 s = imms & levels;
    const u32 r = immr & levels;
    if (s == levels) {
        return std::nullopt;
    }
    const u64 emask = esize == 64 ? ~u64{0} : (u64{1} << esize) - 1;
    const u64 welem = (u64{1} << (s + 1)) - 1;
    const u64 elem = r == 0 ? welem : ((welem >> r) | (welem << (esize - r))) & emask;
    u64 out = 0;
    for (u32 k = 0; k < width; k += esize) {
        out |= elem << k;
    }
    return width == 64 ? out : out & 0xFFFFFFFF;
}

/// Resolves the value an MSR FPCR writes, backward from the write within its block (one code window).
class FpcrResolver {
public:
    /// `data` holds `size` bytes, code first (a module's data after it is read for LDR literal only).
    FpcrResolver(const u8* data_, u64 size_) : data(data_), size(size_) {}

    FpcrValue Resolve(u32 reg, u64 at, int depth = 0, std::optional<u64> lo = std::nullopt) const {
        if (reg == 31) {
            return FpcrValue::Const(0);
        }
        if (depth > 8) {
            return FpcrValue::Unknown();
        }
        const u64 low = lo ? *lo : (at > 4 * window ? at - 4 * window : 0);
        for (u64 i = at; i >= low + 4;) {
            i -= 4;
            const u32 w = Word(i);
            // An unconditional branch, return or padding before the write: the block starts after it.
            if ((w & 0xFC000000) == 0x14000000 || (w & 0xFFFFFC1F) == 0xD65F0000 || (w & 0xFFFFFC1F) == 0xD61F0000 ||
                (w & 0xFFFFFBFF) == 0xD65F0BFF || w == 0) {
                return FpcrValue::Unknown();
            }
            if (const auto v = Writes(w, reg, i, depth, low)) {
                return *v;
            }
        }
        return FpcrValue::Unknown();
    }

    u32 Word(u64 off) const {
        return GetLe<u32>(data + off);
    }

private:
    static constexpr u64 window = 24;

    FpcrValue Source(u32 rn, u64 i, int depth, u64 lo) const {
        return rn == 31 ? FpcrValue::Const(0) : Resolve(rn, i, depth + 1, lo);
    }

    /// Nothing if `w` does not write X<reg>; else the value written.
    std::optional<FpcrValue> Writes(u32 w, u32 reg, u64 i, int depth, u64 lo) const {
        const u32 rd = w & 31;
        const auto unknown = std::optional<FpcrValue>{FpcrValue::Unknown()};
        // Calls clobber x0-x18 and x30.
        if ((w & 0xFC000000) == 0x94000000 || (w & 0xFFFFFC1F) == 0xD63F0000 || (w & 0xFEFFF800) == 0xD63F0800) {
            return reg <= 18 || reg == 30 ? unknown : std::nullopt;
        }
        if ((w & 0x7C000000) == 0x34000000 || (w & 0xFF000010) == 0x54000000) {  // CBZ/TBZ, B.cond
            return std::nullopt;
        }
        const auto logical = [](u32 opc, FpcrValue a, FpcrValue b) {
            return opc == 1 ? FpcrOr(a, b) : opc == 2 ? FpcrXor(a, b) : FpcrAnd(a, b);
        };
        if ((w & 0x1F800000) == 0x12800000) {  // MOVN, MOVZ, MOVK
            if (rd != reg) {
                return std::nullopt;
            }
            const u32 hw = (w >> 21) & 3;
            const u64 imm = u64{(w >> 5) & 0xFFFF} << (16 * hw);
            switch ((w >> 29) & 3) {
            case 0:
                return FpcrValue::Const(~imm);
            case 2:
                return FpcrValue::Const(imm);
            case 3:
                return FpcrInsert(Resolve(reg, i, depth + 1, lo), FpcrValue::Const(imm),
                                  static_cast<u32>(u64{0xFFFF} << (16 * hw)));
            default:
                return unknown;
            }
        }
        if ((w & 0x1F800000) == 0x12000000) {  // logical immediate
            if (rd != reg) {
                return std::nullopt;
            }
            const auto m = DecodeBitMask((w >> 22) & 1, (w >> 10) & 0x3F, (w >> 16) & 0x3F, (w >> 31) ? 64 : 32);
            if (!m) {
                return unknown;
            }
            return logical((w >> 29) & 3, Source((w >> 5) & 31, i, depth, lo), FpcrValue::Const(*m));
        }
        if ((w & 0x1F000000) == 0x0A000000) {  // logical shifted register, shift 0 only
            if (rd != reg) {
                return std::nullopt;
            }
            if ((w >> 10) & 0x3F) {
                return unknown;
            }
            FpcrValue b = Source((w >> 16) & 31, i, depth, lo);
            if ((w >> 21) & 1) {
                b = FpcrNot(b);
            }
            return logical((w >> 29) & 3, Source((w >> 5) & 31, i, depth, lo), b);
        }
        if ((w & 0x3FE00C00) == 0x1A800000) {  // CSEL
            if (rd != reg) {
                return std::nullopt;
            }
            return FpcrSelect(Source((w >> 5) & 31, i, depth, lo), Source((w >> 16) & 31, i, depth, lo));
        }
        if ((w & 0x1F800000) == 0x13000000) {  // SBFM, BFM, UBFM
            if (rd != reg) {
                return std::nullopt;
            }
            const u32 opc = (w >> 29) & 3;
            const u32 immr = (w >> 16) & 0x3F;
            const u32 imms = (w >> 10) & 0x3F;
            const u32 width = (w >> 31) ? 64 : 32;
            if (opc != 1 && opc != 2) {
                return unknown;
            }
            const FpcrValue a = Source((w >> 5) & 31, i, depth, lo);
            u32 fw, pos, shift;
            if (imms >= immr) {
                fw = imms - immr + 1, pos = 0, shift = immr;
            } else {
                fw = imms + 1, pos = width - immr, shift = 0;
            }
            const u64 fm = fw >= 64 ? ~u64{0} : (u64{1} << fw) - 1;
            const auto move = [&](u32 x) {
                return pos >= 32 ? 0u : static_cast<u32>((((u64{x} >> shift) & fm) << pos) & 0xFFFFFFFF);
            };
            const FpcrValue moved{move(a.s), move(a.c), 0};
            const u32 mask = pos >= 32 ? 0u : static_cast<u32>((fm << pos) & 0xFFFFFFFF);
            return FpcrInsert(opc == 2 ? FpcrValue::Const(0) : Resolve(reg, i, depth + 1, lo), moved, mask);
        }
        if ((w & 0xFFFFFFE0) == 0xD53B4400) {  // MRS Xt, FPCR
            return rd == reg ? std::optional{FpcrValue::Old()} : std::nullopt;
        }
        if ((w & 0xFFF00000) == 0xD5300000) {  // other MRS
            return rd == reg ? unknown : std::nullopt;
        }
        if ((w & 0xBF000000) == 0x18000000) {  // LDR literal (W, X): the low 32 bits
            if (rd != reg) {
                return std::nullopt;
            }
            const s64 off = static_cast<s64>(i) + SignExtend((w >> 5) & 0x7FFFF, 19) * 4;
            if (off < 0 || static_cast<u64>(off) + 4 > size) {
                return unknown;
            }
            return FpcrValue::Const(GetLe<u32>(data + off));
        }
        if ((w & 0x0A000000) == 0x08000000) {  // loads and stores
            const u32 rn = (w >> 5) & 31;
            const bool pair = (w & 0x3A000000) == 0x28000000;
            const bool exclusive = (w & 0x3F000000) == 0x08000000;
            const bool writeback =
                pair ? (((w >> 23) & 3) == 1 || ((w >> 23) & 3) == 3) : (w & 0x3B200400) == 0x38000400;
            if (writeback && rn == reg) {
                return unknown;
            }
            if (exclusive) {
                return rd == reg || ((w >> 16) & 31) == reg ? unknown : std::nullopt;
            }
            const bool load = pair ? ((w >> 22) & 1) != 0 : ((w >> 22) & 3) != 0;
            if (load && (rd == reg || (pair && ((w >> 10) & 31) == reg))) {
                return unknown;
            }
            return std::nullopt;
        }
        if ((w & 0xFFFFF01F) == 0xD503201F || (w & 0xFFF00000) == 0xD5000000 || (w & 0xFFF00000) == 0xD5100000) {
            return std::nullopt;  // hints, MSR, SYS
        }
        // Anything else that names X<reg> as its destination: unknown (conservative).
        return rd == reg ? unknown : std::nullopt;
    }

    const u8* data;
    u64 size;
};

constexpr u32 key_fpcr_mask = 0x07C80000;  // the FPCR bits a block key keeps (A64::LocationDescriptor)

} // namespace

std::unique_ptr<JitCodeImage> JitCodeImage::Capture(Kernel::KProcess& process) {
    auto image = std::unique_ptr<JitCodeImage>(new JitCodeImage());
    auto& page_table = process.GetPageTable();
    auto& memory = process.GetMemory();
    std::vector<std::pair<u64, u64>> blocks;  // executable code blocks, merged when adjacent
    u64 cur = 0;
    for (int guard = 0; guard < 1 << 20; ++guard) {
        Kernel::KMemoryInfo mem_info{};
        Kernel::Svc::PageInfo page_info{};
        if (page_table.QueryInfo(std::addressof(mem_info), std::addressof(page_info), cur).IsError()) {
            break;
        }
        const auto info = mem_info.GetSvcMemoryInfo();
        if (info.permission == Kernel::Svc::MemoryPermission::ReadExecute &&
            (info.state == Kernel::Svc::MemoryState::Code || info.state == Kernel::Svc::MemoryState::AliasCode)) {
            if (!blocks.empty() && blocks.back().first + blocks.back().second == info.base_address) {
                blocks.back().second += info.size;
            } else {
                blocks.emplace_back(info.base_address, info.size);
            }
        }
        const u64 next = info.base_address + info.size;
        if (next <= cur) {
            break;
        }
        cur = next;
    }
    for (const auto& [base, size] : blocks) {
        u32 mod0_offset = 0;
        u32 magic = 0;
        const u64 module_end = GetInteger(GetModuleEnd(&process, base)) + 1;
        const bool module = module_end > base + size && module_end - base < (u64{1} << 32) &&
                            memory.ReadBlock(base + 4, &mod0_offset, 4) && mod0_offset + 4 <= module_end - base &&
                            memory.ReadBlock(base + mod0_offset, &magic, 4) && magic == mod0_magic;
        if (module) {
            Module m;
            m.base = base;
            m.text_size = size;
            m.bytes.resize(module_end - base);
            memory.ReadBlock(base, m.bytes.data(), m.bytes.size());
            image->modules.push_back(std::move(m));
        } else {
            std::vector<u8> bytes(size);
            memory.ReadBlock(base, bytes.data(), bytes.size());
            image->loose.push_back(std::move(bytes));
            image->windows.push_back({base, size, image->loose.back().data(), -1});
        }
    }
    for (size_t i = 0; i < image->modules.size(); ++i) {
        const Module& m = image->modules[i];
        image->windows.push_back({m.base, m.text_size, m.bytes.data(), static_cast<int>(i)});
    }
    // NVN-HLE thunk slots: in the sdk module's data segment (RW), so not found by the walk above.
    auto& bridge = Core::NvnHle::Bridge::Instance();
    if (process.IsApplication() && bridge.IsActive() && bridge.RegionBase() != 0) {
        const u64 base = bridge.RegionBase();
        std::vector<u8> bytes(Core::NvnHle::Thunk::CodeBytes);
        if (memory.ReadBlock(base, bytes.data(), bytes.size())) {
            image->thunk_base = base;
            image->loose.push_back(std::move(bytes));
            image->windows.push_back({base, Core::NvnHle::Thunk::CodeBytes, image->loose.back().data(), -1});
        }
    }
    std::sort(image->windows.begin(), image->windows.end(),
              [](const Window& a, const Window& b) { return a.base < b.base; });
    // Overlaps cannot happen between page-table blocks; drop a thunk window that lies inside one anyway.
    std::vector<Window> kept;
    for (const auto& w : image->windows) {
        if (!kept.empty() && w.base < kept.back().base + kept.back().size) {
            continue;
        }
        kept.push_back(w);
    }
    image->windows = std::move(kept);
    u64 h = 0x4E32504C;
    for (const auto& w : image->windows) {
        h = Common::CityHash64WithSeeds(reinterpret_cast<const char*>(w.data), static_cast<size_t>(w.size), h,
                                        w.base ^ (w.size << 1));
    }
    image->hash = h;
    return image;
}

u64 JitCodeImage::CodeBytes() const {
    u64 n = 0;
    for (const auto& w : windows) {
        n += w.size;
    }
    return n;
}

int JitCodeImage::Find(u64 pc) const {
    auto it = std::upper_bound(windows.begin(), windows.end(), pc,
                               [](u64 v, const Window& w) { return v < w.base; });
    if (it == windows.begin()) {
        return -1;
    }
    --it;
    if (pc >= it->base + it->size) {
        return -1;
    }
    return static_cast<int>(it - windows.begin());
}

JitFpcrHint JitStaticFpcrHint(const JitCodeImage& image) {
    JitFpcrHint hint;
    constexpr u32 start = jit_code_list_first_boot_fpcr & key_fpcr_mask;
    std::map<std::pair<u32, u32>, std::size_t> effects;  // (set, clear) key bits -> writes that change `start`
    std::size_t no_change = 0;
    for (const auto& w : image.Windows()) {
        u64 size = w.size;
        if (w.module >= 0) {
            size = image.Modules()[static_cast<size_t>(w.module)].bytes.size();  // literals may sit in .rodata
        }
        const FpcrResolver resolver{w.data, size};
        for (u64 off = 0; off + 4 <= w.size; off += 4) {
            const u32 x = resolver.Word(off);
            if ((x & 0xFFFFFFE0) != 0xD51B4400) {  // MSR FPCR, Xt
                continue;
            }
            ++hint.writes;
            const FpcrValue v = resolver.Resolve(x & 31, off);
            const u32 set = v.s & key_fpcr_mask;
            const u32 clear = v.c & key_fpcr_mask;
            if (set != 0 || clear != 0) {
                ++hint.resolved;
                // A write that leaves the thread-start FPCR as it is (msr fpcr, xzr from 0: a library resetting
                // the default) says nothing about the mode the game picks: it must not outvote the game's write.
                if (((start & ~clear) | set) == start) {
                    ++no_change;
                } else {
                    ++effects[{set, clear}];
                }
            }
        }
    }
    std::size_t best = 0;
    bool tie = false;
    std::pair<u32, u32> chosen{};
    for (const auto& [effect, n] : effects) {
        hint.detail += fmt::format("{}set {:08X} clear {:08X}: {}", hint.detail.empty() ? "" : ", ", effect.first,
                                   effect.second, n);
        if (n > best) {
            best = n, chosen = effect, tie = false;
        } else if (n == best) {
            tie = true;
        }
    }
    if (best != 0 && !tie) {
        hint.fpcr = ((jit_code_list_first_boot_fpcr & ~chosen.second) | chosen.first) & key_fpcr_mask;
    }
    if (hint.detail.empty()) {
        hint.detail = hint.resolved == 0 ? "none sets or clears a key bit" : "none changes the thread-start FPCR";
    } else if (tie) {
        hint.detail += " (tie: thread-start FPCR kept)";
    }
    if (no_change != 0) {
        hint.detail += fmt::format("; {} leave the thread-start FPCR unchanged, not counted", no_change);
    }
    return hint;
}

std::vector<std::pair<u32, std::size_t>> JitManifestFpcrMix(std::span<const JitBlockCache::ManifestEntry> manifest) {
    std::map<u32, std::size_t> counts;
    for (const auto& e : manifest) {
        ++counts[DescriptorFpcr(e.descriptor)];
    }
    std::vector<std::pair<u32, std::size_t>> mix(counts.begin(), counts.end());
    std::stable_sort(mix.begin(), mix.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    return mix;
}

bool JitPrebuildListMatches(const JitPrebuildList& list, bool generated, const JitCodeImage& image,
                            ArmDynarmic64& jit, std::span<const std::pair<u32, std::size_t>> manifest_mix,
                            std::string& why) {
    const auto& h = list.Header();
    const u32 generator = GetLe<u32>(h.data() + 12);
    const u64 hash = GetLe<u64>(h.data() + 24);
    if (generated) {
        if (generator != jit_code_list_generator) {
            why = fmt::format("made by generator version {} (this is {})", generator, jit_code_list_generator);
            return false;
        }
        if (hash != image.Hash()) {
            why = fmt::format("made for code with image hash {:016X}, this code hashes to {:016X}", hash,
                              image.Hash());
            return false;
        }
        // Does the manifest (or the override) contradict the list's FPCR choice?
        const u32 list_fpcr = GetLe<u32>(h.data() + 40);
        const u32 list_manifest = GetLe<u32>(h.data() + 44);  // manifest blocks the list was made with
        if (const char* env = std::getenv("EDEN_JIT_CACHE_GENERATE_FPCR"); env && env[0] != '\0') {
            const u32 want = DescriptorFpcr(Descriptor(0, static_cast<u32>(std::strtoul(env, nullptr, 16))));
            if (want != list_fpcr) {
                why = fmt::format("made for common FPCR {:08X}, EDEN_JIT_CACHE_GENERATE_FPCR asks for {:08X}",
                                  list_fpcr, want);
                return false;
            }
        } else {
            std::size_t total = 0, in_list_mode = 0;
            for (const auto& [f, n] : manifest_mix) {
                total += n;
                if (f == list_fpcr) {
                    in_list_mode = n;
                }
            }
            constexpr std::size_t min_manifest = 4096;
            if (total >= min_manifest) {
                const auto [top, top_n] = manifest_mix.front();
                if (top != list_fpcr && top_n > 2 * in_list_mode) {
                    why = fmt::format("made for common FPCR {:08X}, but the manifest runs {} of {} blocks in FPCR "
                                      "{:08X} and {} in the list's mode",
                                      list_fpcr, top_n, total, top, in_list_mode);
                    return false;
                }
                if (list_manifest == 0 && (total - in_list_mode) * 4 >= total) {
                    why = fmt::format("made on a first boot for FPCR {:08X}; the manifest runs {} of {} blocks in "
                                      "other modes",
                                      list_fpcr, total - in_list_mode, total);
                    return false;
                }
            }
        }
        why = fmt::format("image hash {:016X} matches; common FPCR {:08X}, made with {} manifest blocks", hash,
                          list_fpcr, list_manifest);
        return true;
    }
    if (hash == image.Hash()) {
        why = fmt::format("image hash {:016X} matches", hash);
        return true;
    }
    // Made by another tool, which hashes the image its own way: check a sample of entries against this code.
    const size_t n = list.Size();
    const size_t samples = std::min<size_t>(n, 4096);
    size_t matched = 0;
    for (size_t k = 0; k < samples; ++k) {
        const JitPrebuildList::Entry e = list.At(samples == n ? k : k * (n / samples));
        if (jit.GuestCodeHash(DescriptorPc(e.descriptor), e.end_pc) == std::optional<u64>{e.guest_hash}) {
            ++matched;
        }
    }
    const bool ok = samples != 0 && matched * 100 >= samples * 95;
    why = fmt::format("image hash {:016X} is not this code's ({:016X}; the list was made by another tool); {} of {} "
                      "sampled entries match this code{}",
                      hash, image.Hash(), matched, samples, ok ? "" : " (95% needed)");
    return ok;
}

std::optional<std::size_t> GenerateJitCodeList(const JitCodeImage& image, std::span<ArmDynarmic64* const> jits,
                                               std::span<const JitBlockCache::ManifestEntry> manifest,
                                               u32 default_fpcr, const std::string& default_from, u64 program_id,
                                               const std::filesystem::path& out,
                                               const JitCodeListProgress& progress) {
    const auto start = Clock::now();
    const auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
    };
    if (jits.empty() || image.Windows().empty()) {
        LOG_WARNING(Core_ARM, "JIT code list: no code found to scan");
        return std::nullopt;
    }

    // Seeds.
    std::vector<u64> seeds;
    std::vector<std::pair<u64, u64>> fde_ranges;
    SeedCounts counts;
    for (const auto& m : image.Modules()) {
        ModuleSeeds(m, image, seeds, fde_ranges, counts);
    }
    if (image.ThunkBase() != 0) {
        for (u32 slot = 0; slot < Core::NvnHle::Thunk::SlotCount; ++slot) {
            seeds.push_back(image.ThunkBase() + Core::NvnHle::Thunk::SlotOffset(slot));
            ++counts.thunk;
        }
    }
    for (const auto& e : manifest) {
        seeds.push_back(DescriptorPc(e.descriptor));
        ++counts.manifest;
    }
    for (const auto& w : image.Windows()) {
        SweepSeeds(w, seeds, counts.sweep);
    }
    const auto seeded = Clock::now();

    // Descent with the JIT's translator, one thread per JIT; seeds handed out in chunks, each chunk followed to
    // the end before the next.
    VisitMap visited(image);
    const size_t thread_count = jits.size();
    std::vector<std::vector<Found>> found(thread_count);
    std::atomic<size_t> next_seed{0};
    std::atomic<size_t> scanned{0};
    std::atomic<size_t> not_cacheable{0};
    std::atomic<size_t> running{thread_count};
    constexpr size_t chunk = 1024;
    constexpr u32 scan_fpcr = 0;  // block ends and successors do not depend on FPCR; 0 as in every measured run
    std::vector<std::thread> threads;
    for (size_t t = 0; t < thread_count; ++t) {
        threads.emplace_back([&, t, jit = jits[t]] {
            Common::SetCurrentThreadName("JitCodeScan");
            std::vector<u64> stack;
            auto& mine = found[t];
            size_t local_scanned = 0, local_nc = 0;
            for (;;) {
                const size_t begin = next_seed.fetch_add(chunk, std::memory_order_relaxed);
                if (begin >= seeds.size()) {
                    break;
                }
                const size_t end = std::min(seeds.size(), begin + chunk);
                for (size_t i = begin; i < end; ++i) {
                    if (visited.Mark(seeds[i])) {
                        stack.push_back(seeds[i]);
                    }
                    while (!stack.empty()) {
                        const u64 pc = stack.back();
                        stack.pop_back();
                        const auto r = jit->ScanBlock(Descriptor(pc, scan_fpcr));
                        ++local_scanned;
                        if (!r) {
                            continue;
                        }
                        if (r->hashed && r->cacheable) {
                            mine.push_back({pc, r->end_pc, r->code_hash});
                        } else {
                            ++local_nc;
                        }
                        for (u8 s = 0; s < r->successor_count; ++s) {
                            if (visited.Mark(r->successors[s])) {
                                stack.push_back(r->successors[s]);
                            }
                        }
                    }
                }
            }
            scanned.fetch_add(local_scanned, std::memory_order_relaxed);
            not_cacheable.fetch_add(local_nc, std::memory_order_relaxed);
            running.fetch_sub(1, std::memory_order_release);
        });
    }
    while (running.load(std::memory_order_acquire) != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
        if (progress) {
            progress(std::min(next_seed.load(std::memory_order_relaxed), seeds.size()), seeds.size());
        }
    }
    for (auto& thread : threads) {
        thread.join();
    }
    const size_t seed_count = seeds.size();
    std::vector<u64>().swap(seeds);
    size_t total_found = 0;
    for (const auto& f : found) {
        total_found += f.size();
    }
    std::vector<Found> blocks;
    blocks.reserve(total_found);
    for (auto& f : found) {
        blocks.insert(blocks.end(), f.begin(), f.end());
        std::vector<Found>().swap(f);
    }
    std::sort(blocks.begin(), blocks.end(), [](const Found& a, const Found& b) { return a.pc < b.pc; });
    const auto descended = Clock::now();

    // Policy.
    std::map<u32, size_t> manifest_fpcrs;
    std::unordered_set<u64> executed;
    executed.reserve(manifest.size());
    for (const auto& e : manifest) {
        ++manifest_fpcrs[DescriptorFpcr(e.descriptor)];
        executed.insert(e.descriptor);
    }
    u32 common = default_fpcr;
    std::string common_from = "no manifest: " + default_from;
    if (const char* env = std::getenv("EDEN_JIT_CACHE_GENERATE_FPCR"); env && env[0] != '\0') {
        common = static_cast<u32>(std::strtoul(env, nullptr, 16));
        common_from = "EDEN_JIT_CACHE_GENERATE_FPCR";
    } else if (!manifest_fpcrs.empty()) {
        common = std::max_element(manifest_fpcrs.begin(), manifest_fpcrs.end(),
                                  [](const auto& a, const auto& b) { return a.second < b.second; })
                     ->first;
        common_from = "the manifest's most frequent FPCR";
    }
    common = DescriptorFpcr(Descriptor(0, common));  // the bits a block key keeps
    std::vector<u32> played_modes{common};
    for (const auto& [f, n] : manifest_fpcrs) {
        if (f != common) {
            played_modes.push_back(f);
        }
    }
    // Function units: intervals between FDE boundaries; a unit holding a manifest PC is "played".
    std::vector<u64> bounds;
    bounds.reserve(fde_ranges.size() * 2);
    for (const auto& [lo, hi] : fde_ranges) {
        bounds.push_back(lo);
        bounds.push_back(hi);
    }
    std::sort(bounds.begin(), bounds.end());
    bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());
    const auto unit_of = [&](u64 pc) {
        return static_cast<size_t>(std::upper_bound(bounds.begin(), bounds.end(), pc) - bounds.begin());
    };
    std::vector<bool> played(bounds.size() + 1);
    for (const auto& e : manifest) {
        played[unit_of(DescriptorPc(e.descriptor))] = true;
    }

    using F = JitPrebuildList::Flags;
    const auto temp = JitBlockCache::TempPathFor(out);
    ListWriter writer(temp);
    std::map<u32, size_t> per_fpcr;
    std::unordered_set<u64> emitted_executed;
    size_t n_executed = 0, n_learned = 0, n_played = 0, n_rest = 0;
    for (int pass = 0; pass < 3; ++pass) {
        if (pass == 1) {
            // Manifest keys discovery did not produce (code reached only through indirect branches no seed
            // names), with the manifest's own end and hash.
            for (const auto& e : manifest) {
                if (!emitted_executed.contains(e.descriptor)) {
                    writer.Add(e.descriptor, e.end_pc, e.code_hash,
                               static_cast<u16>(F::Executed | F::PlayedFunction | F::LearnedOnly | F::Generated |
                                                (DescriptorFpcr(e.descriptor) == common ? F::CommonMode : 0)));
                    ++per_fpcr[DescriptorFpcr(e.descriptor)];
                    ++n_learned;
                }
            }
        }
        for (const auto& b : blocks) {
            const bool in_played = played[unit_of(b.pc)];
            const size_t modes = in_played ? played_modes.size() : 1;
            for (size_t k = 0; k < modes; ++k) {
                const u32 fpcr = played_modes[k];
                const u64 d = Descriptor(b.pc, fpcr);
                const bool is_executed = executed.contains(d);
                const int want = is_executed ? 0 : in_played ? 1 : 2;
                if ((pass == 0 && want != 0) || (pass == 1 && want != 1) || (pass == 2 && want != 2)) {
                    continue;
                }
                if (is_executed) {
                    emitted_executed.insert(d);
                }
                writer.Add(d, b.end_pc, b.code_hash,
                           static_cast<u16>((in_played ? F::PlayedFunction : 0) | (is_executed ? F::Executed : 0) |
                                            (fpcr == common ? F::CommonMode : 0) | F::Generated));
                ++per_fpcr[fpcr];
                ++(pass == 0 ? n_executed : pass == 1 ? n_played : n_rest);
            }
        }
    }
    const size_t entries = writer.Count();
    if (!writer.Finish(program_id, image.Hash(), common,
                       static_cast<u32>(std::min<std::size_t>(manifest.size(), 0xFFFFFFFF)))) {
        std::error_code ec;
        std::filesystem::remove(temp, ec);
        LOG_WARNING(Core_ARM, "JIT code list: cannot write {}", temp.string());
        return std::nullopt;
    }
    std::error_code ec;
    if (!JitBlockCache::ReplaceDurably(temp, out, ec)) {
        std::filesystem::remove(temp, ec);
        LOG_WARNING(Core_ARM, "JIT code list: cannot write {}", out.string());
        return std::nullopt;
    }
    const auto written = Clock::now();

    std::string fpcr_text;
    for (const auto& [f, n] : per_fpcr) {
        fpcr_text += fmt::format("{}FPCR {:08X}: {}", fpcr_text.empty() ? "" : ", ", f, n);
    }
    LOG_INFO(Core_ARM,
             "JIT code list: {} windows, {} MiB of code, {} modules, image hash {:016X}; seeds {} (FDE {}, "
             "relocations {}, ADR {}, symbols {}, entries {}, thunk slots {}, manifest {}, sweep {}); {} FDE ranges",
             image.Windows().size(), image.CodeBytes() >> 20, image.Modules().size(), image.Hash(), seed_count,
             counts.fde, counts.reloc, counts.adr, counts.sym, counts.entry, counts.thunk, counts.manifest,
             counts.sweep, fde_ranges.size());
    LOG_INFO(Core_ARM,
             "JIT code list: {} blocks scanned on {} threads, {} kept, {} not cacheable or not hashable; common "
             "FPCR {:08X} ({}), {} FPCR modes in played functions, {} played function units of {}",
             scanned.load(), thread_count, blocks.size(), not_cacheable.load(), common, common_from,
             played_modes.size(), std::count(played.begin(), played.end(), true), played.size());
    LOG_INFO(Core_ARM,
             "JIT code list: wrote {} entries to {} ({} manifest keys found, {} manifest keys not found, {} rest of "
             "played functions, {} rest); per FPCR: {}; {} ms (seeds {} ms, scan {} ms, write {} ms)",
             entries, out.string(), n_executed, n_learned, n_played, n_rest, fpcr_text, ms(start, written),
             ms(start, seeded), ms(seeded, descended), ms(descended, written));
    return entries;
}

} // namespace Core
