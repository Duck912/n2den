// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/nvn/prerec_probe.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <emmintrin.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "video_core/nvn/guest_memory.h"

namespace VideoCommon::Nvn::PreRec {

namespace Detail {
namespace {
int ReadMode() {
    const char* value = std::getenv("EDEN_NX2_NVN_PREREC_PROBE");
    if (value == nullptr || *value == 0) {
        return 0;
    }
    const int parsed = std::atoi(value);
    if (parsed < 0 || parsed > 3 || !Common::TlRing::On()) {
        return 0;
    }
    return parsed;
}
} // namespace
const int mode = ReadMode();
thread_local bool ra_ahead = false;
thread_local bool ra_trace = false;
thread_local u32 ra_trace_fired = 0;
} // namespace Detail

namespace {
struct RaScope {
    u64 begin{};
    u64 end{}; ///< empty when begin == end
};
thread_local RaScope ra_scope;
thread_local RaReadStats ra_stats;
} // namespace

void RaNote(Cls cls, u64 va, u64 size) {
    const size_t c = static_cast<size_t>(cls);
    if (c < ra_stats.calls.size()) {
        ++ra_stats.calls[c];
        ra_stats.bytes[c] += size;
        ra_stats.last_va[c] = va;
    }
    ra_scope = {va, va + size};
}

bool RaCovered(u64 va, u64 size) noexcept {
    const u64 end = va + (size == 0 ? 1 : size);
    return ra_scope.begin < ra_scope.end && ra_scope.begin <= va && end <= ra_scope.end;
}

namespace {
/// Call sites (diagnostic) of unlogged accesses and of host-mapping lookups while recording
/// ahead: a few frames of the stack, as offsets into the executable, with a count each;
/// symbolized offline against the PDB. EDEN_NX2_NVN_RA_SITES=1 only; stacks are taken for the
/// first CAPTURE events (the rest are counted as uncaptured).
struct RaSite {
    std::array<u64, 8> frames{};
    u64 count{};
    u64 first_va{};
    u64 size{};
};
struct RaSiteTable {
    static constexpr size_t MAX_SITES = 128;
    static constexpr u64 CAPTURE = 20000;
    std::array<RaSite, MAX_SITES> sites;
    size_t count{};
    u64 captured{};
    u64 uncaptured{};
    u64 overflow{};

    void Note(u64 va, u64 size) noexcept {
#ifdef _WIN32
        // Opt-in (EDEN_NX2_NVN_RA_SITES=1): a stack walk per event slows the heads of the
        // loading phase, which can change what they meet (timing-dependent pool uploads).
        static const bool on = [] {
            const char* const value = std::getenv("EDEN_NX2_NVN_RA_SITES");
            return value != nullptr && value[0] == '1';
        }();
        if (!on || captured == CAPTURE) {
            ++uncaptured;
            return;
        }
        ++captured;
        void* raw[10]{};
        const USHORT n = RtlCaptureStackBackTrace(2, 10, raw, nullptr);
        static const u64 base = reinterpret_cast<u64>(GetModuleHandleW(nullptr));
        std::array<u64, 8> frames{};
        for (USHORT f = 0; f < n && f < frames.size(); ++f) {
            frames[f] = reinterpret_cast<u64>(raw[f]) - base;
        }
        for (size_t k = 0; k < count; ++k) {
            if (sites[k].frames == frames) {
                ++sites[k].count;
                return;
            }
        }
        if (count == MAX_SITES) {
            ++overflow;
            return;
        }
        sites[count++] = {.frames = frames, .count = 1, .first_va = va, .size = size};
#else
        (void)va;
        (void)size;
        ++uncaptured;
#endif
    }

    std::string Format() const {
        std::string out;
        char buf[48];
        for (size_t k = 0; k < count; ++k) {
            const RaSite& site = sites[k];
            std::snprintf(buf, sizeof(buf), " | site %zu: count %llu first va 0x%llx size %llu", k,
                          static_cast<unsigned long long>(site.count),
                          static_cast<unsigned long long>(site.first_va),
                          static_cast<unsigned long long>(site.size));
            out += buf;
            out += " frames (exe rva):";
            for (const u64 frame : site.frames) {
                std::snprintf(buf, sizeof(buf), " 0x%llx", static_cast<unsigned long long>(frame));
                out += buf;
            }
        }
        out += " | uncaptured " + std::to_string(uncaptured) + " overflow " +
               std::to_string(overflow);
        return out;
    }
};
thread_local RaSiteTable ra_unlogged_sites;
thread_local RaSiteTable ra_mapping_sites;
} // namespace

void RaCheck(u64 va, u64 size) noexcept {
    if (Detail::ra_ahead && !RaCovered(va, size)) [[unlikely]] {
        if (ra_stats.unlogged++ == 0) {
            ra_stats.first_unlogged = va;
        }
        ra_stats.last_unlogged = va;
        ra_stats.last_unlogged_size = size;
        ra_unlogged_sites.Note(va, size);
    }
}

void RaMapping(u64 va) noexcept {
    if (Detail::ra_ahead) [[unlikely]] {
        ++ra_stats.mapping_lookups;
        ra_mapping_sites.Note(va, 1);
    }
}

std::string RaUnloggedSites() {
    return ra_unlogged_sites.Format();
}

std::string RaMappingSites() {
    return ra_mapping_sites.Format();
}

void RaResetScope() noexcept {
    ra_scope = {};
}

RaReadStats& RaStats() noexcept {
    return ra_stats;
}

namespace {

constexpr u64 PAGE = 4096;
constexpr u64 LINE = 64;
constexpr u64 BIG = u64{64} << 10;
constexpr u32 FULL = 0xFFFFFFFFu;
constexpr u32 LINEMODE = 0xFFFFFFFEu; ///< every set line counts whole (line-granular classes)
constexpr u64 EMPTY = ~u64{0};
constexpr size_t NCLS = static_cast<size_t>(Cls::Count);
/// 0 trigger->suspend (mode 3 only), 1 suspend->wake (W minus W'), 2 wake->signal (W'),
/// 3 signal->end of the frame's recording.
constexpr int STAGES = 4;
constexpr u64 TIER1_BUDGET = u64{8} << 20;
constexpr u64 TIER2_BUDGET = u64{16} << 20;
constexpr size_t LOG_PAGES = size_t{1} << 17;
constexpr size_t LOG_MASKS = size_t{1} << 14;
constexpr size_t SNAP_PAGES = size_t{1} << 17;
constexpr size_t SNAP_DIFFS = 8192;
constexpr size_t MAX_HITS = 30000;
constexpr size_t MAX_EHITS = 60000;
constexpr size_t MAX_FRAMES = 8192;
constexpr size_t MAX_OPS = 32;
constexpr size_t NLOGS = 6;
constexpr size_t MAX_EPOCHS = 8192;

const char* const CLS_NAMES[NCLS] = {
    "MirrorCheck", "MirrorCopy",  "VramMirror",    "VramWanted",  "TexSig",      "TexFull",
    "TexBytes",    "TexGather",   "TexTakeover",   "UniformWord", "HandleMem",   "OracleUniform",
    "UniformCopy", "GenIndex",    "PoolUpload",    "EventWord",   "TexReadback", "SparseGrowth",
};

constexpr u32 Bit(Cls c) {
    return 1u << static_cast<u32>(c);
}
/// Record-time decisions taken from guest write tracking over whole ranges: measured with the
/// trackers' write epochs, kept per pool.
constexpr u32 DECISION = Bit(Cls::MirrorCheck) | Bit(Cls::VramMirror) | Bit(Cls::VramWanted) |
                         Bit(Cls::TexTakeover) | Bit(Cls::SparseGrowth);
/// Bytes captured on the CPU, small: snapshot tier 1 (diffed at wake and signal).
constexpr u32 TIER1 = Bit(Cls::UniformWord) | Bit(Cls::HandleMem) | Bit(Cls::UniformCopy) |
                      Bit(Cls::GenIndex) | Bit(Cls::EventWord) | Bit(Cls::OracleUniform);
/// Bytes captured on the CPU, larger: tier 2 (diffed at signal).
constexpr u32 TIER2 = Bit(Cls::TexSig) | Bit(Cls::MirrorCopy) | Bit(Cls::TexFull) |
                      Bit(Cls::TexBytes) | Bit(Cls::TexGather) | Bit(Cls::TexReadback) |
                      Bit(Cls::PoolUpload);
constexpr u32 LINE_GRAN = Bit(Cls::TexSig);

bool IsPageClass(Cls cls) {
    return cls == Cls::MirrorCopy;
}

u64 Now() {
    return Common::TlRing::Now();
}

u64 LineBytes(u64 line_va, u64 begin, u64 end) {
    const u64 a = (std::max)(line_va, begin) - line_va;
    const u64 b = (std::min)(line_va + LINE, end) - line_va;
    if (b <= a) {
        return 0;
    }
    const u64 len = b - a;
    const u64 bits = len >= 64 ? ~u64{0} : ((u64{1} << len) - 1);
    return bits << a;
}

u64 DiffMask(const u8* a, const u8* b) {
    u64 eq = 0;
    for (int i = 0; i < 4; ++i) {
        const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i * 16));
        const __m128i y = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i * 16));
        eq |= static_cast<u64>(static_cast<u32>(_mm_movemask_epi8(_mm_cmpeq_epi8(x, y))) & 0xFFFFu)
              << (i * 16);
    }
    return ~eq;
}

u64 HashPage(u64 page) {
    return (page * 0x9E3779B97F4A7C15ull) >> 20;
}

struct PageEnt {
    u64 page{EMPTY};
    u64 lines{};
    u64 id{};
    u32 cls{};
    u32 bm{}; ///< 0: none; FULL: whole page; LINEMODE: set lines whole; else masks index + 1
};

struct BigEnt {
    u64 va{};
    u64 size{};
    u64 id{};
    u32 cls{};
};

struct PoolUse {
    u32 cls{};
    u64 bytes{};
    u64 calls{};
};

/// One tracker write epoch the replay registered (a pool mirror, a Backing::Mirror pool, ...).
struct EpochEnt {
    u64 va{};
    u64 size{};
    u64 serial{};
    Cls cls{};
    std::shared_ptr<const std::atomic<u64>> epoch;
};

/// Epoch movement of one entry around one frame: A prev pass->suspend, B suspend->wake,
/// C wake->signal, D signal->pass.
struct EpochDelta {
    u32 index{};
    u64 a{};
    u64 b{};
    u64 c{};
    u64 d{};
};

struct MapRec {
    u64 tsc;
    u64 va;
    u64 size;
    u32 tid;
    bool flush;
};

/// The reads of one frame (between two passes of a gating wait).
struct FrameLog {
    std::vector<PageEnt> table = std::vector<PageEnt>(LOG_PAGES);
    std::vector<u32> used;
    std::vector<std::array<u64, 64>> masks = std::vector<std::array<u64, 64>>(LOG_MASKS);
    u32 masks_used{};
    std::vector<BigEnt> bigs;
    std::unordered_map<u64, PoolUse> pools; ///< decision classes, by pool serial
    std::array<u64, NCLS> calls{};
    std::array<u64, NCLS> bytes{};
    u64 overflow_pages{};
    u64 overflow_masks{};
    // Meta.
    u64 wait_value{};
    u64 wait_address{};
    u64 tsc_pass{};
    u64 tsc_end{};
    u64 tsc_block{};
    u64 block_seq{};
    u64 tsc_eventcpu{};
    u64 tsc_wb{};
    u64 tsc_wake{};
    u64 tsc_signal{};
    u32 wake_tid{};
    u32 signal_tid{};
    u64 signal_value{};
    bool end_blocked{};
    std::array<u16, MAX_OPS> ops{};
    u32 nops{};
    u64 ops_total{};
    std::array<u32, 128> op_hist{};
    u64 present_begin{};
    std::vector<EpochDelta> epochs; ///< movement around this frame's wait (nonzero only)
    std::unordered_map<const void*, u32> wcalls; ///< HLE calls between wake and signal
    std::unordered_map<u32, u32> wcall_tids;
    std::vector<MapRec> maps;
    std::vector<BigEnt> linesets; ///< SampleGuestBytes line sets (va, size), expanded later
    u32 other_signals{};

    FrameLog() {
        used.reserve(LOG_PAGES);
        bigs.reserve(1024);
        pools.reserve(1024);
        epochs.reserve(256);
    }
    void Clear() {
        for (const u32 slot : used) {
            table[slot] = PageEnt{};
        }
        used.clear();
        masks_used = 0;
        bigs.clear();
        pools.clear();
        calls.fill(0);
        bytes.fill(0);
        overflow_pages = overflow_masks = 0;
        wait_value = wait_address = tsc_pass = tsc_end = tsc_block = block_seq = tsc_eventcpu = 0;
        tsc_wb = tsc_wake = tsc_signal = signal_value = 0;
        wake_tid = signal_tid = 0;
        end_blocked = false;
        nops = 0;
        ops_total = 0;
        op_hist.fill(0);
        present_begin = 0;
        epochs.clear();
        wcalls.clear();
        wcall_tids.clear();
        maps.clear();
        linesets.clear();
        other_signals = 0;
    }
    PageEnt* Insert(u64 page) {
        const size_t mask = table.size() - 1;
        size_t slot = static_cast<size_t>(HashPage(page)) & mask;
        for (size_t probes = 0; probes < table.size(); ++probes) {
            PageEnt& e = table[slot];
            if (e.page == page) {
                return &e;
            }
            if (e.page == EMPTY) {
                if (used.size() >= table.size() * 3 / 4) {
                    ++overflow_pages;
                    return nullptr;
                }
                e.page = page;
                used.push_back(static_cast<u32>(slot));
                return &e;
            }
            slot = (slot + 1) & mask;
        }
        ++overflow_pages;
        return nullptr;
    }
    void AddSmall(Cls cls, u64 begin, u64 end, u64 id) {
        const u32 bit = Bit(cls);
        const bool full_pages = IsPageClass(cls);
        const bool line_gran = (bit & LINE_GRAN) != 0;
        for (u64 page = begin / PAGE; page * PAGE < end; ++page) {
            PageEnt* e = Insert(page);
            if (e == nullptr) {
                return;
            }
            if (e->cls == 0) {
                e->id = id;
            }
            e->cls |= bit;
            if (e->bm == FULL) {
                continue;
            }
            const u64 pva = page * PAGE;
            if (full_pages || (begin <= pva && end >= pva + PAGE)) {
                e->bm = FULL;
                e->lines = ~u64{0};
                continue;
            }
            const u64 a = (std::max)(begin, pva);
            const u64 b = (std::min)(end, pva + PAGE);
            if (line_gran || e->bm == LINEMODE) {
                if (e->bm == 0) {
                    e->bm = LINEMODE;
                }
                for (u64 line = (a - pva) / LINE; pva + line * LINE < b; ++line) {
                    e->lines |= u64{1} << line;
                    if (e->bm != LINEMODE) {
                        masks[e->bm - 1][line] = ~u64{0};
                    }
                }
                continue;
            }
            if (e->bm == 0) {
                if (masks_used >= masks.size()) {
                    ++overflow_masks;
                    e->bm = LINEMODE;
                    for (u64 line = (a - pva) / LINE; pva + line * LINE < b; ++line) {
                        e->lines |= u64{1} << line;
                    }
                    continue;
                }
                masks[masks_used].fill(0);
                e->bm = ++masks_used;
            }
            auto& m = masks[e->bm - 1];
            for (u64 line = (a - pva) / LINE; pva + line * LINE < b; ++line) {
                const u64 bits = LineBytes(pva + line * LINE, a, b);
                if (bits != 0) {
                    m[line] |= bits;
                    e->lines |= u64{1} << line;
                }
            }
        }
    }
    void AddBig(Cls cls, u64 begin, u64 end, u64 id) {
        const u32 bit = Bit(cls);
        for (BigEnt& b : bigs) {
            if (b.va == begin && b.size == end - begin) {
                b.cls |= bit;
                return;
            }
        }
        if (bigs.size() < 4096) {
            bigs.push_back({begin, end - begin, id, bit});
        }
    }
    [[nodiscard]] u64 ByteMask(const PageEnt& e, u64 line) const {
        if (((e.lines >> line) & 1) == 0) {
            return 0;
        }
        return (e.bm == FULL || e.bm == LINEMODE) ? ~u64{0} : masks[e.bm - 1][line];
    }
};

struct ChangeRun {
    u64 begin;
    u64 end;
    u64 ndiff;
};

struct SnapPage {
    u64 page{EMPTY};
    u64 lines{};
    u64 off{};
    std::array<u64, STAGES> changed{};
    u32 diff{};
    u32 unreadable{};
};

struct SnapBig {
    u64 va{};
    u64 size{};
    u64 off{};
    u64 id{};
    u32 cls{};
    std::array<std::vector<ChangeRun>, STAGES> runs;
};

/// Bytes of one tier of the predicted read set, compared (and refreshed) at each stage.
struct Snap {
    std::vector<SnapPage> table = std::vector<SnapPage>(SNAP_PAGES);
    std::vector<u32> used;
    std::vector<u8> buffer;
    std::vector<std::array<std::array<u64, 64>, STAGES>> diffs;
    std::vector<SnapBig> bigs;
    int tier{};
    u64 page_bytes{};
    u64 big_bytes{};
    u64 skipped_bytes{};
    u64 overflow{};
    u64 diffs_overflow{};
    u64 trigger_tsc{};
    u64 snap_begin{};
    u64 snap_end{};
    std::array<u64, STAGES> diff_begin{};
    std::array<u64, STAGES> diff_end{};
    u64 signal_tsc{};
    bool signaled{};
    bool live{};

    Snap() {
        used.reserve(SNAP_PAGES);
        diffs.reserve(SNAP_DIFFS);
    }
    void Clear() {
        for (const u32 slot : used) {
            table[slot] = SnapPage{};
        }
        used.clear();
        diffs.clear();
        bigs.clear();
        page_bytes = big_bytes = skipped_bytes = overflow = diffs_overflow = 0;
        trigger_tsc = snap_begin = snap_end = signal_tsc = 0;
        diff_begin.fill(0);
        diff_end.fill(0);
        signaled = false;
        live = false;
    }
    SnapPage* Find(u64 page) {
        const size_t mask = table.size() - 1;
        size_t slot = static_cast<size_t>(HashPage(page)) & mask;
        for (size_t probes = 0; probes < table.size(); ++probes) {
            SnapPage& e = table[slot];
            if (e.page == page) {
                return &e;
            }
            if (e.page == EMPTY) {
                return nullptr;
            }
            slot = (slot + 1) & mask;
        }
        return nullptr;
    }
    SnapPage* Insert(u64 page) {
        const size_t mask = table.size() - 1;
        size_t slot = static_cast<size_t>(HashPage(page)) & mask;
        for (size_t probes = 0; probes < table.size(); ++probes) {
            SnapPage& e = table[slot];
            if (e.page == page) {
                return &e;
            }
            if (e.page == EMPTY) {
                if (used.size() >= table.size() * 3 / 4) {
                    ++overflow;
                    return nullptr;
                }
                e.page = page;
                used.push_back(static_cast<u32>(slot));
                return &e;
            }
            slot = (slot + 1) & mask;
        }
        ++overflow;
        return nullptr;
    }
};

struct Hit {
    u64 frame;
    u64 present;
    u64 va;
    u64 len;
    u64 ndiff;
    u64 id;
    u32 cls;
    u32 stage;
};

struct EpochHit {
    u64 frame;
    u64 present;
    u64 serial;
    u64 va;
    u64 size;
    u64 b;
    u64 c;
    u64 r_bytes;
    u32 r_cls;
    Cls cls;
};

struct FrameRow {
    u64 index{};
    u64 value{};
    u64 address{};
    u64 present{};
    u64 tsc_pass{};
    u64 tsc_end{};
    u64 tsc_block{};
    u64 block_seq{};
    u64 tsc_eventcpu{};
    u64 tsc_wb{};
    u64 tsc_wake{};
    u64 tsc_signal{};
    u32 wake_tid{};
    u32 signal_tid{};
    std::array<u64, 2> snap_begin{};
    std::array<u64, 2> snap_end{};
    std::array<std::array<u64, STAGES>, 2> diff_begin{};
    std::array<std::array<u64, STAGES>, 2> diff_end{};
    std::array<u64, 2> snap_bytes{};
    std::array<u64, 2> snap_skipped{};
    std::array<u64, 2> r_bytes{};
    std::array<u64, 2> r_covered{};
    std::array<std::array<u64, STAGES>, 2> hit_bytes{};
    u64 decision_pools{};
    u64 decision_bytes{};
    std::array<u64, 4> epoch_pages{};    ///< A B C D over all entries
    std::array<u64, 4> epoch_pages_r{};  ///< ... entries whose pool this frame's decisions read
    u64 overflow{};
    bool analyzed{};
    bool end_blocked{};
    u64 ops_total{};
    u32 wait_events{};
    u32 signal_events{};
    u32 fence_syncs{};
    u32 draws{};
    u32 other_signals{};
    std::array<u16, MAX_OPS> ops{};
    u32 nops{};
    std::vector<std::pair<const void*, u32>> wcalls;
    std::vector<std::pair<u32, u32>> wcall_tids;
    std::vector<MapRec> maps;
};

enum class JobKind { Snapshot, Block, Wake, Signal, Analyze, Dump };

struct Job {
    JobKind kind;
    u64 tsc{};
    FrameLog* flog{};
};

struct ClassTotals {
    u64 calls{};
    u64 raw_bytes{};
    u64 r_bytes{};
    u64 covered{};
    std::array<u64, STAGES> hit_bytes{};
    std::array<u64, STAGES> hit_frames{};
    u64 frames_read{};
    u64 outside_calls{};
    u64 outside_bytes{};
};

class Probe {
public:
    Probe() {
        for (auto& flog : logs) {
            flog = std::make_unique<FrameLog>();
            free_logs.push_back(flog.get());
        }
        for (auto& snap : snaps) {
            snap = std::make_unique<Snap>();
        }
        frames.reserve(MAX_FRAMES);
        hits.reserve(MAX_HITS);
        ehits.reserve(MAX_EHITS);
        epochs.reserve(MAX_EPOCHS);
        for (auto& s : samples) {
            s.assign(MAX_EPOCHS, 0);
        }
        if (const char* w = std::getenv("EDEN_NX2_NVN_PREREC_WATCH"); w != nullptr && *w != 0) {
            unsigned long long va = 0, size = 0;
            if (std::sscanf(w, "%llx:%llu", &va, &size) == 2 && size <= (u64{1} << 20)) {
                watch_va = va;
                watch_size = size;
            }
        }
        thread = std::thread([this] { Run(); });
        fast_thread = std::thread([this] { RunFast(); });
    }

    // ---- read path (replay thread, mostly) ----
    void NoteRead(Cls cls, u64 va, u64 size, u64 id) {
        if (size == 0) {
            return;
        }
        Lock();
        FrameLog* const flog = cur;
        const size_t c = static_cast<size_t>(cls);
        if (flog == nullptr) {
            outside_calls[c] += 1;
            outside_bytes[c] += size;
            Unlock();
            return;
        }
        flog->calls[c] += 1;
        flog->bytes[c] += size;
        if ((Bit(cls) & DECISION) != 0) {
            auto& slot = pool_cache[static_cast<size_t>(id * 0x9E3779B97F4A7C15ull >> 54) &
                                    (pool_cache.size() - 1)];
            if (slot.gen == gen && slot.id == id && slot.cls == c) {
                Unlock();
                return;
            }
            slot = {id, c, gen};
            PoolUse& use = flog->pools[id];
            use.cls |= Bit(cls);
            use.bytes = (std::max)(use.bytes, size);
            ++use.calls;
            Unlock();
            return;
        }
        const u64 key = va ^ (size * 0x9E3779B97F4A7C15ull) ^ (u64{c} << 58);
        auto& slot = dedupe[static_cast<size_t>((key >> 7) ^ key) & (dedupe.size() - 1)];
        if (slot.gen == gen && slot.va == va && slot.size == size && slot.cls == c) {
            Unlock();
            return;
        }
        slot = {va, size, c, gen};
        u64 begin = va;
        u64 end = va + size;
        if (IsPageClass(cls)) {
            begin = begin / PAGE * PAGE;
            end = (end + PAGE - 1) / PAGE * PAGE;
        }
        if (end - begin >= BIG) {
            flog->AddBig(cls, begin, end, id);
        } else {
            flog->AddSmall(cls, begin, end, id);
        }
        Unlock();
    }

    void NoteLineSet(Cls cls, u64 va, u64 size, u64 id) {
        Lock();
        if (FrameLog* const flog = cur) {
            flog->calls[static_cast<size_t>(cls)] += 1;
            flog->bytes[static_cast<size_t>(cls)] += 256 * LINE;
            if (flog->linesets.size() < 65536) {
                flog->linesets.push_back({va, size, id, Bit(cls)});
            }
        } else {
            outside_calls[static_cast<size_t>(cls)] += 1;
        }
        Unlock();
    }

    void NoteOp(u16 op) {
        if (FrameLog* const flog = cur) {
            ++flog->ops_total;
            if (op < flog->op_hist.size()) {
                ++flog->op_hist[op];
            }
            if (flog->nops < MAX_OPS) {
                flog->ops[flog->nops++] = op;
            }
        } else {
            ++ops_outside;
            if (op == 51 || op == 52) {
                ++draws_outside;
            }
        }
    }

    void RegisterEpoch(u64 va, u64 size, u64 serial, Cls cls,
                       std::shared_ptr<const std::atomic<u64>> epoch) {
        if (!epoch) {
            return;
        }
        std::scoped_lock lk{epoch_mutex};
        if (epochs.size() >= MAX_EPOCHS) {
            ++epochs_dropped;
            return;
        }
        epochs.push_back({va, size, serial, cls, std::move(epoch)});
        epoch_count.store(epochs.size(), std::memory_order_release);
    }

    // ---- replay events ----
    void OnWaitBlocked(u64 address, u32 value, u64 seq) {
        std::scoped_lock lk{state_mutex};
        if (address != last_block_address || value != last_block_value) {
            last_block_address = address;
            last_block_value = value;
            ++block_counts[address];
            ++blocks_seen;
        }
        if (!learned) {
            Learn();
            return;
        }
        if (!gating.contains(address)) {
            ++other_blocks;
            return;
        }
        if (replay_state == 2) {
            return; // evaluated again while suspended
        }
        const u64 tsc = Now();
        replay_state = 2;
        block_tsc = tsc;
        block_seq = seq;
        block_value = value;
        Sample(SAMPLE_BLOCK);
        Lock();
        FrameLog* const ended = cur;
        cur = nullptr;
        Unlock();
        Push(Job{JobKind::Block, tsc});
        if (ended != nullptr) {
            ended->tsc_end = tsc;
            ended->end_blocked = true;
            Push(Job{JobKind::Analyze, tsc, ended});
        }
    }

    void OnWaitPassed(u64 address, u32 value) {
        std::scoped_lock lk{state_mutex};
        if (!learned || !gating.contains(address)) {
            return;
        }
        const u64 tsc = Now();
        Sample(SAMPLE_PASS);
        Lock();
        FrameLog* const ended = cur;
        cur = nullptr;
        Unlock();
        if (ended != nullptr) {
            ended->tsc_end = tsc;
            ended->end_blocked = false;
            Push(Job{JobKind::Analyze, tsc, ended});
        }
        FrameLog* next = nullptr;
        {
            std::scoped_lock fl{free_mutex};
            if (!free_logs.empty()) {
                next = free_logs.back();
                free_logs.pop_back();
            }
        }
        const bool blocked = replay_state == 2;
        if (next == nullptr) {
            ++frames_dropped;
        } else {
            next->Clear();
            next->wait_value = value;
            next->wait_address = address;
            next->tsc_pass = tsc;
            next->present_begin = present.load(std::memory_order_relaxed);
            if (blocked) {
                next->tsc_block = block_tsc;
                next->block_seq = block_seq;
            }
            next->tsc_eventcpu = eventcpu_tsc;
            next->tsc_wb = game_wb;
            next->tsc_wake = game_wake;
            next->tsc_signal = game_signal;
            next->wake_tid = game_wake_tid;
            next->signal_tid = game_signal_tid;
            next->signal_value = game_signal_value;
            next->other_signals = game_other_signals;
            next->wcalls = std::move(game_calls);
            next->wcall_tids = std::move(game_call_tids);
            next->maps = std::move(game_maps);
            // Epoch movement around this wait.
            const size_t n = (std::min)(epoch_count.load(std::memory_order_acquire), MAX_EPOCHS);
            const bool have = blocked && have_prev_pass && game_signal > block_tsc;
            if (have) {
                auto& sp = samples[SAMPLE_PREV_PASS];
                auto& sb = samples[SAMPLE_BLOCK];
                auto& sw = samples[SAMPLE_WAKE];
                auto& ss = samples[SAMPLE_SIGNAL];
                auto& sx = samples[SAMPLE_PASS];
                const bool wake_inside = game_wake > block_tsc && game_wake < game_signal;
                for (size_t i = 0; i < n; ++i) {
                    EpochDelta d{static_cast<u32>(i), sb[i] - sp[i],
                                 wake_inside ? sw[i] - sb[i] : 0,
                                 wake_inside ? ss[i] - sw[i] : ss[i] - sb[i], sx[i] - ss[i]};
                    if (d.a != 0 || d.b != 0 || d.c != 0 || d.d != 0) {
                        next->epochs.push_back(d);
                    }
                }
            }
        }
        game_calls.clear();
        game_call_tids.clear();
        game_maps.clear();
        game_other_signals = 0;
        samples[SAMPLE_PREV_PASS] = samples[SAMPLE_PASS];
        have_prev_pass = true;
        replay_state = 1;
        Lock();
        cur = next;
        ++gen;
        Unlock();
    }

    void OnEventCpu(u64 address) {
        std::scoped_lock lk{state_mutex};
        ++eventcpu_counts[address];
        if (gating.contains(address)) {
            eventcpu_tsc = Now();
        }
    }

    // ---- game events ----
    void OnSyncWaitBegin(bool blocking, u32 tid) {
        if (!blocking) {
            return;
        }
        std::scoped_lock lk{state_mutex};
        const u64 tsc = Now();
        game_wb = tsc;
        wb_tid = tid;
        wb_open = true;
        maps_open.store(true, std::memory_order_relaxed);
    }

    void OnSyncWaitEnd(u32 tid) {
        std::scoped_lock lk{state_mutex};
        if (!wb_open || tid != wb_tid) {
            return;
        }
        wb_open = false;
        const u64 tsc = Now();
        game_wake = tsc;
        game_wake_tid = tid;
        Sample(SAMPLE_WAKE);
        in_wprime.store(true, std::memory_order_relaxed);
        Push(Job{JobKind::Wake, tsc});
    }

    void NoteSignalTid(u32 tid) {
        signal_tid.store(tid, std::memory_order_relaxed);
    }

    void OnEventSignal(u64 address, u32 written) {
        std::scoped_lock lk{state_mutex};
        ++cpu_signal_counts[address];
        if (!learned || !gating.contains(address)) {
            ++game_other_signals;
            return;
        }
        const u64 tsc = Now();
        Sample(SAMPLE_SIGNAL);
        game_signal = tsc;
        game_signal_value = written;
        game_signal_tid = signal_tid.load(std::memory_order_relaxed);
        in_wprime.store(false, std::memory_order_relaxed);
        maps_open.store(false, std::memory_order_relaxed);
        Push(Job{JobKind::Signal, tsc});
    }

    void OnGameCall(const void* handler, u32 tid) {
        if (!in_wprime.load(std::memory_order_relaxed)) {
            return;
        }
        std::scoped_lock lk{state_mutex};
        if (!in_wprime.load(std::memory_order_relaxed)) {
            return;
        }
        ++game_calls[handler];
        ++game_call_tids[tid];
    }

    void OnMappedRange(bool flush, u64 va, u64 size, u32 tid) {
        if (!maps_open.load(std::memory_order_relaxed)) {
            return;
        }
        std::scoped_lock lk{state_mutex};
        if (game_maps.size() < 1024) {
            game_maps.push_back({Now(), va, size, tid, flush});
        }
    }

    void NameHandler(const void* handler, std::string_view name) {
        std::scoped_lock lk{names_mutex};
        names.emplace(handler, std::string{name});
    }

    void SetGuest(GuestMemory* memory) {
        std::unique_lock lk{guest_mutex};
        guest = memory;
    }

    void OnPresent(u64 count) {
        present.store(count, std::memory_order_relaxed);
        if (count == Common::TlRing::Detail::config.last && !dump_asked.exchange(true)) {
            Push(Job{JobKind::Dump, Now()});
        }
    }

private:
    enum SampleSlot { SAMPLE_PREV_PASS, SAMPLE_BLOCK, SAMPLE_WAKE, SAMPLE_SIGNAL, SAMPLE_PASS, NSAMPLES };

    void Sample(SampleSlot slot) {
        const size_t n = (std::min)(epoch_count.load(std::memory_order_acquire), MAX_EPOCHS);
        std::scoped_lock lk{epoch_mutex};
        auto& s = samples[slot];
        for (size_t i = 0; i < n && i < epochs.size(); ++i) {
            s[i] = epochs[i].epoch->load(std::memory_order_acquire);
        }
    }

    void Learn() {
        if (blocks_seen < 24 || cpu_signal_counts.empty()) {
            return;
        }
        for (const auto& [a, n] : block_counts) {
            if (cpu_signal_counts.contains(a)) {
                gating.insert(a);
            }
        }
        learned = !gating.empty();
    }

    void Lock() {
        while (spin.test_and_set(std::memory_order_acquire)) {
            _mm_pause();
        }
    }
    void Unlock() {
        spin.clear(std::memory_order_release);
    }
    void Push(Job job) {
        const bool fast = job.kind == JobKind::Block || job.kind == JobKind::Wake ||
                          job.kind == JobKind::Signal;
        const bool slow = job.kind != JobKind::Wake;
        {
            std::scoped_lock lk{job_mutex};
            if (fast) {
                fast_jobs.push_back(job);
            }
            if (slow) {
                jobs.push_back(job);
            }
        }
        if (fast) {
            fast_cv.notify_one();
        }
        if (slow) {
            job_cv.notify_one();
        }
    }

    void PollEpochs() {
        if (!Common::TlRing::Armed() || dumped.load(std::memory_order_acquire)) {
            return;
        }
        const size_t n = (std::min)(epoch_count.load(std::memory_order_acquire), MAX_EPOCHS);
        std::scoped_lock lk{epoch_mutex};
        if (poll_last.size() < n) {
            const size_t old = poll_last.size();
            poll_last.resize(n);
            for (size_t i = old; i < n; ++i) {
                poll_last[i] = epochs[i].epoch->load(std::memory_order_acquire);
            }
        }
        const u64 tsc = Now();
        for (size_t i = 0; i < n; ++i) {
            const u64 v = epochs[i].epoch->load(std::memory_order_acquire);
            if (v != poll_last[i]) {
                if (epoch_events.size() < (size_t{1} << 20)) {
                    epoch_events.push_back({tsc, static_cast<u32>(i), v - poll_last[i]});
                }
                poll_last[i] = v;
            }
        }
        last_poll_tsc = tsc;
        PollWatch(tsc);
    }

    /// EDEN_NX2_NVN_PREREC_WATCH=<va hex>:<size>: a guest range compared with its last copy at
    /// every poll (~100 us); each change is logged with its time (when a hit range is written).
    void PollWatch(u64 tsc) {
        if (watch_size == 0 || guest == nullptr) {
            return;
        }
        if (watch_copy.size() != watch_size) {
            watch_copy.assign(watch_size, 0);
            ForEachPiece(watch_va, watch_size, [&](u64 va, u64 len, const u8* src) {
                if (src != nullptr) {
                    std::memcpy(watch_copy.data() + (va - watch_va), src, len);
                }
            });
            return;
        }
        ForEachPiece(watch_va, watch_size, [&](u64 va, u64 len, const u8* src) {
            if (src == nullptr) {
                return;
            }
            u8* dst = watch_copy.data() + (va - watch_va);
            if (std::memcmp(src, dst, len) == 0) {
                return;
            }
            u64 first = ~u64{0};
            u64 last = 0;
            u64 nd = 0;
            for (u64 i = 0; i < len; ++i) {
                if (src[i] != dst[i]) {
                    first = (std::min)(first, i);
                    last = i;
                    ++nd;
                }
            }
            if (watch_events.size() < (size_t{1} << 18)) {
                watch_events.push_back({tsc, va - watch_va + first, last - first + 1, nd});
            }
            std::memcpy(dst, src, len);
        });
    }

    void RunFast() {
        for (;;) {
            Job job;
            {
                std::unique_lock lk{job_mutex};
                // Between jobs: poll the tracker write epochs every ~100 us and log each movement
                // with its time (when CPU writes happen, independent of the replay's timing).
                while (fast_jobs.empty()) {
                    lk.unlock();
                    PollEpochs();
                    lk.lock();
                    if (fast_jobs.empty()) {
                        fast_cv.wait_for(lk, std::chrono::microseconds{100});
                    }
                }
                job = fast_jobs.front();
                fast_jobs.pop_front();
            }
            PollEpochs();
            if (dumped.load(std::memory_order_acquire)) {
                continue;
            }
            std::shared_lock gl{guest_mutex};
            switch (job.kind) {
            case JobKind::Block:
                if (Detail::mode == 1) {
                    BuildSnapshot(0, job.tsc);
                } else if (current[0] != nullptr) {
                    Diff(*current[0], 0);
                }
                break;
            case JobKind::Wake:
                if (current[0] != nullptr) {
                    Diff(*current[0], 1);
                }
                break;
            case JobKind::Signal:
                if (current[0] != nullptr) {
                    Diff(*current[0], 2);
                    std::scoped_lock sl{snap_mutex};
                    current[0]->signaled = true;
                    current[0]->signal_tsc = job.tsc;
                    current[0] = nullptr;
                }
                if (Detail::mode == 3) {
                    BuildSnapshot(0, Now());
                }
                break;
            default:
                break;
            }
        }
    }

    void Run() {
        for (;;) {
            Job job;
            {
                std::unique_lock lk{job_mutex};
                job_cv.wait(lk, [this] { return !jobs.empty(); });
                job = jobs.front();
                jobs.pop_front();
                max_backlog = (std::max)(max_backlog, static_cast<u64>(jobs.size()));
            }
            if (dumped.load(std::memory_order_acquire)) {
                if (job.kind == JobKind::Analyze && job.flog) {
                    ReleaseLog(job.flog);
                }
                continue;
            }
            std::shared_lock gl{guest_mutex};
            switch (job.kind) {
            case JobKind::Block:
                if (current[1] != nullptr) {
                    Diff(*current[1], 0);
                }
                break;
            case JobKind::Signal:
                if (current[1] != nullptr) {
                    Diff(*current[1], 2);
                    std::scoped_lock sl{snap_mutex};
                    current[1]->signaled = true;
                    current[1]->signal_tsc = job.tsc;
                    current[1] = nullptr;
                }
                // Tier 2 is too large to copy inside W: its window starts here (a superset of
                // the next W; stage 0 separates what changed before the next suspension).
                BuildSnapshot(1, Now());
                break;
            case JobKind::Analyze:
                Analyze(*job.flog);
                break;
            case JobKind::Dump:
                Dump();
                dumped.store(true, std::memory_order_release);
                break;
            default:
                break;
            }
        }
    }

    void ReleaseLog(FrameLog* flog) {
        std::scoped_lock fl{free_mutex};
        free_logs.push_back(flog);
    }

    Snap* FreeSnap(int tier) {
        std::scoped_lock sl{snap_mutex};
        for (size_t i = static_cast<size_t>(tier) * 3; i < static_cast<size_t>(tier) * 3 + 3; ++i) {
            if (!snaps[i]->live) {
                snaps[i]->live = true;
                return snaps[i].get();
            }
        }
        ++snaps_exhausted;
        return nullptr;
    }

    void BuildSnapshot(int tier, u64 trigger) {
        if (current[tier] != nullptr) {
            std::scoped_lock sl{snap_mutex};
            current[tier]->live = false; // never signaled
            current[tier] = nullptr;
        }
        std::scoped_lock hl{history_mutex};
        if (history.empty() || guest == nullptr) {
            return;
        }
        Snap* const s = FreeSnap(tier);
        if (s == nullptr) {
            return;
        }
        s->Clear();
        s->live = true;
        s->tier = tier;
        s->trigger_tsc = trigger;
        s->snap_begin = Now();
        const u32 want = tier == 0 ? TIER1 : TIER2;
        const u64 budget = tier == 0 ? TIER1_BUDGET : TIER2_BUDGET;
        for (const FrameLog* flog : history) {
            for (const u32 slot : flog->used) {
                const PageEnt& e = flog->table[slot];
                if ((e.cls & want) == 0) {
                    continue;
                }
                SnapPage* sp = s->Insert(e.page);
                if (sp != nullptr) {
                    sp->lines |= e.lines;
                }
            }
        }
        u64 off = 0;
        for (const u32 slot : s->used) {
            SnapPage& sp = s->table[slot];
            const u64 bytes = static_cast<u64>(std::popcount(sp.lines)) * LINE;
            if (off + bytes > budget) {
                s->skipped_bytes += bytes;
                sp.lines = 0;
                continue;
            }
            sp.off = off;
            off += bytes;
        }
        s->page_bytes = off;
        std::vector<const BigEnt*> bigs;
        for (const FrameLog* flog : history) {
            for (const BigEnt& b : flog->bigs) {
                if ((b.cls & want) == 0) {
                    continue;
                }
                bool dup = false;
                for (const BigEnt* other : bigs) {
                    if (other->va == b.va && other->size == b.size) {
                        dup = true;
                        break;
                    }
                }
                if (!dup) {
                    bigs.push_back(&b);
                }
            }
        }
        std::sort(bigs.begin(), bigs.end(),
                  [](const BigEnt* a, const BigEnt* b) { return a->size < b->size; });
        for (const BigEnt* b : bigs) {
            if (off + b->size > budget) {
                s->skipped_bytes += b->size;
                continue;
            }
            SnapBig& sb = s->bigs.emplace_back();
            sb.va = b->va;
            sb.size = b->size;
            sb.off = off;
            sb.id = b->id;
            sb.cls = b->cls;
            off += b->size;
            s->big_bytes += b->size;
        }
        std::sort(s->bigs.begin(), s->bigs.end(),
                  [](const SnapBig& a, const SnapBig& b) { return a.va < b.va; });
        if (s->buffer.size() < off) {
            s->buffer.resize(off);
        }
        for (const u32 slot : s->used) {
            SnapPage& sp = s->table[slot];
            if (sp.lines == 0) {
                continue;
            }
            const u8* src = guest->Pointer(sp.page * PAGE);
            u8* dst = s->buffer.data() + sp.off;
            if (src == nullptr) {
                sp.unreadable = 1;
                continue;
            }
            u64 lines = sp.lines;
            while (lines != 0) {
                const int line = std::countr_zero(lines);
                lines &= lines - 1;
                std::memcpy(dst, src + line * LINE, LINE);
                dst += LINE;
            }
        }
        for (SnapBig& sb : s->bigs) {
            ForEachPiece(sb.va, sb.size, [&](u64 va, u64 len, const u8* src) {
                u8* dst = s->buffer.data() + sb.off + (va - sb.va);
                if (src != nullptr) {
                    std::memcpy(dst, src, len);
                } else {
                    std::memset(dst, 0, len);
                }
            });
        }
        s->snap_end = Now();
        current[tier] = s;
    }

    template <typename F>
    void ForEachPiece(u64 va, u64 size, F&& f) {
        u64 at = va;
        const u64 end = va + size;
        while (at < end) {
            const u64 page_end = (at / PAGE + 1) * PAGE;
            const u64 len = (std::min)(page_end, end) - at;
            f(at, len, guest->Pointer(at));
            at += len;
        }
    }

    void Diff(Snap& s, int stage) {
        if (guest == nullptr) {
            return;
        }
        s.diff_begin[stage] = Now();
        for (const u32 slot : s.used) {
            SnapPage& sp = s.table[slot];
            if (sp.unreadable != 0 || sp.lines == 0) {
                continue;
            }
            const u8* cur_page = guest->Pointer(sp.page * PAGE);
            if (cur_page == nullptr) {
                continue;
            }
            u8* snap_line = s.buffer.data() + sp.off;
            u64 lines = sp.lines;
            while (lines != 0) {
                const int line = std::countr_zero(lines);
                lines &= lines - 1;
                const u8* now = cur_page + line * LINE;
                if (std::memcmp(now, snap_line, LINE) != 0) {
                    const u64 dm = DiffMask(now, snap_line);
                    sp.changed[stage] |= u64{1} << line;
                    if (sp.diff == 0) {
                        if (s.diffs.size() < SNAP_DIFFS) {
                            s.diffs.emplace_back();
                            for (auto& st : s.diffs.back()) {
                                st.fill(0);
                            }
                            sp.diff = static_cast<u32>(s.diffs.size());
                        } else {
                            ++s.diffs_overflow;
                        }
                    }
                    if (sp.diff != 0) {
                        s.diffs[sp.diff - 1][stage][line] |= dm;
                    }
                    std::memcpy(snap_line, now, LINE);
                }
                snap_line += LINE;
            }
        }
        for (SnapBig& sb : s.bigs) {
            auto& runs = sb.runs[stage];
            ForEachPiece(sb.va, sb.size, [&](u64 va, u64 len, const u8* src) {
                if (src == nullptr) {
                    return;
                }
                u8* dst = s.buffer.data() + sb.off + (va - sb.va);
                if (std::memcmp(src, dst, len) == 0) {
                    return;
                }
                for (u64 i = 0; i < len; i += LINE) {
                    const u64 n = (std::min)(LINE, len - i);
                    if (std::memcmp(src + i, dst + i, n) == 0) {
                        continue;
                    }
                    u64 nd = 0;
                    for (u64 j = 0; j < n; ++j) {
                        nd += src[i + j] != dst[i + j] ? 1 : 0;
                    }
                    const u64 b = va + i;
                    if (!runs.empty() && runs.back().end == b) {
                        runs.back().end = b + n;
                        runs.back().ndiff += nd;
                    } else {
                        runs.push_back({b, b + n, nd});
                    }
                    std::memcpy(dst + i, src + i, n);
                }
            });
        }
        s.diff_end[stage] = Now();
    }

    static u64 BigRunBytes(const Snap& s, int stage, u64 begin, u64 end) {
        u64 total = 0;
        for (const SnapBig& sb : s.bigs) {
            if (sb.va >= end || sb.va + sb.size <= begin) {
                continue;
            }
            for (const ChangeRun& r : sb.runs[stage]) {
                const u64 a = (std::max)(r.begin, begin);
                const u64 b = (std::min)(r.end, end);
                if (a < b) {
                    total += (r.ndiff * (b - a) + (r.end - r.begin) - 1) / (r.end - r.begin);
                }
            }
        }
        return total;
    }
    static u64 BigCovered(const Snap& s, u64 begin, u64 end) {
        u64 covered = 0;
        u64 at = begin;
        for (const SnapBig& sb : s.bigs) {
            if (sb.va + sb.size <= at) {
                continue;
            }
            if (sb.va >= end) {
                break;
            }
            const u64 a = (std::max)(sb.va, at);
            const u64 b = (std::min)(sb.va + sb.size, end);
            if (a < b) {
                covered += b - a;
                at = b;
            }
        }
        return covered;
    }

    /// Content analysis of one tier of the frame's reads against its snapshot.
    void AnalyzeTier(FrameLog& flog, Snap* s, int tier, FrameRow& row,
                     std::array<u64, NCLS>& cls_r, std::array<u64, NCLS>& cls_cov,
                     std::array<std::array<u64, STAGES>, NCLS>& cls_hit) {
        const u32 want = tier == 0 ? TIER1 : TIER2;
        for (const u32 slot : flog.used) {
            const PageEnt& e = flog.table[slot];
            const u32 ecls = e.cls & want;
            if (ecls == 0) {
                continue;
            }
            const SnapPage* sp = s ? s->Find(e.page) : nullptr;
            const u64 pva = e.page * PAGE;
            const bool big_page = s != nullptr && sp == nullptr && !s->bigs.empty() &&
                                  BigCovered(*s, pva, pva + PAGE) >= PAGE;
            u64 r = 0;
            u64 cov = 0;
            std::array<u64, STAGES> hit{};
            std::array<u64, STAGES> first{};
            std::array<u64, STAGES> last{};
            u64 lines = e.lines;
            while (lines != 0) {
                const int line = std::countr_zero(lines);
                lines &= lines - 1;
                const u64 rm = flog.ByteMask(e, static_cast<u64>(line));
                const u64 rb = static_cast<u64>(std::popcount(rm));
                r += rb;
                if (s == nullptr) {
                    continue;
                }
                const u64 line_va = pva + static_cast<u64>(line) * LINE;
                if (sp != nullptr && ((sp->lines >> line) & 1) != 0) {
                    cov += rb;
                    for (int st = 0; st < STAGES; ++st) {
                        if (((sp->changed[st] >> line) & 1) == 0) {
                            continue;
                        }
                        const u64 dm = sp->diff != 0 ? s->diffs[sp->diff - 1][st][line] : ~u64{0};
                        const u64 n = static_cast<u64>(std::popcount(rm & dm));
                        if (n != 0) {
                            if (hit[st] == 0) {
                                first[st] = line_va;
                            }
                            hit[st] += n;
                            last[st] = line_va + LINE;
                        }
                    }
                } else if (big_page) {
                    cov += rb;
                    for (int st = 0; st < STAGES; ++st) {
                        const u64 n = BigRunBytes(*s, st, line_va, line_va + LINE);
                        if (n != 0) {
                            if (hit[st] == 0) {
                                first[st] = line_va;
                            }
                            hit[st] += (std::min)(n, rb);
                            last[st] = line_va + LINE;
                        }
                    }
                }
            }
            row.r_bytes[tier] += r;
            row.r_covered[tier] += cov;
            for (size_t c = 0; c < NCLS; ++c) {
                if ((ecls >> c) & 1) {
                    cls_r[c] += r;
                    cls_cov[c] += cov;
                    for (int st = 0; st < STAGES; ++st) {
                        cls_hit[c][st] += hit[st];
                    }
                }
            }
            for (int st = 0; st < STAGES; ++st) {
                row.hit_bytes[tier][st] += hit[st];
                if (hit[st] != 0 && hits.size() < MAX_HITS) {
                    hits.push_back({row.index, flog.present_begin, first[st], last[st] - first[st],
                                    hit[st], e.id, ecls, static_cast<u32>(st)});
                }
            }
        }
        for (const BigEnt& b : flog.bigs) {
            const u32 bcls = b.cls & want;
            if (bcls == 0) {
                continue;
            }
            u64 cov = 0;
            std::array<u64, STAGES> hit{};
            if (s != nullptr) {
                cov = BigCovered(*s, b.va, b.va + b.size);
                for (int st = 0; st < STAGES; ++st) {
                    hit[st] = BigRunBytes(*s, st, b.va, b.va + b.size);
                }
            }
            row.r_bytes[tier] += b.size;
            row.r_covered[tier] += cov;
            for (size_t c = 0; c < NCLS; ++c) {
                if ((bcls >> c) & 1) {
                    cls_r[c] += b.size;
                    cls_cov[c] += cov;
                    for (int st = 0; st < STAGES; ++st) {
                        cls_hit[c][st] += hit[st];
                    }
                }
            }
            for (int st = 0; st < STAGES; ++st) {
                row.hit_bytes[tier][st] += hit[st];
                if (hit[st] != 0 && hits.size() < MAX_HITS) {
                    hits.push_back({row.index, flog.present_begin, b.va, b.size, hit[st], b.id, bcls,
                                    static_cast<u32>(st) | 0x100u});
                }
            }
        }
    }

    void Analyze(FrameLog& flog) {
        // The signature lines, as SampleGuestBytes reads them (expanded here, off the replay).
        for (const BigEnt& ls : flog.linesets) {
            const Cls cls = static_cast<Cls>(std::countr_zero(ls.cls));
            const u64 last_line = (ls.size - LINE) / LINE;
            for (u64 i = 0; i < 256; ++i) {
                const u64 va = ls.va + (last_line * i / 255) * LINE;
                flog.AddSmall(cls, va, va + LINE, ls.id);
            }
        }
        FrameRow row;
        row.index = frame_index++;
        row.value = flog.wait_value;
        row.address = flog.wait_address;
        row.present = flog.present_begin;
        row.tsc_pass = flog.tsc_pass;
        row.tsc_end = flog.tsc_end;
        row.tsc_block = flog.tsc_block;
        row.block_seq = flog.block_seq;
        row.tsc_eventcpu = flog.tsc_eventcpu;
        row.tsc_wb = flog.tsc_wb;
        row.tsc_wake = flog.tsc_wake;
        row.tsc_signal = flog.tsc_signal;
        row.wake_tid = flog.wake_tid;
        row.signal_tid = flog.signal_tid;
        row.end_blocked = flog.end_blocked;
        row.ops_total = flog.ops_total;
        row.wait_events = flog.op_hist[11];
        row.signal_events = flog.op_hist[9];
        row.fence_syncs = flog.op_hist[8];
        row.draws = flog.op_hist[51] + flog.op_hist[52];
        row.other_signals = flog.other_signals;
        row.ops = flog.ops;
        row.nops = flog.nops;
        row.overflow = flog.overflow_pages + flog.overflow_masks;
        row.wcalls.assign(flog.wcalls.begin(), flog.wcalls.end());
        row.wcall_tids.assign(flog.wcall_tids.begin(), flog.wcall_tids.end());
        row.maps = flog.maps;
        for (size_t c = 0; c < NCLS; ++c) {
            totals[c].calls += flog.calls[c];
            totals[c].raw_bytes += flog.bytes[c];
        }
        // The snapshots of this frame: signaled at or before the pass that started it.
        std::array<Snap*, 2> s{};
        std::unique_lock sel{snap_mutex};
        for (auto& snap : snaps) {
            if (snap->live && snap->signaled && snap->signal_tsc <= flog.tsc_pass + 1) {
                Snap*& slot = s[snap->tier];
                if (slot == nullptr || snap->signal_tsc > slot->signal_tsc) {
                    slot = snap.get();
                }
            }
        }
        sel.unlock();
        std::array<u64, NCLS> cls_r{};
        std::array<u64, NCLS> cls_cov{};
        std::array<std::array<u64, STAGES>, NCLS> cls_hit{};
        for (int t = 0; t < 2; ++t) {
            if (s[t] != nullptr) {
                if (t == 0) {
                    Diff(*s[t], 3);
                }
                row.snap_begin[t] = s[t]->snap_begin;
                row.snap_end[t] = s[t]->snap_end;
                row.diff_begin[t] = s[t]->diff_begin;
                row.diff_end[t] = s[t]->diff_end;
                row.snap_bytes[t] = s[t]->page_bytes + s[t]->big_bytes;
                row.snap_skipped[t] = s[t]->skipped_bytes;
            }
            AnalyzeTier(flog, s[t], t, row, cls_r, cls_cov, cls_hit);
        }
        row.analyzed = s[0] != nullptr;
        // Decisions: tracker write epochs of the pools this frame's decisions read.
        for (const auto& [serial, use] : flog.pools) {
            ++row.decision_pools;
            row.decision_bytes += use.bytes;
            for (size_t c = 0; c < NCLS; ++c) {
                if ((use.cls >> c) & 1) {
                    cls_r[c] += use.bytes;
                }
            }
        }
        {
            std::scoped_lock lk{epoch_mutex};
            for (const EpochDelta& d : flog.epochs) {
                if (d.index >= epochs.size()) {
                    continue;
                }
                const EpochEnt& e = epochs[d.index];
                row.epoch_pages[0] += d.a;
                row.epoch_pages[1] += d.b;
                row.epoch_pages[2] += d.c;
                row.epoch_pages[3] += d.d;
                const auto it = flog.pools.find(e.serial);
                const bool read = it != flog.pools.end();
                if (read) {
                    row.epoch_pages_r[0] += d.a;
                    row.epoch_pages_r[1] += d.b;
                    row.epoch_pages_r[2] += d.c;
                    row.epoch_pages_r[3] += d.d;
                }
                epoch_totals[static_cast<size_t>(e.cls)][0] += d.a;
                epoch_totals[static_cast<size_t>(e.cls)][1] += d.b;
                epoch_totals[static_cast<size_t>(e.cls)][2] += d.c;
                epoch_totals[static_cast<size_t>(e.cls)][3] += d.d;
                if (read) {
                    epoch_totals_r[static_cast<size_t>(e.cls)][0] += d.a;
                    epoch_totals_r[static_cast<size_t>(e.cls)][1] += d.b;
                    epoch_totals_r[static_cast<size_t>(e.cls)][2] += d.c;
                    epoch_totals_r[static_cast<size_t>(e.cls)][3] += d.d;
                }
                if ((d.b != 0 || d.c != 0) && ehits.size() < MAX_EHITS) {
                    ehits.push_back({row.index, flog.present_begin, e.serial, e.va, e.size, d.b,
                                     d.c, read ? it->second.bytes : 0, read ? it->second.cls : 0,
                                     e.cls});
                }
            }
        }
        if (row.analyzed) {
            ++frames_analyzed;
        }
        if (flog.tsc_block != 0) {
            ++frames_blocked;
        }
        for (size_t c = 0; c < NCLS; ++c) {
            totals[c].r_bytes += cls_r[c];
            totals[c].covered += cls_cov[c];
            if (cls_r[c] != 0) {
                ++totals[c].frames_read;
            }
            for (int st = 0; st < STAGES; ++st) {
                totals[c].hit_bytes[st] += cls_hit[c][st];
                if (cls_hit[c][st] != 0) {
                    ++totals[c].hit_frames[st];
                }
            }
        }
        if (frames.size() < MAX_FRAMES) {
            frames.push_back(std::move(row));
        }
        std::scoped_lock rl{snap_mutex};
        for (int t = 0; t < 2; ++t) {
            if (s[t] != nullptr) {
                for (auto& snap : snaps) {
                    if (snap->live && snap->signaled && snap->tier == t &&
                        snap->signal_tsc <= s[t]->signal_tsc) {
                        snap->live = false;
                    }
                }
            }
        }
        std::scoped_lock hl{history_mutex};
        history.push_back(&flog);
        if (history.size() > 3) {
            ReleaseLog(history.front());
            history.erase(history.begin());
        }
    }

    void Dump() {
        const std::string path = Common::TlRing::Detail::config.path + ".prerec.tsv";
        std::FILE* f = std::fopen(path.c_str(), "w");
        if (f == nullptr) {
            return;
        }
        std::fprintf(f, "# prerec probe v3 mode %d; frames %zu analyzed %llu blocked %llu dropped %llu "
                        "snaps exhausted %llu other blocks %llu ops outside %llu draws outside %llu "
                        "max backlog %llu epochs %zu dropped %llu\n",
                     Detail::mode, frames.size(), static_cast<unsigned long long>(frames_analyzed),
                     static_cast<unsigned long long>(frames_blocked),
                     static_cast<unsigned long long>(frames_dropped),
                     static_cast<unsigned long long>(snaps_exhausted),
                     static_cast<unsigned long long>(other_blocks),
                     static_cast<unsigned long long>(ops_outside),
                     static_cast<unsigned long long>(draws_outside),
                     static_cast<unsigned long long>(max_backlog), epochs.size(),
                     static_cast<unsigned long long>(epochs_dropped));
        {
            std::scoped_lock lk{state_mutex};
            for (const u64 a : gating) {
                std::fprintf(f, "# gating %#llx\n", static_cast<unsigned long long>(a));
            }
            for (const auto& [a, n] : block_counts) {
                std::fprintf(f, "# first-time blocks on %#llx: %llu\n",
                             static_cast<unsigned long long>(a), static_cast<unsigned long long>(n));
            }
            for (const auto& [a, n] : cpu_signal_counts) {
                std::fprintf(f, "# CPU signals (nvnEventSignal) of %#llx: %llu\n",
                             static_cast<unsigned long long>(a), static_cast<unsigned long long>(n));
            }
            for (const auto& [a, n] : eventcpu_counts) {
                std::fprintf(f, "# EventCpu commands taken for %#llx: %llu\n",
                             static_cast<unsigned long long>(a), static_cast<unsigned long long>(n));
            }
        }
        std::fprintf(f, "[classes]\nclass\tcalls\traw_bytes\tr_bytes\tcovered\thit0\thit1\thit2\thit3"
                        "\thf0\thf1\thf2\thf3\tframes_read\toutside_calls\toutside_bytes\tepA\tepB\tepC\t"
                        "epD\tepA_r\tepB_r\tepC_r\tepD_r\n");
        for (size_t c = 0; c < NCLS; ++c) {
            const ClassTotals& t = totals[c];
            std::fprintf(f, "%s\t%llu\t%llu\t%llu\t%llu", CLS_NAMES[c],
                         static_cast<unsigned long long>(t.calls),
                         static_cast<unsigned long long>(t.raw_bytes),
                         static_cast<unsigned long long>(t.r_bytes),
                         static_cast<unsigned long long>(t.covered));
            for (int st = 0; st < STAGES; ++st) {
                std::fprintf(f, "\t%llu", static_cast<unsigned long long>(t.hit_bytes[st]));
            }
            for (int st = 0; st < STAGES; ++st) {
                std::fprintf(f, "\t%llu", static_cast<unsigned long long>(t.hit_frames[st]));
            }
            std::fprintf(f, "\t%llu\t%llu\t%llu", static_cast<unsigned long long>(t.frames_read),
                         static_cast<unsigned long long>(outside_calls[c]),
                         static_cast<unsigned long long>(outside_bytes[c]));
            for (int i = 0; i < 4; ++i) {
                std::fprintf(f, "\t%llu", static_cast<unsigned long long>(epoch_totals[c][i]));
            }
            for (int i = 0; i < 4; ++i) {
                std::fprintf(f, "\t%llu", static_cast<unsigned long long>(epoch_totals_r[c][i]));
            }
            std::fprintf(f, "\n");
        }
        std::fprintf(f, "[frames]\nindex\tvalue\taddress\tpresent\tpass\tend\tblock\tblock_seq\teventcpu\t"
                        "wb\twake\tsignal\twake_tid\tsignal_tid");
        for (int t = 0; t < 2; ++t) {
            std::fprintf(f, "\tt%d_snap_begin\tt%d_snap_end\tt%d_d0b\tt%d_d0e\tt%d_d1b\tt%d_d1e\tt%d_d2b\t"
                            "t%d_d2e\tt%d_d3b\tt%d_d3e\tt%d_snap_bytes\tt%d_skipped\tt%d_r\tt%d_cov\t"
                            "t%d_h0\tt%d_h1\tt%d_h2\tt%d_h3",
                         t, t, t, t, t, t, t, t, t, t, t, t, t, t, t, t, t, t);
        }
        std::fprintf(f, "\tdec_pools\tdec_bytes\tepA\tepB\tepC\tepD\tepA_r\tepB_r\tepC_r\tepD_r\toverflow\t"
                        "analyzed\tend_blocked\tops\twait_events\tsignal_events\tfence_syncs\tdraws\t"
                        "other_signals\tfirst_ops\n");
        for (const FrameRow& r : frames) {
            std::fprintf(f, "%llu\t%llu\t%#llx\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%u\t%u",
                         static_cast<unsigned long long>(r.index),
                         static_cast<unsigned long long>(r.value),
                         static_cast<unsigned long long>(r.address),
                         static_cast<unsigned long long>(r.present),
                         static_cast<unsigned long long>(r.tsc_pass),
                         static_cast<unsigned long long>(r.tsc_end),
                         static_cast<unsigned long long>(r.tsc_block),
                         static_cast<unsigned long long>(r.block_seq),
                         static_cast<unsigned long long>(r.tsc_eventcpu),
                         static_cast<unsigned long long>(r.tsc_wb),
                         static_cast<unsigned long long>(r.tsc_wake),
                         static_cast<unsigned long long>(r.tsc_signal), r.wake_tid, r.signal_tid);
            for (int t = 0; t < 2; ++t) {
                std::fprintf(f, "\t%llu\t%llu", static_cast<unsigned long long>(r.snap_begin[t]),
                             static_cast<unsigned long long>(r.snap_end[t]));
                for (int st = 0; st < STAGES; ++st) {
                    std::fprintf(f, "\t%llu\t%llu",
                                 static_cast<unsigned long long>(r.diff_begin[t][st]),
                                 static_cast<unsigned long long>(r.diff_end[t][st]));
                }
                std::fprintf(f, "\t%llu\t%llu\t%llu\t%llu",
                             static_cast<unsigned long long>(r.snap_bytes[t]),
                             static_cast<unsigned long long>(r.snap_skipped[t]),
                             static_cast<unsigned long long>(r.r_bytes[t]),
                             static_cast<unsigned long long>(r.r_covered[t]));
                for (int st = 0; st < STAGES; ++st) {
                    std::fprintf(f, "\t%llu", static_cast<unsigned long long>(r.hit_bytes[t][st]));
                }
            }
            std::fprintf(f, "\t%llu\t%llu", static_cast<unsigned long long>(r.decision_pools),
                         static_cast<unsigned long long>(r.decision_bytes));
            for (int i = 0; i < 4; ++i) {
                std::fprintf(f, "\t%llu", static_cast<unsigned long long>(r.epoch_pages[i]));
            }
            for (int i = 0; i < 4; ++i) {
                std::fprintf(f, "\t%llu", static_cast<unsigned long long>(r.epoch_pages_r[i]));
            }
            std::fprintf(f, "\t%llu\t%d\t%d\t%llu\t%u\t%u\t%u\t%u\t%u\t",
                         static_cast<unsigned long long>(r.overflow), r.analyzed ? 1 : 0,
                         r.end_blocked ? 1 : 0, static_cast<unsigned long long>(r.ops_total),
                         r.wait_events, r.signal_events, r.fence_syncs, r.draws, r.other_signals);
            for (u32 i = 0; i < r.nops; ++i) {
                std::fprintf(f, "%s%u", i ? "," : "", r.ops[i]);
            }
            std::fprintf(f, "\n");
        }
        const auto classes = [](u32 mask) {
            std::string out;
            for (size_t c = 0; c < NCLS; ++c) {
                if ((mask >> c) & 1) {
                    out += out.empty() ? "" : ",";
                    out += CLS_NAMES[c];
                }
            }
            return out.empty() ? std::string{"-"} : out;
        };
        std::fprintf(f, "[hits]\nframe\tpresent\tstage\tbig\tclasses\tid\tva\tlen\tndiff\n");
        for (const Hit& h : hits) {
            std::fprintf(f, "%llu\t%llu\t%u\t%u\t%s\t%llu\t%#llx\t%llu\t%llu\n",
                         static_cast<unsigned long long>(h.frame),
                         static_cast<unsigned long long>(h.present), h.stage & 0xFF,
                         (h.stage >> 8) & 1, classes(h.cls).c_str(),
                         static_cast<unsigned long long>(h.id),
                         static_cast<unsigned long long>(h.va),
                         static_cast<unsigned long long>(h.len),
                         static_cast<unsigned long long>(h.ndiff));
        }
        std::fprintf(f, "[epoch_hits]\nframe\tpresent\tcls\tserial\tva\tsize\tpagesB\tpagesC\tr_bytes\tr_cls\n");
        for (const EpochHit& h : ehits) {
            std::fprintf(f, "%llu\t%llu\t%s\t%llu\t%#llx\t%llu\t%llu\t%llu\t%llu\t%s\n",
                         static_cast<unsigned long long>(h.frame),
                         static_cast<unsigned long long>(h.present),
                         CLS_NAMES[static_cast<size_t>(h.cls)],
                         static_cast<unsigned long long>(h.serial),
                         static_cast<unsigned long long>(h.va),
                         static_cast<unsigned long long>(h.size),
                         static_cast<unsigned long long>(h.b), static_cast<unsigned long long>(h.c),
                         static_cast<unsigned long long>(h.r_bytes), classes(h.r_cls).c_str());
        }
        std::fprintf(f, "[epochs]\nidx\tcls\tserial\tva\tsize\n");
        {
            std::scoped_lock lk{epoch_mutex};
            for (size_t i = 0; i < epochs.size(); ++i) {
                std::fprintf(f, "%zu\t%s\t%llu\t%#llx\t%llu\n", i,
                             CLS_NAMES[static_cast<size_t>(epochs[i].cls)],
                             static_cast<unsigned long long>(epochs[i].serial),
                             static_cast<unsigned long long>(epochs[i].va),
                             static_cast<unsigned long long>(epochs[i].size));
            }
        }
        std::fprintf(f, "[calls]\nframe\tname\tcount\n");
        {
            std::scoped_lock lk{names_mutex};
            for (const FrameRow& r : frames) {
                for (const auto& [handler, n] : r.wcalls) {
                    const auto it = names.find(handler);
                    std::fprintf(f, "%llu\t%s\t%u\n", static_cast<unsigned long long>(r.index),
                                 it != names.end() ? it->second.c_str() : "?", n);
                }
                for (const auto& [tid, n] : r.wcall_tids) {
                    std::fprintf(f, "%llu\ttid:%u\t%u\n", static_cast<unsigned long long>(r.index),
                                 tid, n);
                }
            }
        }
        {
            std::scoped_lock lk{epoch_mutex};
            std::fprintf(f, "[watch]\ntsc\toffset\tlen\tndiff\n");
            for (const WatchEvent& e : watch_events) {
                std::fprintf(f, "%llu\t%llu\t%llu\t%llu\n", static_cast<unsigned long long>(e.tsc),
                             static_cast<unsigned long long>(e.offset),
                             static_cast<unsigned long long>(e.len),
                             static_cast<unsigned long long>(e.ndiff));
            }
            std::fprintf(f, "[epoch_events]\ntsc\tidx\tdelta\n");
            for (const EpochEvent& e : epoch_events) {
                std::fprintf(f, "%llu\t%u\t%llu\n", static_cast<unsigned long long>(e.tsc), e.index,
                             static_cast<unsigned long long>(e.delta));
            }
        }
        std::fprintf(f, "[maps]\nframe\ttsc\tkind\ttid\tva\tsize\n");
        for (const FrameRow& r : frames) {
            for (const MapRec& m : r.maps) {
                std::fprintf(f, "%llu\t%llu\t%s\t%u\t%#llx\t%llu\n",
                             static_cast<unsigned long long>(r.index),
                             static_cast<unsigned long long>(m.tsc), m.flush ? "flush" : "invalidate",
                             m.tid, static_cast<unsigned long long>(m.va),
                             static_cast<unsigned long long>(m.size));
            }
        }
        std::fclose(f);
    }

    // Read path.
    std::atomic_flag spin = ATOMIC_FLAG_INIT;
    FrameLog* cur{};
    u64 gen{1};
    struct DedupeSlot {
        u64 va{};
        u64 size{};
        size_t cls{};
        u64 gen{};
    };
    std::array<DedupeSlot, 4096> dedupe{};
    struct PoolSlot {
        u64 id{};
        size_t cls{};
        u64 gen{};
    };
    std::array<PoolSlot, 1024> pool_cache{};
    std::array<u64, NCLS> outside_calls{};
    std::array<u64, NCLS> outside_bytes{};
    u64 ops_outside{};
    u64 draws_outside{};

    // Events (state_mutex).
    std::mutex state_mutex;
    std::unordered_set<u64> gating;
    bool learned{};
    std::unordered_map<u64, u64> block_counts;
    std::unordered_map<u64, u64> cpu_signal_counts;
    std::unordered_map<u64, u64> eventcpu_counts;
    u64 last_block_address{};
    u32 last_block_value{};
    u64 blocks_seen{};
    int replay_state{};
    u64 block_tsc{};
    u64 block_seq{};
    u32 block_value{};
    u64 eventcpu_tsc{};
    u64 other_blocks{};
    u64 frames_dropped{};
    u64 game_wb{};
    u64 game_wake{};
    u64 game_signal{};
    u32 game_wake_tid{};
    u32 game_signal_tid{};
    u64 game_signal_value{};
    u32 game_other_signals{};
    std::unordered_map<const void*, u32> game_calls;
    std::unordered_map<u32, u32> game_call_tids;
    std::vector<MapRec> game_maps;
    std::atomic<bool> maps_open{false};
    u32 wb_tid{};
    bool wb_open{};
    bool have_prev_pass{};
    std::atomic<bool> in_wprime{false};
    std::atomic<u32> signal_tid{};
    std::atomic<u64> present{};
    std::atomic<bool> dump_asked{false};

    // Epochs.
    std::mutex epoch_mutex;
    std::vector<EpochEnt> epochs;
    std::atomic<size_t> epoch_count{0};
    u64 epochs_dropped{};
    std::array<std::vector<u64>, NSAMPLES> samples;
    std::array<std::array<u64, 4>, NCLS> epoch_totals{};
    struct EpochEvent {
        u64 tsc;
        u32 index;
        u64 delta;
    };
    std::vector<u64> poll_last;
    struct WatchEvent {
        u64 tsc;
        u64 offset;
        u64 len;
        u64 ndiff;
    };
    std::vector<WatchEvent> watch_events;
    std::vector<u8> watch_copy;
    u64 watch_va{};
    u64 watch_size{};
    std::vector<EpochEvent> epoch_events;
    u64 last_poll_tsc{};
    std::array<std::array<u64, 4>, NCLS> epoch_totals_r{};

    // Logs.
    std::array<std::unique_ptr<FrameLog>, NLOGS> logs;
    std::mutex free_mutex;
    std::vector<FrameLog*> free_logs;

    // Probe thread.
    std::mutex job_mutex;
    std::condition_variable job_cv;
    std::deque<Job> jobs;
    std::thread thread;
    std::deque<Job> fast_jobs;
    std::condition_variable fast_cv;
    std::thread fast_thread;
    std::mutex snap_mutex;
    std::mutex history_mutex;
    std::shared_mutex guest_mutex;
    GuestMemory* guest{};
    std::array<std::unique_ptr<Snap>, 6> snaps;
    std::array<Snap*, 2> current{};
    std::vector<FrameLog*> history;
    std::vector<FrameRow> frames;
    std::vector<Hit> hits;
    std::vector<EpochHit> ehits;
    std::array<ClassTotals, NCLS> totals{};
    u64 frame_index{};
    u64 frames_analyzed{};
    u64 frames_blocked{};
    u64 snaps_exhausted{};
    u64 max_backlog{};
    std::atomic<bool> dumped{false};

    std::mutex names_mutex;
    std::unordered_map<const void*, std::string> names;
};

Probe& Get() {
    static Probe* const probe = new Probe(); // never destroyed: threads may still call
    return *probe;
}

thread_local int suppress_tex_bytes = 0;

} // namespace

void NoteReadSlow(Cls cls, u64 va, u64 size, u64 id) {
    if (cls == Cls::TexBytes && suppress_tex_bytes > 0) {
        return;
    }
    Get().NoteRead(cls, va, size, id);
}

void NoteLinesSlow(Cls cls, u64 va, u64 size, u64 id) {
    constexpr u64 LINES = 256;
    if (size <= LINE * LINES) {
        Get().NoteRead(cls, va, size, id);
        return;
    }
    Get().NoteLineSet(cls, va, size, id);
}

SuppressTexBytes::SuppressTexBytes() {
    ++suppress_tex_bytes;
}
SuppressTexBytes::~SuppressTexBytes() {
    --suppress_tex_bytes;
}
bool TexBytesSuppressed() noexcept {
    return suppress_tex_bytes > 0;
}

void SetGuest(GuestMemory* memory) {
    if (On()) {
        Get().SetGuest(memory);
    }
}

void RegisterEpoch(u64 va, u64 size, u64 serial, Cls cls,
                   std::shared_ptr<const std::atomic<u64>> epoch) {
    if (On()) {
        Get().RegisterEpoch(va, size, serial, cls, std::move(epoch));
    }
}

void OnPresent(u64 count) {
    if (On()) {
        Get().OnPresent(count);
    }
}

void OnWaitBlocked(u64 address, u32 /*mode*/, u32 value, u64 seq, u32 /*queue*/) {
    if (Active()) {
        Get().OnWaitBlocked(address, value, seq);
    }
}

void OnWaitPassed(u64 address, u32 value) {
    if (Active()) {
        Get().OnWaitPassed(address, value);
    }
}

void OnEventCpu(u64 address) {
    if (Active()) {
        Get().OnEventCpu(address);
    }
}

void NoteOpSlow(u16 op) {
    Get().NoteOp(op);
}

void OnSyncWaitBegin(u64 /*fence*/, bool blocking, u32 tid) {
    if (Active()) {
        Get().OnSyncWaitBegin(blocking, tid);
    }
}

void OnSyncWaitEnd(u64 /*fence*/, u32 tid) {
    if (Active()) {
        Get().OnSyncWaitEnd(tid);
    }
}

void NoteSignalTid(u32 tid) {
    if (Active()) {
        Get().NoteSignalTid(tid);
    }
}

void OnEventSignal(u64 address, u32 written, u32 /*mode*/) {
    if (Active()) {
        Get().OnEventSignal(address, written);
    }
}

void OnGameCall(const void* handler, u32 tid) {
    if (Active()) {
        Get().OnGameCall(handler, tid);
    }
}

void OnMappedRange(bool flush, u64 va, u64 size, u32 tid) {
    if (Active()) {
        Get().OnMappedRange(flush, va, size, tid);
    }
}

void NameHandler(const void* handler, std::string_view name) {
    if (On()) {
        Get().NameHandler(handler, name);
    }
}

} // namespace VideoCommon::Nvn::PreRec
