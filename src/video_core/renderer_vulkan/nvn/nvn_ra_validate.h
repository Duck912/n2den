// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Record-ahead stage 2: draws and
// dispatches recorded ahead with VALIDATE-ONLY release checks (EDEN_NX2_NVN_RA_VALIDATE_ONLY=1,
// measurement only, off by default). The read classes whose exact repair comes in later stages
// (Backing::Mirror pages, texture coherency checks, texture bytes) are logged while recording
// ahead; at the release the same hashes the repair will need are taken again, side-effect free,
// and the first mismatch latches record-ahead off for the boot. Not exact under a change: the
// held work of the gate that finds the mismatch is submitted as recorded (one possibly wrong
// frame). The VRAM class uses the exact stage-1 repair (nvn_ra_vram.h). Rare classes (no repair
// and no check) latch off when read ahead. No Vulkan calls in here (unit tested in
// src/tests/nvn/record_ahead.cpp).

#pragma once

#include <array>
#include <bit>
#include <cstring>
#include <vector>

#include "common/common_types.h"
#include "video_core/nvn/prerec_probe.h"
#include "video_core/nvn/record.h"
#include "video_core/renderer_vulkan/nvn/nvn_ra_vram.h"

namespace Vulkan::Nvn {

/// What a validate-only check or tripwire found (the class of a latch).
enum class RaCheck : u8 {
    MirrorPage = 0,   ///< Backing::Mirror page decided ahead: guest bytes differ from its hash
    MirrorCopy,       ///< Backing::Mirror staging bytes copied ahead differ (late-bound again)
    TexCheck,         ///< texture coherency check ahead: signature / full hash differs
    TexBytes,         ///< texture bytes read ahead (upload, CPU copy) differ
    VramA2,           ///< after the VRAM repair a page a decision used is valid but written
    Unadmitted,       ///< a read ahead of a class with neither repair nor check
    PipelineNotReady, ///< a draw ahead found its pipeline still compiling (skipped)
    TexPoolGen,       ///< (stop, not a latch) a texture pool set's resolve generation moved
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(RaCheck::Count)> RA_CHECK_NAMES{
    "mirror-page", "mirror-copy",        "tex-check",   "tex-bytes",
    "vram-a2",     "unadmitted-class",   "pipeline-not-ready", "texpool-gen",
};

struct RaCheckStats {
    u64 logged{};     ///< entries logged ahead
    u64 checked{};    ///< entries looked at at the release
    u64 hashed{};     ///< ... of them hashed (the rest proven unchanged without a hash)
    u64 mismatches{}; ///< must be 0 for exactness (each latches off)
    u64 ns{};         ///< release time spent on this class
};

namespace PreRecCls = VideoCommon::Nvn::PreRec;

/// Read classes with neither a repair nor a release check in this stage: a read of one while
/// recording ahead latches record-ahead off (EventWord also has its pre-op stop: unseen events).
[[nodiscard]] constexpr bool RaUnadmittedClass(PreRecCls::Cls cls) noexcept {
    switch (cls) {
    case PreRecCls::Cls::TexGather:
    case PreRecCls::Cls::TexTakeover:
    case PreRecCls::Cls::OracleUniform:
    case PreRecCls::Cls::GenIndex:
    case PreRecCls::Cls::PoolUpload:
    case PreRecCls::Cls::EventWord:
    case PreRecCls::Cls::TexReadback:
    case PreRecCls::Cls::SparseGrowth:
        return true;
    default:
        return false;
    }
}

/// Validate-only admission of the draw-path ops (the stage 2 allow-list).
enum class RaAdmit : u8 {
    NotDrawOp, ///< not one of the stage 2 ops (the stage 1 rules decide)
    Yes,
    Indirect,  ///< indirect or count-buffer draw / dispatch: the GPU reads its arguments (stop)
    GenIndex,  ///< indexed quads / quad strip / line loop: indices converted on the CPU (stop)
    Malformed, ///< payload too short (stop; the recorder reports it)
};

[[nodiscard]] inline RaAdmit RaAdmitDrawOp(u16 op, const void* payload, size_t size) noexcept {
    namespace VC = VideoCommon::Nvn;
    switch (static_cast<VC::Op>(op)) {
    case VC::Op::Draw: {
        if (size < sizeof(VC::DrawPayload)) {
            return RaAdmit::Malformed;
        }
        VC::DrawPayload p;
        std::memcpy(&p, payload, sizeof(p));
        return p.indirect != 0 || p.count_address != 0 ? RaAdmit::Indirect : RaAdmit::Yes;
    }
    case VC::Op::DrawIndexed: {
        if (size < sizeof(VC::DrawIndexedPayload)) {
            return RaAdmit::Malformed;
        }
        VC::DrawIndexedPayload p;
        std::memcpy(&p, payload, sizeof(p));
        if (p.indirect != 0 || p.count_address != 0) {
            return RaAdmit::Indirect;
        }
        // DrawContext: quad_list (7), quad_strip (8), line_loop (2) convert the game's indices
        // on the CPU (GenerateIndices reads them: GenIndex).
        return p.mode == 2 || p.mode == 7 || p.mode == 8 ? RaAdmit::GenIndex : RaAdmit::Yes;
    }
    case VC::Op::DrawMeshTasks: {
        if (size < sizeof(VC::DrawMeshTasksPayload)) {
            return RaAdmit::Malformed;
        }
        VC::DrawMeshTasksPayload p;
        std::memcpy(&p, payload, sizeof(p));
        return p.indirect != 0 || p.count_address != 0 ? RaAdmit::Indirect : RaAdmit::Yes;
    }
    case VC::Op::Dispatch: {
        if (size < sizeof(VC::DispatchPayload)) {
            return RaAdmit::Malformed;
        }
        VC::DispatchPayload p;
        std::memcpy(&p, payload, sizeof(p));
        return p.indirect != 0 ? RaAdmit::Indirect : RaAdmit::Yes;
    }
    case VC::Op::SetRenderTargets:
    case VC::Op::ClearColor:
    case VC::Op::ClearDepthStencil:
    case VC::Op::UpdateUniformBuffer:
    case VC::Op::SetRasterizerDiscard:
    case VC::Op::SetPointSize:
    case VC::Op::SetLineWidth:
        return RaAdmit::Yes;
    default:
        return RaAdmit::NotDrawOp;
    }
}

/// Stage 2 ops that touch resources (images, pools, mirrors, layouts): non-held queues' resource
/// ops go behind them (ordering contagion).
[[nodiscard]] constexpr bool RaDrawTouching(u16 op) noexcept {
    namespace VC = VideoCommon::Nvn;
    switch (static_cast<VC::Op>(op)) {
    case VC::Op::Draw:
    case VC::Op::DrawIndexed:
    case VC::Op::DrawMeshTasks:
    case VC::Op::Dispatch:
    case VC::Op::SetRenderTargets:
    case VC::Op::ClearColor:
    case VC::Op::ClearDepthStencil:
    case VC::Op::UpdateUniformBuffer:
        return true;
    default:
        return false;
    }
}

/// A set of page numbers (a bit each), grown on demand.
class RaPageBits {
public:
    /// True when the page was not in the set yet.
    bool Insert(u64 page) {
        const size_t word = static_cast<size_t>(page / 64);
        if (word >= bits.size()) {
            bits.resize(word + 1, 0);
        }
        const u64 mask = u64{1} << (page % 64);
        if ((bits[word] & mask) != 0) {
            return false;
        }
        bits[word] |= mask;
        ++count;
        return true;
    }
    [[nodiscard]] bool Test(u64 page) const noexcept {
        const size_t word = static_cast<size_t>(page / 64);
        return word < bits.size() && ((bits[word] >> (page % 64)) & 1) != 0;
    }
    [[nodiscard]] u64 Count() const noexcept {
        return count;
    }
    template <typename F>
    void ForEach(F&& f) const {
        for (size_t word = 0; word < bits.size(); ++word) {
            u64 w = bits[word];
            while (w != 0) {
                const u64 bit = static_cast<u64>(std::countr_zero(w));
                w &= w - 1;
                f(static_cast<u64>(word) * 64 + bit);
            }
        }
    }
    void Clear() noexcept {
        bits.clear();
        count = 0;
    }

private:
    std::vector<u64> bits;
    u64 count{};
};

/// A Backing::Mirror page a decision recorded ahead used, at the release.
struct RaMirrorPageState {
    bool valid{};         ///< MirrorPage::valid
    bool trusted{};       ///< MirrorPage::trusted
    bool tracked{};       ///< the pool's CPU write tracker is registered
    bool written{};       ///< the tracker shows a CPU write of the page (not taken yet)
    bool split{};         ///< split pool (raw stores reach its guest bytes past the tracker)
    u64 page_raw_epoch{}; ///< MirrorPage::raw_epoch
    u64 epoch_now{};      ///< Impl::split_raw_epoch at the release
};

enum class RaPageVerdict : u8 {
    Unchanged, ///< still trusted: nothing can have changed its guest bytes unseen
    EpochOnly, ///< trusted except that a raw store completed since (a current pre-hash may decide)
    Hash,      ///< hash the guest page and compare it with MirrorPage::hash
};

/// The release check of one page: the decision today would take after the signal, side-effect
/// free (no tracker take, no epoch adopted).
[[nodiscard]] constexpr RaPageVerdict RaMirrorPageVerdict(const RaMirrorPageState& s) noexcept {
    if (!s.valid || !s.trusted || !s.tracked || s.written) {
        return RaPageVerdict::Hash;
    }
    if (s.split && s.page_raw_epoch != s.epoch_now) {
        return RaPageVerdict::EpochOnly;
    }
    return RaPageVerdict::Unchanged;
}

/// EDEN_NX2_NVN_RA_INJECT=vramepoch: the pages of a logged pool the release repair treats as
/// written although the CPU did not write them: every page a decision recorded ahead treated as
/// valid or current that was valid before the repair (the mirror holds the guest bytes there, so
/// the prologue copies equal bytes: the copy path runs with real data and changes nothing).
[[nodiscard]] inline RaRuns RaVramInjectRuns(const RaVramLog& log, const MirrorPages& before) {
    const u64 size = log.tv_take.Size();
    RaPageSet pages{size};
    for (size_t word = 0; word < log.tv_take.Words(); ++word) {
        u64 used = log.tv_take.Word(word) | log.tv_current.Word(word);
        while (used != 0) {
            const u64 bit = static_cast<u64>(std::countr_zero(used));
            used &= used - 1;
            const u64 page = static_cast<u64>(word) * 64 + bit;
            if (before.AllValid(page * RaPageSet::PAGE, RaPageSet::PAGE)) {
                pages.Set(page * RaPageSet::PAGE, RaPageSet::PAGE);
            }
        }
    }
    return pages.Runs();
}

/// The pages of a logged pool the VRAM A2 check looks at after the repair: every page a decision
/// recorded ahead used (TV_take | TV_current | CA). A2 counts those valid in the mirror while the
/// tracker shows them written (a stale page the repair left valid).
[[nodiscard]] inline RaRuns RaVramA2Pages(const RaVramLog& log) {
    const u64 size = log.tv_take.Size();
    RaPageSet pages{size};
    for (size_t word = 0; word < log.tv_take.Words(); ++word) {
        pages.OrWord(word,
                     log.tv_take.Word(word) | log.tv_current.Word(word) | log.copied.Word(word));
    }
    return pages.Runs();
}

/// The VRAM A2 verdict on one page a decision used, after the release repair.
enum class RaVramA2Verdict {
    Clean,               ///< invalid in the mirror, or not written
    WrittenAfterCollect, ///< valid and written, and the write epoch moved since the collect read it
    Positive,            ///< valid and written with an unmoved epoch: latched
};

/// `valid`: the mirror holds the page (MirrorPages::AllValid). `written`: the tracker shows it
/// dirty (GuestMemory::AnyWritten). `epoch_collect`: the write epoch the repair's collect read
/// BEFORE its TakeWritten (PoolMirror::epoch_clean right after MirrorCollectCpuWrites; when the
/// collect returned early it is the value it compared equal). `epoch_now`: the epoch at the check.
///
/// The tracker (core/memory/page_tracker_state.cpp) sets a dirty bit only in OnWrite and OnRemap,
/// each with an epoch increment under its lock, and TakeWritten over the whole registration cleans
/// every dirty bit. So a page dirty at the check either turned dirty after the collect read the
/// epoch (the epoch moved past `epoch_collect`), or the collect did not cover it. The epoch must be
/// the one read BEFORE the TakeWritten: a write landing between the TakeWritten and a later read
/// (the guest writing the pool during the release) leaves an epoch read after the collect equal to
/// the current one while the page is dirty (boot ra2g1). Without a write epoch nothing tells the
/// cases apart: positive.
[[nodiscard]] constexpr RaVramA2Verdict RaVramA2Classify(bool valid, bool written, bool has_epoch,
                                                         u64 epoch_collect, u64 epoch_now) {
    if (!valid || !written) {
        return RaVramA2Verdict::Clean;
    }
    if (has_epoch && epoch_now != epoch_collect) {
        return RaVramA2Verdict::WrittenAfterCollect;
    }
    return RaVramA2Verdict::Positive;
}

} // namespace Vulkan::Nvn
