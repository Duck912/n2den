// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <functional>
#include <map>
#include <optional>
#include <vector>

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/memory/page_tracker_state.h"
#include "video_core/nvn/prerec_probe.h"
#include "video_core/nvn/record_ahead_gate.h"
#include "video_core/renderer_vulkan/nvn/nvn_ra_validate.h"
#include "video_core/renderer_vulkan/nvn/nvn_ra_vram.h"

namespace {

using VideoCommon::Nvn::EVENT_WAIT_EQUAL;
using VideoCommon::Nvn::EVENT_WAIT_GEQUAL_WRAP;

struct Batch {
    u32 queue{};
    u64 value{}; ///< timeline value of the batch's last segment
};
using Gate = VideoCommon::Nvn::RecordAheadGate<Batch>;

constexpr u64 WORD_A = 0x290d675000;
constexpr u64 WORD_B = 0x290d675004;

/// Models the replay's submit routing: a held queue's batch is held, anything else goes to the
/// GPU at once.
struct Replay {
    Gate gate;
    std::vector<Batch> gpu; ///< vkQueueSubmit order

    void Submit(u32 queue, u64 seq, u64 value) {
        if (gate.Held(queue)) {
            gate.Hold(queue, seq, Batch{queue, value});
        } else {
            gpu.push_back({queue, value});
        }
    }
    void Release() {
        for (auto& entry : gate.Take()) {
            gpu.push_back(entry.batch);
        }
    }
};

} // Anonymous namespace

TEST_CASE("RecordAheadGate: held work is released in recording order", "[nvn][record_ahead]") {
    Replay replay;
    // Queue 0 records before its gate wait: submitted as usual.
    replay.Submit(0, 10, 4);
    // Queue 0 passes the wait on word A provisionally; its open segment is value 5.
    replay.gate.Pass({WORD_A, EVENT_WAIT_EQUAL, 7, 0, 10}, 5);
    REQUIRE(replay.gate.Active());
    REQUIRE(replay.gate.Held(0));
    REQUIRE_FALSE(replay.gate.Held(1));
    // Queue 1 is not behind the gate: its work reaches the GPU while queue 0 is held.
    replay.Submit(1, 11, 20);
    // Queue 2 joins on word B (its pre-wait work went first).
    replay.Submit(2, 12, 30);
    replay.gate.Pass({WORD_B, EVENT_WAIT_GEQUAL_WRAP, 3, 2, 12}, 31);
    replay.Submit(0, 10, 5);
    replay.Submit(2, 12, 31);
    replay.Submit(0, 13, 6);
    replay.Submit(1, 14, 21);
    replay.Submit(2, 12, 32);
    REQUIRE(replay.gate.MinHeldSeq() == 10);

    const std::vector<std::pair<u32, u64>> before{{0, 4}, {1, 20}, {2, 30}, {1, 21}};
    REQUIRE(replay.gpu.size() == before.size());
    for (size_t i = 0; i < before.size(); ++i) {
        REQUIRE(replay.gpu[i].queue == before[i].first);
        REQUIRE(replay.gpu[i].value == before[i].second);
    }
    replay.Release();
    REQUIRE_FALSE(replay.gate.Active());
    REQUIRE_FALSE(replay.gate.Held(0));
    const std::vector<std::pair<u32, u64>> after{{0, 4},  {1, 20}, {2, 30}, {1, 21},
                                                 {0, 5},  {2, 31}, {0, 6},  {2, 32}};
    REQUIRE(replay.gpu.size() == after.size());
    for (size_t i = 0; i < after.size(); ++i) {
        REQUIRE(replay.gpu[i].queue == after[i].first);
        REQUIRE(replay.gpu[i].value == after[i].second);
    }
    // Per queue, timeline values reach the GPU in increasing order (recording order).
    std::map<u32, u64> last;
    for (const Batch& batch : replay.gpu) {
        REQUIRE(batch.value > last[batch.queue]);
        last[batch.queue] = batch.value;
    }
}

TEST_CASE("RecordAheadGate: contagion holds a queue that waits on held work",
          "[nvn][record_ahead]") {
    Replay replay;
    replay.gate.Pass({WORD_A, EVENT_WAIT_EQUAL, 1, 0, 1}, 8);
    // Values of queue 0 below its first held one were submitted: waiting on them needs no hold.
    REQUIRE_FALSE(replay.gate.NamesHeld(0, 7));
    REQUIRE(replay.gate.NamesHeld(0, 8));
    REQUIRE(replay.gate.NamesHeld(0, 9));
    // Queue 1 holds nothing: a wait on it names no held work.
    REQUIRE_FALSE(replay.gate.NamesHeld(1, 100));

    // Queue 1 records a wait on queue 0's held value 9: from its next segment (value 40) it is
    // held; what it submitted before (39) went to the GPU.
    replay.Submit(1, 2, 39);
    REQUIRE(replay.gate.NamesHeld(0, 9));
    replay.gate.HoldQueue(1, 40);
    REQUIRE(replay.gate.Held(1));
    REQUIRE(replay.gate.HeldFrom(1) == 40);
    // A second contagion of the same queue keeps the first value.
    replay.gate.HoldQueue(1, 45);
    REQUIRE(replay.gate.HeldFrom(1) == 40);
    replay.Submit(0, 1, 9);
    replay.Submit(1, 2, 40);
    REQUIRE(replay.gpu.size() == 1);
    // Queue 2 waits on queue 1's held value: transitively held.
    REQUIRE(replay.gate.NamesHeld(1, 40));
    replay.gate.HoldQueue(2, 70);
    replay.Submit(2, 3, 70);
    replay.Release();
    REQUIRE(replay.gpu.size() == 4);
    REQUIRE(replay.gpu[1].value == 9);
    REQUIRE(replay.gpu[2].value == 40);
    REQUIRE(replay.gpu[3].value == 70);
    // After the release nothing is held and no wait names held work.
    REQUIRE_FALSE(replay.gate.NamesHeld(0, 9));
    REQUIRE_FALSE(replay.gate.Held(2));
}

TEST_CASE("RecordAheadGate: a misprediction keeps everything held (no rollback)",
          "[nvn][record_ahead]") {
    Replay replay;
    std::map<u64, u32> words{{WORD_A, 6}, {WORD_B, 2}};
    const auto word = [&words](u64 address) { return words[address]; };
    replay.gate.Pass({WORD_A, EVENT_WAIT_EQUAL, 7, 0, 1}, 3);
    replay.gate.Pass({WORD_B, EVENT_WAIT_GEQUAL_WRAP, 3, 1, 2}, 10);
    replay.Submit(0, 1, 3);
    replay.Submit(1, 2, 10);
    REQUIRE(replay.gate.IsGateWord(WORD_A));
    REQUIRE(replay.gate.IsGateWord(WORD_B));
    REQUIRE_FALSE(replay.gate.IsGateWord(0x1000));

    // The guest's first change of A is not the predicted value: nothing is released.
    words[WORD_A] = 8;
    REQUIRE_FALSE(replay.gate.AllSatisfied(word));
    REQUIRE(replay.gate.Active());
    REQUIRE(replay.gate.HeldEntries().size() == 2);
    REQUIRE(replay.gpu.empty());
    // Held work recorded after the misprediction stays in order behind it.
    replay.Submit(0, 1, 4);
    // A satisfied, B not yet: still held (a multi-word gate waits for every word).
    words[WORD_A] = 7;
    REQUIRE_FALSE(replay.gate.AllSatisfied(word));
    // B wraps past its target: every provisional wait is satisfied.
    words[WORD_B] = 3;
    REQUIRE(replay.gate.AllSatisfied(word));
    replay.Release();
    REQUIRE(replay.gpu.size() == 3);
    REQUIRE(replay.gpu[0].value == 3);
    REQUIRE(replay.gpu[1].value == 10);
    REQUIRE(replay.gpu[2].value == 4);
}

TEST_CASE("RecordAheadGate: an empty gate holds nothing", "[nvn][record_ahead]") {
    Gate gate;
    REQUIRE_FALSE(gate.Active());
    REQUIRE_FALSE(gate.Held(0));
    REQUIRE_FALSE(gate.NamesHeld(0, 1));
    REQUIRE(gate.MinHeldSeq() == ~u64{0});
    REQUIRE(gate.Last() == nullptr);
    REQUIRE(gate.Take().empty());
}

// --- Frozen write view and release repair of VRAM-mirrored pools ---
//
// A scripted model of a pool mirror: MirrorPages (the real class) plus a fake page tracker with
// the GuestMemory rules (armed pages turn dirty on a CPU write, the write epoch grows, TakeWritten
// returns and cleans dirty armed pages). Today (T) runs every decision after the CPU write; ahead
// (A) runs the same decisions before it, with the frozen view, and repairs at the release. The
// tests check that the state after A + repair equals T's: valid bits, armed and dirty pages, the
// clean epoch; for TV_current-only pages, that every later decision answers the same.

namespace {

using Vulkan::Nvn::MirrorPages;
using Vulkan::Nvn::RaRuns;
using Vulkan::Nvn::RaVramLog;
using Vulkan::Nvn::RaVramRepairPlan;

constexpr u64 PAGE = MirrorPages::PAGE;
constexpr u64 POOL = 8 * PAGE;

struct FakeTracker {
    std::vector<bool> armed = std::vector<bool>(POOL / PAGE, false);
    std::vector<bool> dirty = std::vector<bool>(POOL / PAGE, false);
    u64 epoch{};

    void CpuWrite(u64 page) {
        if (armed[page] && !dirty[page]) {
            dirty[page] = true;
            ++epoch;
        }
    }
    void Arm(u64 begin, u64 end) {
        for (u64 page = begin / PAGE; page * PAGE < end; ++page) {
            armed[page] = true;
            dirty[page] = false;
        }
    }
    void TakeWritten(RaRuns& runs) {
        for (u64 page = 0; page < dirty.size(); ++page) {
            if (armed[page] && dirty[page]) {
                dirty[page] = false;
                if (!runs.empty() && runs.back().second == page * PAGE) {
                    runs.back().second += PAGE;
                } else {
                    runs.emplace_back(page * PAGE, page * PAGE + PAGE);
                }
            }
        }
    }
    [[nodiscard]] bool AnyWritten(u64 offset, u64 size) const {
        for (u64 page = offset / PAGE; page * PAGE < offset + size; ++page) {
            if (armed[page] && dirty[page]) {
                return true;
            }
        }
        return false;
    }
};

/// One pool mirror as Context::Impl handles it (nvn_pool_mirror.cpp), with the record-ahead hooks.
struct MirrorModel {
    MirrorPages pages{POOL};
    FakeTracker tracker;
    u64 epoch_clean{~u64{0}};
    std::vector<std::pair<u64, u64>> copies; ///< guest -> mirror copies recorded (any time)
    bool ahead{false};                        ///< decisions are recorded ahead
    bool frozen_view{true};                   ///< false: the broken variant (no frozen view)
    std::optional<RaVramLog> log;
    /// Live, non-pending mirror entries of the open segment (RecordMirrorCopies cuts around them).
    std::vector<std::pair<u64, u64>> live_entries;

    // MirrorCollectCpuWrites
    void Collect(RaRuns* taken = nullptr) {
        if (tracker.epoch == epoch_clean) {
            return;
        }
        const u64 epoch = tracker.epoch;
        RaRuns written;
        tracker.TakeWritten(written);
        for (const auto& [begin, end] : written) {
            pages.Invalidate(begin, end - begin);
            if (taken != nullptr) {
                taken->emplace_back(begin, end);
            }
        }
        epoch_clean = epoch;
    }
    /// The record-ahead hook before a decision: true = frozen (skip the tracker).
    bool Frozen() {
        if (!ahead || !frozen_view) {
            return false;
        }
        if (!log) {
            Collect(); // the first decision ahead: what the CPU wrote so far
            log.emplace(POOL, epoch_clean);
        }
        ++log->decisions;
        return true;
    }
    // MirrorTakeStale + RecordMirrorCopies (entry exclusion)
    void Take(u64 offset, u64 size) {
        const bool frozen = Frozen();
        if (!frozen) {
            Collect();
        }
        RaRuns runs;
        pages.TakeInvalid(offset, size, runs);
        if (frozen) {
            log->NoteTake(offset, size, runs, 0);
        }
        for (const auto& [begin, end] : runs) {
            tracker.Arm(begin, end);
        }
        for (const auto& [begin, end] : runs) {
            bool excluded = false;
            for (const auto& [entry_begin, entry_end] : live_entries) {
                if (begin < entry_end && entry_begin < end) {
                    excluded = true; // the whole run stays stale
                }
            }
            if (excluded) {
                pages.Invalidate(begin, end - begin);
            } else {
                copies.emplace_back(begin, end);
            }
        }
    }
    // MirrorCurrent
    bool Current(u64 offset, u64 size) {
        if (Frozen()) {
            const bool current = pages.AllValid(offset, size);
            if (current) {
                log->NoteCurrent(offset, size);
            }
            return current;
        }
        if (!pages.AllValid(offset, size)) {
            return false;
        }
        return tracker.epoch == epoch_clean || !tracker.AnyWritten(offset, size);
    }
    /// A CpuWrite command (FlushMappedRange): today invalidates now; ahead on a logged pool it is
    /// deferred to the release.
    void FlushCommand(u64 offset, u64 size) {
        if (ahead && frozen_view && log) {
            log->flushed.emplace_back(offset, offset + size);
            return;
        }
        pages.Invalidate(offset, size);
    }
    /// Recorder::NotePoolWrite (a GPU write through the pool's buffer, recorded).
    void PoolWrite(u64 offset, u64 size) {
        pages.Invalidate(offset, size);
    }
    /// The release repair (RaVramRepairAll).
    void Release() {
        ahead = false;
        if (!log) {
            return;
        }
        const MirrorPages before = pages;
        for (const auto& [begin, end] : log->flushed) {
            pages.Invalidate(begin, end - begin);
        }
        RaRuns written;
        Collect(&written);
        const auto repair = RaVramRepairPlan(*log, before, pages, written);
        for (const auto& [begin, end] : repair.copies) {
            copies.emplace_back(begin, end); // prologue
        }
        for (const auto& [begin, end] : repair.revalidate) {
            tracker.Arm(begin, end);
        }
        log.reset();
    }
    void EndSegment() {
        live_entries.clear();
    }
};

/// Two models start from the same state.
struct Pair {
    MirrorModel today;
    MirrorModel ahead;
    void Both(const std::function<void(MirrorModel&)>& step) {
        step(today);
        step(ahead);
    }
};

void RequireSameState(const MirrorModel& a, const MirrorModel& b) {
    for (u64 page = 0; page < POOL / PAGE; ++page) {
        INFO("page " << page);
        REQUIRE(a.pages.AllValid(page * PAGE, PAGE) == b.pages.AllValid(page * PAGE, PAGE));
        REQUIRE(a.tracker.armed[page] == b.tracker.armed[page]);
        REQUIRE(a.tracker.dirty[page] == b.tracker.dirty[page]);
    }
    REQUIRE(a.epoch_clean == b.epoch_clean);
}

/// Every page answers MirrorCurrent the same, and a later whole-pool take copies the same runs and
/// leaves the same state.
void RequireSameAnswers(MirrorModel& a, MirrorModel& b) {
    for (u64 page = 0; page < POOL / PAGE; ++page) {
        INFO("page " << page);
        REQUIRE(a.Current(page * PAGE, PAGE) == b.Current(page * PAGE, PAGE));
    }
    a.copies.clear();
    b.copies.clear();
    a.Take(0, POOL);
    b.Take(0, POOL);
    REQUIRE(a.copies == b.copies);
    RequireSameState(a, b);
}

/// Pages 0-7 copied in and armed (a warm pool), segment ended.
Pair Warm() {
    Pair pair;
    pair.Both([](MirrorModel& m) {
        m.Take(0, POOL);
        m.EndSegment();
        m.copies.clear();
    });
    return pair;
}

constexpr u64 P = 3; ///< the page the CPU writes

} // Anonymous namespace

TEST_CASE("RaVram: the counter-example (a live entry over a page written mid-span)",
          "[nvn][record_ahead]") {
    // Today: the write precedes the gated recording. E0 creates entry X inside page P (pending:
    // its own take is not excluded) and copies P; Dk finds P valid.
    Pair pair = Warm();
    MirrorModel& t = pair.today;
    t.tracker.CpuWrite(P);
    t.Take(P * PAGE + 64, 128); // E0
    t.live_entries.emplace_back(P * PAGE + 64, P * PAGE + 192);
    t.Take(P * PAGE, PAGE);     // Dk
    t.EndSegment();

    // Without the frozen view: E0 ahead sees P valid; the write lands; Dk collects it, takes P,
    // meets live X and leaves the whole run stale: bookkeeping differs from today's.
    MirrorModel broken = Warm().ahead;
    broken.ahead = true;
    broken.frozen_view = false;
    broken.Take(P * PAGE + 64, 128);
    broken.live_entries.emplace_back(P * PAGE + 64, P * PAGE + 192);
    broken.tracker.CpuWrite(P);
    broken.Take(P * PAGE, PAGE);
    broken.EndSegment();
    broken.ahead = false;
    REQUIRE_FALSE(broken.pages.AllValid(P * PAGE, PAGE));
    REQUIRE(t.pages.AllValid(P * PAGE, PAGE));

    // Ahead with the frozen view: Dk does not see the write; the release copies P in the prologue
    // and leaves it valid and armed, as today.
    MirrorModel& a = pair.ahead;
    a.ahead = true;
    a.Take(P * PAGE + 64, 128);
    a.live_entries.emplace_back(P * PAGE + 64, P * PAGE + 192);
    a.tracker.CpuWrite(P);
    a.Take(P * PAGE, PAGE);
    a.EndSegment();
    REQUIRE(a.copies.empty());
    a.Release();
    REQUIRE(a.copies == std::vector<std::pair<u64, u64>>{{P * PAGE, P * PAGE + PAGE}});
    RequireSameState(t, a);
}

TEST_CASE("RaVram: a page copied ahead is valid and armed again after the release",
          "[nvn][record_ahead]") {
    Pair pair = Warm();
    pair.Both([](MirrorModel& m) { m.pages.Invalidate(P * PAGE, PAGE); }); // stale before the gate
    MirrorModel& t = pair.today;
    t.tracker.CpuWrite(P);
    t.Take(P * PAGE, PAGE);

    MirrorModel& a = pair.ahead;
    a.ahead = true;
    a.Take(P * PAGE, PAGE); // copied ahead (the GPU copy runs after the release)
    a.tracker.CpuWrite(P);  // armed by the ahead take: dirty now
    REQUIRE(a.tracker.dirty[P]);
    a.Release();
    REQUIRE(a.pages.AllValid(P * PAGE, PAGE));
    REQUIRE(a.tracker.armed[P]);
    REQUIRE_FALSE(a.tracker.dirty[P]);
    // Copy-ahead pages get no prologue copy (their held copy reads the bytes at GPU time).
    REQUIRE(a.copies == std::vector<std::pair<u64, u64>>{{P * PAGE, P * PAGE + PAGE}});
    REQUIRE(t.copies == a.copies);
    RequireSameState(t, a);
}

TEST_CASE("RaVram: pages only MirrorCurrent treated as current stay invalid",
          "[nvn][record_ahead]") {
    Pair pair = Warm();
    MirrorModel& t = pair.today;
    t.tracker.CpuWrite(P);
    REQUIRE_FALSE(t.Current(P * PAGE, PAGE)); // today: dirty, so the binding reads guest memory

    MirrorModel& a = pair.ahead;
    a.ahead = true;
    REQUIRE(a.Current(P * PAGE, PAGE)); // ahead: the mirror, which the prologue refreshes
    a.tracker.CpuWrite(P);
    a.Release();
    REQUIRE(a.copies == std::vector<std::pair<u64, u64>>{{P * PAGE, P * PAGE + PAGE}});
    REQUIRE_FALSE(a.pages.AllValid(P * PAGE, PAGE));
    // Today has P valid but dirty, ahead has it invalid: every later decision answers the same.
    RequireSameAnswers(t, a);
}

TEST_CASE("RaVram: a CpuWrite command on a logged pool is applied at the release",
          "[nvn][record_ahead]") {
    Pair pair = Warm();
    MirrorModel& t = pair.today;
    t.tracker.CpuWrite(P);
    t.FlushCommand(P * PAGE, PAGE);
    t.FlushCommand(6 * PAGE, 16); // a flush of a page the tracker did not see written
    t.Take(0, 4 * PAGE);
    t.Take(6 * PAGE, PAGE);

    MirrorModel& a = pair.ahead;
    a.ahead = true;
    a.Take(0, 4 * PAGE); // logs the pool (TV_take 0-3)
    a.tracker.CpuWrite(P);
    a.FlushCommand(P * PAGE, PAGE);
    a.FlushCommand(6 * PAGE, 16);
    REQUIRE(a.pages.AllValid(0, POOL)); // deferred
    a.Take(6 * PAGE, PAGE);
    a.Release();
    REQUIRE(a.copies ==
            std::vector<std::pair<u64, u64>>{{P * PAGE, P * PAGE + PAGE}, {6 * PAGE, 7 * PAGE}});
    REQUIRE(t.copies == a.copies);
    RequireSameState(t, a);
}

TEST_CASE("RaVram: a page a held GPU write invalidated after a TV decision stays invalid",
          "[nvn][record_ahead]") {
    Pair pair = Warm();
    MirrorModel& t = pair.today;
    t.tracker.CpuWrite(P);
    t.Take(P * PAGE, PAGE);      // copies P (after the write)
    t.PoolWrite(P * PAGE, 256);  // a held draw writes the guest bytes

    MirrorModel& a = pair.ahead;
    a.ahead = true;
    a.Take(P * PAGE, PAGE);      // P valid: TV_take
    a.PoolWrite(P * PAGE, 256);  // same op
    a.tracker.CpuWrite(P);
    a.Release();
    // The prologue still copies P (the draw between read the mirror), P stays invalid.
    REQUIRE(a.copies == std::vector<std::pair<u64, u64>>{{P * PAGE, P * PAGE + PAGE}});
    REQUIRE_FALSE(a.pages.AllValid(P * PAGE, PAGE));
    RequireSameState(t, a);
}

TEST_CASE("RaVram: nothing written in the span: the release changes nothing",
          "[nvn][record_ahead]") {
    Pair pair = Warm();
    pair.today.Take(0, POOL);
    MirrorModel& a = pair.ahead;
    a.ahead = true;
    a.Take(0, POOL);
    REQUIRE(a.Current(PAGE, PAGE));
    a.Release();
    REQUIRE(a.copies.empty());
    RequireSameState(pair.today, a);
}

// --- Stage 2: validate-only admission and release checks (nvn_ra_validate.h) ---

namespace {

using Vulkan::Nvn::RaAdmit;
using Vulkan::Nvn::RaAdmitDrawOp;
using Vulkan::Nvn::RaMirrorPageState;
using Vulkan::Nvn::RaMirrorPageVerdict;
using Vulkan::Nvn::RaPageBits;
using Vulkan::Nvn::RaPageVerdict;
namespace VCN = VideoCommon::Nvn;

template <typename P>
RaAdmit Admit(VCN::Op op, const P& payload) {
    return RaAdmitDrawOp(static_cast<u16>(op), &payload, sizeof(payload));
}

} // Anonymous namespace

TEST_CASE("RaValidate: draw-path admission (direct only, CPU index conversion stops)",
          "[nvn][record_ahead]") {
    VCN::DrawPayload draw{};
    draw.mode = 4;
    REQUIRE(Admit(VCN::Op::Draw, draw) == RaAdmit::Yes);
    draw.mode = 7; // quads without indices: generated indices, no guest read
    REQUIRE(Admit(VCN::Op::Draw, draw) == RaAdmit::Yes);
    draw.indirect = 0x1000;
    REQUIRE(Admit(VCN::Op::Draw, draw) == RaAdmit::Indirect);

    VCN::DrawIndexedPayload indexed{};
    indexed.mode = 4;
    REQUIRE(Admit(VCN::Op::DrawIndexed, indexed) == RaAdmit::Yes);
    for (const u32 mode : {2u, 7u, 8u}) {
        indexed.mode = mode;
        REQUIRE(Admit(VCN::Op::DrawIndexed, indexed) == RaAdmit::GenIndex);
    }
    indexed.mode = 4;
    indexed.count_address = 0x2000;
    REQUIRE(Admit(VCN::Op::DrawIndexed, indexed) == RaAdmit::Indirect);

    VCN::DispatchPayload dispatch{};
    REQUIRE(Admit(VCN::Op::Dispatch, dispatch) == RaAdmit::Yes);
    dispatch.indirect = 0x3000;
    REQUIRE(Admit(VCN::Op::Dispatch, dispatch) == RaAdmit::Indirect);

    VCN::DrawMeshTasksPayload mesh{};
    REQUIRE(Admit(VCN::Op::DrawMeshTasks, mesh) == RaAdmit::Yes);
    mesh.indirect = 0x4000;
    REQUIRE(Admit(VCN::Op::DrawMeshTasks, mesh) == RaAdmit::Indirect);

    const u32 word = 0;
    REQUIRE(RaAdmitDrawOp(static_cast<u16>(VCN::Op::Draw), &word, sizeof(word)) ==
            RaAdmit::Malformed);
    for (const VCN::Op op : {VCN::Op::SetRenderTargets, VCN::Op::ClearColor,
                             VCN::Op::ClearDepthStencil, VCN::Op::UpdateUniformBuffer,
                             VCN::Op::SetRasterizerDiscard, VCN::Op::SetPointSize,
                             VCN::Op::SetLineWidth}) {
        REQUIRE(RaAdmitDrawOp(static_cast<u16>(op), nullptr, 0) == RaAdmit::Yes);
    }
    for (const VCN::Op op : {VCN::Op::CopyBufferToBuffer, VCN::Op::ClearTexture,
                             VCN::Op::DrawTexture, VCN::Op::EvaluateDlss, VCN::Op::BindProgram}) {
        REQUIRE(RaAdmitDrawOp(static_cast<u16>(op), nullptr, 0) == RaAdmit::NotDrawOp);
    }
    REQUIRE(Vulkan::Nvn::RaDrawTouching(static_cast<u16>(VCN::Op::Draw)));
    REQUIRE_FALSE(Vulkan::Nvn::RaDrawTouching(static_cast<u16>(VCN::Op::SetLineWidth)));
}

TEST_CASE("RaValidate: rare classes latch, checked classes do not", "[nvn][record_ahead]") {
    using C = VCN::PreRec::Cls;
    for (const C cls : {C::TexGather, C::TexTakeover, C::OracleUniform, C::GenIndex,
                        C::PoolUpload, C::EventWord, C::TexReadback, C::SparseGrowth}) {
        REQUIRE(Vulkan::Nvn::RaUnadmittedClass(cls));
    }
    for (const C cls : {C::MirrorCheck, C::MirrorCopy, C::VramMirror, C::VramWanted, C::TexSig,
                        C::TexFull, C::TexBytes, C::UniformWord, C::HandleMem, C::UniformCopy}) {
        REQUIRE_FALSE(Vulkan::Nvn::RaUnadmittedClass(cls));
    }
}

TEST_CASE("RaValidate: Backing::Mirror page verdict at the release", "[nvn][record_ahead]") {
    RaMirrorPageState s{.valid = true,
                        .trusted = true,
                        .tracked = true,
                        .written = false,
                        .split = false,
                        .page_raw_epoch = 5,
                        .epoch_now = 6};
    REQUIRE(RaMirrorPageVerdict(s) == RaPageVerdict::Unchanged); // not split: epoch ignored
    s.split = true;
    REQUIRE(RaMirrorPageVerdict(s) == RaPageVerdict::EpochOnly);
    s.page_raw_epoch = 6;
    REQUIRE(RaMirrorPageVerdict(s) == RaPageVerdict::Unchanged);
    auto hash_if = [&](auto change) {
        RaMirrorPageState t = s;
        change(t);
        return RaMirrorPageVerdict(t) == RaPageVerdict::Hash;
    };
    REQUIRE(hash_if([](RaMirrorPageState& t) { t.written = true; }));
    REQUIRE(hash_if([](RaMirrorPageState& t) { t.trusted = false; }));
    REQUIRE(hash_if([](RaMirrorPageState& t) { t.tracked = false; }));
    REQUIRE(hash_if([](RaMirrorPageState& t) { t.valid = false; }));
    // A write wins over an epoch move: hashed, never decided by a pre-hash.
    REQUIRE(hash_if([](RaMirrorPageState& t) {
        t.written = true;
        t.page_raw_epoch = 1;
    }));
}

TEST_CASE("RaValidate: page set inserts once and iterates in order", "[nvn][record_ahead]") {
    RaPageBits bits;
    REQUIRE(bits.Insert(3));
    REQUIRE_FALSE(bits.Insert(3));
    REQUIRE(bits.Insert(200));
    REQUIRE(bits.Insert(64));
    REQUIRE(bits.Count() == 3);
    std::vector<u64> seen;
    bits.ForEach([&](u64 page) { seen.push_back(page); });
    REQUIRE(seen == std::vector<u64>{3, 64, 200});
    REQUIRE(bits.Test(64));
    REQUIRE_FALSE(bits.Test(65));
    bits.Clear();
    REQUIRE(bits.Count() == 0);
    REQUIRE_FALSE(bits.Test(3));
}

TEST_CASE("RaValidate: vramepoch injection copies only valid pages a decision used",
          "[nvn][record_ahead]") {
    using Vulkan::Nvn::MirrorPages;
    using Vulkan::Nvn::RaRuns;
    using Vulkan::Nvn::RaVramLog;
    constexpr u64 PG = Vulkan::Nvn::RaPageSet::PAGE;
    constexpr u64 SIZE = 8 * PG;
    RaVramLog log{SIZE, 0};
    log.tv_take.Set(0, 2 * PG);      // pages 0-1 taken while valid
    log.tv_current.Set(5 * PG, PG);  // page 5 current
    log.copied.Set(3 * PG, PG);      // page 3 copied ahead (not TV)
    MirrorPages before{SIZE};
    before.Set(0, PG);      // page 0 valid
    before.Set(5 * PG, PG); // page 5 valid (page 1 invalid: not copied)
    before.Set(3 * PG, PG);
    const RaRuns runs = Vulkan::Nvn::RaVramInjectRuns(log, before);
    REQUIRE(runs == RaRuns{{0, PG}, {5 * PG, 6 * PG}});
    // A2 looks at every page a decision used (TV_take | TV_current | CA).
    const RaRuns a2 = Vulkan::Nvn::RaVramA2Pages(log);
    REQUIRE(a2 == RaRuns{{0, 2 * PG}, {3 * PG, 4 * PG}, {5 * PG, 6 * PG}});
    // The repair with the injected runs: equal-byte copies of pages 0 and 5, page 0 valid again.
    MirrorPages pages = before;
    const auto repair = Vulkan::Nvn::RaVramRepairPlan(log, before, pages, runs);
    REQUIRE(repair.copies == RaRuns{{0, PG}, {5 * PG, 6 * PG}});
    REQUIRE(repair.revalidate == RaRuns{{0, PG}});
    REQUIRE(pages.AllValid(0, PG));
    REQUIRE(pages.AllValid(5 * PG, PG));
}

TEST_CASE("RaReadLog: a mapping lookup is not a read; an access outside the note is unlogged",
          "[nvn][record_ahead]") {
    namespace PreRec = VideoCommon::Nvn::PreRec;
    PreRec::RaReadStats& stats = PreRec::RaStats();
    const u64 unlogged0 = stats.unlogged;
    const u64 mapping0 = stats.mapping_lookups;
    PreRec::Detail::ra_ahead = true;
    PreRec::RaResetScope();
    PreRec::RaMapping(0x10000);                     // pointer taken, bytes not read
    REQUIRE(stats.unlogged == unlogged0);
    REQUIRE(stats.mapping_lookups == mapping0 + 1);
    PreRec::RaNote(PreRec::Cls::UniformWord, 0x20000, 64);
    PreRec::RaCheck(0x20000, 64);                   // covered by the op's note
    PreRec::RaCheck(0x20010, 1);
    REQUIRE(stats.unlogged == unlogged0);
    PreRec::RaCheck(0x20040, 1);                    // one byte past the note
    REQUIRE(stats.unlogged == unlogged0 + 1);
    PreRec::RaResetScope();
    PreRec::RaCheck(0x20000, 4);                    // the next op: the note no longer covers it
    REQUIRE(stats.unlogged == unlogged0 + 2);
    PreRec::Detail::ra_ahead = false;
    PreRec::RaCheck(0x30000, 4);                    // not recording ahead: nothing counted
    PreRec::RaMapping(0x30000);
    REQUIRE(stats.unlogged == unlogged0 + 2);
    REQUIRE(stats.mapping_lookups == mapping0 + 1);
}

TEST_CASE("RaVram A2: a page written inside the repair's collect is not a positive (ra2g1)",
          "[nvn][record_ahead]") {
    using Vulkan::Nvn::RaVramA2Classify;
    using Vulkan::Nvn::RaVramA2Verdict;
    struct Sink final : Core::Memory::PageClaimSink {
        void ApplyClaim(u64, u64, Core::Memory::PageClaim) override {}
        bool DeviceCached(u64) const override {
            return false;
        }
    } sink;
    // The real tracker of the pool mirrors, pages 100-107 = pool pages 0-7.
    Core::Memory::PageTrackerState tracker{sink};
    const u64 id = tracker.Register(100, 108, false);
    REQUIRE(id != 0);
    tracker.Arm(id, 100, 108);
    const auto epoch = tracker.WriteEpoch(id);
    REQUIRE(epoch != nullptr);
    tracker.OnWrite(101, 102); // written during the span: the repair's collect takes it
    // MirrorCollectCpuWrites: the epoch is read first (it becomes epoch_clean), then TakeWritten.
    const u64 epoch_collect = epoch->load();
    std::vector<std::pair<u64, u64>> runs;
    tracker.CollectDirty(id, 100, 108, runs);
    REQUIRE(runs == std::vector<std::pair<u64, u64>>{{101, 102}});
    REQUIRE_FALSE(tracker.AnyDirty(id, 100, 108));
    // The guest writes pool pages 2-3 right after the TakeWritten (one write over both pages),
    // before the repair reads the epoch "after the collect".
    tracker.OnWrite(102, 104);
    const u64 epoch_after_collect = epoch->load();
    const u64 epoch_now = epoch->load();
    // ra2g1's log: dirty, not in the collected runs, epoch after the collect == epoch now.
    REQUIRE(tracker.AnyDirty(id, 102, 103));
    REQUIRE(tracker.AnyDirty(id, 103, 104));
    REQUIRE(epoch_after_collect == epoch_now);
    REQUIRE(epoch_now == epoch_collect + 2); // one increment per page turned dirty
    // The baseline read after the collect latched it; the one the collect read does not.
    REQUIRE(RaVramA2Classify(true, true, true, epoch_after_collect, epoch_now) ==
            RaVramA2Verdict::Positive);
    REQUIRE(RaVramA2Classify(true, true, true, epoch_collect, epoch_now) ==
            RaVramA2Verdict::WrittenAfterCollect);
    // The invariant the check stands on: after a whole collect with no write since its epoch
    // read, no page is dirty.
    const u64 epoch_full = epoch->load();
    runs.clear();
    tracker.CollectDirty(id, 100, 108, runs);
    REQUIRE(runs == std::vector<std::pair<u64, u64>>{{102, 104}});
    REQUIRE_FALSE(tracker.AnyDirty(id, 100, 108));
    REQUIRE(epoch->load() == epoch_full);
    // Still caught: a collect that misses a dirty page (here a partial range) leaves it dirty with
    // the epoch unmoved since the collect's read.
    tracker.OnWrite(106, 107);
    tracker.OnWrite(101, 102);
    const u64 epoch_partial = epoch->load();
    runs.clear();
    tracker.CollectDirty(id, 100, 104, runs);
    REQUIRE(runs == std::vector<std::pair<u64, u64>>{{101, 102}});
    REQUIRE(tracker.AnyDirty(id, 106, 107));
    REQUIRE(RaVramA2Classify(true, true, true, epoch_partial, epoch->load()) ==
            RaVramA2Verdict::Positive);
    // Not a finding: invalid in the mirror, or not written. No write epoch: positive.
    REQUIRE(RaVramA2Classify(false, true, true, 1, 2) == RaVramA2Verdict::Clean);
    REQUIRE(RaVramA2Classify(true, false, true, 1, 1) == RaVramA2Verdict::Clean);
    REQUIRE(RaVramA2Classify(true, true, false, 1, 2) == RaVramA2Verdict::Positive);
}
