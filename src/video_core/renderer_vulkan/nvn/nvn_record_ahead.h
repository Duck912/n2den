// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Record-ahead (EDEN_NX2_NVN_RECORD_AHEAD=1, opt-in, off by default).
// When a stream reaches a WaitEvent on an event
// word the guest signals from the CPU, and the word is not signalled yet, the replay may pass the
// wait provisionally and keep recording instead of idling. From then on everything that stream's
// queue records is HELD (no vkQueueSubmit, no flight marked submitted), and so is everything a
// queue records after it waited on held work (contagion). Other queues, and what any queue
// recorded before, are submitted as before. When the guest's CPU changes satisfy every wait passed
// provisionally, the held segments are submitted in recording order in one vkQueueSubmit. A
// misprediction keeps holding (no rollback). Held queues record ahead op by op only while each op
// is on the allow-list; anything else ends recording ahead for this gate and the held streams idle
// until the signal as before.

#pragma once

#include <array>
#include <chrono>
#include <memory>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "video_core/nvn/record_ahead_gate.h"
#include "video_core/renderer_vulkan/nvn/nvn_ra_validate.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan::Nvn {

struct InFlight;
struct Pool;

/// Why recording ahead stopped (counted per gate).
enum class RaStop : u32 {
    None = 0,
    Op,            ///< an op not on the allow-list
    Present,       ///< a present item while a gate is open
    NewSubmission, ///< a new submission begins on a held queue
    UnseenEvent,   ///< a wait/signal on an event the model has not seen (it would read guest memory)
    GateSignal,    ///< a GPU signal of a gate word
    CmdBufCap,     ///< held command buffer cap
    TimeCap,       ///< ahead recording time cap
    Snapshot,      ///< uniform snapshot active for a held submission
    Precondition,  ///< a frame-level precondition (debug/diag/rescale/no conditional rendering...)
    Tripwire,      ///< a sync helper ran while holding (FlushAndWait, Abort, held-seq wait)
    RareClass,     ///< stage 2: an op predicted to read a rare class (GenIndex by draw mode)
    TexPoolGen,    ///< stage 2: the queue's texture pool set resolve generation moved ahead
    PoolContents,  ///< stage 2: a pool upload or sparse growth is due (PoolUpload / SparseGrowth)
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(RaStop::Count)> RA_STOP_NAMES{
    "none",       "op",       "present",  "new-submission", "unseen-event", "gate-signal",
    "cmdbuf-cap", "time-cap", "snapshot", "precondition",   "tripwire",
    "rare-class", "texpool-gen", "pool-contents",
};

/// One vkQueueSubmit worth of segments a held queue recorded.
struct RaHeldBatch {
    struct Seg {
        VkCommandBuffer raw{};
        std::vector<VkSemaphore> waits;
        std::vector<u64> wait_values;
        u64 signal_value{};
    };
    InFlight* flight{};
    VkSemaphore timeline{};
    std::vector<Seg> segs;
    bool mark{false}; ///< the recorder finished: MarkSubmitted at release
    u64 mark_value{};
};

/// A private uniform copy recorded ahead (UniformCopy): guest bytes memcpy'd into a held flight's
/// arena with the inline updates applied on top. Late-bound: taken again at the release into the
/// same memory (held commands have not run), the updates re-applied in order.
struct RaUniformCopy {
    u8* pointer{};
    u64 address{};
    u64 size{};
    std::vector<std::pair<u64, std::vector<u8>>> patches; ///< (offset in the copy, bytes), in order
    /// The bytes written to `pointer` at record time (guest bytes with the patches applied), kept
    /// in host memory: `pointer` is mapped upload memory (slow to read back), and nothing else
    /// writes it before the release (held commands have not run).
    std::vector<u8> written;
};

/// Learned facts about one event word the guest changes from the CPU.
struct RaWord {
    u64 cpu_changes{};
    u64 stable{};                          ///< consecutive CPU changes that satisfied every blocked wait
    std::vector<std::pair<u32, u32>> pending; ///< (mode, value) of waits blocked since the last change
};

/// Ops that only change the recording queue's own state or the replay's sync model (contagion
/// covers the latter): a non-held queue may record them while held work touched resources.
[[nodiscard]] bool RaPureOp(u16 op) noexcept;

/// EDEN_NX2_NVN_HEADTRACE=1 (measurement only): after each CPU event change that unblocks a
/// stream (the frame's gate), every op the replay records within HEAD_MS: count and recording time
/// per queue and op kind, and the record-time read classes (PreRec::Cls) the ops touch: first use
/// per queue (up to END_MS) and the recording time of the ops that touched each class.
struct HeadTrace {
    static constexpr u32 QUEUES = 8;
    static constexpr u32 OPS = 96;
    static constexpr u16 SUBMIT_CHECK = OPS - 1; ///< pseudo kind: the early-submit check (+ submit)
    static constexpr u32 CLASSES = 18;
    static constexpr float HEAD_MS = 5.0f;
    static constexpr float END_MS = 8.0f;
    bool enabled{};
    bool on{};
    std::chrono::steady_clock::time_point t0{};
    bool have_prev{};
    u32 prev_queue{};
    u16 prev_op{};
    float prev_ms{};
    std::chrono::steady_clock::time_point prev_t{};
    std::array<u32, QUEUES> seen{}; ///< classes used so far in this head, per queue
    // Accumulated over the heads since the last log line.
    u64 heads{};
    std::array<u64, QUEUES> rec_ns{};
    std::array<std::array<u64, OPS>, QUEUES> op_count{};
    std::array<std::array<u64, OPS>, QUEUES> op_ns{};
    std::array<std::array<u64, CLASSES>, QUEUES> cls_ns{};
    std::array<std::array<std::vector<float>, CLASSES>, QUEUES> cls_first{};
    u64 last_log_present{};
};

/// RaStopNow's queue argument for a stop of every held queue.
inline constexpr u32 RA_ALL_QUEUES = ~u32{0};

/// One copy of the release prologue (repairs: guest -> VRAM mirror copy-ins).
struct RaPrologueCopy {
    VkBuffer src{};
    VkBuffer dst{};
    VkBufferCopy region{};
};

struct RecordAhead {
    bool enabled{};
    bool verify{};
    bool latched_off{};
    u32 inject{}; ///< EDEN_NX2_NVN_RA_INJECT: 1 = mispredict, 2 = unlogged, 3 = prologue,
                  ///< 4 = vramepoch
    /// EDEN_NX2_NVN_RA_VALIDATE_ONLY=1 (stage 2, measurement only): draws and dispatches are
    /// recorded ahead; the read classes without an exact repair yet are checked at the release
    /// (nvn_ra_validate.h) and the first mismatch latches off. Not exact under a change.
    bool validate_only{};
    /// Suspend / Finish of a held queue's recorder (held work recorded after RecordList
    /// returned): its mirror decisions are recorded ahead too (frozen view).
    bool tail_held{};
    std::unordered_map<u64, RaWord> words;

    // Open gate.
    VideoCommon::Nvn::RecordAheadGate<RaHeldBatch> gate;
    /// Recording ahead stopped for every held queue (a gate-wide reason: present, caps,
    /// snapshot, tripwire): no queue joins or becomes held any more.
    bool stopped{};
    RaStop stop_reason{RaStop::None};
    /// Held queues that stopped recording ahead on their own (an op not allowed, an unseen event,
    /// a gate signal, a new submission): the others keep recording ahead, and a stopped queue does
    /// not block another queue's contagion.
    std::vector<std::pair<u32, RaStop>> stopped_queues;
    [[nodiscard]] bool Stopped(u32 queue) const noexcept {
        if (stopped) {
            return true;
        }
        for (const auto& entry : stopped_queues) {
            if (entry.first == queue) {
                return true;
            }
        }
        return false;
    }
    std::chrono::steady_clock::time_point start{};
    u64 held_cmdbufs{};
    u64 frame_ops{};
    u64 frame_draws{};
    bool frame_mispredicted{};
    /// A held queue recorded an op that touches resources (counter writes into pools): non-held
    /// queues' resource ops are then held behind it (ordering contagion), and E8 commands are
    /// exactness failures.
    bool held_touched{};
    /// EDEN_NX2_NVN_RA_INJECT=mispredict (verify mode): this gate's release is deferred once.
    bool inject_deferred{};
    /// Uniform and handle words read on the CPU while recording ahead (bindless handles, ref and
    /// offset words: UniformWord, HandleMem). Their value chose what held commands bind, which
    /// cannot be repaired: every pair is compared with guest memory at the release, in production
    /// too; a mismatch latches record-ahead off for the boot (logged once).
    struct HandleWord {
        u64 address{};
        u32 value{};
        bool ok{}; ///< the read succeeded
    };
    std::vector<HandleWord> handle_words;
    std::vector<RaUniformCopy> uniform_copies;

    /// Release prologue (design 2.4.5): one command buffer submitted before every held batch,
    /// with a leading ALL_COMMANDS barrier; it signals `prologue_timeline` and the first held
    /// segment of every held queue waits for that value. Recorded only when there is a copy.
    std::vector<RaPrologueCopy> prologue_copies;
    vk::Semaphore prologue_timeline;
    u64 prologue_value{};
    /// Submitted prologue command buffers and their timeline values (back to the free list once
    /// the value is reached).
    std::vector<std::pair<VkCommandBuffer, u64>> prologue_retire;
    /// EDEN_NX2_NVN_RA_INJECT=prologue: two pages; every release's prologue copies the first into
    /// the second.
    vk::Buffer inject_scratch;

    // Stage 2 validate-only logs (nvn_ra_validate.h), filled while recording ahead, checked and
    // cleared at the release.
    struct MirrorPool {
        std::shared_ptr<Pool> pool;
        RaPageBits pages; ///< Backing::Mirror pages a decision recorded ahead used
    };
    std::vector<MirrorPool> mirror_pools;
    struct MirrorCopy {
        u8* pointer{}; ///< staging bytes (held: the copy has not run)
        std::shared_ptr<Pool> pool;
        u64 begin{};
        u64 length{};
    };
    std::vector<MirrorCopy> mirror_copies;
    struct TexCheck {
        std::shared_ptr<const Pool> pool;
        u64 storage_address{}; ///< identity
        u64 offset{};
        u64 size{};
        u64 signature{};
        u64 full_hash{};
        bool full{};
    };
    std::vector<TexCheck> tex_checks;
    struct TexBytes {
        std::shared_ptr<const Pool> pool;
        u64 offset{};
        u64 size{};
        u64 hash{};
    };
    mutable std::vector<TexBytes> tex_bytes;
    mutable std::array<RaCheckStats, static_cast<size_t>(RaCheck::Count)> checks{};
    /// PreRec::RaStats().calls of the rare classes as of the last release.
    std::array<u64, static_cast<size_t>(VideoCommon::Nvn::PreRec::Cls::Count)> rare_calls{};
    /// Texture pool set generation per held queue at its first draw-path op ahead.
    std::vector<std::pair<u32, u64>> texpool_gen;
    u64 latches{};           ///< latch events logged (all classes)
    u64 validate_skipped{};  ///< releases with guest memory stopped (nothing validated)
    u64 lost_entries{};      ///< logged entries whose pool went away before the release
    u64 e8_validated{};      ///< NotifyHostWrite while held work touched resources (validate-only)
    u64 dispatches_ahead{};
    u64 gates_with_draws{};
    u64 frame_dispatches{};
    u64 ns_ops_ahead{};      ///< recording time of the ops recorded ahead (head moved)
    u64 ns_ops_ahead_draw_gates{}; ///< ... in gates that recorded a draw or dispatch ahead
    u64 ns_gate_ops{};       ///< ... in the open gate
    std::chrono::steady_clock::time_point op_t0{};
    bool op_timer{};
    u64 ns_validate{};       ///< release time of the stage 2 checks
    // Diagnostics: unlogged reads by the op kind recorded ahead when they happened (80 = outside
    // an op), with the latest va / size; release time split.
    u16 last_op{0xFFFF};
    u64 unlogged_seen{};
    std::array<u64, 81> unlogged_by_op{};
    std::array<u64, 81> unlogged_va{};
    std::array<u64, 81> unlogged_size{};
    u64 ns_rel_words{};      ///< handle words + uniform copies
    u64 ns_rel_handles{};    ///< of which: handle-word validation
    u64 ns_rel_ucopies{};    ///< of which: uniform-copy late binding
    u64 ucopy_bytes{};       ///< bytes of the uniform copies late-bound at releases
    u64 raw_rebuilds_ahead{};     ///< raw global table rebuilds while recording ahead
    u64 raw_rebuild_last_present{}; ///< present of the latest of them
    u64 pools_backed_ahead{};     ///< BackPoolLocked on the replay thread while recording ahead
    u64 vram_a2_after_collect{}; ///< A2 positives written by the guest after the repair's collect
    u64 vram_a2_in_collect{};    ///< of them, written inside the collect (after its epoch read)
    u64 rel_flag_ahead{};    ///< releases entered with the thread's ra_ahead flag still set
    u64 ns_rel_vram{};       ///< VRAM repair
    u64 ns_rel_prologue{};   ///< prologue recording
    u64 ns_rel_submit{};     ///< vkQueueSubmit (with the submit mutex)
    u64 ns_rel_mark{};       ///< MarkSubmitted of the held flights
    u64 rel_segments{};      ///< segments submitted at releases
    [[nodiscard]] bool Active() const noexcept {
        return gate.Active();
    }
    /// What `queue` records now is held.
    [[nodiscard]] bool Held(u32 queue) const noexcept {
        return gate.Held(queue);
    }

    // Counters (cumulative; one log line every 600 presents).
    std::array<u64, static_cast<size_t>(RaStop::Count)> stops{};
    u64 gates{};
    u64 releases{};
    u64 releases_timeout{};
    u64 mispredicts{};
    u64 ops_ahead{};
    u64 draws_ahead{};
    u64 yields{};
    u64 held_batches{};
    u64 held_segments{};
    u64 provisional_waits{};   ///< waits passed provisionally (all gate words)
    u64 gate_words_max{};      ///< most distinct gate words in one gate
    u64 contagion_holds{};     ///< queues made held by a wait on held work
    u64 contagion_blocks{};    ///< waits on held work that blocked (recording ahead stopped)
    u64 order_violations{};    ///< non-held submissions that waited on held values (must be 0)
    u64 ordering_holds{};      ///< non-held queues held because they touched resources after held work
    u64 e8_commands{};         ///< MapVirtual/host write/TextureWritten/FlushTexture while a gate was open
    u64 e8_failures{};         ///< ... while held work had touched resources (must be 0)
    u64 injected{};            ///< fault injection: deferred releases / unlogged reads made
    u64 ops_by_kind[80]{};     ///< ops recorded ahead per op number (verify log)
    u64 dropped{};             ///< gates ended without submitting (device loss, shutdown)
    u64 immediate_submits{};
    u64 tripwires{};
    u64 handle_words_logged{};
    u64 handle_word_mismatches{}; ///< must be 0 (exactness failure, latched off)
    u64 uniform_copies_logged{};
    u64 uniform_copies_changed{};  ///< A1: guest bytes changed before the signal (re-taken)
    u64 uniform_copies_different{}; ///< A2: still different after the re-take (must be 0)
    u64 queue_stops{};            ///< per-queue stops (the other held queues kept recording)
    u64 prologue_submits{};
    u64 prologue_copies_total{};
    u64 prologue_bytes{};
    u64 prologue_waits_patched{}; ///< first held segments given the prologue wait
    u64 prologue_failed{};        ///< prologue recording threw (release went on without it)
    // VRAM frozen write view and release repair.
    u64 vram_logged_pools{};      ///< pools a decision recorded ahead used (first decisions)
    u64 vram_frozen_decisions{};  ///< decisions recorded ahead on logged pools
    u64 vram_unheld_decisions{};  ///< decisions not recorded ahead on a logged pool (must be 0)
    u64 vram_deferred_commands{}; ///< CpuWrite/NotifyHostWrite commands deferred to the release
    u64 vram_repairs{};           ///< logged pools whose epoch moved or that had deferred commands
    u64 vram_changed_pages{};
    u64 vram_copy_pages{};        ///< prologue copy-ins
    u64 vram_revalidated_pages{};
    u64 vram_current_only_pages{};
    u64 vram_stopped_skips{};     ///< repairs with guest memory stopped (nothing copied)
    u64 vram_lost_pools{};        ///< logged pools whose mirror went away before the release
    u64 vram_abandoned{};         ///< logged pools of a gate dropped without a release
    u64 ns_release_late_bind{};
    u64 ns_ahead{};      ///< recording time spent ahead
    u64 ns_release{};    ///< time spent in the release step
    u64 frames_with_ahead{};
    u64 last_log_present{};
};

} // namespace Vulkan::Nvn
