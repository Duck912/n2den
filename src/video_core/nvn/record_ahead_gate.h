// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/nvn/queue_sync.h"

namespace VideoCommon::Nvn {

// Record-ahead gate bookkeeping (EDEN_NX2_NVN_RECORD_AHEAD; no Vulkan in here, so the rules are
// unit tested in src/tests/nvn/record_ahead.cpp).
//
// A gate is the set of event words the guest changes from the CPU that the replay passed waits on
// provisionally. A queue is HELD from the moment one of its streams passed such a wait, or from
// the moment one of its streams recorded a wait on work that is held (contagion: its event or sync
// dependency names a held segment). Only what a held queue records from then on is held; what any
// queue recorded before (and everything non-held queues record) is submitted as usual. Held work
// is released in recording order when every provisional wait is satisfied by the words' CPU values
// (no rollback: a change that does not satisfy them keeps everything held).
template <typename Batch>
class RecordAheadGate {
public:
    struct Provisional {
        u64 address{};
        u32 mode{};
        u32 value{};
        u32 queue{};
        u64 seq{};
    };

    [[nodiscard]] bool Active() const noexcept {
        return active;
    }

    /// Whether what `queue` records now is held.
    [[nodiscard]] bool Held(u32 queue) const noexcept {
        return active && std::ranges::any_of(held_from, [queue](const auto& entry) {
                   return entry.first == queue;
               });
    }

    /// The first held timeline value of `queue` (0 when the queue is not held).
    [[nodiscard]] u64 HeldFrom(u32 queue) const noexcept {
        for (const auto& [q, value] : held_from) {
            if (q == queue) {
                return value;
            }
        }
        return 0;
    }

    /// Whether a wait for timeline value `value` of `queue` names held work: a stream that records
    /// such a wait becomes held (contagion).
    [[nodiscard]] bool NamesHeld(u32 queue, u64 value) const noexcept {
        if (!active) {
            return false;
        }
        const u64 from = HeldFrom(queue);
        return from != 0 && value >= from;
    }

    /// A stream of `queue` passed a wait on a CPU-gated word provisionally; the queue's segments
    /// from timeline value `first_held` on are held. Opens the gate if needed.
    void Pass(const Provisional& wait, u64 first_held) {
        active = true;
        provisional.push_back(wait);
        HoldQueue(wait.queue, first_held);
    }

    /// `queue` is held from timeline value `first_held` on (no-op when it is held already).
    void HoldQueue(u32 queue, u64 first_held) {
        active = true;
        if (HeldFrom(queue) == 0) {
            held_from.emplace_back(queue, first_held);
        }
    }

    /// A batch a held queue submitted (kept in call order across all queues).
    void Hold(u32 queue, u64 seq, Batch&& batch) {
        min_seq = held.empty() ? seq : std::min(min_seq, seq);
        held.push_back({queue, seq, std::move(batch)});
    }

    /// Lowest flight seq among held batches (~0 when none): a CPU wait for a completion at or
    /// after it would wait for held work.
    [[nodiscard]] u64 MinHeldSeq() const noexcept {
        return held.empty() ? ~u64{0} : min_seq;
    }

    [[nodiscard]] bool IsGateWord(u64 address) const noexcept {
        return std::ranges::any_of(provisional,
                                   [address](const Provisional& p) { return p.address == address; });
    }

    /// Every provisional wait is satisfied by `word(address)` (the CPU value as the event model
    /// has it after the guest's change).
    [[nodiscard]] bool AllSatisfied(const std::function<u32(u64)>& word) const {
        return std::ranges::all_of(provisional, [&word](const Provisional& p) {
            return EventModel::Satisfies(word(p.address), p.mode, p.value);
        });
    }

    struct Entry {
        u32 queue{};
        u64 seq{};
        Batch batch;
    };

    /// Ends the gate: the held batches in recording order (the caller submits them in this order,
    /// or fails their flights), and every queue is no longer held.
    [[nodiscard]] std::vector<Entry> Take() {
        std::vector<Entry> out = std::move(held);
        held.clear();
        provisional.clear();
        held_from.clear();
        active = false;
        min_seq = 0;
        return out;
    }

    /// The batch held last (nullptr when none).
    [[nodiscard]] Entry* Last() noexcept {
        return held.empty() ? nullptr : &held.back();
    }

    [[nodiscard]] const std::vector<Entry>& HeldEntries() const noexcept {
        return held;
    }
    [[nodiscard]] const std::vector<Provisional>& ProvisionalWaits() const noexcept {
        return provisional;
    }
    [[nodiscard]] const std::vector<std::pair<u32, u64>>& HeldQueues() const noexcept {
        return held_from;
    }

private:
    bool active{false};
    std::vector<Provisional> provisional;
    std::vector<std::pair<u32, u64>> held_from; ///< (queue, first held timeline value)
    std::vector<Entry> held;
    u64 min_seq{0};
};

} // namespace VideoCommon::Nvn
