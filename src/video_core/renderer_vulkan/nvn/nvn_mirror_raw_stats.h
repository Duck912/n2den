// SPDX-FileCopyrightText: Copyright 2026 n2den Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <functional>

#include "common/common_types.h"

namespace VideoCommon::Nvn {
class GuestMemory;
}

namespace Vulkan::Nvn {

struct Pool;

/// Split-pool mirror raw-epoch statistics (EDEN_NVN_MIRROR_RAW_STATS=1; measurement only, every
/// decision is unchanged). Replay thread only. Counts, per window of presents: the Backing::Mirror
/// pages hashed again because a raw-storing submission completed (split pools: `raw-epoch`), how
/// many of them actually differed from the mirror's hash, by pool and by what consumed the range;
/// raw-storing draws/dispatches and submissions, by program; and a sweep (every window) of every
/// trusted page of every split mirror pool seen, hashed against the stored hash, to find the pages
/// GPU raw stores (or any writer the tracker does not see) changed, used or not.
namespace MirrorRawStats {

/// What consumed a mirror range (DrawPlan::RangeUse::consumer, the hint below).
enum Consumer : u8 {
    Other = 0,
    Vertex = 1,
    Index = 2,
    Uniform = 3,
    Storage = 4,
    Indirect = 5,
    Count = 6,
};

[[nodiscard]] bool Enabled();

/// The consumer of the mirror range the next MirrorNeedsUpload / UploadRange looks at (set by the
/// draw path around its calls; Other elsewhere).
inline thread_local u8 consumer_hint = Other;
/// The programs (a hash of the stages' program hashes) and the layout's write-through flag of the
/// draw whose raw stores are noted next (set by the draw path when Enabled()).
inline thread_local u64 raw_program_hint = 0;
inline thread_local bool raw_through_hint = false;

/// A hashed mirror page: `kind` 0 first, 1 untrusted, 2 raw-epoch; `changed`: the guest hash
/// differed from the mirror's (or the page was not valid).
void NotePage(Pool& pool, u64 page, u32 kind, bool changed);
/// A trusted page skipped (no hash).
void NoteTrusted(Pool& pool, u64 page);
/// A pool took a new raw epoch.
void NoteAdopted();
/// A draw/dispatch with raw global stores: `program` identifies its programs, `first_in_flight`
/// whether it is the first one of its submission, `written_through` the layout flag.
void NoteRawDraw(u64 program, bool first_in_flight, bool written_through);
/// Present `index` was recorded: logs a window and runs the sweep every N presents. `hash` is
/// Recorder::GuestPageHash; `memory` tells CPU writes the tracker holds but nobody took yet.
void OnPresent(u64 index, const std::function<u64(Pool&, u64)>& hash,
               VideoCommon::Nvn::GuestMemory* memory);

} // namespace MirrorRawStats

} // namespace Vulkan::Nvn
