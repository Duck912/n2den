// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <mutex>
#include <unordered_map>

#include "common/common_types.h"

// [DEBUG] Mario Kart World uploads the same 144-682 MiB images again every frame, about 2.2 GB a
// frame on its title loop. For images above Threshold
// this records what last marked one CPU-modified (a CPU write or a GPU unmap over its range, and
// that range), so the upload log can say why it uploads again or that the image is new.
namespace VideoCommon::BigImageTrace {

inline constexpr u64 Threshold = 64ULL << 20;

struct Note {
    const char* reason{}; ///< what marked it modified since its last upload; null for a new image
    u64 address{};        ///< start of the write or unmap that did
    u64 size{};
    u32 marks{};   ///< times it was marked modified
    u32 uploads{}; ///< uploads so far, this one included
};

namespace Detail {
inline std::mutex mutex;
inline std::unordered_map<u32, Note> notes;
} // namespace Detail

/// The image `id` was marked CPU-modified by `reason` over [address, address + size).
inline void Mark(u32 id, const char* reason, u64 address, u64 size) {
    std::scoped_lock lock{Detail::mutex};
    Note& note = Detail::notes[id];
    note.reason = reason;
    note.address = address;
    note.size = size;
    ++note.marks;
}

/// Counts an upload of `id` and returns its note as it was before; the reason is cleared.
inline Note Upload(u32 id) {
    std::scoped_lock lock{Detail::mutex};
    Note& note = Detail::notes[id];
    ++note.uploads;
    const Note copy = note;
    note.reason = nullptr;
    return copy;
}

/// The image `id` was deleted: its slot may be reused by another image.
inline void Forget(u32 id) {
    std::scoped_lock lock{Detail::mutex};
    Detail::notes.erase(id);
}

} // namespace VideoCommon::BigImageTrace
