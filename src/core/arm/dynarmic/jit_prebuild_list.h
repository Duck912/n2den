// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>

#include "common/common_types.h"

namespace Core {

/// A prebuild list: blocks an offline generator found in a title's code (static discovery plus blocks seen in
/// play), for the JIT cache builder to compile before the guest first runs. It holds block keys, block ends and
/// guest-code hashes only, never guest code bytes. Read-only here; the format is the one the offline exporter
/// writes (format v1, little-endian):
///   header (48 bytes): magic "N2PL", version (4), entry size (4) = 32, generator version (4; 0 for an
///                      offline list), program id (8), image hash (8), entry count (8), common FPCR (4),
///                      manifest blocks the list was made with (4; 0 = made with no manifest, a first boot;
///                      written by the emulator's own generator from version 3, reserved otherwise);
///   entries (32 bytes): descriptor (8), end PC (8), guest hash (8), flags (2), guest instructions (2),
///                      reserved (4).
/// The descriptor is the JIT's block key (IR::LocationDescriptor value: PC, FPCR mode bits, single-step), the
/// guest hash the one Dynarmic records carry; entries come in priority order (blocks seen in play first).
/// The file is memory-mapped while open, so a list of millions of entries costs no heap.
class JitPrebuildList {
public:
    enum Flags : u16 {
        PlayedFunction = 1 << 0,  ///< in a function that ran in a recorded session
        Executed = 1 << 1,        ///< this exact key ran in a recorded session
        CommonMode = 1 << 2,      ///< FPCR is the title's common mode
        LearnedOnly = 1 << 3,     ///< not found statically; taken from the recorded keys
        Generated = 1 << 5,       ///< written by the emulator's own generator (jit_code_discovery.h)
    };

    struct Entry {
        u64 descriptor;
        u64 end_pc;
        u64 guest_hash;
        u16 flags;
    };

    static constexpr size_t header_size = 48;
    static constexpr size_t entry_size = 32;

    /// Maps and checks a list (magic, version, entry size, size, program id). nullptr if it is missing or not a
    /// valid list for `program_id`; `error` (if given) then says why (empty when the file does not exist).
    static std::unique_ptr<JitPrebuildList> Open(const std::filesystem::path& path, u64 program_id,
                                                 std::string* error = nullptr);

    /// Copies a list to `destination` (a temp file, then renamed) after checking it is a valid list for
    /// `program_id`. Does nothing if `destination` already holds a list with the same size and header. Returns
    /// true if `destination` now holds the list, and sets `changed` if it was (re)written.
    static bool Import(const std::filesystem::path& source, const std::filesystem::path& destination,
                       u64 program_id, bool& changed);

    ~JitPrebuildList();
    JitPrebuildList(const JitPrebuildList&) = delete;
    JitPrebuildList& operator=(const JitPrebuildList&) = delete;

    size_t Size() const {
        return count;
    }
    Entry At(size_t i) const;

    /// The header's bytes: identifies the list's content for "already built" checks (count, program id, image
    /// hash of the code it was generated from).
    const std::array<u8, header_size>& Header() const {
        return header;
    }

private:
    JitPrebuildList() = default;

    std::array<u8, header_size> header{};
    size_t count = 0;
    const u8* view = nullptr;
    u64 view_size = 0;
#ifdef _WIN32
    void* file_handle = nullptr;
    void* mapping_handle = nullptr;
#else
    int file_descriptor = -1;
#endif
};

} // namespace Core
