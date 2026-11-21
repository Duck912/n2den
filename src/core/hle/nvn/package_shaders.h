// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <map>
#include <span>
#include <string>

#include "common/common_types.h"
#include "core/file_sys/vfs/vfs_types.h"

namespace Core::NvnHle {

/// One NVN2 shader stage found in a title's package: the control section (always 0x100 bytes, as
/// the recorder copies it from guest memory; zero-padded past the end of its container) and the
/// code area the control section sizes (+0x44). Return false to end the scan.
using PackageStageVisitor =
    std::function<bool(u32 stage, std::span<const u8> control, std::span<const u8> code)>;

struct PackageShaderScan {
    u64 files{};              ///< files in the tree
    u64 files_parsed{};       ///< files read whole (a container was recognised or hinted)
    u64 bytes_read{};         ///< bytes read from the package (heads included)
    u64 bytes_inflated{};     ///< zstd output
    u64 stages{};             ///< stages handed to the visitor (duplicates included)
    u64 dictionaries{};       ///< zstd dictionaries loaded
    u64 skipped_files{};      ///< read or inflation failures, too large
    u64 skipped_containers{}; ///< recognised but malformed (never fatal)
    u64 zstd_undecodable{};   ///< zstd files whose first bytes did not inflate (e.g. no dictionary)
    std::map<std::string, u64> containers; ///< parsed containers by kind
    double seconds{};
    bool stopped{}; ///< the visitor ended the scan

    [[nodiscard]] std::string Summary() const;
};

/// Walks `root` (a mounted RomFS: base and update as the game sees it) for NVN2 shader stages in
/// the containers games ship them in: BNSH (also inside other files, e.g. shader archives and
/// effect sets), agl's BAHS, NDSR and NDVE, SARC archives of any of these, and zstd frames (with
/// the title's dictionaries when it has a "ZsDic" pack). Container layouts follow the format
/// notes of Longshot's contributors (nx2/bake_ds/romenum.py). Unknown or malformed containers
/// are counted and skipped; nothing is written anywhere.
PackageShaderScan ScanPackageShaders(const FileSys::VirtualDir& root,
                                     const PackageStageVisitor& visit);

} // namespace Core::NvnHle
