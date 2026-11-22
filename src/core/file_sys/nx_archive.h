// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "core/file_sys/vfs/vfs_types.h"

namespace FileSys {

/// An NX archive (https://github.com/fab918/nxformat), read with the project's header-only reader.
/// Opening reads only the catalog: titles, versions and artwork. The file tree and the block index
/// are loaded by Mount(), which a game list scan never needs.
///
/// Only archives that store a game as loose exefs/ and romfs/ trees can be booted (what Tools >
/// Convert to NX and nxconvert's pack-nup and pack-tree write). Archives of native NCA files are
/// recognised and reported as unsupported.
class NxArchive {
public:
    /// The program this archive boots: the newest update of its first application, or the base
    /// when there is no update. Every NX snapshot is a complete view, so nothing is layered.
    struct Selection {
        bool bootable{};
        /// Stored as NCA files, which this loader does not read.
        bool native{};
        u64 application_id{};
        u64 program_id{};
        u32 snapshot{};
        u32 numeric_version{};
        std::string title;
        std::string publisher;
        std::string display_version;
    };

    /// Quick magic check without parsing.
    static bool IsNxFile(const VirtualFile& file);

    /// Returns nullptr when the file is not a readable NX archive.
    static std::shared_ptr<NxArchive> Open(VirtualFile file);

    ~NxArchive();

    const Selection& GetSelection() const;

    /// Control data (NACP) of the selected program; empty when the archive stores none.
    std::vector<u8> ReadNacp();
    /// Icon (JPEG or PNG) of the application; empty when the archive stores none.
    std::vector<u8> ReadIcon();

    /// Root directory of the selected snapshot, holding exefs/ and romfs/. The first call loads
    /// the file tree and block index. Files and directories keep the archive open on their own.
    VirtualDir Mount();

    /// Add-on content stored by its add-on title id (nxconvert pack-nup / pack-tree dlc/ folders).
    struct AddOn {
        u64 title_id{};
        u32 snapshot{};
    };
    /// Add-ons of the selected application, newest version of each title id.
    std::vector<AddOn> GetAddOns() const;
    /// Root directory of any snapshot of the archive (an add-on's holds romfs/).
    VirtualDir MountSnapshot(u32 snapshot);

    /// Reads and checks every block. Mounts the archive if needed.
    bool VerifyAll();

private:
    struct Impl;
    explicit NxArchive(std::unique_ptr<Impl> impl_);
    std::unique_ptr<Impl> impl;
};

} // namespace FileSys
