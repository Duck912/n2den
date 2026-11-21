// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/loader/loader.h"

namespace FileSys {
class NACP;
class NxArchive;
} // namespace FileSys

namespace Loader {

/// Loads an NX archive (https://github.com/fab918/nxformat) that stores a game as loose exefs/ and
/// romfs/ trees. The newest update in the archive is booted; each NX version is a complete view,
/// so there is no layering to do here.
class AppLoader_NX final : public AppLoader {
public:
    explicit AppLoader_NX(FileSys::VirtualFile file_);
    ~AppLoader_NX() override;

    /// FileType::NX, or FileType::Error if the file is not an NX archive.
    static FileType IdentifyType(const FileSys::VirtualFile& nx_file);

    /// Whether the archive holds a program this loader can boot.
    static bool IsBootable(const FileSys::VirtualFile& nx_file);

    FileType GetFileType() const override {
        return FileType::NX;
    }

    LoadResult Load(Kernel::KProcess& process, Core::System& system) override;

    ResultStatus VerifyIntegrity(std::function<bool(size_t, size_t)> progress_callback) override;

    ResultStatus ReadRomFS(FileSys::VirtualFile& out_file) override;
    bool IsRomFSUpdatable() const override;
    ResultStatus ReadProgramId(u64& out_program_id) override;
    ResultStatus ReadProgramIds(std::vector<u64>& out_program_ids) override;
    ResultStatus ReadIcon(std::vector<u8>& buffer) override;
    ResultStatus ReadTitle(std::string& title) override;
    ResultStatus ReadControlData(FileSys::NACP& control) override;
    ResultStatus ReadNSOModules(Modules& modules) override;

private:
    /// Loads the file tree and prepares the ExeFS loader. Not done for a game list scan.
    ResultStatus Mount();

    std::shared_ptr<FileSys::NxArchive> archive;
    FileSys::VirtualDir root;
    std::unique_ptr<AppLoader> secondary_loader;
    std::unique_ptr<FileSys::NACP> nacp_file;
    std::vector<u8> icon;
    u64 title_id{};
    bool has_program{};
};

} // namespace Loader
