// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>

#include "common/logging.h"
#include "core/core.h"
#include "core/file_sys/control_metadata.h"
#include "core/file_sys/nx_archive.h"
#include "core/file_sys/common_funcs.h"
#include "core/file_sys/romfs.h"
#include "core/file_sys/romfs_factory.h"
#include "core/file_sys/vfs/vfs.h"
#include "core/file_sys/vfs/vfs_vector.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "core/loader/deconstructed_rom_directory.h"
#include "core/loader/nx.h"

namespace Loader {
namespace {

template <std::size_t N>
void CopyText(std::array<char, N>& destination, std::string_view text) {
    const std::size_t length = (std::min)(text.size(), destination.size() - 1);
    std::memcpy(destination.data(), text.data(), length);
    destination[length] = '\0';
}

/// Control data for an archive that stores none: just what its catalog names.
std::unique_ptr<FileSys::NACP> BuildNacp(const FileSys::NxArchive::Selection& selection) {
    if (selection.title.empty() && selection.publisher.empty() &&
        selection.display_version.empty()) {
        return nullptr;
    }
    auto nacp = std::make_unique<FileSys::NACP>();
    nacp->raw.supported_languages = static_cast<FileSys::SupportedLanguage>(0x3FFFF);
    auto& entry = nacp->raw.language_entries.language_entries[0];
    CopyText(entry.application_name, selection.title);
    CopyText(entry.developer_name, selection.publisher);
    CopyText(nacp->raw.version_string, selection.display_version);
    nacp->raw.save_data_owner_id = selection.application_id;

    nacp->language_entries.resize(16);
    std::memcpy(nacp->language_entries.data(), nacp->raw.language_entries.language_entries.data(),
                sizeof(nacp->raw.language_entries.language_entries));
    return nacp;
}

} // namespace

AppLoader_NX::AppLoader_NX(FileSys::VirtualFile file_)
    : AppLoader(std::move(file_)), archive(FileSys::NxArchive::Open(file)) {
    if (archive == nullptr) {
        return;
    }
    const auto& selection = archive->GetSelection();
    if (selection.native) {
        LOG_ERROR(Loader, "NX archive stores NCA files; only exefs/romfs trees are supported");
        return;
    }
    if (!selection.bootable) {
        LOG_ERROR(Loader, "NX archive has no base or update program");
        return;
    }
    title_id = selection.application_id;
    has_program = true;

    if (auto bytes = archive->ReadNacp(); bytes.size() >= sizeof(FileSys::RawNACP)) {
        nacp_file = std::make_unique<FileSys::NACP>(
            std::make_shared<FileSys::VectorVfsFile>(std::move(bytes), "control.nacp"));
    } else {
        nacp_file = BuildNacp(selection);
    }
    icon = archive->ReadIcon();
}

AppLoader_NX::~AppLoader_NX() = default;

FileType AppLoader_NX::IdentifyType(const FileSys::VirtualFile& nx_file) {
    return FileSys::NxArchive::IsNxFile(nx_file) ? FileType::NX : FileType::Error;
}

bool AppLoader_NX::IsBootable(const FileSys::VirtualFile& nx_file) {
    const auto opened = FileSys::NxArchive::Open(nx_file);
    return opened != nullptr && opened->GetSelection().bootable;
}

ResultStatus AppLoader_NX::Mount() {
    if (archive == nullptr) {
        return ResultStatus::ErrorNXBadFormat;
    }
    if (!has_program) {
        return ResultStatus::ErrorNXNoProgram;
    }
    if (secondary_loader != nullptr) {
        return ResultStatus::Success;
    }
    root = archive->Mount();
    if (root == nullptr) {
        return ResultStatus::ErrorNXBadFormat;
    }
    const auto exefs = root->GetSubdirectory("exefs");
    if (exefs == nullptr || exefs->GetFile("main") == nullptr ||
        exefs->GetFile("main.npdm") == nullptr) {
        LOG_ERROR(Loader, "NX archive has no usable ExeFS");
        return ResultStatus::ErrorNXNoProgram;
    }
    secondary_loader = std::make_unique<AppLoader_DeconstructedRomDirectory>(exefs, true);
    return ResultStatus::Success;
}

AppLoader::LoadResult AppLoader_NX::Load(Kernel::KProcess& process, Core::System& system) {
    if (is_loaded) {
        return {ResultStatus::ErrorAlreadyLoaded, {}};
    }
    if (const auto status = Mount(); status != ResultStatus::Success) {
        return {status, {}};
    }

    const auto result = secondary_loader->Load(process, system);
    if (result.first != ResultStatus::Success) {
        return result;
    }

    system.GetFileSystemController().RegisterProcess(
        process.GetProcessId(), title_id,
        std::make_shared<FileSys::RomFSFactory>(*this, system.GetContentProvider(),
                                                system.GetFileSystemController()));

    // Add-on content (dlc/<title id>/<version>/romfs in the packed tree) is served to the game
    // through aoc:u and OpenDataStorageByDataId.
    for (const auto& add_on : archive->GetAddOns()) {
        const auto add_on_root = archive->MountSnapshot(add_on.snapshot);
        auto romfs = add_on_root ? add_on_root->GetSubdirectory("romfs") : nullptr;
        if (romfs == nullptr) {
            romfs = add_on_root;
        }
        if (romfs == nullptr ||
            (romfs->GetFiles().empty() && romfs->GetSubdirectories().empty())) {
            continue;
        }
        // aoc:u looks add-ons up by the running program, which may sit outside its application's
        // range (Kirby S2 Edition: application 01004D300C5AE000, program 04004D300C5AE001).
        system.GetFileSystemController().RegisterLooseAddOnContent(
            FileSys::GetBaseTitleID(archive->GetSelection().program_id), add_on.title_id,
            FileSys::CreateRomFS(romfs));
    }

    is_loaded = true;
    return result;
}

ResultStatus AppLoader_NX::VerifyIntegrity(std::function<bool(size_t, size_t)> progress_callback) {
    if (archive == nullptr) {
        return ResultStatus::ErrorNXBadFormat;
    }
    // The reader checks the whole archive in one call and reports no progress.
    if (progress_callback && !progress_callback(0, 1)) {
        return ResultStatus::ErrorIntegrityVerificationFailed;
    }
    return archive->VerifyAll() ? ResultStatus::Success
                                : ResultStatus::ErrorIntegrityVerificationFailed;
}

ResultStatus AppLoader_NX::ReadRomFS(FileSys::VirtualFile& out_file) {
    if (Mount() != ResultStatus::Success) {
        return ResultStatus::ErrorNoRomFS;
    }
    const auto romfs = root->GetSubdirectory("romfs");
    if (romfs == nullptr) {
        return ResultStatus::ErrorNoRomFS;
    }
    out_file = FileSys::CreateRomFS(romfs);
    return out_file != nullptr ? ResultStatus::Success : ResultStatus::ErrorNoRomFS;
}

bool AppLoader_NX::IsRomFSUpdatable() const {
    // The selected version is already a complete tree.
    return false;
}

ResultStatus AppLoader_NX::ReadProgramId(u64& out_program_id) {
    if (!has_program) {
        return ResultStatus::ErrorNotInitialized;
    }
    out_program_id = title_id;
    return ResultStatus::Success;
}

ResultStatus AppLoader_NX::ReadProgramIds(std::vector<u64>& out_program_ids) {
    if (!has_program) {
        return ResultStatus::ErrorNotInitialized;
    }
    out_program_ids.push_back(title_id);
    return ResultStatus::Success;
}

ResultStatus AppLoader_NX::ReadIcon(std::vector<u8>& buffer) {
    if (icon.empty()) {
        return ResultStatus::ErrorNoIcon;
    }
    buffer = icon;
    return ResultStatus::Success;
}

ResultStatus AppLoader_NX::ReadTitle(std::string& title) {
    if (nacp_file == nullptr) {
        return ResultStatus::ErrorNoControl;
    }
    title = nacp_file->GetApplicationName();
    if (title.empty() && archive != nullptr) {
        title = archive->GetSelection().title;
    }
    return title.empty() ? ResultStatus::ErrorNoControl : ResultStatus::Success;
}

ResultStatus AppLoader_NX::ReadControlData(FileSys::NACP& control) {
    if (nacp_file == nullptr) {
        return ResultStatus::ErrorNoControl;
    }
    control = *nacp_file;
    return ResultStatus::Success;
}

ResultStatus AppLoader_NX::ReadNSOModules(Modules& modules) {
    if (Mount() != ResultStatus::Success) {
        return ResultStatus::ErrorNotInitialized;
    }
    return secondary_loader->ReadNSOModules(modules);
}

} // namespace Loader
