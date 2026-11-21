// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// The VFS adapters follow nxformat's own emulator integration (integration/nxemu-reader.patch,
// MIT, Copyright (c) the nxformat authors).

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <utility>

#include <zstd.h>

#include "common/fs/path_util.h"
#include "common/logging.h"
#include "core/file_sys/nx_archive.h"
#include "core/file_sys/vfs/vfs.h"

#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include <nx_disk_source.hpp>
#include <nx_session.hpp>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

// nx_disk_source.hpp brings in windows.h, whose macros rename these VFS methods.
#undef CreateFile
#undef DeleteFile
#undef CreateDirectory

namespace FileSys {
namespace {

constexpr u32 Absent = 0xffffffffu;
constexpr std::size_t MaxBlockSize = 1024 * 1024;

struct DCtxDeleter {
    void operator()(ZSTD_DCtx* context) const {
        ZSTD_freeDCtx(context);
    }
};

/// Decodes one NX block. NX allows exactly one standard frame per block: no dictionary, a declared
/// content size equal to the block, a window of at most 1 MiB and no trailing bytes.
bool DecodeBlock(void*, const std::uint8_t* frame, std::size_t size, std::uint8_t* output,
                 std::size_t capacity, std::size_t* written, std::size_t* consumed) {
    static constexpr std::array<std::uint8_t, 4> magic{0x28, 0xb5, 0x2f, 0xfd};
    if (size < 6 || capacity > MaxBlockSize || std::memcmp(frame, magic.data(), magic.size()) != 0 ||
        ZSTD_getFrameContentSize(frame, size) != capacity ||
        ZSTD_getDictID_fromFrame(frame, size) != 0 ||
        ZSTD_findFrameCompressedSize(frame, size) != size) {
        return false;
    }
    const unsigned descriptor = frame[4];
    if ((descriptor & 0x1bu) != 0) {
        return false; // Reserved bits or a dictionary ID field.
    }
    if ((descriptor & 0x20u) == 0) {
        const auto base = std::uint64_t{1} << (10 + (frame[5] >> 3));
        if (base + (base >> 3) * (frame[5] & 7u) > MaxBlockSize) {
            return false;
        }
    }
    thread_local std::unique_ptr<ZSTD_DCtx, DCtxDeleter> context;
    if (context == nullptr) {
        context.reset(ZSTD_createDCtx());
        if (context == nullptr) {
            return false;
        }
    }
    const auto result = ZSTD_decompressDCtx(context.get(), output, capacity, frame, size);
    if (ZSTD_isError(result) || result != capacity) {
        return false;
    }
    *written = result;
    *consumed = size;
    return true;
}

/// Reads through the VFS file, one read at a time. Used when the file has no host path to pin.
class VfsSource final : public nx::RandomAccessSource {
public:
    explicit VfsSource(VirtualFile file_) : file{std::move(file_)}, file_size{file->GetSize()} {}

    std::uint64_t size() const noexcept override {
        return file_size;
    }

    bool read_at(std::uint64_t offset, std::uint8_t* output, std::size_t length) override {
        if (offset > file_size || length > file_size - offset ||
            offset > (std::numeric_limits<std::size_t>::max)()) {
            return false;
        }
        std::scoped_lock lock{mutex};
        return file->Read(output, length, static_cast<std::size_t>(offset)) == length;
    }

private:
    VirtualFile file;
    std::uint64_t file_size;
    std::mutex mutex;
};

std::shared_ptr<nx::RandomAccessSource> MakeSource(const VirtualFile& file) {
    // A pinned OS handle reads in parallel; the VFS route is the fallback.
    const std::string path = file->GetFullPath();
    if (!path.empty()) {
        std::error_code ec;
        const auto host_path = Common::FS::ToU8String(path);
        if (std::filesystem::is_regular_file(std::filesystem::path{host_path}, ec)) {
            if (auto pinned = nx::DiskSource::open(std::filesystem::path{host_path})) {
                return std::move(*pinned.value);
            }
        }
    }
    return std::make_shared<VfsSource>(file);
}

class NxFile final : public VfsFile {
public:
    NxFile(std::shared_ptr<nx::Session> session_, nx::FileInfo info_, std::string name_,
           VirtualDir parent_)
        : session{std::move(session_)}, info{info_}, name{std::move(name_)},
          parent{std::move(parent_)}, cursor{std::make_shared<nx::ReadCursor>()} {}

    std::string GetName() const override {
        return name;
    }
    std::size_t GetSize() const override {
        return static_cast<std::size_t>(info.size);
    }
    bool Resize(std::size_t) override {
        return false;
    }
    VirtualDir GetContainingDirectory() const override {
        return parent;
    }
    bool IsWritable() const override {
        return false;
    }
    bool IsReadable() const override {
        return true;
    }
    std::size_t Read(u8* data, std::size_t length, std::size_t offset) const override {
        if (offset > info.size || (length != 0 && data == nullptr)) {
            return 0;
        }
        length = static_cast<std::size_t>((std::min<u64>)(length, info.size - offset));
        std::size_t done = 0;
        while (done < length) {
            const auto part = static_cast<std::size_t>(
                (std::min<u64>)(length - done, session->max_read_bytes()));
            if (part == 0 || !session->read(info.id, offset + done, part, data + done, part, cursor)) {
                break;
            }
            done += part;
        }
        return done;
    }
    std::size_t Write(const u8*, std::size_t, std::size_t) override {
        return 0;
    }
    bool Rename(std::string_view) override {
        return false;
    }

private:
    std::shared_ptr<nx::Session> session;
    nx::FileInfo info;
    std::string name;
    VirtualDir parent;
    std::shared_ptr<nx::ReadCursor> cursor;
};

class NxDirectory final : public ReadOnlyVfsDirectory,
                          public std::enable_shared_from_this<NxDirectory> {
public:
    NxDirectory(std::shared_ptr<nx::Session> session_, nx::DirectoryHandle handle_,
                std::string name_, VirtualDir parent_ = nullptr)
        : session{std::move(session_)}, handle{handle_}, name{std::move(name_)},
          parent{std::move(parent_)} {}

    std::string GetName() const override {
        return name;
    }
    VirtualDir GetParentDirectory() const override {
        return parent;
    }

    VirtualFile GetFile(std::string_view file_name) const override {
        const auto entry = session->archive().lookup(handle, file_name);
        if (!entry || entry.value->type != nx::EntryType::file) {
            return nullptr;
        }
        return MakeFile(*entry.value);
    }

    VirtualDir GetSubdirectory(std::string_view dir_name) const override {
        const auto entry = session->archive().lookup(handle, dir_name);
        if (!entry || entry.value->type != nx::EntryType::directory) {
            return nullptr;
        }
        return std::make_shared<NxDirectory>(session, entry.value->directory,
                                             std::string{entry.value->name}, Self());
    }

    std::vector<VirtualFile> GetFiles() const override {
        std::vector<VirtualFile> result;
        session->archive().enumerate(handle, [&](const nx::DirectoryEntryView& entry) {
            if (entry.type == nx::EntryType::file) {
                if (auto file = MakeFile(entry)) {
                    result.push_back(std::move(file));
                }
            }
            return true;
        });
        return result;
    }

    std::vector<VirtualDir> GetSubdirectories() const override {
        std::vector<VirtualDir> result;
        const auto self = Self();
        session->archive().enumerate(handle, [&](const nx::DirectoryEntryView& entry) {
            if (entry.type == nx::EntryType::directory) {
                result.push_back(std::make_shared<NxDirectory>(session, entry.directory,
                                                               std::string{entry.name}, self));
            }
            return true;
        });
        return result;
    }

private:
    VirtualDir Self() const {
        return std::const_pointer_cast<NxDirectory>(shared_from_this());
    }

    VirtualFile MakeFile(const nx::DirectoryEntryView& entry) const {
        const auto info = session->archive().file_info(entry.file);
        if (!info || info.value->size > (std::numeric_limits<std::size_t>::max)()) {
            return nullptr;
        }
        return std::make_shared<NxFile>(session, *info.value, std::string{entry.name}, Self());
    }

    std::shared_ptr<nx::Session> session;
    nx::DirectoryHandle handle;
    std::string name;
    VirtualDir parent;
};

std::vector<u8> ReadMedia(nx::Archive& archive, u32 index) {
    if (index == Absent) {
        return {};
    }
    auto bytes = archive.read_media(index);
    if (!bytes) {
        LOG_WARNING(Loader, "NX: cannot read media {}: {}", index, bytes.error.message);
        return {};
    }
    return std::move(*bytes.value);
}

} // namespace

struct NxArchive::Impl {
    VirtualFile file;
    std::shared_ptr<nx::RandomAccessSource> source;
    /// Catalog only (never mounted): enough for the game list.
    nx::Archive catalog;
    Selection selection;
    u32 base_snapshot{Absent};
    u32 application{};

    std::mutex mount_mutex;
    std::shared_ptr<nx::Session> session;

    void Select() {
        const auto& applications = catalog.applications();
        const auto& snapshots = catalog.snapshots();
        const auto& programs = catalog.programs();
        if (applications.empty()) {
            return;
        }
        if (applications.size() > 1) {
            LOG_WARNING(Loader, "NX archive holds {} applications; using the first",
                        applications.size());
        }
        application = 0;
        const auto& app = applications[application];

        // Types: 0 base, 1 update, 2 DLC, 3 standalone update, 4 unknown.
        u32 best = Absent;
        for (u32 i = app.first_snapshot; i < app.first_snapshot + app.snapshot_count; ++i) {
            const auto& snapshot = snapshots[i];
            const bool executable = snapshot.type == 0 || snapshot.type == 1 || snapshot.type == 3;
            if (!executable || snapshot.program_index >= programs.size() ||
                (snapshot.flags & 2) != 0) {
                continue;
            }
            if (snapshot.type == 0 && base_snapshot == Absent) {
                base_snapshot = i;
            }
            const auto newer = [&](const nx::Snapshot& other) {
                const bool update = snapshot.type != 0;
                const bool other_update = other.type != 0;
                return update != other_update ? update
                                              : snapshot.numeric_version > other.numeric_version;
            };
            if (best == Absent || newer(snapshots[best])) {
                best = i;
            }
        }
        if (best == Absent) {
            return;
        }

        const auto& snapshot = snapshots[best];
        static constexpr std::array<std::uint8_t, 16> no_content_id{};
        selection.native = snapshot.content_id != no_content_id;
        selection.bootable = !selection.native;
        selection.application_id = app.id;
        selection.program_id = programs[snapshot.program_index].id;
        selection.snapshot = best;
        selection.numeric_version = snapshot.numeric_version;
        selection.title = std::string{catalog.title(application)};
        selection.publisher = std::string{catalog.publisher(application)};
        selection.display_version = std::string{catalog.snapshot_display_version(best)};
    }
};

NxArchive::NxArchive(std::unique_ptr<Impl> impl_) : impl{std::move(impl_)} {}

NxArchive::~NxArchive() = default;

bool NxArchive::IsNxFile(const VirtualFile& file) {
    static constexpr std::array<u8, 4> magic{'N', 'X', 'A', 'R'};
    std::array<u8, 4> header{};
    return file != nullptr && file->GetSize() >= 256 &&
           file->Read(header.data(), header.size(), 0) == header.size() && header == magic;
}

std::shared_ptr<NxArchive> NxArchive::Open(VirtualFile file) {
    if (!IsNxFile(file)) {
        return nullptr;
    }
    auto impl = std::make_unique<Impl>();
    impl->file = std::move(file);
    impl->source = MakeSource(impl->file);

    nx::OpenOptions options;
    options.zstd = &DecodeBlock;
    auto archive = nx::Archive::open(*impl->source, options);
    if (!archive) {
        LOG_ERROR(Loader, "NX: cannot open {}: {}", impl->file->GetName(), archive.error.message);
        return nullptr;
    }
    impl->catalog = std::move(*archive.value);
    impl->Select();
    return std::shared_ptr<NxArchive>(new NxArchive(std::move(impl)));
}

const NxArchive::Selection& NxArchive::GetSelection() const {
    return impl->selection;
}

std::vector<u8> NxArchive::ReadNacp() {
    const auto& snapshots = impl->catalog.snapshots();
    // The selected version's own control data, else its base's, else the application's.
    for (const u32 index : {impl->selection.snapshot, impl->base_snapshot}) {
        if (index < snapshots.size() && snapshots[index].nacp_index != Absent) {
            if (auto bytes = ReadMedia(impl->catalog, snapshots[index].nacp_index); !bytes.empty()) {
                return bytes;
            }
        }
    }
    const auto& applications = impl->catalog.applications();
    if (impl->application < applications.size()) {
        return ReadMedia(impl->catalog, applications[impl->application].nacp_index);
    }
    return {};
}

std::vector<u8> NxArchive::ReadIcon() {
    const auto& applications = impl->catalog.applications();
    if (impl->application >= applications.size()) {
        return {};
    }
    return ReadMedia(impl->catalog, applications[impl->application].icon_index);
}

VirtualDir NxArchive::Mount() {
    if (!impl->selection.bootable) {
        return nullptr;
    }
    std::scoped_lock lock{impl->mount_mutex};
    if (impl->session == nullptr) {
        nx::SessionOptions options;
        options.reader.zstd = &DecodeBlock;
        auto session = nx::Session::open(impl->source, options);
        if (!session) {
            LOG_ERROR(Loader, "NX: cannot mount {}: {}", impl->file->GetName(),
                      session.error.message);
            return nullptr;
        }
        impl->session = std::move(*session.value);
    }
    const auto root = impl->session->archive().root_directory(impl->selection.snapshot);
    if (!root) {
        return nullptr;
    }
    return std::make_shared<NxDirectory>(impl->session, *root.value, "nx");
}

bool NxArchive::VerifyAll() {
    if (Mount() == nullptr) {
        return false;
    }
    const auto status = impl->session->verify_all();
    if (!status) {
        LOG_ERROR(Loader, "NX: verification failed: {}", status.error.message);
    }
    return status.ok();
}

} // namespace FileSys
