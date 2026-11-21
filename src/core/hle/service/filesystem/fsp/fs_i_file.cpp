// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>

#include "common/frame_stats.h"
#include "core/debugger/nx2_debug_tools.h"
#include "core/file_sys/errors.h"
#include "core/hle/service/cmif_serialization.h"
#include "core/hle/service/filesystem/fsp/fs_i_file.h"

namespace Service::FileSystem {

namespace {
/// Frame statistics (EDEN_NX2_FRAME_STATS) and sched_trace span of one read: guest asset loading.
class ReadTimer {
public:
    explicit ReadTimer(s64 length_) : length{length_} {
        active = Common::FrameStats::Enabled() ||
                 Core::Nx2Debug::SchedTrace::enabled.load(std::memory_order_relaxed);
        if (active) {
            start = std::chrono::steady_clock::now();
        }
    }
    ~ReadTimer() {
        if (!active) {
            return;
        }
        const auto end = std::chrono::steady_clock::now();
        Common::FrameStats::AddStall(
            Common::FrameStats::Cause::FsRead,
            static_cast<u64>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()));
        Common::FrameStats::AddCount(Common::FrameStats::Counter::FsReadKiB,
                                     static_cast<u64>(length > 0 ? length : 0) >> 10);
        if (Core::Nx2Debug::SchedTrace::enabled.load(std::memory_order_relaxed)) {
            Core::Nx2Debug::SchedTrace::RecordHostSpan(
                Core::Nx2Debug::SchedTrace::HostSpanKind::FsRead, start, end,
                static_cast<u64>(length > 0 ? length : 0));
        }
    }

private:
    s64 length;
    bool active;
    std::chrono::steady_clock::time_point start{};
};
} // Anonymous namespace


IFile::IFile(Core::System& system_, FileSys::VirtualFile file_)
    : ServiceFramework{system_, "IFile"}, backend{std::make_unique<FileSys::Fsa::IFile>(file_)} {
    // clang-format off
    static const FunctionInfo functions[] = {
        {0, D<&IFile::Read>, "Read"},
        {1, D<&IFile::Write>, "Write"},
        {2, D<&IFile::Flush>, "Flush"},
        {3, D<&IFile::SetSize>, "SetSize"},
        {4, D<&IFile::GetSize>, "GetSize"},
        {5, nullptr, "OperateRange"},
        {6, nullptr, "OperateRangeWithBuffer"},
    };
    // clang-format on
    RegisterHandlers(functions);
}

Result IFile::Read(
    FileSys::ReadOption option, Out<s64> out_size, s64 offset,
    const OutBuffer<BufferAttr_HipcMapAlias | BufferAttr_HipcMapTransferAllowsNonSecure> out_buffer,
    s64 size) {
    LOG_DEBUG(Service_FS, "called, option={}, offset={:#x}, length={}", option.value, offset,
              size);

    // Read the data from the Storage backend
    const ReadTimer timer{size};
    R_RETURN(
        backend->Read(reinterpret_cast<size_t*>(out_size.Get()), offset, out_buffer.data(), size));
}

Result IFile::Write(
    const InBuffer<BufferAttr_HipcMapAlias | BufferAttr_HipcMapTransferAllowsNonSecure> buffer,
    FileSys::WriteOption option, s64 offset, s64 size) {
    LOG_DEBUG(Service_FS, "called, option={}, offset={:#x}, length={}", option.value, offset,
              size);

    R_RETURN(backend->Write(offset, buffer.data(), size, option));
}

Result IFile::Flush() {
    LOG_DEBUG(Service_FS, "called");

    R_RETURN(backend->Flush());
}

Result IFile::SetSize(s64 size) {
    LOG_DEBUG(Service_FS, "called, size={}", size);

    R_RETURN(backend->SetSize(size));
}

Result IFile::GetSize(Out<s64> out_size) {
    LOG_DEBUG(Service_FS, "called");

    R_RETURN(backend->GetSize(out_size));
}

} // namespace Service::FileSystem
