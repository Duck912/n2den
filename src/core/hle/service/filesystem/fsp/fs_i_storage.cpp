// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>

#include "common/frame_stats.h"
#include "core/debugger/nx2_debug_tools.h"
#include "core/file_sys/errors.h"
#include "core/hle/service/cmif_serialization.h"
#include "core/hle/service/filesystem/fsp/fs_i_storage.h"

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


IStorage::IStorage(Core::System& system_, FileSys::VirtualFile backend_)
    : ServiceFramework{system_, "IStorage"}, backend(std::move(backend_)) {
    static const FunctionInfo functions[] = {
        {0, D<&IStorage::Read>, "Read"},
        {1, nullptr, "Write"},
        {2, nullptr, "Flush"},
        {3, nullptr, "SetSize"},
        {4, D<&IStorage::GetSize>, "GetSize"},
        {5, nullptr, "OperateRange"},
    };
    RegisterHandlers(functions);
}

Result IStorage::Read(
    OutBuffer<BufferAttr_HipcMapAlias | BufferAttr_HipcMapTransferAllowsNonSecure> out_bytes,
    s64 offset, s64 length) {
    LOG_DEBUG(Service_FS, "called, offset={:#x}, length={}", offset, length);

    R_UNLESS(length >= 0, FileSys::ResultInvalidSize);
    R_UNLESS(offset >= 0, FileSys::ResultInvalidOffset);

    // Read the data from the Storage backend
    const ReadTimer timer{length};
    backend->Read(out_bytes.data(), length, offset);

    R_SUCCEED();
}

Result IStorage::GetSize(Out<u64> out_size) {
    *out_size = backend->GetSize();

    LOG_DEBUG(Service_FS, "called, size={}", *out_size);

    R_SUCCEED();
}

} // namespace Service::FileSystem
