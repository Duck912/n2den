// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// NVN-HLE D3D12 skeleton sink. Nothing the game records is
// executed: submissions complete in seq order as soon as the queue passes them, presents clear the
// back buffer to a colour derived from the present count, every other call is a counted no-op.
// The completion semantics are the Vulkan sink's (renderer_vulkan/nvn/DESIGN.md, "Ownership and
// threads"), so the recorder's event words, retires and guest waits behave as in its device-lost
// mode and the game keeps running.

#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "video_core/nvn/sink.h"

namespace Core::Frontend {
class EmuWindow;
}

namespace VideoCore {
struct RendererSettings;
}

namespace D3D12 {

class Device;
class SwapChain;

class NvnContext final : public VideoCommon::Nvn::Sink {
public:
    /// Starts the replay and completion threads. Throws std::runtime_error when the D3D12 objects
    /// it needs (fence, command list) cannot be created.
    NvnContext(Device& device, SwapChain& swap_chain, Core::Frontend::EmuWindow& render_window,
               VideoCore::RendererSettings& renderer_settings);
    ~NvnContext() override;

    NvnContext(const NvnContext&) = delete;
    NvnContext& operator=(const NvnContext&) = delete;

    void CreatePool(const VideoCommon::Nvn::PoolDesc& desc) override;
    void CreateTexture(const VideoCommon::Nvn::TextureDesc& desc) override;
    void CreateSampler(const VideoCommon::Nvn::SamplerDesc& desc) override;
    void CreateProgram(const VideoCommon::Nvn::ProgramDesc& desc) override;
    void Destroy(u64 serial) override;
    void Submit(VideoCommon::Nvn::Submission&& submission) override;
    void Present(const VideoCommon::Nvn::PresentRequest& request) override;
    bool Wait(u64 seq, std::chrono::nanoseconds timeout) override;
    void SetCompletionCallback(std::function<void(u64 seq)> callback) override;
    void SetFailureCallback(std::function<void(std::string reason)> callback) override;
    void SetGuestMemory(VideoCommon::Nvn::GuestMemory* memory) override;
    void SetDeviceModes(u32 window_origin, u32 depth_mode) override;
    void SetTitleId(u64 title_id) override;
    void CollectStats(std::vector<VideoCommon::Nvn::SinkStat>& out) const override;
    void MapVirtual(u64 pool_serial,
                    std::vector<VideoCommon::Nvn::VirtualMapping>&& mappings) override;
    void NotifyCpuWrite(u64 guest_address, u64 size) override;
    void NotifyHostWrite(u64 guest_address, u64 size) override;
    void NotifyEventCpu(u64 address, VideoCommon::Nvn::EventCpuChange change,
                        u32 value) override;
    void NotifyTextureWritten(u64 texture_serial) override;
    bool FlushTextureToGuest(u64 texture_serial) override;
    [[nodiscard]] std::string DescribeState() const override;
    [[nodiscard]] std::string DescribeMemory() const override;
    void NoteTextureHolders(VideoCommon::Nvn::TextureHolderReport&& report) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace D3D12
