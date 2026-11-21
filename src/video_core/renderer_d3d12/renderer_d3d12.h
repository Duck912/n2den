// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// NVN-HLE on D3D12, skeleton. Chosen by
// CreateRenderer only when the NVN-HLE renderer is D3D12 (EDEN_NVN_RENDERER=d3d12 or the
// nvn_renderer setting) while NVN-HLE resolves on; it owns the D3D12
// device, the DXGI flip swap chain on the render window and the NVN sink. The emulated GPU draws
// nothing under NVN-HLE, so the rasterizer is the null one.

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "video_core/renderer_base.h"
#include "video_core/renderer_null/null_rasterizer.h"

namespace D3D12 {

class Device;
class SwapChain;
class NvnContext;

class RendererD3D12 final : public VideoCore::RendererBase {
public:
    /// Creates the renderer, or returns nullptr (logged) when the D3D12 device, the swap chain or
    /// the sink cannot be created. On failure `graphics_context` still holds the context, so the
    /// caller can build another renderer with it.
    [[nodiscard]] static std::unique_ptr<VideoCore::RendererBase> TryCreate(
        Core::Frontend::EmuWindow& emu_window, Tegra::GPU& emulated_gpu,
        std::unique_ptr<Core::Frontend::GraphicsContext>& graphics_context);

    ~RendererD3D12() override;

    /// Emulated framebuffers (boot logos, applet layers): not drawn yet, counted.
    void Composite(std::span<const Tegra::FramebufferConfig> layers) override;

    /// A black capture of the right size.
    std::vector<u8> GetAppletCaptureBuffer() override;

    VideoCore::RasterizerInterface* ReadRasterizer() override {
        return &rasterizer;
    }

    [[nodiscard]] std::string GetDeviceVendor() const override;

    [[nodiscard]] VideoCommon::Nvn::Sink* NvnSink() override;

    [[nodiscard]] bool NvnOnD3D12() const override {
        return true;
    }

private:
    /// Built without a graphics context; TryCreate hands it over as its last step.
    RendererD3D12(Core::Frontend::EmuWindow& emu_window, Tegra::GPU& gpu_,
                  std::unique_ptr<Device> device_, std::unique_ptr<SwapChain> swap_chain_);

    Tegra::GPU& gpu;
    Null::RasterizerNull rasterizer;
    std::atomic<u64> composites{};
    // Destroyed in reverse order: the sink first (its threads use the swap chain and the queue),
    // then the swap chain, then the device.
    std::unique_ptr<Device> device;
    std::unique_ptr<SwapChain> swap_chain;
    std::unique_ptr<NvnContext> nvn_context;
};

} // namespace D3D12
