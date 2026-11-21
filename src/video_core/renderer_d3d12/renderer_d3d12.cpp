// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <exception>
#include <utility>

#include <fmt/format.h>

#include "common/logging.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"
#include "video_core/capture.h"
#include "video_core/renderer_d3d12/d3d12_device.h"
#include "video_core/renderer_d3d12/d3d12_swapchain.h"
#include "video_core/renderer_d3d12/nvn/d3d12_nvn_context.h"
#include "video_core/renderer_d3d12/renderer_d3d12.h"

namespace D3D12 {

std::unique_ptr<VideoCore::RendererBase> RendererD3D12::TryCreate(
    Core::Frontend::EmuWindow& emu_window, Tegra::GPU& emulated_gpu,
    std::unique_ptr<Core::Frontend::GraphicsContext>& graphics_context) {
    const auto& window_info = emu_window.GetWindowInfo();
    if (window_info.type != Core::Frontend::WindowSystemType::Windows ||
        window_info.render_surface == nullptr) {
        LOG_ERROR(Render, "[nvn] d3d12: the render window has no HWND");
        return nullptr;
    }
    std::unique_ptr<RendererD3D12> renderer;
    try {
        auto new_device = std::make_unique<Device>();
        const Layout::FramebufferLayout& layout = emu_window.GetFramebufferLayout();
        const auto hwnd = static_cast<HWND>(window_info.render_surface);
        auto new_swap_chain =
            std::make_unique<SwapChain>(*new_device, hwnd, layout.width, layout.height);
        renderer.reset(new RendererD3D12(emu_window, emulated_gpu, std::move(new_device),
                                         std::move(new_swap_chain)));
        renderer->nvn_context = std::make_unique<NvnContext>(
            *renderer->device, *renderer->swap_chain, emu_window, renderer->renderer_settings);
    } catch (const std::exception& exception) {
        // The graphics context was not touched: the caller builds the fallback with it.
        LOG_ERROR(Render, "[nvn] d3d12: renderer creation failed: {}", exception.what());
        return nullptr;
    }
    LOG_INFO(Render, "[nvn] renderer: D3D12 skeleton (NVN-HLE renderer) on \"{}\"",
             renderer->device->AdapterName());
    // Last, once nothing can throw: the renderer takes the graphics context (RendererBase was
    // built without one; nothing reads it before the GPU thread starts).
    renderer->context = std::move(graphics_context);
    return renderer;
}

RendererD3D12::RendererD3D12(Core::Frontend::EmuWindow& emu_window, Tegra::GPU& gpu_,
                             std::unique_ptr<Device> device_,
                             std::unique_ptr<SwapChain> swap_chain_)
    : RendererBase(emu_window, nullptr), gpu{gpu_}, rasterizer{gpu_},
      device{std::move(device_)}, swap_chain{std::move(swap_chain_)} {}

RendererD3D12::~RendererD3D12() {
    nvn_context.reset();
    const u64 count = composites.load(std::memory_order_relaxed);
    if (count != 0) {
        LOG_INFO(Render, "[nvn] d3d12 skeleton: {} emulated composites not drawn", count);
    }
}

void RendererD3D12::Composite(std::span<const Tegra::FramebufferConfig> layers) {
    if (layers.empty()) {
        return;
    }
    composites.fetch_add(1, std::memory_order_relaxed);
    gpu.RendererFrameEndNotify();
    render_window.OnFrameDisplayed();
}

std::vector<u8> RendererD3D12::GetAppletCaptureBuffer() {
    return std::vector<u8>(VideoCore::Capture::TiledSize);
}

std::string RendererD3D12::GetDeviceVendor() const {
    return fmt::format("{} (D3D12)", device->AdapterName());
}

VideoCommon::Nvn::Sink* RendererD3D12::NvnSink() {
    return nvn_context.get();
}

} // namespace D3D12
