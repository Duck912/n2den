// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <memory>
#include <string_view>

#include "common/logging.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/host1x/host1x.h"
#include "video_core/renderer_base.h"
#ifdef HAS_D3D12_NVN
#include "video_core/renderer_d3d12/renderer_d3d12.h"
#endif
#include "video_core/renderer_null/renderer_null.h"
#ifdef HAS_OPENGL
#include "video_core/renderer_opengl/renderer_opengl.h"
#endif
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/video_core.h"

namespace {

#ifdef HAS_D3D12_NVN
/// NVN-HLE on D3D12 (experimental): only while NVN-HLE is on and the NVN-HLE renderer is Direct3D
/// 12 (EDEN_NVN_RENDERER=d3d12, or the nvn_renderer setting when the variable is unset:
/// Settings::NvnRendererChoice). Returns nullptr, with `context` untouched, in every other case
/// and when the D3D12 renderer fails.
std::unique_ptr<VideoCore::RendererBase> TryCreateD3D12Nvn(
    Core::Frontend::EmuWindow& emu_window, Tegra::GPU& gpu,
    std::unique_ptr<Core::Frontend::GraphicsContext>& context) {
    const bool hle = Settings::ResolveNvnHle().hle;
    if (!hle) {
        // The setting is read only under NVN-HLE; only the variable is worth a warning.
        const char* const choice = std::getenv("EDEN_NVN_RENDERER");
        if (choice != nullptr && std::string_view{choice} == "d3d12") {
            LOG_WARNING(HW_GPU,
                        "EDEN_NVN_RENDERER=d3d12 ignored: the game does not run on NVN-HLE");
        }
        return nullptr;
    }
    if (Settings::NvnRendererChoice() != Settings::NvnRenderer::D3D12) {
        return nullptr;
    }
    auto renderer = D3D12::RendererD3D12::TryCreate(emu_window, gpu, context);
    if (!renderer) {
        LOG_WARNING(HW_GPU, "[nvn] D3D12 renderer unavailable, using Vulkan");
    }
    return renderer;
}
#endif

std::unique_ptr<VideoCore::RendererBase> CreateRenderer(Core::System& system, Core::Frontend::EmuWindow& emu_window, Tegra::GPU& gpu, std::unique_ptr<Core::Frontend::GraphicsContext> context) {
    [[maybe_unused]] auto& device_memory = system.Host1x().MemoryManager();
    switch (Settings::values.renderer_backend.GetValue()) {
#ifdef HAS_OPENGL
    case Settings::RendererBackend::OpenGL_GLSL:
    case Settings::RendererBackend::OpenGL_GLASM:
    case Settings::RendererBackend::OpenGL_SPIRV:
        return std::make_unique<OpenGL::RendererOpenGL>(emu_window, device_memory, gpu, std::move(context));
#endif
    case Settings::RendererBackend::Vulkan:
#ifdef HAS_D3D12_NVN
        if (auto renderer = TryCreateD3D12Nvn(emu_window, gpu, context)) {
            return renderer;
        }
#endif
        return std::make_unique<Vulkan::RendererVulkan>(emu_window, device_memory, gpu, std::move(context));
    case Settings::RendererBackend::Null:
        return std::make_unique<Null::RendererNull>(emu_window, gpu, std::move(context));
    default:
        return nullptr;
    }
}

} // Anonymous namespace

namespace VideoCore {

/// @brief Creates an emulated GPU instance using the given system context.
void CreateGPU(std::optional<Tegra::GPU>& gpu, Core::Frontend::EmuWindow& emu_window, Core::System& system) {
    Settings::UpdateRescalingInfo();

    const auto nvdec_value = Settings::values.nvdec_emulation.GetValue();
    const bool use_nvdec = nvdec_value != Settings::NvdecEmulation::Off;
    const bool use_async = Settings::values.use_asynchronous_gpu_emulation.GetValue();
    gpu.emplace(system, use_async, use_nvdec);
    auto context = emu_window.CreateSharedContext();
    auto scope = context->Acquire();
    try {
        auto renderer = CreateRenderer(system, emu_window, *gpu, std::move(context));
        gpu->BindRenderer(std::move(renderer));
    } catch (const std::runtime_error& exception) {
        scope.Cancel();
        LOG_ERROR(HW_GPU, "Failed to initialize GPU: {}", exception.what());
        gpu.reset();
    }
}

} // namespace VideoCore
