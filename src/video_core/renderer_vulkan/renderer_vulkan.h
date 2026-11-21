// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once
#include <vector>
#include <chrono>
#include <array>

#include <memory>
#include <string>
#include <variant>

#include "common/dynamic_library.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_base.h"
#ifdef HAS_LSFG
#include "video_core/renderer_vulkan/present/frame_gen.h"
#endif
#include "video_core/renderer_vulkan/vk_blit_screen.h"
#include "video_core/renderer_vulkan/vk_present_manager.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "video_core/renderer_vulkan/vk_turbo_mode.h"
#include "video_core/load_progress.h"
#include "video_core/renderer_vulkan/nvn/nvn_context.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Core::Memory {
class Memory;
}

namespace Tegra {
class GPU;
}

namespace Vulkan {

Device CreateDevice(const vk::Instance& instance, const vk::InstanceDispatch& dld,
                    VkSurfaceKHR surface);

class RendererVulkan final : public VideoCore::RendererBase {
public:
    explicit RendererVulkan(Core::Frontend::EmuWindow& emu_window,
                            Tegra::MaxwellDeviceMemoryManager& device_memory_, Tegra::GPU& gpu_,
                            std::unique_ptr<Core::Frontend::GraphicsContext> context_);
    ~RendererVulkan() override;

    void Composite(std::span<const Tegra::FramebufferConfig> framebuffers) override;

    std::vector<u8> GetAppletCaptureBuffer() override;

    VideoCore::RasterizerInterface* ReadRasterizer() override {
        return &rasterizer;
    }

    [[nodiscard]] std::string GetDeviceVendor() const override {
        return device.GetDriverName();
    }

    [[nodiscard]] VideoCommon::Nvn::Sink* NvnSink() override {
        return nvn_context ? &*nvn_context : nullptr;
    }

    [[nodiscard]] VideoCore::LoadProgressSink* NvnLoadProgress() override {
        return &nvn_load_progress;
    }

    // Enhanced platform-specific initialization
    void InitializePlatformSpecific();

private:
    void InterpolateFrames(Frame* prev_frame, Frame* curr_frame);
    Frame* previous_frame = nullptr;  // Store the previous frame for interpolation
    VkCommandBuffer BeginSingleTimeCommands();
    void EndSingleTimeCommands(VkCommandBuffer command_buffer);
    void Report() const;

    vk::Buffer RenderToBuffer(std::span<const Tegra::FramebufferConfig> framebuffers,
                              const Layout::FramebufferLayout& layout, VkFormat format,
                              VkDeviceSize buffer_size);
    void RenderScreenshot(std::span<const Tegra::FramebufferConfig> framebuffers);
    /// EDEN_N2DEN_LUMA_TRACE=1: mean brightness of every presented frame, read back a few frames
    /// later without waiting, and one-frame outliers logged (flashing that is too short to catch in
    /// screenshots).
    void TraceLuma(std::span<const Tegra::FramebufferConfig> framebuffers);
    void RenderAppletCaptureLayer(std::span<const Tegra::FramebufferConfig> framebuffers);

    std::vector<Tegra::FramebufferConfig> applet_capture_layers;
    std::vector<Tegra::FramebufferConfig> screenshot_layer_scratch;

    Tegra::MaxwellDeviceMemoryManager& device_memory;
    Tegra::GPU& gpu;

    std::shared_ptr<Common::DynamicLibrary> library;
    vk::InstanceDispatch dld;
#ifdef EGLANTINE_STREAMLINE
    bool streamline_started; ///< NVIDIA Streamline starts before the instance.
#endif

    // Keep original handles for compatibility with existing code
    vk::Instance instance;

    vk::DebugUtilsMessenger debug_messenger;

    vk::SurfaceKHR surface;

    Device device;
    MemoryAllocator memory_allocator;
    StateTracker state_tracker;
    Scheduler scheduler;
    Swapchain swapchain;
    PresentManager present_manager;
    BlitScreen blit_swapchain;
    BlitScreen blit_capture;
    BlitScreen blit_applet;
    BlitScreen blit_luma;
    RasterizerVulkan rasterizer;
#ifdef HAS_LSFG
    FrameGen frame_gen;
#endif
    std::optional<TurboMode> turbo_mode;
    /// nvn-hle: where its pipeline cache's prepare step and disk preload report (the Qt frontend
    /// forwards it to the loading screen), and the Vulkan sink itself while the game runs on nvn-hle.
    VideoCore::LoadProgressSink nvn_load_progress;
    std::optional<Nvn::Context> nvn_context;

    Frame applet_frame;

    struct LumaSlot {
        Frame frame{};
        vk::Buffer buffer;
        u64 tick{};
        u64 index{};
        u64 skips{}; ///< Common::N2denTrace::pipeline_skips when the frame was captured
        bool pending{};
    };
    struct LumaValue {
        u64 frame{};
        float mean{};
        u64 skips{};
    };
    std::array<LumaSlot, 6> luma_slots;
    u64 luma_frames{};
    std::vector<LumaValue> luma_values; ///< finished readbacks
    struct LumaDumpFrame {
        u64 frame{};
        float mean{};
        std::vector<u8> bgra;
    };
    std::vector<LumaDumpFrame> luma_dump; ///< EDEN_N2DEN_LUMA_DUMP: the last three frames
    u32 luma_dumps{};
    std::chrono::steady_clock::time_point luma_report{};
};

} // namespace Vulkan
