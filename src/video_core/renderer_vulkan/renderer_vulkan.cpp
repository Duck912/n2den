// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <array>
#include <cstring>
#include <utility>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <fmt/ranges.h>

#include "common/logging.h"
#include "common/n2den_trace.h"
#include <ranges>
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core_timing.h"
#include "core/frontend/graphics_context.h"
#include "video_core/back_thread_stats.h"
#include "video_core/capture.h"
#include "video_core/gpu.h"
#include "video_core/present.h"
#include "video_core/renderer_vulkan/present/util.h"
#ifdef HAS_RESHADE
#include "video_core/post_processing/fx_chain.h"
#endif
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/nvn/nvn_settings_notice.h"
#include "video_core/shader_log.h"
#include "video_core/renderer_vulkan/vk_blit_screen.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "video_core/textures/decoders.h"
#include "video_core/vulkan_common/vulkan_debug_callback.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_instance.h"
#include "video_core/vulkan_common/vulkan_library.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_surface.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#ifdef EGLANTINE_STREAMLINE
#include "video_core/vulkan_common/streamline.h"
#endif
#ifdef __ANDROID__
#include <jni.h>
#endif
namespace Vulkan {
namespace {

constexpr VkExtent2D CaptureImageSize{
    .width = VideoCore::Capture::LinearWidth,
    .height = VideoCore::Capture::LinearHeight,
};

#ifdef HAS_LSFG
[[nodiscard]] VkExtent2D GuestExtent(std::span<const Tegra::FramebufferConfig> framebuffers) {
    if (framebuffers.empty()) {
        return VkExtent2D{};
    }

    const auto& framebuffer = framebuffers.front();
    if (framebuffer.crop_rect.IsEmpty()) {
        return VkExtent2D{.width = framebuffer.width, .height = framebuffer.height};
    }
    return VkExtent2D{
        .width = static_cast<u32>(framebuffer.crop_rect.GetWidth()),
        .height = static_cast<u32>(framebuffer.crop_rect.GetHeight()),
    };
}
#endif

constexpr VkExtent3D CaptureImageExtent{
    .width = VideoCore::Capture::LinearWidth,
    .height = VideoCore::Capture::LinearHeight,
    .depth = VideoCore::Capture::LinearDepth,
};

constexpr VkFormat CaptureFormat = VK_FORMAT_A8B8G8R8_UNORM_PACK32;

std::string GetReadableVersion(u32 version) {
    return fmt::format("{}.{}.{}", VK_VERSION_MAJOR(version), VK_VERSION_MINOR(version),
                       VK_VERSION_PATCH(version));
}

std::string GetDriverVersion(const Device& device) {
    // Extracted from
    // https://github.com/SaschaWillems/vulkan.gpuinfo.org/blob/5dddea46ea1120b0df14eef8f15ff8e318e35462/functions.php#L308-L314
    const u32 version = device.GetDriverVersion();

    if (device.GetDriverID() == VK_DRIVER_ID_NVIDIA_PROPRIETARY) {
        const u32 major = (version >> 22) & 0x3ff;
        const u32 minor = (version >> 14) & 0x0ff;
        const u32 secondary = (version >> 6) & 0x0ff;
        const u32 tertiary = version & 0x003f;
        return fmt::format("{}.{}.{}.{}", major, minor, secondary, tertiary);
    }
    if (device.GetDriverID() == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS) {
        const u32 major = version >> 14;
        const u32 minor = version & 0x3fff;
        return fmt::format("{}.{}", major, minor);
    }
    return GetReadableVersion(version);
}

std::string BuildCommaSeparatedExtensions(
    const std::set<std::string, std::less<>>& available_extensions) {
    return fmt::format("{}", fmt::join(available_extensions, ","));
}

} // Anonymous namespace

Device CreateDevice(const vk::Instance& instance, const vk::InstanceDispatch& dld, VkSurfaceKHR surface) {
    const std::vector<VkPhysicalDevice> devices = instance.EnumeratePhysicalDevices();
    const u32 device_index = Settings::values.vulkan_device.GetValue();
    if (device_index >= u32(devices.size())) {
        LOG_ERROR(Render_Vulkan, "Invalid device index {}!", device_index);
        throw vk::Exception(VK_ERROR_INITIALIZATION_FAILED);
    }
    const vk::PhysicalDevice physical_device(devices[device_index], dld);
    return Device(*instance, physical_device, surface, dld);
}

RendererVulkan::RendererVulkan(Core::Frontend::EmuWindow& emu_window,
                               Tegra::MaxwellDeviceMemoryManager& device_memory_,
                               Tegra::GPU& gpu_,
                               std::unique_ptr<Core::Frontend::GraphicsContext> context_)
try
    : RendererBase(emu_window, std::move(context_))
    , device_memory(device_memory_)
    , gpu(gpu_)
    , library(OpenLibrary(context.get()))
    , dld()
#ifdef EGLANTINE_STREAMLINE
    , streamline_started(Streamline::Start())
#endif
    // Create raw Vulkan instance first
    , instance(CreateInstance(*library,
                            dld,
                            VK_API_VERSION_1_1,
                            render_window.GetWindowInfo().type,
                            Settings::values.renderer_debug.GetValue()))
    // Create debug messenger if debug is enabled
    , debug_messenger(Settings::values.renderer_debug ? CreateDebugUtilsCallback(instance)
                                                    : vk::DebugUtilsMessenger{})
    // Create surface
    , surface(CreateSurface(instance, render_window.GetWindowInfo()))
    , device(CreateDevice(instance, dld, *surface))
    , memory_allocator(device)
    , state_tracker()
    , scheduler(device, state_tracker)
    , swapchain(*surface,
                device,
                scheduler,
               render_window.GetFramebufferLayout().width,
               render_window.GetFramebufferLayout().height)
    , present_manager(instance,
                      render_window,
                      device,
                      memory_allocator,
                      scheduler,
                      swapchain,
                      surface)
    , blit_swapchain(device_memory,
                   device,
                   memory_allocator,
                   present_manager,
                   scheduler,
                   PresentFiltersForDisplay)
    , blit_capture(device_memory,
                   device,
                   memory_allocator,
                   present_manager,
                   scheduler,
                   PresentFiltersForDisplay)
    , blit_applet(device_memory,
                  device,
                  memory_allocator,
                  present_manager,
                  scheduler,
                  PresentFiltersForAppletCapture)
    , blit_luma(device_memory,
                device,
                memory_allocator,
                present_manager,
                scheduler,
                PresentFiltersForDisplay)
    , rasterizer(render_window, gpu, device_memory, device, memory_allocator, state_tracker, scheduler)
#ifdef HAS_LSFG
    , frame_gen(memory_allocator, scheduler)
#endif
{

    if (Settings::values.renderer_force_max_clock.GetValue() && device.ShouldBoostClocks()) {
        turbo_mode.emplace(instance, dld);
        scheduler.RegisterOnSubmit([this] { turbo_mode->QueueSubmitted(); });
    }

#ifdef HAS_RESHADE
    VideoCore::FxChain::Instance().LoadFromSettings();
#endif

    // nvn-hle: the Vulkan sink, only when the game runs on nvn-hle (graphics_path, EDEN_NVN_HLE).
    // With the emulated GPU path nothing here runs.
    if (const Settings::NvnHleChoice choice = Settings::ResolveNvnHle(); choice.hle) {
        // The replay needs buffer device addresses and 64-bit shader integers (raw global
        // memory). A device without them runs the game on the emulated GPU instead.
        const bool device_ok =
            device.IsBufferDeviceAddressSupported() && device.IsShaderInt64Supported();
        if (!device_ok) {
            LOG_WARNING(Render_Vulkan,
                        "[nvn] graphics path: emulated GPU (the Vulkan device lacks buffer device "
                        "addresses or 64-bit shader integers, which nvn-hle needs)");
        } else {
            try {
                // Force maximum clocks: the replay submits to the queue itself, so it feeds
                // TurboMode (the scheduler's submit hook never fires for its work).
                nvn_context.emplace(
                    device, scheduler, present_manager, render_window, gpu.ShaderNotify(),
                    nvn_load_progress,
                    turbo_mode ? std::function<void()>{[this] { turbo_mode->QueueSubmitted(); }}
                               : std::function<void()>{});
                nvn_context->SetScreenshotRequests(renderer_settings);
                LOG_INFO(Render_Vulkan, "[nvn] Vulkan replay context created");
                VideoCommon::ShaderLog::Start();
                // The replay decodes ASTC itself into RGBA8 stand-ins and re-encodes them to BC
                // per storage, but MaxwellToVK::SurfaceFormat (shared with the emulated path)
                // picks its host format from the effective recompression: a BC mode would give
                // every decoded stand-in a BC image. The saved setting stays untouched.
                Settings::SetAstcRecompressionOverride(Settings::AstcRecompression::Uncompressed);
                if (const auto gaps = Nvn::SettingsNotApplied(Nvn::CurrentPresentSettings());
                    !gaps.empty()) {
                    LOG_WARNING(Render_Vulkan,
                                "[nvn] settings nvn-hle does not apply: {} (set Graphics path to "
                                "Emulated GPU to use them)",
                                fmt::join(gaps, "; "));
                }
            } catch (const std::exception& exception) {
                nvn_context.reset();
                LOG_ERROR(Render_Vulkan,
                          "[nvn] graphics path: emulated GPU (the Vulkan replay context failed: {})",
                          exception.what());
            }
        }
    }

    Report();
} catch (const vk::Exception& exception) {
    LOG_ERROR(Render_Vulkan, "Vulkan initialization failed with error: {}", exception.what());
    throw std::runtime_error{fmt::format("Vulkan initialization error {}", exception.what())};
}

RendererVulkan::~RendererVulkan() {
    // nvn-hle's replay threads submit to the same queue: stop them before anything they use goes.
    nvn_context.reset();
    VideoCommon::ShaderLog::Stop();
    scheduler.RegisterOnSubmit([] {});
    void(device.GetLogical().WaitIdle());
}

void RendererVulkan::Composite(std::span<const Tegra::FramebufferConfig> framebuffers) {
    const VideoCommon::BackStats::ScopedSharedTime composite_time{VideoCommon::BackStats::shared.composite};
    SCOPE_EXIT {
        render_window.OnFrameDisplayed();
    };

    // Swapchain::Create leaves the format undefined while the surface has a zero extent (for
    // example a minimized window). Creating the present frame image with VK_FORMAT_UNDEFINED
    // makes the NVIDIA driver divide by zero, and there is no swapchain to present to anyway.
    // This also covers the applet capture frame, which is allocated through the same present
    // manager.
    if (swapchain.GetImageFormat() == VK_FORMAT_UNDEFINED) {
        static bool logged = false;
        if (!std::exchange(logged, true)) {
            LOG_WARNING(Render_Vulkan, "Skipping composite while the swapchain has no format");
        }
        return;
    }

    RenderAppletCaptureLayer(framebuffers);

    if (!render_window.IsShown()) {
        return;
    }

    RenderScreenshot(framebuffers);
    TraceLuma(framebuffers);
    Frame* frame = present_manager.GetRenderFrame();

    scheduler.RequestOutsideRenderPassOperationContext();
    blit_swapchain.DrawToFrame(device, rasterizer, frame, framebuffers,
                               render_window.GetFramebufferLayout(),
                               present_manager.SwapchainImageCount(),
                               swapchain.GetImageViewFormat());

#ifdef HAS_LSFG
    void(frame_gen.WantedGenerations(present_manager.MaxExtraFrames()));

    frame_gen.Process(device, frame, present_manager.SwapchainImageFormat(),
                      GuestExtent(framebuffers));

    const size_t generated_frames = frame_gen.GeneratedFrameCount();
    for (size_t generation = 0; generation < generated_frames; ++generation) {
        Frame* generated = present_manager.GetRenderFrame();
        blit_swapchain.PrepareFrame(device, generated, render_window.GetFramebufferLayout());
        frame_gen.GenerateInto(device, generated, generation);
        scheduler.Flush(*generated->render_ready);
        present_manager.Present(generated);
    }
#endif

    scheduler.Flush(*frame->render_ready);

    present_manager.Present(frame);
#ifdef HAS_LSFG
    scheduler.DispatchWork();
#endif

    gpu.RendererFrameEndNotify();
    rasterizer.TickFrame();
}

void RendererVulkan::Report() const {
    using namespace Common::Literals;
    const std::string vendor_name{device.GetVendorName()};
    const std::string model_name{device.GetModelName()};
    const std::string driver_version = GetDriverVersion(device);
    const std::string driver_name = fmt::format("{} {}", vendor_name, driver_version);

    const std::string api_version = GetReadableVersion(device.ApiVersion());

    const std::string extensions = BuildCommaSeparatedExtensions(device.GetAvailableExtensions());

    const auto available_vram = static_cast<f64>(device.GetDeviceLocalMemory()) / f64{1_GiB};

    LOG_INFO(Render_Vulkan, "Driver: {}", driver_name);
    LOG_INFO(Render_Vulkan, "Device: {}", model_name);
    LOG_INFO(Render_Vulkan, "Vulkan: {}", api_version);
    LOG_INFO(Render_Vulkan, "Available VRAM: {:.2f} GiB", available_vram);
}

vk::Buffer RendererVulkan::RenderToBuffer(std::span<const Tegra::FramebufferConfig> framebuffers,
                                          const Layout::FramebufferLayout& layout, VkFormat format,
                                          VkDeviceSize buffer_size) {
    auto frame = [&]() {
        Frame f{};
        f.image =
            CreateWrappedImage(memory_allocator, VkExtent2D{layout.width, layout.height}, format);
        f.image_view = CreateWrappedImageView(device, f.image, format);
        f.framebuffer = blit_capture.CreateFramebuffer(device, layout, *f.image_view, format);
        return f;
    }();

    auto dst_buffer = CreateWrappedBuffer(memory_allocator, buffer_size, MemoryUsage::Download);
    blit_capture.DrawToFrame(device, rasterizer, &frame, framebuffers, layout, 1, format);

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([&](vk::CommandBuffer cmdbuf) {
        DownloadColorImage(cmdbuf, *frame.image, *dst_buffer,
                           VkExtent3D{layout.width, layout.height, 1});
    });

    // Ensure the copy is fully completed before saving the capture
    scheduler.Finish();

    // Copy backing image data to the capture buffer
    dst_buffer.Invalidate();
    return dst_buffer;
}

void RendererVulkan::RenderScreenshot(std::span<const Tegra::FramebufferConfig> framebuffers) {
    // Without nvn-hle nothing else claims: this is true whenever a request is pending.
    if (!renderer_settings.ClaimScreenshot()) {
        return;
    }

    const auto screenshot_layers = Tegra::FilterLayerStack(
        framebuffers, renderer_settings.screenshot_layer_stack, screenshot_layer_scratch);

    const auto& layout{renderer_settings.screenshot_framebuffer_layout};
    const auto dst_buffer = RenderToBuffer(screenshot_layers, layout, VK_FORMAT_B8G8R8A8_UNORM,
                                           layout.width * layout.height * 4);

    std::memcpy(renderer_settings.screenshot_bits, dst_buffer.Mapped().data(),
                dst_buffer.Mapped().size());
    renderer_settings.screenshot_complete_callback(false);
    renderer_settings.FinishScreenshot();
}

void RendererVulkan::TraceLuma(std::span<const Tegra::FramebufferConfig> framebuffers) {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_N2DEN_LUMA_TRACE");
        return value && value[0] != '\0' && value[0] != '0';
    }();
    if (!enabled || framebuffers.empty()) {
        return;
    }
    // [DEBUG] EDEN_N2DEN_LUMA_DUMP=<dir>: frames are read back at 320x180 and the first 12
    // one-frame outliers are saved there as PPM with the frames before and after.
    static const std::string dump_dir = [] {
        const char* const value = std::getenv("EDEN_N2DEN_LUMA_DUMP");
        return std::string{value ? value : ""};
    }();
    const u32 width = dump_dir.empty() ? 64 : 320;
    const u32 height = dump_dir.empty() ? 36 : 180;
    // Finished readbacks: mean of the BGRA pixels' Rec. 709 luma.
    for (LumaSlot& slot : luma_slots) {
        if (!slot.pending || !scheduler.IsFree(slot.tick)) {
            continue;
        }
        slot.buffer.Invalidate();
        const auto bytes = slot.buffer.Mapped();
        double sum = 0.0;
        for (size_t i = 0; i + 3 < bytes.size(); i += 4) {
            sum += 0.0722 * bytes[i] + 0.7152 * bytes[i + 1] + 0.2126 * bytes[i + 2];
        }
        const float mean = static_cast<float>(sum / (width * height));
        luma_values.push_back(LumaValue{slot.index, mean, slot.skips});
        slot.pending = false;
        if (!dump_dir.empty()) {
            // The last three frames read back: an outlier in the middle is saved with both.
            luma_dump.push_back(LumaDumpFrame{slot.index, mean, {bytes.begin(), bytes.end()}});
            if (luma_dump.size() > 3) {
                luma_dump.erase(luma_dump.begin());
            }
            if (luma_dump.size() == 3 && luma_dumps < 12 &&
                luma_dump[0].frame + 1 == luma_dump[1].frame &&
                luma_dump[1].frame + 1 == luma_dump[2].frame) {
                const float before = luma_dump[0].mean;
                const float after = luma_dump[2].mean;
                if (std::abs(luma_dump[1].mean - (before + after) * 0.5f) > 4.0f &&
                    std::abs(before - after) < 2.0f) {
                    ++luma_dumps;
                    for (const LumaDumpFrame& frame : luma_dump) {
                        const std::string path =
                            fmt::format("{}/luma-{:06}-of-{:06}.ppm", dump_dir, frame.frame,
                                        luma_dump[1].frame);
                        if (std::FILE* file = std::fopen(path.c_str(), "wb")) {
                            fmt::print(file, "P6\n{} {}\n255\n", width, height);
                            for (size_t i = 0; i + 3 < frame.bgra.size(); i += 4) {
                                const u8 rgb[3]{frame.bgra[i + 2], frame.bgra[i + 1], frame.bgra[i]};
                                std::fwrite(rgb, 1, 3, file);
                            }
                            std::fclose(file);
                        }
                    }
                }
            }
        }
    }
    // Every 4 s: one-frame outliers among the frames read back in order (a frame far from both
    // neighbours, which agree with each other). An outlier in a frame that skipped draws while
    // their pipelines were made is marked "s" and counted apart: missing objects, not stale data.
    const auto now = std::chrono::steady_clock::now();
    if (luma_report == std::chrono::steady_clock::time_point{}) {
        luma_report = now;
    }
    if (now - luma_report >= std::chrono::seconds{4} && luma_values.size() >= 3) {
        luma_report = now;
        std::ranges::sort(luma_values, {}, &LumaValue::frame);
        u32 outliers = 0;
        u32 clean_outliers = 0;
        u32 contiguous = 0;
        std::string list;
        double total = 0.0;
        for (size_t i = 1; i + 1 < luma_values.size(); ++i) {
            const LumaValue& current = luma_values[i];
            total += current.mean;
            if (luma_values[i - 1].frame + 1 != current.frame ||
                current.frame + 1 != luma_values[i + 1].frame) {
                continue;
            }
            ++contiguous;
            const float before = luma_values[i - 1].mean;
            const float after = luma_values[i + 1].mean;
            const float neighbours = (before + after) * 0.5f;
            if (std::abs(current.mean - neighbours) > 4.0f && std::abs(before - after) < 2.0f) {
                ++outliers;
                const bool skipped = current.skips != luma_values[i - 1].skips;
                clean_outliers += skipped ? 0 : 1;
                if (outliers <= 12) {
                    list += fmt::format(" {}:{:+.1f}{}", current.frame, current.mean - neighbours,
                                        skipped ? "s" : "");
                }
            }
        }
        LOG_INFO(Render_Vulkan,
                 "Luma trace: {} frames ({} with both neighbours), mean {:.1f}, one-frame outliers "
                 "{}, {} without skipped draws{}",
                 luma_values.size(), contiguous,
                 total / static_cast<double>((std::max)(luma_values.size(), size_t{3}) - 2), outliers,
                 clean_outliers, list);
        // Keep the last two frames: the next window's first frames need their neighbours.
        luma_values.erase(luma_values.begin(), luma_values.end() - 2);
    }
    // A new capture into a free slot (a frame is skipped while all are in flight).
    const auto free_slot = std::ranges::find(luma_slots, false, &LumaSlot::pending);
    ++luma_frames;
    Common::N2denTrace::back_frames.store(static_cast<u32>(luma_frames), std::memory_order_release);
    if (free_slot == luma_slots.end()) {
        return;
    }
    if (!free_slot->buffer) {
        free_slot->buffer = CreateWrappedBuffer(memory_allocator, width * height * 4, MemoryUsage::Download);
    }
    const Layout::FramebufferLayout layout = Layout::DefaultFrameLayout(width, height);
    blit_luma.DrawToFrame(device, rasterizer, &free_slot->frame, framebuffers, layout,
                          luma_slots.size(), VK_FORMAT_B8G8R8A8_UNORM);
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([image = *free_slot->frame.image, buffer = *free_slot->buffer, width, height](vk::CommandBuffer cmdbuf) {
        DownloadColorImage(cmdbuf, image, buffer, VkExtent3D{width, height, 1});
    });
    free_slot->tick = scheduler.CurrentTick();
    free_slot->index = luma_frames;
    free_slot->skips = Common::N2denTrace::pipeline_skips.load(std::memory_order_relaxed);
    free_slot->pending = true;
}

std::vector<u8> RendererVulkan::GetAppletCaptureBuffer() {
    using namespace VideoCore::Capture;

    std::vector<u8> out(VideoCore::Capture::TiledSize);

    if (!applet_frame.image) {
        return out;
    }

    const auto dst_buffer =
        CreateWrappedBuffer(memory_allocator, VideoCore::Capture::TiledSize, MemoryUsage::Download);

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([&](vk::CommandBuffer cmdbuf) {
        DownloadColorImage(cmdbuf, *applet_frame.image, *dst_buffer, CaptureImageExtent);
    });

    // Ensure the copy is fully completed before writing the capture
    scheduler.Finish();

    // Swizzle image data to the capture buffer
    dst_buffer.Invalidate();
    Tegra::Texture::SwizzleTexture(out, dst_buffer.Mapped(), BytesPerPixel, LinearWidth,
                                   LinearHeight, LinearDepth, BlockHeight, BlockDepth);

    return out;
}

void RendererVulkan::RenderAppletCaptureLayer(
    std::span<const Tegra::FramebufferConfig> framebuffers) {
    const auto capture_layers = Tegra::FilterLayerStack(
        framebuffers, Service::Nvnflinger::LayerStackId::LastFrame, applet_capture_layers);

    if (capture_layers.empty())
        return;

    if (!applet_frame.image) {
        applet_frame.image = CreateWrappedImage(memory_allocator, CaptureImageSize, CaptureFormat);
        applet_frame.image_view = CreateWrappedImageView(device, applet_frame.image, CaptureFormat);
        applet_frame.framebuffer = blit_applet.CreateFramebuffer(device,
            VideoCore::Capture::Layout, *applet_frame.image_view, CaptureFormat);
    }

    // The capture is drawn every frame. With one set of layer resources each frame waited for
    // the host GPU to finish the previous one; rotating through three lets the GPU thread stay
    // ahead like the swapchain blit does.
    constexpr size_t CaptureResourceSets = 3;
    scheduler.RequestOutsideRenderPassOperationContext();
    blit_applet.DrawToFrame(device, rasterizer, &applet_frame, capture_layers, VideoCore::Capture::Layout,
                            CaptureResourceSets, CaptureFormat);
}

} // namespace Vulkan
