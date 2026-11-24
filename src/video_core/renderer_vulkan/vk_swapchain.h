// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <vector>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/present/present_semaphore_ticks.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Layout {
struct FramebufferLayout;
}

namespace Vulkan {

/// Settings::values.async_presentation, or EDEN_ASYNC_PRESENTATION=0/1 when set (a test override
/// that leaves the saved setting alone).
[[nodiscard]] bool AsyncPresentationEnabled();

class Device;
class Scheduler;

class Swapchain {
public:
    explicit Swapchain(
        VkSurfaceKHR_T* surface,
        const Device& device,
        Scheduler& scheduler,
        u32 width,
        u32 height);
    ~Swapchain();

    /// Creates (or recreates) the swapchain with a given size.
    void Create(
        VkSurfaceKHR_T* surface,
        u32 width,
        u32 height);

    /// Acquires the next image in the swapchain, waits as needed.
    /// `renderer_paced`: the frame was rendered through `scheduler`, whose tick then paces the
    /// image. False for frames another submitter rendered (nvn-hle's replay): the renderer's
    /// scheduler holds none of their work, so those frames are paced by their own fences and by
    /// the present timeline (`PresentTimeline`).
    bool AcquireNextImage(bool renderer_paced = true);

    /// Timeline semaphore the submission that waits on `CurrentPresentSemaphore` of a frame that
    /// is not renderer paced signals, with `NextPresentTimelineValue`; null unless nvn-hle runs.
    /// Recorded by `MarkPresentTimelineSubmitted` once submitted.
    [[nodiscard]] VkSemaphore PresentTimeline() const {
        return *present_timeline;
    }

    [[nodiscard]] u64 NextPresentTimelineValue() const {
        return present_semaphore_ticks.Next();
    }

    /// The submission waiting on `CurrentPresentSemaphore` was submitted and signals
    /// `NextPresentTimelineValue`: the next acquire with this semaphore waits for it to have run.
    void MarkPresentTimelineSubmitted() {
        present_semaphore_ticks.Submitted(frame_index);
    }

    /// Presents the rendered image to the swapchain.
    void Present(VkSemaphore render_semaphore);

    /// Returns true when the swapchain needs to be recreated.
    bool NeedsRecreation() const {
        return IsSubOptimal() || NeedsPresentModeUpdate();
    }

    /// Returns true when the swapchain is outdated.
    bool IsOutDated() const {
        return is_outdated;
    }

    /// Returns true when the swapchain is suboptimal.
    bool IsSubOptimal() const {
        return is_suboptimal;
    }

    VkExtent2D GetSize() const {
        return extent;
    }

    std::size_t GetImageCount() const {
        return image_count;
    }

    std::size_t GetImageIndex() const {
        return image_index;
    }

    std::size_t GetFrameIndex() const {
        return frame_index;
    }

    VkImage GetImageIndex(std::size_t index) const {
        return images[index];
    }

    VkImage CurrentImage() const {
        return images[image_index];
    }

    VkFormat GetImageViewFormat() const {
        return image_view_format;
    }

    VkFormat GetImageFormat() const {
        return surface_format.format;
    }

    VkSemaphore CurrentPresentSemaphore() const {
        return *present_semaphores[frame_index];
    }

    VkSemaphore CurrentRenderSemaphore() const {
        return *render_semaphores[image_index];
    }

    u32 GetWidth() const {
        return width;
    }

    u32 GetHeight() const {
        return height;
    }

    VkExtent2D GetExtent() const {
        return extent;
    }

private:
    void CreateSwapchain(const VkSurfaceCapabilitiesKHR& capabilities);
    void CreateSemaphores();
    void CreateImageViews();

    void Destroy();

    bool NeedsPresentModeUpdate() const;

    VkSurfaceKHR_T* surface;

    const Device& device;
    Scheduler& scheduler;

    vk::SwapchainKHR swapchain;

    std::size_t image_count{};
    std::vector<VkImage> images;
    std::vector<u64> resource_ticks;
    std::vector<vk::Semaphore> present_semaphores;
    std::vector<vk::Semaphore> render_semaphores;
    /// Present timeline (see PresentTimeline) and, per present semaphore, its value signalled
    /// by the submission that last waited on it. Only nvn-hle creates and uses them.
    vk::Semaphore present_timeline;
    PresentSemaphoreTicks present_semaphore_ticks;

    u32 width;
    u32 height;

    u32 image_index{};
    u32 frame_index{};

    VkFormat image_view_format{};
    VkExtent2D extent{};
    VkPresentModeKHR present_mode{};
    VkSurfaceFormatKHR surface_format{};
    bool is_hdr{}; ///< The surface is HDR10 (hdr_output on and offered by the display).
    bool has_imm{false};
    bool has_mailbox{false};
    bool has_fifo_relaxed{false};

    bool is_outdated{};
    bool is_suboptimal{};
};

} // namespace Vulkan
