// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA DLSS / DLAA (through NVIDIA Streamline) recorded on the NVN-HLE sink's own command
// buffer, in place of a game's upscaling network or of the game's own DLSS evaluation. A twin of
// the emulated path's StreamlineDlssUpscaler (vk_streamline_dlss.cpp), which records through the
// emulated GPU's Scheduler: this one shares no state with it, uses its own Streamline viewports
// (never 0, the emulated path's) and only runs with EDEN_NVN_DLSS=1.

#pragma once

#include <array>
#include <map>
#include <memory>
#include <vector>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
class Device;
class MemoryAllocator;
} // namespace Vulkan

namespace Vulkan::Nvn {

class SinkDlss {
public:
    /// One image DLSS reads: a single-level, single-layer view, and the rectangle it reads in it.
    struct Image {
        VkImage image{};
        VkImageView view{};
        VkFormat format{VK_FORMAT_UNDEFINED};
        VkImageUsageFlags usage{};
        VkExtent2D size{}; ///< the viewed level's size
        u32 x{};
        u32 y{};
        u32 width{};
        u32 height{};
        [[nodiscard]] bool Valid() const noexcept {
            return image != VK_NULL_HANDLE && view != VK_NULL_HANDLE;
        }
    };

    /// One evaluation. The colour, depth and motion vector images are in the GENERAL layout and
    /// ordered after their writers when Evaluate records (the caller's layout tracking does it).
    struct Inputs {
        Image color; ///< the render-size picture (color.width x color.height)
        /// The scene depth at the colour's rectangle size. Invalid: a far-plane depth is used.
        Image depth;
        /// Motion to the previous frame. Invalid: zero motion. A replaced network's is the
        /// reprojected camera motion of nvn_temporal.h (EDEN_NVN_DLSS_TEMPORAL=1).
        Image motion_vectors;
        VkExtent2D output{}; ///< the output rectangle's size
        /// This frame's sample offset in render pixels (x right, y down: the content's shift). A
        /// replaced network's is the viewport jitter of nvn_temporal.h, else 0.
        std::array<f32, 2> jitter{};
        /// Streamline's mvecScale: the motion vectors times it are normalized.
        std::array<f32, 2> motion_scale{1.0f, 1.0f};
        bool motion_vectors_jittered{};
        bool depth_inverted{};
        bool hdr{};           ///< linear colour (colorBuffersHDR)
        /// The colour is display-encoded (sRGB) and the replaced network stores linear colour:
        /// the output is decoded with the sRGB curve after the evaluation (nvn_srgb_decode.comp).
        bool decode_srgb_output{};
        bool auto_exposure{}; ///< DLSS computes the exposure (no exposure image is tagged)
        f32 pre_exposure{1.0f};
        /// The caller's history reset (the game's flag); the size/gap/feature rules add to it.
        bool reset{};
        /// The Streamline viewport (one DLSS feature and history each); never 0.
        u32 viewport{};
        /// The sink's present counter: a gap of more than two presents resets the history, and
        /// evaluations of one present share a frame token.
        u64 present{};
    };

    /// The sink evaluates DLSS (default off): EDEN_NVN_DLSS=1, or the NVN-HLE upscaler setting
    /// when the variable is unset (Settings::NvnSinkDlss).
    [[nodiscard]] static bool Enabled();

    /// The process's SinkDlss for `allocator` (one per NVN context: every queue's DrawContext
    /// shares it), made on first use. Null (with one log line) when Streamline has no DLSS for
    /// this device or this build has no Streamline.
    [[nodiscard]] static std::shared_ptr<SinkDlss> Create(const Device& device,
                                                          MemoryAllocator& allocator);

    SinkDlss(const Device& device, MemoryAllocator& allocator, void* set_options,
             void* optimal_settings);
    ~SinkDlss();
    SinkDlss(const SinkDlss&) = delete;
    SinkDlss& operator=(const SinkDlss&) = delete;

    /// Records one evaluation into `cmd` (outside a render pass). On true the result is in
    /// OutputImage(in.viewport): RGBA16F, in.output's size, GENERAL layout, its DLSS writes made
    /// visible to transfer reads. The images it made are added to `keepalive` (the flight's).
    /// False when DLSS cannot run (logged once; DLSS then stays off for the session): nothing
    /// was written and the caller writes the output with another stage.
    bool Evaluate(vk::CommandBuffer cmd, const Inputs& in,
                  std::vector<std::shared_ptr<void>>& keepalive);

    [[nodiscard]] VkImage OutputImage(u32 viewport) const;

    /// A Streamline call failed: DLSS is off for the session.
    [[nodiscard]] bool Failed() const noexcept {
        return failed;
    }

    /// The viewport of a replaced network (no game id): never 0 (the emulated path's), and
    /// distinct per SinkDlss.
    [[nodiscard]] u32 NetworkViewport() const noexcept {
        return network_viewport;
    }

    /// Every evaluation the session recorded (for the counter line).
    [[nodiscard]] u64 Evaluations() const noexcept {
        return evaluations;
    }

private:
    /// One owned image and its view (kept alive by the flights that use it).
    struct OwnedImage {
        vk::Image image;
        vk::ImageView view;
        VkFormat format{};
        VkExtent2D extent{};
    };
    /// One Streamline viewport: its feature options and output.
    struct Viewport {
        std::shared_ptr<OwnedImage> output;
        VkExtent2D render{};
        VkExtent2D output_extent{};
        u32 mode{};
        bool hdr{};
        bool auto_exposure{};
        bool depth_inverted{};
        f32 pre_exposure{1.0f};
        bool options_set{};
        u64 last_present{}; ///< the last evaluation's (or the creation's) present
        bool evaluated{};
    };

    std::shared_ptr<OwnedImage> MakeImage(VkFormat format, VkExtent2D extent,
                                          VkImageUsageFlags usage);
    /// A render-size image cleared to `value` (zero motion, far-plane depth), made on first use
    /// and when the size changes; the clear is recorded into `cmd`.
    std::shared_ptr<OwnedImage> ConstantImage(vk::CommandBuffer cmd,
                                              std::shared_ptr<OwnedImage>& slot, VkFormat format,
                                              VkExtent2D extent, f32 value);
    /// EDEN_NVN_DLSS_MODE's mode when it fits the sizes, else (warned once) the automatic one;
    /// sl::DLSSMode::eOff when DLSS gave ranges and none holds the render size.
    [[nodiscard]] u32 ChooseMode(VkExtent2D render, VkExtent2D output) const;
    /// Whether DLSS accepts `render` for `mode` at `output` (its optimal range, else the ratio).
    [[nodiscard]] bool ModeFits(u32 mode, VkExtent2D render, VkExtent2D output) const;
    /// Frees the features (and drops the outputs) of viewports idle for RETIRE_PRESENTS presents:
    /// a game makes a new DLSS id per nvnDlss initialisation. Their flights retired long ago.
    void RetireIdleViewports(u64 present, u32 keep);
    void Fail(const char* what, int result);
    /// The sRGB decode pass over the output (made on first use); false when it cannot run.
    bool EnsureDecodePass();

    const Device& device;
    MemoryAllocator& allocator;
    void* set_options{};      ///< slDLSSSetOptions
    void* optimal_settings{}; ///< slDLSSGetOptimalSettings (may be null)
    u32 network_viewport{};
    std::map<u32, Viewport> viewports;
    u64 last_retire_present{};
    std::shared_ptr<OwnedImage> zero_motion;
    std::shared_ptr<OwnedImage> far_depth;
    f32 far_depth_value{-1.0f};
    u64 evaluations{};
    bool failed{};
    vk::DescriptorSetLayout decode_set_layout;
    vk::PipelineLayout decode_layout;
    vk::ShaderModule decode_module;
    vk::Pipeline decode_pipeline;
    bool decode_failed{};
};

} // namespace Vulkan::Nvn
