// SPDX-FileCopyrightText: Copyright 2026 n2den Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// Upscaling of the NVN replay. Two callers write a rectangle of one texture from a rectangle of
// another, and both run through `RunUpscale`:
//   * a game's upscaling network, the NDSR compute dispatch (`UpscaleNetwork`: DK's, OoT's), which
//     the replay replaces; it binds only a colour image (OoT: a packed copy of the scene, so the
//     scene colour is traced back through the compute chain), so the host makes the motion
//     vectors and the exposure DLSS needs, and uses the scene's depth when the game's passes
//     show which one it is;
//   * the game's own nvnCommandBufferEvaluateDlss (Op::EvaluateDlss, `EvaluateDlssOp`), where the
//     game supplies depth, motion vectors, jitter, exposure and the history reset.
// The user's upscaler setting decides what writes the output: DLSS when it was asked for and can
// evaluate; FSR1 or bilinear otherwise, a failed DLSS evaluation included, so a texture the game
// expects to be written never stays unwritten.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <set>
#include <string>
#include <string_view>

#include <fmt/format.h>

#include "common/boot_counter.h"
#include "common/logging.h"
#include "common/settings.h"
#include "video_core/nvn/upscaler_status.h"
#include "video_core/renderer_vulkan/nvn/nvn_dlss.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw.h"
#include "video_core/renderer_vulkan/nvn/nvn_format.h"
#include "video_core/renderer_vulkan/nvn/nvn_temporal.h"
#include "video_core/renderer_vulkan/present/fsr.h"
#include "video_core/surface.h"
#include "video_core/upscaler.h"
#include "video_core/vulkan_common/cooperative_matrix.h"

namespace Vulkan::Nvn {

namespace {

/// One line the first time each stage replaces a game's own upscale (`what`).
void LogUpscaleStage(std::string_view what, const char* stage) {
    static Common::BootCounter boot;
    static std::set<std::pair<std::string, std::string>> logged;
    if (boot.First()) {
        logged.clear(); // the first line of a new boot
    }
    if (logged.emplace(what, stage).second) {
        LOG_INFO(Render_Vulkan, "[nvn] {} replaced by {} stage", what, stage);
    }
}

/// One line per distinct game DLSS evaluation shape: the input and output rectangles and formats,
/// the NVN flags, which helper images resolved and the stage that wrote the output.
void LogGameDlssEvaluation(std::string line) {
    static Common::BootCounter boot;
    static std::set<std::string> logged;
    if (boot.First()) {
        logged.clear();
    }
    if (logged.insert(line).second) {
        LOG_INFO(Render_Vulkan, "[nvn] {}", line);
    }
}

/// One line the first time a game's upscaling network is replaced by a stage at an input size.
void LogNetworkReplacement(const char* stage, const VideoCore::Upscaler::Rect& input,
                           const VideoCore::Upscaler::Rect& output, std::string_view basis) {
    static Common::BootCounter boot;
    static std::set<std::string> logged;
    if (boot.First()) {
        logged.clear();
    }
    std::string line = fmt::format("NDSR upscale replaced by {} (input {}x{} -> {}x{} {})", stage,
                                   input.width, input.height, output.width, output.height, basis);
    if (logged.insert(line).second) {
        LOG_INFO(Render_Vulkan, "[nvn] {}", line);
    }
}

/// What writes a replaced 1:1 network's output. `copy` is a plain blit of the scene colour
/// (Longshot's Upscaler "Off"); `fsr1` the FSR1 path at 1:1; `dlaa` DLAA through Streamline (needs
/// the sink's DLSS, SinkDlss::Enabled), the copy when it cannot run. EDEN_NVN_SAME_SIZE_STAGE
/// (copy | fsr1 | dlaa) when it is set; else the NVN-HLE upscaler setting: DLSS / DLAA -> dlaa,
/// FSR 1 -> fsr1, anything else -> copy (the default, as without the variable before).
enum class SameSizeStage { Copy, Fsr1, Dlaa };

[[nodiscard]] SameSizeStage SameSizeStageChoice() {
    // The variable is read once per process; the setting on every call (it changes per game).
    static const std::optional<SameSizeStage> environment =
        []() -> std::optional<SameSizeStage> {
        const char* value = std::getenv("EDEN_NVN_SAME_SIZE_STAGE");
        if (value == nullptr || value[0] == '\0') {
            return std::nullopt;
        }
        const std::string_view text{value};
        if (text == "fsr1") {
            return SameSizeStage::Fsr1;
        }
        if (text == "dlaa") {
            return SameSizeStage::Dlaa;
        }
        if (text != "copy") {
            LOG_WARNING(Render_Vulkan,
                        "[nvn] EDEN_NVN_SAME_SIZE_STAGE={} is not copy, fsr1 or dlaa: copy",
                        value);
        }
        return SameSizeStage::Copy;
    }();
    SameSizeStage stage = SameSizeStage::Copy;
    if (environment.has_value()) {
        stage = *environment;
    } else {
        switch (Settings::values.nvn_upscaler.GetValue()) {
        case Settings::NvnUpscaler::Dlss:
            stage = SameSizeStage::Dlaa;
            break;
        case Settings::NvnUpscaler::Fsr1:
            stage = SameSizeStage::Fsr1;
            break;
        case Settings::NvnUpscaler::Auto:
        case Settings::NvnUpscaler::Copy:
            stage = SameSizeStage::Copy;
            break;
        }
    }
    if (stage == SameSizeStage::Dlaa && !SinkDlss::Enabled()) {
        static Common::BootCounter warned;
        if (warned.First()) {
            LOG_WARNING(Render_Vulkan, "[nvn] the same-size stage is DLAA but the sink's DLSS is "
                                       "off (EDEN_NVN_DLSS): the copy writes the output");
        }
    }
    return stage;
}

/// An optional 0/1 environment override: -1 unset, else 0 or 1 (read once per variable by the
/// callers' statics).
[[nodiscard]] int EnvOverride(const char* name) {
    const char* const value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return -1;
    }
    return value[0] != '0' ? 1 : 0;
}

[[nodiscard]] bool Overridden(int override_value, bool fallback) {
    return override_value < 0 ? fallback : override_value != 0;
}

/// Colour DLSS may treat as linear HDR: a float format (the traced scene colour is
/// B10G11R11_UFLOAT, linear).
[[nodiscard]] bool IsFloatColour(VkFormat format) {
    switch (format) {
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
    case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R16G16B16_SFLOAT:
    case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_R32G32B32_SFLOAT:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] u32 LevelSize(u32 size, u32 level) {
    return (std::max)(size >> level, 1u);
}

void NoteUpscale(Context::Impl& impl, const auto& job, const char* stage) {
    using VideoCore::GameFrameSize::Stage;
    impl.ndsr_upscaled_since_present = true;
    const std::string_view name{stage};
    impl.dlss_upscaled_since_present = name.starts_with("DLSS") || name.starts_with("DLAA");
    impl.upscaled_stage = name.starts_with("DLSS") ? Stage::Dlss
                          : name.starts_with("DLAA") ? Stage::Dlaa
                          : name.starts_with("FSR1") ? Stage::Fsr1
                                                     : Stage::Bilinear;
    if (!impl.dlss_upscaled_since_present) {
        // DLSS / DLAA name their own mode for the status bar (SinkDlss::Evaluate).
        namespace Status = VideoCommon::Nvn::UpscalerStatus;
        Status::Note(name.starts_with("FSR1") ? Status::Stage::Fsr1 : Status::Stage::Copy);
    }
    impl.upscaled_input_width = job.input_rect.width;
    impl.upscaled_input_height = job.input_rect.height;
    impl.upscaled_output_width = job.output_rect.width;
    impl.upscaled_output_height = job.output_rect.height;
}

/// The format an image is read or written as: the view's, when the game's view reinterprets it.
[[nodiscard]] VkFormat ViewFormat(const Texture& texture, VkFormat requested) {
    return requested != VK_FORMAT_UNDEFINED ? requested : texture.host.format;
}

} // namespace

bool DrawContext::EvaluateDlss(const UpscaleJob& job, bool dlaa,
                               const VideoCore::Upscaler::Rect* input_rect_override) {
    // EDEN_NVN_DLSS_TEMPORAL=1, a replaced network: every return before the evaluation tells
    // the temporal inputs DLSS did not take this present (no jitter, no camera pair after it).
    const bool temporal_job = temporal && job.host_inputs && !job.game_dlss;
    const auto not_evaluated = [&] {
        if (temporal_job) {
            temporal->EndEvaluation(impl.presents_recorded, false, 0, false);
        }
        return false;
    };
    // Off unless the sink's DLSS is on (SinkDlss::Enabled): the caller's FSR1 / bilinear stage
    // writes the output, as before.
    if (!SinkDlss::Enabled() || sink_dlss_refused) {
        return not_evaluated();
    }
    if (!sink_dlss) {
        // One per context, shared by its queues (Streamline's state is process-wide).
        sink_dlss = SinkDlss::Create(impl.device, *impl.allocator);
        if (!sink_dlss) {
            sink_dlss_refused = true; // Create said why, once
            return not_evaluated();
        }
    }
    if (sink_dlss->Failed()) {
        sink_dlss_refused = true; // it said why, once
        return not_evaluated();
    }
    // DLSS picks DLAA from the sizes (render == output); `dlaa` is what RunUpscale asked for.
    static_cast<void>(dlaa);
    const VideoCore::Upscaler::Rect& input_rect =
        input_rect_override != nullptr ? *input_rect_override : job.input_rect;
    const VideoCore::Upscaler::Rect& output_rect = job.output_rect;
    // One single-level, single-layer view of a job image, at (x, y) with the given size.
    const auto image_of = [&](const UpscaleImage& image, VkImageAspectFlags aspect, u32 x, u32 y,
                              u32 width, u32 height) -> SinkDlss::Image {
        if (image.texture == nullptr || !image.texture->storage ||
            !image.texture->storage->image) {
            return {};
        }
        Texture& texture = *image.texture;
        const ViewKey key{
            .format = ViewFormat(texture, image.format),
            .type = VK_IMAGE_VIEW_TYPE_2D,
            .aspect = aspect,
            .base_level = image.level,
            .levels = 1,
            .base_layer = image.layer,
            .layers = 1,
        };
        const VkImageView view = resources.GetView(texture, key);
        if (!view) {
            return {};
        }
        return SinkDlss::Image{
            .image = *texture.storage->image,
            .view = view,
            .format = key.format,
            .usage = texture.storage->usage,
            .size = {LevelSize(texture.desc.width, image.level),
                     LevelSize(texture.desc.height, image.level)},
            .x = x,
            .y = y,
            .width = width,
            .height = height,
        };
    };
    SinkDlss::Inputs in;
    in.color = image_of(job.input, VK_IMAGE_ASPECT_COLOR_BIT, input_rect.x, input_rect.y,
                        input_rect.width, input_rect.height);
    if (!in.color.Valid()) {
        return not_evaluated();
    }
    const auto sampled = [](const UpscaleImage& image) {
        return image.texture != nullptr && image.texture->storage &&
               (image.texture->storage->usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0;
    };
    // The scene depth (NVN's default 0 near .. 1 far, not reversed, for a replaced network: the
    // game's flag for its own DLSS); none, or one that cannot be sampled, is the far plane. Read
    // through its depth aspect, or as a colour image (a depth the game keeps in R32F).
    if (sampled(job.depth)) {
        const VkImageAspectFlags aspect =
            (job.depth.texture->storage->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) != 0
                ? VK_IMAGE_ASPECT_DEPTH_BIT
                : VK_IMAGE_ASPECT_COLOR_BIT;
        in.depth = image_of(job.depth, aspect, job.depth.x, job.depth.y, input_rect.width,
                            input_rect.height);
    }
    // The game's motion vectors (its own DLSS). A replaced network has none: DLSS gets zero
    // motion, or with EDEN_NVN_DLSS_TEMPORAL=1 the reprojected camera motion (below).
    if (sampled(job.motion_vectors)) {
        in.motion_vectors = image_of(
            job.motion_vectors, VK_IMAGE_ASPECT_COLOR_BIT, job.motion_vectors.x,
            job.motion_vectors.y, job.motion_width != 0 ? job.motion_width : input_rect.width,
            job.motion_height != 0 ? job.motion_height : input_rect.height);
    }
    in.output = {output_rect.width, output_rect.height};
    // The game's jitter; a replaced network's is the viewport jitter of its scene draws (below,
    // EDEN_NVN_DLSS_TEMPORAL=1), else 0.
    in.jitter = {job.jitter[0], job.jitter[1]};
    in.motion_scale = {job.motion_scale[0], job.motion_scale[1]};
    in.motion_vectors_jittered = job.motion_vectors_jittered;
    static const int depth_inverted_env = EnvOverride("EDEN_NVN_DLSS_DEPTH_INVERTED");
    static const int hdr_env = EnvOverride("EDEN_NVN_DLSS_HDR");
    static const int auto_exposure_env = EnvOverride("EDEN_NVN_DLSS_AUTO_EXPOSURE");
    in.depth_inverted = Overridden(depth_inverted_env, job.depth_inverted);
    // Colour: a replaced network's input is the traced scene colour, linear float
    // (B10G11R11_UFLOAT) that can exceed 1 and has no exposure: the HDR path with DLSS's auto
    // exposure, pre-exposure 1 (as the emulated path's default). A non-float colour is LDR. The
    // game's own DLSS keeps its flags (an exposure image is not tagged: auto exposure instead).
    const bool hdr = job.host_inputs ? IsFloatColour(in.color.format) : job.hdr_color;
    in.hdr = Overridden(hdr_env, hdr);
    in.auto_exposure = Overridden(
        auto_exposure_env, job.host_inputs
                               ? in.hdr
                               : job.auto_exposure || (in.hdr && job.exposure.texture != nullptr));
    in.pre_exposure = !job.host_inputs && job.pre_exposure > 0.0f ? job.pre_exposure : 1.0f;
    in.reset = job.reset;
    in.viewport = job.viewport != 0 ? job.viewport : sink_dlss->NetworkViewport();
    in.present = impl.presents_recorded;

    // The inputs as DLSS reads them, ordered after their writers by the sink's tracking.
    std::array<std::pair<Storage*, VkImageLayout>, 3> reads;
    size_t read_count = 0;
    reads[read_count++] = {job.input.texture->storage.get(), VK_IMAGE_LAYOUT_GENERAL};
    if (in.depth.Valid()) {
        reads[read_count++] = {job.depth.texture->storage.get(), VK_IMAGE_LAYOUT_GENERAL};
    }
    if (in.motion_vectors.Valid()) {
        reads[read_count++] = {job.motion_vectors.texture->storage.get(),
                               VK_IMAGE_LAYOUT_GENERAL};
    }
    host->HostTransitionMany(std::span<const std::pair<Storage*, VkImageLayout>>{reads.data(),
                                                                                 read_count});
    host->HostFlushBarrier();
    // EDEN_NVN_DLSS_TEMPORAL=1, a replaced network: the jitter its scene draws used and the
    // camera motion (a compute pass from the depth DLSS reads); a new history whenever no valid
    // camera pair exists. Its pass binds a compute pipeline: the dirty marking below covers it.
    SinkDlss::Image motion_view; // EDEN_NVN_DLSS_DEBUG=motion: shown in place of the output
    if (temporal_job) {
        const SinkTemporal::Frame frame =
            temporal->BeginEvaluation(impl.device, *impl.allocator, host->HostCmd(), in.present,
                                      in.depth, {in.color.width, in.color.height}, in.output,
                                      in.color.size, this, host->HostFlight().keepalive);
        in.jitter = frame.jitter;
        if (frame.motion.Valid() && SinkTemporal::DebugMotionView()) {
            motion_view = frame.motion;
        }
        if (frame.motion.Valid()) {
            in.motion_vectors = frame.motion;
            in.motion_scale = frame.motion_scale;
            in.motion_vectors_jittered = false;
        }
        in.reset = in.reset || frame.reset;
        if (frame.depth_inverted_known && depth_inverted_env < 0) {
            in.depth_inverted = frame.depth_inverted;
        }
    }
    const bool evaluated =
        sink_dlss->Evaluate(host->HostCmd(), in, host->HostFlight().keepalive);
    if (temporal_job) {
        temporal->EndEvaluation(in.present, evaluated, job.scene_depth_serial, in.reset);
    }
    // DLSS binds its own pipelines, descriptors and push constants (and may have, on a failure):
    // nothing the draw path believes bound survives. Later commands order after its accesses.
    pipeline_dirty = true;
    vertex_dirty = true;
    dynamic_dirty = ~0u;
    bound_pipeline = VK_NULL_HANDLE;
    bound_state.Reset();
    vertex_input_set = false;
    fixed_state_set = false;
    host->HostMarkWritten();
    if (!evaluated) {
        sink_dlss_refused = sink_dlss->Failed();
        return false;
    }
    // DLSS's own RGBA16F output into the output rectangle (format conversion by the blit, 1:1).
    Texture& output = *job.output.texture;
    host->HostTransition(*output.storage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    host->HostFlushBarrier();
    const VkImageBlit region{
        .srcSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .srcOffsets = {{0, 0, 0},
                       {static_cast<s32>(output_rect.width), static_cast<s32>(output_rect.height),
                        1}},
        .dstSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, job.output.level,
                                                   job.output.layer, 1},
        .dstOffsets = {{static_cast<s32>(output_rect.x), static_cast<s32>(output_rect.y), 0},
                       {static_cast<s32>(output_rect.x + output_rect.width),
                        static_cast<s32>(output_rect.y + output_rect.height), 1}},
    };
    if (motion_view.Valid()) {
        // The debug view: the motion image (render size) scaled into the output rectangle.
        host->HostCmd().PipelineBarrier(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
            VkImageMemoryBarrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = motion_view.image,
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
            });
        VkImageBlit view = region;
        view.srcOffsets[1] = {static_cast<s32>(motion_view.width),
                              static_cast<s32>(motion_view.height), 1};
        host->HostCmd().BlitImage(motion_view.image, VK_IMAGE_LAYOUT_GENERAL,
                                  *output.storage->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  view, VK_FILTER_LINEAR);
    } else {
        host->HostCmd().BlitImage(sink_dlss->OutputImage(in.viewport), VK_IMAGE_LAYOUT_GENERAL,
                                  *output.storage->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  region, VK_FILTER_NEAREST);
    }
    host->HostTransition(*output.storage, VK_IMAGE_LAYOUT_GENERAL);
    host->HostNoteTouched(job.output.texture->storage);
    host->HostMarkWritten();
    return true;
}

bool DrawContext::SameSizeDlaa(const UpscaleJob& job) {
    // The guest rectangles are 1:1, but a scaled twin feeds another size: as RunUpscale's DLAA
    // request, an input a little larger is cropped, a smaller one is DLSS, a much larger one
    // keeps the copy (so no size DLSS rejects can switch it off for the session).
    using VideoCore::Upscaler::DlaaRequestStage;
    switch (VideoCore::Upscaler::SelectDlaaRequestStage(job.input_rect.width,
                                                        job.input_rect.height,
                                                        job.output_rect.width,
                                                        job.output_rect.height)) {
    case DlaaRequestStage::Dlaa:
        return EvaluateDlss(job, true);
    case DlaaRequestStage::DlaaCrop: {
        VideoCore::Upscaler::Rect cropped = job.input_rect;
        cropped.width = job.output_rect.width;
        cropped.height = job.output_rect.height;
        return EvaluateDlss(job, true, &cropped);
    }
    case DlaaRequestStage::Dlss:
        return EvaluateDlss(job, false);
    default:
        return false;
    }
}

bool DrawContext::UpscaleWithoutDlss(const UpscaleJob& job, bool use_fsr) {
    Texture& input = *job.input.texture;
    Texture& output = *job.output.texture;
    const VideoCore::Upscaler::Rect& input_rect = job.input_rect;
    const VideoCore::Upscaler::Rect& output_rect = job.output_rect;
    const VkImageSubresourceLayers input_layers{VK_IMAGE_ASPECT_COLOR_BIT, job.input.level,
                                                job.input.layer, 1};
    const VkImageSubresourceLayers output_layers{VK_IMAGE_ASPECT_COLOR_BIT, job.output.level,
                                                 job.output.layer, 1};
    if (use_fsr) {
        const VkExtent2D extent{output_rect.width, output_rect.height};
        if (!impl.upscaler_fsr || !impl.upscaler_fsr_extent ||
            impl.upscaler_fsr_extent->width != extent.width ||
            impl.upscaler_fsr_extent->height != extent.height) {
            impl.upscaler_fsr =
                std::make_shared<::Vulkan::FSR>(impl.device, *impl.allocator, 1, extent);
            impl.upscaler_fsr_extent = extent;
        }
        host->HostFlight().keepalive.push_back(impl.upscaler_fsr);
        ViewKey key{
            .format = ViewFormat(input, job.input.format),
            .type = VK_IMAGE_VIEW_TYPE_2D,
            .aspect = input.storage->aspect,
            .base_level = job.input.level,
            .levels = 1,
            .base_layer = job.input.layer,
            .layers = 1,
        };
        const VkImageView input_view = resources.GetView(input, key);
        if (!input_view) {
            return false;
        }
        host->HostTransition(*input.storage, VK_IMAGE_LAYOUT_GENERAL);
        host->HostFlushBarrier();
        const u32 input_width = LevelSize(input.desc.width, job.input.level);
        const u32 input_height = LevelSize(input.desc.height, job.input.level);
        const Common::Rectangle<f32> crop{
            static_cast<f32>(input_rect.x) / input_width,
            static_cast<f32>(input_rect.y) / input_height,
            static_cast<f32>(input_rect.x + input_rect.width) / input_width,
            static_cast<f32>(input_rect.y + input_rect.height) / input_height,
        };
        impl.upscaler_fsr->Record(impl.device, host->HostCmd(), 0, *input.storage->image,
                                  input_view, {input_width, input_height}, crop);
        host->HostFlight().keepalive.push_back(impl.upscaler_fsr->TakeRecordResources());
        host->HostTransition(*output.storage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        host->HostFlushBarrier();
        const VkImageBlit region{
            .srcSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffsets = {{0, 0, 0},
                           {static_cast<s32>(output_rect.width),
                            static_cast<s32>(output_rect.height), 1}},
            .dstSubresource = output_layers,
            .dstOffsets = {{static_cast<s32>(output_rect.x), static_cast<s32>(output_rect.y), 0},
                           {static_cast<s32>(output_rect.x + output_rect.width),
                            static_cast<s32>(output_rect.y + output_rect.height), 1}},
        };
        host->HostCmd().BlitImage(impl.upscaler_fsr->ResultImage(0), VK_IMAGE_LAYOUT_GENERAL,
                                  *output.storage->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  region, VK_FILTER_LINEAR);
    } else {
        host->HostTransition(*input.storage, VK_IMAGE_LAYOUT_GENERAL);
        host->HostTransition(*output.storage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        host->HostFlushBarrier();
        const VkImageBlit region{
            .srcSubresource = input_layers,
            .srcOffsets = {{static_cast<s32>(input_rect.x), static_cast<s32>(input_rect.y), 0},
                           {static_cast<s32>(input_rect.x + input_rect.width),
                            static_cast<s32>(input_rect.y + input_rect.height), 1}},
            .dstSubresource = output_layers,
            .dstOffsets = {{static_cast<s32>(output_rect.x), static_cast<s32>(output_rect.y), 0},
                           {static_cast<s32>(output_rect.x + output_rect.width),
                            static_cast<s32>(output_rect.y + output_rect.height), 1}},
        };
        host->HostCmd().BlitImage(*input.storage->image, VK_IMAGE_LAYOUT_GENERAL,
                                  *output.storage->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                  region, VK_FILTER_LINEAR);
    }
    host->HostTransition(*output.storage, VK_IMAGE_LAYOUT_GENERAL);
    host->HostNoteTouched(job.output.texture->storage);
    return true;
}

bool DrawContext::RunUpscale(const UpscaleJob& job, const char*& stage) {
    const bool same_extent = job.input_rect.width == job.output_rect.width &&
                             job.input_rect.height == job.output_rect.height;
    const Settings::Upscaler setting = VideoCore::Upscaler::Effective();
    auto backend = VideoCore::Upscaler::SelectBackend(setting, false);
    if ((job.game_dlss || job.prefer_dlss) &&
        (setting == Settings::Upscaler::Auto || setting == Settings::Upscaler::Game)) {
        // The game's own DLSS is the game's upscaler: the real DLSS (DLAA at 1:1) stands in for
        // it; DK's NDSR has no such counterpart (its network is not NGX's).
        backend = same_extent ? VideoCore::Upscaler::Backend::Dlaa
                              : VideoCore::Upscaler::Backend::Dlss;
    }
    if (job.host_inputs && !job.game_dlss && Settings::NvnBilinearUpscale() &&
        (setting == Settings::Upscaler::Auto || setting == Settings::Upscaler::Game)) {
        // NVN-HLE upscaler setting "Plain copy": a replaced network is written by a plain
        // bilinear blit, no FSR1 (EDEN_NVN_DLSS=1 below still wins).
        backend = VideoCore::Upscaler::Backend::Bilinear;
    }
    if (job.host_inputs && SinkDlss::Enabled() &&
        (setting == Settings::Upscaler::Auto || setting == Settings::Upscaler::Game)) {
        // EDEN_NVN_DLSS=1 or the NVN-HLE upscaler setting DLSS / DLAA: a replaced network is
        // written by the real DLSS (DLAA at 1:1); FSR1 still writes it whenever DLSS cannot run.
        backend = same_extent ? VideoCore::Upscaler::Backend::Dlaa
                              : VideoCore::Upscaler::Backend::Dlss;
    }
    if (job.host_inputs) {
        // A network replacement fed its scaled twin: at ratio 1 DLSS is DLAA (a two-pixel twin
        // overhang, OoT 854x480 at 3X, is cropped), whatever DLSS/Auto setting asked for it.
        backend = VideoCore::Upscaler::PromoteNearNativeToDlaa(
            backend, job.input_rect.width, job.input_rect.height, job.output_rect.width,
            job.output_rect.height);
    }
    const bool dlss_requested = backend == VideoCore::Upscaler::Backend::Dlss ||
                                backend == VideoCore::Upscaler::Backend::Dlaa;
    bool dlaa_input_too_large = false;
    // The stage that actually writes the output: DLSS when it can evaluate; otherwise FSR1 for
    // FSR and for a DLSS request that cannot run (unsupported device, missing runtime, a failed
    // evaluation); bilinear only when the user asked for it. The output is always written.
    if (dlss_requested) {
        const bool dlaa = backend == VideoCore::Upscaler::Backend::Dlaa;
        if (dlaa) {
            const auto choice = VideoCore::Upscaler::SelectDlaaRequestStage(
                job.input_rect.width, job.input_rect.height, job.output_rect.width,
                job.output_rect.height);
            using VideoCore::Upscaler::DlaaRequestStage;
            if (choice == DlaaRequestStage::Dlaa) {
                if (EvaluateDlss(job, true)) {
                    stage = "DLAA";
                    return true;
                }
            } else if (choice == DlaaRequestStage::DlaaCrop) {
                // Keep the top/left origin: cropping only the right/bottom excess preserves the
                // scene's pixel centres instead of shifting the image by half a texel.
                VideoCore::Upscaler::Rect cropped = job.input_rect;
                cropped.width = job.output_rect.width;
                cropped.height = job.output_rect.height;
                if (EvaluateDlss(job, true, &cropped)) {
                    stage = "DLAA (output-size crop)";
                    return true;
                }
            } else if (choice == DlaaRequestStage::Dlss) {
                if (EvaluateDlss(job, false)) {
                    stage = "DLSS (DLAA requested)";
                    return true;
                }
            } else {
                // Deliberately retain the old path for a substantially larger input. Downsampling
                // before DLAA needs its own history/motion-vector design; FSR1 is not hidden as
                // DLAA and the transition log names the actual sizes.
                dlaa_input_too_large = true;
            }
        } else if (EvaluateDlss(job, false)) {
            stage = "DLSS";
            return true;
        }
    }
    const bool use_fsr = backend == VideoCore::Upscaler::Backend::Fsr || dlss_requested;
    stage = dlaa_input_too_large ? "FSR1 (DLAA requested, input larger than output)"
            : use_fsr         ? (dlss_requested ? "FSR1 (DLSS unavailable)" : "FSR1")
                              : "bilinear";
    if (UpscaleWithoutDlss(job, use_fsr)) {
        return true;
    }
    if (!use_fsr) {
        return false;
    }
    // FSR records nothing when it fails (no view of the input): a plain bilinear blit still
    // writes the output.
    stage = "bilinear (FSR unavailable)";
    return UpscaleWithoutDlss(job, false);
}

bool DrawContext::UpscaleTwinOf(Texture& texture, u32 level, std::shared_ptr<Texture>& out,
                                const char** why) {
    const auto refuse = [&](const char* reason) {
        if (why != nullptr) {
            *why = reason;
        }
        return false;
    };
    Storage& storage = *texture.storage;
    if (!storage.scaled) {
        return refuse("the image was never drawn at the scaled size");
    }
    if (storage.scale_owner != nullptr) {
        return refuse("the image is itself a twin");
    }
    if (level != 0) {
        return refuse("mip level above 0");
    }
    if (!Rescale::TwinIsCurrent(storage.scale_state)) {
        return refuse("the twin is stale: the guest-sized image was written after it");
    }
    if (host->HostNeedsNativeUpdate(storage)) {
        return refuse("the twin lacks guest bytes written to the image");
    }
    out = impl.ScaledTextureOf(texture);
    if (out == nullptr || !impl.EnsureImage(*out->storage)) {
        return refuse("the twin image could not be created");
    }
    return true;
}

VideoCore::Upscaler::Backend DrawContext::SelectedUpscaleBackend(
    Settings::Upscaler requested) const {
    const bool setting_allows_tensor =
        requested == Settings::Upscaler::Auto || requested == Settings::Upscaler::Game;
    // The game's own network needs cooperative matrices, which Eglantine's lowering does not
    // use: the network is always replaced.
    static_cast<void>(setting_allows_tensor);
    constexpr bool tensor_supported = false;
    return VideoCore::Upscaler::SelectBackend(requested, tensor_supported);
}

bool DrawContext::NetworkReplacementWanted() const {
    const Settings::Upscaler requested = VideoCore::Upscaler::Effective();
    return VideoCore::Upscaler::ReplacesGameNetwork(SelectedUpscaleBackend(requested), requested,
                                                    impl.RescaleFactor().Active());
}

void DrawContext::NoteComputeChain() {
    const StageBindings& bound = stages[STAGE_COMPUTE];
    VideoCore::Upscaler::ComputeChainEntry entry;
    size_t count = 0;
    for (const u64 handle : bound.combined) {
        if (handle == 0) {
            continue;
        }
        if (const auto& texture = TextureOfId(VideoCommon::Nvn::TextureIdOf(static_cast<u32>(handle)))) {
            entry.sampled[count++] = VideoCore::Upscaler::NetworkTextureSerial(
                texture->serial, texture->desc.view_of);
            if (count == entry.sampled.size()) {
                break;
            }
        }
    }
    count = 0;
    for (const u64 handle : bound.image) {
        if (handle == 0) {
            continue;
        }
        if (const auto& texture = TextureOfId(VideoCommon::Nvn::TextureIdOf(static_cast<u32>(handle)))) {
            entry.written[count++] = VideoCore::Upscaler::NetworkTextureSerial(
                texture->serial, texture->desc.view_of);
            if (count == entry.written.size()) {
                break;
            }
        }
    }
    impl.compute_chain[impl.compute_chain_next] = entry;
    impl.compute_chain_next = (impl.compute_chain_next + 1) % impl.compute_chain.size();
    impl.compute_chain_count =
        (std::min)(impl.compute_chain_count + 1, static_cast<u32>(impl.compute_chain.size()));
}

bool DrawContext::ComputeProgramUsesTensorMath(u64 hash, const std::vector<u8>& code,
                                               u32 instruction_bytes) {
    const auto [it, inserted] = impl.program_tensor_math.try_emplace(hash, false);
    if (inserted) {
        const size_t bytes = (std::min<size_t>)(code.size(), instruction_bytes);
        it->second = VideoCore::Upscaler::CodeUsesTensorMath({code.data(), bytes});
    }
    return it->second;
}

bool DrawContext::UpscaleNetwork(const VideoCore::Upscaler::Rect& input_rect,
                                 const VideoCore::Upscaler::Rect& output_rect, u32 output_id) {
    const StageBindings& bound = stages[STAGE_COMPUTE];
    if (bound.combined.empty()) {
        return false;
    }
    const u32 input_id = VideoCommon::Nvn::TextureIdOf(static_cast<u32>(bound.combined[0]));
    const std::shared_ptr<Texture> network_input = TextureOfId(input_id);
    const std::shared_ptr<Texture> output = TextureOfId(output_id);
    if (!network_input || !output || !network_input->storage || !output->storage ||
        !network_input->storage->image || !output->storage->image) {
        static Common::BootCounter resource_logged;
        if (resource_logged.First()) {
            LOG_INFO(Render_Vulkan,
                     "[nvn] NDSR replacement skipped (resources): input={} output={} "
                     "input storage={} output storage={} input image={} output image={}",
                     network_input != nullptr, output != nullptr,
                     network_input && network_input->storage, output && output->storage,
                     network_input && network_input->storage && network_input->storage->image,
                     output && output->storage && output->storage->image);
        }
        return false;
    }
    // The scene colour the network consumes: its input, or what an earlier compute dispatch
    // packed it from.
    const auto is_colour = [&](u64 serial) {
        const auto it = impl.textures.find(serial);
        if (it == impl.textures.end() || !it->second || !it->second->storage ||
            !it->second->storage->image) {
            return false;
        }
        const Texture& texture = *it->second;
        const Storage& storage = *texture.storage;
        return (storage.aspect & VK_IMAGE_ASPECT_COLOR_BIT) != 0 && !texture.host.depth &&
               !VideoCore::Surface::IsPixelFormatInteger(texture.host.pixel_format) &&
               storage.type == VK_IMAGE_TYPE_2D && storage.samples == VK_SAMPLE_COUNT_1_BIT &&
               texture.desc.width >= input_rect.x + input_rect.width &&
               texture.desc.height >= input_rect.y + input_rect.height;
    };
    std::array<VideoCore::Upscaler::ComputeChainEntry, 16> chain;
    const u32 chain_size = static_cast<u32>(impl.compute_chain.size());
    for (u32 i = 0; i < impl.compute_chain_count; ++i) {
        chain[i] = impl.compute_chain[(impl.compute_chain_next + chain_size - 1 - i) % chain_size];
    }
    const u64 colour_serial = VideoCore::Upscaler::FindNetworkColour(
        {chain.data(), impl.compute_chain_count}, network_input->serial, is_colour);
    const auto colour_it = colour_serial != 0 ? impl.textures.find(colour_serial) : impl.textures.end();
    std::shared_ptr<Texture> colour = colour_it != impl.textures.end() ? colour_it->second : nullptr;
    if (!colour || !is_colour(colour_serial)) {
        static Common::BootCounter no_colour_logged;
        if (no_colour_logged.First()) {
            LOG_WARNING(Render_Vulkan,
                        "[nvn] NDSR upscale not replaced: the network's input {}x{} (format {}) is "
                        "not a colour and no scene colour was found behind it; the game's own "
                        "network runs",
                        network_input->desc.width, network_input->desc.height,
                        static_cast<u32>(network_input->host.format));
        }
        return false;
    }
    // The scene's depth: the depth of the last pass that drew into that colour, at its size.
    std::shared_ptr<Texture> depth;
    if (const auto it = impl.pass_depth_of_color.find(colour->serial);
        it != impl.pass_depth_of_color.end()) {
        if (const auto texture = impl.textures.find(it->second);
            texture != impl.textures.end() && texture->second && texture->second->storage &&
            texture->second->storage->image && texture->second->host.depth &&
            (texture->second->storage->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) != 0 &&
            texture->second->storage->type == VK_IMAGE_TYPE_2D &&
            texture->second->storage->samples == VK_SAMPLE_COUNT_1_BIT &&
            texture->second->desc.width == colour->desc.width &&
            texture->second->desc.height == colour->desc.height) {
            depth = texture->second;
        }
    }
    if (!host->HostEnsureContents(*colour->storage, false) ||
        !host->HostEnsureContents(*output->storage, true) ||
        (depth && !host->HostEnsureContents(*depth->storage, false))) {
        return false;
    }
    const u64 seq = host->HostFlight().seq;
    impl.NoteUse(colour->serial, seq);
    impl.NoteUse(output->serial, seq);
    if (depth) {
        impl.NoteUse(depth->serial, seq);
    }

    UpscaleJob job;
    job.input.texture = colour.get();
    job.output.texture = output.get();
    job.depth.texture = depth.get();
    job.input_rect = input_rect;
    job.output_rect = output_rect;
    job.host_inputs = true;
    job.scene_depth_serial = depth ? depth->serial : 0;
    // The Resolution setting: the scene was drawn at the scaled size, and the upscaler is fed the
    // twin (and the depth twin) with the rectangle in its pixels; anything the twins cannot serve
    // leaves the guest-sized images, and the log line says why. A twin the output's size or
    // smaller (DLSS), or larger by at most DlaaCropSlack (DLAA crops it: OoT's 854x480 at 3X is
    // 2562x1440 for a 2560x1440 output), is evaluated as it is.
    const Rescale::Factor factor = impl.RescaleFactor();
    std::array<std::shared_ptr<Texture>, 2> twins;
    std::string basis = "native"; // 1X: nothing else to say
    if (factor.Active()) {
        const Rescale::Rect scaled = Rescale::ScaleRect(
            factor, {static_cast<s32>(input_rect.x), static_cast<s32>(input_rect.y),
                     input_rect.width, input_rect.height});
        const char* twin_refusal = nullptr;
        if (scaled.width == 0 || scaled.height == 0) {
            basis = "native: empty scaled input rectangle";
        } else if (!VideoCore::Upscaler::UpscaleInputFitsOutput(
                       scaled.width, scaled.height, output_rect.width, output_rect.height)) {
            basis = fmt::format("native: the scaled input {}x{} does not fit the {}x{} output",
                                scaled.width, scaled.height, output_rect.width,
                                output_rect.height);
        } else if (!UpscaleTwinOf(*colour, 0, twins[0], &twin_refusal) || twins[0] == nullptr) {
            basis = fmt::format("native: no usable colour twin ({})",
                                twin_refusal != nullptr ? twin_refusal : "none");
        } else {
            job.input.texture = twins[0].get();
            job.input_rect = {static_cast<u32>(scaled.x), static_cast<u32>(scaled.y), scaled.width,
                              scaled.height};
            // Depth must match the colour's size: its twin, or none (the far plane).
            const char* depth_refusal = nullptr;
            if (depth && UpscaleTwinOf(*depth, 0, twins[1], &depth_refusal) && twins[1] != nullptr) {
                job.depth.texture = twins[1].get();
                basis = "twin";
            } else {
                job.depth.texture = nullptr;
                basis = fmt::format("twin, far-plane depth ({})",
                                    depth ? (depth_refusal != nullptr ? depth_refusal : "none")
                                          : "no scene depth found");
            }
        }
    }
    job.input.x = job.input_rect.x;
    job.input.y = job.input_rect.y;
    job.depth.x = job.input_rect.x;
    job.depth.y = job.input_rect.y;
    job.output.x = job.output_rect.x;
    job.output.y = job.output_rect.y;
    job.prefer_dlss = factor.Active() &&
                      VideoCore::Upscaler::Effective() == Settings::Upscaler::Auto;
    const char* stage = nullptr;
    if (input_rect.width == output_rect.width && input_rect.height == output_rect.height) {
        // A 1:1 network (only accepted under EDEN_NVN_REPLACE_SAME_SIZE_NETWORK=1, see
        // nvn_compute.cpp): the stage is the experiment's, not the upscaler setting's.
        // `colour` is the scene colour traced back through the packing prepass (linear, as for
        // a real upscale), not the network's sqrt-encoded input, so it is written as it is.
        const SameSizeStage choice = SameSizeStageChoice();
        if (choice == SameSizeStage::Dlaa && SameSizeDlaa(job)) {
            stage = "DLAA (same size)";
        } else if (choice == SameSizeStage::Dlaa) {
            stage = "copy (same size, DLAA unavailable)";
            if (!UpscaleWithoutDlss(job, false)) {
                return false;
            }
        } else {
            const bool use_fsr = choice == SameSizeStage::Fsr1;
            stage = use_fsr ? "FSR1 (same size)" : "copy (same size)";
            if (!UpscaleWithoutDlss(job, use_fsr)) {
                if (!use_fsr) {
                    return false;
                }
                stage = "copy (same size, FSR unavailable)";
                if (!UpscaleWithoutDlss(job, false)) {
                    return false;
                }
            }
        }
        static Common::BootCounter colour_logged;
        if (colour_logged.First()) {
            LOG_INFO(Render_Vulkan,
                     "[nvn] same-size network: colour {} {}x{} format {} ({}), output {}x{} "
                     "format {}",
                     colour->serial, colour->desc.width, colour->desc.height,
                     static_cast<u32>(colour->host.format),
                     colour->serial == network_input->serial
                         ? "the network's own input: no packing prepass was found"
                         : "traced behind the packing prepass",
                     output->desc.width, output->desc.height,
                     static_cast<u32>(output->host.format));
        }
    } else if (!RunUpscale(job, stage)) {
        return false;
    }
    NoteUpscale(impl, job, stage);
    LogNetworkReplacement(stage, job.input_rect, job.output_rect, basis);
    return true;
}

void DrawContext::EvaluateDlssOp(const VC::EvaluateDlssPayload& p) {
    // The textures stay alive for the whole op, whatever the replay retires meanwhile.
    std::array<std::shared_ptr<Texture>, 6> held;
    std::array<std::shared_ptr<Texture>, 4> twins; ///< the scaled twins evaluated instead
    const auto resolve = [&](size_t slot, const VC::RenderTargetRef& ref, UpscaleImage& image) {
        image = {};
        if (ref.texture.serial == 0) {
            return false;
        }
        const auto it = impl.textures.find(ref.texture.serial);
        if (it == impl.textures.end() || !it->second || !it->second->storage) {
            return false;
        }
        impl.NoteUse(ref.texture.serial, host->HostFlight().seq);
        held[slot] = it->second;
        image.texture = it->second.get();
        image.level = ref.texture.level;
        image.layer = ref.texture.layer;
        if (ref.view.set_format) {
            if (const auto format = ResolveHostFormat(impl.device, ref.view.format)) {
                image.format = format->format;
            }
        }
        return true;
    };
    UpscaleJob job;
    if (!resolve(0, p.input, job.input) || !resolve(1, p.output, job.output)) {
        host->HostFailed("EvaluateDlss", "the input or the output texture is not a live texture");
        return;
    }
    resolve(2, p.motion_vectors, job.motion_vectors);
    resolve(3, p.depth, job.depth);
    resolve(4, p.bias, job.bias);
    resolve(5, p.exposure, job.exposure);

    // An image is read or written in one single-sample 2D level and layer of its texture.
    const auto usable = [](const UpscaleImage& image) {
        const Storage& storage = *image.texture->storage;
        return image.level < storage.levels && image.layer < storage.layers &&
               storage.type == VK_IMAGE_TYPE_2D && storage.samples == VK_SAMPLE_COUNT_1_BIT;
    };
    const auto level_width = [](const UpscaleImage& image) {
        return LevelSize(image.texture->desc.width, image.level);
    };
    const auto level_height = [](const UpscaleImage& image) {
        return LevelSize(image.texture->desc.height, image.level);
    };
    if (!usable(job.input) || !usable(job.output)) {
        host->HostFailed("EvaluateDlss",
                         fmt::format("the input (level {} layer {}) or the output (level {} layer "
                                     "{}) is not a single-sample 2D level of its texture",
                                     job.input.level, job.input.layer, job.output.level,
                                     job.output.layer));
        return;
    }
    // The SDK forwards the rectangles unchecked (OoT sdk+0x65a4e0) and a GPU drops writes outside
    // an image, so the part of the output inside its level is written, from the matching part of
    // the input.
    const VideoCore::Upscaler::GameRect game_input{p.input_base.x, p.input_base.y, p.input_width,
                                                   p.input_height};
    const VideoCore::Upscaler::GameRect game_output{p.output_base.x, p.output_base.y,
                                                    p.output_width, p.output_height};
    const auto rects = VideoCore::Upscaler::ClipUpscaleRects(
        game_input, game_output, level_width(job.input), level_height(job.input),
        level_width(job.output), level_height(job.output));
    if (!rects) {
        host->HostFailed(
            "EvaluateDlss",
            fmt::format("input rectangle {},{} {}x{} or output rectangle {},{} {}x{} has no part "
                        "inside its texture level",
                        p.input_base.x, p.input_base.y, p.input_width, p.input_height,
                        p.output_base.x, p.output_base.y, p.output_width, p.output_height));
        return;
    }
    if (rects->clipped) {
        static Common::BootCounter clip_logged;
        if (clip_logged.First()) {
            LOG_WARNING(Render_Vulkan,
                        "[nvn] EvaluateDlss: input rectangle {},{} {}x{} -> output rectangle "
                        "{},{} {}x{} lies partly outside its texture level; upscaling {},{} {}x{} "
                        "-> {},{} {}x{}",
                        p.input_base.x, p.input_base.y, p.input_width, p.input_height,
                        p.output_base.x, p.output_base.y, p.output_width, p.output_height,
                        rects->input.x, rects->input.y, rects->input.width, rects->input.height,
                        rects->output.x, rects->output.y, rects->output.width,
                        rects->output.height);
        }
    }
    job.input_rect = rects->input;
    job.output_rect = rects->output;
    job.input.x = job.input_rect.x;
    job.input.y = job.input_rect.y;
    job.output.x = job.output_rect.x;
    job.output.y = job.output_rect.y;
    // Resolution scaling: the game's colour, depth and (low-resolution) motion vectors were drawn
    // at the scaled size, and DLSS takes any input size up to the output's: it is evaluated on the
    // twins, with the rectangles, the jitter and the motion scale in their pixels, so the output
    // is rebuilt from more samples. Anything else (an input that is not a current twin, an input
    // larger than the output, a depth or motion vector image drawn at the guest size) leaves every
    // image at the guest size: EnsureContents makes the guest-sized images current.
    const bool motion_low_res = (p.flags & VC::DLSS_FLAG_MV_LOW_RES) != 0;
    const Rescale::Factor factor = impl.RescaleFactor();
    const VideoCore::Upscaler::Rect guest_input_rect = job.input_rect;
    bool dlss_scaled = false;
    if (factor.Active()) {
        const auto twin_of = [&](const UpscaleImage& image, std::shared_ptr<Texture>& out) {
            return image.texture == nullptr || UpscaleTwinOf(*image.texture, image.level, out);
        };
        const Rescale::Rect scaled_input = Rescale::ScaleRect(
            factor, {static_cast<s32>(guest_input_rect.x), static_cast<s32>(guest_input_rect.y),
                     guest_input_rect.width, guest_input_rect.height});
        dlss_scaled = scaled_input.width != 0 && scaled_input.height != 0 &&
                      scaled_input.width <= job.output_rect.width &&
                      scaled_input.height <= job.output_rect.height &&
                      job.input.texture != nullptr && twin_of(job.input, twins[0]) &&
                      twins[0] != nullptr && twin_of(job.depth, twins[1]) &&
                      (!motion_low_res || twin_of(job.motion_vectors, twins[2])) &&
                      twin_of(job.bias, twins[3]);
        if (dlss_scaled) {
            job.input.texture = twins[0].get();
            if (twins[1]) {
                job.depth.texture = twins[1].get();
            }
            if (twins[2]) {
                job.motion_vectors.texture = twins[2].get();
            }
            if (twins[3]) {
                job.bias.texture = twins[3].get();
            }
            job.input_rect = {static_cast<u32>(scaled_input.x), static_cast<u32>(scaled_input.y),
                              scaled_input.width, scaled_input.height};
            job.input.x = job.input_rect.x;
            job.input.y = job.input_rect.y;
        }
    }
    const auto scaled_at = [&](s64 value) -> s64 {
        return dlss_scaled ? factor.ScaleStart(static_cast<s32>(value)) : value;
    };

    if ((job.input.texture->storage->aspect & VK_IMAGE_ASPECT_COLOR_BIT) == 0 ||
        (job.output.texture->storage->aspect & VK_IMAGE_ASPECT_COLOR_BIT) == 0) {
        host->HostUnsupported("EvaluateDlss", "the input or the output texture is not a colour image");
        return;
    }
    if (job.input.texture->storage == job.output.texture->storage &&
        job.input.level == job.output.level && job.input.layer == job.output.layer) {
        host->HostFailed("EvaluateDlss", "the input and the output are the same image");
        return;
    }
    // NGX flags as the SDK forwards them (Initialize, OoT sdk+0x659e40).
    // The inputs that only help DLSS: one that cannot be used is left out, and the evaluation
    // then falls back to a plain upscale (DLSS needs the depth and the motion vectors). Their
    // rectangles follow the input rectangle (the motion vectors the output one when they are at
    // full resolution), moved as clipping moved it.
    const auto optional = [&](UpscaleImage& image, s64 x, s64 y, u32 width, u32 height,
                              const char* name) {
        if (image.texture == nullptr) {
            return;
        }
        const bool placed = usable(image) && x >= 0 && y >= 0 && x <= UINT32_MAX &&
                            y <= UINT32_MAX &&
                            VideoCore::Upscaler::IsValidRect(
                                {static_cast<u32>(x), static_cast<u32>(y), width, height},
                                level_width(image), level_height(image));
        if (!placed || !host->HostEnsureContents(*image.texture->storage, false)) {
            host->HostUnsupported("EvaluateDlss",
                                  fmt::format("the {} image cannot be used at {},{} {}x{}; DLSS "
                                              "is evaluated without it",
                                              name, x, y, width, height));
            image = {};
            return;
        }
        image.x = static_cast<u32>(x);
        image.y = static_cast<u32>(y);
    };
    const s64 input_dx = s64{job.input_rect.x} - p.input_base.x;
    const s64 input_dy = s64{job.input_rect.y} - p.input_base.y;
    const s64 output_dx = s64{job.output_rect.x} - p.output_base.x;
    const s64 output_dy = s64{job.output_rect.y} - p.output_base.y;
    // NGX's motion vectors are at the output resolution unless they are low-res ones.
    const VideoCore::Upscaler::Rect& motion_rect = motion_low_res ? job.input_rect : job.output_rect;
    job.motion_width = motion_rect.width;
    job.motion_height = motion_rect.height;
    // (A scaled evaluation reads the low-resolution images at their scaled positions.)
    const bool motion_scaled = dlss_scaled && motion_low_res;
    optional(job.motion_vectors,
             motion_scaled ? scaled_at(p.motion_vector_base.x + input_dx)
                           : p.motion_vector_base.x + (motion_low_res ? input_dx : output_dx),
             motion_scaled ? scaled_at(p.motion_vector_base.y + input_dy)
                           : p.motion_vector_base.y + (motion_low_res ? input_dy : output_dy),
             job.motion_width, job.motion_height, "motion vector");
    optional(job.depth, scaled_at(p.depth_base.x + input_dx), scaled_at(p.depth_base.y + input_dy),
             job.input_rect.width, job.input_rect.height, "depth");
    optional(job.bias, scaled_at(p.bias_base.x + input_dx), scaled_at(p.bias_base.y + input_dy),
             job.input_rect.width, job.input_rect.height, "bias mask");
    optional(job.exposure, 0, 0, 1, 1, "exposure");
    // The output's other texels survive unless the rectangle covers the whole level.
    const Storage& output_storage = *job.output.texture->storage;
    const bool covers_output =
        job.output_rect.x == 0 && job.output_rect.y == 0 && output_storage.levels == 1 &&
        output_storage.layers == 1 && job.output_rect.width == level_width(job.output) &&
        job.output_rect.height == level_height(job.output);
    if (!host->HostEnsureContents(*job.input.texture->storage, false) ||
        !host->HostEnsureContents(*job.output.texture->storage, covers_output)) {
        host->HostFailed("EvaluateDlss", "the input or the output image could not be created");
        return;
    }

    // The jitter is in pixels of the input and the game's vectors in pixels of its own render size:
    // in a scaled evaluation both are in scaled pixels.
    const f32 input_pixels = dlss_scaled ? factor.UpFactor() : 1.0f;
    const f32 motion_pixels = motion_scaled ? factor.UpFactor() : 1.0f;
    job.jitter[0] = p.jitter[0] * input_pixels;
    job.jitter[1] = p.jitter[1] * input_pixels;
    // The game's scale takes its vectors to pixels of the motion vector rectangle (NGX's MV
    // scale); Streamline's takes them to normalized units of that rectangle.
    job.motion_scale[0] = p.motion_scale[0] * motion_pixels / static_cast<f32>(job.motion_width);
    job.motion_scale[1] = p.motion_scale[1] * motion_pixels / static_cast<f32>(job.motion_height);
    job.pre_exposure = p.pre_exposure;
    job.reset = p.history_reset != 0;
    job.depth_inverted = (p.flags & VC::DLSS_FLAG_DEPTH_INVERTED) != 0;
    // NVN's bit 0 is NGX IsHDR inverted: clear means HDR colour.
    job.hdr_color = (p.flags & VC::DLSS_FLAG_LDR) == 0;
    job.auto_exposure = (p.flags & VC::DLSS_FLAG_AUTO_EXPOSURE) != 0;
    job.motion_vectors_jittered = (p.flags & VC::DLSS_FLAG_MV_JITTERED) != 0;
    job.game_dlss = true;
    job.viewport = p.network;
    job.game_method = p.method;
    job.game_preset = p.preset;

    const char* stage = nullptr;
    if (!RunUpscale(job, stage)) {
        host->HostFailed("EvaluateDlss", "no upscale could be recorded for the output");
        return;
    }
    host->HostMarkWritten();
    NoteUpscale(impl, job, stage);
    LogUpscaleStage("game DLSS evaluation", stage);
    const auto format_of = [](const UpscaleImage& image) {
        if (image.texture == nullptr) {
            return -1;
        }
        return static_cast<int>(image.format != VK_FORMAT_UNDEFINED
                                    ? image.format
                                    : image.texture->storage->host.format);
    };
    LogGameDlssEvaluation(fmt::format(
        "game DLSS evaluation: input {}x{} at {},{} format {} -> output {}x{} at {},{} format {}, "
        "flags {:#x}, motion vectors {} ({}x{}), depth {}, bias {}, exposure {}, scaled {}; "
        "written by the {} stage",
        job.input_rect.width, job.input_rect.height, job.input_rect.x, job.input_rect.y,
        format_of(job.input), job.output_rect.width, job.output_rect.height, job.output_rect.x,
        job.output_rect.y, format_of(job.output), p.flags,
        job.motion_vectors.texture ? "yes" : "no", job.motion_width, job.motion_height,
        job.depth.texture ? "yes" : "no", job.bias.texture ? "yes" : "no",
        job.exposure.texture ? "yes" : "no", dlss_scaled ? "yes" : "no",
        stage ? stage : "?"));
}

} // namespace Vulkan::Nvn
