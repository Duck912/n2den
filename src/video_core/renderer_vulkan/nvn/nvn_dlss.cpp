// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// NVIDIA DLSS / DLAA through NVIDIA Streamline on the NVN-HLE sink's command buffer.
// The mode choice by the actual render/output sizes and the reset
// rules follow the method Longshot (Hookshot fork, GPL-3.0-or-later) describes for its NGX path;
// no code was taken from it.
//
// Streamline per evaluation, as the emulated path's StreamlineDlssUpscaler does it: a frame token
// (one per sink present here; the emulated presenter is idle under nvn-hle, and the sink drives
// no Reflex/PCL markers, which DLSS super resolution does not need), the common constants, the
// DLSS options when the feature changes, then slEvaluateFeature with local resource tags. Each
// network / game DLSS id is its own Streamline viewport (never 0, the emulated path's viewport),
// so the two users can never share a DLSS history.

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string_view>
#include <utility>

#include "common/logging.h"
#include "common/settings.h"
#include "video_core/nvn/upscaler_status.h"
#include "video_core/renderer_vulkan/nvn/nvn_dlss.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

#ifdef EGLANTINE_STREAMLINE
#include "video_core/vulkan_common/streamline.h"
#include "video_core/vulkan_common/streamline_api.h"
#endif

namespace Vulkan::Nvn {

namespace {

[[nodiscard, maybe_unused]] const char* EnvValue(const char* name) {
    const char* const value = std::getenv(name);
    return value != nullptr && value[0] != '\0' ? value : nullptr;
}

#ifdef EGLANTINE_STREAMLINE

/// EDEN_NVN_DLSS_MODE: dlaa | quality | balanced | performance | auto (default).
enum class ModeSetting { Auto, Dlaa, Quality, Balanced, Performance };

[[nodiscard]] ModeSetting ModeFromEnv() {
    static const ModeSetting mode = [] {
        const char* const value = EnvValue("EDEN_NVN_DLSS_MODE");
        if (value == nullptr) {
            return ModeSetting::Auto;
        }
        const std::string_view text{value};
        if (text == "dlaa") {
            return ModeSetting::Dlaa;
        }
        if (text == "quality") {
            return ModeSetting::Quality;
        }
        if (text == "balanced") {
            return ModeSetting::Balanced;
        }
        if (text == "performance") {
            return ModeSetting::Performance;
        }
        if (text != "auto") {
            LOG_WARNING(Render_Vulkan,
                        "[nvn] dlss: EDEN_NVN_DLSS_MODE={} is not dlaa, quality, balanced, "
                        "performance or auto: auto",
                        value);
        }
        return ModeSetting::Auto;
    }();
    return mode;
}

/// EDEN_NVN_DLSS_RESET: always (a new history every evaluation: spatial only, the first eye
/// test) | auto (default: on a new feature, a size change or a gap of more than two presents).
[[nodiscard]] bool ResetAlways() {
    static const bool always = [] {
        const char* const value = EnvValue("EDEN_NVN_DLSS_RESET");
        const bool chosen = value != nullptr && std::string_view{value} == "always";
        if (value != nullptr && !chosen && std::string_view{value} != "auto") {
            LOG_WARNING(Render_Vulkan,
                        "[nvn] dlss: EDEN_NVN_DLSS_RESET={} is neither always nor auto: auto",
                        value);
        }
        return chosen;
    }();
    return always;
}

constexpr VkFormat OUTPUT_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
// Streamline may clear the output before DLSS writes it; the sink blits from it.
constexpr VkImageUsageFlags OUTPUT_USAGE = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_DST_BIT;
constexpr VkImageUsageFlags CONSTANT_USAGE =
    VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
constexpr VkFormat ZERO_MOTION_FORMAT = VK_FORMAT_R16G16_SFLOAT;
constexpr VkFormat FAR_DEPTH_FORMAT = VK_FORMAT_R32_SFLOAT;
constexpr VkImageSubresourceRange COLOR_RANGE{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

/// The DLSS model, as the emulated path picks it (EDEN_NX2_DLSS_PRESET=J|K|L|M|default; K),
/// read once.
[[nodiscard]] sl::DLSSPreset Preset() {
    static const sl::DLSSPreset preset = [] {
        const char* const value = std::getenv("EDEN_NX2_DLSS_PRESET");
        if (value && value[0] >= 'J' && value[0] <= 'M') {
            return static_cast<sl::DLSSPreset>(static_cast<u32>(sl::DLSSPreset::ePresetJ) +
                                               static_cast<u32>(value[0] - 'J'));
        }
        if (value && std::string_view{value} == "default") {
            return sl::DLSSPreset::eDefault;
        }
        return sl::DLSSPreset::ePresetK;
    }();
    return preset;
}

/// Viewports idle this many presents are freed (RetireIdleViewports).
constexpr u64 RETIRE_PRESENTS = 120;
/// The replaced-network viewport of the first SinkDlss ("NVN" 01); later ones count up.
constexpr u32 NETWORK_VIEWPORT_BASE = 0x4e564e01;

/// Streamline's state is process-wide: the SinkDlss of every NVN context (normally one) records
/// under this lock, and frame indices come from one counter, so they never repeat or go back.
struct SharedState {
    std::mutex mutex;
    /// One SinkDlss per allocator (per NVN context), shared by the context's queues.
    std::vector<std::pair<const MemoryAllocator*, std::weak_ptr<SinkDlss>>> instances;
    u32 instance_count{};
    /// Frame tokens: one per (SinkDlss, present); a new one when a viewport evaluates twice in
    /// one present.
    u32 frame_index{};
    const SinkDlss* token_owner{};
    u64 token_present{~u64{0}};
    std::set<u32> token_viewports;
};

[[nodiscard]] SharedState& Shared() {
    static SharedState state;
    return state;
}

[[nodiscard]] const char* ModeName(sl::DLSSMode mode) {
    switch (mode) {
    case sl::DLSSMode::eDLAA:
        return "DLAA";
    case sl::DLSSMode::eMaxQuality:
        return "Quality";
    case sl::DLSSMode::eBalanced:
        return "Balanced";
    case sl::DLSSMode::eMaxPerformance:
        return "Performance";
    case sl::DLSSMode::eUltraPerformance:
        return "UltraPerformance";
    case sl::DLSSMode::eUltraQuality:
        return "UltraQuality";
    default:
        return "Off";
    }
}

[[nodiscard]] sl::DLSSMode ModeByRatio(VkExtent2D render, VkExtent2D output) {
    const f32 ratio = static_cast<f32>(render.width) / static_cast<f32>(output.width);
    if (ratio >= 0.99f) {
        return sl::DLSSMode::eDLAA;
    }
    if (ratio >= 0.65f) {
        return sl::DLSSMode::eMaxQuality;
    }
    if (ratio >= 0.57f) {
        return sl::DLSSMode::eBalanced;
    }
    if (ratio >= 0.49f) {
        return sl::DLSSMode::eMaxPerformance;
    }
    return sl::DLSSMode::eUltraPerformance;
}

[[nodiscard]] sl::float4x4 Identity() {
    sl::float4x4 matrix;
    matrix.setRow(0, {1.0f, 0.0f, 0.0f, 0.0f});
    matrix.setRow(1, {0.0f, 1.0f, 0.0f, 0.0f});
    matrix.setRow(2, {0.0f, 0.0f, 1.0f, 0.0f});
    matrix.setRow(3, {0.0f, 0.0f, 0.0f, 1.0f});
    return matrix;
}

[[nodiscard]] sl::Resource Resource(VkImage image, VkImageView view, VkFormat format,
                                    VkExtent2D size, VkImageUsageFlags usage) {
    sl::Resource resource{sl::ResourceType::eTex2d, image, nullptr, view,
                          static_cast<uint32_t>(VK_IMAGE_LAYOUT_GENERAL)};
    resource.width = size.width;
    resource.height = size.height;
    resource.nativeFormat = static_cast<uint32_t>(format);
    resource.mipLevels = 1;
    resource.arrayLayers = 1;
    resource.flags = 0;
    resource.usage = usage;
    return resource;
}

[[nodiscard]] VkImageMemoryBarrier ImageBarrier(VkImage image, VkAccessFlags src_access,
                                                VkAccessFlags dst_access, VkImageLayout old_layout,
                                                VkImageLayout new_layout) {
    return VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = COLOR_RANGE,
    };
}

#endif // EGLANTINE_STREAMLINE

} // namespace

bool SinkDlss::Enabled() {
    // EDEN_NVN_DLSS when set, else the NVN-HLE upscaler setting (Settings::NvnSinkDlss).
    return Settings::NvnSinkDlss();
}

SinkDlss::SinkDlss(const Device& device_, MemoryAllocator& allocator_, void* set_options_,
                   void* optimal_settings_)
    : device{device_}, allocator{allocator_}, set_options{set_options_},
      optimal_settings{optimal_settings_} {
#ifdef EGLANTINE_STREAMLINE
    // (Create holds the shared lock.)
    network_viewport = NETWORK_VIEWPORT_BASE + Shared().instance_count++;
#endif
}

#ifdef EGLANTINE_STREAMLINE

std::shared_ptr<SinkDlss> SinkDlss::Create(const Device& device, MemoryAllocator& allocator) {
    static bool logged = false;
    const auto refuse = [&](const char* why, int result) -> std::shared_ptr<SinkDlss> {
        if (!logged) {
            logged = true;
            LOG_WARNING(Render_Vulkan,
                        "[nvn] dlss: DLSS is on but cannot run: {} (result {}); the "
                        "replacement falls back to its copy / FSR1 stage",
                        why, result);
        }
        return nullptr;
    };
    if (!Streamline::IsDlssAvailable()) {
        return refuse("NVIDIA Streamline has no DLSS on this device (see its start-up lines)", 0);
    }
    const Streamline::Api& api = Streamline::GetApi();
    void* set_options = nullptr;
    if (const sl::Result result =
            api.get_feature_function(sl::kFeatureDLSS, "slDLSSSetOptions", set_options);
        result != sl::Result::eOk || set_options == nullptr) {
        return refuse("no slDLSSSetOptions", static_cast<int>(result));
    }
    void* optimal_settings = nullptr;
    if (api.get_feature_function(sl::kFeatureDLSS, "slDLSSGetOptimalSettings",
                                 optimal_settings) != sl::Result::eOk) {
        optimal_settings = nullptr; // modes by ratio only
    }
    SharedState& shared = Shared();
    const std::scoped_lock lock{shared.mutex};
    std::erase_if(shared.instances, [](const auto& entry) { return entry.second.expired(); });
    for (const auto& [owner, weak] : shared.instances) {
        if (owner == &allocator) {
            if (auto existing = weak.lock()) {
                return existing;
            }
        }
    }
    auto created = std::make_shared<SinkDlss>(device, allocator, set_options, optimal_settings);
    shared.instances.emplace_back(&allocator, created);
    return created;
}

SinkDlss::~SinkDlss() {
    const bool any_feature = std::ranges::any_of(
        viewports, [](const auto& entry) { return entry.second.options_set; });
    if (!any_feature || !Streamline::IsDlssAvailable()) {
        return;
    }
    // The features' resources may still be read by a flight in progress.
    const std::scoped_lock lock{Shared().mutex};
    void(device.GetLogical().WaitIdle());
    for (const auto& [id, viewport] : viewports) {
        if (viewport.options_set) {
            Streamline::GetApi().free_resources(sl::kFeatureDLSS, sl::ViewportHandle{id});
        }
    }
}

void SinkDlss::Fail(const char* what, int result) {
    if (!failed) {
        LOG_ERROR(Render_Vulkan,
                  "[nvn] dlss: {} failed (Streamline result {}); DLSS stays off for the session, "
                  "the replacement falls back to its copy / FSR1 stage",
                  what, result);
    }
    failed = true;
}

std::shared_ptr<SinkDlss::OwnedImage> SinkDlss::MakeImage(VkFormat format, VkExtent2D extent,
                                                          VkImageUsageFlags usage) {
    auto owned = std::make_shared<OwnedImage>();
    owned->format = format;
    owned->extent = extent;
    owned->image = allocator.CreateImage(VkImageCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {extent.width, extent.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    });
    owned->view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = *owned->image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .components{},
        .subresourceRange = COLOR_RANGE,
    });
    return owned;
}

std::shared_ptr<SinkDlss::OwnedImage> SinkDlss::ConstantImage(vk::CommandBuffer cmd,
                                                              std::shared_ptr<OwnedImage>& slot,
                                                              VkFormat format, VkExtent2D extent,
                                                              f32 value) {
    if (slot && slot->extent.width == extent.width && slot->extent.height == extent.height) {
        return slot;
    }
    // The previous one (another size) lives on in the flights that read it.
    slot = MakeImage(format, extent, CONSTANT_USAGE);
    const VkImage image = *slot->image;
    cmd.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                        ImageBarrier(image, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL));
    VkClearColorValue clear{};
    clear.float32[0] = value;
    cmd.ClearColorImage(image, VK_IMAGE_LAYOUT_GENERAL, clear, COLOR_RANGE);
    cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                        ImageBarrier(image, VK_ACCESS_TRANSFER_WRITE_BIT,
                                     VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                                     VK_IMAGE_LAYOUT_GENERAL));
    return slot;
}

namespace {

/// The optimal render range, with 1/64 of the output as slack, holds `render`.
[[nodiscard]] bool RangeHolds(const sl::DLSSOptimalSettings& settings, VkExtent2D render,
                              VkExtent2D output) {
    const u32 slack_w = output.width / 64;
    const u32 slack_h = output.height / 64;
    return render.width + slack_w >= settings.renderWidthMin &&
           render.width <= settings.renderWidthMax + slack_w &&
           render.height + slack_h >= settings.renderHeightMin &&
           render.height <= settings.renderHeightMax + slack_h;
}

/// slDLSSGetOptimalSettings (`function`) for `mode` at `output`; false when it has no range.
[[nodiscard]] bool OptimalRange(void* function, u32 mode, VkExtent2D output,
                                sl::DLSSOptimalSettings& settings) {
    if (function == nullptr) {
        return false;
    }
    const auto get = reinterpret_cast<PFun_slDLSSGetOptimalSettings*>(function);
    sl::DLSSOptions options{};
    options.mode = static_cast<sl::DLSSMode>(mode);
    options.outputWidth = output.width;
    options.outputHeight = output.height;
    settings = {};
    const bool ok = get(options, settings) == sl::Result::eOk && settings.renderWidthMax != 0;
    // Each mode's range, once per output size: the mode choice below depends on them.
    static std::set<std::pair<u32, u64>> logged;
    if (logged.emplace(mode, (u64{output.width} << 32) | output.height).second) {
        LOG_INFO(Render_Vulkan,
                 "[nvn] dlss: mode {} at {}x{}: optimal {}x{}, range {}x{} .. {}x{}{}",
                 ModeName(static_cast<sl::DLSSMode>(mode)), output.width, output.height,
                 settings.optimalRenderWidth, settings.optimalRenderHeight,
                 settings.renderWidthMin, settings.renderHeightMin, settings.renderWidthMax,
                 settings.renderHeightMax, ok ? "" : " (not available)");
    }
    return ok;
}

} // Anonymous namespace

bool SinkDlss::ModeFits(u32 mode_value, VkExtent2D render, VkExtent2D output) const {
    const auto mode = static_cast<sl::DLSSMode>(mode_value);
    if (mode == sl::DLSSMode::eDLAA) {
        return render.width == output.width && render.height == output.height;
    }
    if (optimal_settings == nullptr) {
        return ModeByRatio(render, output) == mode;
    }
    sl::DLSSOptimalSettings settings{};
    return OptimalRange(optimal_settings, mode_value, output, settings) &&
           RangeHolds(settings, render, output);
}

u32 SinkDlss::ChooseMode(VkExtent2D render, VkExtent2D output) const {
    sl::DLSSMode forced = sl::DLSSMode::eOff;
    switch (ModeFromEnv()) {
    case ModeSetting::Dlaa:
        forced = sl::DLSSMode::eDLAA;
        break;
    case ModeSetting::Quality:
        forced = sl::DLSSMode::eMaxQuality;
        break;
    case ModeSetting::Balanced:
        forced = sl::DLSSMode::eBalanced;
        break;
    case ModeSetting::Performance:
        forced = sl::DLSSMode::eMaxPerformance;
        break;
    case ModeSetting::Auto:
        break;
    }
    if (forced != sl::DLSSMode::eOff) {
        if (ModeFits(static_cast<u32>(forced), render, output)) {
            return static_cast<u32>(forced);
        }
        // A mode DLSS would reject must not switch DLSS off for the session: auto instead.
        static std::set<std::pair<u64, u64>> warned;
        const u64 sizes_in = (u64{render.width} << 32) | render.height;
        const u64 sizes_out = (u64{output.width} << 32) | output.height;
        if (warned.emplace(sizes_in, sizes_out).second) {
            LOG_WARNING(Render_Vulkan,
                        "[nvn] dlss: EDEN_NVN_DLSS_MODE={} does not fit {}x{} -> {}x{}: the "
                        "mode is chosen from the sizes",
                        ModeName(forced), render.width, render.height, output.width,
                        output.height);
        }
    }
    // Auto: DLAA at 1:1, else the mode whose optimal render range holds the actual render size.
    if (render.width == output.width && render.height == output.height) {
        return static_cast<u32>(sl::DLSSMode::eDLAA);
    }
    if (optimal_settings != nullptr) {
        // The ranges may overlap: of the modes whose range holds the size, the one whose optimal
        // width is nearest. A size that no range holds (DLSS evaluates it with an invalid
        // parameter, 0xbad00005) gets no mode when DLSS gave ranges at all.
        u32 best = static_cast<u32>(sl::DLSSMode::eOff);
        u32 best_distance = ~0u;
        bool any_range = false;
        for (const sl::DLSSMode mode : {sl::DLSSMode::eMaxQuality, sl::DLSSMode::eBalanced,
                                        sl::DLSSMode::eMaxPerformance,
                                        sl::DLSSMode::eUltraPerformance}) {
            sl::DLSSOptimalSettings settings{};
            if (!OptimalRange(optimal_settings, static_cast<u32>(mode), output, settings)) {
                continue;
            }
            any_range = true;
            if (!RangeHolds(settings, render, output)) {
                continue;
            }
            const u32 optimal = settings.optimalRenderWidth;
            const u32 distance = optimal > render.width ? optimal - render.width
                                                        : render.width - optimal;
            if (distance < best_distance) {
                best = static_cast<u32>(mode);
                best_distance = distance;
            }
        }
        if (any_range) {
            return best;
        }
    }
    return static_cast<u32>(ModeByRatio(render, output));
}

void SinkDlss::RetireIdleViewports(u64 present, u32 keep) {
    if (present == last_retire_present) {
        return; // once per present
    }
    last_retire_present = present;
    for (auto it = viewports.begin(); it != viewports.end();) {
        if (it->first == keep || present <= it->second.last_present + RETIRE_PRESENTS) {
            ++it;
            continue;
        }
        if (it->second.options_set) {
            Streamline::GetApi().free_resources(sl::kFeatureDLSS, sl::ViewportHandle{it->first});
            LOG_INFO(Render_Vulkan, "[nvn] dlss: viewport {:#x} idle for {} presents: freed",
                     it->first, present - it->second.last_present);
        }
        it = viewports.erase(it); // its output dies with the (long retired) flights' references
    }
}

bool SinkDlss::Evaluate(vk::CommandBuffer cmd, const Inputs& in,
                        std::vector<std::shared_ptr<void>>& keepalive) {
    if (failed || !Streamline::IsDlssAvailable()) {
        return false;
    }
    if (!in.color.Valid() || in.color.width == 0 || in.color.height == 0 || in.output.width == 0 ||
        in.output.height == 0 || in.viewport == 0) {
        return false;
    }
    SharedState& shared = Shared();
    const std::scoped_lock lock{shared.mutex};
    RetireIdleViewports(in.present, in.viewport);
    const VkExtent2D render{in.color.width, in.color.height};
    const sl::ViewportHandle viewport{in.viewport};
    const auto [state_it, fresh] = viewports.try_emplace(in.viewport);
    Viewport& state = state_it->second;
    if (fresh) {
        state.last_present = in.present; // the retirement clock starts now
    }
    // A game's dynamic resolution moves the render size every few frames: the mode in use stays
    // while the new size is in its render range (DLSS takes any size of the range per evaluation,
    // the colour tag's extent), so the feature and its history survive the step.
    const bool keep_mode = state.options_set && state.output_extent.width == in.output.width &&
                           state.output_extent.height == in.output.height &&
                           ModeFromEnv() == ModeSetting::Auto &&
                           ModeFits(state.mode, render, in.output);
    const u32 mode = keep_mode ? state.mode : ChooseMode(render, in.output);
    if (mode == static_cast<u32>(sl::DLSSMode::eOff)) {
        // Between two modes' ranges (a game's dynamic resolution passing through): DLSS would
        // reject the evaluation. The caller's copy / FSR1 stage writes this frame; the feature
        // and its history stay as they are.
        static std::set<std::pair<u64, u64>> warned;
        if (warned.emplace((u64{render.width} << 32) | render.height,
                           (u64{in.output.width} << 32) | in.output.height)
                .second) {
            LOG_INFO(Render_Vulkan,
                     "[nvn] dlss: {}x{} -> {}x{} is in no DLSS mode's render range: the "
                     "copy / FSR1 stage writes those frames",
                     render.width, render.height, in.output.width, in.output.height);
        }
        return false;
    }
    const bool feature_changed =
        !state.options_set || state.output_extent.width != in.output.width ||
        state.output_extent.height != in.output.height || state.mode != mode ||
        state.hdr != in.hdr || state.auto_exposure != in.auto_exposure ||
        state.depth_inverted != in.depth_inverted;
    // A new pre-exposure only needs the options again, not a new history.
    const bool options_changed = feature_changed || state.pre_exposure != in.pre_exposure;

    // Images: the output once per extent (the old one is freed when its flights retire), the
    // zero motion vectors and the far-plane depth only when the caller has none.
    bool output_made = false;
    std::shared_ptr<OwnedImage> zero;
    std::shared_ptr<OwnedImage> far_plane;
    try {
        if (!state.output || state.output->extent.width != in.output.width ||
            state.output->extent.height != in.output.height) {
            state.output = MakeImage(OUTPUT_FORMAT, in.output, OUTPUT_USAGE);
            output_made = true;
        }
        if (!in.motion_vectors.Valid()) {
            zero = ConstantImage(cmd, zero_motion, ZERO_MOTION_FORMAT, render, 0.0f);
        }
        if (!in.depth.Valid()) {
            const f32 far_value = in.depth_inverted ? 0.0f : 1.0f;
            if (far_depth && far_depth_value != far_value) {
                far_depth.reset();
            }
            far_depth_value = far_value;
            far_plane = ConstantImage(cmd, far_depth, FAR_DEPTH_FORMAT, render, far_value);
        }
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "[nvn] dlss: an image could not be created ({})",
                  exception.what());
        state.output.reset();
        Fail("image creation", 0);
        return false;
    }
    keepalive.push_back(state.output);
    if (zero) {
        keepalive.push_back(zero);
    }
    if (far_plane) {
        keepalive.push_back(far_plane);
    }

    const sl::DLSSPreset preset = Preset();
    if (options_changed) {
        sl::DLSSOptions options{};
        options.mode = static_cast<sl::DLSSMode>(mode);
        options.outputWidth = in.output.width;
        options.outputHeight = in.output.height;
        options.preExposure = in.pre_exposure;
        options.colorBuffersHDR = in.hdr ? sl::eTrue : sl::eFalse;
        options.useAutoExposure = in.auto_exposure ? sl::eTrue : sl::eFalse;
        options.dlaaPreset = preset;
        options.qualityPreset = preset;
        options.balancedPreset = preset;
        options.performancePreset = preset;
        options.ultraPerformancePreset = preset;
        options.ultraQualityPreset = preset;
        const auto set = reinterpret_cast<PFun_slDLSSSetOptions*>(set_options);
        if (const sl::Result result = set(viewport, options); result != sl::Result::eOk) {
            Fail("slDLSSSetOptions", static_cast<int>(result));
            return false;
        }
        state.options_set = true;
        state.render = render;
        state.output_extent = in.output;
        state.mode = mode;
        state.hdr = in.hdr;
        state.auto_exposure = in.auto_exposure;
        state.depth_inverted = in.depth_inverted;
        state.pre_exposure = in.pre_exposure;
    }
    if (feature_changed) {
        LOG_INFO(Render_Vulkan,
                 "[nvn] dlss: feature created {}x{} -> {}x{} mode {} (viewport {:#x}, preset {}, "
                 "{} colour{}, depth {}{}, motion {})",
                 render.width, render.height, in.output.width, in.output.height,
                 ModeName(static_cast<sl::DLSSMode>(mode)), in.viewport,
                 static_cast<int>(preset), in.hdr ? "HDR" : "LDR",
                 in.auto_exposure ? " + auto exposure" : "",
                 in.depth.Valid() ? "scene" : "far plane",
                 in.depth_inverted ? " (inverted)" : "",
                 in.motion_vectors.Valid() ? "given" : "zero");
    }
    // The history: the caller's flag, a new feature, the first evaluation or a gap of more than
    // two presents (a menu, a load), or every evaluation with EDEN_NVN_DLSS_RESET=always.
    const bool reset = in.reset || feature_changed || !state.evaluated ||
                       in.present > state.last_present + 2 || ResetAlways();

    // One frame token per sink present; a viewport evaluated twice in one present gets a new one.
    if (shared.token_owner != this || in.present != shared.token_present ||
        shared.token_viewports.contains(in.viewport)) {
        ++shared.frame_index;
        shared.token_owner = this;
        shared.token_present = in.present;
        shared.token_viewports.clear();
    }
    shared.token_viewports.insert(in.viewport);
    const Streamline::Api& api = Streamline::GetApi();
    sl::FrameToken* frame{};
    u32 index = shared.frame_index;
    if (const sl::Result result = api.get_new_frame_token(frame, &index);
        result != sl::Result::eOk || frame == nullptr) {
        Fail("slGetNewFrameToken", static_cast<int>(result));
        return false;
    }

    // DLSS super resolution only reads the jitter, motion and depth conventions; the camera
    // fields are for frame generation, which this does not use.
    const f32 rw = static_cast<f32>(render.width);
    const f32 rh = static_cast<f32>(render.height);
    sl::Constants constants{};
    constants.cameraViewToClip = Identity();
    constants.clipToCameraView = Identity();
    constants.clipToLensClip = Identity();
    constants.clipToPrevClip = Identity();
    constants.prevClipToClip = Identity();
    constants.jitterOffset = {in.jitter[0], in.jitter[1]};
    constants.mvecScale = in.motion_vectors.Valid()
                              ? sl::float2{in.motion_scale[0], in.motion_scale[1]}
                              : sl::float2{1.0f, 1.0f};
    constants.cameraPinholeOffset = {0.0f, 0.0f};
    constants.cameraPos = {0.0f, 0.0f, 0.0f};
    constants.cameraUp = {0.0f, 1.0f, 0.0f};
    constants.cameraRight = {1.0f, 0.0f, 0.0f};
    constants.cameraFwd = {0.0f, 0.0f, 1.0f};
    constants.cameraNear = 0.1f;
    constants.cameraFar = 1000.0f;
    constants.cameraFOV = 1.0f;
    constants.cameraAspectRatio = rw / rh;
    constants.depthInverted = in.depth_inverted ? sl::eTrue : sl::eFalse;
    constants.cameraMotionIncluded = sl::eTrue;
    constants.motionVectors3D = sl::eFalse;
    constants.reset = reset ? sl::eTrue : sl::eFalse;
    constants.motionVectorsJittered = in.motion_vectors_jittered ? sl::eTrue : sl::eFalse;
    if (const sl::Result result = api.set_constants(constants, *frame, viewport);
        result != sl::Result::eOk) {
        Fail("slSetConstants", static_cast<int>(result));
        return false;
    }

    // The output: made now (undefined), or read by the previous evaluation's blit.
    const VkImage output_image = *state.output->image;
    cmd.PipelineBarrier(
        output_made ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
        ImageBarrier(output_image, output_made ? 0 : VK_ACCESS_TRANSFER_READ_BIT,
                     VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                         VK_ACCESS_TRANSFER_WRITE_BIT,
                     output_made ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
                     VK_IMAGE_LAYOUT_GENERAL));

    const Image depth_image =
        in.depth.Valid()
            ? in.depth
            : Image{*far_plane->image, *far_plane->view, FAR_DEPTH_FORMAT, CONSTANT_USAGE, render,
                    0, 0, render.width, render.height};
    const Image motion_image =
        in.motion_vectors.Valid()
            ? in.motion_vectors
            : Image{*zero->image, *zero->view, ZERO_MOTION_FORMAT, CONSTANT_USAGE, render, 0, 0,
                    render.width, render.height};
    sl::Resource color = Resource(in.color.image, in.color.view, in.color.format, in.color.size,
                                  in.color.usage);
    sl::Resource depth = Resource(depth_image.image, depth_image.view, depth_image.format,
                                  depth_image.size, depth_image.usage);
    sl::Resource motion = Resource(motion_image.image, motion_image.view, motion_image.format,
                                   motion_image.size, motion_image.usage);
    sl::Resource output = Resource(output_image, *state.output->view, OUTPUT_FORMAT, in.output,
                                   OUTPUT_USAGE);
    const sl::Extent color_area{in.color.y, in.color.x, in.color.width, in.color.height};
    const sl::Extent depth_area{depth_image.y, depth_image.x, depth_image.width,
                                depth_image.height};
    const sl::Extent motion_area{motion_image.y, motion_image.x, motion_image.width,
                                 motion_image.height};
    const sl::Extent output_area{0, 0, in.output.width, in.output.height};
    // Local tags: everything stays put until the evaluation is recorded.
    sl::ResourceTag color_tag{&color, sl::kBufferTypeScalingInputColor,
                              sl::ResourceLifecycle::eValidUntilEvaluate, &color_area};
    sl::ResourceTag depth_tag{&depth, sl::kBufferTypeDepth,
                              sl::ResourceLifecycle::eValidUntilEvaluate, &depth_area};
    sl::ResourceTag motion_tag{&motion, sl::kBufferTypeMotionVectors,
                               sl::ResourceLifecycle::eValidUntilEvaluate, &motion_area};
    sl::ResourceTag output_tag{&output, sl::kBufferTypeScalingOutputColor,
                               sl::ResourceLifecycle::eValidUntilEvaluate, &output_area};
    const sl::BaseStructure* inputs[] = {&viewport, &color_tag, &depth_tag, &motion_tag,
                                         &output_tag};
    if (const sl::Result result =
            api.evaluate_feature(sl::kFeatureDLSS, *frame, inputs, std::size(inputs),
                                 reinterpret_cast<sl::CommandBuffer*>(*cmd));
        result != sl::Result::eOk) {
        Fail("slEvaluateFeature (DLSS)", static_cast<int>(result));
        return false;
    }
    // DLSS's writes (compute, or a clear) before the caller's transfer read of the output.
    cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                        ImageBarrier(output_image,
                                     VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                                     VK_ACCESS_TRANSFER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                                     VK_IMAGE_LAYOUT_GENERAL));
    state.evaluated = true;
    state.last_present = in.present;
    {
        // The status bar names the mode DLSS ran in, and whether it had jitter and motion.
        namespace Status = VideoCommon::Nvn::UpscalerStatus;
        Status::Stage shown = Status::Stage::DlssUltraPerformance;
        switch (static_cast<sl::DLSSMode>(mode)) {
        case sl::DLSSMode::eDLAA:
            shown = Status::Stage::Dlaa;
            break;
        case sl::DLSSMode::eUltraQuality:
            shown = Status::Stage::DlssUltraQuality;
            break;
        case sl::DLSSMode::eMaxQuality:
            shown = Status::Stage::DlssQuality;
            break;
        case sl::DLSSMode::eBalanced:
            shown = Status::Stage::DlssBalanced;
            break;
        case sl::DLSSMode::eMaxPerformance:
            shown = Status::Stage::DlssPerformance;
            break;
        default:
            break;
        }
        const bool network = (in.viewport & ~0xffu) == (NETWORK_VIEWPORT_BASE & ~0xffu);
        Status::Note(shown, in.motion_vectors.Valid(),
                     network ? Status::Slot::Network : Status::Slot::Game);
    }
    ++evaluations;
    if (evaluations % 600 == 0) {
        LOG_INFO(Render_Vulkan, "[nvn] dlss: {} evaluations ({}x{} -> {}x{} mode {}{})",
                 evaluations, render.width, render.height, in.output.width, in.output.height,
                 ModeName(static_cast<sl::DLSSMode>(mode)), reset ? ", reset" : "");
    }
    return true;
}

VkImage SinkDlss::OutputImage(u32 viewport) const {
    const auto it = viewports.find(viewport);
    return it != viewports.end() && it->second.output ? *it->second.output->image
                                                      : VkImage{VK_NULL_HANDLE};
}

#else // EGLANTINE_STREAMLINE

std::shared_ptr<SinkDlss> SinkDlss::Create(const Device& device, MemoryAllocator& allocator) {
    static_cast<void>(device);
    static_cast<void>(allocator);
    static bool logged = false;
    if (!logged) {
        logged = true;
        LOG_WARNING(Render_Vulkan, "[nvn] dlss: DLSS is on but this build has no NVIDIA "
                                   "Streamline (ENABLE_NVIDIA_DLSS); the replacement falls back");
    }
    return nullptr;
}

SinkDlss::~SinkDlss() = default;

bool SinkDlss::Evaluate(vk::CommandBuffer cmd, const Inputs& in,
                        std::vector<std::shared_ptr<void>>& keepalive) {
    static_cast<void>(cmd);
    static_cast<void>(in);
    static_cast<void>(keepalive);
    return false;
}

VkImage SinkDlss::OutputImage(u32 viewport) const {
    static_cast<void>(viewport);
    return VK_NULL_HANDLE;
}

#endif // EGLANTINE_STREAMLINE

} // namespace Vulkan::Nvn
