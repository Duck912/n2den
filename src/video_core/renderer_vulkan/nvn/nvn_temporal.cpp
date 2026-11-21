// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-FileCopyrightText: Copyright (C) 2026 the Longshot contributors
// SPDX-FileCopyrightText: Copyright (C) 2026 NotRexed and the Hookshot contributors
// SPDX-FileCopyrightText: Copyright (C) 2026 hckrman101 (the temporal chain, Hookshot #841)
// SPDX-License-Identifier: GPL-3.0-or-later

// Temporal inputs of the sink's DLSS for a replaced upscaling network (see
// nvn_temporal.h). The method follows what Longshot (Hookshot
// fork, GPL-3.0-or-later; the temporal chain credited to hckrman101) documents for its NGX path:
// the camera block layout, Halton (2, 3) jitter from index 1 over clamp(8 (out/render)^2, 8, 64)
// presents, and the depth reprojection through this and the previous frame's view-projection.
// Provenance: `Reprojection` is adapted from Longshot's dlss_math.h `reprojection` (commit e41de66:
// the same augmented-matrix Gauss-Jordan inverse with partial pivoting, thresholds and float-range
// check), `Phases` from its `dlss_jitter_phases`, and the near/far/fov formulas of the camera log
// from vk_temporal.cpp; modified for n2den on 2026-11-16 (GPL-3.0 section 5a). The rest was written
// for n2den. Unlike Longshot, which jitters by patching the game's projection, the sink jitters by
// a sub-pixel viewport offset on the scene draws at replay, so the captured matrices are
// unjittered and need no unjitter step.
//
// Conventions:
//  - Jitter: a viewport moved by (dx, dy) framebuffer pixels moves the content by (dx, dy), x
//    right, y down (the image's first row is its top, nvn_viewport.h). DLSS (Streamline's
//    jitterOffset, as NGX's InJitterOffset) takes that content shift in render pixels, x right /
//    y down: Longshot passes (jitter_ndc.x * w / 2, -jitter_ndc.y * h / 2) for a projection jitter
//    that moves NDC by jitter_ndc, which is the same shift. EDEN_NVN_DLSS_JITTER_SIGN=x,y (+1 or
//    -1 each) flips what DLSS is told (not the offset drawn) for the eye test: with a wrong sign a
//    still scene shimmers.
//  - Motion: render pixels, from this frame's position to the previous frame's (previous minus
//    current), x right / y down, mvecScale = (1 / render width, 1 / render height), not jittered
//    (the unjittered position is the pixel centre minus the jitter).
//  - Depth: stored = min + (max - min) * t with viewport 0's range; clip z = 2t - 1 for NVN's
//    default (OpenGL) depth mode, t for the 0..1 mode. Whether depth is inverted (near stores
//    more than far) is read from the projection and the range, not assumed.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <string_view>
#include <utility>

#include "common/boot_counter.h"
#include "common/logging.h"
#include "common/settings.h"
#include "video_core/host_shaders/nvn_dlss_motion_comp_spv.h"
#include "video_core/renderer_vulkan/nvn/nvn_temporal.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"

namespace Vulkan::Nvn {

namespace {

[[nodiscard]] const char* EnvValue(const char* name) {
    const char* const value = std::getenv(name);
    return value != nullptr && value[0] != '\0' ? value : nullptr;
}

/// A 0/1 switch: unset is `fallback`, "0" off, anything else on.
[[nodiscard]] bool EnvSwitch(const char* name, bool fallback) {
    const char* const value = EnvValue(name);
    return value == nullptr ? fallback : value[0] != '0';
}

/// EDEN_NVN_DLSS_JITTER=0|1 (1 by default).
[[nodiscard]] bool JitterEnabled() {
    static const bool enabled = EnvSwitch("EDEN_NVN_DLSS_JITTER", true);
    return enabled;
}

/// EDEN_NVN_DLSS_MVEC=0|1 (1 by default).
[[nodiscard]] bool MotionEnabled() {
    static const bool enabled = EnvSwitch("EDEN_NVN_DLSS_MVEC", true);
    return enabled;
}

/// EDEN_NVN_DLSS_JITTER_SIGN=x,y: +1 / -1 per axis on the jitter DLSS is told ("1,1" default).
[[nodiscard]] std::array<f32, 2> JitterSign() {
    static const std::array<f32, 2> sign = [] {
        std::array<f32, 2> result{1.0f, 1.0f};
        const char* const value = EnvValue("EDEN_NVN_DLSS_JITTER_SIGN");
        if (value == nullptr) {
            return result;
        }
        char* end = nullptr;
        const f32 x = std::strtof(value, &end);
        const bool comma = end != nullptr && *end == ',';
        const f32 y = comma ? std::strtof(end + 1, &end) : 0.0f;
        if (!comma || x == 0.0f || y == 0.0f) {
            LOG_WARNING(Render_Vulkan,
                        "[nvn] dlss temporal: EDEN_NVN_DLSS_JITTER_SIGN={} is not x,y with x and "
                        "y +1 or -1: 1,1",
                        value);
            return result;
        }
        result = {x < 0.0f ? -1.0f : 1.0f, y < 0.0f ? -1.0f : 1.0f};
        LOG_INFO(Render_Vulkan,
                 "[nvn] dlss temporal: the jitter DLSS is told is multiplied by "
                 "({}, {})",
                 result[0], result[1]);
        return result;
    }();
    return sign;
}

/// The radical inverse of `index` in `base` (the Halton sequence), in [0, 1).
[[nodiscard]] f32 RadicalInverse(u32 index, u32 base) {
    f32 fraction = 1.0f;
    f32 result = 0.0f;
    for (u32 i = index; i > 0; i /= base) {
        fraction /= static_cast<f32>(base);
        result += fraction * static_cast<f32>(i % base);
    }
    return result;
}

/// The jitter sequence's length: about 8 samples per output pixel, 8 to 64 presents.
[[nodiscard]] u32 Phases(VkExtent2D render, VkExtent2D output) {
    if (render.width == 0 || render.height == 0) {
        return 8;
    }
    const f64 area = static_cast<f64>(output.width) * output.height /
                     (static_cast<f64>(render.width) * render.height);
    return std::clamp(static_cast<u32>(8.0 * area + 0.5), 8u, 64u);
}

/// previous * inverse(current), both row major, in double; false when an input is not finite,
/// `current` is singular or the result does not fit floats.
[[nodiscard]] bool Reprojection(const std::array<f32, 16>& current,
                                const std::array<f32, 16>& previous, std::array<f32, 16>& out) {
    // [current | identity] reduced to [identity | inverse] (partial pivoting).
    std::array<std::array<f64, 8>, 4> m{};
    for (u32 r = 0; r < 4; ++r) {
        for (u32 c = 0; c < 4; ++c) {
            if (!std::isfinite(current[r * 4 + c]) || !std::isfinite(previous[r * 4 + c])) {
                return false;
            }
            m[r][c] = current[r * 4 + c];
            m[r][c + 4] = r == c ? 1.0 : 0.0;
        }
    }
    for (u32 c = 0; c < 4; ++c) {
        u32 pivot = c;
        for (u32 r = c + 1; r < 4; ++r) {
            if (std::abs(m[r][c]) > std::abs(m[pivot][c])) {
                pivot = r;
            }
        }
        if (std::abs(m[pivot][c]) < 1e-12) {
            return false;
        }
        std::swap(m[pivot], m[c]);
        const f64 scale = 1.0 / m[c][c];
        for (f64& value : m[c]) {
            value *= scale;
        }
        for (u32 r = 0; r < 4; ++r) {
            if (r == c || m[r][c] == 0.0) {
                continue;
            }
            const f64 factor = m[r][c];
            for (u32 k = 0; k < 8; ++k) {
                m[r][k] -= factor * m[c][k];
            }
        }
    }
    for (u32 r = 0; r < 4; ++r) {
        for (u32 c = 0; c < 4; ++c) {
            f64 sum = 0.0;
            for (u32 k = 0; k < 4; ++k) {
                sum += static_cast<f64>(previous[r * 4 + k]) * m[k][c + 4];
            }
            if (!std::isfinite(sum) || std::abs(sum) > std::numeric_limits<f32>::max()) {
                return false;
            }
            out[r * 4 + c] = static_cast<f32>(sum);
        }
    }
    return true;
}

/// The depth a view-space z (< 0, in front) is stored as, for projection p (row major).
[[nodiscard]] f64 StoredDepth(const std::array<f32, 16>& p, u32 depth_mode, f32 depth_min,
                              f32 depth_max, f64 view_z) {
    const f64 w = -view_z; // the w row is (0, 0, -1, 0)
    const f64 ndc = (p[10] * view_z + p[11]) / w;
    const f64 t = depth_mode == 0 ? (ndc + 1.0) * 0.5 : ndc;
    return depth_min + (static_cast<f64>(depth_max) - depth_min) * t;
}

/// The push constant block of nvn_dlss_motion.comp.
struct MotionConstants {
    std::array<f32, 16> reproject; ///< column major (GLSL mat4)
    std::array<f32, 2> jitter;
    std::array<f32, 2> size;
    std::array<s32, 2> origin;
    f32 depth_min;
    f32 depth_scale;
    u32 flags;
};
static_assert(sizeof(MotionConstants) == 100);

constexpr VkFormat MOTION_FORMAT = VK_FORMAT_R16G16_SFLOAT;
/// (transfer source: EDEN_NVN_DLSS_DEBUG=motion blits it into the network's output)
constexpr VkImageUsageFlags MOTION_USAGE =
    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
constexpr VkImageSubresourceRange COLOR_RANGE{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
/// Camera blocks read per scene frame at most (a game whose scene draws have none stops early).
constexpr u32 CAMERA_ATTEMPTS = 32;
/// Scene frames in a row without an accepted camera that hold the last one before a reset.
constexpr u32 MAX_HELD_FRAMES = 2;
/// Evaluations between counter lines.
constexpr u64 STATS_FRAMES = 600;

/// The process's SinkTemporal per NVN context.
struct Registry {
    std::mutex mutex;
    std::vector<std::pair<const void*, std::weak_ptr<SinkTemporal>>> instances;
};

[[nodiscard]] Registry& Instances() {
    static Registry registry;
    return registry;
}

} // namespace

/// The motion pass's Vulkan objects and its current output image.
struct SinkTemporal::MotionPass {
    struct Image {
        vk::Image image;
        vk::ImageView view;
        VkExtent2D extent{};
    };
    vk::DescriptorSetLayout set_layout;
    vk::PipelineLayout pipeline_layout;
    vk::ShaderModule module;
    vk::Pipeline pipeline;
    vk::Sampler sampler;
    /// One output image per evaluating queue (`owner`): two queues' flights may run at once.
    std::vector<std::pair<const void*, std::shared_ptr<Image>>> images;
};

bool SinkTemporal::DebugMotionView() {
    static const bool enabled = [] {
        const char* const value = EnvValue("EDEN_NVN_DLSS_DEBUG");
        const bool on = value != nullptr && std::string_view{value} == "motion";
        if (on) {
            LOG_INFO(Render_Vulkan, "[nvn] dlss temporal: EDEN_NVN_DLSS_DEBUG=motion: the "
                                    "replaced network's output shows the motion vectors");
        }
        return on;
    }();
    return enabled;
}

bool SinkTemporal::Enabled() {
    // EDEN_NVN_DLSS_TEMPORAL when set, else the NVN-HLE setting "DLSS jitter and motion vectors"
    // with the upscaler at DLSS / DLAA (Settings::NvnDlssTemporalRequested). Settings change per
    // game, so this is decided on every call and logged once per boot.
    if (!Settings::NvnDlssTemporalRequested()) {
        return false;
    }
    if (!SinkDlss::Enabled()) {
        static Common::BootCounter warned;
        if (warned.First()) {
            LOG_WARNING(Render_Vulkan, "[nvn] dlss temporal: asked for, but the sink's DLSS is off "
                                       "(EDEN_NVN_DLSS): off");
        }
        return false;
    }
    static Common::BootCounter logged;
    if (logged.First()) {
        LOG_INFO(Render_Vulkan,
                 "[nvn] dlss temporal: on (jitter {}, motion vectors {}) for replaced networks",
                 JitterEnabled() ? "on" : "off", MotionEnabled() ? "on" : "off");
        static_cast<void>(JitterSign()); // logs an override once
    }
    return true;
}

std::shared_ptr<SinkTemporal> SinkTemporal::For(const void* owner) {
    Registry& registry = Instances();
    const std::scoped_lock lock{registry.mutex};
    std::erase_if(registry.instances, [](const auto& entry) { return entry.second.expired(); });
    for (const auto& [key, weak] : registry.instances) {
        if (key == owner) {
            if (auto existing = weak.lock()) {
                return existing;
            }
        }
    }
    auto created = std::make_shared<SinkTemporal>();
    registry.instances.emplace_back(owner, created);
    return created;
}

SinkTemporal::SinkTemporal() = default;
SinkTemporal::~SinkTemporal() = default;

bool SinkTemporal::IsScenePass(u64 depth_serial, u64 present) {
    const std::scoped_lock lock{mutex};
    // The scene frame after the last evaluation: from that evaluation's present (a game may
    // record the next frame's scene before it presents; the replay then sees those draws after
    // the network in the same present) to the next present, or the one after that for a game
    // whose last two evaluations were two presents apart. A stretch of presents that never
    // reaches DLSS shows at most that many jittered pictures.
    const u64 window = std::clamp(evaluation_spacing, u64{1}, u64{2});
    return scene_known && depth_serial == scene_depth && present >= scene_present &&
           present <= scene_present + window;
}

bool SinkTemporal::IsSceneCameraDraw(u64 present, VkExtent2D extent, u64 camera_binding) {
    if (camera_binding == 0) {
        return false;
    }
    const std::scoped_lock lock{mutex};
    const u64 window = std::clamp(evaluation_spacing, u64{1}, u64{2});
    return scene_known && present >= scene_present && present <= scene_present + window &&
           jitter_epoch == epoch && camera_binding == camera_address &&
           extent.width == jitter_extent.width && extent.height == jitter_extent.height;
}

SinkTemporal::SceneDraw SinkTemporal::OnSceneDraw(u64 present, VkExtent2D extent, bool depth_write,
                                                  u64 camera_binding, bool copy_pass) {
    const std::scoped_lock lock{mutex};
    SceneDraw out;
    // Scene frames are counted by evaluations (`epoch`), not presents: the draws between two
    // evaluations are the picture the second one gets, whichever present the replay saw them in.
    if (present == evaluated_present) {
        ++window_after_network; // (diagnostic: the next frame's scene, in the network's present)
    }
    if (jitter_epoch != epoch) {
        // This scene frame's first scene draw: its jitter (indexed by evaluations, so a game
        // that presents more often than it draws its scene still walks the whole sequence).
        jitter_epoch = epoch;
        const u32 index =
            static_cast<u32>(evaluation_count % phases) + 1; // from 1: 0 is the corner
        jitter_offset = {RadicalInverse(index, 2) - 0.5f, RadicalInverse(index, 3) - 0.5f};
        jitter_extent = extent;
        jitter_draws = 0;
        skipped_draws = 0;
    }
    // The camera comes from the scene's geometry: depth-writing draws.
    if (MotionEnabled() && depth_write && !copy_pass && !(camera.valid && camera.epoch == epoch)) {
        if (camera_attempt_epoch != epoch) {
            camera_attempt_epoch = epoch;
            camera_attempts = 0;
        }
        if (camera_attempts < CAMERA_ATTEMPTS) {
            ++camera_attempts;
            out.wants_camera = true;
        }
    }
    // Jittered: depth-writing draws (the scene's geometry, its depth pre-pass), and depth-tested
    // draws that do not write depth only when they bind the accepted camera block (transparent
    // geometry, particles). A depth-tested full-screen pass reading the G-buffer (lighting, fog,
    // SSAO) neither writes depth nor binds the camera block: jittering it would shift the
    // already jittered G-buffer again through its interpolated coordinates.
    const bool camera_drawn = camera_binding != 0 && camera_binding == camera_address;
    if (!depth_write && !camera_drawn) {
        ++skipped_draws;
        return out;
    }
    // With motion vectors on, the jitter waits for a valid motion pass at the last evaluation:
    // a jitter whose history is reset every frame only shimmers.
    if (!JitterEnabled() || (MotionEnabled() && !last_motion_valid) || extent.width == 0 ||
        extent.height == 0) {
        return out;
    }
    ++jitter_draws;
    if (copy_pass) {
        ++window_copy_pass;
    }
    out.jitter = true;
    out.offset = jitter_offset;
    return out;
}

void SinkTemporal::Reject(const char* why, u32 a, u32 b, u32 c, u32 d) {
    if (reject_logged) {
        return;
    }
    reject_logged = true;
    LOG_INFO(Render_Vulkan,
             "[nvn] dlss temporal: a scene draw's uniform binding 0 is not a camera block: {} "
             "({:#x} {:#x} {:#x} {:#x}); later draws are still tried (logged once)",
             why, a, b, c, d);
}

void SinkTemporal::OfferCamera(u64 present, const std::function<std::optional<u32>(u32)>& read,
                               VkExtent2D target, u32 depth_mode, f32 depth_min, f32 depth_max,
                               bool mesh, u64 binding_address) {
    const auto as_float = [](u32 bits) {
        f32 value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    };
    // The checks first (6 words), the matrices only for a block that passes them.
    std::array<u32, 4> w_row{};
    std::array<u32, 2> size{};
    for (u32 i = 0; i < 4; ++i) {
        const std::optional<u32> word = read(0xa0 + i * 4);
        if (!word) {
            const std::scoped_lock lock{mutex};
            Reject("the binding is unbound or smaller than 0x128 bytes");
            return;
        }
        w_row[i] = *word;
    }
    for (u32 i = 0; i < 2; ++i) {
        const std::optional<u32> word = read(0x120 + i * 4);
        if (!word) {
            const std::scoped_lock lock{mutex};
            Reject("the binding is smaller than 0x128 bytes");
            return;
        }
        size[i] = *word;
    }
    if (as_float(w_row[0]) != 0.0f || as_float(w_row[1]) != 0.0f || as_float(w_row[2]) != -1.0f ||
        as_float(w_row[3]) != 0.0f) {
        const std::scoped_lock lock{mutex};
        Reject("the projection's w row at +0xa0 is not (0, 0, -1, 0)", w_row[0], w_row[1], w_row[2],
               w_row[3]);
        return;
    }
    if (as_float(size[0]) != static_cast<f32>(target.width) ||
        as_float(size[1]) != static_cast<f32>(target.height)) {
        const std::scoped_lock lock{mutex};
        Reject("the size at +0x120 is not the scene target's (target width, height follow)",
               size[0], size[1], target.width, target.height);
        return;
    }
    Camera found;
    for (u32 i = 0; i < 16; ++i) {
        const std::optional<u32> vp = read(0x30 + i * 4);
        const std::optional<u32> p = read(0x70 + i * 4);
        if (!vp || !p) {
            const std::scoped_lock lock{mutex};
            Reject("the matrices at +0x30 / +0x70 are unreadable");
            return;
        }
        found.view_projection[i] = as_float(*vp);
        found.projection[i] = as_float(*p);
    }
    const auto finite = [](const std::array<f32, 16>& m) {
        return std::ranges::all_of(m, [](f32 v) { return std::isfinite(v); });
    };
    const std::array<f32, 16>& p = found.projection;
    if (!finite(found.view_projection) || !finite(p)) {
        const std::scoped_lock lock{mutex};
        Reject("a matrix element is not finite");
        return;
    }
    if (p[0] == 0.0f || p[5] == 0.0f || p[11] == 0.0f || depth_max == depth_min) {
        const std::scoped_lock lock{mutex};
        Reject("the projection or the depth range is degenerate (P00, P11, P23 follow)",
               std::bit_cast<u32>(p[0]), std::bit_cast<u32>(p[5]), std::bit_cast<u32>(p[11]));
        return;
    }
    found.depth_mode = depth_mode;
    found.depth_min = depth_min;
    found.depth_max = depth_max;
    found.present = present;
    // (the epoch is taken under the lock below)
    found.valid = true;
    const bool inverted = StoredDepth(p, depth_mode, depth_min, depth_max, -1.0) >
                          StoredDepth(p, depth_mode, depth_min, depth_max, -1000.0);

    const std::scoped_lock lock{mutex};
    if (camera.valid && camera.epoch == epoch) {
        return; // the first accepted block of a scene frame wins
    }
    found.epoch = epoch;
    camera = found;
    camera_address = binding_address;
    depth_inverted = inverted;
    depth_inverted_known = true;
    if (!camera_logged) {
        camera_logged = true;
        // An OpenGL-style projection: P22 = -(f + n) / (f - n), P23 = -2fn / (f - n).
        const f32 near_z = depth_mode == 0 ? p[11] / (p[10] - 1.0f) : p[11] / p[10];
        const f32 far_z = depth_mode == 0 ? p[11] / (p[10] + 1.0f) : p[11] / (p[10] + 1.0f);
        const f32 fov = 2.0f * std::atan(1.0f / p[5]) * 57.29578f;
        LOG_INFO(Render_Vulkan,
                 "[nvn] dlss temporal: camera block accepted ({} uniform binding 0: "
                 "view-projection +0x30, projection +0x70, w row +0xa0, target size +0x120 = "
                 "{}x{}); near {}, far {}, vfov {:.1f} deg, clip depth {}, depth range {}..{}, "
                 "depth {}",
                 mesh ? "mesh" : "vertex", target.width, target.height, near_z, far_z, fov,
                 depth_mode == 0 ? "-1..1" : "0..1", depth_min, depth_max,
                 inverted ? "inverted" : "not inverted");
    }
}

bool SinkTemporal::EnsureMotionPass(const Device& device) {
    if (motion) {
        return true;
    }
    if (motion_failed) {
        return false;
    }
    if (!device.IsKhrPushDescriptorSupported() || device.MaxPushDescriptors() < 2 ||
        !device.IsStorageImageExtendedFormatsSupported()) {
        motion_failed = true;
        LOG_WARNING(Render_Vulkan,
                    "[nvn] dlss temporal: no push descriptors or no RG16F storage images "
                    "(shaderStorageImageExtendedFormats) on this device: no motion vectors (no "
                    "jitter, DLSS history resets every frame)");
        return false;
    }
    try {
        auto pass = std::make_shared<MotionPass>();
        const vk::Device& logical = device.GetLogical();
        const std::array<VkDescriptorSetLayoutBinding, 2> bindings{{
            {
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr,
            },
            {
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr,
            },
        }};
        pass->set_layout = logical.CreateDescriptorSetLayout({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        });
        const VkPushConstantRange push_range{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(MotionConstants),
        };
        pass->pipeline_layout = logical.CreatePipelineLayout({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = 1,
            .pSetLayouts = pass->set_layout.address(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_range,
        });
        const std::span<const u32> code = NVN_DLSS_MOTION_COMP_SPV;
        pass->module = logical.CreateShaderModule({
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = static_cast<u32>(code.size_bytes()),
            .pCode = code.data(),
        });
        pass->pipeline = logical.CreateComputePipeline(
            VkComputePipelineCreateInfo{
                .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                    .module = *pass->module,
                    .pName = "main",
                    .pSpecializationInfo = nullptr,
                },
                .layout = *pass->pipeline_layout,
                .basePipelineHandle = {},
                .basePipelineIndex = 0,
            },
            // (the shared pipeline cache is externally synchronized: this small one skips it)
            VK_NULL_HANDLE);
        pass->sampler = logical.CreateSampler({
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .magFilter = VK_FILTER_NEAREST,
            .minFilter = VK_FILTER_NEAREST,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .mipLodBias = 0.0f,
            .anisotropyEnable = VK_FALSE,
            .maxAnisotropy = 1.0f,
            .compareEnable = VK_FALSE,
            .compareOp = VK_COMPARE_OP_NEVER,
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
            .unnormalizedCoordinates = VK_FALSE,
        });
        motion = std::move(pass);
    } catch (const vk::Exception& exception) {
        motion_failed = true;
        LOG_ERROR(Render_Vulkan,
                  "[nvn] dlss temporal: the motion pass could not be made ({}): no motion "
                  "vectors (DLSS history resets every frame)",
                  exception.what());
        return false;
    }
    return true;
}

std::shared_ptr<void> SinkTemporal::MotionImage(const Device& device, MemoryAllocator& allocator,
                                                const void* owner, VkExtent2D extent,
                                                SinkDlss::Image& out) {
    auto entry = std::ranges::find_if(motion->images,
                                      [owner](const auto& image) { return image.first == owner; });
    if (entry == motion->images.end()) {
        motion->images.emplace_back(owner, nullptr);
        entry = std::prev(motion->images.end());
    }
    std::shared_ptr<MotionPass::Image>& slot = entry->second;
    if (!slot || slot->extent.width != extent.width || slot->extent.height != extent.height) {
        // The previous one (another size) lives on in the flights that read it.
        auto made = std::make_shared<MotionPass::Image>();
        made->extent = extent;
        made->image = allocator.CreateImage(VkImageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = MOTION_FORMAT,
            .extent = {extent.width, extent.height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = MOTION_USAGE,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        });
        made->view = device.GetLogical().CreateImageView(VkImageViewCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = *made->image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = MOTION_FORMAT,
            .components{},
            .subresourceRange = COLOR_RANGE,
        });
        slot = std::move(made);
    }
    out = SinkDlss::Image{
        .image = *slot->image,
        .view = *slot->view,
        .format = MOTION_FORMAT,
        .usage = MOTION_USAGE,
        .size = extent,
        .x = 0,
        .y = 0,
        .width = extent.width,
        .height = extent.height,
    };
    return slot;
}

SinkTemporal::Frame SinkTemporal::BeginEvaluation(const Device& device, MemoryAllocator& allocator,
                                                  vk::CommandBuffer cmd, u64 present,
                                                  const SinkDlss::Image& depth, VkExtent2D render,
                                                  VkExtent2D output, VkExtent2D color_image,
                                                  const void* owner,
                                                  std::vector<std::shared_ptr<void>>& keepalive) {
    const std::scoped_lock lock{mutex};
    Frame frame;
    evaluated_present = present;
    phases = Phases(render, output);
    // The jitter the scene draws of this present used, in render pixels: the drawn images are
    // the one DLSS reads (a rectangle of it), or a twin of another size it was not given.
    std::array<f32, 2> content{};
    if (jitter_epoch == epoch && jitter_draws > 0 && jitter_extent.width != 0 &&
        jitter_extent.height != 0 && color_image.width != 0 && color_image.height != 0) {
        content = {jitter_offset[0] * static_cast<f32>(color_image.width) / jitter_extent.width,
                   jitter_offset[1] * static_cast<f32>(color_image.height) / jitter_extent.height};
        const std::array<f32, 2> sign = JitterSign();
        frame.jitter = {content[0] * sign[0], content[1] * sign[1]};
        window_jittered += jitter_draws;
    }
    bool camera_now = camera.valid && camera.epoch == epoch;
    if (jitter_epoch != epoch) {
        ++window_no_scene; // no scene draw since the last evaluation
    } else if (MotionEnabled() && !camera_now && camera_attempt_epoch == epoch &&
               camera_attempts >= CAMERA_ATTEMPTS) {
        ++window_cap_hit;
    }
    if (camera_now && camera.present != present) {
        ++window_earlier_present; // the scene was drawn in an earlier present than its network
    }
    frame.depth_inverted_known = depth_inverted_known;
    frame.depth_inverted = depth_inverted;
    if (!MotionEnabled()) {
        // Zero motion and the history running: the jitter alone (a still camera's eye test).
        return frame;
    }
    if (camera_now) {
        ++window_camera;
        held_frames = 0;
    } else if (previous.valid && held_frames < MAX_HELD_FRAMES) {
        // One scene frame (or two) without an accepted camera: hold the last one (zero camera
        // motion for that frame) instead of resetting the history and, through the missing
        // previous camera, the next frame's as well.
        camera = previous;
        camera.epoch = epoch;
        camera_now = true;
        ++held_frames;
        ++window_held;
    }
    std::array<f32, 16> reproject{};
    const bool pair = camera_now && previous.valid && depth.Valid() &&
                      Reprojection(camera.view_projection, previous.view_projection, reproject);
    if (pair && EnsureMotionPass(device)) {
        try {
            SinkDlss::Image image;
            // The scene viewport is assumed to cover exactly the DLSS input rectangle (render):
            // the shader maps NDC to that rectangle's pixels.
            std::shared_ptr<void> owned = MotionImage(device, allocator, owner, render, image);
            keepalive.push_back(std::move(owned));
            keepalive.push_back(motion); // its pipeline, layout and sampler outlive this flight
            MotionConstants constants{};
            for (u32 r = 0; r < 4; ++r) {
                for (u32 c = 0; c < 4; ++c) {
                    constants.reproject[c * 4 + r] = reproject[r * 4 + c];
                }
            }
            constants.jitter = content;
            constants.size = {static_cast<f32>(render.width), static_cast<f32>(render.height)};
            constants.origin = {static_cast<s32>(depth.x), static_cast<s32>(depth.y)};
            constants.depth_min = camera.depth_min;
            constants.depth_scale = 1.0f / (camera.depth_max - camera.depth_min);
            constants.flags = (camera.depth_mode == 0 ? 1u : 0u) | (DebugMotionView() ? 2u : 0u);
            // (the previous evaluation's DLSS read it: an execution dependency suffices, the
            // contents are rewritten whole)
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                VkImageMemoryBarrier{
                                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                    .pNext = nullptr,
                                    .srcAccessMask = 0,
                                    .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .image = image.image,
                                    .subresourceRange = COLOR_RANGE,
                                });
            const VkDescriptorImageInfo depth_info{
                .sampler = *motion->sampler,
                .imageView = depth.view,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            };
            const VkDescriptorImageInfo motion_info{
                .sampler = VK_NULL_HANDLE,
                .imageView = image.view,
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            };
            const std::array<VkWriteDescriptorSet, 2> writes{{
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = VK_NULL_HANDLE,
                    .dstBinding = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &depth_info,
                    .pBufferInfo = nullptr,
                    .pTexelBufferView = nullptr,
                },
                {
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = VK_NULL_HANDLE,
                    .dstBinding = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .pImageInfo = &motion_info,
                    .pBufferInfo = nullptr,
                    .pTexelBufferView = nullptr,
                },
            }};
            cmd.BindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, *motion->pipeline);
            cmd.PushDescriptorSetKHR(VK_PIPELINE_BIND_POINT_COMPUTE, *motion->pipeline_layout, 0,
                                     writes);
            cmd.PushConstants(*motion->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, constants);
            cmd.Dispatch((render.width + 7) / 8, (render.height + 7) / 8, 1);
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                                VkImageMemoryBarrier{
                                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                    .pNext = nullptr,
                                    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                                    .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .image = image.image,
                                    .subresourceRange = COLOR_RANGE,
                                });
            frame.motion = image;
            frame.motion_scale = {1.0f / static_cast<f32>(render.width),
                                  1.0f / static_cast<f32>(render.height)};
            ++motion_passes;
        } catch (const vk::Exception& exception) {
            // Only the image is dropped: earlier flights keep the pass's objects alive.
            motion_failed = true;
            LOG_ERROR(Render_Vulkan,
                      "[nvn] dlss temporal: the motion image could not be made ({}): no motion "
                      "vectors",
                      exception.what());
        }
    }
    // No valid pair (no camera this present or the one before, a singular matrix, no scene
    // depth, no motion pass): the history has no mapping to this camera.
    frame.reset = !frame.motion.Valid();
    previous = camera_now ? camera : Camera{};
    last_motion_valid = frame.motion.Valid();
    return frame;
}

void SinkTemporal::EndEvaluation(u64 present, bool evaluated, u64 depth_serial, bool reset) {
    const std::scoped_lock lock{mutex};
    const u32 frame_skipped = jitter_epoch == epoch ? skipped_draws : 0;
    ++epoch; // the next scene frame starts
    if (!evaluated) {
        // DLSS did not take this picture: no jitter until it does again, and no camera pair
        // across it (its history never saw this camera).
        scene_known = false;
        previous = Camera{};
        last_motion_valid = false;
        held_frames = 0;
        evaluated_present = present;
        ++window_not_evaluated;
        return;
    }
    if (last_evaluation_present != ~u64{0} && present > last_evaluation_present) {
        evaluation_spacing = present - last_evaluation_present;
    }
    last_evaluation_present = present;
    ++evaluation_count;
    if (depth_serial != 0) {
        scene_depth = depth_serial;
        scene_present = present;
        scene_known = true;
    } else {
        scene_known = false;
    }
    window_skipped += frame_skipped;
    ++frames;
    ++window_frames;
    if (reset) {
        ++resets;
    }
    if (window_frames >= STATS_FRAMES) {
        LOG_INFO(Render_Vulkan,
                 "[nvn] dlss temporal: {} frames, camera valid {:.0f}%, jittered draws/frame "
                 "{:.1f} (of them in passes of a copy of the scene depth {:.1f}), mvec passes {}, "
                 "resets {} (depth-tested scene draws not jittered/frame "
                 "{:.1f}, jitter phases {}; last {}: camera held {}, no scene draws {}, camera "
                 "cap hit {}, scene drawn in an earlier present {}, scene draws in the network's "
                 "present after it/frame {:.1f}, not evaluated {})",
                 frames, 100.0 * static_cast<f64>(window_camera) / window_frames,
                 static_cast<f64>(window_jittered) / window_frames,
                 static_cast<f64>(window_copy_pass) / window_frames, motion_passes, resets,
                 static_cast<f64>(window_skipped) / window_frames, phases, window_frames,
                 window_held, window_no_scene, window_cap_hit, window_earlier_present,
                 static_cast<f64>(window_after_network) / window_frames, window_not_evaluated);
        window_frames = 0;
        window_camera = 0;
        window_jittered = 0;
        window_skipped = 0;
        window_held = 0;
        window_no_scene = 0;
        window_cap_hit = 0;
        window_earlier_present = 0;
        window_after_network = 0;
        window_not_evaluated = 0;
        window_copy_pass = 0;
    }
}

} // namespace Vulkan::Nvn
