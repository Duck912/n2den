// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// What a temporal upscaler needs for a replaced upscaling network,
// without patching the game: the scene camera read from the scene draws' uniform data,
// a sub-pixel viewport jitter on the scene draws at replay, and camera motion vectors made by
// reprojecting the scene depth. Only with EDEN_NVN_DLSS_TEMPORAL=1 (and EDEN_NVN_DLSS=1); the
// sub-switches EDEN_NVN_DLSS_JITTER=0|1 and EDEN_NVN_DLSS_MVEC=0|1 (both 1 by default) let the
// jitter and the motion vectors be tested separately.
//
// The camera block layout (vertex uniform binding 0: view-projection at +0x30 and projection at
// +0x70, both row major, the projection's w row (0, 0, -1, 0) at +0xa0, the target size as floats
// at +0x120) is the one Longshot (a Hookshot fork, GPL-3.0-or-later) documents for Ocarina of
// Time; it is accepted only by its own checks, never by a title id. The method (Halton (2, 3)
// jitter from index 1, the sequence length, the reprojection) follows Longshot's description;
// the code was written for n2den.

#pragma once

#include <array>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/nvn/nvn_dlss.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
class Device;
class MemoryAllocator;
} // namespace Vulkan

namespace Vulkan::Nvn {

class SinkTemporal {
public:
    /// The temporal inputs are asked for (EDEN_NVN_DLSS_TEMPORAL=1, or the NVN-HLE setting when
    /// the variable is unset) and the sink's DLSS is on (SinkDlss::Enabled); warns once per boot
    /// when only the first holds.
    [[nodiscard]] static bool Enabled();

    /// EDEN_NVN_DLSS_DEBUG=motion: the motion pass writes 0.5 + motion / 32 (x red,
    /// y green) and the caller shows that image in place of the DLSS output.
    [[nodiscard]] static bool DebugMotionView();

    /// The process's SinkTemporal for one NVN context (`owner`: its Impl), shared by its queues.
    [[nodiscard]] static std::shared_ptr<SinkTemporal> For(const void* owner);

    // ---- the replay's scene draws ----

    /// Whether a pass with the depth attachment `depth_serial` (a guest texture serial) draws the
    /// scene of present `present`: its depth is the scene depth of the replaced network DLSS
    /// evaluated last, in this present or the one before (two before when the last evaluations
    /// were two presents apart: a game drawing its scene every second present).
    [[nodiscard]] bool IsScenePass(u64 depth_serial, u64 present);

    /// One depth-tested draw of a scene pass, into images of `extent` pixels. `depth_write`: the
    /// draw writes depth; `camera_binding`: the address of its vertex (mesh) uniform binding 0.
    /// Jittered: depth-writing draws, and draws that bind the accepted camera block.
    struct SceneDraw {
        bool jitter{};               ///< offset the draw's viewports by `offset`
        std::array<f32, 2> offset{}; ///< framebuffer pixels (x right, y down)
        bool wants_camera{};         ///< the caller offers the draw's camera block
    };
    /// `copy_pass`: the draw is not in a pass of the scene depth but IsSceneCameraDraw said yes
    /// (it is jittered as a draw that binds the camera block, and offers no camera).
    [[nodiscard]] SceneDraw OnSceneDraw(u64 present, VkExtent2D extent, bool depth_write,
                                        u64 camera_binding, bool copy_pass);

    /// Whether a depth-tested draw outside a scene pass, into images of `extent` pixels, is still
    /// one of the scene's: in the scene frame's window, after its first scene draw, at the scene
    /// draws' size, binding the accepted camera block (`camera_binding`) as its vertex (mesh)
    /// uniform binding 0. Such a draw tests against a copy of the scene depth (or draws more of
    /// the scene into another depth) and needs the scene's jitter.
    [[nodiscard]] bool IsSceneCameraDraw(u64 present, VkExtent2D extent, u64 camera_binding);

    /// A draw's camera block, read by `read` (a byte offset into the draw's vertex uniform
    /// binding 0, nullopt when unreadable). `target` is the scene target's guest size,
    /// `depth_mode` NVN's (0: clip depth -1..1, 1: 0..1), `depth_min/max` viewport 0's range,
    /// `binding_address` the binding's guest address (kept to recognise draws that bind it).
    void OfferCamera(u64 present, const std::function<std::optional<u32>(u32)>& read,
                     VkExtent2D target, u32 depth_mode, f32 depth_min, f32 depth_max, bool mesh,
                     u64 binding_address);

    // ---- the replaced network's DLSS evaluation ----

    /// What the evaluation of present `present` gets.
    struct Frame {
        std::array<f32, 2> jitter{}; ///< render pixels, EDEN_NVN_DLSS_JITTER_SIGN applied
        SinkDlss::Image motion;      ///< RG16F render-size camera motion; invalid: zero motion
        std::array<f32, 2> motion_scale{1.0f, 1.0f};
        bool reset{}; ///< a new history (no valid camera pair)
        bool depth_inverted_known{};
        bool depth_inverted{};
    };

    /// Before SinkDlss::Evaluate: the jitter the scene draws of this present used, and (motion
    /// vectors on) the motion pass recorded into `cmd` from `depth` (GENERAL, ordered after its
    /// writers; invalid: no motion, a reset). `render` is the colour rectangle DLSS reads,
    /// `color_image` the size of the image it is in (the jitter is scaled from the drawn images'
    /// size to it: the guest image when the scene was drawn into a twin DLSS does not read).
    /// `owner` (the evaluating DrawContext) has its own motion image: two queues' flights may
    /// run at the same time. The camera pair itself is shared by the context: one replaced
    /// network per frame is expected (two queues evaluating would alternate the pair, giving
    /// wrong motion, not a crash). Outside a render pass; binds a compute pipeline.
    [[nodiscard]] Frame BeginEvaluation(const Device& device, MemoryAllocator& allocator,
                                        vk::CommandBuffer cmd, u64 present,
                                        const SinkDlss::Image& depth, VkExtent2D render,
                                        VkExtent2D output, VkExtent2D color_image,
                                        const void* owner,
                                        std::vector<std::shared_ptr<void>>& keepalive);

    /// After it, and for a replaced network DLSS did not evaluate (`evaluated` false: no jitter
    /// and no camera pair until it evaluates again): `depth_serial` becomes the scene depth the
    /// next present jitters (0: none), and the frame is counted.
    void EndEvaluation(u64 present, bool evaluated, u64 depth_serial, bool reset);

    SinkTemporal();
    ~SinkTemporal();
    SinkTemporal(const SinkTemporal&) = delete;
    SinkTemporal& operator=(const SinkTemporal&) = delete;

private:
    /// One accepted camera block.
    struct Camera {
        std::array<f32, 16> view_projection{}; ///< row major (clip = M * position)
        std::array<f32, 16> projection{};      ///< row major
        u32 depth_mode{};
        f32 depth_min{};
        f32 depth_max{1.0f};
        u64 present{~u64{0}}; ///< the present its draw was replayed in (diagnostics)
        u64 epoch{~u64{0}};   ///< the scene frame (epoch) it was accepted in
        bool valid{};
    };
    struct MotionPass;

    /// The motion pass's pipeline (made on first use; false when the device cannot).
    bool EnsureMotionPass(const Device& device);
    /// `owner`'s motion image at `extent` (a new one when the size changes).
    std::shared_ptr<void> MotionImage(const Device& device, MemoryAllocator& allocator,
                                      const void* owner, VkExtent2D extent, SinkDlss::Image& out);
    void Reject(const char* why, u32 a = 0, u32 b = 0, u32 c = 0, u32 d = 0);

    std::mutex mutex;
    // The scene, learnt at the last replaced network DLSS evaluated.
    u64 scene_depth{};
    u64 scene_present{};
    bool scene_known{};
    /// The present of the last evaluation: draws of that present after it are not the scene's.
    u64 evaluated_present{~u64{0}};
    /// Whether the last evaluation got valid motion vectors (with them on the jitter waits for
    /// that: a jitter with a reset every frame shimmers).
    bool last_motion_valid{};
    /// Evaluations so far (the jitter's index), the last one's present and the spacing of the
    /// last two (the scene-pass window).
    u64 evaluation_count{};
    u64 last_evaluation_present{~u64{0}};
    u64 evaluation_spacing{1};
    /// The scene frame: incremented by every EndEvaluation (the draws after an evaluation are
    /// the next evaluation's picture, whatever present the replay sees them in).
    u64 epoch{};
    u32 held_frames{}; ///< scene frames in a row that held the last camera
    u32 phases{8};
    // This scene frame's jitter, as the scene draws used it.
    u64 jitter_epoch{~u64{0}};
    std::array<f32, 2> jitter_offset{};
    VkExtent2D jitter_extent{};
    u32 jitter_draws{};
    u32 skipped_draws{}; ///< depth-tested scene draws the rule left unjittered
    u32 camera_attempts{};
    u64 camera_attempt_epoch{~u64{0}};
    Camera camera;         ///< this scene frame's (camera.epoch)
    Camera previous;       ///< the one the last evaluation used
    u64 camera_address{};  ///< the accepted block's binding address
    bool depth_inverted{}; ///< from the last accepted camera
    bool depth_inverted_known{};
    // The motion pass.
    std::shared_ptr<MotionPass> motion; ///< also held by every flight that used it
    bool motion_failed{};
    // Logs and counters.
    bool camera_logged{};
    bool reject_logged{};
    u64 frames{};
    u64 window_frames{};
    u64 window_camera{};
    u64 window_jittered{};
    u64 window_skipped{};
    u64 window_held{};
    u64 window_no_scene{};
    u64 window_cap_hit{};
    u64 window_earlier_present{};
    u64 window_after_network{};
    u64 window_not_evaluated{};
    u64 window_copy_pass{}; ///< draws jittered by IsSceneCameraDraw
    u64 motion_passes{};
    u64 resets{};
};

} // namespace Vulkan::Nvn
