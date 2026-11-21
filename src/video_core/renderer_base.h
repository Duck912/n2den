// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <functional>
#include <memory>

#include "common/common_funcs.h"
#include "common/common_types.h"
#include "core/frontend/framebuffer_layout.h"
#include "video_core/gpu.h"
#include "video_core/rasterizer_interface.h"

namespace VideoCommon::Nvn {
class Sink;
} // namespace VideoCommon::Nvn

namespace Core::Frontend {
class EmuWindow;
class GraphicsContext;
} // namespace Core::Frontend

namespace VideoCore {

class LoadProgressSink;

struct RendererSettings {
    // Screenshot
    std::atomic<bool> screenshot_requested{false};
    void* screenshot_bits{};
    std::function<void(bool)> screenshot_complete_callback;
    Layout::FramebufferLayout screenshot_framebuffer_layout;
    Service::Nvnflinger::LayerStackId screenshot_layer_stack{Service::Nvnflinger::LayerStackId::Default};
    /// A renderer is rendering the pending request (see ClaimScreenshot).
    std::atomic<bool> screenshot_claimed{false};

    /// Called by a renderer about to render the pending screenshot request, from any thread: true
    /// for exactly one caller per request. The Vulkan Composite path and the nvn-hle replay both
    /// present frames, so both ask before they touch screenshot_bits.
    [[nodiscard]] bool ClaimScreenshot() {
        if (!screenshot_requested.load() || screenshot_claimed.exchange(true)) {
            return false;
        }
        // The request may have been finished between the two steps above.
        if (!screenshot_requested.load()) {
            screenshot_claimed.store(false);
            return false;
        }
        return true;
    }

    /// The claimant could not render the request: it stays pending for the next frame.
    void ReleaseScreenshotClaim() {
        screenshot_claimed.store(false);
    }

    /// The request is answered (the callback has been called): a new one may be made.
    void FinishScreenshot() {
        screenshot_requested.store(false);
        screenshot_claimed.store(false);
    }
};

class RendererBase {
public:
    YUZU_NON_COPYABLE(RendererBase);
    YUZU_NON_MOVEABLE(RendererBase);

    explicit RendererBase(Core::Frontend::EmuWindow& window,
                          std::unique_ptr<Core::Frontend::GraphicsContext> context);
    virtual ~RendererBase();

    /// Finalize rendering the guest frame and draw into the presentation texture
    virtual void Composite(std::span<const Tegra::FramebufferConfig> layers) = 0;

    /// Get the tiled applet layer capture buffer
    virtual std::vector<u8> GetAppletCaptureBuffer() = 0;

    [[nodiscard]] virtual RasterizerInterface* ReadRasterizer() = 0;

    [[nodiscard]] virtual std::string GetDeviceVendor() const = 0;

    /// nvn-hle (NVN high-level emulation) sink. Only the Vulkan renderer will provide one, and only
    /// while the game runs on nvn-hle (nullptr otherwise, and always until the Vulkan sink is in).
    [[nodiscard]] virtual VideoCommon::Nvn::Sink* NvnSink() {
        return nullptr;
    }

    /// Where the nvn-hle pipeline cache reports its boot work (prepare step, pipeline rebuild), which
    /// runs on its own threads after the frontend's load calls. nullptr when the renderer has none.
    [[nodiscard]] virtual LoadProgressSink* NvnLoadProgress() {
        return nullptr;
    }

    /// Whether this is NVN-HLE's Direct3D 12 renderer (the status bar names it).
    [[nodiscard]] virtual bool NvnOnD3D12() const {
        return false;
    }

    // Getter/setter functions:
    // ------------------------

    [[nodiscard]] f32 GetCurrentFPS() const {
        return m_current_fps;
    }

    [[nodiscard]] int GetCurrentFrame() const {
        return m_current_frame;
    }

    [[nodiscard]] Core::Frontend::GraphicsContext& Context() {
        return *context;
    }

    [[nodiscard]] const Core::Frontend::GraphicsContext& Context() const {
        return *context;
    }

    [[nodiscard]] Core::Frontend::EmuWindow& GetRenderWindow() {
        return render_window;
    }

    [[nodiscard]] const Core::Frontend::EmuWindow& GetRenderWindow() const {
        return render_window;
    }

    [[nodiscard]] RendererSettings& Settings() {
        return renderer_settings;
    }

    [[nodiscard]] const RendererSettings& Settings() const {
        return renderer_settings;
    }

    /// Refreshes the settings common to all renderers
    void RefreshBaseSettings();

    /// Returns true if a screenshot is being processed
    bool IsScreenshotPending() const;

    /// Request a screenshot of the next frame.
    void RequestScreenshot(void* data, std::function<void(bool)> callback,
                           const Layout::FramebufferLayout& layout,
                           Service::Nvnflinger::LayerStackId layer_stack =
                               Service::Nvnflinger::LayerStackId::Default);

protected:
    Core::Frontend::EmuWindow& render_window; ///< Reference to the render window handle.
    std::unique_ptr<Core::Frontend::GraphicsContext> context;
    f32 m_current_fps = 0.0f; ///< Current framerate, should be set by the renderer
    int m_current_frame = 0;  ///< Current frame, should be set by the renderer

    RendererSettings renderer_settings;

private:
    /// Updates the framebuffer layout of the contained render window handle.
    void UpdateCurrentFramebufferLayout();
};

} // namespace VideoCore
