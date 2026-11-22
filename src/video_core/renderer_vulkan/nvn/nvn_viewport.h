// SPDX-FileCopyrightText: Copyright 2026 n2den Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// The window-space rules of the NVN replay: where an NVN viewport / scissor rectangle lands in the
// Vulkan framebuffer and which winding is a front face. Pure functions (tests: tests/nvn/draw.cpp).
//
// NVN clip space is OpenGL's (+y up) and the first row of a render target is its top, whatever the
// window origin mode. A Vulkan viewport therefore always has a negative height (clip +y lands on
// the first row of the rectangle) and the winding stays the NVN one. The origin only decides where
// the rectangle sits: rows count from the top for UPPER_LEFT and from the bottom for LOWER_LEFT.
// Found on DK (UPPER_LEFT): with a positive height the G-buffer is stored upside down and text
// rendered to a texture is sampled flipped (the world banner).

#pragma once

#include <algorithm>
#include <cstdlib>

#include "common/common_types.h"

namespace Vulkan::Nvn {

struct ViewportRect {
    float x;
    float y;
    float width;
    float height;
};

/// `rect` in NVN window space (y counted from the origin). `invert_y` is the experiment knob
/// EDEN_NX2_NVN_FLIP_VIEWPORT (positive height).
[[nodiscard]] constexpr ViewportRect ToVulkanViewport(ViewportRect rect, float surface_height,
                                                      bool lower_left, bool invert_y = false) {
    const float top = lower_left ? surface_height - (rect.y + rect.height) : rect.y;
    if (invert_y) {
        return {rect.x, top, rect.width, rect.height};
    }
    return {rect.x, top + rect.height, rect.width, -rect.height};
}

/// LOWER_LEFT games store render targets bottom-up, as OpenGL does: window row y is image row y,
/// clip +y points to higher rows, and textures they render are sampled the same way. The window is
/// flipped once when presented (PresentFlipsRows). Found on Kirby and the Forgotten Land (never
/// sets an origin, so NVN's default LOWER_LEFT): with the UPPER_LEFT rule every render-to-texture
/// layer (text, buttons) came out upside down, and a sub-target viewport landed in the wrong half.
/// EDEN_NX2_NVN_LOWER_LEFT_ROWS=0 restores the old rule (the image's first row is its top).
[[nodiscard]] inline bool LowerLeftStoresRowsBottomUp() {
    static const bool enabled = [] {
        const char* value = std::getenv("EDEN_NX2_NVN_LOWER_LEFT_ROWS");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

/// Whether draws use LOWER_LEFT window space: the game's device origin (0 lower left, 1 upper left)
/// unless EDEN_NX2_NVN_ORIGIN=lower|upper overrides it (experiments).
[[nodiscard]] inline bool EffectiveLowerLeft(u32 device_origin) {
    static const int origin_override = [] {
        const char* value = std::getenv("EDEN_NX2_NVN_ORIGIN");
        return value == nullptr ? -1 : (value[0] == 'u' ? 1 : 0);
    }();
    return origin_override >= 0 ? origin_override == 0 : device_origin == 0;
}

/// The window image is stored bottom-up and must be flipped once when presented.
[[nodiscard]] inline bool PresentFlipsRows(u32 device_origin) {
    return EffectiveLowerLeft(device_origin) && LowerLeftStoresRowsBottomUp();
}

/// Viewport for a LOWER_LEFT game whose rows are stored bottom-up: positive height at `rect.y`.
[[nodiscard]] constexpr ViewportRect ToVulkanViewportBottomUp(ViewportRect rect) {
    return {rect.x, rect.y, rect.width, rect.height};
}

/// First row of a scissor rectangle of `height` rows starting at `y` in NVN window space.
[[nodiscard]] constexpr s32 ScissorTopRow(s32 y, s32 height, s32 surface_height, bool lower_left) {
    return lower_left ? surface_height - (y + (std::max)(height, 0)) : y;
}

/// Vulkan `frontFace == COUNTER_CLOCKWISE` for the NVN front face (non-zero: counter-clockwise).
/// `invert` is the experiment knob EDEN_NX2_NVN_FACE=1.
[[nodiscard]] constexpr bool VulkanFrontFaceCcw(bool nvn_ccw, bool invert = false) {
    return nvn_ccw != invert;
}

} // namespace Vulkan::Nvn
