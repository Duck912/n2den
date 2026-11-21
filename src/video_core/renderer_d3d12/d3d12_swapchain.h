// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// NVN-HLE D3D12 skeleton: the flip-model swap chain on the render window's HWND.

#pragma once

#include <array>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "common/common_types.h"

namespace D3D12 {

using Microsoft::WRL::ComPtr;

class Device;

/// FLIP_DISCARD, BUFFER_COUNT back buffers, tearing when the system supports it. Used by one
/// thread at a time (the NVN replay thread once the game runs). The constructor throws
/// std::runtime_error on failure.
class SwapChain {
public:
    static constexpr u32 BUFFER_COUNT = 3;
    static constexpr DXGI_FORMAT FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;

    SwapChain(Device& device, HWND hwnd, u32 width, u32 height);
    ~SwapChain();

    SwapChain(const SwapChain&) = delete;
    SwapChain& operator=(const SwapChain&) = delete;

    /// Resizes the buffers when the size changed. The caller must have waited for every GPU use
    /// of the back buffers. A zero size is ignored (minimized window).
    bool Resize(u32 width, u32 height);

    [[nodiscard]] u32 Width() const {
        return width;
    }
    [[nodiscard]] u32 Height() const {
        return height;
    }
    [[nodiscard]] u32 CurrentIndex() const;
    [[nodiscard]] ID3D12Resource* BackBuffer(u32 index) const {
        return buffers[index].Get();
    }
    [[nodiscard]] D3D12_CPU_DESCRIPTOR_HANDLE Rtv(u32 index) const;

    /// Presents with `interval` (0..4); interval 0 tears when supported. Returns the result.
    HRESULT Present(u32 interval);

private:
    bool AcquireBuffers();

    Device& device;
    HWND hwnd;
    ComPtr<IDXGISwapChain3> swap_chain;
    ComPtr<ID3D12DescriptorHeap> rtv_heap;
    std::array<ComPtr<ID3D12Resource>, BUFFER_COUNT> buffers;
    u32 rtv_stride{};
    u32 width{};
    u32 height{};
    UINT flags{};
};

} // namespace D3D12
