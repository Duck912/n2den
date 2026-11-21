// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <stdexcept>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/renderer_d3d12/d3d12_device.h"
#include "video_core/renderer_d3d12/d3d12_swapchain.h"

namespace D3D12 {

SwapChain::SwapChain(Device& device_, HWND hwnd_, u32 width_, u32 height_)
    : device{device_}, hwnd{hwnd_}, width{std::max(width_, 1u)}, height{std::max(height_, 1u)} {
    flags = device.TearingSupported() ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
    const DXGI_SWAP_CHAIN_DESC1 desc{
        .Width = width,
        .Height = height,
        .Format = FORMAT,
        .Stereo = FALSE,
        .SampleDesc = {.Count = 1, .Quality = 0},
        .BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT,
        .BufferCount = BUFFER_COUNT,
        .Scaling = DXGI_SCALING_STRETCH,
        .SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD,
        .AlphaMode = DXGI_ALPHA_MODE_IGNORE,
        .Flags = flags,
    };
    ComPtr<IDXGISwapChain1> swap_chain1;
    if (HRESULT result = device.Factory()->CreateSwapChainForHwnd(
            device.DirectQueue(), hwnd, &desc, nullptr, nullptr, &swap_chain1);
        FAILED(result)) {
        throw std::runtime_error{
            fmt::format("CreateSwapChainForHwnd failed: {}", ResultText(result))};
    }
    if (HRESULT result = swap_chain1.As(&swap_chain); FAILED(result)) {
        throw std::runtime_error{
            fmt::format("IDXGISwapChain3 unavailable: {}", ResultText(result))};
    }
    // The frontend owns fullscreen; DXGI must not react to Alt+Enter on the window.
    device.Factory()->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);

    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{
        .Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
        .NumDescriptors = BUFFER_COUNT,
        .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        .NodeMask = 0,
    };
    if (HRESULT result = device.Get()->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap));
        FAILED(result)) {
        throw std::runtime_error{
            fmt::format("CreateDescriptorHeap (RTV) failed: {}", ResultText(result))};
    }
    rtv_stride = device.Get()->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    if (!AcquireBuffers()) {
        throw std::runtime_error{"swap chain buffers unavailable"};
    }
    LOG_INFO(Render, "[nvn] d3d12: flip-discard swap chain {}x{}, {} buffers, tearing {}", width,
             height, BUFFER_COUNT, flags != 0 ? "on" : "off");
}

SwapChain::~SwapChain() = default;

bool SwapChain::AcquireBuffers() {
    for (u32 index = 0; index < BUFFER_COUNT; ++index) {
        if (!device.Check(swap_chain->GetBuffer(index, IID_PPV_ARGS(&buffers[index])),
                          "IDXGISwapChain::GetBuffer")) {
            return false;
        }
        device.Get()->CreateRenderTargetView(buffers[index].Get(), nullptr, Rtv(index));
    }
    return true;
}

bool SwapChain::Resize(u32 new_width, u32 new_height) {
    if (new_width == 0 || new_height == 0 || (new_width == width && new_height == height)) {
        return true;
    }
    for (auto& buffer : buffers) {
        buffer.Reset();
    }
    if (!device.Check(swap_chain->ResizeBuffers(BUFFER_COUNT, new_width, new_height, FORMAT, flags),
                      "IDXGISwapChain::ResizeBuffers")) {
        return false;
    }
    width = new_width;
    height = new_height;
    LOG_INFO(Render, "[nvn] d3d12: swap chain resized to {}x{}", width, height);
    return AcquireBuffers();
}

u32 SwapChain::CurrentIndex() const {
    return swap_chain->GetCurrentBackBufferIndex();
}

D3D12_CPU_DESCRIPTOR_HANDLE SwapChain::Rtv(u32 index) const {
    D3D12_CPU_DESCRIPTOR_HANDLE handle = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    handle.ptr += static_cast<SIZE_T>(index) * rtv_stride;
    return handle;
}

HRESULT SwapChain::Present(u32 interval) {
    const u32 sync = std::min(interval, 4u);
    const UINT present_flags = sync == 0 && flags != 0 ? DXGI_PRESENT_ALLOW_TEARING : 0;
    return swap_chain->Present(sync, present_flags);
}

} // namespace D3D12
