// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

// NVN-HLE D3D12 skeleton: the D3D12 device and its direct queue.

#pragma once

#include <atomic>
#include <string>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "common/common_types.h"

namespace D3D12 {

using Microsoft::WRL::ComPtr;

/// DXGI 1.6 factory, the high-performance adapter, an ID3D12Device and one direct queue.
/// The constructor throws std::runtime_error when any of it cannot be created.
class Device {
public:
    Device();
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    [[nodiscard]] ID3D12Device* Get() const {
        return device.Get();
    }
    [[nodiscard]] IDXGIFactory4* Factory() const {
        return factory.Get();
    }
    [[nodiscard]] ID3D12CommandQueue* DirectQueue() const {
        return queue.Get();
    }
    [[nodiscard]] bool TearingSupported() const {
        return tearing_supported;
    }
    [[nodiscard]] const std::string& AdapterName() const {
        return adapter_name;
    }
    [[nodiscard]] u32 VendorId() const {
        return vendor_id;
    }

    /// True once the device was removed (the reason is logged once, by the first caller).
    [[nodiscard]] bool IsRemoved() const {
        return removed.load(std::memory_order_acquire);
    }

    /// Checks a result of a device or swap chain call: logs `what` on failure and, for a removed
    /// device, the removal reason (once). Returns true on success.
    bool Check(HRESULT result, const char* what);

private:
    void LogFeatures() const;

    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    std::string adapter_name;
    u32 vendor_id{};
    bool tearing_supported{};
    bool debug_layer{};
    std::atomic<bool> removed{false};
};

/// Readable form of an HRESULT for the log.
[[nodiscard]] std::string ResultText(HRESULT result);

} // namespace D3D12
