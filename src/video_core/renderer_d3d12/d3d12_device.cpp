// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <fmt/format.h>

#include "common/logging.h"
#include "common/string_util.h"
#include "video_core/renderer_d3d12/d3d12_device.h"

namespace D3D12 {

namespace {

bool DebugLayerRequested() {
    const char* const value = std::getenv("EDEN_D3D12_DEBUG");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

const char* FeatureLevelText(D3D_FEATURE_LEVEL level) {
    switch (level) {
    case D3D_FEATURE_LEVEL_11_0:
        return "11_0";
    case D3D_FEATURE_LEVEL_11_1:
        return "11_1";
    case D3D_FEATURE_LEVEL_12_0:
        return "12_0";
    case D3D_FEATURE_LEVEL_12_1:
        return "12_1";
    case D3D_FEATURE_LEVEL_12_2:
        return "12_2";
    default:
        return "other";
    }
}

const char* RemovalReasonText(HRESULT reason) {
    switch (reason) {
    case DXGI_ERROR_DEVICE_HUNG:
        return "DXGI_ERROR_DEVICE_HUNG";
    case DXGI_ERROR_DEVICE_REMOVED:
        return "DXGI_ERROR_DEVICE_REMOVED";
    case DXGI_ERROR_DEVICE_RESET:
        return "DXGI_ERROR_DEVICE_RESET";
    case DXGI_ERROR_DRIVER_INTERNAL_ERROR:
        return "DXGI_ERROR_DRIVER_INTERNAL_ERROR";
    case DXGI_ERROR_INVALID_CALL:
        return "DXGI_ERROR_INVALID_CALL";
    case S_OK:
        return "S_OK";
    default:
        return "unknown";
    }
}

/// d3d12.dll and dxgi.dll are loaded at run time, only when this renderer is chosen: linking their
/// import libraries into the static video_core would make every process start load them.
struct EntryPoints {
    PFN_D3D12_CREATE_DEVICE create_device{};
    PFN_D3D12_GET_DEBUG_INTERFACE get_debug_interface{};
    HRESULT(WINAPI* create_factory2)(UINT, REFIID, void**){};
};

template <typename T>
void Resolve(HMODULE module, const char* name, T& out) {
    out = module != nullptr ? reinterpret_cast<T>(GetProcAddress(module, name)) : nullptr;
}

/// Loaded once and never unloaded (objects of the libraries may outlive any one renderer).
const EntryPoints& LoadEntryPoints() {
    static const EntryPoints entry_points = [] {
        EntryPoints result;
        const HMODULE d3d12 = LoadLibraryExW(L"d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        const HMODULE dxgi = LoadLibraryExW(L"dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        Resolve(d3d12, "D3D12CreateDevice", result.create_device);
        Resolve(d3d12, "D3D12GetDebugInterface", result.get_debug_interface);
        Resolve(dxgi, "CreateDXGIFactory2", result.create_factory2);
        return result;
    }();
    return entry_points;
}

} // Anonymous namespace

std::string ResultText(HRESULT result) {
    return fmt::format("{:#010x}", static_cast<u32>(result));
}

Device::Device() {
    const EntryPoints& api = LoadEntryPoints();
    if (api.create_device == nullptr || api.create_factory2 == nullptr) {
        throw std::runtime_error{"d3d12.dll or dxgi.dll (CreateDXGIFactory2) is unavailable"};
    }
    debug_layer = DebugLayerRequested();
    if (debug_layer) {
        ComPtr<ID3D12Debug> debug;
        if (api.get_debug_interface != nullptr &&
            SUCCEEDED(api.get_debug_interface(IID_PPV_ARGS(&debug)))) {
            debug->EnableDebugLayer();
            LOG_INFO(Render, "[nvn] d3d12: debug layer enabled (EDEN_D3D12_DEBUG=1)");
        } else {
            debug_layer = false;
            LOG_WARNING(Render, "[nvn] d3d12: EDEN_D3D12_DEBUG=1 but the debug layer is not "
                                "installed (Graphics Tools optional feature)");
        }
    }

    const UINT factory_flags = debug_layer ? DXGI_CREATE_FACTORY_DEBUG : 0;
    if (HRESULT result = api.create_factory2(factory_flags, IID_PPV_ARGS(&factory));
        FAILED(result)) {
        throw std::runtime_error{fmt::format("CreateDXGIFactory2 failed: {}", ResultText(result))};
    }

    const auto try_adapter = [&api](IDXGIAdapter1* candidate) {
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(candidate->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
            return false;
        }
        return SUCCEEDED(api.create_device(candidate, D3D_FEATURE_LEVEL_12_0,
                                           __uuidof(ID3D12Device), nullptr));
    };
    ComPtr<IDXGIFactory6> factory6;
    if (SUCCEEDED(factory.As(&factory6))) {
        for (UINT index = 0;; ++index) {
            ComPtr<IDXGIAdapter1> candidate;
            if (factory6->EnumAdapterByGpuPreference(index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                     IID_PPV_ARGS(&candidate)) ==
                DXGI_ERROR_NOT_FOUND) {
                break;
            }
            if (candidate && try_adapter(candidate.Get())) {
                adapter = candidate;
                break;
            }
        }
    }
    if (!adapter) {
        for (UINT index = 0;; ++index) {
            ComPtr<IDXGIAdapter1> candidate;
            if (factory->EnumAdapters1(index, &candidate) == DXGI_ERROR_NOT_FOUND) {
                break;
            }
            if (try_adapter(candidate.Get())) {
                adapter = candidate;
                break;
            }
        }
    }
    if (!adapter) {
        throw std::runtime_error{"no hardware adapter with D3D12 feature level 12_0"};
    }
    DXGI_ADAPTER_DESC1 desc{};
    adapter->GetDesc1(&desc);
    adapter_name = Common::UTF16ToUTF8(std::wstring_view{desc.Description});
    vendor_id = desc.VendorId;

    if (HRESULT result =
            api.create_device(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device));
        FAILED(result)) {
        throw std::runtime_error{fmt::format("D3D12CreateDevice failed: {}", ResultText(result))};
    }

    const D3D12_COMMAND_QUEUE_DESC queue_desc{
        .Type = D3D12_COMMAND_LIST_TYPE_DIRECT,
        .Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL,
        .Flags = D3D12_COMMAND_QUEUE_FLAG_NONE,
        .NodeMask = 0,
    };
    if (HRESULT result = device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue));
        FAILED(result)) {
        throw std::runtime_error{
            fmt::format("CreateCommandQueue failed: {}", ResultText(result))};
    }
    queue->SetName(L"NVN-HLE direct queue");

    ComPtr<IDXGIFactory5> factory5;
    if (SUCCEEDED(factory.As(&factory5))) {
        BOOL allow_tearing = FALSE;
        if (SUCCEEDED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                                                    &allow_tearing, sizeof(allow_tearing)))) {
            tearing_supported = allow_tearing != FALSE;
        }
    }

    LOG_INFO(Render, "[nvn] d3d12: adapter \"{}\" (vendor {:#06x}, device {:#06x}, {} MiB VRAM)",
             adapter_name, desc.VendorId, desc.DeviceId, desc.DedicatedVideoMemory >> 20);
    LogFeatures();
}

Device::~Device() = default;

void Device::LogFeatures() const {
    static constexpr std::array levels{D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_11_1,
                                       D3D_FEATURE_LEVEL_12_0, D3D_FEATURE_LEVEL_12_1,
                                       D3D_FEATURE_LEVEL_12_2};
    D3D12_FEATURE_DATA_FEATURE_LEVELS feature_levels{
        .NumFeatureLevels = static_cast<UINT>(levels.size()),
        .pFeatureLevelsRequested = levels.data(),
        .MaxSupportedFeatureLevel = D3D_FEATURE_LEVEL_11_0,
    };
    device->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &feature_levels,
                                sizeof(feature_levels));

    // The highest shader model the runtime knows answers; lower it until the query succeeds.
    D3D12_FEATURE_DATA_SHADER_MODEL shader_model{.HighestShaderModel = D3D_HIGHEST_SHADER_MODEL};
    while (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shader_model,
                                              sizeof(shader_model))) &&
           shader_model.HighestShaderModel > D3D_SHADER_MODEL_5_1) {
        shader_model.HighestShaderModel =
            static_cast<D3D_SHADER_MODEL>(shader_model.HighestShaderModel - 1);
    }

    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options));
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 options1{};
    device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &options1, sizeof(options1));
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 options7{};
    device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &options7, sizeof(options7));
    D3D12_FEATURE_DATA_D3D12_OPTIONS12 options12{};
    const bool has_options12 = SUCCEEDED(
        device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &options12, sizeof(options12)));

    const u32 sm = static_cast<u32>(shader_model.HighestShaderModel);
    LOG_INFO(Render,
             "[nvn] d3d12 features: feature level {}, shader model {}.{}, resource binding tier "
             "{}, tiled resources tier {}, mesh shader tier {}, enhanced barriers {}, typed UAV "
             "additional formats {}, wave ops {} (lanes {}-{}), tearing {}, debug layer {}",
             FeatureLevelText(feature_levels.MaxSupportedFeatureLevel), sm >> 4, sm & 0xf,
             static_cast<int>(options.ResourceBindingTier),
             static_cast<int>(options.TiledResourcesTier),
             static_cast<int>(options7.MeshShaderTier),
             has_options12 ? (options12.EnhancedBarriersSupported ? "yes" : "no") : "unknown",
             options.TypedUAVLoadAdditionalFormats ? "yes" : "no",
             options1.WaveOps ? "yes" : "no", options1.WaveLaneCountMin,
             options1.WaveLaneCountMax, tearing_supported ? "yes" : "no",
             debug_layer ? "on" : "off");
}

bool Device::Check(HRESULT result, const char* what) {
    if (SUCCEEDED(result)) {
        return true;
    }
    const bool lost = result == DXGI_ERROR_DEVICE_REMOVED || result == DXGI_ERROR_DEVICE_RESET ||
                      result == DXGI_ERROR_DEVICE_HUNG ||
                      result == DXGI_ERROR_DRIVER_INTERNAL_ERROR;
    if (!lost) {
        LOG_ERROR(Render, "[nvn] d3d12: {} failed: {}", what, ResultText(result));
        return false;
    }
    if (!removed.exchange(true, std::memory_order_acq_rel)) {
        const HRESULT reason = device->GetDeviceRemovedReason();
        LOG_CRITICAL(Render, "[nvn] d3d12: device removed during {}: {} (reason {} {})", what,
                     ResultText(result), ResultText(reason), RemovalReasonText(reason));
    }
    return false;
}

} // namespace D3D12
