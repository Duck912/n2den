// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <cstdlib>
#if defined(__linux__)
#include <execinfo.h>
#endif
#include <bit>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/common_types.h"
#include "common/literals.h"
#include "common/logging.h"
#include <ranges>
#include "video_core/vulkan_common/vma.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "common/settings.h"

namespace Vulkan {

namespace {
/// Warns when an allocation that prefers device-local memory lands elsewhere (VMA silently picks
/// another memory type once the device-local heap is over budget). Placement in system memory
/// makes every GPU access to it cross PCIe.
void CheckPlacement(VmaAllocator allocator, VmaAllocation allocation, VkDeviceSize size,
                    const char* kind) {
    VkMemoryPropertyFlags flags{};
    vmaGetAllocationMemoryProperties(allocator, allocation, &flags);
    if (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
        return;
    }
    static std::atomic<u32> count{};
    static std::atomic<u64> bytes{};
    const u32 n = ++count;
    const u64 total = bytes += size;
    if (n <= 64 || std::has_single_bit(n)) {
        VmaBudget budgets[VK_MAX_MEMORY_HEAPS]{};
        vmaGetHeapBudgets(allocator, budgets);
        LOG_WARNING(Render_Vulkan,
                    "Device-local {} placed in non-device-local memory: size={} flags={:#x} "
                    "(n={}, total={} MiB; heap0 usage={} MiB budget={} MiB)",
                    kind, size, flags, n, total >> 20, budgets[0].usage >> 20,
                    budgets[0].budget >> 20);
#if defined(__linux__)
        if (n <= 4) {
            void* frames[12];
            const int depth = backtrace(frames, 12);
            char** names = backtrace_symbols(frames, depth);
            for (int i = 1; names && i < depth; ++i) {
                LOG_WARNING(Render_Vulkan, "  placement caller {}: {}", i, names[i]);
            }
            std::free(names);
        }
#endif
    }
}
} // namespace
namespace {

// Helpers translating MemoryUsage to flags/usage

    [[maybe_unused]] VkMemoryPropertyFlags MemoryUsagePropertyFlags(MemoryUsage usage) {
        switch (usage) {
            case MemoryUsage::DeviceLocal:
                return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            case MemoryUsage::Upload:
                return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            case MemoryUsage::Download:
                return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                        VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
            case MemoryUsage::Stream:
                return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        }
        ASSERT_MSG(false, "Invalid memory usage={}", usage);
        return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }

    [[nodiscard]] VkMemoryPropertyFlags MemoryUsagePreferredVmaFlags(MemoryUsage usage) {
        if (usage == MemoryUsage::Download) {
            return VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        }
        return usage != MemoryUsage::DeviceLocal ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                                    : VkMemoryPropertyFlagBits{};
    }

    [[nodiscard]] VmaAllocationCreateFlags MemoryUsageVmaFlags(MemoryUsage usage) {
        switch (usage) {
            case MemoryUsage::Upload:
            case MemoryUsage::Stream:
                return VMA_ALLOCATION_CREATE_MAPPED_BIT |
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            case MemoryUsage::Download:
                return VMA_ALLOCATION_CREATE_MAPPED_BIT |
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
            case MemoryUsage::DeviceLocal:
                return {};
        }
        return {};
    }

    [[nodiscard]] VmaMemoryUsage MemoryUsageVma(MemoryUsage usage) {
        switch (usage) {
            case MemoryUsage::DeviceLocal:
            case MemoryUsage::Stream:
                return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            case MemoryUsage::Upload:
            case MemoryUsage::Download:
                return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        }
        return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    }

    /// Strict mode's over-budget allocations: the first ones and then every 256th are logged.
    [[nodiscard]] bool LogStrictFallback(std::atomic<u64>& counter) {
        const u64 n = counter.load(std::memory_order_relaxed);
        return n <= 32 || (n & 0xFF) == 0;
    }
} // namespace

const char* AllocationKindName(AllocationKind kind) {
    switch (kind) {
    case AllocationKind::Other:
        return "other";
    case AllocationKind::StorageImage:
        return "storage_images";
    case AllocationKind::PoolBuffer:
        return "pool_buffers";
    case AllocationKind::Shadow:
        return "shadows";
    case AllocationKind::Mirror:
        return "mirrors";
    case AllocationKind::Arena:
        return "arenas";
    case AllocationKind::Staging:
        return "staging";
    case AllocationKind::Count:
        break;
    }
    return "?";
}

bool CountSpill(AllocationSpills& spills, AllocationKind kind, bool wanted_device_local,
                VkMemoryPropertyFlags flags, u64 size) {
    const size_t index = static_cast<size_t>(kind);
    if (!wanted_device_local || (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0 ||
        index >= AllocationSpills::KINDS) {
        return false;
    }
    spills.count[index].fetch_add(1, std::memory_order_relaxed);
    spills.bytes[index].fetch_add(size, std::memory_order_relaxed);
    return true;
}

//MemoryCommit is now VMA-backed
MemoryCommit::MemoryCommit(VmaAllocator alloc, VmaAllocation a,
                            const VmaAllocationInfo &info) noexcept
        : allocator{alloc}, allocation{a}, memory{info.deviceMemory},
            offset{info.offset}, size{info.size}, mapped_ptr{info.pMappedData} {
    // Log GPU memory allocation
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_memory_tracking.GetValue()) {
        GPU::Logging::GPULogger::GetInstance().LogMemoryAllocation(
            reinterpret_cast<uintptr_t>(memory),
            static_cast<u64>(size),
            0  // Memory property flags (not easily available from VMA)
        );
    }
}

MemoryCommit::~MemoryCommit() { Release(); }

MemoryCommit::MemoryCommit(MemoryCommit &&rhs) noexcept
        : allocator{std::exchange(rhs.allocator, nullptr)},
            allocation{std::exchange(rhs.allocation, nullptr)},
            memory{std::exchange(rhs.memory, VK_NULL_HANDLE)},
            offset{std::exchange(rhs.offset, 0)},
            size{std::exchange(rhs.size, 0)},
            mapped_ptr{std::exchange(rhs.mapped_ptr, nullptr)} {}

MemoryCommit &MemoryCommit::operator=(MemoryCommit &&rhs) noexcept {
    if (this != &rhs) {
        Release();
        allocator = std::exchange(rhs.allocator, nullptr);
        allocation = std::exchange(rhs.allocation, nullptr);
        memory = std::exchange(rhs.memory, VK_NULL_HANDLE);
        offset = std::exchange(rhs.offset, 0);
        size = std::exchange(rhs.size, 0);
        mapped_ptr = std::exchange(rhs.mapped_ptr, nullptr);
    }
    return *this;
}

std::span<u8> MemoryCommit::Map()
{
    if (!allocation) return {};
    if (!mapped_ptr) {
        if (vmaMapMemory(allocator, allocation, &mapped_ptr) != VK_SUCCESS) return {};
    }
    const size_t n = static_cast<size_t>(std::min<VkDeviceSize>(size,
                                                                (std::numeric_limits<size_t>::max)()));
    return std::span<u8>{static_cast<u8 *>(mapped_ptr), n};
}

std::span<const u8> MemoryCommit::Map() const
{
    if (!allocation) return {};
    if (!mapped_ptr) {
        void *p = nullptr;
        if (vmaMapMemory(allocator, allocation, &p) != VK_SUCCESS) return {};
        const_cast<MemoryCommit *>(this)->mapped_ptr = p;
    }
    const size_t n = static_cast<size_t>(std::min<VkDeviceSize>(size,
                                                                (std::numeric_limits<size_t>::max)()));
    return std::span<const u8>{static_cast<const u8 *>(mapped_ptr), n};
}

void MemoryCommit::Unmap()
{
    if (allocation && mapped_ptr) {
        vmaUnmapMemory(allocator, allocation);
        mapped_ptr = nullptr;
    }
}

void MemoryCommit::Release() {
    if (allocation && allocator) {
        // Log GPU memory deallocation
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_memory_tracking.GetValue() &&
            memory != VK_NULL_HANDLE) {
            GPU::Logging::GPULogger::GetInstance().LogMemoryDeallocation(
                reinterpret_cast<uintptr_t>(memory)
            );
        }

        if (mapped_ptr) {
            vmaUnmapMemory(allocator, allocation);
            mapped_ptr = nullptr;
        }
        vmaFreeMemory(allocator, allocation);
    }
    allocation = nullptr;
    allocator = nullptr;
    memory = VK_NULL_HANDLE;
    offset = 0;
    size = 0;
}

MemoryAllocator::MemoryAllocator(const Device &device_)
        : MemoryAllocator{device_, device_.GetAllocator()} {}

MemoryAllocator::MemoryAllocator(const Device &device_, VmaAllocator allocator_)
        : device{device_}, allocator{allocator_},
            properties{device_.GetPhysical().GetMemoryProperties().memoryProperties},
            buffer_image_granularity{
                    device_.GetPhysical().GetProperties().limits.bufferImageGranularity},
            own_allocator{allocator_ != device_.GetAllocator()} {

    // Preserve the previous "RenderDoc small heap" trimming behavior that we had in original vma minus the heap bug
    if (device.HasDebuggingToolAttached())
    {
        using namespace Common::Literals;
        ForEachDeviceLocalHostVisibleHeap(device, [this](size_t heap_idx, VkMemoryHeap &heap) {
            if (heap.size <= 256_MiB) {
                for (u32 t = 0; t < properties.memoryTypeCount; ++t) {
                    if (properties.memoryTypes[t].heapIndex == heap_idx) {
                        valid_memory_types &= ~(1u << t);
                    }
                }
            }
        });
    }
}

MemoryAllocator::~MemoryAllocator() = default;

vk::Image MemoryAllocator::CreateImage(const VkImageCreateInfo &ci, bool dedicated,
                                       AllocationKind kind) const
{
    if (own_allocator) {
        return CreateImageOverBudget(ci, dedicated, kind);
    }
    // The device's allocator (the emulated GPU): within the budget, no fallback.
    const VmaAllocationCreateInfo alloc_ci = {
            .flags = static_cast<VmaAllocationCreateFlags>(VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT) |
                     (dedicated ? VmaAllocationCreateFlags{VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT}
                                : VmaAllocationCreateFlags{0}),
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            .requiredFlags = 0,
            .preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            .memoryTypeBits = 0,
            .pool = VK_NULL_HANDLE,
            .pUserData = nullptr,
            .priority = 0.f,
    };

    VkImage handle{};
    VmaAllocation allocation{};
    VmaAllocationInfo alloc_info{};
    vk::Check(vmaCreateImage(allocator, &ci, &alloc_ci, &handle, &allocation, &alloc_info));
    CheckPlacement(allocator, allocation, alloc_info.size, "image");
    if (spills != nullptr) {
        VkMemoryPropertyFlags property_flags{};
        vmaGetAllocationMemoryProperties(allocator, allocation, &property_flags);
        CountSpill(*spills, kind, true, property_flags, alloc_info.size);
    }

    // Log GPU memory allocation for images
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_memory_tracking.GetValue()) {
        GPU::Logging::GPULogger::GetInstance().LogMemoryAllocation(
            reinterpret_cast<uintptr_t>(alloc_info.deviceMemory),
            static_cast<u64>(alloc_info.size),
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
    }

    return vk::Image(handle, ci.usage, *device.GetLogical(), allocator, allocation,
                        device.GetDispatchLoader());
}

vk::Image MemoryAllocator::CreateImageOverBudget(const VkImageCreateInfo &ci, bool dedicated,
                                                 AllocationKind kind) const
{
    const VmaAllocationCreateInfo alloc_ci = {
            // Best fit: images freed by the thousand leave holes in the blocks, which new ones
            // should fill before VMA takes another block.
            .flags = static_cast<VmaAllocationCreateFlags>(VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT |
                                                           VMA_ALLOCATION_CREATE_STRATEGY_MIN_MEMORY_BIT) |
                     (dedicated ? VmaAllocationCreateFlags{VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT}
                                : VmaAllocationCreateFlags{0}),
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            // Strict: no silent fallback to a host memory type within the budget.
            .requiredFlags = device_strict ? VkMemoryPropertyFlags{VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT}
                                           : VkMemoryPropertyFlags{0},
            .preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            .memoryTypeBits = 0,
            .pool = VK_NULL_HANDLE,
            // The kind, for a detailed statistics dump of an allocator that opted in
            // (TrackSpills; "CustomData" there, kind + 1). Never read back.
            .pUserData = spills != nullptr
                             ? reinterpret_cast<void*>(static_cast<uintptr_t>(kind) + 1)
                             : nullptr,
            .priority = 0.f,
    };

    VkImage handle{};
    VmaAllocation allocation{};
    VmaAllocationInfo alloc_info{};
    // Images this large (title 1's sparse texture arrays reach 4.5G texels, ~23 GB) are never
    // backed: VMA would quietly place them in system memory once VRAM is full. They take the
    // unbacked sparse fallback below.
    const u64 texels = u64{ci.extent.width} * ci.extent.height * ci.extent.depth * ci.arrayLayers;
    const bool huge = texels > (u64{1} << 28);
    VkResult result = huge ? VK_ERROR_OUT_OF_DEVICE_MEMORY
                           : vmaCreateImage(allocator, &ci, &alloc_ci, &handle, &allocation, &alloc_info);
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY && !huge) {
        // Over the device memory budget (Switch 2 titles fill 8 GB cards): let the driver page
        // it out rather than failing the draw and aborting.
        static std::atomic<u32> logged{};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
            LOG_WARNING(Render_Vulkan,
                        "Image {}x{}x{} ({} levels, {} layers, format {}) is over the memory "
                        "budget; allocating it anyway",
                        ci.extent.width, ci.extent.height, ci.extent.depth, ci.mipLevels,
                        ci.arrayLayers, static_cast<u32>(ci.format));
        }
        pressure_events.fetch_add(1, std::memory_order_relaxed);
        VmaAllocationCreateInfo relaxed = alloc_ci;
        relaxed.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        result = vmaCreateImage(allocator, &ci, &relaxed, &handle, &allocation, &alloc_info);
        if (result == VK_SUCCESS && device_strict && spills != nullptr) {
            spills->over_budget_count.fetch_add(1, std::memory_order_relaxed);
            spills->over_budget_bytes.fetch_add(alloc_info.size, std::memory_order_relaxed);
            if (LogStrictFallback(spills->over_budget_count)) {
                LOG_WARNING(Render_Vulkan,
                            "[vma strict] {} image {}x{}x{} ({} MiB) placed in device-local "
                            "memory over the budget (#{})",
                            AllocationKindName(kind), ci.extent.width, ci.extent.height,
                            ci.extent.depth, alloc_info.size >> 20,
                            spills->over_budget_count.load(std::memory_order_relaxed));
            }
        }
    }
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY && !huge) {
        // Still out of device memory: a real image in host memory (slow, but valid) beats one
        // with nothing bound, which AMD's drivers do not survive being sampled.
        VmaAllocationCreateInfo host = alloc_ci;
        host.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        host.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        host.requiredFlags = 0;
        host.preferredFlags = 0;
        result = vmaCreateImage(allocator, &ci, &host, &handle, &allocation, &alloc_info);
        if (result == VK_SUCCESS && device_strict && spills != nullptr) {
            // The last resort of strict mode: every one is logged.
            spills->host_fallback_count.fetch_add(1, std::memory_order_relaxed);
            spills->host_fallback_bytes.fetch_add(alloc_info.size, std::memory_order_relaxed);
            LOG_WARNING(Render_Vulkan,
                        "[vma strict] {} image {}x{}x{} ({} MiB): no device-local memory left, "
                        "placed in host memory (#{})",
                        AllocationKindName(kind), ci.extent.width, ci.extent.height,
                        ci.extent.depth, alloc_info.size >> 20,
                        spills->host_fallback_count.load(std::memory_order_relaxed));
        }
        static std::atomic<u32> host_logged{};
        if (result == VK_SUCCESS && !device_strict &&
            host_logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            LOG_WARNING(Render_Vulkan,
                        "Image {}x{}x{} ({} levels, {} layers, format {}) is out of device memory; "
                        "placed in host memory",
                        ci.extent.width, ci.extent.height, ci.extent.depth, ci.mipLevels,
                        ci.arrayLayers, static_cast<u32>(ci.format));
        }
    }
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY && ci.imageType != VK_IMAGE_TYPE_1D) {
        if (!huge) {
            pressure_events.fetch_add(1, std::memory_order_relaxed);
        }
        // Too big to back at all, e.g. title 1's sparse virtual texture arrays (8192x8192,
        // 68 layers, ~23 GB): on the console only resident tiles have memory. Use an image with
        // nothing bound, so views and copies still work, instead of aborting.
        if (vk::Image sparse = CreateUnbackedImage(ci)) {
            LOG_WARNING(Render_Vulkan,
                        "Image {}x{}x{} ({} levels, {} layers, format {}) does not fit; using an "
                        "unbacked sparse image",
                        ci.extent.width, ci.extent.height, ci.extent.depth, ci.mipLevels,
                        ci.arrayLayers, static_cast<u32>(ci.format));
            return sparse;
        }
    }
    if (result != VK_SUCCESS) {
        LOG_CRITICAL(Render_Vulkan, "Image {}x{}x{} ({} levels, {} layers, format {}) failed: {}",
                     ci.extent.width, ci.extent.height, ci.extent.depth, ci.mipLevels,
                     ci.arrayLayers, static_cast<u32>(ci.format), static_cast<int>(result));
    }
    vk::Check(result);

    if (spills != nullptr) {
        VkMemoryPropertyFlags property_flags{};
        vmaGetAllocationMemoryProperties(allocator, allocation, &property_flags);
        CountSpill(*spills, kind, true, property_flags, alloc_info.size);
    }

    // Log GPU memory allocation for images
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_memory_tracking.GetValue()) {
        GPU::Logging::GPULogger::GetInstance().LogMemoryAllocation(
            reinterpret_cast<uintptr_t>(alloc_info.deviceMemory),
            static_cast<u64>(alloc_info.size),
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
    }

    return vk::Image(handle, ci.usage, *device.GetLogical(), allocator, allocation,
                        device.GetDispatchLoader());
}

vk::Image MemoryAllocator::CreateUnbackedImage(const VkImageCreateInfo &ci, bool aliased) const {
    if (ci.imageType == VK_IMAGE_TYPE_1D) {
        return vk::Image{};
    }
    VkImageCreateInfo sparse_ci = ci;
    sparse_ci.flags |= VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
    if (aliased) {
        sparse_ci.flags |= VK_IMAGE_CREATE_SPARSE_ALIASED_BIT;
    }
    VkImage sparse{};
    if (device.GetDispatchLoader().vkCreateImage(*device.GetLogical(), &sparse_ci, nullptr,
                                                 &sparse) != VK_SUCCESS) {
        return vk::Image{};
    }
    // No allocation: vmaDestroyImage (vk::Image's release) destroys just the image.
    return vk::Image(sparse, ci.usage, *device.GetLogical(), allocator, VK_NULL_HANDLE,
                     device.GetDispatchLoader());
}

VmaAllocation MemoryAllocator::AllocateSparseMemory(const VkMemoryRequirements &requirements,
                                                   VkDeviceMemory &memory,
                                                   VkDeviceSize &offset) const {
    // Eglantine also keeps these in a pool of their own (Device::SeparatePool), but only for the
    // device's shared allocator; the sink's allocator, the only caller, always took this path.
    const VmaAllocationCreateInfo alloc_ci{
        .flags = 0,
        .usage = VMA_MEMORY_USAGE_UNKNOWN,
        .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        .preferredFlags = 0,
        .memoryTypeBits = requirements.memoryTypeBits,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
        .priority = 0.f,
    };
    VmaAllocation allocation{};
    VmaAllocationInfo info{};
    if (vmaAllocateMemory(allocator, &requirements, &alloc_ci, &allocation, &info) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    memory = info.deviceMemory;
    offset = info.offset;
    return allocation;
}

void MemoryAllocator::FreeSparseMemory(VmaAllocation allocation) const {
    if (allocation != VK_NULL_HANDLE) {
        vmaFreeMemory(allocator, allocation);
    }
}

vk::Buffer MemoryAllocator::CreateBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage) const {
    return CreateBuffer(ci, usage, AllocationKind::Other);
}

vk::Buffer MemoryAllocator::CreateBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage,
                                         AllocationKind kind) const {
    // MESA will do memcpy() if not marked as host cached, so just force mark it for most buffers
    auto const anv_flags = (usage == MemoryUsage::Stream
        && device.GetDriverID() == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA)
        ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0;
    // Strict (TrackSpills; never set on the device's allocator): a DeviceLocal buffer gets no
    // silent fallback to a host memory type within the budget (Stream buffers fall back to host
    // memory by design).
    const bool strict = device_strict && usage == MemoryUsage::DeviceLocal;
    const VmaAllocationCreateInfo alloc_ci = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | MemoryUsageVmaFlags(usage),
        .usage = MemoryUsageVma(usage),
        .requiredFlags = strict ? VkMemoryPropertyFlags{VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT}
                                : VkMemoryPropertyFlags{0},
        .preferredFlags = MemoryUsagePreferredVmaFlags(usage) | anv_flags,
        .memoryTypeBits = usage == MemoryUsage::Stream ? 0u : valid_memory_types,
        .pool = VK_NULL_HANDLE,
        // The kind, for a detailed statistics dump (see CreateImageOverBudget).
        .pUserData = spills != nullptr ? reinterpret_cast<void*>(static_cast<uintptr_t>(kind) + 1)
                                       : nullptr,
        .priority = 0.f,
    };

    VkBuffer handle{};
    VmaAllocationInfo alloc_info{};
    VmaAllocation allocation{};
    VkMemoryPropertyFlags property_flags{};

    VkResult result = vmaCreateBuffer(allocator, &ci, &alloc_ci, &handle, &allocation, &alloc_info);
    if (result != VK_SUCCESS && strict) {
        // Over the budget: device-local anyway, the driver pages.
        VmaAllocationCreateInfo relaxed = alloc_ci;
        relaxed.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        result = vmaCreateBuffer(allocator, &ci, &relaxed, &handle, &allocation, &alloc_info);
        if (result == VK_SUCCESS && spills != nullptr) {
            spills->over_budget_count.fetch_add(1, std::memory_order_relaxed);
            spills->over_budget_bytes.fetch_add(alloc_info.size, std::memory_order_relaxed);
            if (LogStrictFallback(spills->over_budget_count)) {
                LOG_WARNING(Render_Vulkan,
                            "[vma strict] {} buffer ({} MiB) placed in device-local memory "
                            "over the budget (#{})",
                            AllocationKindName(kind), alloc_info.size >> 20,
                            spills->over_budget_count.load(std::memory_order_relaxed));
            }
        }
        if (result != VK_SUCCESS) {
            // No device-local memory type can hold it: host memory, the last resort.
            VmaAllocationCreateInfo host = relaxed;
            host.requiredFlags = 0;
            result = vmaCreateBuffer(allocator, &ci, &host, &handle, &allocation, &alloc_info);
            if (result == VK_SUCCESS && spills != nullptr) {
                spills->host_fallback_count.fetch_add(1, std::memory_order_relaxed);
                spills->host_fallback_bytes.fetch_add(alloc_info.size,
                                                      std::memory_order_relaxed);
                LOG_WARNING(Render_Vulkan,
                            "[vma strict] {} buffer ({} MiB): no device-local memory left, "
                            "placed in host memory (#{})",
                            AllocationKindName(kind), alloc_info.size >> 20,
                            spills->host_fallback_count.load(std::memory_order_relaxed));
            }
        }
    }
    if (result != VK_SUCCESS) {
        LOG_CRITICAL(Render_Vulkan, "Buffer allocation failed: bytes={}, usage={}, buffer_flags={:#x}, result={}",
                     ci.size, static_cast<u32>(usage), ci.usage, static_cast<s32>(result));
    }
    vk::Check(result);
    vmaGetAllocationMemoryProperties(allocator, allocation, &property_flags);
    if (spills != nullptr) {
        // Counted per kind instead of the placement warning.
        CountSpill(*spills, kind,
                   usage == MemoryUsage::DeviceLocal || usage == MemoryUsage::Stream,
                   property_flags, alloc_info.size);
    } else if (usage == MemoryUsage::DeviceLocal) {
        CheckPlacement(allocator, allocation, alloc_info.size, "buffer");
    }

    // Log GPU memory allocation for buffers
    if (GPU::Logging::IsActive() &&
        Settings::values.gpu_log_memory_tracking.GetValue()) {
        GPU::Logging::GPULogger::GetInstance().LogMemoryAllocation(
            reinterpret_cast<uintptr_t>(alloc_info.deviceMemory),
            static_cast<u64>(alloc_info.size),
            property_flags
        );
    }

    u8 *data = reinterpret_cast<u8 *>(alloc_info.pMappedData);
    const std::span<u8> mapped_data = data ? std::span<u8>{data, ci.size} : std::span<u8>{};
    const bool is_coherent = (property_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

    const vk::MemoryLocation location{
        .memory = alloc_info.deviceMemory,
        .offset = alloc_info.offset,
        .memory_type = alloc_info.memoryType,
    };
    return vk::Buffer(handle, *device.GetLogical(), allocator, allocation, mapped_data, is_coherent,
                      location, device.GetDispatchLoader());
}

vk::Buffer MemoryAllocator::CreateBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage,
                                         VkDeviceSize min_alignment) const {
    if (min_alignment <= 1) {
        return CreateBuffer(ci, usage);
    }
    VkMemoryPropertyFlags anv_flags = 0;
    if (usage == MemoryUsage::Stream &&
        device.GetDriverID() == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA) {
        anv_flags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    }
    u32 memory_type_bits = valid_memory_types;
    if (usage == MemoryUsage::Stream) {
        memory_type_bits = 0u;
    }
    const VmaAllocationCreateInfo alloc_ci = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | MemoryUsageVmaFlags(usage),
        .usage = MemoryUsageVma(usage),
        .requiredFlags = 0,
        .preferredFlags = MemoryUsagePreferredVmaFlags(usage) | anv_flags,
        .memoryTypeBits = memory_type_bits,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
        .priority = 0.f,
    };

    VkBuffer handle{};
    VmaAllocationInfo alloc_info{};
    VmaAllocation allocation{};
    VkMemoryPropertyFlags property_flags{};

    vk::Check(vmaCreateBufferWithAlignment(allocator, &ci, &alloc_ci, min_alignment, &handle,
                                           &allocation, &alloc_info));
    vmaGetAllocationMemoryProperties(allocator, allocation, &property_flags);
    if (usage == MemoryUsage::DeviceLocal) {
        CheckPlacement(allocator, allocation, alloc_info.size, "buffer");
    }

    u8 *data = reinterpret_cast<u8 *>(alloc_info.pMappedData);
    std::span<u8> mapped_data{};
    if (data) {
        mapped_data = std::span<u8>{data, ci.size};
    }
    const bool is_coherent = (property_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

    const vk::MemoryLocation location{
        .memory = alloc_info.deviceMemory,
        .offset = alloc_info.offset,
        .memory_type = alloc_info.memoryType,
    };
    return vk::Buffer(handle, *device.GetLogical(), allocator, allocation, mapped_data, is_coherent,
                      location, device.GetDispatchLoader());
}

MemoryCommit MemoryAllocator::Commit(const VkMemoryRequirements &reqs, MemoryUsage usage)
{
    const auto vma_usage = MemoryUsageVma(usage);
    VmaAllocationCreateInfo ci{};
    ci.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | MemoryUsageVmaFlags(usage);
    ci.usage = vma_usage;
    ci.memoryTypeBits = reqs.memoryTypeBits & valid_memory_types;
    ci.requiredFlags = 0;
    ci.preferredFlags = MemoryUsagePreferredVmaFlags(usage);

    VmaAllocation a{};
    VmaAllocationInfo info{};

    VkResult res = vmaAllocateMemory(allocator, &reqs, &ci, &a, &info);

    if (res != VK_SUCCESS) {
        // Relax 1: drop budget constraint
        auto ci2 = ci;
        ci2.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        res = vmaAllocateMemory(allocator, &reqs, &ci2, &a, &info);

        // Relax 2: if we preferred DEVICE_LOCAL, drop that preference
        if (res != VK_SUCCESS && (ci.preferredFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            auto ci3 = ci2;
            ci3.preferredFlags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            res = vmaAllocateMemory(allocator, &reqs, &ci3, &a, &info);
        }
    }

    vk::Check(res);
    if (usage == MemoryUsage::DeviceLocal) {
        CheckPlacement(allocator, a, info.size, "commit");
    }
    return MemoryCommit(allocator, a, info);
}

MemoryCommit MemoryAllocator::Commit(const vk::Buffer &buffer, MemoryUsage usage) {
    // Allocate memory appropriate for this buffer automatically
    const auto vma_usage = MemoryUsageVma(usage);

    VmaAllocationCreateInfo ci{};
    ci.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | MemoryUsageVmaFlags(usage);
    ci.usage = vma_usage;
    ci.requiredFlags = 0;
    ci.preferredFlags = MemoryUsagePreferredVmaFlags(usage);
    ci.pool = VK_NULL_HANDLE;
    ci.pUserData = nullptr;
    ci.priority = 0.0f;

    const VkBuffer raw = *buffer;

    VmaAllocation a{};
    VmaAllocationInfo info{};

    // Let VMA infer memory requirements from the buffer
    VkResult res = vmaAllocateMemoryForBuffer(allocator, raw, &ci, &a, &info);

    if (res != VK_SUCCESS) {
        auto ci2 = ci;
        ci2.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        res = vmaAllocateMemoryForBuffer(allocator, raw, &ci2, &a, &info);

        if (res != VK_SUCCESS && (ci.preferredFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            auto ci3 = ci2;
            ci3.preferredFlags &= ~VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            res = vmaAllocateMemoryForBuffer(allocator, raw, &ci3, &a, &info);
        }
    }

    vk::Check(res);
    vk::Check(vmaBindBufferMemory2(allocator, a, 0, raw, nullptr));
    return MemoryCommit(allocator, a, info);
}

} // namespace Vulkan
