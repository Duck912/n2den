// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/vulkan_common/vma.h"

namespace Vulkan {

    class Device;

/// Hints and requirements for the backing memory type of a commit
    enum class MemoryUsage {
        DeviceLocal, ///< Requests device local host visible buffer, falling back to device local memory.
        Upload,      ///< Requires a host visible memory type optimized for CPU to GPU uploads
        Download,    ///< Requires a host visible memory type optimized for GPU to CPU readbacks
        Stream,      ///< Requests device local host visible buffer, falling back host memory.
    };

/// What an allocation is for, for the spill counters (MemoryAllocator::TrackSpills).
    enum class AllocationKind : u8 {
        Other,
        StorageImage,
        PoolBuffer,
        Shadow,
        Mirror,
        Arena,
        Staging,
        Count,
    };

    [[nodiscard]] const char* AllocationKindName(AllocationKind kind);

/// Allocations that asked for device-local memory (images, DeviceLocal and Stream buffers) and
/// were placed in a memory type without DEVICE_LOCAL: VMA moves on to the next memory type once
/// the device heap is at its budget, and the allocation stays in system memory for its lifetime.
/// Cumulative. Only allocators that opted in (TrackSpills) count.
    struct AllocationSpills {
        static constexpr size_t KINDS = static_cast<size_t>(AllocationKind::Count);
        std::array<std::atomic<u64>, KINDS> count{};
        std::array<std::atomic<u64>, KINDS> bytes{};
        /// Strict mode: device-local allocations made over the budget (the driver pages them) ...
        std::atomic<u64> over_budget_count{};
        std::atomic<u64> over_budget_bytes{};
        /// ... and the ones no device-local memory type took, placed in host memory.
        std::atomic<u64> host_fallback_count{};
        std::atomic<u64> host_fallback_bytes{};
    };

/// Counts `size` bytes of `kind` in `spills` when the allocation wanted device-local memory and
/// got a memory type of `flags` without DEVICE_LOCAL. Returns whether it counted.
    bool CountSpill(AllocationSpills& spills, AllocationKind kind, bool wanted_device_local,
                    VkMemoryPropertyFlags flags, u64 size);

    template<typename F>
    void ForEachDeviceLocalHostVisibleHeap(const Device &device, F &&f) {
        auto memory_props = device.GetPhysical().GetMemoryProperties().memoryProperties;
        for (size_t i = 0; i < memory_props.memoryTypeCount; i++) {
            auto &memory_type = memory_props.memoryTypes[i];
            if ((memory_type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                (memory_type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                f(memory_type.heapIndex, memory_props.memoryHeaps[memory_type.heapIndex]);
            }
        }
    }

/// Ownership handle of a memory commitment (real VMA allocation).
    class MemoryCommit {
    public:
        MemoryCommit() noexcept = default;

        MemoryCommit(VmaAllocator allocator, VmaAllocation allocation,
                     const VmaAllocationInfo &info) noexcept;

        ~MemoryCommit();

        MemoryCommit(const MemoryCommit &) = delete;

        MemoryCommit &operator=(const MemoryCommit &) = delete;

        MemoryCommit(MemoryCommit &&) noexcept;

        MemoryCommit &operator=(MemoryCommit &&) noexcept;

        [[nodiscard]] std::span<u8> Map();

        [[nodiscard]] std::span<const u8> Map() const;

        void Unmap();

        explicit operator bool() const noexcept { return allocation != nullptr; }

        VkDeviceMemory Memory() const noexcept { return memory; }

        VkDeviceSize Offset() const noexcept { return offset; }

        VkDeviceSize Size() const noexcept { return size; }

        VmaAllocation Allocation() const noexcept { return allocation; }

    private:
        void Release();

        VmaAllocator allocator{};   ///< VMA allocator
        VmaAllocation allocation{};  ///< VMA allocation handle
        VkDeviceMemory memory{};      ///< Underlying VkDeviceMemory chosen by VMA
        VkDeviceSize offset{};      ///< Offset of this allocation inside VkDeviceMemory
        VkDeviceSize size{};        ///< Size of the allocation
        void *mapped_ptr{};  ///< Optional persistent mapped pointer
    };

/// Memory allocator container.
/// Allocates and releases memory allocations on demand.
    class MemoryAllocator {
    public:
        /**
         * Construct memory allocator
         *
         * @param device_  Device to allocate from
         *
         * @throw vk::Exception on failure
         */
        explicit MemoryAllocator(const Device &device_);

        /// Allocates from `allocator_` instead of the device's shared one (which is externally
        /// synchronized). Used by the NVN sink, which must not race the render thread. An
        /// allocator made this way also takes the sink's image policy (see CreateImage); the
        /// one-argument constructor keeps the emulated GPU's.
        explicit MemoryAllocator(const Device &device_, VmaAllocator allocator_);

        ~MemoryAllocator();

        MemoryAllocator &operator=(const MemoryAllocator &) = delete;

        MemoryAllocator(const MemoryAllocator &) = delete;

        /// Counts this allocator's spills into `spills_` (null: none). `device_strict_`: images
        /// and DeviceLocal buffers require a DEVICE_LOCAL memory type, over the budget too (the
        /// driver pages), and take host memory only when no device-local type can hold them.
        /// Set before the allocator is used.
        void TrackSpills(AllocationSpills* spills_, bool device_strict_) noexcept {
            spills = spills_;
            device_strict = device_strict_;
        }

        /// `dedicated`: own device memory, returned to the driver on destruction (short-lived
        /// images would otherwise leave holes in the shared blocks).
        /// The device's allocator (one-argument constructor) allocates within the budget and
        /// throws when that fails. An allocator over its own VmaAllocator (the NVN sink's) goes
        /// over the budget, then to host memory, then to an unbacked sparse image instead.
        vk::Image CreateImage(const VkImageCreateInfo &ci, bool dedicated = false,
                              AllocationKind kind = AllocationKind::Other) const;

        /// A sparse resident image with nothing bound (reads return zero, writes are dropped), or
        /// a null image when the device can't make one.
        /// aliased: its sparse blocks may share memory (VK_IMAGE_CREATE_SPARSE_ALIASED_BIT).
        vk::Image CreateUnbackedImage(const VkImageCreateInfo &ci, bool aliased = false) const;

        /// Device memory to bind to part of a sparse resource. Null when there is none left.
        VmaAllocation AllocateSparseMemory(const VkMemoryRequirements &requirements,
                                           VkDeviceMemory &memory, VkDeviceSize &offset) const;

        void FreeSparseMemory(VmaAllocation allocation) const;

        /// Counts the images that did not fit the device memory budget (made anyway, in host
        /// memory or with nothing bound). Only the sink's image policy counts.
        u32 MemoryPressureEvents() const noexcept {
            return pressure_events.load(std::memory_order_relaxed);
        }

        vk::Buffer CreateBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage) const;

        vk::Buffer CreateBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage,
                                AllocationKind kind) const;

        vk::Buffer CreateBuffer(const VkBufferCreateInfo &ci, MemoryUsage usage,
                                VkDeviceSize min_alignment) const;

        /**
         * Commits a memory with the specified requirements.
         *
         * @param requirements Requirements returned from a Vulkan call.
         * @param usage        Indicates how the memory will be used.
         *
         * @returns A memory commit.
         */
        MemoryCommit Commit(const VkMemoryRequirements &requirements, MemoryUsage usage);

        /// Commits memory required by the buffer and binds it (for buffers created outside VMA).
        MemoryCommit Commit(const vk::Buffer &buffer, MemoryUsage usage);

    private:
        /// CreateImage for an allocator over its own VmaAllocator.
        vk::Image CreateImageOverBudget(const VkImageCreateInfo &ci, bool dedicated,
                                        AllocationKind kind) const;

        static bool IsAutoUsage(VmaMemoryUsage u) noexcept {
            switch (u) {
                case VMA_MEMORY_USAGE_AUTO:
                case VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE:
                case VMA_MEMORY_USAGE_AUTO_PREFER_HOST:
                    return true;
                default:
                    return false;
            }
        }

        const Device &device;                              ///< Device handle.
        VmaAllocator allocator;                           ///< VMA allocator.
        const VkPhysicalDeviceMemoryProperties properties; ///< Physical device memory properties.
        VkDeviceSize buffer_image_granularity;            ///< Adjacent buffer/image granularity
        u32 valid_memory_types{~0u};
        /// Constructed over a VmaAllocator of its own: the NVN sink's image policy.
        const bool own_allocator;
        mutable std::atomic<u32> pressure_events{0};
        AllocationSpills* spills{};
        bool device_strict{};
    };

} // namespace Vulkan
