// SPDX-FileCopyrightText: Copyright 2026 n2den Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <string>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T*;

namespace Vulkan {
class Device;
}

namespace Vulkan::Nvn {

/// What a memory pool buffer can be used for.
constexpr VkBufferUsageFlags POOL_BUFFER_USAGE =
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

/// POOL_BUFFER_USAGE plus a device address (raw global memory, VK_KHR_buffer_device_address): used
/// when the device has the feature. Imports allocate their memory with DEVICE_ADDRESS then.
constexpr VkBufferUsageFlags POOL_BUFFER_USAGE_ADDRESSABLE =
    POOL_BUFFER_USAGE | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

/// A VkBuffer over imported host memory (VK_EXT_external_memory_host). The host range must stay
/// mapped for as long as the buffer exists.
struct ImportedBuffer {
    vk::DeviceMemory memory; // declared first: destroyed after the buffer
    vk::Buffer buffer;
    VkDeviceSize size{};
};

/// A large host range imported once as device memory; several pool buffers are bound into it at
/// their offsets (one import costs a few ms; see Common::HostMemory::BackingViewSize).
struct ImportedChunk {
    vk::DeviceMemory memory;
    u8* host{};
    VkDeviceSize size{};
    u32 memory_type_bits{}; ///< host-pointer types usable with pool buffers
    u32 type_index{};
    /// Guest (GPU) address of `host`: the chunk is [guest_base, guest_base + size) contiguous.
    u64 guest_base{};
    /// One buffer over the whole chunk with a device address, created the first time raw global
    /// memory needs the chunk (declared last: destroyed before the memory).
    vk::Buffer address_buffer;
    u64 device_address{};
};

/// VK_EXT_external_memory_host import. Loads vkGetMemoryHostPointerPropertiesEXT with
/// vkGetDeviceProcAddr and reads minImportedHostPointerAlignment itself.
class HostImporter {
public:
    /// `allocator` is stored in imported buffers so they are destroyed with the right owner; it is
    /// never used to allocate.
    HostImporter(const Device& device, VmaAllocator allocator);

    [[nodiscard]] bool Available() const noexcept {
        return get_properties != nullptr;
    }

    /// Called with true right before and with false right after the vkAllocateMemory of an import
    /// (and at no other time), on the importing thread. On the NVIDIA Windows driver that call
    /// reads byte 0 of every imported page and writes it back, not atomically, so whoever imports
    /// live guest memory holds the guest's stores to the range off for its duration
    /// (GuestMemory::BeginImportGate).
    using AllocateHook = std::function<void(bool begin)>;

    [[nodiscard]] u64 Alignment() const noexcept {
        return min_alignment;
    }

    /// True when `ptr`/`size` satisfy the import alignment.
    [[nodiscard]] bool IsAligned(const void* ptr, u64 size) const noexcept;

    /// Imports [ptr, ptr + size) as a buffer with `usage`. Returns VK_SUCCESS or the failing
    /// result; `error` is filled with a short reason on failure.
    VkResult Import(void* ptr, VkDeviceSize size, VkBufferUsageFlags usage, ImportedBuffer& out,
                    std::string& error, const AllocateHook* hook = nullptr) const;

    /// Gives every page of [ptr, ptr + size) its host page before an import. Pages nobody has
    /// touched yet have none: vkAllocateMemory would fault each one in (43 ms for 256 MiB) while
    /// it holds the driver lock every vkQueueSubmit of the device waits for. Reading one byte per
    /// page outside that lock does the same work (~50 ms per 256 MiB on the calling thread), and
    /// the import then only pins resident pages (~3 ms per 256 MiB). A read leaves the bytes as
    /// they are while the guest keeps writing them. On Windows only committed readable pages are
    /// read (the deferred-commit backing has reserved pages nobody was given). The imports call
    /// it themselves; calling it earlier, without locks held, moves the work off them.
    /// `window`: the range is a chunk window (pages around a pool), not a pool's own pages.
    /// Returns about how many bytes had no host page before, i.e. the RAM the call cost (0 with
    /// EDEN_NVN_IMPORT_RESIDENT_DIAG=0 or where it cannot be measured).
    static u64 MakeResident(const u8* ptr, VkDeviceSize size, bool window);

    /// What MakeResident has touched so far, chunk windows and pools' own pages apart.
    struct ResidentStats {
        u64 calls{};
        u64 bytes{};     ///< touched (a range touched twice counts twice)
        u64 new_bytes{}; ///< of that, about how much had no host page before
    };
    [[nodiscard]] static ResidentStats ResidentTotals(bool window) noexcept;

    /// Whether some page of [ptr, ptr + size) has a host page at that address (the page is in
    /// the process' working set: somebody read or wrote it through this mapping). Touches
    /// nothing, so it gives no page RAM; `ptr` need not even be mapped. A page reached only
    /// through another mapping of the same memory does not count: ask for each mapping. True
    /// when the host cannot tell (not Windows, the query failed): the caller then treats the
    /// range as used. About 0.1-0.3 us per page.
    [[nodiscard]] static bool AnyResident(const u8* ptr, u64 size);

    /// Where an import's time went: making the pages resident (MakeResident),
    /// vkGetMemoryHostPointerPropertiesEXT and vkAllocateMemory.
    struct ImportTiming {
        u64 resident_ns{};
        u64 props_ns{};
        u64 alloc_ns{};
    };

    /// Imports [ptr, ptr + size) as device memory for buffers with `usage` (chunk of many pools).
    /// `timing`, when given, receives where the time went.
    VkResult ImportChunk(u8* ptr, VkDeviceSize size, VkBufferUsageFlags usage, ImportedChunk& out,
                         std::string& error, ImportTiming* timing = nullptr,
                         const AllocateHook* hook = nullptr) const;

    /// Creates a buffer of `size` bytes with `usage` over `chunk` at `offset` (a multiple of the
    /// buffer's alignment requirement).
    VkResult CreateBufferInChunk(const ImportedChunk& chunk, VkDeviceSize offset, VkDeviceSize size,
                                 VkBufferUsageFlags usage, vk::Buffer& out,
                                 std::string& error) const;

private:
    /// Reads one byte of every readable page of [ptr, ptr + size).
    static void TouchPages(const u8* ptr, VkDeviceSize size);

    const Device& device;
    VmaAllocator allocator;
    PFN_vkGetMemoryHostPointerPropertiesEXT get_properties{};
    u64 min_alignment{};
};

} // namespace Vulkan::Nvn
