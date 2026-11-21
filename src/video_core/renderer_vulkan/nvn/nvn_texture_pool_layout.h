// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// The layout side of the NVN replay's texture pool descriptor set (nvn_texture_pool.h): when
// nvn-hle programs are translated with texture pool lookups, and the set layout the pipeline
// cache and the replay both create (identically defined, so their sets are compatible).

#pragma once

#include <array>
#include <cstdlib>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan::Nvn {

inline constexpr u32 POOL_VIEW_CAPACITY = Shader::TEXTURE_POOL_VIEW_CAPACITY;
inline constexpr u32 POOL_SAMPLER_CAPACITY = 4096; ///< A handle holds a 12-bit sampler index

/// The view arrays of the set (one per image type, Shader::TexturePoolBinding). Multisampled
/// arrays are not provided: a program with a multisampled pool lookup is translated without the
/// pool (PipelineCache::Impl::Translate).
inline constexpr std::array POOL_VIEW_TYPES{
    Shader::TextureType::Color1D,   Shader::TextureType::ColorArray1D,
    Shader::TextureType::Color2D,   Shader::TextureType::ColorArray2D,
    Shader::TextureType::Color3D,   Shader::TextureType::ColorCube,
    Shader::TextureType::ColorArrayCube,
};

/// Position of `type` in POOL_VIEW_TYPES (POOL_VIEW_TYPES.size(): no array).
[[nodiscard]] constexpr size_t PoolSlotOf(Shader::TextureType type) noexcept {
    if (type == Shader::TextureType::Color2DRect) {
        type = Shader::TextureType::Color2D;
    }
    for (size_t slot = 0; slot < POOL_VIEW_TYPES.size(); ++slot) {
        if (POOL_VIEW_TYPES[slot] == type) {
            return slot;
        }
    }
    return POOL_VIEW_TYPES.size();
}

/// The POOL_VIEW_TYPES positions of a DescriptorLayout::texture_pool_types mask.
[[nodiscard]] constexpr u32 PoolSlotsOf(u32 type_mask) noexcept {
    u32 slots = 0;
    for (u32 type = 0; type < Shader::NUM_TEXTURE_TYPES; ++type) {
        if ((type_mask >> type) & 1) {
            const size_t slot = PoolSlotOf(static_cast<Shader::TextureType>(type));
            if (slot < POOL_VIEW_TYPES.size()) {
                slots |= 1u << slot;
            }
        }
    }
    return slots;
}

/// Every pool lookup of `info` has a view array in the set.
[[nodiscard]] inline bool PoolLookupsSupported(const Shader::Info& info) noexcept {
    for (const Shader::TexturePoolDescriptor& desc : info.texture_pool_descriptors) {
        if (desc.is_multisample || PoolSlotOf(desc.type) == POOL_VIEW_TYPES.size()) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] constexpr VkImageViewType PoolViewType(size_t slot) noexcept {
    switch (POOL_VIEW_TYPES[slot]) {
    case Shader::TextureType::Color1D:
        return VK_IMAGE_VIEW_TYPE_1D;
    case Shader::TextureType::ColorArray1D:
        return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
    case Shader::TextureType::ColorArray2D:
        return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    case Shader::TextureType::Color3D:
        return VK_IMAGE_VIEW_TYPE_3D;
    case Shader::TextureType::ColorCube:
        return VK_IMAGE_VIEW_TYPE_CUBE;
    case Shader::TextureType::ColorArrayCube:
        return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    default:
        return VK_IMAGE_VIEW_TYPE_2D;
    }
}

/// EDEN_NVN_NO_TEXTURE_POOL=1: nvn-hle programs are translated without the texture pool (a handle
/// read per invocation gets the placeholder handle and samples the dummy, as before).
[[nodiscard]] inline bool TexturePoolDisabledByEnvironment() {
    static const bool disabled = [] {
        const char* const value = std::getenv("EDEN_NVN_NO_TEXTURE_POOL");
        return value != nullptr && value[0] != 0 && value[0] != '0';
    }();
    return disabled;
}

/// The device can hold the set: run-time sized, non-uniformly indexed, partially bound arrays,
/// and (as vk_texture_pool.h: half of the per-set limits, the other half for set 0) the arrays.
[[nodiscard]] inline bool TexturePoolDeviceSupported(const Device& device) {
    if (!device.IsTexturePoolSupported()) {
        return false;
    }
    const u64 view_limit = u64{device.GetMaxDescriptorSetSampledImages()} / 2;
    const u64 sampler_limit = u64{device.GetMaxDescriptorSetSamplers()} / 2;
    return view_limit >= u64{POOL_VIEW_CAPACITY} * POOL_VIEW_TYPES.size() &&
           sampler_limit >= POOL_SAMPLER_CAPACITY;
}

/// nvn-hle programs are translated with texture pool lookups (TexturePoolMode::Fallback).
[[nodiscard]] inline bool TexturePoolEnabled(const Device& device) {
    return !TexturePoolDisabledByEnvironment() && TexturePoolDeviceSupported(device);
}

/// Stages of the set's bindings; the pipeline cache and the replay create identically defined
/// layouts from this (descriptor sets of one are compatible with the other).
[[nodiscard]] inline VkShaderStageFlags PoolStages(const Device& device) {
    VkShaderStageFlags stages = VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT;
    if (device.IsExtMeshShaderSupported()) {
        stages |= VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT;
    }
    return stages;
}

/// The layout of the set: the sampler array, one view array per POOL_VIEW_TYPES entry and the
/// feedback buffer, every binding partially bound (arrays of types no pipeline reads are never
/// written).
[[nodiscard]] inline vk::DescriptorSetLayout CreateTexturePoolSetLayout(const Device& device) {
    const VkShaderStageFlags stages = PoolStages(device);
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.push_back({
        .binding = Shader::TEXTURE_POOL_SAMPLER_BINDING,
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
        .descriptorCount = POOL_SAMPLER_CAPACITY,
        .stageFlags = stages,
        .pImmutableSamplers = nullptr,
    });
    for (const Shader::TextureType type : POOL_VIEW_TYPES) {
        bindings.push_back({
            .binding = Shader::TexturePoolBinding(type, false),
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .descriptorCount = POOL_VIEW_CAPACITY,
            .stageFlags = stages,
            .pImmutableSamplers = nullptr,
        });
    }
    bindings.push_back({
        .binding = Shader::TEXTURE_POOL_FEEDBACK_BINDING,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = stages,
        .pImmutableSamplers = nullptr,
    });
    const std::vector<VkDescriptorBindingFlags> binding_flags(
        bindings.size(), VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
    const VkDescriptorSetLayoutBindingFlagsCreateInfo flags_ci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .pNext = nullptr,
        .bindingCount = static_cast<u32>(binding_flags.size()),
        .pBindingFlags = binding_flags.data(),
    };
    return device.GetLogical().CreateDescriptorSetLayout({
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &flags_ci,
        .flags = 0,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    });
}

} // namespace Vulkan::Nvn
