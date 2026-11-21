// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// The texture pool descriptor set of the NVN replay (Shader::TEXTURE_POOL_DESCRIPTOR_SET): what a
// program reads when it looks a texture handle up per invocation (Ocarina of Time's lit fragment
// programs read the shadow map of each light from the light records). The emulated GPU path has
// its own (vk_texture_pool.h); this one serves nvn-hle pipelines from the replay's textures.
//
// Element i of the view arrays is the texture at global id `texture_pool_base + i`, element j of
// the sampler array the sampler at `sampler_pool_base + j` (shaders index with the pool-relative
// handle bits). Only elements a shader marked in the feedback buffer recently are resolved; the
// others hold the null view (a depth dummy, so depth-compare lookups stay defined) and the
// fallback sampler.
//
// One set per draw context (guest queue) that draws with pool lookups: the resolve state, marks
// and sets of a queue are its own. The desired contents change only when something can have
// changed: once per flight (new feedback marks, mark aging), per submission (the id snapshot),
// when a texture or sampler the queue resolved is released, when the pool bases or the image
// types pipelines read change. A set that recorded commands may still read is never written; a
// change goes into a set whose last use completed (only the elements it holds differently are
// written), a new one, or a one-off set kept alive by the flight. Only when all of those are
// exhausted is the current set bound as it is, and only if every image it names is still in the
// layout it states. Drawing never waits for the GPU (creating the set fills its depth null views
// once with a one-time submission).
//
// Limitations: an element reads null until its mark is seen and the next resolve ran; a shadow
// map that is a render target of the draw being recorded is not copied (the per-draw path's
// copy-on-read): its element keeps the read-only layout the set states while the image is an
// attachment (valid only while the draw does not look it up; counted as target-of-draw images);
// pool elements read the guest-size image, never a scaled twin.

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw_resources.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"
#include "video_core/renderer_vulkan/nvn/nvn_texture_pool_layout.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan::Nvn {

/// The replay's texture pool set (owned by DrawResources; replay thread only).
///
/// Per pool draw: BeginDraw returns the generation of the desired contents. When it moved since
/// the draw context last resolved, the context resolves again: BeginResolve, SetView / SetSampler
/// for the elements that resolve (views once the images exist), EndResolve. Then Commit returns
/// the set to bind as set 1.
class TexturePoolSet {
public:
    static constexpr size_t MAX_SETS = 8;

    TexturePoolSet(Context::Impl& impl_, DrawResources& resources_, u32 queue_)
        : impl{impl_}, resources{resources_}, device{impl_.device}, queue{queue_} {
        try {
            layout = CreateTexturePoolSetLayout(device);
            feedback = impl.CreateHostBuffer(FEEDBACK_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                             MemoryUsage::Download);
        } catch (const vk::Exception& exception) {
            LOG_ERROR(Render_Vulkan, "[nvn] texture pool: set creation failed: {}",
                      exception.what());
            return;
        }
        if (!feedback.IsHostVisible()) {
            LOG_ERROR(Render_Vulkan, "[nvn] texture pool: the feedback buffer is not host visible");
            return;
        }
        std::memset(feedback.Mapped().data(), 0, FEEDBACK_SIZE);
        if (!feedback.IsHostCoherent()) {
            feedback.Flush();
        }
        view_version.assign(POOL_VIEW_CAPACITY, 0);
        view_serial.assign(POOL_VIEW_CAPACITY, 0);
        view_touched.assign(POOL_VIEW_CAPACITY, 0);
        view_resolved.assign(POOL_VIEW_CAPACITY, 0);
        sampler_version.assign(POOL_SAMPLER_CAPACITY, 0);
        sampler_serial.assign(POOL_SAMPLER_CAPACITY, 0);
        sampler_touched.assign(POOL_SAMPLER_CAPACITY, 0);
        sampler_resolved.assign(POOL_SAMPLER_CAPACITY, 0);
        seen_words.assign(Shader::TEXTURE_POOL_FEEDBACK_WORDS, 0);
        sampler_infos.assign(POOL_SAMPLER_CAPACITY, NullSampler());
        static const bool aging = [] {
            const char* const value = std::getenv("EDEN_POOL_MARK_AGING");
            return !(value && value[0] == '0');
        }();
        aging_enabled = aging;
        static const u64 window = [] {
            const char* const value = std::getenv("EDEN_NVN_POOL_MARK_WINDOW");
            const u64 parsed = value != nullptr ? std::strtoull(value, nullptr, 10) : 0;
            return parsed != 0 ? parsed : DEFAULT_WINDOW_PRESENTS;
        }();
        window_presents = window;
        CreateDepthDummies();
        resources.texture_pools.push_back(this);
        ok = true;
        LOG_INFO(Render_Vulkan,
                 "[nvn] texture pool (queue {}): set ready ({} views per image type x {} types, {} samplers; "
                 "device limits {} sampled images, {} samplers per set); mark aging {} ({} "
                 "presents x {} windows)",
                 queue, POOL_VIEW_CAPACITY, POOL_VIEW_TYPES.size(), POOL_SAMPLER_CAPACITY,
                 device.GetMaxDescriptorSetSampledImages(), device.GetMaxDescriptorSetSamplers(),
                 aging_enabled ? "on" : "off", window_presents, HISTORY_WINDOWS);
    }

    ~TexturePoolSet() {
        std::erase(resources.texture_pools, this);
    }

    TexturePoolSet(const TexturePoolSet&) = delete;
    TexturePoolSet& operator=(const TexturePoolSet&) = delete;

    [[nodiscard]] bool Ok() const noexcept {
        return ok;
    }
    /// Record-ahead (stage 2): the generation moves that came from GPU feedback, mark aging or
    /// released textures / samplers (inputs that depend on when the draw is recorded relative to
    /// completions and commands), and the queue of this set.
    [[nodiscard]] u64 TimedGeneration() const noexcept {
        return timed_generation;
    }
    [[nodiscard]] u32 Queue() const noexcept {
        return queue;
    }

    /// Phase A of every pool draw: `type_mask` is DescriptorLayout::texture_pool_types of the
    /// pipeline, `frame` the presents recorded (mark aging), the pool bases name the guest ids of
    /// element 0, `seq` the flight being recorded (feedback is scanned once per flight). Returns
    /// the generation of the desired contents: the caller resolves again when it moved.
    [[nodiscard]] u64 BeginDraw(u32 type_mask, u64 frame, u32 texture_base, u32 sampler_base,
                                u64 seq) {
        const u32 slots = desired_slots | PoolSlotsOf(type_mask);
        if (slots != desired_slots) {
            for (size_t slot = 0; slot < POOL_VIEW_TYPES.size(); ++slot) {
                if (((slots >> slot) & 1) && view_infos[slot].empty()) {
                    view_infos[slot].assign(POOL_VIEW_CAPACITY, NullView(slot));
                }
            }
            desired_slots = slots;
            ++desired_serial;
            ++generation;
        }
        if (texture_base != bases[0] || sampler_base != bases[1]) {
            bases = {texture_base, sampler_base};
            ++stats.base_changes;
            ++generation;
        }
        if (seq != scanned_seq) {
            scanned_seq = seq;
            if (ScanFeedback()) {
                ++generation;
                ++timed_generation;
            }
            if (AgeMarks(frame)) {
                ++generation;
                ++timed_generation;
            }
        }
        ProcessReleases();
        return generation;
    }

    [[nodiscard]] std::span<const u32> RequestedViews() const noexcept {
        return requested_views;
    }
    [[nodiscard]] std::span<const u32> RequestedSamplers() const noexcept {
        return requested_samplers;
    }
    /// POOL_VIEW_TYPES positions some pool pipeline reads.
    [[nodiscard]] u32 UsedSlots() const noexcept {
        return desired_slots;
    }

    /// DrawResources::NoteViewsReleased: the views of texture `serial` were released.
    void NoteReleased(u64 serial) {
        if (released_overflow) {
            return;
        }
        if (released.size() >= 65536) {
            released.clear();
            released_overflow = true;
            return;
        }
        released.push_back(serial);
    }

    /// Starts resolving the requested elements (every one is null unless set before EndResolve).
    void BeginResolve() {
        for (const u32 index : requested_views) {
            view_resolved[index] = 0;
        }
        for (const u32 index : requested_samplers) {
            sampler_resolved[index] = 0;
        }
        resolved_textures.clear();
        ++stats.resolves;
    }

    /// The resolve names texture object `serial` (the draw context keeps pointers to it until the
    /// next resolve): its release moves the generation.
    void NoteResolvedTexture(u64 serial) {
        resolved_textures.insert(serial);
    }

    /// The texture id of element `index` names no live texture: the element leaves the requested
    /// list. Its feedback bit stays seen (shaders keep setting it) until the next window, when
    /// dead ids are looked at again (RetryDead), so a dead id costs one resolve per window, not
    /// one per flight.
    void DropDead(u32 index) {
        if (std::erase(requested_views, index) != 0) {
            NullAllSlots(index);
            dead_views.push_back(index);
            ++stats.dropped_dead;
        }
    }

    /// Element `index` of the view array `slot` is `view` in `layout` (texture object `serial`).
    void SetView(u32 index, size_t slot, VkImageView view, VkImageLayout image_layout, u64 serial) {
        view_resolved[index] |= static_cast<u8>(1u << slot);
        PutView(index, slot, view, image_layout, serial);
    }

    /// Element `index` of the sampler array is `sampler` (sampler object `serial`, 0: none).
    void SetSampler(u32 index, VkSampler sampler, u64 serial) {
        sampler_resolved[index] = 1;
        if (sampler == VK_NULL_HANDLE) {
            sampler = *resources.fallback_sampler;
            serial = 0;
        }
        Touch(sampler_touched, touched_samplers, index);
        VkDescriptorImageInfo& info = sampler_infos[index];
        if (info.sampler == sampler && sampler_serial[index] == serial) {
            return;
        }
        info.sampler = sampler;
        sampler_serial[index] = serial;
        ++sampler_version[index];
        ++desired_serial;
    }

    /// Ends a resolve: unresolved elements hold null; `images` is the number of distinct images
    /// the resolved elements keep in use (heartbeat).
    void EndResolve(u32 images) {
        u32 views_live = 0;
        for (const u32 index : requested_views) {
            const u8 resolved = view_resolved[index];
            views_live += resolved != 0 ? 1u : 0u;
            for (size_t slot = 0; slot < POOL_VIEW_TYPES.size(); ++slot) {
                if (((desired_slots >> slot) & 1) && !((resolved >> slot) & 1)) {
                    PutView(index, slot, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, 0);
                }
            }
        }
        u32 samplers_live = 0;
        for (const u32 index : requested_samplers) {
            if (sampler_resolved[index] == 0) {
                SetSampler(index, VK_NULL_HANDLE, 0);
            }
            samplers_live += sampler_serial[index] != 0 ? 1u : 0u;
        }
        // Which elements name which texture: a released serial invalidates exactly those.
        live_views.clear();
        for (const u32 index : requested_views) {
            if (view_resolved[index] != 0 && view_serial[index] != 0) {
                live_views[view_serial[index]].push_back(index);
            }
        }
        stats.views_live = views_live;
        stats.samplers_live = samplers_live;
        stats.images_live = images;
    }

    /// The image of element `index` (every slot that holds it) is in `image_layout` now.
    void UpdateLayout(u32 index, u32 slot, VkImageLayout image_layout) {
        VkDescriptorImageInfo& info = view_infos[slot][index];
        if (info.imageLayout == image_layout) {
            return;
        }
        info.imageLayout = image_layout;
        ++view_version[index];
        ++desired_serial;
        ++stats.layout_updates;
    }

    /// Pool images that were render targets of the draw being recorded (heartbeat).
    void NoteTargetOfDraw(u32 images) noexcept {
        stats.target_of_draw += images;
    }

    /// Time the draw context spent in pool planning (heartbeat: ms per second).
    void AddPlanTime(std::chrono::nanoseconds time) noexcept {
        stats.plan_ns += static_cast<u64>(time.count());
    }

    /// An image the desired contents name and the layout their descriptors state.
    struct HeldImage {
        std::weak_ptr<Storage> storage;
        VkImageLayout layout;
    };

    /// The set to bind as set 1 for the draw being recorded in the flight numbered `seq`; null
    /// when nothing can be bound (the draw is skipped). `images`: the images the desired
    /// contents name, with their layouts; `keepalive`: the flight's (one-off sets).
    [[nodiscard]] VkDescriptorSet Commit(u64 seq, std::span<const HeldImage> images,
                                         std::vector<std::shared_ptr<void>>& keepalive) {
        if (oneoff_seq != seq) {
            oneoff.reset(); // the flight that holds it keeps it alive
            oneoff_seq = seq;
            oneoffs_in_flight = 0;
        }
        SetState* bind = nullptr;
        if (current != nullptr && Matches(*current)) {
            bind = current;
        } else if (oneoff && Matches(*oneoff)) {
            bind = oneoff.get();
        } else if (SetState* const state = AcquireSet()) {
            Write(*state, images);
            current = state;
            bind = state;
        } else if (oneoffs_in_flight < MAX_ONEOFF_PER_FLIGHT) {
            // Every set is read by recorded commands: a set of its own for the rest of this
            // flight, freed with it.
            std::unique_ptr<SetState> made = NewSet();
            if (made) {
                oneoff = std::shared_ptr<SetState>(std::move(made));
                ++oneoffs_in_flight;
                ++stats.oneoff_sets;
                Write(*oneoff, images);
                keepalive.push_back(oneoff);
                bind = oneoff.get();
            }
        }
        if (bind == nullptr) {
            if (current != nullptr && Bindable(*current) && InStatedLayouts(*current)) {
                // Bound as it is: it names no released view or sampler and every image it names
                // is in the layout it states; the changes reach a later draw.
                bind = current;
                ++stats.stale_binds;
            } else {
                ++stats.no_set;
                Heartbeat();
                return VK_NULL_HANDLE;
            }
        }
        if (!bind->bound || bind->noted_seq != seq) {
            // What the set names is used by this flight: textures and samplers are not destroyed,
            // images not released (Context::Impl::ReleaseStorageImage), before it completed. A set
            // names only elements marked within the aging window (or, stale, until shortly
            // before), so superseded images still age out (ReleaseSupersededImages).
            const u64 presents = impl.stats.presents.load(std::memory_order_relaxed);
            for (const HeldImage& held : bind->held_images) {
                if (const std::shared_ptr<Storage> storage = held.storage.lock()) {
                    storage->last_used_seq = (std::max)(storage->last_used_seq, seq);
                    storage->last_used_present = (std::max)(storage->last_used_present, presents);
                }
            }
            for (const u64 serial : bind->held_views) {
                impl.NoteUse(serial, seq);
            }
            for (const u64 serial : bind->held_samplers) {
                impl.NoteUse(serial, seq);
            }
            bind->noted_seq = seq;
        }
        bind->bound = true;
        bind->last_seq = (std::max)(bind->last_seq, seq);
        ++stats.draws;
        Heartbeat();
        return bind->set;
    }

private:
    struct SetState {
        vk::DescriptorPool pool;
        VkDescriptorSet set{VK_NULL_HANDLE};
        bool written{};
        bool bound{};
        /// Names a view or sampler that was released since it was written: never bound before
        /// it is written again.
        bool poisoned{};
        u64 last_seq{};
        u64 noted_seq{};
        u32 slots{};
        u64 synced_serial{};
        std::vector<u32> view_stamp;    ///< view_version of each element when written
        std::vector<u32> sampler_stamp; ///< sampler_version of each element when written
        std::unordered_set<u64> held_views;    ///< texture serials the set names
        std::unordered_set<u64> held_samplers; ///< sampler serials the set names
        std::vector<HeldImage> held_images;    ///< images the set names, layouts it states
    };

    struct DepthDummy {
        vk::Image image;
        vk::ImageView view;
    };

    static constexpr VkDeviceSize FEEDBACK_SIZE{Shader::TEXTURE_POOL_FEEDBACK_WORDS * sizeof(u32)};
    /// nvn-hle: Ocarina of Time registers its shadow maps at new texture ids every frame (about
    /// 120 a frame); an element must be marked again within HISTORY_WINDOWS windows of this many
    /// presents to stay resolved, so superseded ids drop out within ~16 presents.
    static constexpr u64 DEFAULT_WINDOW_PRESENTS{8};
    static constexpr size_t HISTORY_WINDOWS{2};
    static constexpr u32 IMAGE_WORDS{POOL_VIEW_CAPACITY / 32};
    static constexpr u32 SAMPLER_WORDS{POOL_SAMPLER_CAPACITY / 32};
    /// D16_UNORM: sampled images with linear filtering are required for it (D32_SFLOAT does not
    /// guarantee linear filtering); still checked, the colour dummy otherwise.
    static constexpr VkFormat DEPTH_DUMMY_FORMAT{VK_FORMAT_D16_UNORM};
    static constexpr u32 MAX_ONEOFF_PER_FLIGHT{4};

    [[nodiscard]] VkDescriptorImageInfo NullSampler() const {
        return VkDescriptorImageInfo{
            .sampler = *resources.fallback_sampler,
            .imageView = VK_NULL_HANDLE,
            .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
    }

    /// The null view of a view array: a depth image holding 1.0 (a depth-compare lookup of it
    /// is defined, a plain lookup reads 1.0), or the replay's colour dummy where no depth image
    /// can be made (3D and 1D types, or the format unsupported). The null sampler stays the
    /// non-compare fallback: a compare sampler would make plain lookups undefined instead, and
    /// a guest sampler is resolved after its first mark.
    [[nodiscard]] VkDescriptorImageInfo NullView(size_t slot) {
        VkImageView& dummy = null_views[slot];
        if (dummy == VK_NULL_HANDLE) {
            dummy = depth_dummies[slot].view ? *depth_dummies[slot].view
                                             : resources.DummyView(PoolViewType(slot));
        }
        return VkDescriptorImageInfo{
            .sampler = VK_NULL_HANDLE,
            .imageView = dummy,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
    }

    void CreateDepthDummies() {
        const bool depth_format_ok = device.IsFormatSupported(
            DEPTH_DUMMY_FORMAT,
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                VK_FORMAT_FEATURE_TRANSFER_DST_BIT,
            FormatType::Optimal);
        for (size_t slot = 0; slot < POOL_VIEW_TYPES.size(); ++slot) {
            const VkImageViewType view_type = PoolViewType(slot);
            // 2D, 2D array, cube, cube array only: no 3D depth images, 1D depth images are not
            // guaranteed.
            if (view_type == VK_IMAGE_VIEW_TYPE_3D || view_type == VK_IMAGE_VIEW_TYPE_1D ||
                view_type == VK_IMAGE_VIEW_TYPE_1D_ARRAY || !depth_format_ok) {
                continue;
            }
            const bool cube =
                view_type == VK_IMAGE_VIEW_TYPE_CUBE || view_type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
            const bool one_d =
                view_type == VK_IMAGE_VIEW_TYPE_1D || view_type == VK_IMAGE_VIEW_TYPE_1D_ARRAY;
            const u32 layers = cube ? 6u
                               : (view_type == VK_IMAGE_VIEW_TYPE_1D_ARRAY ||
                                  view_type == VK_IMAGE_VIEW_TYPE_2D_ARRAY)
                                   ? 2u
                                   : 1u;
            DepthDummy& dummy = depth_dummies[slot];
            try {
                dummy.image = impl.allocator->CreateImage({
                    .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = cube ? VkImageCreateFlags{VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT}
                                  : VkImageCreateFlags{0},
                    .imageType = one_d ? VK_IMAGE_TYPE_1D : VK_IMAGE_TYPE_2D,
                    .format = DEPTH_DUMMY_FORMAT,
                    .extent = {1, 1, 1},
                    .mipLevels = 1,
                    .arrayLayers = layers,
                    .samples = VK_SAMPLE_COUNT_1_BIT,
                    .tiling = VK_IMAGE_TILING_OPTIMAL,
                    .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                    .queueFamilyIndexCount = 0,
                    .pQueueFamilyIndices = nullptr,
                    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                });
                dummy.view = device.GetLogical().CreateImageView({
                    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .image = *dummy.image,
                    .viewType = view_type,
                    .format = DEPTH_DUMMY_FORMAT,
                    .components{},
                    .subresourceRange{
                        .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = (view_type == VK_IMAGE_VIEW_TYPE_1D ||
                                       view_type == VK_IMAGE_VIEW_TYPE_2D)
                                          ? 1u
                                          : layers,
                    },
                });
                // Depth 1.0 in every layer (copied: the dispatch has no depth clear of images).
                vk::Buffer ones = impl.CreateHostBuffer(sizeof(u16) * 8,
                                                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                        MemoryUsage::Upload);
                std::array<u16, 8> one_values{};
                one_values.fill(0xffff);
                std::memcpy(ones.Mapped().data(), one_values.data(), sizeof(one_values));
                ones.Flush();
                const VkImage image = *dummy.image;
                const VkBuffer source = *ones;
                impl.ImmediateSubmit([image, source, layers](vk::CommandBuffer cmd) {
                    const VkImageSubresourceRange range{
                        .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = layers,
                    };
                    VkImageMemoryBarrier barrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = 0,
                        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = image,
                        .subresourceRange = range,
                    };
                    cmd.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, barrier);
                    const VkBufferImageCopy copy{
                        .bufferOffset = 0,
                        .bufferRowLength = 0,
                        .bufferImageHeight = 0,
                        .imageSubresource{
                            .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
                            .mipLevel = 0,
                            .baseArrayLayer = 0,
                            .layerCount = layers,
                        },
                        .imageOffset = {0, 0, 0},
                        .imageExtent = {1, 1, 1},
                    };
                    cmd.CopyBufferToImage(source, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                          copy);
                    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, barrier);
                });
            } catch (const vk::Exception& exception) {
                LOG_WARNING(Render_Vulkan,
                            "[nvn] texture pool: depth null view (type {}) not created, the "
                            "colour dummy is used: {}",
                            static_cast<int>(view_type), exception.what());
                dummy.view = vk::ImageView{};
                dummy.image = vk::Image{};
            }
        }
    }

    static void Touch(std::vector<u8>& touched, std::vector<u32>& list, u32 index) {
        if (touched[index] == 0) {
            touched[index] = 1;
            list.push_back(index);
        }
    }

    void PutView(u32 index, size_t slot, VkImageView view, VkImageLayout image_layout,
                 u64 serial) {
        if (view == VK_NULL_HANDLE) {
            const VkDescriptorImageInfo null_view = NullView(slot);
            view = null_view.imageView;
            image_layout = null_view.imageLayout;
            serial = 0;
        }
        Touch(view_touched, touched_views, index);
        VkDescriptorImageInfo& info = view_infos[slot][index];
        if (info.imageView == view && info.imageLayout == image_layout &&
            (serial == 0 || view_serial[index] == serial)) {
            return;
        }
        info.imageView = view;
        info.imageLayout = image_layout;
        if (serial != 0) {
            view_serial[index] = serial;
        }
        ++view_version[index];
        ++desired_serial;
    }

    void NullAllSlots(u32 index) {
        for (size_t slot = 0; slot < POOL_VIEW_TYPES.size(); ++slot) {
            if ((desired_slots >> slot) & 1) {
                PutView(index, slot, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, 0);
            }
        }
    }

    /// Element `index` names a released view: its desired value is invalid until resolved again
    /// (a placeholder value no PutView can match, so the next resolve writes it anew).
    void InvalidateView(u32 index) {
        for (size_t slot = 0; slot < POOL_VIEW_TYPES.size(); ++slot) {
            if ((desired_slots >> slot) & 1) {
                view_infos[slot][index].imageView = VK_NULL_HANDLE;
            }
        }
        view_serial[index] = 0;
        ++view_version[index];
        ++desired_serial;
        if (!((seen_words[index / 32] >> (index % 32)) & 1)) {
            NullAllSlots(index); // not requested: no resolve fills it, it is null now
        }
    }

    /// Textures whose views the replay released (NoteReleased, from DrawResources) and
    /// released samplers: the sets that name one are poisoned, the elements resolved again.
    void ProcessReleases() {
        if (released_overflow) {
            released_overflow = false;
            released.clear();
            ForEachSet([](SetState& state) { state.poisoned = true; });
            for (const u32 index : touched_views) {
                InvalidateView(index);
            }
            live_views.clear();
            resolved_textures.clear();
            ++generation;
            ++timed_generation; // command-driven (record-ahead)
        }
        if (!released.empty()) {
            for (const u64 serial : released) {
                ForEachSet([serial](SetState& state) {
                    if (state.held_views.contains(serial)) {
                        state.poisoned = true;
                    }
                });
                if (resolved_textures.erase(serial) != 0) {
                    ++generation;
                    ++timed_generation; // command-driven (record-ahead)
                }
                const auto it = live_views.find(serial);
                if (it == live_views.end()) {
                    continue;
                }
                for (const u32 index : it->second) {
                    InvalidateView(index);
                }
                live_views.erase(it);
                ++generation;
                ++timed_generation; // command-driven (record-ahead)
            }
            released.clear();
        }
        if (resources.samplers_epoch != seen_samplers_epoch) {
            seen_samplers_epoch = resources.samplers_epoch;
            ForEachSet([](SetState& state) {
                if (!state.held_samplers.empty()) {
                    state.poisoned = true;
                }
            });
            for (const u32 index : touched_samplers) {
                sampler_infos[index].sampler = VK_NULL_HANDLE; // placeholder until resolved
                sampler_serial[index] = ~u64{0};
                ++sampler_version[index];
            }
            ++desired_serial;
            ++generation;
            ++timed_generation; // command-driven (record-ahead)
        }
    }

    /// Adds the elements shaders marked since the last scan to the requested lists; true when
    /// any was added. The last view element stays null: shaders clamp out-of-range indices to it.
    bool ScanFeedback() {
        if (!feedback.IsHostCoherent()) {
            feedback.Invalidate();
        }
        const auto* const words{reinterpret_cast<const volatile u32*>(feedback.Mapped().data())};
        bool added = false;
        const auto scan = [&](u32 first_word, u32 num_words, u32 limit, std::vector<u32>& out,
                              bool images) {
            for (u32 w = 0; w < num_words; ++w) {
                u32 fresh{words[first_word + w] & ~seen_words[first_word + w]};
                if (fresh == 0) {
                    continue;
                }
                seen_words[first_word + w] |= fresh;
                while (fresh != 0) {
                    const u32 index{w * 32 + static_cast<u32>(std::countr_zero(fresh))};
                    fresh &= fresh - 1;
                    if (index >= limit) {
                        continue;
                    }
                    out.push_back(index);
                    added = true;
                    if (stats.logged_marks < 32) {
                        ++stats.logged_marks;
                        LOG_INFO(Render_Vulkan, "[nvn] texture pool: {} {} marked (guest id {})",
                                 images ? "image" : "sampler", index,
                                 (images ? bases[0] : bases[1]) + index);
                    }
                }
            }
        };
        scan(0, IMAGE_WORDS, POOL_VIEW_CAPACITY - 1, requested_views, true);
        scan(Shader::TEXTURE_POOL_FEEDBACK_SAMPLER_BASE / 32, SAMPLER_WORDS, POOL_SAMPLER_CAPACITY,
             requested_samplers, false);
        return added;
    }

    /// Each window of `window_presents` presents the image marks are copied into a history ring
    /// and cleared (a shader sets a bit again on its next lookup), and elements no shader marked
    /// in the last HISTORY_WINDOWS windows are dropped: null again, no longer resolved, kept in
    /// layout or noted as used. EDEN_POOL_MARK_AGING=0 keeps every mark; EDEN_NVN_POOL_MARK_WINDOW
    /// sets the window. True when any element was dropped.
    bool AgeMarks(u64 frame) {
        if (frame < window_start + window_presents) {
            return false;
        }
        window_start = frame;
        RetryDead();
        if (!aging_enabled) {
            return false;
        }
        auto* const words{reinterpret_cast<volatile u32*>(feedback.Mapped().data())};
        std::vector<u32>& snapshot{mark_history[history_next]};
        snapshot.resize(IMAGE_WORDS);
        for (u32 w = 0; w < IMAGE_WORDS; ++w) {
            snapshot[w] = words[w];
            words[w] = 0;
        }
        if (!feedback.IsHostCoherent()) {
            feedback.Flush();
        }
        history_next = (history_next + 1) % HISTORY_WINDOWS;
        history_filled = (std::min)(history_filled + 1, HISTORY_WINDOWS);
        if (history_filled < HISTORY_WINDOWS) {
            return false;
        }
        recent_marks.assign(IMAGE_WORDS, 0);
        for (const std::vector<u32>& past : mark_history) {
            for (u32 w = 0; w < IMAGE_WORDS; ++w) {
                recent_marks[w] |= past[w];
            }
        }
        size_t dropped = 0;
        std::erase_if(requested_views, [&](u32 index) {
            if ((recent_marks[index / 32] >> (index % 32)) & 1) {
                return false;
            }
            seen_words[index / 32] &= ~(1u << (index % 32)); // A new mark is taken in again
            NullAllSlots(index);
            ++dropped;
            return true;
        });
        if (dropped == 0) {
            return false;
        }
        stats.dropped_aged += dropped;
        if (stats.drop_logs < 16) {
            ++stats.drop_logs;
            LOG_INFO(Render_Vulkan,
                     "[nvn] texture pool: dropped {} images no shader sampled in {} presents ({} "
                     "still marked)",
                     dropped, HISTORY_WINDOWS * window_presents, requested_views.size());
        }
        return true;
    }

    /// Dead ids are taken in again by their next mark (the id may name a texture by now).
    void RetryDead() {
        for (const u32 index : dead_views) {
            seen_words[index / 32] &= ~(1u << (index % 32));
        }
        dead_views.clear();
    }

    template <typename Func>
    void ForEachSet(Func&& func) {
        for (const auto& state : sets) {
            func(*state);
        }
        if (oneoff) {
            func(*oneoff);
        }
    }

    [[nodiscard]] bool Bindable(const SetState& state) const noexcept {
        return state.written && !state.poisoned && (state.slots & desired_slots) == desired_slots;
    }

    [[nodiscard]] bool Matches(const SetState& state) const noexcept {
        return Bindable(state) && state.slots == desired_slots &&
               state.synced_serial == desired_serial;
    }

    /// Every image `state` names is alive and in the layout the set states for it.
    [[nodiscard]] bool InStatedLayouts(const SetState& state) const {
        for (const HeldImage& held : state.held_images) {
            const std::shared_ptr<Storage> storage = held.storage.lock();
            if (!storage || !storage->image || storage->current_layout != held.layout) {
                return false;
            }
        }
        return true;
    }

    /// A set no recorded command reads (its last use completed, or never bound): of those, the
    /// one that needs the fewest writes (an unpoisoned set with the newest contents); a new one
    /// while there is room; else null. Never waits.
    [[nodiscard]] SetState* AcquireSet() {
        const u64 completed = impl.completed_seq.load(std::memory_order_acquire);
        SetState* best = nullptr;
        for (const auto& state : sets) {
            if (state->bound && state->last_seq > completed) {
                continue;
            }
            if (best == nullptr || (best->poisoned && !state->poisoned) ||
                (best->poisoned == state->poisoned && state->synced_serial > best->synced_serial)) {
                best = state.get();
            }
        }
        if (best != nullptr) {
            return best;
        }
        if (sets.size() < MAX_SETS) {
            return CreateSet();
        }
        return nullptr;
    }

    [[nodiscard]] SetState* CreateSet() {
        std::unique_ptr<SetState> state = NewSet();
        if (!state) {
            return nullptr;
        }
        sets.push_back(std::move(state));
        return sets.back().get();
    }

    [[nodiscard]] std::unique_ptr<SetState> NewSet() {
        auto state = std::make_unique<SetState>();
        const std::array<VkDescriptorPoolSize, 3> sizes{{
            {VK_DESCRIPTOR_TYPE_SAMPLER, POOL_SAMPLER_CAPACITY},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
             POOL_VIEW_CAPACITY * static_cast<u32>(POOL_VIEW_TYPES.size())},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
        }};
        try {
            state->pool = device.GetLogical().CreateDescriptorPool({
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .maxSets = 1,
                .poolSizeCount = static_cast<u32>(sizes.size()),
                .pPoolSizes = sizes.data(),
            });
        } catch (const vk::Exception& exception) {
            LOG_ERROR(Render_Vulkan, "[nvn] texture pool: descriptor pool creation failed: {}",
                      exception.what());
            return nullptr;
        }
        const VkDescriptorSetLayout set_layout = *layout;
        const VkDescriptorSetAllocateInfo allocate_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = *state->pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &set_layout,
        };
        if (impl.dld.vkAllocateDescriptorSets(impl.logical, &allocate_info, &state->set) !=
            VK_SUCCESS) {
            LOG_ERROR(Render_Vulkan, "[nvn] texture pool: descriptor set allocation failed");
            return nullptr;
        }
        state->view_stamp.assign(POOL_VIEW_CAPACITY, 0);
        state->sampler_stamp.assign(POOL_SAMPLER_CAPACITY, 0);
        return state;
    }

    /// Makes `state` (read by no recorded command) hold the desired elements.
    void Write(SetState& state, std::span<const HeldImage> images) {
        writes.clear();
        const auto write = [&](u32 binding, u32 element, u32 count, VkDescriptorType type,
                               const VkDescriptorImageInfo* infos) {
            writes.push_back({
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = state.set,
                .dstBinding = binding,
                .dstArrayElement = element,
                .descriptorCount = count,
                .descriptorType = type,
                .pImageInfo = infos,
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            });
        };
        if (!state.written || state.slots != desired_slots) {
            ++stats.full_writes;
            write(Shader::TEXTURE_POOL_SAMPLER_BINDING, 0, POOL_SAMPLER_CAPACITY,
                  VK_DESCRIPTOR_TYPE_SAMPLER, sampler_infos.data());
            for (size_t slot = 0; slot < POOL_VIEW_TYPES.size(); ++slot) {
                if ((desired_slots >> slot) & 1) {
                    write(Shader::TexturePoolBinding(POOL_VIEW_TYPES[slot], false), 0,
                          POOL_VIEW_CAPACITY, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                          view_infos[slot].data());
                }
            }
            state.view_stamp = view_version;
            state.sampler_stamp = sampler_version;
        } else {
            ++stats.partial_writes;
            for (const u32 index : touched_views) {
                if (state.view_stamp[index] == view_version[index]) {
                    continue;
                }
                state.view_stamp[index] = view_version[index];
                ++stats.partial_elements;
                for (size_t slot = 0; slot < POOL_VIEW_TYPES.size(); ++slot) {
                    if ((desired_slots >> slot) & 1) {
                        write(Shader::TexturePoolBinding(POOL_VIEW_TYPES[slot], false), index, 1,
                              VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &view_infos[slot][index]);
                    }
                }
            }
            for (const u32 index : touched_samplers) {
                if (state.sampler_stamp[index] == sampler_version[index]) {
                    continue;
                }
                state.sampler_stamp[index] = sampler_version[index];
                ++stats.partial_elements;
                write(Shader::TEXTURE_POOL_SAMPLER_BINDING, index, 1, VK_DESCRIPTOR_TYPE_SAMPLER,
                      &sampler_infos[index]);
            }
        }
        const VkDescriptorBufferInfo feedback_info{
            .buffer = *feedback,
            .offset = 0,
            .range = FEEDBACK_SIZE,
        };
        if (!state.written) {
            writes.push_back({
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = state.set,
                .dstBinding = Shader::TEXTURE_POOL_FEEDBACK_BINDING,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = nullptr,
                .pBufferInfo = &feedback_info,
                .pTexelBufferView = nullptr,
            });
        }
        if (!writes.empty()) {
            impl.dld.vkUpdateDescriptorSets(impl.logical, static_cast<u32>(writes.size()),
                                            writes.data(), 0, nullptr);
        }
        state.written = true;
        state.poisoned = false;
        state.slots = desired_slots;
        state.synced_serial = desired_serial;
        state.held_views.clear();
        for (const auto& [serial, indices] : live_views) {
            state.held_views.insert(serial);
        }
        state.held_images.assign(images.begin(), images.end());
        state.held_samplers.clear();
        for (const u32 index : requested_samplers) {
            if (sampler_serial[index] != 0 && sampler_serial[index] != ~u64{0}) {
                state.held_samplers.insert(sampler_serial[index]);
            }
        }
    }

    /// `[nvn] texture pool: ...` once on first use and then every ten seconds of pool draws.
    void Heartbeat() {
        const auto now = std::chrono::steady_clock::now();
        const bool first = !stats.logged_first;
        if (!first && now - stats.logged_at < std::chrono::seconds{10}) {
            return;
        }
        const double seconds =
            first ? 1.0
                  : std::chrono::duration<double>(now - stats.logged_at).count();
        const double plan_ms_per_s = static_cast<double>(stats.plan_ns) / 1e6 / seconds;
        stats.plan_ns = 0;
        stats.logged_first = true;
        stats.logged_at = now;
        const u64 completed = impl.completed_seq.load(std::memory_order_acquire);
        size_t in_use = 0;
        for (const auto& state : sets) {
            in_use += state->bound && state->last_seq > completed ? 1 : 0;
        }
        LOG_INFO(Render_Vulkan,
                 "[nvn] texture pool (queue {}): {} views / {} samplers resolved, {} / {} marked "
                 "live, {} images kept in use, {} dropped by aging, {} dropped dead, planning "
                 "{:.2f} ms/s, {} pool draws, {} resolves, sets {} ({} in use), {} one-off sets, "
                 "writes {} full / {} partial ({} elements), {} layout updates, {} stale binds, "
                 "{} draws without a set, {} target-of-draw images, {} pool base changes{}",
                 queue, stats.views_live, stats.samplers_live, requested_views.size(),
                 requested_samplers.size(), stats.images_live, stats.dropped_aged,
                 stats.dropped_dead, plan_ms_per_s, stats.draws, stats.resolves, sets.size(),
                 in_use, stats.oneoff_sets, stats.full_writes, stats.partial_writes, stats.partial_elements,
                 stats.layout_updates, stats.stale_binds, stats.no_set, stats.target_of_draw,
                 stats.base_changes,
                 first ? " (first use)" : "");
    }

    Context::Impl& impl;
    DrawResources& resources;
    const Device& device;
    u32 queue;
    bool ok{};
    bool aging_enabled{true};
    u64 window_presents{DEFAULT_WINDOW_PRESENTS};
    vk::DescriptorSetLayout layout;
    vk::Buffer feedback;
    std::array<DepthDummy, POOL_VIEW_TYPES.size()> depth_dummies;
    std::array<VkImageView, POOL_VIEW_TYPES.size()> null_views{};
    std::vector<std::unique_ptr<SetState>> sets;
    SetState* current{};
    std::vector<VkWriteDescriptorSet> writes;

    // What the set should hold now.
    u64 generation{1};     ///< Moves when the context has to resolve again
    u64 timed_generation{}; ///< ... the moves TimedGeneration counts (record-ahead)
    u64 scanned_seq{~u64{0}};
    std::array<u32, 2> bases{}; ///< texture, sampler pool base of the last draw
    u32 desired_slots{};
    u64 desired_serial{1}; ///< Counts the changes to the desired elements
    std::array<std::vector<VkDescriptorImageInfo>, POOL_VIEW_TYPES.size()> view_infos;
    std::vector<VkDescriptorImageInfo> sampler_infos;
    std::vector<u32> view_version;
    std::vector<u64> view_serial;
    std::vector<u8> view_touched;
    std::vector<u8> view_resolved; ///< POOL_VIEW_TYPES positions set by the last resolve
    std::vector<u32> touched_views; ///< Elements that ever held something other than null
    std::unordered_map<u64, std::vector<u32>> live_views; ///< texture serial -> elements
    std::unordered_set<u64> resolved_textures; ///< every texture the last resolve named
    std::vector<u32> sampler_version;
    std::vector<u64> sampler_serial;
    std::vector<u8> sampler_touched;
    std::vector<u8> sampler_resolved;
    std::vector<u32> touched_samplers;
    std::vector<u32> requested_views;    ///< Texture pool elements shaders marked
    std::vector<u32> requested_samplers; ///< Sampler pool elements shaders marked
    std::vector<u32> seen_words;         ///< Feedback bits already taken into the lists
    std::vector<u32> dead_views;         ///< Dropped as dead since the last window (RetryDead)
    std::vector<u64> released;           ///< NoteReleased since the last draw
    bool released_overflow{false};
    std::shared_ptr<SetState> oneoff; ///< One-off set of the flight `oneoff_seq`
    u64 oneoff_seq{~u64{0}};
    u32 oneoffs_in_flight{};
    u64 seen_samplers_epoch{};

    // Mark aging.
    u64 window_start{};
    std::array<std::vector<u32>, HISTORY_WINDOWS> mark_history;
    size_t history_next{};
    size_t history_filled{};
    std::vector<u32> recent_marks;

    struct {
        u64 draws{};
        u64 resolves{};
        u64 full_writes{};
        u64 partial_writes{};
        u64 partial_elements{};
        u64 layout_updates{};
        u64 stale_binds{};
        u64 oneoff_sets{};
        u64 target_of_draw{};
        u64 no_set{};
        u64 dropped_aged{};
        u64 dropped_dead{};
        u64 base_changes{};
        u64 plan_ns{};
        u32 views_live{};
        u32 samplers_live{};
        u32 images_live{};
        u32 logged_marks{};
        u32 drop_logs{};
        bool logged_first{};
        std::chrono::steady_clock::time_point logged_at{};
    } stats;
};

} // namespace Vulkan::Nvn
