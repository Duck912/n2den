// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/frame_snapshot.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/compact_texture_table.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

/// The texture pool descriptor set (Shader::TEXTURE_POOL_DESCRIPTOR_SET): the guest's sampler
/// table and texture table as descriptor arrays a shader indexes with the two halves of a
/// texture handle (handle >> 20 and handle & 0xfffff), one view array per image type, plus the
/// feedback buffer in which shaders mark the elements they look up.
///
/// Only marked elements are resolved. Switch 2 games keep tens of thousands of entries in the
/// texture table (Ocarina of Time: 33,068) and a pool draw samples a handful of them; resolving
/// the whole table would create and upload an image for every texture the game has loaded, on
/// every draw. An element is null (the null view, the null sampler) until a shader has marked
/// it and the host has seen the mark, so its first lookups read zero; from then on it is
/// resolved with every pool draw, like a per-draw texture (PrepareImageView keeps it current).
///
/// The set is laid out the same for every pipeline, so one set serves all of them. A set that
/// commands already recorded may read is never rewritten: a change goes into another set (only
/// the elements that set holds differently are written), and the old set is retired with the
/// scheduler tick that last used it.
class TexturePoolSet {
public:
    static constexpr u32 VIEW_CAPACITY = Shader::TEXTURE_POOL_VIEW_CAPACITY; ///< Per view array
    static constexpr u32 SAMPLER_CAPACITY = 4096; ///< A handle holds a 12-bit sampler index
    static constexpr u32 MAX_SETS = 8;

    /// View arrays the set holds. Multisampled arrays are not provided yet.
    static constexpr std::array VIEW_TYPES{
        Shader::TextureType::Color1D,   Shader::TextureType::ColorArray1D,
        Shader::TextureType::Color2D,   Shader::TextureType::ColorArray2D,
        Shader::TextureType::Color3D,   Shader::TextureType::ColorCube,
        Shader::TextureType::ColorArrayCube,
    };

    explicit TexturePoolSet(const Device& device_, Scheduler& scheduler_)
        : device{device_}, scheduler{scheduler_} {
        // The device's limits count every set of a pipeline layout: leave half to set 0.
        const u64 view_limit{u64{device.GetMaxDescriptorSetSampledImages()} / 2};
        const u64 sampler_limit{u64{device.GetMaxDescriptorSetSamplers()} / 2};
        supported = view_limit >= u64{VIEW_CAPACITY} * VIEW_TYPES.size() &&
                    sampler_limit >= SAMPLER_CAPACITY;
        if (const char* const env = std::getenv("EDEN_NX2_TEXTURE_POOL_FEEDBACK")) {
            feedback_enabled = std::string_view{env} != "0";
        }
        CreateLayout();
        CreateFeedbackBuffer();
        LOG_INFO(Render_Vulkan,
                 "Texture pool set: {} views per image type, {} samplers, feedback {} "
                 "(device limits {} sampled images, {} samplers per set)",
                 VIEW_CAPACITY, SAMPLER_CAPACITY, feedback_enabled ? "on" : "OFF (all null)",
                 device.GetMaxDescriptorSetSampledImages(), device.GetMaxDescriptorSetSamplers());
    }

    /// Whether a program's pool lookups can be served; a pipeline that fails this is not built.
    [[nodiscard]] static bool Supports(const Shader::Info& info) noexcept {
        for (const auto& desc : info.texture_pool_descriptors) {
            if (desc.is_multisample || SlotOf(desc.type) == VIEW_TYPES.size()) {
                return false;
            }
        }
        return true;
    }

    /// Registers the view arrays a pipeline stage reads; only those are filled. Pipelines are
    /// built on worker threads, hence the atomic.
    void Use(const Shader::Info& info) {
        for (const auto& desc : info.texture_pool_descriptors) {
            used_slots_shared.fetch_or(1U << SlotOf(desc.type));
        }
    }

    /// The layout of the set, for descriptor set 1 of a pipeline layout.
    [[nodiscard]] VkDescriptorSetLayout Layout() const noexcept {
        return *layout;
    }

    /// Brings the set up to date with the guest's tables (limits are the last valid indices)
    /// and returns the set to bind for the draw being recorded.
    [[nodiscard]] VkDescriptorSet Update(TextureCache& texture_cache, u32 tic_limit, u32 tsc_limit) {
        if (!supported) {
            throw Shader::RuntimeError("Texture pool needs {} sampled images per set",
                                       u64{VIEW_CAPACITY} * VIEW_TYPES.size() * 2);
        }
        // The last view element stays null: shaders clamp out-of-range indices to it.
        if (tic_limit >= VIEW_CAPACITY - 1 || tsc_limit >= SAMPLER_CAPACITY) {
            throw Shader::RuntimeError(
                "Texture pool larger than the pool descriptor set (image limit {} of {}, sampler limit {} of {})",
                tic_limit, VIEW_CAPACITY - 1, tsc_limit, SAMPLER_CAPACITY);
        }
        if (desired_view_ids.empty()) {
            InitState(texture_cache);
        }
        ScanFeedback(texture_cache, tic_limit, tsc_limit);
        AgeMarks(texture_cache);

        const u32 used_slots{used_slots_shared.load()};
        const u64 generation{texture_cache.ViewGeneration()};
        const bool refresh_all{used_slots != desired_slots || generation != desired_generation};
        desired_slots = used_slots;
        desired_generation = generation;
        if (refresh_all) {
            // Arrays a pipeline started to use hold the null view where nothing is marked.
            for (size_t slot = 0; slot < VIEW_TYPES.size(); ++slot) {
                if (((used_slots >> slot) & 1) && view_infos[slot].empty()) {
                    view_infos[slot].assign(
                        VIEW_CAPACITY,
                        ViewInfo(texture_cache, VideoCommon::NULL_IMAGE_VIEW_ID, VIEW_TYPES[slot]));
                }
            }
        }

        // Resolve the marked elements; this also keeps their contents current. The marked elements
        // only grow in number (4,700 after twenty minutes of Ocarina of Time) and a draw changes
        // none or a few of them: the texture cache keeps what they resolve to and redoes only what
        // can have changed (TextureCache::ResolvePoolViews), and names the elements whose view
        // differs from the previous draw's, the only ones looked at here.
        const auto set_view{[&](size_t position, VideoCommon::ImageViewId id) {
            const u32 index{requested_views[position]};
            if (!refresh_all && desired_view_ids[index] == id) {
                return;
            }
            desired_view_ids[index] = id;
            ++desired_serial;
            for (size_t slot = 0; slot < VIEW_TYPES.size(); ++slot) {
                if ((used_slots >> slot) & 1) {
                    view_infos[slot][index] = ViewInfo(texture_cache, id, VIEW_TYPES[slot]);
                }
            }
        }};
        if (VideoCommon::HeapReuseEnabled() && !VideoCommon::FrameSnapshot::Part(1)) {
            // Until this draw's changes are taken, the next draw has to look at every element.
            const bool every_view{refresh_all || !views_incremental};
            views_incremental = false;
            const VideoCommon::PoolViewsResult resolved{texture_cache.ResolvePoolViews(requested_views)};
            if (every_view || resolved.all_changed) {
                for (size_t position = 0; position < resolved.ids.size(); ++position) {
                    set_view(position, resolved.ids[position]);
                }
            } else {
                for (const u32 position : resolved.changed) {
                    set_view(position, resolved.ids[position]);
                }
            }
            views_incremental = true;
        } else {
            // Every element, on every draw (EDEN_NO_HEAP_REUSE, frame snapshots).
            static thread_local std::vector<VideoCommon::ImageViewInOut> views;
            views.clear();
            for (const u32 index : requested_views) {
                views.push_back({.index = index, .blacklist = false, .id = {}});
            }
            if (!views.empty()) {
                texture_cache.FillImageViews(views, false, false);
            }
            for (size_t position = 0; position < views.size(); ++position) {
                set_view(position, views[position].id);
            }
            views_incremental = false;
        }
        for (const u32 index : requested_samplers) {
            const VideoCommon::SamplerId id{index <= tsc_limit ? texture_cache.GetSamplerId(index, false)
                                                               : VideoCommon::NULL_SAMPLER_ID};
            if (!refresh_all && desired_sampler_ids[index] == id) {
                continue;
            }
            desired_sampler_ids[index] = id;
            ++desired_serial;
            sampler_infos[index] = SamplerInfo(texture_cache, id);
        }

        if (current == nullptr || !Matches(*current)) {
            WriteNewSet();
        }
        current->tick = scheduler.CurrentTick();
        ++stats.updates;
        if (stats.updates % 16384 == 0) {
            LOG_INFO(Render_Vulkan,
                     "Texture pool: {} updates, {} images and {} samplers marked, sets written {} "
                     "full / {} partial ({} elements), {} waits, {} images dropped unused",
                     stats.updates, requested_views.size(), requested_samplers.size(),
                     stats.full_writes, stats.partial_writes, stats.partial_elements, stats.waits,
                     stats.dropped);
        }
        return current->set;
    }

private:
    struct SetState {
        VkDescriptorSet set{};
        u64 tick{};
        bool written{};
        u32 slots{};
        u64 generation{};
        u64 synced_serial{}; ///< desired_serial when the set was last made to hold the desired ids
        std::vector<VideoCommon::ImageViewId> view_ids; ///< What the set holds, by element
        std::vector<VideoCommon::SamplerId> sampler_ids;
        /// Dropped elements the set still holds a view for: written with the next partial write
        /// (the requested lists no longer name them), and the set matches nothing until then.
        std::vector<u32> pending_null;
    };

    [[nodiscard]] static constexpr size_t SlotOf(Shader::TextureType type) noexcept {
        if (type == Shader::TextureType::Color2DRect) {
            type = Shader::TextureType::Color2D;
        }
        for (size_t slot = 0; slot < VIEW_TYPES.size(); ++slot) {
            if (VIEW_TYPES[slot] == type) {
                return slot;
            }
        }
        return VIEW_TYPES.size();
    }

    [[nodiscard]] static VkDescriptorImageInfo ViewInfo(TextureCache& texture_cache,
                                                        VideoCommon::ImageViewId id,
                                                        Shader::TextureType type) {
        // A table element of another image type (or an integer format, which the float arrays
        // cannot hold) is the null view of this type: a lookup of it stays defined.
        ImageView& image_view{texture_cache.GetImageView(id)};
        VkImageView handle{image_view.Handle(type)};
        if (id != VideoCommon::NULL_IMAGE_VIEW_ID &&
            VideoCore::Surface::IsPixelFormatInteger(image_view.format)) {
            handle = VK_NULL_HANDLE;
        }
        if (handle == VK_NULL_HANDLE) {
            handle = texture_cache.GetImageView(VideoCommon::NULL_IMAGE_VIEW_ID).Handle(type);
        }
        return VkDescriptorImageInfo{
            .sampler = VK_NULL_HANDLE,
            .imageView = handle,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
    }

    [[nodiscard]] static VkDescriptorImageInfo SamplerInfo(TextureCache& texture_cache,
                                                           VideoCommon::SamplerId id) {
        // The sampler as its table entry states it. The per-view variants the per-draw path
        // picks (Sampler::HandleFor) do not exist here: view and sampler meet in the shader.
        return VkDescriptorImageInfo{
            .sampler = texture_cache.GetSampler(id).Handle(),
            .imageView = VK_NULL_HANDLE,
            .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
    }

    void InitState(TextureCache& texture_cache) {
        desired_view_ids.assign(VIEW_CAPACITY, VideoCommon::NULL_IMAGE_VIEW_ID);
        desired_sampler_ids.assign(SAMPLER_CAPACITY, VideoCommon::NULL_SAMPLER_ID);
        sampler_infos.assign(SAMPLER_CAPACITY,
                             SamplerInfo(texture_cache, VideoCommon::NULL_SAMPLER_ID));
        seen_words.assign(Shader::TEXTURE_POOL_FEEDBACK_WORDS, 0);
        desired_slots = ~0U; // Forces the first refresh, which fills the view arrays
    }

    /// Adds the elements shaders marked since the last scan to the resolved lists.
    void ScanFeedback(TextureCache& texture_cache, u32 tic_limit, u32 tsc_limit) {
        if (!feedback_enabled) {
            return;
        }
        if (!feedback.IsHostCoherent()) {
            feedback.Invalidate();
        }
        const auto* const words{reinterpret_cast<const volatile u32*>(feedback.Mapped().data())};
        const auto scan = [&](u32 first_bit, u32 count, std::vector<u32>& out, bool images) {
            const u32 first_word{first_bit / 32};
            const u32 num_words{(count + 31) / 32};
            for (u32 w = 0; w < num_words; ++w) {
                u32 fresh{words[first_word + w] & ~seen_words[first_word + w]};
                if (fresh == 0) {
                    continue;
                }
                seen_words[first_word + w] |= fresh;
                while (fresh != 0) {
                    const u32 index{w * 32 + static_cast<u32>(std::countr_zero(fresh))};
                    fresh &= fresh - 1;
                    out.push_back(index);
                    if (stats.logged_marks < 96) {
                        ++stats.logged_marks;
                        LogMark(texture_cache, index, images);
                    }
                }
            }
        };
        scan(0, tic_limit + 1, requested_views, true);
        scan(Shader::TEXTURE_POOL_FEEDBACK_SAMPLER_BASE, tsc_limit + 1, requested_samplers, false);
    }

    /// Marks are permanent on the GPU side, and a marked element is resolved, kept current and
    /// LRU-touched on every pool draw: every texture a shader ever sampled stayed resident, and
    /// walking around Ocarina of Time grew video memory past the card (25 GB on a 16 GB card,
    /// fps 35 -> 12). Each window of WINDOW_FRAMES frames the image marks are copied into a
    /// history ring and cleared (a shader sets a bit again on its next lookup of the element), and
    /// elements no shader marked in the last windows are dropped: their descriptors become null,
    /// their images are no longer touched, and the garbage collector can free them. A dropped
    /// element that is sampled again reads null until the host sees the new mark (one or two
    /// frames), as on its first use. The window count shrinks with memory pressure.
    /// EDEN_POOL_MARK_AGING=0 keeps every mark, as before.
    void AgeMarks(TextureCache& texture_cache) {
        static const bool enabled = [] {
            const char* const value = std::getenv("EDEN_POOL_MARK_AGING");
            return !(value && value[0] == '0');
        }();
        if (!enabled || !feedback_enabled) {
            return;
        }
        const u64 tick{texture_cache.FrameTick()};
        if (tick < window_start + WINDOW_FRAMES) {
            return;
        }
        window_start = tick;
        constexpr u32 image_words{Shader::TEXTURE_POOL_FEEDBACK_SAMPLER_BASE / 32};
        auto* const words{reinterpret_cast<volatile u32*>(feedback.Mapped().data())};
        std::vector<u32>& snapshot{mark_history[history_next]};
        snapshot.resize(image_words);
        for (u32 w = 0; w < image_words; ++w) {
            snapshot[w] = words[w];
            words[w] = 0;
        }
        if (!feedback.IsHostCoherent()) {
            feedback.Flush();
        }
        history_next = (history_next + 1) % HISTORY_WINDOWS;
        history_filled = (std::min)(history_filled + 1, HISTORY_WINDOWS);

        // A mark the GPU sets late lands in the next window: never look at fewer than two.
        const int pressure{texture_cache.MemoryPressure()};
        const size_t keep{pressure >= 2 ? size_t{2} : pressure == 1 ? size_t{4} : HISTORY_WINDOWS};
        if (history_filled < keep) {
            return;
        }
        recent_marks.assign(image_words, 0);
        for (size_t k = 0; k < keep; ++k) {
            const std::vector<u32>& past{
                mark_history[(history_next + HISTORY_WINDOWS - 1 - k) % HISTORY_WINDOWS]};
            for (u32 w = 0; w < image_words; ++w) {
                recent_marks[w] |= past[w];
            }
        }
        dropped_views.clear();
        std::erase_if(requested_views, [&](u32 index) {
            if ((recent_marks[index / 32] >> (index % 32)) & 1) {
                return false;
            }
            dropped_views.push_back(index);
            seen_words[index / 32] &= ~(1U << (index % 32)); // A new mark is taken in again
            return true;
        });
        if (dropped_views.empty()) {
            return;
        }
        for (const u32 index : dropped_views) {
            desired_view_ids[index] = VideoCommon::NULL_IMAGE_VIEW_ID;
            for (size_t slot = 0; slot < VIEW_TYPES.size(); ++slot) {
                if (!view_infos[slot].empty()) {
                    view_infos[slot][index] =
                        ViewInfo(texture_cache, VideoCommon::NULL_IMAGE_VIEW_ID, VIEW_TYPES[slot]);
                }
            }
        }
        ++desired_serial;
        // Every set that holds one of them must drop it before it is bound again: the garbage
        // collector may now destroy the view (sets already recorded are covered by the texture
        // cache delaying destruction for several frames).
        for (const auto& state : sets) {
            if (!state->written) {
                continue;
            }
            for (const u32 index : dropped_views) {
                if (state->view_ids[index] != VideoCommon::NULL_IMAGE_VIEW_ID) {
                    state->pending_null.push_back(index);
                }
            }
        }
        // The incremental resolver assumes the list only grows.
        texture_cache.ResetPoolViews();
        views_incremental = false;
        stats.dropped += dropped_views.size();
        if (stats.drop_logs < 32) {
            ++stats.drop_logs;
            LOG_INFO(Render_Vulkan,
                     "Texture pool: dropped {} images no shader sampled in {} frames ({} still "
                     "marked, memory pressure {})",
                     dropped_views.size(), keep * WINDOW_FRAMES, requested_views.size(), pressure);
        }
    }

    void LogMark(TextureCache& texture_cache, u32 index, bool image) {
        if (!image) {
            LOG_INFO(Render_Vulkan, "Texture pool: sampler {} marked", index);
            return;
        }
        std::array<VideoCommon::ImageViewInOut, 1> one{};
        one[0].index = index;
        texture_cache.FillImageViews(one, false, false);
        if (one[0].id == VideoCommon::NULL_IMAGE_VIEW_ID) {
            LOG_INFO(Render_Vulkan, "Texture pool: image {} marked (null entry)", index);
            return;
        }
        const ImageView& view{texture_cache.GetImageView(one[0].id)};
        LOG_INFO(Render_Vulkan, "Texture pool: image {} marked: {}x{}x{} format {} view type {}", index,
                 view.size.width, view.size.height, view.size.depth, static_cast<u32>(view.format),
                 static_cast<u32>(view.type));
    }

    /// Whether a set holds what the guest's tables resolve to now.
    [[nodiscard]] bool Matches(SetState& state) const {
        if (!state.written || state.slots != desired_slots ||
            state.generation != desired_generation || !state.pending_null.empty()) {
            return false;
        }
        // A written set holds the desired ids of every element as they were then (desired_serial
        // counts every change to them): nothing to compare while that has not moved.
        if (state.synced_serial == desired_serial) {
            return true;
        }
        for (const u32 index : requested_views) {
            if (state.view_ids[index] != desired_view_ids[index]) {
                return false;
            }
        }
        for (const u32 index : requested_samplers) {
            if (state.sampler_ids[index] != desired_sampler_ids[index]) {
                return false;
            }
        }
        state.synced_serial = desired_serial; // Changed and changed back
        return true;
    }

    void CreateLayout() {
        constexpr VkShaderStageFlags stages{VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT};
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.push_back({
            .binding = Shader::TEXTURE_POOL_SAMPLER_BINDING,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
            .descriptorCount = SAMPLER_CAPACITY,
            .stageFlags = stages,
            .pImmutableSamplers = nullptr,
        });
        for (const Shader::TextureType type : VIEW_TYPES) {
            bindings.push_back({
                .binding = Shader::TexturePoolBinding(type, false),
                .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                .descriptorCount = VIEW_CAPACITY,
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
        // Arrays of types no pipeline reads are never written.
        const std::vector<VkDescriptorBindingFlags> binding_flags(
            bindings.size(), VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
        const VkDescriptorSetLayoutBindingFlagsCreateInfo flags_ci{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
            .pNext = nullptr,
            .bindingCount = static_cast<u32>(binding_flags.size()),
            .pBindingFlags = binding_flags.data(),
        };
        layout = device.GetLogical().CreateDescriptorSetLayout({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = &flags_ci,
            .flags = 0,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        });
        const std::array<VkDescriptorPoolSize, 3> sizes{{
            {VK_DESCRIPTOR_TYPE_SAMPLER, MAX_SETS * SAMPLER_CAPACITY},
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
             MAX_SETS * VIEW_CAPACITY * static_cast<u32>(VIEW_TYPES.size())},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MAX_SETS},
        }};
        pool = device.GetLogical().CreateDescriptorPool({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .maxSets = MAX_SETS,
            .poolSizeCount = static_cast<u32>(sizes.size()),
            .pPoolSizes = sizes.data(),
        });
    }

    void CreateFeedbackBuffer() {
        const MemoryAllocator allocator{device};
        feedback = allocator.CreateBuffer(
            VkBufferCreateInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .size = FEEDBACK_SIZE,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .queueFamilyIndexCount = 0,
                .pQueueFamilyIndices = nullptr,
            },
            MemoryUsage::Download);
        if (!feedback.IsHostVisible()) {
            supported = false;
            LOG_ERROR(Render_Vulkan, "Texture pool feedback buffer is not host visible");
            return;
        }
        std::memset(feedback.Mapped().data(), 0, FEEDBACK_SIZE);
        if (!feedback.IsHostCoherent()) {
            feedback.Flush();
        }
    }

    /// A set no recorded command reads: a retired one whose tick completed, a new one while the
    /// pool has room, else the oldest retired one after waiting for its tick.
    [[nodiscard]] SetState& AcquireSet() {
        if (current != nullptr) {
            retired.push_back(current);
            current = nullptr;
        }
        for (size_t i = 0; i < retired.size(); ++i) {
            if (scheduler.IsFree(retired[i]->tick)) {
                SetState* const state{retired[i]};
                retired.erase(retired.begin() + static_cast<std::ptrdiff_t>(i));
                return *state;
            }
        }
        if (sets.size() < MAX_SETS) {
            const VkDescriptorSetLayout set_layout{*layout};
            vk::DescriptorSets allocated{pool.Allocate({
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .pNext = nullptr,
                .descriptorPool = *pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &set_layout,
            })};
            auto state{std::make_unique<SetState>()};
            state->set = allocated[0];
            allocations.push_back(std::move(allocated));
            sets.push_back(std::move(state));
            return *sets.back();
        }
        SetState* const oldest{retired.front()};
        retired.erase(retired.begin());
        ++stats.waits;
        scheduler.Flush();
        scheduler.Wait(oldest->tick);
        return *oldest;
    }

    /// Makes a set no recorded command reads hold the desired elements; it becomes current.
    void WriteNewSet() {
        SetState& state{AcquireSet()};
        current = &state;
        std::vector<VkWriteDescriptorSet> writes;
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
        const bool full{!state.written || state.slots != desired_slots};
        const bool all_marked{state.generation != desired_generation};
        if (full) {
            ++stats.full_writes;
            write(Shader::TEXTURE_POOL_SAMPLER_BINDING, 0, SAMPLER_CAPACITY,
                  VK_DESCRIPTOR_TYPE_SAMPLER, sampler_infos.data());
            for (size_t slot = 0; slot < VIEW_TYPES.size(); ++slot) {
                if ((desired_slots >> slot) & 1) {
                    write(Shader::TexturePoolBinding(VIEW_TYPES[slot], false), 0, VIEW_CAPACITY,
                          VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, view_infos[slot].data());
                }
            }
            state.view_ids = desired_view_ids;
            state.sampler_ids = desired_sampler_ids;
            state.pending_null.clear();
        } else {
            ++stats.partial_writes;
            // Dropped elements, as the desired arrays hold them now (null, or a view again if
            // re-marked since). The infos are the arrays' own: they outlive the update call.
            for (const u32 index : state.pending_null) {
                state.view_ids[index] = desired_view_ids[index];
                ++stats.partial_elements;
                for (size_t slot = 0; slot < VIEW_TYPES.size(); ++slot) {
                    if ((desired_slots >> slot) & 1) {
                        write(Shader::TexturePoolBinding(VIEW_TYPES[slot], false), index, 1,
                              VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &view_infos[slot][index]);
                    }
                }
            }
            state.pending_null.clear();
            for (const u32 index : requested_views) {
                if (!all_marked && state.view_ids[index] == desired_view_ids[index]) {
                    continue;
                }
                state.view_ids[index] = desired_view_ids[index];
                ++stats.partial_elements;
                for (size_t slot = 0; slot < VIEW_TYPES.size(); ++slot) {
                    if ((desired_slots >> slot) & 1) {
                        write(Shader::TexturePoolBinding(VIEW_TYPES[slot], false), index, 1,
                              VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, &view_infos[slot][index]);
                    }
                }
            }
            for (const u32 index : requested_samplers) {
                if (!all_marked && state.sampler_ids[index] == desired_sampler_ids[index]) {
                    continue;
                }
                state.sampler_ids[index] = desired_sampler_ids[index];
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
            device.GetLogical().UpdateDescriptorSets(writes, {});
        }
        state.written = true;
        state.slots = desired_slots;
        state.generation = desired_generation;
        state.synced_serial = desired_serial;
    }

    static constexpr VkDeviceSize FEEDBACK_SIZE{Shader::TEXTURE_POOL_FEEDBACK_WORDS * sizeof(u32)};
    static constexpr u64 WINDOW_FRAMES{30};
    static constexpr size_t HISTORY_WINDOWS{20}; ///< About 600 frames without memory pressure

    const Device& device;
    Scheduler& scheduler;
    bool supported{};
    bool feedback_enabled{true};
    vk::DescriptorSetLayout layout;
    vk::DescriptorPool pool;
    vk::Buffer feedback;
    std::vector<vk::DescriptorSets> allocations;
    std::vector<std::unique_ptr<SetState>> sets;
    std::vector<SetState*> retired;
    SetState* current{};
    std::atomic<u32> used_slots_shared{}; ///< View arrays some pipeline reads, by VIEW_TYPES position

    // What the set should hold now.
    u32 desired_slots{};
    u64 desired_generation{};
    std::vector<u32> requested_views;    ///< Texture table elements shaders marked
    std::vector<u32> requested_samplers; ///< Sampler table elements shaders marked
    std::vector<u32> seen_words;         ///< Feedback bits already taken into the lists
    u64 desired_serial{1};               ///< Counts the changes to the desired ids below
    bool views_incremental{};            ///< The last draw's views came from ResolvePoolViews
    std::vector<VideoCommon::ImageViewId> desired_view_ids;
    std::vector<VideoCommon::SamplerId> desired_sampler_ids;
    std::array<std::vector<VkDescriptorImageInfo>, VIEW_TYPES.size()> view_infos;
    std::vector<VkDescriptorImageInfo> sampler_infos;

    // Mark aging (AgeMarks).
    u64 window_start{};
    std::array<std::vector<u32>, HISTORY_WINDOWS> mark_history; ///< Image marks per window
    size_t history_next{};
    size_t history_filled{};
    std::vector<u32> recent_marks;
    std::vector<u32> dropped_views;

    struct {
        u64 updates{};
        u64 full_writes{};
        u64 partial_writes{};
        u64 partial_elements{};
        u64 waits{};
        u64 dropped{};
        u32 logged_marks{};
        u32 drop_logs{};
    } stats;
};

} // namespace Vulkan
