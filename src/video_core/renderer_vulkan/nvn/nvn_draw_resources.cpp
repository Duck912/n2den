// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 n2den Contributors
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <unordered_set>
#include <mutex>
#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <optional>

#include <fmt/format.h>

#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/scm_rev.h"
#include "common/settings.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw_resources.h"
#include "video_core/renderer_vulkan/nvn/nvn_sampler_aniso.h"
#include "video_core/renderer_vulkan/nvn/nvn_texture_pool.h"
#include "video_core/sampler_overrides.h"

namespace Vulkan::Nvn {
namespace {

constexpr u64 ARENA_CHUNK_SIZE = u64{4} << 20;
constexpr u64 DUMMY_BUFFER_SIZE = 64 * 1024;
constexpr u64 SHADOW_CHUNK_SIZE = u64{8} << 20;
/// Chunks kept for reuse: the vertex staging of a busy frame (nvn_vertex_stage.h, 20-40 MiB at
/// DK's city spot) fills a few chunks per submission, for every submission in flight, and a slow
/// GPU keeps more frames in flight. With 16 kept, a run that shared the GPU with another emulator
/// created and destroyed a chunk every third frame.
constexpr u32 MAX_FREE_SHADOWS = 32;
constexpr u64 SHADOW_KEEP_MAX = u64{32} << 20;

/// What a shadow of a storage range is bound as, and copied from and to the pool buffer; staged
/// vertex and index ranges (nvn_vertex_stage.h) are carved from the same chunks.
constexpr VkBufferUsageFlags SHADOW_USAGE =
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

constexpr VkBufferUsageFlags ARENA_USAGE =
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

VkSamplerAddressMode WrapMode(u32 nvn) {
    switch (nvn) {
    case 0: // CLAMP
    case 7: // CLAMP_TO_EDGE
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case 1:
        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case 2: // MIRROR_CLAMP
    case 3: // MIRROR_CLAMP_TO_EDGE
        return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
    case 4: // MIRROR_CLAMP_TO_BORDER
    case 5: // CLAMP_TO_BORDER
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case 6:
        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    default:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    }
}

VkCompareOp CompareOp(u32 nvn) {
    // NVNcompareFunc NEVER=1 .. ALWAYS=8 -> VK_COMPARE_OP_NEVER=0 .. ALWAYS=7
    return static_cast<VkCompareOp>(std::clamp<u32>(nvn, 1, 8) - 1);
}

/// The sampler an NVN sampler descriptor asks for, before the diagnostic overrides, the reduction
/// mode and the border colour (GetSampler adds those). `steps`: the doublings the Anisotropic
/// Filtering setting adds (nvn_sampler_aniso.h); 0 is the guest's own sampler.
VkSamplerCreateInfo SamplerInfoOf(const VC::SamplerDesc& d, s32 steps, f32 max_limit) {
    const u32 min = d.min_filter;
    const bool mipmapped = min >= 2;
    const SamplerAnisotropyFields anisotropy = SamplerAnisotropyOf(d, steps, max_limit);
    return VkSamplerCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = d.mag_filter == 1 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .minFilter = (min & 1) != 0 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .mipmapMode = min >= 4 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = WrapMode(d.wrap_s),
        .addressModeV = WrapMode(d.wrap_t),
        .addressModeW = WrapMode(d.wrap_r),
        .mipLodBias = d.lod_bias,
        .anisotropyEnable = anisotropy.enable,
        .maxAnisotropy = anisotropy.max_anisotropy,
        .compareEnable = static_cast<VkBool32>(d.compare_mode == 1),
        .compareOp = CompareOp(d.compare_func),
        .minLod = mipmapped ? d.lod_min : 0.0f,
        // A filter without mipmaps samples level 0 only.
        .maxLod = mipmapped ? d.lod_max : 0.25f,
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };
}

VkFormat DummyFormat() {
    return VK_FORMAT_R8G8B8A8_UNORM;
}

} // namespace

DrawResources::DrawResources(Context::Impl& impl_) : impl{impl_}, clears{impl_.device} {
    dummy_buffer = impl.CreateHostBuffer(DUMMY_BUFFER_SIZE, ARENA_USAGE, MemoryUsage::Upload);
    std::memset(dummy_buffer.Mapped().data(), 0, DUMMY_BUFFER_SIZE);
    dummy_buffer.Flush();
    sink_buffer =
        impl.CreateHostBuffer(SINK_SLICE_SIZE * SINK_SLICES, ARENA_USAGE, MemoryUsage::Upload);
    std::memset(sink_buffer.Mapped().data(), 0, SINK_SLICE_SIZE * SINK_SLICES);
    sink_buffer.Flush();
    fallback_sampler = impl.device.GetLogical().CreateSampler({
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_NEVER,
        .minLod = 0.0f,
        .maxLod = VK_LOD_CLAMP_NONE,
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    });
    max_sampler_anisotropy =
        (std::max)(1.0f, impl.device.GetPhysical().GetProperties().limits.maxSamplerAnisotropy);
    anisotropy_steps = CurrentAnisotropyBoostSteps();
}

PipelineCache& DrawResources::Cache() {
    if (!cache_instance) {
        std::scoped_lock lock{cache_mutex};
        PipelineCacheOptions options;
        options.async_compile = Settings::values.use_asynchronous_shaders.GetValue();
        options.graphics_pipeline_library = Settings::values.nvn_pipeline_library.GetValue();
        options.sync_diet = Settings::values.nvn_sync_diet.GetValue();
        options.shader_notify = impl.options.shader_notify;
        options.load_progress = impl.options.load_progress;
        cache_instance = std::make_unique<PipelineCache>(impl.device, std::move(options));
    }
    return *cache_instance;
}

void DrawResources::CollectCacheRows(std::vector<PipelineCache::Row>& rows) {
    std::scoped_lock lock{cache_mutex};
    if (cache_instance) {
        cache_instance->CollectRows(rows);
    }
}

void DrawResources::OpenCache(u64 title_id) {
    if (cache_instance) {
        return;
    }
    PipelineCacheOptions options;
    options.title_id = title_id;
    options.revision = Common::g_scm_rev;
    options.async_compile = Settings::values.use_asynchronous_shaders.GetValue();
    options.graphics_pipeline_library = Settings::values.nvn_pipeline_library.GetValue();
    options.sync_diet = Settings::values.nvn_sync_diet.GetValue();
    options.shader_notify = impl.options.shader_notify;
    options.load_progress = impl.options.load_progress;
    if (options.async_compile) {
        LOG_INFO(Render_Vulkan,
                 "[nvn] asynchronous shaders on: run-time pipeline compiles run on workers; the "
                 "draws that may be skipped are skipped until they finish, dispatches wait{}",
                 options.graphics_pipeline_library
                     ? " (nvn_pipeline_library: library parts are compiled and linked there too)"
                     : "");
    }
    LOG_INFO(Render_Vulkan, "[nvn] pass boundary syncs: {}",
             options.sync_diet ? "one barrier (nvn_sync_diet)"
                               : "barrier plus the render passes' two external dependencies");
    options.driver_cache_file = Settings::values.use_vulkan_driver_pipeline_cache.GetValue();
    if (!Settings::values.use_disk_shader_cache.GetValue()) {
        // As on the emulated GPU (no LoadDiskResources, no cache file): nothing is read from or
        // written to the profile's shader directory; every pipeline is translated and compiled
        // when a draw first needs it.
        LOG_INFO(Render_Vulkan,
                 "[nvn] disk shader cache off (use_disk_shader_cache): pipelines are not loaded "
                 "from or saved to disk");
    } else if (const char* own_dir = std::getenv("EDEN_NX2_NVN_CACHE_DIR");
               own_dir != nullptr && *own_dir != '\0') {
        // A/B runs that must not share a profile's cache at all (the cache files of one
        // directory are pruned to a few). The cache creates the directory.
        options.directory = own_dir;
        LOG_INFO(Render_Vulkan, "[nvn] pipeline cache directory: {}", options.directory.string());
    } else {
        const auto shader_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir);
        const auto title_dir = shader_dir / fmt::format("{:016x}", title_id);
        if (Common::FS::CreateDir(shader_dir) && Common::FS::CreateDir(title_dir)) {
            options.directory = title_dir;
        } else {
            LOG_ERROR(Render_Vulkan, "[nvn] pipeline cache: cannot create {}", title_dir.string());
        }
    }
    auto cache = std::make_unique<PipelineCache>(impl.device, std::move(options));
    PipelineCache* const opened = cache.get();
    {
        std::scoped_lock lock{cache_mutex};
        cache_instance = std::move(cache);
    }
    // The replay runs this when the guest names its title, before its first frame: every cached
    // pipeline is rebuilt before any draw or dispatch can need it, so a cold driver cache costs
    // boot time instead of stalls in play. EDEN_NX2_NVN_BLOCKING_PRELOAD=0: rebuild in the
    // background while the game runs (lookups that need a key still queued wait or skip).
    const char* const blocking = std::getenv("EDEN_NX2_NVN_BLOCKING_PRELOAD");
    if (blocking == nullptr || *blocking == '\0' || std::strcmp(blocking, "0") != 0) {
        opened->WaitPreload();
    } else {
        LOG_INFO(Render_Vulkan, "[nvn] pipeline cache: rebuilt in the background "
                                "(EDEN_NX2_NVN_BLOCKING_PRELOAD=0)");
    }
}
DrawResources::~DrawResources() = default;

void DrawResources::TracePass(u32 width, u32 height) {
    ++frame_trace.passes[{width, height}];
}

void DrawResources::TraceDispatch(bool tensor_like) {
    ++frame_trace.dispatches;
    frame_trace.dispatches_128 += tensor_like ? 1 : 0;
}

void DrawResources::TracePresent(u64 present) {
    FrameTrace& t = frame_trace;
    // The main scene target is the largest render pass extent that more than one pass uses; the
    // line is logged every 30th present and whenever that size changes.
    std::pair<u32, u32> main{};
    u64 main_area = 0;
    u32 total = 0;
    for (const auto& [size, count] : t.passes) {
        total += count;
        const u64 area = u64{size.first} * size.second;
        if (count >= 2 && area > main_area) {
            main = size;
            main_area = area;
        }
    }
    const auto now = std::chrono::steady_clock::now();
    const double ms = t.last_present == std::chrono::steady_clock::time_point{}
                          ? 0.0
                          : std::chrono::duration<double, std::milli>(now - t.last_present).count();
    t.last_present = now;
    if (present % 30 == 0 || main != t.last_main) {
        std::vector<std::pair<std::pair<u32, u32>, u32>> sizes(t.passes.begin(), t.passes.end());
        std::sort(sizes.begin(), sizes.end(), [](const auto& a, const auto& b) {
            return u64{a.first.first} * a.first.second > u64{b.first.first} * b.first.second;
        });
        std::string list;
        for (size_t i = 0; i < sizes.size() && i < 6; ++i) {
            list += fmt::format(" {}x{}:{}", sizes[i].first.first, sizes[i].first.second,
                                sizes[i].second);
        }
        LOG_INFO(Render_Vulkan,
                 "[nvn] tstrace present {}: main target {}x{}, {} passes, sizes (largest first){}, "
                 "{} dispatches ({} with 128 threads in x), {:.1f} ms since the previous present",
                 present, main.first, main.second, total, list, t.dispatches, t.dispatches_128, ms);
    }
    t.last_main = main;
    t.passes.clear();
    t.dispatches = 0;
    t.dispatches_128 = 0;
}

void DrawResources::NoteAsyncSkips(u64 present) {
    const u64 draws = counters.draws_skipped_async.load(std::memory_order_relaxed);
    if (draws == noted_draw_skips) {
        return;
    }
    counters.frames_with_async_skips.fetch_add(1, std::memory_order_relaxed);
    // One line per frame with skips, for the pop-in analysis; bounded so a pathological run
    // cannot flood the log.
    constexpr u64 MAX_LINES = 6000;
    if (async_skip_lines < MAX_LINES) {
        ++async_skip_lines;
        LOG_INFO(Render_Vulkan,
                 "[nvn] async pipelines: present {}: {} draws skipped awaiting a pipeline", present,
                 draws - noted_draw_skips);
    }
    noted_draw_skips = draws;
}

void DrawResources::OnProgramCreated(const std::shared_ptr<const VC::ProgramDesc>& program) {
    const VC::ProgramDesc& desc = *program;
    std::array<u64, STAGE_COUNT> hashes{};
    std::array<StageProgram, STAGE_COUNT> stages{};
    for (u32 stage = 0; stage < STAGE_COUNT; ++stage) {
        if (!desc.control[stage].empty() && !desc.code[stage].empty()) {
            hashes[stage] = HashProgramStage(desc.control[stage], desc.code[stage]);
            stages[stage] = {hashes[stage], &desc.control[stage], &desc.code[stage]};
        }
    }
    program_hashes[desc.serial] = hashes;
    // EDEN_NX2_NVN_LOG_CONTROL=1 (diagnostic): each program stage's control section, once per
    // stage hash, as hex (finding control flags such as early fragment tests).
    static const bool log_control = [] {
        const char* value = std::getenv("EDEN_NX2_NVN_LOG_CONTROL");
        return value != nullptr && value[0] == '1';
    }();
    if (log_control) {
        static std::mutex logged_mutex;
        static std::unordered_set<u64> logged;
        for (u32 stage = 0; stage < STAGE_COUNT; ++stage) {
            if (hashes[stage] == 0) {
                continue;
            }
            {
                std::scoped_lock lock{logged_mutex};
                if (!logged.insert(hashes[stage]).second) {
                    continue;
                }
            }
            std::string hex;
            for (const u8 byte : desc.control[stage]) {
                hex += fmt::format("{:02x}", byte);
            }
            LOG_INFO(Render_Vulkan, "[nvn] control s{} {:016x} {}", stage, hashes[stage], hex);
        }
    }
    PipelineCache* cache{};
    {
        std::scoped_lock lock{cache_mutex};
        cache = cache_instance.get();
    }
    if (cache != nullptr && cache->UsesGraphicsPipelineLibrary()) {
        // The stage pointers stay valid: the cache keeps `program` until it is done.
        cache->PrepareProgram(stages, program);
    }
    if (cache != nullptr) {
        // The local shader store, and the state list's pipelines that waited for these stages
        // (queued now, before PrepareCompute below would queue a default-answer compile).
        cache->NoteProgram(stages, program);
    }
    // A compute pipeline depends on nothing a dispatch sets but its texture answers, so it is
    // compiled now, off the replay thread, instead of at the first dispatch (which would stall
    // the frame for the whole compile). EDEN_NX2_NVN_COMPUTE_AT_CREATE: unset or 1 = at once on
    // the preload workers (eager), "idle" = at low priority while no asynchronous compile waits,
    // 0 = at first dispatch.
    enum class ComputeAtCreation { Off, Eager, Idle };
    static const ComputeAtCreation compute_at_creation = [] {
        const char* value = std::getenv("EDEN_NX2_NVN_COMPUTE_AT_CREATE");
        ComputeAtCreation mode = ComputeAtCreation::Eager;
        if (value != nullptr && std::strcmp(value, "0") == 0) {
            mode = ComputeAtCreation::Off;
        } else if (value != nullptr && std::strcmp(value, "idle") == 0) {
            mode = ComputeAtCreation::Idle;
        }
        LOG_INFO(Render_Vulkan, "[nvn] compute pipelines compiled {}",
                 mode == ComputeAtCreation::Off
                     ? "at first dispatch (EDEN_NX2_NVN_COMPUTE_AT_CREATE=0)"
                 : mode == ComputeAtCreation::Idle
                     ? "when the program is created, at idle priority "
                       "(EDEN_NX2_NVN_COMPUTE_AT_CREATE=idle)"
                     : "when the program is created (EDEN_NX2_NVN_COMPUTE_AT_CREATE=idle: at idle "
                       "priority, 0: at first dispatch)");
        return mode;
    }();
    if (cache != nullptr && compute_at_creation != ComputeAtCreation::Off &&
        stages[STAGE_COMPUTE].hash != 0) {
        cache->PrepareCompute(stages[STAGE_COMPUTE], program,
                              compute_at_creation == ComputeAtCreation::Idle);
    }
}

void DrawResources::RefreshSamplerSetting() {
    const s32 steps = CurrentAnisotropyBoostSteps();
    if (steps != anisotropy_steps) {
        anisotropy_steps = steps;
        ++samplers_epoch;
    }
}

VkSampler DrawResources::GetSampler(u64 serial, bool boosted) {
    const auto desc_it = impl.samplers.find(serial);
    if (desc_it == impl.samplers.end()) {
        return *fallback_sampler;
    }
    const VC::SamplerDesc& d = desc_it->second;
    // The key: a sampler the setting leaves alone is the guest's own for every setting, one the
    // setting changes is built per setting value, so a changed setting never reuses a VkSampler.
    const s32 steps = SamplerStepsKey(d, boosted ? anisotropy_steps : 0, max_sampler_anisotropy);
    if (const auto it = samplers.find(serial); it != samplers.end()) {
        for (const SamplerVariant& variant : it->second) {
            if (variant.steps == steps) {
                return *variant.sampler;
            }
        }
    }
    VkSamplerCreateInfo info = SamplerInfoOf(d, steps, max_sampler_anisotropy);
    // EDEN_NX2_FORCE_FILTER / EDEN_NX2_FORCE_ANISO / EDEN_NX2_SAMPLER_DIAGNOSTICS
    // (video_core/sampler_overrides.h): the same diagnostics as the legacy Sampler.
    bool forced = VideoCore::SamplerOverrides::ApplyFilters(info.magFilter, info.minFilter,
                                                            info.mipmapMode, info.maxAnisotropy);
    forced |= VideoCore::SamplerOverrides::ApplyLod(info.minLod, info.maxLod);
    info.anisotropyEnable = static_cast<VkBool32>(info.maxAnisotropy > 1.0f);
    if (VideoCore::SamplerOverrides::ShouldLog()) {
        LOG_INFO(
            Render_Vulkan,
            "[nvn] sampler {:#x}: mag={} min={} mip={} lod=[{}, {}] bias={} aniso={} forced={}",
            serial, static_cast<u32>(info.magFilter), static_cast<u32>(info.minFilter),
            static_cast<u32>(info.mipmapMode), info.minLod, info.maxLod, info.mipLodBias,
            info.maxAnisotropy, forced);
    }
    VkSamplerReductionModeCreateInfo reduction{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO,
        .pNext = nullptr,
        .reductionMode = d.reduction == 1   ? VK_SAMPLER_REDUCTION_MODE_MIN
                         : d.reduction == 2 ? VK_SAMPLER_REDUCTION_MODE_MAX
                                            : VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE,
    };
    if (d.reduction != 0 && impl.device.IsExtSamplerFilterMinmaxSupported()) {
        info.pNext = &reduction;
    }
    VkSamplerCustomBorderColorCreateInfoEXT custom_border{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT,
        .pNext = info.pNext,
        .customBorderColor = {.float32 = {d.border[0], d.border[1], d.border[2], d.border[3]}},
        .format = VK_FORMAT_UNDEFINED,
    };
    const bool border_used = info.addressModeU == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
                             info.addressModeV == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
                             info.addressModeW == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    if (border_used) {
        if (impl.device.IsCustomBorderColorUsable()) {
            info.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
            info.pNext = &custom_border;
        } else {
            const bool opaque = d.border[3] > 0.5f;
            const bool white = d.border[0] > 0.5f;
            info.borderColor = !opaque ? VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK
                               : white ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE
                                       : VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        }
    }
    try {
        auto& variants = samplers[serial];
        variants.push_back({steps, impl.device.GetLogical().CreateSampler(info)});
        return *variants.back().sampler;
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "[nvn] sampler {:#x} creation failed: {}", serial,
                  exception.what());
        return *fallback_sampler;
    }
}

std::optional<VideoCommon::PathDiff::Sampler> DrawResources::DescribeSampler(u64 serial) const {
    const auto desc_it = impl.samplers.find(serial);
    if (desc_it == impl.samplers.end()) {
        return std::nullopt;
    }
    const VC::SamplerDesc& d = desc_it->second;
    // What the emulated path reports too: the sampler with the setting applied, before its
    // per-view reduction.
    const VkSamplerCreateInfo info = SamplerInfoOf(
        d, SamplerStepsKey(d, anisotropy_steps, max_sampler_anisotropy), max_sampler_anisotropy);
    const bool border_used = info.addressModeU == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
                             info.addressModeV == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
                             info.addressModeW == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    return VideoCommon::PathDiff::Sampler{
        .mag_filter = static_cast<u32>(info.magFilter),
        .min_filter = static_cast<u32>(info.minFilter),
        .mipmap_mode = static_cast<u32>(info.mipmapMode),
        .address = {static_cast<u32>(info.addressModeU), static_cast<u32>(info.addressModeV),
                    static_cast<u32>(info.addressModeW)},
        .lod_bias = info.mipLodBias,
        .min_lod = info.minLod,
        .max_lod = info.maxLod,
        .anisotropy = info.anisotropyEnable != VK_FALSE,
        .max_anisotropy = info.maxAnisotropy,
        .compare = info.compareEnable != VK_FALSE,
        .compare_op = static_cast<u32>(info.compareOp),
        // Only meaningful when an address mode clamps to the border (0 otherwise, both paths).
        .border_color = static_cast<u32>(!border_used ? VkBorderColor{}
                                         : impl.device.IsCustomBorderColorUsable()
                                             ? VK_BORDER_COLOR_FLOAT_CUSTOM_EXT
                                             : info.borderColor),
        .border = {d.border[0], d.border[1], d.border[2], d.border[3]},
        .reduction =
            static_cast<u32>(d.reduction == 1   ? VK_SAMPLER_REDUCTION_MODE_MIN
                             : d.reduction == 2 ? VK_SAMPLER_REDUCTION_MODE_MAX
                                                : VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE),
    };
}

void DrawResources::ReleaseSampler(u64 serial, u64 after_seq) {
    const auto it = samplers.find(serial);
    if (it == samplers.end()) {
        return;
    }
    auto variants = std::make_shared<std::vector<SamplerVariant>>(std::move(it->second));
    samplers.erase(it);
    ++samplers_epoch;
    impl.retired.push_back({after_seq, [variants]() mutable { variants->clear(); }});
}

VkImageView DrawResources::GetView(const Texture& texture, const ViewKey& key, u64* id) {
    // A stand-in changes BC format when a later upload classifies it differently (smart BC,
    // RefitStandin recreates its image as BC1/BC4/BC5/BC3). A ViewKey made before that (a clear's,
    // an attachment's, one memoised by a caller) still names the old format, and a view of another
    // BC class than the image's is invalid usage: NVIDIA loses the device on it. Such a draw is
    // left without the texture for this binding (the next binding pass rebuilds the key from the
    // storage, ViewKeyOf) instead of creating the view.
    const ViewKey& key_valid = key;
    if (texture.storage && texture.storage->host.bc_standin &&
        !ViewsBcStandin(key.format, texture.storage->host.format)) {
        static std::atomic<u32> skipped{0};
        if (skipped.fetch_add(1) < 16) {
            LOG_WARNING(Render_Vulkan,
                        "[nvn] stand-in view {:#x} left out: texture {:#x}, storage format {:#x} "
                        "(a view key from before the stand-in's format changed)",
                        static_cast<int>(key.format), texture.serial,
                        static_cast<int>(texture.storage->host.format));
        }
        return VK_NULL_HANDLE;
    }
    // EDEN_NVN_VIEW_CACHE (default on): the views of the storage's image, shared by every texture
    // object over it. DK Bananza initializes about 14 new texture objects on the same storages
    // every frame; each would otherwise create (and keep) views of its own.
    static const bool view_cache = [] {
        const char* env = std::getenv("EDEN_NVN_VIEW_CACHE");
        const bool on = env == nullptr || env[0] != '0';
        if (!on) {
            LOG_INFO(Render_Vulkan, "[nvn] EDEN_NVN_VIEW_CACHE=0: image views per texture object");
        }
        return on;
    }();
    Storage& storage = *texture.storage;
    StorageViewCache* shared = nullptr;
    TextureViews* views = nullptr;
    if (view_cache && storage.image) {
        if (!storage.shared_views || storage.shared_views->image != *storage.image) {
            if (storage.shared_views) {
                // A replaced image takes the views first (TakeStorageViews). This is an image
                // swapped without that: the textures that took the old views are told (their
                // descriptor sets are dropped) and the old views go after the submissions
                // recorded so far, as TakeStorageViews' would.
                if (!storage.shared_views->views.empty() && stale_view_caches++ < 16) {
                    LOG_WARNING(Render_Vulkan,
                                "[nvn] view cache: storage image replaced without its views "
                                "(texture {:#x}); old views retired",
                                texture.serial);
                }
                RetireViews(TakeStorageViews(storage), storage);
            }
            storage.shared_views = std::make_shared<StorageViewCache>();
            storage.shared_views->image = *storage.image;
            storage.shared_views->id = next_view_cache_id++;
        }
        shared = storage.shared_views.get();
        // Registered once per cache: the texture's serial is told when these views go.
        if (texture.view_slot != shared || texture.view_slot_epoch != shared->id) {
            auto& fresh = const_cast<Texture&>(texture);
            shared->holders.push_back(texture.serial);
            fresh.view_slot = shared;
            fresh.view_slot_epoch = shared->id;
        }
        for (const StorageViewCache::Entry& entry : shared->views) {
            if (entry.key == key_valid) {
                ++view_cache_hits;
                if (id != nullptr) {
                    *id = entry.id;
                }
                return *entry.view;
            }
        }
    } else {
        // The texture remembers its texture_views node (stable until TakeViews erases it).
        if (texture.view_slot == nullptr || texture.view_slot_epoch != views_epoch) {
            auto& fresh = const_cast<Texture&>(texture);
            fresh.view_slot = &texture_views[texture.serial];
            fresh.view_slot_epoch = views_epoch;
        }
        views = static_cast<TextureViews*>(texture.view_slot);
        if (!views->storage) {
            views->storage = texture.storage;
        }
        for (const ViewRef& ref : views->views) {
            if (ref.key == key_valid) {
                if (id != nullptr) {
                    *id = ref.id;
                }
                return ref.handle;
            }
        }
    }
    if (id != nullptr) {
        *id = 0;
    }
    const auto swizzle = [&](u32 shift) {
        return static_cast<VkComponentSwizzle>((key.swizzle >> shift) & 0xF);
    };
    // A view inherits the image's usage; a format without storage support (sRGB) must not claim it.
    VkImageViewUsageCreateInfo usage_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
        .pNext = nullptr,
        .usage = texture.storage->usage,
    };
    if ((usage_info.usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0 &&
        !impl.device.IsFormatSupported(key.format, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT,
                                       FormatType::Optimal)) {
        usage_info.usage &= ~VK_IMAGE_USAGE_STORAGE_BIT;
    }
    try {
        vk::ImageView view = impl.device.GetLogical().CreateImageView({
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = &usage_info,
            .flags = 0,
            .image = *texture.storage->image,
            .viewType = key.type,
            .format = key.format,
            .components{
                .r = swizzle(0),
                .g = swizzle(4),
                .b = swizzle(8),
                .a = swizzle(12),
            },
            .subresourceRange{
                .aspectMask = key.aspect,
                .baseMipLevel = key.base_level,
                .levelCount = key.levels,
                .baseArrayLayer = key.base_layer,
                .layerCount = key.layers,
            },
        });
        const VkImageView handle = *view;
        const u64 view_id = next_view_id++;
        ++view_cache_creates;
        if (shared != nullptr) {
            shared->views.push_back({key_valid, std::move(view), view_id});
        } else {
            views->owned.push_back(std::move(view));
            views->views.push_back({key_valid, handle, view_id});
        }
        if (id != nullptr) {
            *id = view_id;
        }
        return handle;
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan,
                  "[nvn] image view for texture {:#x} (format {}, type {}, levels {}+{}, layers "
                  "{}+{}) failed: {}",
                  texture.serial, static_cast<int>(key.format), static_cast<int>(key.type),
                  key.base_level, key.levels, key.base_layer, key.layers, exception.what());
        return VK_NULL_HANDLE;
    }
}

void DrawResources::NoteViewsReleased(u64 serial) {
    for (TexturePoolSet* const pool : texture_pools) {
        pool->NoteReleased(serial);
    }
}

std::vector<vk::ImageView> DrawResources::TakeStorageViews(const Storage& storage) {
    std::vector<vk::ImageView> taken;
    // Nodes exist only with EDEN_NVN_VIEW_CACHE=0 (or for a storage without an image), so with
    // the cache on this walk is over (almost) nothing.
    if (!texture_views.empty()) {
        for (auto& [serial, views] : texture_views) {
            if (views.storage.get() != &storage) {
                continue;
            }
            if (!views.views.empty()) {
                NoteViewsReleased(serial);
            }
            for (vk::ImageView& view : views.owned) {
                taken.push_back(std::move(view));
            }
            views.owned.clear();
            views.views.clear();
        }
    }
    // The views every texture over the storage shared (EDEN_NVN_VIEW_CACHE): the textures that
    // took them are this cache's holders. Each re-registers with the next cache.
    if (storage.shared_views) {
        for (const u64 serial : storage.shared_views->holders) {
            NoteViewsReleased(serial);
        }
        for (StorageViewCache::Entry& entry : storage.shared_views->views) {
            taken.push_back(std::move(entry.view));
        }
        const_cast<Storage&>(storage).shared_views.reset();
    }
    return taken;
}

void DrawResources::RetireViews(std::vector<vk::ImageView>&& views, const Storage& storage) {
    if (views.empty()) {
        return;
    }
    // After every submission recorded so far (ReleaseStorageImage's bound).
    u64 after = (std::max)({impl.completed_seq.load(std::memory_order_acquire),
                            impl.replay_seq.load(std::memory_order_relaxed), storage.last_used_seq});
    if (storage.touched_seq != ~u64{0}) {
        after = (std::max)(after, storage.touched_seq);
    }
    auto old = std::make_shared<std::vector<vk::ImageView>>(std::move(views));
    impl.retired.push_back({after, [old]() mutable { old->clear(); }});
}

size_t DrawResources::FramebufferKeyHash::operator()(const FramebufferKey& key) const noexcept {
    u64 h = reinterpret_cast<u64>(key.render_pass) * 0x9E3779B97F4A7C15ULL;
    const auto mix = [&h](u64 v) { h = (h ^ v) * 0x100000001B3ULL + (h >> 29); };
    for (u32 i = 0; i < key.count; ++i) {
        mix(key.ids[i]);
    }
    mix(static_cast<u64>(key.width) << 32 | key.height);
    mix(static_cast<u64>(key.layers) << 8 | key.count);
    return static_cast<size_t>(h);
}

std::shared_ptr<vk::Framebuffer> DrawResources::GetFramebuffer(const FramebufferKey& key,
                                                               std::span<const VkImageView> views) {
    static const bool fb_cache = [] {
        const char* env = std::getenv("EDEN_NVN_FB_CACHE");
        const bool on = env == nullptr || env[0] != '0';
        if (!on) {
            LOG_INFO(Render_Vulkan, "[nvn] EDEN_NVN_FB_CACHE=0: a framebuffer per render pass");
        }
        return on;
    }();
    const u64 tick = ++framebuffer_tick;
    if ((tick & 0x7FFF) == 0) {
        LOG_INFO(Render_Vulkan,
                 "[nvn] framebuffer cache: {} entries, {} hits, {} created; image views: {} "
                 "shared hits, {} created, {} texture view nodes, {} images swapped under a cache",
                 framebuffers.size(), framebuffer_hits, framebuffer_creates, view_cache_hits,
                 view_cache_creates, texture_views.size(), stale_view_caches);
    }
    if (fb_cache) {
        if (const auto it = framebuffers.find(key); it != framebuffers.end()) {
            ++framebuffer_hits;
            it->second.last_use = tick;
            return it->second.framebuffer;
        }
    }
    ++framebuffer_creates;
    auto framebuffer = std::make_shared<vk::Framebuffer>(impl.device.GetLogical().CreateFramebuffer({
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .renderPass = key.render_pass,
        .attachmentCount = static_cast<u32>(views.size()),
        .pAttachments = views.data(),
        .width = key.width,
        .height = key.height,
        .layers = key.layers,
    }));
    if (fb_cache) {
        // Bounded: once the cache is large, entries unused for the last 64k pass begins go (the
        // submissions that used them hold references of their own).
        constexpr size_t MAX_FRAMEBUFFERS = 2048;
        if (framebuffers.size() >= MAX_FRAMEBUFFERS) {
            std::erase_if(framebuffers, [tick](const auto& entry) {
                return entry.second.last_use + 65536 < tick;
            });
            if (framebuffers.size() >= MAX_FRAMEBUFFERS) {
                framebuffers.clear();
            }
        }
        framebuffers.emplace(key, FramebufferEntry{framebuffer, tick});
    }
    return framebuffer;
}

std::shared_ptr<TextureViews> DrawResources::TakeViews(u64 serial) {
    // Called for every destroyed texture: the pool set drops what names it.
    NoteViewsReleased(serial);
    const auto it = texture_views.find(serial);
    if (it == texture_views.end()) {
        return nullptr;
    }
    // (Views this texture owns go with it, EDEN_NVN_VIEW_CACHE=0; shared ones stay with the
    // storage.)
    auto result = std::make_shared<TextureViews>(std::move(it->second));
    texture_views.erase(it);
    ++views_epoch;
    return result;
}

DrawResources::Scratch* DrawResources::FindScratch(const Storage& storage) {
    for (Scratch& scratch : scratch_images) {
        if (scratch.storage.get() == &storage) {
            return &scratch;
        }
    }
    return nullptr;
}

DrawResources::Scratch* DrawResources::GetScratch(const Storage& source, u32 slot) {
    for (Scratch& scratch : scratch_images) {
        const Storage& s = *scratch.storage;
        if (scratch.slot == slot && s.host.format == source.host.format && s.type == source.type &&
            s.extent.width == source.extent.width && s.extent.height == source.extent.height &&
            s.extent.depth == source.extent.depth && s.levels == source.levels &&
            s.layers == source.layers && s.aspect == source.aspect) {
            return &scratch;
        }
    }
    auto storage = std::make_shared<Storage>();
    storage->host = source.host;
    storage->type = source.type;
    storage->extent = source.extent;
    storage->levels = source.levels;
    storage->layers = source.layers;
    storage->aspect = source.aspect;
    storage->samples = VK_SAMPLE_COUNT_1_BIT;
    storage->create_flags =
        VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
        (source.type == VK_IMAGE_TYPE_3D
             ? static_cast<VkImageCreateFlags>(VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT)
             : 0u);
    storage->create_flags |= source.create_flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    storage->usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT;
    // Feedback scratch images get their own allocation (EDEN_NX2_NVN_SCRATCH_DEDICATED=0: shared
    // blocks). Fast Fusion lost the device on about one boot in five: Aftermath showed a ROP write
    // refused (illegal access, not a translation fault) inside a fresh scratch sub-allocated in a
    // shared block, during frame 1's feedback copy of its 8181x1721 R8 mip chain. With a dedicated
    // allocation: 0 of 13 boots crashed (3 of 15 without). The cause below the allocation is not
    // known; scratches are few and kept for the session, so their own memory costs nothing.
    static const bool dedicated = [] {
        const char* v = std::getenv("EDEN_NX2_NVN_SCRATCH_DEDICATED");
        return v == nullptr || *v == '\0' || v[0] != '0';
    }();
    try {
        storage->image = impl.allocator->CreateImage({
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = storage->create_flags,
            .imageType = storage->type,
            .format = storage->host.format,
            .extent = storage->extent,
            .mipLevels = storage->levels,
            .arrayLayers = storage->layers,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = storage->usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        }, dedicated);
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "[nvn] feedback scratch image ({}x{}x{}, format {}) failed: {}",
                  source.extent.width, source.extent.height, source.extent.depth,
                  static_cast<int>(source.host.format), exception.what());
        return nullptr;
    }
    // Written by copies only: no guest bytes behind it, nothing to upload or check.
    storage->contents_valid = true;
    storage->origin = StorageOrigin::Gpu;
    auto texture = std::make_shared<Texture>();
    texture->serial = next_scratch_serial++;
    texture->storage = storage;
    texture->host = storage->host;
    scratch_images.push_back({std::move(storage), std::move(texture), slot});
    return &scratch_images.back();
}

DrawResources::Scratch* DrawResources::GetRetypedScratch(const Storage& source, u32 slot,
                                                         bool cube) {
    const VkImageType type = cube ? VK_IMAGE_TYPE_2D : VK_IMAGE_TYPE_3D;
    const u32 layers = cube ? 6u : 1u;
    for (Scratch& scratch : scratch_images) {
        const Storage& s = *scratch.storage;
        if (scratch.slot == slot && s.host.format == source.host.format && s.type == type &&
            s.extent.width == source.extent.width && s.extent.height == source.extent.height &&
            s.extent.depth == 1 && s.layers == layers && s.aspect == source.aspect) {
            return &scratch;
        }
    }
    auto storage = std::make_shared<Storage>();
    storage->host = source.host;
    storage->type = type;
    storage->extent = {source.extent.width, source.extent.height, 1};
    storage->levels = 1;
    storage->layers = layers;
    storage->aspect = source.aspect;
    storage->samples = VK_SAMPLE_COUNT_1_BIT;
    storage->create_flags =
        VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
        (cube ? static_cast<VkImageCreateFlags>(VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) : 0u);
    storage->usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT;
    try {
        storage->image = impl.allocator->CreateImage({
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = storage->create_flags,
            .imageType = storage->type,
            .format = storage->host.format,
            .extent = storage->extent,
            .mipLevels = 1,
            .arrayLayers = layers,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = storage->usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        });
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "[nvn] placeholder image (format {}, {}) failed: {}",
                  static_cast<int>(source.host.format), cube ? "cube" : "3D", exception.what());
        return nullptr;
    }
    storage->contents_valid = true;
    storage->origin = StorageOrigin::Gpu;
    auto texture = std::make_shared<Texture>();
    texture->serial = next_scratch_serial++;
    texture->storage = storage;
    texture->host = storage->host;
    scratch_images.push_back({std::move(storage), std::move(texture), slot});
    return &scratch_images.back();
}

std::shared_ptr<ArenaChunk> DrawResources::AcquireArena(u64 min_size) {
    {
        std::scoped_lock lock{arena_mutex};
        for (auto it = free_arenas.begin(); it != free_arenas.end(); ++it) {
            if ((*it)->size >= min_size) {
                std::shared_ptr<ArenaChunk> chunk = std::move(*it);
                free_arenas.erase(it);
                chunk->used = 0;
                return chunk;
            }
        }
    }
    auto chunk = std::make_shared<ArenaChunk>();
    chunk->size = (std::max)(min_size, ARENA_CHUNK_SIZE);
    chunk->buffer =
        impl.CreateHostBuffer(chunk->size, ARENA_USAGE, MemoryUsage::Upload, AllocationKind::Arena);
    chunk->pointer = chunk->buffer.Mapped().data();
    return chunk;
}

void DrawResources::ReleaseArenas(std::vector<std::shared_ptr<ArenaChunk>>& chunks) {
    std::scoped_lock lock{arena_mutex};
    for (auto& chunk : chunks) {
        // Keep the pool small: oversize one-offs are dropped.
        if (chunk->size == ARENA_CHUNK_SIZE && free_arenas.size() < 8) {
            free_arenas.push_back(std::move(chunk));
        }
    }
    chunks.clear();
}

std::shared_ptr<ShadowChunk> DrawResources::AcquireShadow(u64 min_size) {
    {
        std::scoped_lock lock{shadow_mutex};
        for (auto it = free_shadows.begin(); it != free_shadows.end(); ++it) {
            if ((*it)->size >= min_size) {
                std::shared_ptr<ShadowChunk> chunk = std::move(*it);
                free_shadows.erase(it);
                chunk->used = 0;
                return chunk;
            }
        }
    }
    auto chunk = std::make_shared<ShadowChunk>();
    // A multiple of 256: staged vertex copies by shader write whole 16-byte units up to the end of
    // their allocation (AppendVertexStageTasks).
    chunk->size = ((std::max)(min_size, SHADOW_CHUNK_SIZE) + 255) / 256 * 256;
    try {
        chunk->buffer = impl.CreateHostBuffer(chunk->size, SHADOW_USAGE, MemoryUsage::DeviceLocal,
                                              AllocationKind::Shadow);
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "[nvn] storage shadow memory ({} bytes) failed: {}", chunk->size,
                  exception.what());
        return nullptr;
    }
    const u64 created = counters.shadow_chunks_created.fetch_add(1, std::memory_order_relaxed) + 1;
    // Chunks alive at once (the free list keeps at most MAX_FREE_SHADOWS; the rest belong to
    // submissions that have not completed): each new high mark, 32 chunks apart, is logged.
    const u64 destroyed = counters.shadow_chunks_destroyed.load(std::memory_order_relaxed);
    const u64 alive = created > destroyed ? created - destroyed : 0;
    static std::atomic<u64> alive_mark{64};
    u64 mark = alive_mark.load(std::memory_order_relaxed);
    if (alive >= mark &&
        alive_mark.compare_exchange_strong(mark, alive + 32, std::memory_order_relaxed)) {
        LOG_WARNING(Render_Vulkan,
                    "[nvn] {} shadow chunks alive (at least {} MiB of device memory; {} created, "
                    "{} destroyed, this one {:#x} bytes): submissions hold them until they "
                    "complete",
                    alive, (alive * SHADOW_CHUNK_SIZE) >> 20, created, destroyed, chunk->size);
    }
    return chunk;
}

void DrawResources::ReleaseShadows(std::vector<std::shared_ptr<ShadowChunk>>& chunks) {
    std::scoped_lock lock{shadow_mutex};
    for (auto& chunk : chunks) {
        // Chunks up to SHADOW_KEEP_MAX stay: the alias transfer's scratch buffer is a chunk sized
        // to the shared bytes of two textures, asked for many times a second.
        if (chunk->size <= SHADOW_KEEP_MAX && free_shadows.size() < MAX_FREE_SHADOWS) {
            free_shadows.push_back(std::move(chunk));
        } else {
            counters.shadow_chunks_destroyed.fetch_add(1, std::memory_order_relaxed);
        }
    }
    chunks.clear();
}

VkBuffer DrawResources::DummyBuffer() {
    return *dummy_buffer;
}

u64 DrawResources::SinkSlice(u64 program_hash, u32 stage, u32 slot) {
    const std::tuple<u64, u32, u32> key{program_hash, stage, slot};
    if (const auto it = sink_index.find(key); it != sink_index.end()) {
        return (u64{it->second} + 1) * SINK_SLICE_SIZE;
    }
    if (sink_keys.size() + 1 >= SINK_SLICES) {
        return 0; // the shared slice
    }
    const u32 index = static_cast<u32>(sink_keys.size());
    sink_keys.push_back({program_hash, stage, slot, false});
    sink_index.emplace(key, index);
    return (u64{index} + 1) * SINK_SLICE_SIZE;
}

void DrawResources::CheckSink(u64 present) {
    if (present % SINK_CHECK_PRESENTS != 0) {
        return;
    }
    sink_buffer.Invalidate();
    const u8* const base = sink_buffer.Mapped().data();
    const auto first_nonzero = [&](u64 slice) -> std::optional<std::pair<u64, u32>> {
        const u32* const words = reinterpret_cast<const u32*>(base + slice * SINK_SLICE_SIZE);
        for (u64 i = 0; i < SINK_SLICE_SIZE / sizeof(u32); ++i) {
            if (words[i] != 0) {
                return std::pair<u64, u32>{i * sizeof(u32), words[i]};
            }
        }
        return std::nullopt;
    };
    for (size_t i = 0; i < sink_keys.size(); ++i) {
        SinkKey& key = sink_keys[i];
        if (key.reported) {
            continue;
        }
        if (const auto found = first_nonzero(i + 1)) {
            key.reported = true;
            LOG_WARNING(Render_Vulkan,
                        "[nvn] a shader wrote through a storage buffer binding without a buffer "
                        "(slot unbound, or its range is in no pool): "
                        "program {:016x} stage {} slot {} (word +{:#x} = {:#x}); the hardware "
                        "would fault there",
                        key.program_hash, key.stage, key.slot, found->first, found->second);
        }
    }
    if (!sink_overflow_reported) {
        if (const auto found = first_nonzero(0)) {
            sink_overflow_reported = true;
            LOG_WARNING(Render_Vulkan,
                        "[nvn] a shader wrote through a storage buffer binding without a buffer "
                        "(more unbound slots than the sink has slices; word +{:#x} = {:#x})",
                        found->first, found->second);
        }
    }
}

VkImageView DrawResources::DummyView(VkImageViewType type) {
    size_t index = 0;
    VkImageType image_type = VK_IMAGE_TYPE_2D;
    u32 layers = 1;
    VkImageCreateFlags flags = 0;
    switch (type) {
    case VK_IMAGE_VIEW_TYPE_1D:
        index = 0;
        image_type = VK_IMAGE_TYPE_1D;
        break;
    case VK_IMAGE_VIEW_TYPE_2D:
        index = 1;
        break;
    case VK_IMAGE_VIEW_TYPE_3D:
        index = 2;
        image_type = VK_IMAGE_TYPE_3D;
        break;
    case VK_IMAGE_VIEW_TYPE_CUBE:
        index = 3;
        layers = 6;
        flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        break;
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY:
        index = 4;
        image_type = VK_IMAGE_TYPE_1D;
        layers = 2;
        break;
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY:
        index = 5;
        layers = 2;
        break;
    default: // CUBE_ARRAY
        index = 6;
        layers = 6;
        flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        type = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
        break;
    }
    DummyImage& dummy = dummy_images[index];
    if (dummy.view) {
        return *dummy.view;
    }
    const bool is_array_view = type == VK_IMAGE_VIEW_TYPE_1D_ARRAY ||
                               type == VK_IMAGE_VIEW_TYPE_2D_ARRAY ||
                               type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    try {
        dummy.image = impl.allocator->CreateImage({
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = flags,
            .imageType = image_type,
            .format = DummyFormat(),
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
        dummy.view = impl.device.GetLogical().CreateImageView({
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = *dummy.image,
            .viewType = type,
            .format = DummyFormat(),
            .components{},
            .subresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = (type == VK_IMAGE_VIEW_TYPE_1D || type == VK_IMAGE_VIEW_TYPE_2D ||
                               type == VK_IMAGE_VIEW_TYPE_3D)
                                  ? 1u
                                  : layers,
            },
        });
        static_cast<void>(is_array_view);
        impl.ImmediateSubmit([&](vk::CommandBuffer cmd) {
            const VkImageSubresourceRange range{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
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
                .image = *dummy.image,
                .subresourceRange = range,
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                0, barrier);
            const VkClearColorValue black{.float32 = {0.0f, 0.0f, 0.0f, 0.0f}};
            cmd.ClearColorImage(*dummy.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, black, range);
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                0, barrier);
        });
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "[nvn] dummy image creation failed: {}", exception.what());
        dummy.view = vk::ImageView{};
        return VK_NULL_HANDLE;
    }
    return *dummy.view;
}

VkBuffer DrawResources::ProgramConstants(const PipelineBase& pipeline, u32 stage) {
    auto& buffers = constant_buffers[&pipeline];
    vk::Buffer& buffer = buffers[stage];
    if (buffer) {
        return *buffer;
    }
    const auto& words = pipeline.stages[stage].program_constants;
    const u64 size = (std::max<u64>)(words ? words->size() * sizeof(u32) : 0, 256);
    buffer = impl.CreateHostBuffer((size + 255) / 256 * 256, ARENA_USAGE, MemoryUsage::Upload,
                                   AllocationKind::Arena);
    std::span<u8> mapped = buffer.Mapped();
    std::memset(mapped.data(), 0, mapped.size());
    if (words && !words->empty()) {
        std::memcpy(mapped.data(), words->data(), words->size() * sizeof(u32));
    }
    buffer.Flush();
    return *buffer;
}

} // namespace Vulkan::Nvn
