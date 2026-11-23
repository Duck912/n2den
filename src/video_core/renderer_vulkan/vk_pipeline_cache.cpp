// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include "common/n2den_trace.h"
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <thread>
#include <unordered_set>
#include <vector>
#include <bit>
#include <numeric>
#include "common/cityhash.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/thread_worker.h"
#include "core/core.h"
#include "shader_recompiler/frontend/ampere/native.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/program_header.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/maxwell_to_vk.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/pipeline_statistics.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/native_census.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_reuse.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/shader_cache.h"
#include "video_core/shader_environment.h"
#include "video_core/shader_notify.h"
#include "video_core/surface.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "video_core/n2den_debug_control.h"
#include "common/scope_exit.h"
#include <nlohmann/json.hpp>

#ifdef __ANDROID__
#include "../../android/app/src/main/jni/android_settings.h"
#endif

namespace Vulkan {

namespace {

// n2Den debug server dump_ir: the IR after the translation passes, a summary of the bindings it
// asks for, and the SPIR-V it was emitted to, from the live environment.
std::string DebugIrSummary(const Shader::IR::Program& program) {
    std::string out{Shader::IR::DumpProgram(program)};
    const auto& info{program.info};
    out += fmt::format("\n; untracked global accesses {} ({} stores)\n", info.untracked_global_accesses,
                       info.untracked_global_stores);
    for (const auto& d : info.storage_buffers_descriptors) {
        out += fmt::format("; storage buffer c{}[{:#x}] count {} written {}\n", d.cbuf_index, d.cbuf_offset,
                           d.count, d.is_written);
    }
    for (const auto& d : info.texture_descriptors) {
        out += fmt::format("; texture type {} c{}[{:#x}] count {} depth {}\n", static_cast<u32>(d.type),
                           d.cbuf_index, d.cbuf_offset, d.count, d.is_depth);
    }
    for (const auto& d : info.image_descriptors) {
        out += fmt::format("; image type {} c{}[{:#x}] count {}\n", static_cast<u32>(d.type), d.cbuf_index,
                           d.cbuf_offset, d.count);
    }
    return out;
}

void WriteDebugDump(u64 key, u64 hash, bool native, std::string_view stage, const std::string& ir,
                    std::span<const u32> spirv) {
    try {
        const auto dir{Common::FS::GetEdenPath(Common::FS::EdenPath::DumpDir) / "n2den-debug" / "ir"};
        std::filesystem::create_directories(dir);
        const std::string base{fmt::format("{:016x}-{}-{:016x}-{}", key, native ? "native" : "lowered", hash, stage)};
        const auto ir_path{dir / (base + ".ir.txt")};
        const auto spv_path{dir / (base + ".spv")};
        std::ofstream{ir_path, std::ios::trunc} << ir;
        std::ofstream spv_file{spv_path, std::ios::binary | std::ios::trunc};
        spv_file.write(reinterpret_cast<const char*>(spirv.data()),
                       static_cast<std::streamsize>(spirv.size() * sizeof(u32)));
        VideoCore::N2DenDebug::NoteDump(key, hash, {{"frontend", native ? "native" : "lowered"},
                                                    {"stage", std::string{stage}},
                                                    {"ir", ir_path.generic_string()},
                                                    {"spirv", spv_path.generic_string()},
                                                    {"spirv_words", spirv.size()}});
    } catch (const std::exception& error) {
        LOG_ERROR(Render_Vulkan, "n2Den debug dump of {:016x} failed: {}", key, error.what());
    }
}

using Shader::Backend::SPIRV::EmitSPIRV;
using Shader::Maxwell::ConvertLegacyToGeneric;
using Shader::Maxwell::GenerateGeometryPassthrough;
using Shader::Maxwell::MergeDualVertexPrograms;
using Shader::Maxwell::TranslateProgram;
using VideoCommon::ComputeEnvironment;
using VideoCommon::FileEnvironment;
using VideoCommon::GenericEnvironment;
using VideoCommon::GraphicsEnvironment;

constexpr u32 CACHE_VERSION = 41;
constexpr size_t VULKAN_CACHE_FLUSH_PIPELINES = 128;
constexpr size_t VULKAN_CACHE_FLUSH_MIN_SECONDS = 30;
constexpr std::array<char, 8> VULKAN_CACHE_MAGIC_NUMBER{'y', 'u', 'z', 'u', 'v', 'k', 'c', 'h'};

template <typename Container>
auto MakeSpan(Container& container) {
    return std::span(container.data(), container.size());
}

Shader::OutputTopology MaxwellToOutputTopology(Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::Points:
        return Shader::OutputTopology::PointList;
    case Maxwell::PrimitiveTopology::LineStrip:
        return Shader::OutputTopology::LineStrip;
    default:
        return Shader::OutputTopology::TriangleStrip;
    }
}

Shader::CompareFunction MaxwellToCompareFunction(Maxwell::ComparisonOp comparison) {
    switch (comparison) {
    case Maxwell::ComparisonOp::Never_D3D:
    case Maxwell::ComparisonOp::Never_GL:
        return Shader::CompareFunction::Never;
    case Maxwell::ComparisonOp::Less_D3D:
    case Maxwell::ComparisonOp::Less_GL:
        return Shader::CompareFunction::Less;
    case Maxwell::ComparisonOp::Equal_D3D:
    case Maxwell::ComparisonOp::Equal_GL:
        return Shader::CompareFunction::Equal;
    case Maxwell::ComparisonOp::LessEqual_D3D:
    case Maxwell::ComparisonOp::LessEqual_GL:
        return Shader::CompareFunction::LessThanEqual;
    case Maxwell::ComparisonOp::Greater_D3D:
    case Maxwell::ComparisonOp::Greater_GL:
        return Shader::CompareFunction::Greater;
    case Maxwell::ComparisonOp::NotEqual_D3D:
    case Maxwell::ComparisonOp::NotEqual_GL:
        return Shader::CompareFunction::NotEqual;
    case Maxwell::ComparisonOp::GreaterEqual_D3D:
    case Maxwell::ComparisonOp::GreaterEqual_GL:
        return Shader::CompareFunction::GreaterThanEqual;
    case Maxwell::ComparisonOp::Always_D3D:
    case Maxwell::ComparisonOp::Always_GL:
        return Shader::CompareFunction::Always;
    }
    UNIMPLEMENTED_MSG("Unimplemented comparison op={}", comparison);
    return {};
}

Shader::AttributeType CastAttributeType(const FixedPipelineState::VertexAttribute& attr) {
    if (attr.enabled == 0) {
        return Shader::AttributeType::Disabled;
    }
    switch (attr.Type()) {
    case Maxwell::VertexAttribute::Type::UnusedEnumDoNotUseBecauseItWillGoAway:
        ASSERT_MSG(false, "Invalid vertex attribute type!");
        return Shader::AttributeType::Disabled;
    case Maxwell::VertexAttribute::Type::SNorm:
    case Maxwell::VertexAttribute::Type::UNorm:
    case Maxwell::VertexAttribute::Type::Float:
        return Shader::AttributeType::Float;
    case Maxwell::VertexAttribute::Type::SInt:
        return Shader::AttributeType::SignedInt;
    case Maxwell::VertexAttribute::Type::UInt:
        return Shader::AttributeType::UnsignedInt;
    case Maxwell::VertexAttribute::Type::UScaled:
        return Shader::AttributeType::UnsignedScaled;
    case Maxwell::VertexAttribute::Type::SScaled:
        return Shader::AttributeType::SignedScaled;
    }
    return Shader::AttributeType::Float;
}

Shader::AttributeType AttributeType(const FixedPipelineState& state, size_t index) {
    switch (state.DynamicAttributeType(index)) {
    case 0:
        return Shader::AttributeType::Disabled;
    case 1:
        return Shader::AttributeType::Float;
    case 2:
        return Shader::AttributeType::SignedInt;
    case 3:
        return Shader::AttributeType::UnsignedInt;
    }
    return Shader::AttributeType::Disabled;
}

Shader::InputTopology MaxwellToInputTopology(Maxwell::PrimitiveTopology topology) {
    switch (topology) {
    case Maxwell::PrimitiveTopology::Points:
        return Shader::InputTopology::Points;
    case Maxwell::PrimitiveTopology::Lines:
    case Maxwell::PrimitiveTopology::LineLoop:
    case Maxwell::PrimitiveTopology::LineStrip:
        return Shader::InputTopology::Lines;
    case Maxwell::PrimitiveTopology::Triangles:
    case Maxwell::PrimitiveTopology::TriangleStrip:
    case Maxwell::PrimitiveTopology::TriangleFan:
    case Maxwell::PrimitiveTopology::Quads:
    case Maxwell::PrimitiveTopology::QuadStrip:
    case Maxwell::PrimitiveTopology::Polygon:
    case Maxwell::PrimitiveTopology::Patches:
        return Shader::InputTopology::Triangles;
    case Maxwell::PrimitiveTopology::LinesAdjacency:
    case Maxwell::PrimitiveTopology::LineStripAdjacency:
        return Shader::InputTopology::LinesAdjacency;
    case Maxwell::PrimitiveTopology::TrianglesAdjacency:
    case Maxwell::PrimitiveTopology::TriangleStripAdjacency:
        return Shader::InputTopology::TrianglesAdjacency;
    }
    return {};
}

Shader::TessPrimitive MaxwellToTessPrimitive(u32 raw) {
    switch (static_cast<Maxwell::Tessellation::DomainType>(raw)) {
    case Maxwell::Tessellation::DomainType::Isolines:
        return Shader::TessPrimitive::Isolines;
    case Maxwell::Tessellation::DomainType::Triangles:
        return Shader::TessPrimitive::Triangles;
    case Maxwell::Tessellation::DomainType::Quads:
        return Shader::TessPrimitive::Quads;
    }
    ASSERT(false);
    return Shader::TessPrimitive::Triangles;
}

Shader::TessSpacing MaxwellToTessSpacing(u32 raw) {
    switch (static_cast<Maxwell::Tessellation::Spacing>(raw)) {
    case Maxwell::Tessellation::Spacing::Integer:
        return Shader::TessSpacing::Equal;
    case Maxwell::Tessellation::Spacing::FractionalOdd:
        return Shader::TessSpacing::FractionalOdd;
    case Maxwell::Tessellation::Spacing::FractionalEven:
        return Shader::TessSpacing::FractionalEven;
    }
    ASSERT(false);
    return Shader::TessSpacing::Equal;
}

Shader::AttributeType ColorOutputType(u8 render_target_format) {
    const auto format = static_cast<Tegra::RenderTargetFormat>(render_target_format);
    const auto pixel_format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(format);
    if (!VideoCore::Surface::IsPixelFormatInteger(pixel_format)) {
        return Shader::AttributeType::Float;
    }
    return VideoCore::Surface::IsPixelFormatSignedInteger(pixel_format)
               ? Shader::AttributeType::SignedInt
               : Shader::AttributeType::UnsignedInt;
}

// What translation and SPIR-V emission read of a pipeline key, plus the settings they read
// (see vk_shader_reuse.h). MakeRuntimeInfo and the layer passthrough only get this key, so
// nothing they use can be left out of it. A field is only filled when a stage of the pipeline
// reads it: pipelines differing in state their shaders ignore then share the shaders.
ShaderTranslationKey MakeShaderTranslationKey(const GraphicsPipelineCacheKey& pipeline_key,
                                              const Device& device) {
    const FixedPipelineState& state{pipeline_key.state};
    const bool has_tessellation_eval{pipeline_key.unique_hashes[3] != 0};
    const bool has_geometry{pipeline_key.unique_hashes[4] != 0};
    const bool has_fragment{pipeline_key.unique_hashes[5] != 0};

    ShaderTranslationKey key{};
    key.unique_hashes = pipeline_key.unique_hashes;
    key.input_topology = MaxwellToInputTopology(state.topology);
    key.passthrough_topology = MaxwellToOutputTopology(state.topology);
    key.convert_depth_mode = state.ndc_minus_one_to_one != 0;
    key.force_early_z = state.early_z != 0;
    key.y_negate = state.y_negate != 0;

    if (state.dynamic_vertex_input) {
        for (size_t index = 0; index < Maxwell::NumVertexAttributes; ++index) {
            key.generic_input_types[index] = AttributeType(state, index);
        }
    } else {
        std::ranges::transform(state.attributes, key.generic_input_types.begin(),
                               &CastAttributeType);
    }
    // A fixed point size is written by the last vertex stage when points are drawn, and by a
    // geometry stage that outputs points (a generated passthrough one outputs what is drawn).
    if (key.input_topology == Shader::InputTopology::Points || has_geometry) {
        key.point_size = state.point_size;
    }
    if (state.xfb_enabled) {
        if (device.IsExtTransformFeedbackSupported()) {
            key.xfb_enabled = true;
            key.xfb_state = state.xfb_state;
        } else {
            LOG_WARNING(Render_Vulkan, "XFB requested in pipeline key but device lacks VK_EXT_transform_feedback; ignoring XFB decorations");
        }
    }
    if (has_tessellation_eval) {
        key.tess_clockwise = state.tessellation_clockwise != 0;
        key.tess_primitive = MaxwellToTessPrimitive(state.tessellation_primitive.Value());
        key.tess_spacing = MaxwellToTessSpacing(state.tessellation_spacing);
    }
    if (has_fragment) {
        key.alpha_test_func = MaxwellToCompareFunction(
            state.UnpackComparisonOp(state.alpha_test_func.Value()));
        key.alpha_test_reference = state.alpha_test_ref;
        key.dual_source_blend = state.attachment0_dual_source_blend != 0;
        if (device.IsMoltenVK()) {
            std::ranges::transform(state.color_formats, key.color_output_types.begin(),
                                   &ColorOutputType);
        }
    }

    const auto& resolution{Settings::values.resolution_info};
    key.resolution_up_scale = resolution.up_scale;
    key.resolution_down_shift = resolution.down_shift;
    key.rescaling_active = resolution.active;
    key.renderer_debug = Settings::values.renderer_debug.GetValue();
    key.disable_shader_loop_safety_checks =
        Settings::values.disable_shader_loop_safety_checks.GetValue();
    key.fix_bloom_effects = Settings::values.fix_bloom_effects.GetValue();
    return key;
}

void SetTransformFeedback(Shader::RuntimeInfo& info, const ShaderTranslationKey& key) {
    if (key.xfb_enabled) {
        auto [varyings, count] = VideoCommon::MakeTransformFeedbackVaryings(key.xfb_state);
        info.xfb_varyings = varyings;
        info.xfb_count = count;
    }
}

Shader::RuntimeInfo MakeRuntimeInfo(std::span<const Shader::IR::Program> programs,
                                    const ShaderTranslationKey& key,
                                    const Shader::IR::Program& program,
                                    const Shader::IR::Program* previous_program) {
    Shader::RuntimeInfo info;
    if (previous_program) {
        info.previous_stage_stores = previous_program->info.stores;
        info.previous_stage_legacy_stores_mapping = previous_program->info.legacy_stores_mapping;
        if (previous_program->is_geometry_passthrough) {
            info.previous_stage_stores.mask |= previous_program->info.passthrough.mask;
        }
    } else {
        info.previous_stage_stores.mask.set();
    }
    if (previous_program && previous_program->stage == Shader::Stage::Mesh) {
        for (const u32 line : previous_program->mesh.primitive_lines) {
            const auto attribute{static_cast<Shader::IR::Attribute>(line)};
            if (Shader::IR::IsGeneric(attribute)) {
                info.per_primitive_generics.set(Shader::IR::GenericAttributeIndex(attribute));
            }
        }
    }
    const Shader::Stage stage{program.stage};
    const bool has_geometry{key.unique_hashes[4] != 0 && !programs[4].is_geometry_passthrough};
    const float point_size{std::bit_cast<float>(key.point_size)};
    switch (stage) {
    case Shader::Stage::VertexB:
        if (!has_geometry) {
            if (key.input_topology == Shader::InputTopology::Points) {
                info.fixed_state_point_size = point_size;
            }
            SetTransformFeedback(info, key);
            info.convert_depth_mode = key.convert_depth_mode;
        }
        info.generic_input_types = key.generic_input_types;
        break;
    case Shader::Stage::TessellationEval:
        info.tess_clockwise = key.tess_clockwise;
        info.tess_primitive = key.tess_primitive;
        info.tess_spacing = key.tess_spacing;
        break;
    case Shader::Stage::Mesh:
        info.convert_depth_mode = key.convert_depth_mode;
        if (program.mesh.topology == 0) {
            info.fixed_state_point_size = point_size;
        }
        break;
    case Shader::Stage::Geometry:
        if (program.output_topology == Shader::OutputTopology::PointList) {
            info.fixed_state_point_size = point_size;
        }
        SetTransformFeedback(info, key);
        info.convert_depth_mode = key.convert_depth_mode;
        break;
    case Shader::Stage::Fragment:
        info.alpha_test_func = key.alpha_test_func;
        info.alpha_test_reference = std::bit_cast<float>(key.alpha_test_reference);
        info.dual_source_blend = key.dual_source_blend;
        info.color_output_types = key.color_output_types;
        break;
    default:
        break;
    }
    info.input_topology = key.input_topology;
    info.force_early_z = key.force_early_z;
    info.y_negate = key.y_negate;
    return info;
}

// EDEN_NX2_SHADER_REUSE=0: translate every graphics pipeline's shaders, as before (for A/B
// tests). Otherwise pipelines reuse the shaders of an earlier one when they fit (vk_shader_reuse.h).
bool IsShaderReuseAllowed() {
    const char* const value = std::getenv("EDEN_NX2_SHADER_REUSE");
    return value == nullptr || value[0] != '0';
}

size_t GetTotalPipelineWorkers() {
    const size_t max_core_threads =
        std::max<size_t>(static_cast<size_t>(std::thread::hardware_concurrency()), 2ULL) - 1ULL;
#ifdef __ANDROID__
    const int configured = AndroidSettings::values.pipeline_worker_count.GetValue();
    const int clamped = std::clamp(configured, 2, 8);
    const size_t desired = static_cast<size_t>(clamped);
    if (desired == 0) {
        return 1ULL;
    }
    return std::min(max_core_threads, desired);
#else
    return max_core_threads;
#endif
}

// How long the GPU thread waits for a native graphics pipeline it just queued for translation
// before skipping the draw.
constexpr std::chrono::microseconds TRANSLATION_WAIT_BUDGET{3000};

// Translation threads also run the driver compile of what they translate. They get half of the
// pipeline workers' count, so they cannot hold up the driver compiles queued on those (compute
// pipelines wait for theirs).
size_t GetTotalTranslationWorkers() {
    return std::max<size_t>(1, GetTotalPipelineWorkers() / 2);
}

} // Anonymous namespace

size_t ComputePipelineCacheKey::Hash() const noexcept {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(this), sizeof *this);
    return static_cast<size_t>(hash);
}

bool ComputePipelineCacheKey::operator==(const ComputePipelineCacheKey& rhs) const noexcept {
    return std::memcmp(&rhs, this, sizeof *this) == 0;
}

size_t GraphicsPipelineCacheKey::Hash() const noexcept {
    const u64 hash = Common::CityHash64(reinterpret_cast<const char*>(this), Size());
    return static_cast<size_t>(hash);
}

bool GraphicsPipelineCacheKey::operator==(const GraphicsPipelineCacheKey& rhs) const noexcept {
    return std::memcmp(&rhs, this, Size()) == 0;
}

PipelineCache::PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory_,
                             const Device& device_, Scheduler& scheduler_,
                             DescriptorPool& descriptor_pool_,
                             GuestDescriptorQueue& guest_descriptor_queue_,
                             DescriptorBufferRing& descriptor_buffer_ring_,
                             RenderPassCache& render_pass_cache_, BufferCache& buffer_cache_,
                             TextureCache& texture_cache_, VideoCore::ShaderNotify& shader_notify_)
    : VideoCommon::ShaderCache{device_memory_}, device{device_}, scheduler{scheduler_},
      descriptor_pool{descriptor_pool_}, guest_descriptor_queue{guest_descriptor_queue_},
      descriptor_buffer_ring{descriptor_buffer_ring_},
      render_pass_cache{render_pass_cache_}, buffer_cache{buffer_cache_},
      texture_cache{texture_cache_}, shader_notify{shader_notify_},
      use_asynchronous_shaders{Settings::values.use_asynchronous_shaders.GetValue()},
      use_vulkan_pipeline_cache{Settings::values.use_vulkan_driver_pipeline_cache.GetValue()},
      // Translation threads compile in parallel with the pipeline workers: not on drivers that
      // break with parallel compiles.
      use_async_translation{use_asynchronous_shaders &&
                            !device.HasBrokenParallelShaderCompiling()},
      use_shader_reuse{IsShaderReuseAllowed()},
      translation_wait_budget{TRANSLATION_WAIT_BUDGET},
      num_translation_threads{use_async_translation ? GetTotalTranslationWorkers() : 0},
      workers(device.HasBrokenParallelShaderCompiling() ? 1ULL : GetTotalPipelineWorkers(),
              "VkPipelineBuilder", {}, Common::ThreadPlacement::Background),
      serialization_thread(1, "VkPipelineSerialization", {},
                           Common::ThreadPlacement::Background),
      translation_workers(num_translation_threads, "VkPipelineTranslator", {},
                          Common::ThreadPlacement::Background) {
    // [DEBUG] read-only GPU memory access for the n2Den debug server's read_gpu.
    VideoCore::N2DenDebug::SetGpuReader([this](u64 address, std::span<u8> out) {
        Tegra::MemoryManager* const memory{gpu_memory};
        if (!memory || !memory->IsWithinGPUAddressRange(address) ||
            !memory->IsFullyMappedRange(address, out.size())) {
            return false;
        }
        memory->ReadBlockUnsafe(address, out.data(), out.size());
        return true;
    });
    if (use_async_translation) {
        LOG_INFO(Render_Vulkan,
                 "Native graphics pipelines are translated on {} worker threads, wait budget "
                 "{:.1f} ms",
                 num_translation_threads, translation_wait_budget.count() / 1000.0);
    }
    LOG_INFO(Render_Vulkan, "Graphics pipelines {} the shaders of earlier pipelines that fit them",
             use_shader_reuse ? "reuse" : "do not reuse (EDEN_NX2_SHADER_REUSE=0)");
    const auto& float_control{device.FloatControlProperties()};
    const VkDriverId driver_id{device.GetDriverID()};
    const VkShaderStageFlags subgroup_stages{device.GetSubgroupSupportedStages()};
    const auto subgroup_stage_bit{[subgroup_stages](VkShaderStageFlags flag, Shader::Stage stage) {
        return (subgroup_stages & flag) != 0 ? (1u << static_cast<u32>(stage)) : 0u;
    }};
    u32 supported_subgroup_stages{
        subgroup_stage_bit(VK_SHADER_STAGE_VERTEX_BIT, Shader::Stage::VertexA) |
        subgroup_stage_bit(VK_SHADER_STAGE_VERTEX_BIT, Shader::Stage::VertexB) |
        subgroup_stage_bit(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                           Shader::Stage::TessellationControl) |
        subgroup_stage_bit(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
                           Shader::Stage::TessellationEval) |
        subgroup_stage_bit(VK_SHADER_STAGE_GEOMETRY_BIT, Shader::Stage::Geometry) |
        subgroup_stage_bit(VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Stage::Fragment) |
        subgroup_stage_bit(VK_SHADER_STAGE_COMPUTE_BIT, Shader::Stage::Compute) |
        subgroup_stage_bit(VK_SHADER_STAGE_TASK_BIT_EXT, Shader::Stage::Task) |
        subgroup_stage_bit(VK_SHADER_STAGE_MESH_BIT_EXT, Shader::Stage::Mesh)};
    if (driver_id == VK_DRIVER_ID_MESA_TURNIP) {
        supported_subgroup_stages &= ~(1u << static_cast<u32>(Shader::Stage::Geometry));
    }
    profile = Shader::Profile{
        .supported_spirv = device.SupportedSpirvVersion(),
        .unified_descriptor_binding = true,
        .support_descriptor_aliasing = device.IsDescriptorAliasingSupported(),
        .support_int8 = device.IsInt8Supported(),
        .support_uniform_and_storage_buffer_8bit =
            device.IsUniformAndStorageBuffer8BitAccessSupported(),
        .support_storage_buffer_8bit = device.IsStorageBuffer8BitAccessSupported(),
        .support_int16 = device.IsShaderInt16Supported(),
        .support_uniform_and_storage_buffer_16bit =
            device.IsUniformAndStorageBuffer16BitAccessSupported(),
        .support_storage_buffer_16bit = device.IsStorageBuffer16BitAccessSupported(),
        .support_int64 = device.IsShaderInt64Supported(),
        .support_vertex_instance_id = false,
        .support_float_controls = device.IsKhrShaderFloatControlsSupported(),
        .support_separate_denorm_behavior =
            float_control.denormBehaviorIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
        .support_separate_rounding_mode =
            float_control.roundingModeIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
        .support_fp16_denorm_preserve = float_control.shaderDenormPreserveFloat16 != VK_FALSE,
        .support_fp32_denorm_preserve = float_control.shaderDenormPreserveFloat32 != VK_FALSE,
        .support_fp16_denorm_flush = float_control.shaderDenormFlushToZeroFloat16 != VK_FALSE,
        .support_fp32_denorm_flush = float_control.shaderDenormFlushToZeroFloat32 != VK_FALSE,
        .support_fp16_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat16 != VK_FALSE,
        .support_fp32_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat32 != VK_FALSE,
        .support_fp64_signed_zero_nan_preserve =
            float_control.shaderSignedZeroInfNanPreserveFloat64 != VK_FALSE,
        .support_explicit_workgroup_layout = device.IsKhrWorkgroupMemoryExplicitLayoutSupported(),
        .support_workgroup_layout_8bit_access =
            device.IsWorkgroupMemoryExplicitLayout8BitAccessSupported(),
        .support_workgroup_layout_16bit_access =
            device.IsWorkgroupMemoryExplicitLayout16BitAccessSupported(),
        .support_shader_quad_control = device.IsKhrShaderQuadControlSupported(),
        .support_quad_shuffles = device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_QUAD_BIT),
        .support_vote = device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_VOTE_BIT),
        .support_shuffle_relative =
            device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT),
        .supported_subgroup_stages = supported_subgroup_stages,
        .support_viewport_index_layer_non_geometry =
            device.IsExtShaderViewportIndexLayerSupported(),
        .support_viewport_mask = device.IsNvViewportArray2Supported(),
        .support_typeless_image_loads = device.IsFormatlessImageLoadSupported(),
        .support_demote_to_helper_invocation =
            device.IsExtShaderDemoteToHelperInvocationSupported(),
        .support_int64_atomics = device.IsExtShaderAtomicInt64Supported(),
        .support_shared_int64_atomics = device.IsSharedInt64AtomicsSupported(),
        .support_derivative_control = true,
        .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
        .support_native_ndc = device.IsExtDepthClipControlSupported(),
        .support_scaled_attributes = !device.MustEmulateScaledFormats(),
        .support_multi_viewport = device.SupportsMultiViewport(),
        .support_geometry_streams = device.AreTransformFeedbackGeometryStreamsSupported(),
        .support_sampled_image_array_nonuniform_indexing =
            device.IsSampledImageArrayNonUniformIndexingSupported(),
        .support_storage_image_array_nonuniform_indexing =
            device.IsStorageImageArrayNonUniformIndexingSupported(),
        .support_uniform_texel_buffer_array_nonuniform_indexing =
            device.IsUniformTexelBufferArrayNonUniformIndexingSupported(),
        .support_storage_texel_buffer_array_nonuniform_indexing =
            device.IsStorageTexelBufferArrayNonUniformIndexingSupported(),

        .warp_size_potentially_larger_than_guest = device.IsWarpSizePotentiallyBiggerThanGuest(),

        .lower_left_origin_mode = false,
        .need_declared_frag_colors = false,
        .need_gather_subpixel_offset = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                       driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
                                       driver_id == VK_DRIVER_ID_MESA_RADV ||
                                       driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS ||
                                       driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA ||
                                       driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY ||
                                       driver_id == VK_DRIVER_ID_MESA_TURNIP,

        .has_broken_spirv_clamp = driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS,
        .has_broken_spirv_position_input = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .has_broken_unsigned_image_offsets = false,
        .has_broken_signed_operations = false,
        .has_broken_fp16_float_controls = driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY,
        .has_broken_fp32_denorm_flush = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .ignore_nan_fp_comparisons = false,
        .has_broken_spirv_subgroup_mask_vector_extract_dynamic =
            driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY &&
            device.GetDriverVersion() < VK_MAKE_VERSION(512, 672, 0),
        .has_broken_spirv_subgroup_shuffle = driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY,
        .max_subgroup_size = device.GetMaxSubgroupSize(),
        // The cooperative matrix module keeps the GLSL450 memory model (spirv-val wants the Vulkan
        // one with CooperativeMatrixKHR); verified on NVIDIA only, so other drivers emulate.
        .support_cooperative_matrix_f16_16816 = device.IsCooperativeMatrixF16_16816Supported() &&
                                                driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY,
        .max_compute_shared_memory_size = device.GetMaxComputeSharedMemorySize(),
        .has_broken_robust =
            device.IsNvidia() && device.GetNvidiaArch() <= NvidiaArchitecture::Arch_Pascal,
        .min_ssbo_alignment = device.GetStorageBufferAlignment(),
        .max_user_clip_distances = device.GetMaxUserClipDistances()
    };

    host_info = Shader::HostTranslateInfo{
        .min_ssbo_alignment = device.GetStorageBufferAlignment(),
        .max_per_stage_descriptor_sampled_images = device.GetMaxPerStageDescriptorSampledImages(),
        .max_per_stage_resources = device.GetMaxPerStageResources(),
        .max_per_stage_descriptor_storage_buffers = device.GetMaxPerStageDescriptorStorageBuffers(),
        .max_descriptor_set_samplers = device.GetMaxDescriptorSetSamplers(),
        .max_descriptor_set_uniform_buffers = device.GetMaxDescriptorSetUniformBuffers(),
        .max_descriptor_set_uniform_buffers_dynamic = device.GetMaxDescriptorSetUniformBuffersDynamic(),
        .max_descriptor_set_storage_buffers = device.GetMaxDescriptorSetStorageBuffers(),
        .max_descriptor_set_storage_buffers_dynamic = device.GetMaxDescriptorSetStorageBuffersDynamic(),
        .max_descriptor_set_sampled_images = device.GetMaxDescriptorSetSampledImages(),
        .max_descriptor_set_storage_images = device.GetMaxDescriptorSetStorageImages(),
        .max_descriptor_set_input_attachements = device.GetMaxDescriptorSetInputAttachments(),
        .support_float64 = device.IsFloat64Supported(),
        .support_float16 = device.IsFloat16Supported(),
        .support_int64 = device.IsShaderInt64Supported(),
        .needs_demote_reorder = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
                                driver_id == VK_DRIVER_ID_SAMSUNG_PROPRIETARY,
        .support_snorm_render_buffer = true,
        .support_viewport_index_layer = device.IsExtShaderViewportIndexLayerSupported(),
        .support_viewport_mask = device.IsNvViewportArray2Supported(),
        .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
        .support_conditional_barrier = device.SupportsConditionalBarriers(),
        .single_lane_geometry_subgroups = !profile.SupportsSubgroupStage(Shader::Stage::Geometry),
        .support_buffer_texture_handles = profile.support_sampled_image_array_nonuniform_indexing,
    };
    host_info.ApplyDescriptorLimitPolicy();
    // Texture pool lookups (vk_texture_pool.h). Default fallback: the handles nothing per draw
    // expresses are read from the pool (Ocarina of Time needs it). EDEN_NX2_TEXTURE_POOL=all reads
    // every handle from memory, =off turns the pool off.
    if (device.IsTexturePoolSupported()) {
        const char* const pool_mode = std::getenv("EDEN_NX2_TEXTURE_POOL");
        const std::string_view mode{pool_mode ? pool_mode : "fallback"};
        host_info.texture_pool_mode = mode == "all"        ? Shader::TexturePoolMode::All
                                      : mode == "fallback" ? Shader::TexturePoolMode::Fallback
                                                           : Shader::TexturePoolMode::Off;
    }

    if (device.GetMaxVertexInputAttributes() < Maxwell::NumVertexAttributes) {
        LOG_WARNING(Render_Vulkan, "maxVertexInputAttributes is too low: {} < {}",
                    device.GetMaxVertexInputAttributes(), Maxwell::NumVertexAttributes);
    }
    if (device.GetMaxVertexInputBindings() < Maxwell::NumVertexArrays) {
        LOG_WARNING(Render_Vulkan, "maxVertexInputBindings is too low: {} < {}",
                    device.GetMaxVertexInputBindings(), Maxwell::NumVertexArrays);
    }

    LOG_INFO(Render_Vulkan, "DynamicState setting value: {}", u32(Settings::values.dyna_state.GetValue()));

    dynamic_features = {};
    dynamic_features.driver_id = device.GetDriverID();
    dynamic_features.driver_version = device.GetDriverVersion();

    // User granularity enforced in vulkan_device.cpp switch statement:
    //   Level 0: Core Dynamic States only
    //   Level 1: Core + EDS1
    //   Level 2: Core + EDS1 + EDS2 (accumulative)
    //   Level 3: Core + EDS1 + EDS2 + EDS3 (accumulative)
    // Here we only verify if extensions were successfully loaded by the device

    dynamic_features.has_extended_dynamic_state =
        device.IsExtExtendedDynamicStateSupported();

    dynamic_features.has_extended_dynamic_state_2 =
        device.IsExtExtendedDynamicState2Supported();
    dynamic_features.has_extended_dynamic_state_2_logic_op =
        device.IsExtExtendedDynamicState2ExtrasSupported();
    dynamic_features.has_extended_dynamic_state_2_patch_control_points = false;

    dynamic_features.has_extended_dynamic_state_3_blend =
        device.IsExtExtendedDynamicState3BlendingSupported();
    dynamic_features.has_extended_dynamic_state_3_enables =
        device.IsExtExtendedDynamicState3EnablesSupported();
    dynamic_features.has_color_write_enable =
        device.IsExtColorWriteEnableSupported();
    dynamic_features.has_dynamic_state3_depth_clamp_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3DepthClampEnable();
    dynamic_features.has_dynamic_state3_logic_op_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3LogicOpEnable();
    dynamic_features.has_dynamic_state3_line_stipple_enable =
        dynamic_features.has_extended_dynamic_state_3_enables &&
        device.SupportsDynamicState3LineStippleEnable();

    // VIDS: Independent toggle (not affected by dyna_state levels)
    dynamic_features.has_dynamic_vertex_input =
        device.IsExtVertexInputDynamicStateSupported() &&
        Settings::values.vertex_input_dynamic_state.GetValue();

    dynamic_features.has_provoking_vertex = device.IsExtProvokingVertexSupported();
    dynamic_features.has_provoking_vertex_first_mode =
        device.SupportsProvokingVertexFirstMode();
    dynamic_features.has_provoking_vertex_last_mode =
        device.SupportsProvokingVertexLastMode();
    dynamic_features.has_provoking_vertex_tf_preserve =
        device.SupportsTransformFeedbackProvokingVertexPreservation();
}

PipelineCache::~PipelineCache() {
    VideoCore::N2DenDebug::SetGpuReader({});
    VideoCore::NativeCensus::LogSummary();
    if (use_vulkan_pipeline_cache && !vulkan_pipeline_cache_filename.empty()) {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
    }
}

GraphicsPipeline* PipelineCache::CurrentGraphicsPipeline() {
    if (num_translated.load(std::memory_order_acquire) != 0) {
        AdoptTranslatedGraphicsPipelines();
    }
    if (!RefreshStages(graphics_key.unique_hashes)) {
        current_pipeline = nullptr;
        return nullptr;
    }
    graphics_key.state.Refresh(*maxwell3d, dynamic_features);
    if (VideoCore::N2DenDebug::Enabled() && VideoCore::N2DenDebug::AnyPendingDump(graphics_key.unique_hashes)) {
        DebugDumpCurrentGraphics();
    }

    if (current_pipeline) {
        GraphicsPipeline* const next{current_pipeline->Next(graphics_key)};
        if (next) {
            current_pipeline = next;
            return BuiltPipeline(current_pipeline);
        }
    }
    return CurrentGraphicsPipelineSlowPath();
}

void PipelineCache::DebugDumpCurrentGraphics() try {
    // dump_ir for programs whose pipeline already exists: translate once more for the dump only.
    GraphicsEnvironments environments;
    GetGraphicsEnvironments(environments, graphics_key.unique_hashes);
    main_pools.ReleaseContents();
    const ShaderTranslationKey translation_key{MakeShaderTranslationKey(graphics_key, device)};
    (void)BuildGraphicsShaders(main_pools, translation_key, environments.Span(), graphics_key.Hash());
} catch (const std::exception& error) {
    LOG_ERROR(Render_Vulkan, "n2Den debug dump translation failed: {}", error.what());
    for (const u64 hash : graphics_key.unique_hashes) {
        if (const auto dump_key = VideoCore::N2DenDebug::PendingDump(hash)) {
            VideoCore::N2DenDebug::NoteDump(*dump_key, hash, {{"error", error.what()}});
        }
    }
}

ComputePipeline* PipelineCache::CurrentComputePipeline() {

    const ShaderInfo* const shader{ComputeShader()};
    if (!shader) {
        return nullptr;
    }
    const auto& qmd{kepler_compute->launch_description};
    const ComputePipelineCacheKey key{
        .unique_hash = shader->unique_hash,
        .shared_memory_size = qmd.shared_alloc,
        .workgroup_size{qmd.block_dim_x, qmd.block_dim_y, qmd.block_dim_z},
    };
    const auto [pair, is_new]{compute_cache.try_emplace(key)};
    auto& pipeline{pair->second};
    const bool created{is_new};
    auto* const pipeline_slot{&pair->second};
    SCOPE_EXIT {
        if (VideoCore::N2DenDebug::Enabled() &&
            (created || VideoCore::N2DenDebug::WatchedCompute(key.unique_hash))) {
            SnapshotComputeDispatch(key.unique_hash, pipeline_slot->get());
        }
    };
    if (!is_new && (VideoCore::N2DenDebug::TakeComputeRebuild(key.unique_hash) ||
                    VideoCore::N2DenDebug::PendingDump(key.unique_hash))) {
        // [DEBUG] debug-server request: retranslate against the live state.
        auto rebuilt{CreateComputePipeline(key, shader)};
        LOG_INFO(Render_Vulkan, "Compute pipeline {:016x} rebuilt on debug request ({})",
                 key.unique_hash, rebuilt ? "ok" : "failed");
        if (pipeline) {
            retired_compute_pipelines.push_back(std::move(pipeline));
        }
        pipeline = std::move(rebuilt);
        return pipeline.get();
    }
    if (!is_new) {
        // A native program translated while one of its pointer slots was null (a buffer the game
        // hadn't allocated yet) has no binding for it, and the disk cache keeps that version for
        // later sessions. Rebuild it once the slot holds a buffer (Eglantine 688c6fa7d8).
        //
        // The rebuild is a full driver compile (seconds for OoT's large computes). It runs on a
        // worker without pausing the game; the current pipeline is dispatched meanwhile, as it
        // was before the slot filled, and is replaced once the rebuild is built.
        // [TEST] EDEN_N2DEN_SYNC_POINTER_REBUILD=1: rebuild synchronously (the game pauses), for A/B.
        static const bool sync_rebuild = std::getenv("EDEN_N2DEN_SYNC_POINTER_REBUILD") != nullptr;
        if (const auto pending = compute_rebuilds.find(key); pending != compute_rebuilds.end()) {
            if (!pending->second->IsBuilt()) {
                return pipeline.get();
            }
            std::unique_ptr<ComputePipeline> rebuilt{std::move(pending->second)};
            compute_rebuilds.erase(pending);
            if (!rebuilt->IsBound()) {
                // The driver build failed: keep the current pipeline and stop rebuilding it.
                pipeline->null_pointer_slots.clear();
                retired_compute_pipelines.push_back(std::move(rebuilt));
                return pipeline.get();
            }
            LOG_INFO(Render_Vulkan,
                     "Compute pipeline {:016x} replaced by its background rebuild ({} pointer "
                     "slots, {} still null)",
                     key.unique_hash, rebuilt->pointer_slots.size(),
                     rebuilt->null_pointer_slots.size());
            retired_compute_pipelines.push_back(std::move(pipeline));
            pipeline = std::move(rebuilt);
            return pipeline.get();
        }
        if (pipeline && !pipeline->null_pointer_slots.empty() && pipeline->pointer_rebuilds < 4 &&
            NullPointerSlotFilled(*pipeline)) {
            auto rebuilt{CreateComputePipeline(key, shader, pipeline.get(), !sync_rebuild)};
            if (!rebuilt) {
                pipeline->null_pointer_slots.clear();
            } else if (!sync_rebuild && !rebuilt->IsBuilt()) {
                LOG_INFO(Render_Vulkan,
                         "Compute pipeline {:016x}: a pointer slot null at its translation now "
                         "holds a buffer; rebuilding in the background",
                         key.unique_hash);
                compute_rebuilds.emplace(key, std::move(rebuilt));
            } else {
                LOG_INFO(Render_Vulkan,
                         "Compute pipeline {:016x} rebuilt: a pointer slot null at its translation "
                         "now holds a buffer ({} pointer slots, {} still null)",
                         key.unique_hash, rebuilt->pointer_slots.size(),
                         rebuilt->null_pointer_slots.size());
                retired_compute_pipelines.push_back(std::move(pipeline));
                pipeline = std::move(rebuilt);
            }
        }
        return pipeline.get();
    }
    pipeline = CreateComputePipeline(key, shader);
    return pipeline.get();
}

void PipelineCache::SnapshotComputeDispatch(u64 hash, const ComputePipeline* pipeline) const {
    using nlohmann::json;
    const auto& qmd{kepler_compute->launch_description};
    auto cbufs = json::array();
    for (u32 index = 0; index < qmd.const_buffer_config.size(); ++index) {
        if (((qmd.const_buffer_enable_mask.Value() >> index) & 1) == 0) {
            continue;
        }
        const auto& cbuf{qmd.const_buffer_config[index]};
        const GPUVAddr address{cbuf.Address()};
        const u32 words{std::min<u32>(cbuf.size / 4, 64)};
        auto values = json::array();
        if (address != 0 && gpu_memory->IsFullyMappedRange(address, words * 4)) {
            std::vector<u32> data(words);
            gpu_memory->ReadBlockUnsafe(address, data.data(), words * 4);
            for (const u32 word : data) values.push_back(fmt::format("{:08x}", word));
        }
        cbufs.push_back({{"index", index}, {"address", fmt::format("{:016x}", address)},
                         {"size", static_cast<u32>(cbuf.size)}, {"words", std::move(values)}});
    }
    const auto slots = [](const std::vector<u64>& in) {
        auto out = json::array();
        for (const u64 slot : in) {
            out.push_back(fmt::format("c{}[{:#x}]", slot >> 32, static_cast<u32>(slot)));
        }
        return out;
    };
    json update{{"program_base", fmt::format("{:016x}", kepler_compute->regs.code_loc.Address())},
                {"program_start", static_cast<u32>(qmd.program_start)},
                {"workgroup", {qmd.block_dim_x.Value(), qmd.block_dim_y.Value(), qmd.block_dim_z.Value()}},
                {"pipeline", pipeline ? "built" : "unavailable"},
                {"cbufs", std::move(cbufs)}};
    if (pipeline) {
        update["pointer_slots"] = slots(pipeline->pointer_slots);
        update["null_pointer_slots"] = slots(pipeline->null_pointer_slots);
        update["pointer_rebuilds"] = pipeline->pointer_rebuilds;
    }
    VideoCore::N2DenDebug::NoteCompute(hash, update);
}

bool PipelineCache::NullPointerSlotFilled(const ComputePipeline& pipeline) const {
    const auto& qmd{kepler_compute->launch_description};
    for (const u64 slot : pipeline.null_pointer_slots) {
        const u32 index = static_cast<u32>(slot >> 32);
        const u32 offset = static_cast<u32>(slot);
        if (index >= qmd.const_buffer_config.size() ||
            ((qmd.const_buffer_enable_mask.Value() >> index) & 1) == 0) {
            continue;
        }
        const auto& cbuf{qmd.const_buffer_config[index]};
        if (offset + sizeof(u64) > cbuf.size) {
            continue;
        }
        const GPUVAddr address = gpu_memory->Read<u64>(cbuf.Address() + offset);
        if (address != 0 && gpu_memory->IsWithinGPUAddressRange(address) &&
            gpu_memory->IsFullyMappedRange(address, 1)) {
            return true;
        }
    }
    return false;
}

void PipelineCache::LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                                      const VideoCore::DiskResourceLoadCallback& callback) {
    if (title_id == 0) {
        return;
    }
    if (!pipeline_cache_filename.empty()) {
        serialization_thread.WaitForRequests();
        if (use_vulkan_pipeline_cache && !vulkan_pipeline_cache_filename.empty()) {
            SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                         CACHE_VERSION);
        }
    }

    // EDEN_NX2_NVN_CACHE_DIR (runs that must not share the profile's shader caches): this cache
    // goes to <dir>/legacy too.
    const char* const own_dir = std::getenv("EDEN_NX2_NVN_CACHE_DIR");
    const bool own = own_dir != nullptr && *own_dir != '\0';
    const auto shader_dir{own ? std::filesystem::path{own_dir}
                              : Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir)};
    const auto base_dir{own ? shader_dir / "legacy"
                            : shader_dir / fmt::format("{:016x}", title_id)};
    if (!Common::FS::CreateDirs(base_dir)) {
        LOG_ERROR(Common_Filesystem, "Failed to create pipeline cache directories");
        return;
    }
    pipeline_cache_filename = base_dir / "vulkan.bin";
    mesh_pipeline_cache_filename = base_dir / "vulkan_mesh.bin";

    if (use_vulkan_pipeline_cache) {
        vulkan_pipeline_cache_filename = base_dir / "vulkan_pipelines.bin";
        vulkan_pipeline_cache =
            LoadVulkanPipelineCache(vulkan_pipeline_cache_filename, CACHE_VERSION);
    }

    struct {
        std::mutex mutex;
        size_t total{};
        size_t built{};
        bool has_loaded{};
        std::unique_ptr<PipelineStatistics> statistics;
        std::unordered_set<GraphicsPipelineCacheKey> graphics_keys;
        size_t duplicates{};
        // Entries stored by an older translator and built again by this one: the 8-byte fields to
        // rewrite in the cache file (salt, key hash) so the next boot loads them as current.
        std::vector<std::pair<s64, u64>> file_patches;
        size_t rebuilt{};
        size_t rebuild_failed{};
        size_t stale_skipped{};
    } state;
    const bool rebuild_stale{Settings::values.rebuild_stale_pipeline_cache.GetValue()};
    constexpr u64 current_salt{Shader::Ampere::NATIVE_TRANSLATOR_SALT};

    if (device.IsKhrPipelineExecutablePropertiesEnabled()) {
        state.statistics = std::make_unique<PipelineStatistics>(device);
    }
    // Entries from an older translator wait until the whole file is read: an entry the current
    // translator stored for the same pipeline wins over them, and among themselves the newest does.
    struct StaleCompute {
        ComputePipelineCacheKey key;
        FileEnvironment env;
        std::vector<std::pair<s64, u64>> patches;
    };
    struct StaleGraphics {
        GraphicsPipelineCacheKey key;
        std::vector<FileEnvironment> envs;
        std::vector<std::pair<s64, u64>> patches;
    };
    std::vector<StaleCompute> stale_compute;
    std::vector<StaleGraphics> stale_graphics;
    std::unordered_set<ComputePipelineCacheKey> compute_keys;

    const auto queue_compute{[&](const ComputePipelineCacheKey& key, FileEnvironment env,
                                 std::vector<std::pair<s64, u64>> patches) {
        workers.QueueWork([this, key, env_ = std::move(env), patches_ = std::move(patches), &state,
                           &callback]() mutable {
            ShaderPools pools;
            auto pipeline{CreateComputePipeline(pools, key, env_, state.statistics.get(), false)};
            if (pipeline) {
                env_.StoragePointerSlots(pipeline->pointer_slots, pipeline->null_pointer_slots);
            }
            std::scoped_lock lock{state.mutex};
            if (!patches_.empty()) {
                if (pipeline) {
                    ++state.rebuilt;
                    state.file_patches.insert(state.file_patches.end(), patches_.begin(),
                                              patches_.end());
                } else {
                    ++state.rebuild_failed;
                }
            }
            if (pipeline) {
                // A program rebuilt for a filled pointer slot has a second record: keep the one
                // that binds more pointers.
                auto& slot{compute_cache[key]};
                if (!slot || slot->pointer_slots.size() < pipeline->pointer_slots.size()) {
                    slot = std::move(pipeline);
                }
            }
            ++state.built;
            if (state.has_loaded) {
                callback(VideoCore::LoadCallbackStage::Build, state.built, state.total);
            }
        });
        ++state.total;
    }};
    const auto load_compute{[&](std::ifstream& file, FileEnvironment env) {
        const s64 key_offset{static_cast<s64>(file.tellg())};
        ComputePipelineCacheKey key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));
        if (!env.IsStale()) {
            compute_keys.insert(key);
            queue_compute(key, std::move(env), {});
            return;
        }
        if (!rebuild_stale) {
            ++state.stale_skipped;
            return;
        }
        // The salt is XORed into the program hash, so the hash under the current translator
        // follows from the stored one.
        key.unique_hash ^= env.StoredSalt() ^ current_salt;
        std::vector<std::pair<s64, u64>> patches;
        patches.emplace_back(env.SaltFileOffset(), current_salt);
        patches.emplace_back(
            key_offset + static_cast<s64>(offsetof(ComputePipelineCacheKey, unique_hash)),
            key.unique_hash);
        env.AcceptStale();
        stale_compute.push_back({key, std::move(env), std::move(patches)});
    }};
    // Set while vulkan_mesh.bin loads (after vulkan.bin and its stale entries).
    bool loading_mesh_file = false;
    const auto queue_graphics{[&](GraphicsPipelineCacheKey key, std::vector<FileEnvironment> envs,
                                  std::vector<std::pair<s64, u64>> patches) {
        // Task/mesh pipelines live in vulkan_mesh.bin, with the mesh registers they are translated
        // with; vulkan.bin doesn't carry those, so skip any it holds.
        if (!loading_mesh_file && std::ranges::any_of(envs, [](const FileEnvironment& env) {
                return env.ShaderStage() == Shader::Stage::Task ||
                       env.ShaderStage() == Shader::Stage::Mesh;
            })) {
            return;
        }
        // Older caches stored the alpha reference even with the test off (see
        // FixedPipelineState::Refresh); those keys are never requested again as stored.
        if (key.state.alpha_test_func.Value() ==
            FixedPipelineState::PackComparisonOp(Maxwell::ComparisonOp::Always_GL)) {
            key.state.alpha_test_ref = 0;
        }
        // Runtime keys carry the running driver's id and version (FixedPipelineState::Refresh).
        // After a driver update the stored ones are older, so every pipeline built here would
        // miss at run time and be built and stored again. From Eglantine ee560f97ab.
        key.state.driver_id = dynamic_features.driver_id;
        key.state.driver_version = dynamic_features.driver_version;
        if (!state.graphics_keys.insert(key).second) {
            ++state.duplicates;
            return;
        }
        if ((key.state.extended_dynamic_state != 0) !=
                dynamic_features.has_extended_dynamic_state ||
            (key.state.extended_dynamic_state_2 != 0) !=
                dynamic_features.has_extended_dynamic_state_2 ||
            (key.state.extended_dynamic_state_2_logic_op != 0) !=
                dynamic_features.has_extended_dynamic_state_2_logic_op ||
            (key.state.extended_dynamic_state_3_blend != 0) !=
                dynamic_features.has_extended_dynamic_state_3_blend ||
            (key.state.extended_dynamic_state_3_enables != 0) !=
                dynamic_features.has_extended_dynamic_state_3_enables ||
            (key.state.color_write_enable_dynamic != 0) !=
                dynamic_features.has_color_write_enable ||
            (key.state.dynamic_vertex_input != 0) !=
                dynamic_features.has_dynamic_vertex_input) {
            return;
        }

        const bool key_requests_provoking_last = key.state.provoking_vertex_last != 0;
        if (key_requests_provoking_last && !dynamic_features.has_provoking_vertex_last_mode) {
            return;
        }

        const bool key_uses_transform_feedback = key.state.xfb_enabled != 0;
        if (key_uses_transform_feedback && key_requests_provoking_last &&
            !dynamic_features.has_provoking_vertex_tf_preserve) {
            return;
        }

        workers.QueueWork([this, key, envs_ = std::move(envs), patches_ = std::move(patches),
                           &state, &callback]() mutable {
            ShaderPools pools;
            boost::container::static_vector<Shader::Environment*, 5> env_ptrs;
            for (auto& env : envs_) {
                env_ptrs.push_back(&env);
            }
            auto pipeline{CreateGraphicsPipeline(pools, key, MakeSpan(env_ptrs),
                                                 state.statistics.get(), false)};

            std::scoped_lock lock{state.mutex};
            if (!patches_.empty()) {
                if (pipeline) {
                    ++state.rebuilt;
                    state.file_patches.insert(state.file_patches.end(), patches_.begin(),
                                              patches_.end());
                } else {
                    ++state.rebuild_failed;
                }
            }
            if (pipeline) {
                graphics_cache.emplace(key, std::move(pipeline));
            }
            ++state.built;
            if (state.has_loaded) {
                callback(VideoCore::LoadCallbackStage::Build, state.built, state.total);
            }
        });
        ++state.total;
    }};
    const auto load_graphics{[&](std::ifstream& file, std::vector<FileEnvironment> envs) {
        const s64 key_offset{static_cast<s64>(file.tellg())};
        GraphicsPipelineCacheKey key;
        file.read(reinterpret_cast<char*>(&key), sizeof(key));

        if (!std::ranges::any_of(envs, &FileEnvironment::IsStale)) {
            queue_graphics(key, std::move(envs), {});
            return;
        }
        std::vector<std::pair<s64, u64>> patches;
        {
            // Mesh file entries are not patched in place (the patches go to vulkan.bin): a stale
            // one is skipped and stored again when built.
            if (!rebuild_stale || loading_mesh_file) {
                ++state.stale_skipped;
                return;
            }
            // The stored environments are the programs of the key's non-zero hashes, in order.
            size_t env_index{};
            for (size_t index = 0; index < key.unique_hashes.size(); ++index) {
                if (key.unique_hashes[index] == 0) {
                    continue;
                }
                if (env_index >= envs.size()) {
                    break;
                }
                FileEnvironment& env{envs[env_index++]};
                if (!env.IsStale()) {
                    continue;
                }
                key.unique_hashes[index] ^= env.StoredSalt() ^ current_salt;
                patches.emplace_back(env.SaltFileOffset(), current_salt);
                patches.emplace_back(
                    key_offset + static_cast<s64>(offsetof(GraphicsPipelineCacheKey, unique_hashes) +
                                                  index * sizeof(u64)),
                    key.unique_hashes[index]);
                env.AcceptStale();
            }
            if (std::ranges::any_of(envs, &FileEnvironment::IsStale)) {
                ++state.stale_skipped;
                return;
            }
        }
        stale_graphics.push_back({key, std::move(envs), std::move(patches)});
    }};
    VideoCommon::LoadPipelines(stop_loading, pipeline_cache_filename, CACHE_VERSION, load_compute,
                               load_graphics);

    // Newest first: the file is appended to, so later entries are more recent.
    for (auto it = stale_compute.rbegin(); it != stale_compute.rend(); ++it) {
        if (!compute_keys.insert(it->key).second) {
            ++state.duplicates;
            continue;
        }
        queue_compute(it->key, std::move(it->env), std::move(it->patches));
    }
    for (auto it = stale_graphics.rbegin(); it != stale_graphics.rend(); ++it) {
        queue_graphics(it->key, std::move(it->envs), std::move(it->patches));
    }
    stale_compute.clear();
    stale_graphics.clear();
    if (device.IsExtMeshShaderSupported()) {
        // Ocarina of Time's terrain and foliage: built again every session before this file
        // (Eglantine bc121caf3e by Duck912, after Frizti's 3a2c8980ad).
        loading_mesh_file = true;
        VideoCommon::LoadPipelines(stop_loading, mesh_pipeline_cache_filename, CACHE_VERSION,
                                   load_compute, load_graphics, true);
    }

    LOG_INFO(Render_Vulkan, "Total Pipeline Count: {} ({} duplicate keys skipped)", state.total,
             state.duplicates);
    state.graphics_keys = {};

    std::unique_lock lock{state.mutex};
    callback(VideoCore::LoadCallbackStage::Build, 0, state.total);
    state.has_loaded = true;
    lock.unlock();

    workers.WaitForRequests(stop_loading);

    {
        // Mark the entries that were built again as current, in place: only fixed-size fields
        // change, and nothing appends to the file until the game runs.
        std::scoped_lock patch_lock{state.mutex};
        if (!state.file_patches.empty()) {
            std::fstream cache_file{pipeline_cache_filename,
                                    std::ios::binary | std::ios::in | std::ios::out};
            for (const auto& [offset, value] : state.file_patches) {
                if (!cache_file || offset < 0) {
                    break;
                }
                cache_file.seekp(offset);
                cache_file.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
            if (!cache_file) {
                LOG_ERROR(Render_Vulkan, "Could not update rebuilt entries in the pipeline cache");
            }
        }
        if (state.rebuilt != 0 || state.rebuild_failed != 0 || state.stale_skipped != 0) {
            LOG_INFO(Render_Vulkan,
                     "Pipeline cache entries from an older translator: {} rebuilt, {} could not be "
                     "rebuilt, {} skipped",
                     state.rebuilt, state.rebuild_failed, state.stale_skipped);
        }
    }

    if (use_vulkan_pipeline_cache) {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
        size_t size = 0;
        vulkan_pipeline_cache.Read(&size, nullptr);
        last_cache_size.store(size, std::memory_order_relaxed);
        last_flush = std::chrono::steady_clock::now();
    }

    if (state.statistics) {
        state.statistics->Report();
    }
}

void PipelineCache::QueueVulkanPipelineCacheFlush() {
    if (!use_vulkan_pipeline_cache || vulkan_pipeline_cache_filename.empty()) {
        return;
    }
    if (++pipelines_since_flush < VULKAN_CACHE_FLUSH_PIPELINES) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const auto megabytes = last_cache_size.load(std::memory_order_relaxed) / (1024 * 1024);
    const std::chrono::seconds interval{
        std::max<size_t>(VULKAN_CACHE_FLUSH_MIN_SECONDS, megabytes)};
    if (last_flush.time_since_epoch().count() != 0 && now - last_flush < interval) {
        return;
    }
    if (flush_in_flight.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    pipelines_since_flush = 0;
    last_flush = now;
    serialization_thread.QueueWork([this] {
        SerializeVulkanPipelineCache(vulkan_pipeline_cache_filename, vulkan_pipeline_cache,
                                     CACHE_VERSION);
        size_t size = 0;
        vulkan_pipeline_cache.Read(&size, nullptr);
        last_cache_size.store(size, std::memory_order_relaxed);
        flush_in_flight.store(false, std::memory_order_release);
    });
}

GraphicsPipeline* PipelineCache::CurrentGraphicsPipelineSlowPath() {
    auto it{graphics_cache.find(graphics_key)};
    if (it == graphics_cache.end()) {
        if (translating_graphics.contains(graphics_key)) {
            // Its translation is still running on a worker thread: the draw is dropped on
            // purpose, as for a pipeline the driver has not built yet.
            NoteSkippedDraw("translates");
            draw_deferred = true;
            return nullptr;
        }
        if (!BuildNewGraphicsPipeline()) {
            NoteSkippedDraw("translates");
            draw_deferred = true;
            return nullptr;
        }
        it = graphics_cache.find(graphics_key);
    }
    auto& pipeline{it->second};
    if (!pipeline) {
        return nullptr;
    }
    if (current_pipeline) {
        current_pipeline->AddTransition(pipeline.get());
    }
    current_pipeline = pipeline.get();
    return BuiltPipeline(current_pipeline);
}

GraphicsPipeline* PipelineCache::BuiltPipeline(GraphicsPipeline* pipeline) const noexcept {
    if (pipeline->IsBuilt()) {
        return pipeline;
    }
    if (!use_asynchronous_shaders) {
        return pipeline;
    }
    if (!CanSkipDraw()) {
        // The GPU back thread waits for the driver build before this draw (a frame-time spike
        // as long as the build): log each pipeline once with what made the draw unskippable.
        if (logged_build_waits.size() < 4096 && logged_build_waits.insert(pipeline).second) {
            const auto& draw_state = maxwell3d->draw_manager.draw_state;
            const auto& hashes = pipeline->DebugKey().unique_hashes;
            LOG_INFO(Render_Vulkan,
                     "Draw waits for its pipeline build: shaders {:016x} {:016x}, {} elements "
                     "(index count {}, vertex count {}), depth {}",
                     hashes[1], hashes[5], draw_state.draw_element_count,
                     draw_state.index_buffer.count, draw_state.vertex_buffer.count,
                     maxwell3d->regs.zeta_enable != 0);
        }
        return pipeline;
    }
    // The draw is dropped until the pipeline exists. Native draws are dropped on purpose here, so
    // the rasterizer must not report them as failed native work.
    NoteSkippedDraw("builds");
    draw_deferred = true;
    return nullptr;
}

bool PipelineCache::CanSkipDraw() const noexcept {
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    // [TEST] EDEN_N2DEN_SKIP_RULE for A/B runs:
    //   stale  the rule before draw_element_count (the other count field is left over from the
    //          last draw of the other kind)
    //   count  depth-tested draws, or draws of more than six elements (upstream's rule with the
    //          current draw's count, Eglantine 226162edf5)
    //   scene  (default) depth-tested draws of more than six elements only
    enum class Rule { Stale, Count, Scene };
    static const Rule rule = [] {
        const char* const value = std::getenv("EDEN_N2DEN_SKIP_RULE");
        const std::string_view name{value ? value : ""};
        return name == "stale" ? Rule::Stale : name == "count" ? Rule::Count : Rule::Scene;
    }();
    switch (rule) {
    case Rule::Stale:
        return draw_state.index_buffer.count > 6 && draw_state.vertex_buffer.count > 6;
    case Rule::Count:
        return maxwell3d->regs.zeta_enable || draw_state.draw_element_count > 6;
    case Rule::Scene:
        break;
    }
    // A skipped draw is simply missing from its frame, which only goes unnoticed when the game
    // redraws the target every frame. With the count rule, OoT lost menu graphics and drew its
    // water wrong (draws without depth that render targets the game keeps). Depth-tested draws
    // of more than six elements are scene geometry; the rest wait for their pipeline. Only the
    // current draw's count tells its size: the other count field keeps whatever the last draw
    // of the other kind left there (Eglantine 226162edf5).
    return maxwell3d->regs.zeta_enable && draw_state.draw_element_count > 6;
}

void PipelineCache::NoteSkippedDraw(const char* reason) const {
    Common::N2denTrace::pipeline_skips.fetch_add(1, std::memory_order_relaxed);
    VideoCore::NativeCensus::NoteDrop(std::string_view{reason} == "builds"
                                          ? VideoCore::NativeCensus::Drop::PipelineBuilding
                                          : VideoCore::NativeCensus::Drop::PipelineTranslating);
    // Logged once per pipeline, to find what a skipped draw renders when something is missing.
    if (logged_skips.size() >= 4096 || !logged_skips.insert(graphics_key.Hash()).second) {
        return;
    }
    const auto& regs = maxwell3d->regs;
    LOG_INFO(Render_Vulkan,
             "Draw skipped while its pipeline {}: shaders {:016x} {:016x}, {} elements, depth {}, "
             "rt0 {}x{} format {}",
             reason, graphics_key.unique_hashes[1], graphics_key.unique_hashes[5],
             maxwell3d->draw_manager.draw_state.draw_element_count, regs.zeta_enable != 0,
             regs.rt[0].width, regs.rt[0].height, static_cast<u32>(regs.rt[0].format));
}

bool PipelineCache::BuildNewGraphicsPipeline() {
    if (CanTranslateAsynchronously()) {
        // With an earlier pipeline's shaders only the driver compile is left, and it goes to the
        // pipeline workers: no translation worker is needed.
        if (std::unique_ptr<GraphicsPipeline> pipeline{CreateGraphicsPipelineFromReusedShaders()}) {
            graphics_cache.emplace(graphics_key, std::move(pipeline));
            return true;
        }
        if (std::optional<std::future<void>> done{QueueGraphicsTranslation()}) {
            return WaitForGraphicsTranslation(*done);
        }
    }
    graphics_cache.emplace(graphics_key, CreateGraphicsPipeline());
    return true;
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline(
    ShaderPools& pools, const GraphicsPipelineCacheKey& key,
    std::span<Shader::Environment* const> envs, PipelineStatistics* statistics,
    bool build_in_parallel, bool allow_translation) try {
    auto hash = key.Hash();
    LOG_DEBUG(Render_Vulkan, "{:#016x}", hash);
    const ShaderTranslationKey translation_key{MakeShaderTranslationKey(key, device)};
    const bool debug_dump_pending{VideoCore::N2DenDebug::AnyPendingDump(key.unique_hashes)};
    std::shared_ptr<const ReusableShaders> shaders{
        debug_dump_pending ? nullptr : FindReusableShaders(translation_key, envs, hash)};
    if (!shaders && !allow_translation) {
        return nullptr;
    }
    if (!shaders) {
        const auto translate_start{std::chrono::steady_clock::now()};
        {
            const VideoCore::N2DenDebug::ActivityScope activity{"graphics shader translation", key.unique_hashes[5]};
            shaders = BuildGraphicsShaders(pools, translation_key, envs, hash);
        }
        const auto translate_ms{std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - translate_start).count()};
        if (translate_ms >= 100) {
            LOG_WARNING(Render_Vulkan, "slow graphics shader translation: {} ms, shaders {:016x} {:016x}",
                        translate_ms, key.unique_hashes[1], key.unique_hashes[5]);
        }
        if (!shaders) {
            return nullptr;
        }
        if (use_shader_reuse) {
            shader_reuse.Add(translation_key, shaders);
        }
    }
    Common::ThreadWorker* const thread_worker{build_in_parallel ? &workers : nullptr};
    return std::make_unique<GraphicsPipeline>(
        scheduler, buffer_cache, texture_cache, vulkan_pipeline_cache, &shader_notify, device,
        descriptor_pool, guest_descriptor_queue, descriptor_buffer_ring, thread_worker, statistics,
        render_pass_cache, key, shaders->modules, shaders->InfoPointers());

} catch (const Shader::Exception& exception) {
    auto hash = key.Hash();
    size_t env_index{0};
    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        Shader::Environment& env{*envs[env_index]};
        ++env_index;

        const u32 cfg_offset{static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
        // Rebuilding the CFG for the dump can throw the same error again; never let that escape
        // the handler, or a single bad shader terminates the emulator instead of skipping a draw.
        try {
            Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
            env.Dump(hash, key.unique_hashes[index]);
        } catch (const Shader::Exception&) {
        }
    }
    std::string stage_hashes;
    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (key.unique_hashes[index] != 0) {
            stage_hashes += fmt::format(" stage{}={:016x}", index, key.unique_hashes[index]);
        }
    }
    LOG_ERROR(Render_Vulkan, "{}{}", exception.what(), stage_hashes);
    return nullptr;
}

std::shared_ptr<const ReusableShaders> PipelineCache::FindReusableShaders(
    const ShaderTranslationKey& key, std::span<Shader::Environment* const> envs,
    u64 pipeline_hash) const {
    if (!use_shader_reuse) {
        return nullptr;
    }
    const ShaderReuseCache::Lookup found{shader_reuse.Find(key, envs)};
    if (found.shaders && Settings::values.dump_guest_shaders) {
        // What BuildGraphicsShaders dumps for a pipeline it translates.
        size_t env_index{0};
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] != 0) {
                envs[env_index++]->Dump(pipeline_hash, key.unique_hashes[index]);
            }
        }
    }
    return found.shaders;
}

bool PipelineCache::HasReusableShaders(const GraphicsPipelineCacheKey& key) const {
    return use_shader_reuse && shader_reuse.Contains(MakeShaderTranslationKey(key, device));
}

// Translation (Maxwell code to IR) and SPIR-V emission of a graphics pipeline's programs. Reads
// nothing of the pipeline key but `key`, so the shaders fit every pipeline with that key and
// environments that answer the same (vk_shader_reuse.h). Null for programs with known issues.
std::shared_ptr<ReusableShaders> PipelineCache::BuildGraphicsShaders(
    ShaderPools& pools, const ShaderTranslationKey& key,
    std::span<Shader::Environment* const> pipeline_envs, u64 pipeline_hash) {
    // With reuse on, translation asks the environments through recorders: their answers tell
    // which later pipelines these shaders fit.
    boost::container::static_vector<RecordingEnvironment, Maxwell::MaxShaderProgram> recorders;
    boost::container::static_vector<Shader::Environment*, Maxwell::MaxShaderProgram> envs;
    for (Shader::Environment* const env : pipeline_envs) {
        envs.push_back(use_shader_reuse ? &recorders.emplace_back(*env) : env);
    }

    size_t env_index{0};
    std::array<Shader::IR::Program, Maxwell::MaxShaderProgram> programs;
    std::array<u64, Maxwell::MaxShaderProgram> debug_dump_keys{};
    std::array<bool, Maxwell::MaxShaderProgram> debug_dump_native{};
    std::array<std::string, Maxwell::MaxShaderProgram> debug_dump_ir;
    const bool uses_vertex_a{key.unique_hashes[0] != 0};
    const bool uses_vertex_b{key.unique_hashes[1] != 0};

    // Layer passthrough generation for devices without VK_EXT_shader_viewport_index_layer
    Shader::IR::Program* layer_source_program{};

    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        const bool is_emulated_stage = layer_source_program != nullptr &&
                                       index == static_cast<u32>(Maxwell::ShaderType::Geometry);
        if (key.unique_hashes[index] == 0 && is_emulated_stage) {
            programs[index] = GenerateGeometryPassthrough(pools.inst, pools.block, host_info,
                                                          *layer_source_program,
                                                          key.passthrough_topology);
            continue;
        }
        if (key.unique_hashes[index] == 0) {
            continue;
        }
        Shader::Environment& env{*envs[env_index]};
        ++env_index;

        // Without VK_EXT_mesh_shader the driver cannot build a task/mesh pipeline; the draw is
        // skipped (see the pipeline check in the rasterizer).
        if ((env.ShaderStage() == Shader::Stage::Task ||
             env.ShaderStage() == Shader::Stage::Mesh) &&
            !device.IsExtMeshShaderSupported()) {
            return nullptr;
        }

        const u32 cfg_offset{static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader))};
        Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, index == 0);
        if (!uses_vertex_a || index != 1) {
            // Normal path
            programs[index] = TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
        } else {
            // VertexB path when VertexA is present.
            auto& program_va{programs[0]};
            auto program_vb{TranslateProgram(pools.inst, pools.block, env, cfg, host_info)};
            programs[index] = MergeDualVertexPrograms(program_va, program_vb, env);
        }

        if (programs[index].stage == Shader::Stage::Mesh || programs[index].stage == Shader::Stage::Task) {
            // What each task/mesh program was translated with (mesh registers, outputs).
            const auto& mesh = programs[index].mesh;
            LOG_DEBUG(Render_Vulkan,
                     "Mesh program {:016x} stage {}: topology={} max_vertices={} max_primitives={} "
                     "threads={} vertex_lines={} primitive_lines={} payload_words={} native={}",
                     key.unique_hashes[index], static_cast<u32>(programs[index].stage),
                     mesh.topology, mesh.max_vertices, mesh.max_primitives,
                     programs[index].workgroup_size[0], mesh.vertex_lines.size(),
                     mesh.primitive_lines.size(), mesh.payload_words, env.IsNativeAmpere());
        }

        if (device.GetDriverID() != VK_DRIVER_ID_NVIDIA_PROPRIETARY && Shader::Optimization::HasBrokenPattern(programs[index])) {
            LOG_WARNING(Render_Vulkan, "0x{:016x}: Skipping shader with known issues", key.unique_hashes[index]);
            return nullptr;
        }

        if (const auto dump_key = VideoCore::N2DenDebug::PendingDump(key.unique_hashes[index])) {
            debug_dump_keys[index] = *dump_key;
            debug_dump_native[index] = env.IsNativeAmpere();
            debug_dump_ir[index] = DebugIrSummary(programs[index]);
        }

        if (programs[index].info.untracked_global_accesses != 0) {
            LOG_WARNING(Render_Vulkan,
                        "stage {} {:#016x}: untracked global accesses {}/{} stores, storage buffers {}: {}",
                        index, key.unique_hashes[index],
                        programs[index].info.untracked_global_accesses,
                        programs[index].info.untracked_global_stores,
                        programs[index].info.storage_buffers_descriptors.size(),
                        fmt::join(programs[index].info.untracked_reports, " | "));
        }

        if (Settings::values.dump_guest_shaders) {
            env.Dump(pipeline_hash, key.unique_hashes[index]);
        }

        if (programs[index].info.requires_layer_emulation) {
            layer_source_program = &programs[index];
        }
    }
    std::array<const Shader::Info*, Maxwell::MaxShaderStage> infos{};
    auto modules{std::make_shared<GraphicsShaderModules>()};

    const Shader::IR::Program* previous_stage{};
    Shader::Backend::Bindings binding;
    for (size_t index = uses_vertex_a && uses_vertex_b ? 1 : 0; index < Maxwell::MaxShaderProgram;
         ++index) {
        const bool is_emulated_stage = layer_source_program != nullptr &&
                                       index == static_cast<u32>(Maxwell::ShaderType::Geometry);
        if (key.unique_hashes[index] == 0 && !is_emulated_stage) {
            continue;
        }
        UNIMPLEMENTED_IF(index == 0);

        Shader::IR::Program& program{programs[index]};
        const size_t stage_index{index - 1};
        infos[stage_index] = &program.info;

        const auto runtime_info{MakeRuntimeInfo(programs, key, program, previous_stage)};
        ConvertLegacyToGeneric(program, runtime_info);
        const std::vector<u32> code{EmitSPIRV(profile, runtime_info, program, binding)};
        device.SaveShader(code);
        (*modules)[stage_index] = BuildShader(device, code);
        if (debug_dump_keys[index] != 0) {
            static constexpr std::array dump_stage_names{"vertex_a", "vertex", "tess_control", "tess_eval",
                                                         "geometry", "fragment"};
            WriteDebugDump(debug_dump_keys[index], key.unique_hashes[index], debug_dump_native[index],
                           dump_stage_names[index], debug_dump_ir[index], code);
        }

        // Text log + .spv dump. Text log is gated by gpu_log_level != Off; .spv dump
        // is independent and gated only by gpu_log_shader_dumps.
        const bool should_log = GPU::Logging::IsActive();
        const bool should_dump = Settings::values.gpu_log_shader_dumps.GetValue();
        if (should_log || should_dump) {
            static constexpr std::array stage_names{"vertex", "tess_control", "tess_eval", "geometry", "fragment"};
            const std::string shader_name = fmt::format("shader_{:016x}_{}", key.unique_hashes[index], stage_names[stage_index]);
            if (should_log) {
                const std::string shader_info = fmt::format("SPIR-V size: {} bytes, hash: {:016x}",
                    code.size() * sizeof(u32), key.unique_hashes[index]);
                GPU::Logging::GPULogger::GetInstance().LogShaderCompilation(shader_name, shader_info);
            }
            if (should_dump) {
                GPU::Logging::DumpSpirvShader(key.unique_hashes[index],
                                              std::span<const u32>(code.data(), code.size()));
            }
        }

        if (device.HasDebuggingToolAttached()) {
            const std::string name{fmt::format("Shader {:016x}", key.unique_hashes[index])};
            (*modules)[stage_index].SetObjectNameEXT(name.c_str());
        }
        previous_stage = &program;
    }

    auto shaders{std::make_shared<ReusableShaders>()};
    shaders->modules = std::move(modules);
    for (size_t stage = 0; stage < infos.size(); ++stage) {
        if (infos[stage]) {
            shaders->infos[stage] = *infos[stage];
        }
    }
    for (RecordingEnvironment& recorder : recorders) {
        shaders->reads.push_back(recorder.TakeReads());
    }
    return shaders;
}

namespace {
// Graphics pipelines translated on worker threads: what they still cost the GPU thread
// (environments and snapshots, waiting within the budget) and the workers, logged every 10 s.
// Used on the GPU thread only.
class TranslationReport {
public:
    using Duration = std::chrono::steady_clock::duration;

    void Queued(Duration preparation, size_t snapshot_bytes) {
        ++queued;
        preparation_time += preparation;
        copied_bytes += snapshot_bytes;
        LogEvery10Seconds();
    }

    void NotDetached() {
        ++not_detached;
        LogEvery10Seconds();
    }

    void Waited(Duration wait, bool in_time) {
        wait_time += wait;
        if (in_time) {
            ++ready_in_time;
        }
        LogEvery10Seconds();
    }

    void Adopted(Duration build) {
        ++adopted;
        build_time += build;
        LogEvery10Seconds();
    }

private:
    void LogEvery10Seconds() {
        const auto now = std::chrono::steady_clock::now();
        if (now - window < std::chrono::seconds{10}) {
            return;
        }
        const auto ms = [](Duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
        LOG_INFO(Render_Vulkan,
                 "Graphics pipelines translated on workers in the last 10 s: {} queued ({} KiB "
                 "of state copied), {} ready within the wait budget, {} adopted ({:.1f} ms on "
                 "workers), {} built on the GPU thread (texture pool not contiguous); GPU "
                 "thread: {:.1f} ms preparing, {:.1f} ms waiting",
                 queued, copied_bytes / 1024, ready_in_time, adopted, ms(build_time),
                 not_detached, ms(preparation_time), ms(wait_time));
        // Start a new window.
        *this = TranslationReport{};
    }

    u32 queued{};
    u32 not_detached{};
    u32 ready_in_time{};
    u32 adopted{};
    size_t copied_bytes{};
    Duration preparation_time{};
    Duration wait_time{};
    Duration build_time{};
    std::chrono::steady_clock::time_point window{std::chrono::steady_clock::now()};
};

TranslationReport& GetTranslationReport() {
    static TranslationReport report;
    return report;
}
} // Anonymous namespace

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipeline() {
    GraphicsEnvironments environments;
    GetGraphicsEnvironments(environments, graphics_key.unique_hashes);

    main_pools.ReleaseContents();
    auto pipeline{
        CreateGraphicsPipeline(main_pools, graphics_key, environments.Span(), nullptr, true)};
    if (!pipeline || pipeline_cache_filename.empty()) {
        return pipeline;
    }
    SerializeGraphicsPipeline(graphics_key, std::move(environments.envs));
    QueueVulkanPipelineCacheFlush();
    return pipeline;
}

std::unique_ptr<GraphicsPipeline> PipelineCache::CreateGraphicsPipelineFromReusedShaders() {
    if (!HasReusableShaders(graphics_key)) {
        return nullptr;
    }
    GraphicsEnvironments environments;
    GetGraphicsEnvironments(environments, graphics_key.unique_hashes);
    auto pipeline{CreateGraphicsPipeline(main_pools, graphics_key, environments.Span(), nullptr,
                                         true, false)};
    if (!pipeline) {
        return nullptr;
    }
    if (!pipeline_cache_filename.empty()) {
        SerializeGraphicsPipeline(graphics_key, std::move(environments.envs));
        QueueVulkanPipelineCacheFlush();
    }
    return pipeline;
}

void PipelineCache::SerializeGraphicsPipeline(const GraphicsPipelineCacheKey& key,
                                              GraphicsEnvironmentArray envs) {
    // Task and mesh programs are translated with the mesh registers, which vulkan.bin doesn't
    // carry: they go to vulkan_mesh.bin with them (builds without it never read that file).
    bool is_mesh = false;
    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (key.unique_hashes[index] != 0 && (envs[index].ShaderStage() == Shader::Stage::Task ||
                                              envs[index].ShaderStage() == Shader::Stage::Mesh)) {
            is_mesh = true;
        }
    }
    serialization_thread.QueueWork([this, key, envs = std::move(envs), is_mesh] {
        boost::container::static_vector<const GenericEnvironment*, Maxwell::MaxShaderProgram>
            env_ptrs;
        for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
            if (key.unique_hashes[index] != 0) {
                env_ptrs.push_back(&envs[index]);
            }
        }
        SerializePipeline(key, env_ptrs,
                          is_mesh ? mesh_pipeline_cache_filename : pipeline_cache_filename,
                          CACHE_VERSION, is_mesh);
    });
}

// Asynchronous translation of native graphics pipelines (NX2_SHADER_STUTTER.md).
//
// The GPU thread used to translate a new pipeline (Maxwell code to IR to SPIR-V) before its
// draw, ~2-4 ms each. Now it builds the environments, detaches them from the 3D engine (copies
// what translation reads, see GraphicsEnvironment::Detach) and queues the translation and the
// driver compile on translation_workers. It then waits up to translation_wait_budget for the
// result; if the pipeline is not ready by then the draw is skipped, like a draw whose driver
// compile is not done yet, and a later draw adopts the pipeline.
//
// Only native (Switch 2) programs take this path: their code was lowered when the shader cache
// met them, so translation never reads guest memory for instructions. Compute pipelines stay
// synchronous: a skipped dispatch would lose its outputs.
bool PipelineCache::CanTranslateAsynchronously() const {
    if (!use_async_translation || !CanSkipDraw()) {
        return false;
    }
    if (!maxwell3d->GetNativeMacroEngine().native_turing) {
        return false;
    }
    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (graphics_key.unique_hashes[index] == 0) {
            continue;
        }
        if (!shader_infos[index] || !shader_infos[index]->lowered_native) {
            return false;
        }
    }
    return true;
}

std::optional<std::future<void>> PipelineCache::QueueGraphicsTranslation() {
    const auto begin = std::chrono::steady_clock::now();
    GraphicsEnvironments environments;
    GetGraphicsEnvironments(environments, graphics_key.unique_hashes);

    size_t snapshot_bytes{};
    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (graphics_key.unique_hashes[index] == 0) {
            continue;
        }
        if (!environments.envs[index].Detach(device_memory)) {
            GetTranslationReport().NotDetached();
            return std::nullopt;
        }
        snapshot_bytes += environments.envs[index].SnapshotSizeBytes();
    }

    std::promise<void> done;
    std::future<void> result{done.get_future()};
    translating_graphics.insert(graphics_key);
    translation_workers.QueueWork([this, key = graphics_key, envs = std::move(environments.envs),
                                   done = std::move(done)]() mutable {
        TranslateGraphicsPipeline(key, std::move(envs), std::move(done));
    });
    GetTranslationReport().Queued(std::chrono::steady_clock::now() - begin, snapshot_bytes);
    return result;
}

bool PipelineCache::WaitForGraphicsTranslation(std::future<void>& done) {
    // Waiting only helps when the job starts right away: behind a backlog it would spend the
    // whole budget without getting the pipeline.
    if (translation_wait_budget.count() == 0 ||
        translating_graphics.size() > num_translation_threads) {
        return false;
    }
    const auto begin = std::chrono::steady_clock::now();
    const bool in_time = done.wait_for(translation_wait_budget) == std::future_status::ready;
    GetTranslationReport().Waited(std::chrono::steady_clock::now() - begin, in_time);
    if (!in_time) {
        return false;
    }
    AdoptTranslatedGraphicsPipelines();
    return graphics_cache.contains(graphics_key);
}

void PipelineCache::TranslateGraphicsPipeline(const GraphicsPipelineCacheKey& key,
                                              GraphicsEnvironmentArray envs,
                                              std::promise<void> done) {
    const auto begin = std::chrono::steady_clock::now();
    ShaderPools pools;
    boost::container::static_vector<Shader::Environment*, Maxwell::MaxShaderProgram> env_ptrs;
    for (size_t index = 0; index < Maxwell::MaxShaderProgram; ++index) {
        if (key.unique_hashes[index] != 0) {
            env_ptrs.push_back(&envs[index]);
        }
    }
    std::unique_ptr<GraphicsPipeline> pipeline;
    try {
        // Not built in parallel: the driver compile runs here too, so the pipeline can be drawn
        // with as soon as the GPU thread adopts it.
        pipeline = CreateGraphicsPipeline(pools, key, MakeSpan(env_ptrs), nullptr, false);
    } catch (const std::exception& exception) {
        // Shader errors are handled inside; nothing else may end the worker thread.
        LOG_ERROR(Render_Vulkan, "Graphics pipeline translation failed: {}", exception.what());
    }
    {
        std::scoped_lock lock{translated_mutex};
        translated_graphics.push_back({
            .key = key,
            .pipeline = std::move(pipeline),
            .envs = std::move(envs),
            .build_time = std::chrono::steady_clock::now() - begin,
        });
        num_translated.store(translated_graphics.size(), std::memory_order_release);
    }
    // Signalled after the lock is released: the GPU thread may be waiting to adopt this result.
    done.set_value();
}

void PipelineCache::AdoptTranslatedGraphicsPipelines() {
    std::vector<TranslatedGraphicsPipeline> translated;
    {
        std::scoped_lock lock{translated_mutex};
        translated.swap(translated_graphics);
        num_translated.store(0, std::memory_order_relaxed);
    }
    for (TranslatedGraphicsPipeline& result : translated) {
        translating_graphics.erase(result.key);
        GetTranslationReport().Adopted(result.build_time);
        // What CreateGraphicsPipeline() does after building on the GPU thread.
        if (result.pipeline && !pipeline_cache_filename.empty()) {
            SerializeGraphicsPipeline(result.key, std::move(result.envs));
            QueueVulkanPipelineCacheFlush();
        }
        graphics_cache.emplace(result.key, std::move(result.pipeline));
    }
}

std::unique_ptr<ComputePipeline> PipelineCache::CreateComputePipeline(
    const ComputePipelineCacheKey& key, const ShaderInfo* shader,
    const ComputePipeline* previous, bool background) {
    const GPUVAddr program_base{kepler_compute->regs.code_loc.Address()};
    const auto& qmd{kepler_compute->launch_description};
    ComputeEnvironment env{*kepler_compute, *gpu_memory, program_base, qmd.program_start,
                           shader->lowered_native};
    env.SetCachedSize(shader->size_bytes);
    if (previous) {
        // Keep the buffers the previous build bound, so a slot that empties again later doesn't
        // drop them and start another rebuild.
        env.SeedStoragePointers(previous->pointer_slots);
    }

    main_pools.ReleaseContents();
    const auto build_start{std::chrono::steady_clock::now()};
    std::unique_ptr<ComputePipeline> pipeline;
    {
        const VideoCore::N2DenDebug::ActivityScope activity{"compute pipeline build", key.unique_hash};
        // Debug only: EDEN_NX2_DEBUG_COMPUTE_BUILD_DELAY_MS stretches every compute pipeline build,
        // to reproduce timing bugs that need a slow driver compile (a cold driver shader cache).
        static const int debug_delay_ms = [] {
            const char* const value = std::getenv("EDEN_NX2_DEBUG_COMPUTE_BUILD_DELAY_MS");
            return value ? std::atoi(value) : 0;
        }();
        if (debug_delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds{debug_delay_ms});
        }
        pipeline = CreateComputePipeline(main_pools, key, env, nullptr, true, background);
    }
    const auto build_ms{std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - build_start).count()};
    if (build_ms >= 100) {
        LOG_WARNING(Render_Vulkan, "slow compute pipeline build: {} ms, shader {:016x}", build_ms, key.unique_hash);
    }
    if (pipeline) {
        env.StoragePointerSlots(pipeline->pointer_slots, pipeline->null_pointer_slots);
        pipeline->pointer_rebuilds = previous ? previous->pointer_rebuilds + 1 : 0;
    }
    if (!pipeline || pipeline_cache_filename.empty()) {
        return pipeline;
    }
    serialization_thread.QueueWork([this, key, env_ = std::move(env)] {
        SerializePipeline(key, std::array<const GenericEnvironment*, 1>{&env_},
                          pipeline_cache_filename, CACHE_VERSION);
    });
    QueueVulkanPipelineCacheFlush();
    return pipeline;
}

std::unique_ptr<ComputePipeline> PipelineCache::CreateComputePipeline(
    ShaderPools& pools, const ComputePipelineCacheKey& key, Shader::Environment& env,
    PipelineStatistics* statistics, bool build_in_parallel, bool background) try {
    auto hash = key.Hash();
    if (device.HasBrokenCompute()) {
        LOG_ERROR(Render_Vulkan, "Skipping {:#016x}", hash);
        return nullptr;
    }

    LOG_DEBUG(Render_Vulkan, "{:#016x}", hash);

    Shader::Maxwell::Flow::CFG cfg{env, pools.flow_block, env.StartAddress()};

    // Dump it before error.
    if (Settings::values.dump_guest_shaders) {
        env.Dump(hash, key.unique_hash);
    }

    auto program{TranslateProgram(pools.inst, pools.block, env, cfg, host_info)};
    if (VideoCore::N2DenDebug::Enabled()) {
        auto storage_json = nlohmann::json::array();
        for (const auto& desc : program.info.storage_buffers_descriptors) {
            storage_json.push_back({{"cbuf", fmt::format("c{}[{:#x}]", desc.cbuf_index, desc.cbuf_offset)},
                               {"count", desc.count}, {"written", desc.is_written}});
        }
        VideoCore::N2DenDebug::NoteCompute(key.unique_hash, {
            {"translation_error", ""},
            {"storage_buffers", std::move(storage_json)},
            {"native_global_bounds", program.info.native_global_bounds},
            {"uses_global_memory", program.info.uses_global_memory},
            {"untracked_accesses", program.info.untracked_global_accesses},
            {"untracked_stores", program.info.untracked_global_stores},
            {"untracked_reports", program.info.untracked_reports},
            {"shared_memory", program.shared_memory_size},
        });
    }
    if (program.info.untracked_global_accesses != 0) {
        LOG_WARNING(Render_Vulkan,
                    "compute {:#016x}: untracked global accesses {}/{} stores, storage buffers {}",
                    key.unique_hash, program.info.untracked_global_accesses,
                    program.info.untracked_global_stores,
                    program.info.storage_buffers_descriptors.size());
    }
    const VkDriverIdKHR driver_id = device.GetDriverID();
    const bool needs_shared_mem_clamp =
        driver_id == VK_DRIVER_ID_QUALCOMM_PROPRIETARY ||
        driver_id == VK_DRIVER_ID_ARM_PROPRIETARY;
    const u32 max_shared_memory = device.GetMaxComputeSharedMemorySize();
    if (needs_shared_mem_clamp && program.shared_memory_size > max_shared_memory) {
        LOG_WARNING(Render_Vulkan,
                    "Compute shader {:#016x} requests {}KB shared memory but device max is {}KB - clamping",
                    key.unique_hash,
                    program.shared_memory_size / 1024,
                    max_shared_memory / 1024);
        program.shared_memory_size = max_shared_memory;
    }
    auto compute_profile = profile;
    compute_profile.prefer_subgroup_ballot_reduction =
        env.HasNativeRegisterCopies() && device.IsNvidia() &&
        device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);
    const std::vector<u32> code{EmitSPIRV(compute_profile, program)};
    device.SaveShader(code);
    vk::ShaderModule spv_module{BuildShader(device, code)};
    if (const auto dump_key = VideoCore::N2DenDebug::PendingDump(key.unique_hash)) {
        WriteDebugDump(*dump_key, key.unique_hash, env.IsNativeAmpere(), "compute", DebugIrSummary(program), code);
    }

    // Text log + .spv dump. Same split as the graphics path.
    const bool should_log = GPU::Logging::IsActive();
    const bool should_dump = Settings::values.gpu_log_shader_dumps.GetValue();
    if (should_log || should_dump) {
        const std::string shader_name = fmt::format("shader_{:016x}_compute", key.unique_hash);
        if (should_log) {
            const std::string shader_info = fmt::format("SPIR-V size: {} bytes, hash: {:016x}",
                code.size() * sizeof(u32), key.unique_hash);
            GPU::Logging::GPULogger::GetInstance().LogShaderCompilation(shader_name, shader_info);
        }
        if (should_dump) {
            GPU::Logging::DumpSpirvShader(key.unique_hash,
                                          std::span<const u32>(code.data(), code.size()));
        }
    }

    if (device.HasDebuggingToolAttached()) {
        const auto name{fmt::format("Shader {:016x}", key.unique_hash)};
        spv_module.SetObjectNameEXT(name.c_str());
    }
    Common::ThreadWorker* const thread_worker{build_in_parallel ? &workers : nullptr};
    // Without the shader notifier the build is not a blocking one: the game keeps running.
    VideoCore::ShaderNotify* const notify{background ? nullptr : &shader_notify};
    return std::make_unique<ComputePipeline>(device, scheduler, vulkan_pipeline_cache, descriptor_pool,
                                             guest_descriptor_queue, descriptor_buffer_ring,
                                             thread_worker, statistics,
                                             notify, program.info, std::move(spv_module),
                                             key.unique_hash, env.WorkgroupSize());

} catch (const Shader::Exception& exception) {
    LOG_ERROR(Render_Vulkan, "{}", exception.what());
    VideoCore::N2DenDebug::NoteCompute(key.unique_hash,
                                       {{"translation_error", std::string{exception.what()}}});
    return nullptr;
}

void PipelineCache::SerializeVulkanPipelineCache(const std::filesystem::path& filename,
                                                 const vk::PipelineCache& pipeline_cache,
                                                 u32 cache_version) try {
    // Write to a temporary file and rename it over the old cache, so a stop in the middle of the
    // write keeps the previous cache instead of leaving an empty file that fails to load.
    const std::filesystem::path temporary{filename.string() + ".tmp"};
    std::ofstream file(temporary, std::ios::binary);
    file.exceptions(std::ifstream::failbit);
    if (!file.is_open()) {
        LOG_ERROR(Common_Filesystem, "Failed to open Vulkan driver pipeline cache file {}",
                  Common::FS::PathToUTF8String(temporary));
        return;
    }
    file.write(VULKAN_CACHE_MAGIC_NUMBER.data(), VULKAN_CACHE_MAGIC_NUMBER.size())
        .write(reinterpret_cast<const char*>(&cache_version), sizeof(cache_version));

    size_t cache_size = 0;
    std::vector<char> cache_data;
    if (pipeline_cache) {
        pipeline_cache.Read(&cache_size, nullptr);
        cache_data.resize(cache_size);
        pipeline_cache.Read(&cache_size, cache_data.data());
    }
    file.write(cache_data.data(), cache_size);
    file.close();

    // RenameFile refuses to overwrite, so drop the old cache first. The temporary file is already
    // complete at this point, so a stop here loses the cache but never leaves a truncated one.
    Common::FS::RemoveFile(filename);
    if (!Common::FS::RenameFile(temporary, filename)) {
        LOG_ERROR(Common_Filesystem, "Failed to replace Vulkan driver pipeline cache file {}",
                  Common::FS::PathToUTF8String(filename));
        return;
    }

    LOG_INFO(Render_Vulkan, "Vulkan driver pipelines cached at: {}",
             Common::FS::PathToUTF8String(filename));

} catch (const std::ios_base::failure& e) {
    LOG_ERROR(Common_Filesystem, "{}", e.what());
    const std::filesystem::path temporary{filename.string() + ".tmp"};
    if (!Common::FS::RemoveFile(temporary)) {
        LOG_ERROR(Common_Filesystem, "Failed to delete Vulkan driver pipeline cache file {}",
                  Common::FS::PathToUTF8String(temporary));
    }
}

vk::PipelineCache PipelineCache::LoadVulkanPipelineCache(const std::filesystem::path& filename,
                                                         u32 expected_cache_version) {
    const auto create_pipeline_cache = [this](size_t data_size, const void* data) {
        VkPipelineCacheCreateInfo pipeline_cache_ci = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .initialDataSize = data_size,
            .pInitialData = data};
        return device.GetLogical().CreatePipelineCache(pipeline_cache_ci);
    };
    try {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return create_pipeline_cache(0, nullptr);
        }
        file.exceptions(std::ifstream::failbit);
        const auto end{file.tellg()};
        static constexpr size_t header_size =
            VULKAN_CACHE_MAGIC_NUMBER.size() + sizeof(u32);
        if (end < 0 || static_cast<size_t>(end) < header_size) {
            // A stop in the middle of a write used to leave an empty file; recreate it instead of
            // reading past its end.
            file.close();
            LOG_ERROR(Common_Filesystem,
                      "Truncated Vulkan driver pipeline cache file, recreating it");
            Common::FS::RemoveFile(filename);
            return create_pipeline_cache(0, nullptr);
        }
        file.seekg(0, std::ios::beg);

        std::array<char, 8> magic_number;
        u32 cache_version;
        file.read(magic_number.data(), magic_number.size())
            .read(reinterpret_cast<char*>(&cache_version), sizeof(cache_version));
        if (magic_number != VULKAN_CACHE_MAGIC_NUMBER || cache_version != expected_cache_version) {
            file.close();
            if (Common::FS::RemoveFile(filename)) {
                if (magic_number != VULKAN_CACHE_MAGIC_NUMBER) {
                    LOG_ERROR(Common_Filesystem, "Invalid Vulkan driver pipeline cache file");
                }
                if (cache_version != expected_cache_version) {
                    LOG_INFO(Common_Filesystem, "Deleting old Vulkan driver pipeline cache");
                }
            } else {
                LOG_ERROR(Common_Filesystem,
                          "Invalid Vulkan pipeline cache file and failed to delete it in \"{}\"",
                          Common::FS::PathToUTF8String(filename));
            }
            return create_pipeline_cache(0, nullptr);
        }

        const size_t cache_size = static_cast<size_t>(end) - header_size;
        std::vector<char> cache_data(cache_size);
        file.read(cache_data.data(), cache_size);

        LOG_INFO(Render_Vulkan,
                 "Loaded Vulkan driver pipeline cache: ", Common::FS::PathToUTF8String(filename));

        return create_pipeline_cache(cache_size, cache_data.data());

    } catch (const std::ios_base::failure& e) {
        LOG_ERROR(Common_Filesystem, "{}", e.what());
        if (!Common::FS::RemoveFile(filename)) {
            LOG_ERROR(Common_Filesystem, "Failed to delete Vulkan driver pipeline cache file {}",
                      Common::FS::PathToUTF8String(filename));
        }

        return create_pipeline_cache(0, nullptr);
    }
}

} // namespace Vulkan
