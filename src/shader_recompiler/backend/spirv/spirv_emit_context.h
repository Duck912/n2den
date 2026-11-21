// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <bit>

#include <sirit/sirit.h>
#include "common/container/unordered_map.h"
#include "common/container/unordered_set.h"

#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"
#include "shader_recompiler/shader_info.h"

namespace Shader::Backend::SPIRV {


using Sirit::Id;

class VectorTypes {
public:
    void Define(Sirit::Module& sirit_ctx, Id base_type, std::string_view name);

    [[nodiscard]] Id operator[](size_t size) const noexcept {
        return defs[size - 1];
    }

private:
    std::array<Id, 4> defs{};
};

struct TextureDefinition {
    Id id;
    Id sampled_type;
    Id pointer_type;
    Id image_type;
    u32 count;
    bool is_multisample;
    bool is_integer;
};

/// A texture pool view array (Info::texture_pool_descriptors).
struct TexturePoolDefinition {
    Id id;
    Id image_type;
    Id sampled_type;
    Id pointer_type; ///< To one view of the array
    bool is_multisample;
};

struct TextureBufferDefinition {
    Id id;
    u32 count;
};

struct ImageBufferDefinition {
    Id id;
    Id image_type;
    Id pointer_type;
    u32 count;
    bool is_integer;
};

struct ImageDefinition {
    Id id;
    Id image_type;
    Id pointer_type;
    u32 count;
    bool is_integer;
};

struct UniformDefinitions {
    Id U8{};
    Id S8{};
    Id U16{};
    Id S16{};
    Id U32{};
    Id F32{};
    Id U32x2{};
    Id U32x4{};

    constexpr static size_t NumElements(Id UniformDefinitions::*member_ptr) {
        if (member_ptr == &UniformDefinitions::U8) {
            return 1;
        }
        if (member_ptr == &UniformDefinitions::S8) {
            return 1;
        }
        if (member_ptr == &UniformDefinitions::U16) {
            return 1;
        }
        if (member_ptr == &UniformDefinitions::S16) {
            return 1;
        }
        if (member_ptr == &UniformDefinitions::U32) {
            return 1;
        }
        if (member_ptr == &UniformDefinitions::F32) {
            return 1;
        }
        if (member_ptr == &UniformDefinitions::U32x2) {
            return 2;
        }
        if (member_ptr == &UniformDefinitions::U32x4) {
            return 4;
        }
        ASSERT(false);
        return 1;
    }

    constexpr static bool IsFloat(Id UniformDefinitions::*member_ptr) {
        if (member_ptr == &UniformDefinitions::F32) {
            return true;
        }
        return false;
    }
};

struct StorageTypeDefinition {
    Id array{};
    Id element{};
};

struct StorageTypeDefinitions {
    StorageTypeDefinition U8{};
    StorageTypeDefinition S8{};
    StorageTypeDefinition U16{};
    StorageTypeDefinition S16{};
    StorageTypeDefinition U32{};
    StorageTypeDefinition U64{};
    StorageTypeDefinition F32{};
    StorageTypeDefinition U32x2{};
    StorageTypeDefinition U32x4{};
};

struct StorageDefinitions {
    Id U8{};
    Id S8{};
    Id U16{};
    Id S16{};
    Id U32{};
    Id F32{};
    Id U64{};
    Id U32x2{};
    Id U32x4{};
};

enum class InputGenericLoadOp {
    None,
    Bitcast,
    SToF,
    UToF,
};

struct InputGenericInfo {
    Id id;
    Id pointer_type;
    Id component_type;
    InputGenericLoadOp load_op;
};

struct GenericElementInfo {
    Id id{};
    u32 first_element{};
    u32 num_components{};
};

// Opcode numbers of the OpExtInst markers that stand for the cooperative matrix instructions.
// GLSL.std.450 ends at 81, so these never collide with a real extended instruction.
namespace CoopMarker {
constexpr u32 Load = 0x7e01;
constexpr u32 Store = 0x7e02;
constexpr u32 MulAdd = 0x7e03;
} // namespace CoopMarker

class EmitContext final : public Sirit::Module {
public:
    explicit EmitContext(const Profile& profile, const RuntimeInfo& runtime_info,
                         IR::Program& program, Bindings& binding);
    ~EmitContext();

    [[nodiscard]] Id Def(const IR::Value& value);

    [[nodiscard]] Id BitOffset8(const IR::Value& offset);
    [[nodiscard]] Id BitOffset16(const IR::Value& offset);

    Id Const(u32 value) {
        return CachedConstant(u32_constants, U32[1], value, value);
    }

    Id Const(u32 element_1, u32 element_2) {
        return ConstantComposite(U32[2], Const(element_1), Const(element_2));
    }

    Id Const(u32 element_1, u32 element_2, u32 element_3) {
        return ConstantComposite(U32[3], Const(element_1), Const(element_2), Const(element_3));
    }

    Id Const(u32 element_1, u32 element_2, u32 element_3, u32 element_4) {
        return ConstantComposite(U32[4], Const(element_1), Const(element_2), Const(element_3),
                                 Const(element_4));
    }

    Id SConst(s32 value) {
        return Constant(S32[1], value);
    }

    Id SConst(s32 element_1, s32 element_2) {
        return ConstantComposite(S32[2], SConst(element_1), SConst(element_2));
    }

    Id SConst(s32 element_1, s32 element_2, s32 element_3) {
        return ConstantComposite(S32[3], SConst(element_1), SConst(element_2), SConst(element_3));
    }

    Id SConst(s32 element_1, s32 element_2, s32 element_3, s32 element_4) {
        return ConstantComposite(S32[4], SConst(element_1), SConst(element_2), SConst(element_3),
                                 SConst(element_4));
    }

    Id Const(f32 value) {
        return CachedConstant(f32_constants, F32[1], std::bit_cast<u32>(value), value);
    }

    /// Sirit deduplicates constants, but by hashing a freshly allocated copy of their words on
    /// every call; immediates are emitted over and over, so remember the ids of scalar constants.
    /// Keys are bit patterns, like Sirit's (0.0f and -0.0f stay distinct).
    template <typename T>
    Id CachedConstant(::Common::unordered_map<u32, Id>& cache, Id type, u32 bits, T value) {
        const auto [it, is_new]{cache.try_emplace(bits)};
        if (is_new) {
            it->second = Constant(type, value);
        }
        return it->second;
    }

    const Profile& profile;
    const RuntimeInfo& runtime_info;
    Stage stage{};

    Id void_id{};
    Id U1{};
    Id U8{};
    Id S8{};
    Id U16{};
    Id S16{};
    Id U64{};
    VectorTypes F32;
    VectorTypes U32;
    VectorTypes S32;
    VectorTypes F16;
    VectorTypes F64;

    ::Common::unordered_map<u32, Id> u32_constants;
    ::Common::unordered_map<u32, Id> f32_constants;

    Id true_value{};
    Id false_value{};
    Id u32_zero_value{};
    Id f32_zero_value{};

    UniformDefinitions uniform_types;
    StorageTypeDefinitions storage_types;

    Id private_u32{};

    Id shared_u8{};
    Id shared_u16{};
    Id shared_u32{};
    Id shared_u64{};
    Id shared_u32x2{};
    Id shared_u32x4{};

    Id input_f32{};
    Id input_u32{};
    Id input_s32{};

    Id output_f32{};
    Id output_u32{};

    Id image_buffer_type{};
    Id image_u32{};

    std::array<UniformDefinitions, Info::MAX_CBUFS> cbufs{};
    std::array<StorageDefinitions, Info::MAX_SSBOS> ssbos{};
    std::vector<TextureBufferDefinition> texture_buffers;
    std::vector<ImageBufferDefinition> image_buffers;
    std::vector<TextureDefinition> textures;
    std::vector<ImageDefinition> images;
    std::vector<TexturePoolDefinition> texture_pools;
    Id sampler_pool{};         ///< The sampler pool array, defined with the texture pools
    Id sampler_pool_type{};    ///< OpTypeSampler
    Id sampler_pool_pointer{}; ///< Pointer type to one sampler of the pool
    /// void(u32 bit): marks a pool element as looked up in the feedback buffer (shader_info.h)
    Id texture_pool_feedback_func{};

    Id workgroup_id{};
    Id local_invocation_id{};
    Id invocation_id{};
    Id patch_vertices_in{};
    Id sample_id{};
    Id is_helper_invocation{};
    Id subgroup_local_invocation_id{};
    Id subgroup_mask_eq{};
    Id subgroup_mask_lt{};
    Id subgroup_mask_le{};
    Id subgroup_mask_gt{};
    Id subgroup_mask_ge{};
    Id instance_id{};
    Id instance_index{};
    Id base_instance{};
    Id vertex_id{};
    Id vertex_index{};
    Id draw_index{};
    Id base_vertex{};
    Id front_face{};
    Id point_coord{};
    Id tess_coord{};
    Id clip_distances{};
    Id layer{};
    Id viewport_index{};
    Id viewport_mask{};
    Id primitive_id{};

    Id fswzadd_lut_a{};
    Id fswzadd_lut_b{};

    Id indexed_load_func{};
    Id indexed_store_func{};

    Id rescaling_uniform_constant{};
    Id rescaling_push_constants{};
    Id rescaling_textures_type{};
    Id rescaling_images_type{};
    u32 rescaling_textures_member_index{};
    u32 rescaling_images_member_index{};
    u32 rescaling_downfactor_member_index{};
    u32 texture_rescaling_index{};
    u32 image_rescaling_index{};

    Id render_area_push_constant{};
    u32 render_are_member_index{};

    Id local_memory{};

    bool uses_explicit_workgroup_layout{};
    Id shared_memory_u8{};
    Id shared_memory_u16{};
    Id shared_memory_u32{};
    Id shared_memory_u64{};
    Id shared_memory_u32x2{};
    Id shared_memory_u32x4{};

    Id shared_memory_u32_type{};

    Id shared_store_u8_func{};
    Id shared_store_u16_func{};
    Id increment_cas_shared{};
    Id increment_cas_ssbo{};
    Id decrement_cas_shared{};
    Id decrement_cas_ssbo{};
    Id f32_add_cas{};
    Id f16x2_add_cas{};
    Id f16x2_min_cas{};
    Id f16x2_max_cas{};
    Id f32x2_add_cas{};
    Id f32x2_min_cas{};
    Id f32x2_max_cas{};

    Id write_storage_cas_loop_func{};

    Id load_global_func_u32{};
    Id load_global_func_u32x2{};
    Id load_global_func_u32x4{};
    Id write_global_func_u32{};
    Id write_global_func_u32x2{};
    Id write_global_func_u32x4{};
    std::array<Id, 11> global_atomic_funcs{}; // IAdd, SMin, UMin, SMax, UMax, And, Or, Xor, Exchange, Inc, Dec
    std::array<Id, 9> global_atomic_funcs_u64{}; // 64-bit forms of the first nine
    Id global_atomic_add_f32_func{};              // F32 add through a raw address (CAS loop)
    Id global_atomic_cas_func_u32{};              // 32-bit compare-exchange through a raw address
    Id write_global_subword_func{};
    /// nvn-hle raw global memory (from Eglantine): the load/write_global_func_* and
    /// write_global_subword_func above go through PhysicalStorageBuffer64 pointers and the
    /// atomics are the functions below (see DefineRawGlobalMemoryFunctions); global_atomic_funcs*
    /// stay undefined. `raw_global_atomic_funcs[opcode - GlobalAtomicIAdd32]` is the function
    /// `result(guest address, value)` (`result(guest address, comparator, value)` for
    /// GlobalAtomicCompareExchange32) defined for each GlobalAtomic* opcode the program uses
    /// (Info::global_atomics); invalid for the ones with no definition. Never set for the
    /// emulated GPU.
    static constexpr size_t NUM_GLOBAL_ATOMICS{static_cast<size_t>(IR::Opcode::GlobalAtomicMaxF32x2) -
                                               static_cast<size_t>(IR::Opcode::GlobalAtomicIAdd32) + 1};
    std::array<Id, NUM_GLOBAL_ATOMICS> raw_global_atomic_funcs{};
    bool raw_global_memory{};

    bool need_input_position_indirect{};
    Id input_position{};
    std::array<InputGenericInfo, 32> input_generics{};

    Id output_point_size{};
    Id output_position{};
    std::array<std::array<GenericElementInfo, 4>, 32> output_generics{};

    Id output_tess_level_outer{};
    Id output_tess_level_inner{};
    std::array<Id, 30> patches{};

    std::array<Id, 8> frag_color{};
    Id sample_mask{};
    Id frag_depth{};

    std::vector<Id> interfaces;
    std::vector<std::pair<u32, spv::Op>> opcode_rewrites;

    // HMMA.16816.F16 as a cooperative matrix multiply-add (DefineCooperativeMatrix). Sirit has no
    // cooperative matrix instructions: the types are placeholder structs and the instructions
    // OpExtInst markers on the GLSL set, replaced after assembly (PatchCooperativeMatrix).
    bool uses_cooperative_matrix{};
    Id hmma_func{}; // One DontInline function per module, called by every HMMA
    Id coopmat_glsl{};
    Id coopmat_scratch{};
    Id coopmat_word_pointer{};
    bool coopmat_scratch_in_block{};
    Id coopmat_subgroup_id{};
    Id coopmat_f16{};
    std::array<Id, 3> coopmat_types{};      // A 16x16, B 16x8, accumulator 16x8
    std::array<Id, 3> coopmat_columns{};    // column count constant of each
    std::array<Id, 3> coopmat_uses{};       // MatrixAKHR, MatrixBKHR, MatrixAccumulatorKHR
    Id coopmat_scope{};
    Id coopmat_rows{};

    // Task and mesh programs. Their stage buffer: the task's output region is the task payload
    // (the mesh stage's input region); the mesh output regions live in workgroup memory until
    // the epilogue copies them to the outputs. The two mesh instructions Sirit lacks are emitted
    // as OpStore markers with these operands and replaced after assembly (PatchMeshInstructions).
    struct MeshLineTarget {
        Id variable;    ///< arrayed output
        u32 element;    ///< vector element, or ~0 for a scalar output
        bool is_float;  ///< bitcast the stage buffer word to F32
    };
    Id mesh_payload{};
    Id mesh_payload_u32{};  ///< pointer type to one payload word
    Id mesh_attributes{};
    Id mesh_map{};
    Id mesh_primitive_count{};
    Id mesh_shared_u32{};   ///< pointer type to one workgroup word
    Id mesh_primitive_indices{};
    Id mesh_point_size{}; ///< PointSize output array of a mesh program, if any
    MeshInfo mesh_info{};
    u32 mesh_threads{};
    u32 mesh_attribute_words{};
    u32 mesh_map_bytes{};
    u32 mesh_primitive_base{}; ///< first word of the per-primitive groups
    std::vector<MeshLineTarget> mesh_vertex_targets;
    std::vector<MeshLineTarget> mesh_primitive_targets;
    std::array<Id, 2> mesh_outputs_marker{};
    std::array<Id, 4> mesh_tasks_marker{};

    Id load_const_func_u8{};
    Id load_const_func_u16{};
    Id load_const_func_u32{};
    Id load_const_func_f32{};
    Id load_const_func_u32x2{};
    Id load_const_func_u32x4{};

    // Sirit::Id doesn't play nice with *::set<>
    ::Common::unordered_set<u32> non_uniform_ids;

    bool uses_nonuniform_sampled_image{};
    bool uses_nonuniform_storage_image{};
    bool uses_nonuniform_uniform_texel_buffer{};
    bool uses_nonuniform_storage_texel_buffer{};

private:
    void DefineCommonTypes(const Info& info);
    void DefineCommonConstants();
    void DefineInterfaces(const IR::Program& program);
    void DefineLocalMemory(const IR::Program& program);
    void DefineSharedMemory(const IR::Program& program);
    void DefineCooperativeMatrix(const IR::Program& program);
    void DefineSharedMemoryFunctions(const IR::Program& program);
    void DefineConstantBuffers(const Info& info, u32& binding);
    void DefineConstantBufferIndirectFunctions(const Info& info);
    void DefineStorageBuffers(const Info& info, u32& binding);
    void DefineTextureBuffers(const Info& info, u32& binding);
    void DefineImageBuffers(const Info& info, u32& binding);
    void DefineTextures(const Info& info, u32& binding, u32& scaling_index);
    void DefineTexturePools(const Info& info);
    void DefineImages(const Info& info, u32& binding, u32& scaling_index);
    void DefineAttributeMemAccess(const Info& info);
    void DefineWriteStorageCasLoopFunction(const Info& info);
    void DefineGlobalMemoryFunctions(const Info& info);
    void DefineRawGlobalMemoryFunctions(const Info& info);
    void DefineRescalingInput(const Info& info);
    void DefineRescalingInputPushConstant();
    void DefineRescalingInputUniformConstant();
    void DefineRenderArea(const Info& info);

    void DefineInputs(const IR::Program& program);
    void DefineOutputs(const IR::Program& program);
    void DefineMesh(const IR::Program& program);
};

} // namespace Shader::Backend::SPIRV
