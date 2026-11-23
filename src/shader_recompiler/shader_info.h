// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <bitset>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/frontend/ir/opcodes.h"
#include "shader_recompiler/frontend/ir/type.h"
#include "shader_recompiler/varying_state.h"

#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>

namespace Shader {

enum class ReplaceConstant : u32 {
    BaseInstance,
    BaseVertex,
    DrawID,
};

enum class TextureType : u32 {
    Color1D,
    ColorArray1D,
    Color2D,
    ColorArray2D,
    Color3D,
    ColorCube,
    ColorArrayCube,
    Buffer,
    Color2DRect,
};
constexpr u32 NUM_TEXTURE_TYPES = 9;

enum class TexturePixelFormat {
    A8B8G8R8_UNORM,
    A8B8G8R8_SNORM,
    A8B8G8R8_SINT,
    A8B8G8R8_UINT,
    R5G6B5_UNORM,
    B5G6R5_UNORM,
    A1R5G5B5_UNORM,
    A2B10G10R10_UNORM,
    A2B10G10R10_UINT,
    A2R10G10B10_UNORM,
    A1B5G5R5_UNORM,
    A5B5G5R1_UNORM,
    R8_UNORM,
    R8_SNORM,
    R8_SINT,
    R8_UINT,
    R16G16B16A16_FLOAT,
    R16G16B16A16_UNORM,
    R16G16B16A16_SNORM,
    R16G16B16A16_SINT,
    R16G16B16A16_UINT,
    B10G11R11_FLOAT,
    R32G32B32A32_UINT,
    BC1_RGBA_UNORM,
    BC2_UNORM,
    BC3_UNORM,
    BC4_UNORM,
    BC4_SNORM,
    BC5_UNORM,
    BC5_SNORM,
    BC7_UNORM,
    BC6H_UFLOAT,
    BC6H_SFLOAT,
    ASTC_2D_4X4_UNORM,
    B8G8R8A8_UNORM,
    R32G32B32A32_FLOAT,
    R32G32B32A32_SINT,
    R32G32_FLOAT,
    R32G32_SINT,
    R32_FLOAT,
    R16_FLOAT,
    R16_UNORM,
    R16_SNORM,
    R16_UINT,
    R16_SINT,
    R16G16_UNORM,
    R16G16_FLOAT,
    R16G16_UINT,
    R16G16_SINT,
    R16G16_SNORM,
    R32G32B32_FLOAT,
    A8B8G8R8_SRGB,
    R8G8_UNORM,
    R8G8_SNORM,
    R8G8_SINT,
    R8G8_UINT,
    R32G32_UINT,
    R16G16B16X16_FLOAT,
    R32_UINT,
    R32_SINT,
    ASTC_2D_8X8_UNORM,
    ASTC_2D_8X5_UNORM,
    ASTC_2D_5X4_UNORM,
    B8G8R8A8_SRGB,
    BC1_RGBA_SRGB,
    BC2_SRGB,
    BC3_SRGB,
    BC7_SRGB,
    A4B4G4R4_UNORM,
    G4R4_UNORM,
    ASTC_2D_4X4_SRGB,
    ASTC_2D_8X8_SRGB,
    ASTC_2D_8X5_SRGB,
    ASTC_2D_5X4_SRGB,
    ASTC_2D_5X5_UNORM,
    ASTC_2D_5X5_SRGB,
    ASTC_2D_10X8_UNORM,
    ASTC_2D_10X8_SRGB,
    ASTC_2D_6X6_UNORM,
    ASTC_2D_6X6_SRGB,
    ASTC_2D_10X6_UNORM,
    ASTC_2D_10X6_SRGB,
    ASTC_2D_10X5_UNORM,
    ASTC_2D_10X5_SRGB,
    ASTC_2D_10X10_UNORM,
    ASTC_2D_10X10_SRGB,
    ASTC_2D_12X10_UNORM,
    ASTC_2D_12X10_SRGB,
    ASTC_2D_12X12_UNORM,
    ASTC_2D_12X12_SRGB,
    ASTC_2D_8X6_UNORM,
    ASTC_2D_8X6_SRGB,
    ASTC_2D_6X5_UNORM,
    ASTC_2D_6X5_SRGB,
    E5B9G9R9_FLOAT,
    D32_FLOAT,
    D16_UNORM,
    X8_D24_UNORM,
    S8_UINT,
    D24_UNORM_S8_UINT,
    S8_UINT_D24_UNORM,
    D32_FLOAT_S8_UINT,
};

enum class ImageFormat : u32 {
    Typeless,
    R8_UINT,
    R8_SINT,
    R16_UINT,
    R16_SINT,
    R32_UINT,
    R32G32_UINT,
    R32G32B32A32_UINT,
};

enum class Interpolation {
    Smooth,
    Flat,
    NoPerspective,
};

struct ConstantBufferDescriptor {
    u32 index;
    u32 count;

    auto operator<=>(const ConstantBufferDescriptor&) const = default;
};

struct StorageBufferDescriptor {
    u32 cbuf_index;
    u32 cbuf_offset;
    u32 count;
    bool is_written;
    /// When set, the buffer pointer is read from memory at this offset from the constant buffer
    /// pointer, which is bound as its own descriptor.
    u32 indirect_offset{~0U};
    /// nvn-hle: a global atomic reaches the buffer. Only the nvn-hle storage pass sets it; it is
    /// false in every descriptor of the emulated GPU, so comparisons there are unchanged.
    bool is_atomic_target{};
    /// Emulated GPU, vertex programs: one of the ranges of a pointer table span. The program
    /// stores through pointers it loads from a table at c[cbuf_index][cbuf_offset] indexed by the
    /// vertex (entry i at table + span_entry_offset + i * span_stride). The host reads the draw's
    /// entries and binds up to MAX_TABLE_SPANS coalesced ranges of their targets; this is range
    /// number span_slot. Its base and size reach the program as push constants
    /// (Backend::SPIRV::TableSpanLayout). ~0U for every other descriptor.
    u32 span_slot{~0U};
    u32 span_stride{};
    u32 span_entry_offset{};

    [[nodiscard]] bool IsTableSpan() const noexcept {
        return span_slot != ~0U;
    }

    auto operator<=>(const StorageBufferDescriptor&) const = default;
};

/// `cbuf_index` of the storage buffer descriptor that carries the raw global memory address table
/// (HostTranslateInfo::nvn_raw_global_table_ssbo): not a guest storage buffer, bound by the host.
inline constexpr u32 RAW_GLOBAL_TABLE_CBUF{0xffffffffU};

/// Ranges a table span binds at most (StorageBufferDescriptor::span_slot).
inline constexpr u32 MAX_TABLE_SPANS{8};
/// Bytes a table span range reaches past the last pointer it holds.
inline constexpr u32 TABLE_SPAN_EXTENT{256};

[[nodiscard]] constexpr bool IsRawGlobalTable(const StorageBufferDescriptor& desc) noexcept {
    return desc.cbuf_index == RAW_GLOBAL_TABLE_CBUF;
}

/// The bit of `Info::global_atomics` of a GlobalAtomic* opcode: its position after
/// GlobalAtomicIAdd32 in the opcode list.
[[nodiscard]] constexpr u64 GlobalAtomicBit(IR::Opcode opcode) noexcept {
    return u64{1} << (static_cast<u32>(opcode) - static_cast<u32>(IR::Opcode::GlobalAtomicIAdd32));
}
static_assert(static_cast<u32>(IR::Opcode::GlobalAtomicMaxF32x2) -
                  static_cast<u32>(IR::Opcode::GlobalAtomicIAdd32) < 64,
              "Info::global_atomics has one bit per GlobalAtomic opcode");

struct TextureBufferDescriptor {
    bool has_secondary;
    u32 cbuf_index;
    u32 cbuf_offset;
    u32 shift_left;
    u32 secondary_cbuf_index;
    u32 secondary_cbuf_offset;
    u32 secondary_shift_left;
    u32 count;
    u32 size_shift;

    /// Index in Info::handle_programs when the handle is computed at draw time.
    u32 handle_program{~0U};

    auto operator<=>(const TextureBufferDescriptor&) const = default;
};
using TextureBufferDescriptors = boost::container::small_vector<TextureBufferDescriptor, 6>;

struct ImageBufferDescriptor {
    ImageFormat format;
    bool is_written;
    bool is_read;
    bool is_integer;
    u32 cbuf_index;
    u32 cbuf_offset;
    u32 count;
    u32 size_shift;

    /// Index in Info::handle_programs when the handle is computed at draw time.
    u32 handle_program{~0U};

    auto operator<=>(const ImageBufferDescriptor&) const = default;
};
using ImageBufferDescriptors = boost::container::small_vector<ImageBufferDescriptor, 2>;

struct BufferTextureExclusion {
    u32 offset{};
    u32 mask{};
    u32 value{};
    bool not_equal{};

    bool Matches(u32 word) const { return mask && (((word & mask) != value) == not_equal); }
    auto operator<=>(const BufferTextureExclusion&) const = default;
};

struct BufferTextureSource {
    u32 offset{};
    u32 stride{};
    u32 size{};
    bool enabled{};
    bool width_query{};
    std::array<BufferTextureExclusion, 4> exclusions{};

    auto operator<=>(const BufferTextureSource&) const = default;
};

struct TextureDescriptor {
    TextureType type;
    bool is_depth;
    bool is_multisample;
    bool is_integer;
    bool has_secondary;
    u32 cbuf_index;
    u32 cbuf_offset;
    u32 shift_left;
    u32 secondary_cbuf_index;
    u32 secondary_cbuf_offset;
    u32 secondary_shift_left;
    u32 count;
    u32 size_shift;

    BufferTextureSource buffer_source{};

    /// Index in Info::handle_programs when the handle is computed at draw time.
    u32 handle_program{~0U};

    auto operator<=>(const TextureDescriptor&) const = default;
};
using TextureDescriptors = boost::container::small_vector<TextureDescriptor, 12>;

/// Texture pool lookup (Switch 2 bindless): a texture whose handle is only known when the shader
/// runs is looked up as the hardware does, the texture pool element handle & 0xfffff sampled with
/// the sampler pool element handle >> 20. The pools are run-time sized descriptor arrays in a
/// descriptor set of their own, laid out the same for every pipeline: the sampler pool and one
/// view array per image type. The host keeps that set current from the guest's texture and
/// sampler tables; nothing is resolved per draw.
constexpr u32 TEXTURE_POOL_DESCRIPTOR_SET = 1;
constexpr u32 TEXTURE_POOL_SAMPLER_BINDING = 0;
constexpr u32 TEXTURE_POOL_INDEX_BITS = 20;
constexpr u32 TEXTURE_POOL_INDEX_MASK = (1U << TEXTURE_POOL_INDEX_BITS) - 1;
constexpr u32 NUM_TEXTURE_POOL_BINDINGS = 1 + 2 * NUM_TEXTURE_TYPES;
/// Elements of each texture pool view array. The last element is always the null view: shaders
/// clamp indices to it, so a handle past the guest's table never reads outside the array.
constexpr u32 TEXTURE_POOL_VIEW_CAPACITY = 1U << 16;

/// Texture pool feedback: a storage buffer in the texture pool set holding one bit per pool
/// element a shader has looked up (texture pool elements first, then the sampler pool at
/// TEXTURE_POOL_FEEDBACK_SAMPLER_BASE). The host resolves only the elements that are marked,
/// so a table of tens of thousands of entries costs what the shaders actually sample.
constexpr u32 TEXTURE_POOL_FEEDBACK_BINDING = NUM_TEXTURE_POOL_BINDINGS;
constexpr u32 TEXTURE_POOL_FEEDBACK_SAMPLER_BASE = 1U << TEXTURE_POOL_INDEX_BITS;
constexpr u32 TEXTURE_POOL_FEEDBACK_BITS = TEXTURE_POOL_FEEDBACK_SAMPLER_BASE + 4096;
constexpr u32 TEXTURE_POOL_FEEDBACK_WORDS = TEXTURE_POOL_FEEDBACK_BITS / 32;

/// Binding of the texture pool view array of an image type. Depth-compare and plain lookups of a
/// type share the binding: the views are the same, only the image type the shader declares
/// differs. Multisampled views have their own arrays.
constexpr u32 TexturePoolBinding(TextureType type, bool is_multisample) {
    const u32 slot{static_cast<u32>(type == TextureType::Color2DRect ? TextureType::Color2D : type)};
    return 1 + (is_multisample ? NUM_TEXTURE_TYPES : 0) + slot;
}

/// One view array over the whole texture pool, as a shader declares it.
struct TexturePoolDescriptor {
    TextureType type;
    bool is_depth;
    bool is_multisample;

    auto operator<=>(const TexturePoolDescriptor&) const = default;
};
using TexturePoolDescriptors = boost::container::small_vector<TexturePoolDescriptor, 4>;

struct ImageDescriptor {
    TextureType type;
    ImageFormat format;
    bool is_written;
    bool is_read;
    bool is_integer;
    u32 cbuf_index;
    u32 cbuf_offset;
    u32 count;
    u32 size_shift;

    /// Index in Info::handle_programs when the handle is computed at draw time.
    u32 handle_program{~0U};

    auto operator<=>(const ImageDescriptor&) const = default;
};
using ImageDescriptors = boost::container::small_vector<ImageDescriptor, 4>;

/// Switch 2 (NVN) task and mesh programs: the sizes their mesh registers set and, for mesh
/// programs, the attribute (IR::Attribute index) held by each 128-byte line of a 32-element group
/// in the emulated stage buffer: per-vertex lines from the program's output map, per-primitive
/// lines from the pass-through geometry header bound with it, in attribute order.
struct MeshInfo {
    u32 payload_words{}; ///< Task output region / mesh input region, in 32-bit words
    u32 max_vertices{};
    u32 max_primitives{};
    u32 topology{}; ///< SET_MESH_SHADER_A output topology (0 = points, 1 = lines, 4 = triangles)
    boost::container::static_vector<u32, 136> vertex_lines;
    boost::container::static_vector<u32, 136> primitive_lines;

    /// Vertex indices per primitive, which is also its byte count in the map region.
    [[nodiscard]] u32 PrimitiveVertices() const noexcept {
        return topology == 0 ? 1 : topology == 1 ? 2 : 3;
    }
};

struct Info {
    static constexpr size_t MAX_INDIRECT_CBUFS{14};
    static constexpr size_t MAX_CBUFS{32};
    /// Compute programs can bind up to MAX_SSBOS storage buffers (pointer table entries beyond the
    /// program's own pointers); graphics programs, and computes with EDEN_NX2_WIDE_COMPUTE_SSBOS=0,
    /// keep MAX_GRAPHICS_SSBOS.
    static constexpr size_t MAX_SSBOS{64};
    static constexpr size_t MAX_GRAPHICS_SSBOS{32};

    bool is_task{}; ///< Switch 2 task program (Stage::Task)
    bool is_mesh{}; ///< Switch 2 mesh program (Stage::Mesh)

    bool uses_workgroup_id{};
    bool uses_local_invocation_id{};
    bool uses_invocation_id{};
    bool uses_invocation_info{};
    bool uses_sample_id{};
    bool uses_is_helper_invocation{};
    bool uses_subgroup_invocation_id{};
    bool uses_subgroup_shuffles{};
    bool uses_half_matrix_multiply{};
    bool uses_quad_shuffles{};
    std::array<bool, 30> uses_patches{};

    std::array<Interpolation, 32> interpolation{};
    VaryingState loads;
    VaryingState stores;
    VaryingState passthrough;

    std::map<IR::Attribute, IR::Attribute> legacy_stores_mapping;

    bool loads_indexed_attributes{};

    std::array<bool, 8> stores_frag_color{};
    bool stores_sample_mask{};
    bool stores_frag_depth{};

    bool stores_tess_level_outer{};
    bool stores_tess_level_inner{};

    bool stores_indexed_attributes{};

    bool stores_global_memory{};
    bool uses_local_memory{};

    bool uses_fp16{};
    bool uses_fp64{};
    bool uses_fp16_denorms_flush{};
    bool uses_fp16_denorms_preserve{};
    bool uses_fp32_denorms_flush{};
    bool uses_fp32_denorms_preserve{};
    bool uses_int8{};
    bool uses_int16{};
    bool uses_int64{};
    bool uses_image_1d{};
    bool uses_sampled_1d{};
    bool uses_sparse_residency{};
    bool uses_demote_to_helper_invocation{};
    bool uses_subgroup_vote{};
    bool uses_subgroup_mask{};
    bool uses_fswzadd{};
    bool uses_derivatives{};
    bool uses_typeless_image_reads{};
    bool uses_typeless_image_writes{};
    bool uses_image_buffers{};
    bool uses_shared_increment{};
    bool uses_shared_decrement{};
    bool uses_global_increment{};
    bool uses_global_decrement{};
    bool uses_atomic_f32_add{};
    bool uses_atomic_f16x2_add{};
    bool uses_atomic_f16x2_min{};
    bool uses_atomic_f16x2_max{};
    bool uses_atomic_f32x2_add{};
    bool uses_atomic_f32x2_min{};
    bool uses_atomic_f32x2_max{};
    bool uses_atomic_s32_min{};
    bool uses_atomic_s32_max{};
    bool uses_int64_bit_atomics{};
    bool uses_global_memory{};
    /// Native global accesses whose pointer is chosen at run time: every storage buffer descriptor
    /// is a candidate and its bound length (not an NVN size word) limits the range.
    bool native_global_bounds{};
    /// [DEBUG-untracked] Global accesses that could not be tied to a storage buffer; they stay
    /// global and the emitted lookup returns zero when no bound buffer covers the address.
    u32 untracked_global_accesses{};
    u32 untracked_global_stores{};
    /// [DEBUG-oot-washout] "<opcode> <address IR tree>" per untracked access (first 32).
    std::vector<std::string> untracked_reports;
    bool uses_global_atomics{};
    bool uses_global_subword_write{};
    bool uses_atomic_image_u32{};
    bool uses_shadow_lod{};
    bool uses_rescaling_uniform{};
    bool uses_cbuf_indirect{};
    bool uses_render_area{};

    IR::Type used_constant_buffer_types{};
    IR::Type used_storage_buffer_types{};
    IR::Type used_indirect_cbuf_types{};

    u32 constant_buffer_mask{};
    std::array<u32, MAX_CBUFS> constant_buffer_used_sizes{};
    /// First byte of the driver bank (c0) buffer entry table and its entry count (16 for the
    /// fixed per-stage tables). `nvn_buffer_used` has one bit per entry. `nvn_buffer_window` is
    /// set for nvn-hle's NVN2 compute programs, whose per-shader table is described by
    /// Environment::NvnComputeBufferWindow().
    static constexpr u32 LEGACY_NVN_BUFFERS{16};
    static constexpr size_t MAX_NVN_BUFFERS{64};
    u32 nvn_buffer_base{};
    u32 nvn_buffer_count{LEGACY_NVN_BUFFERS};
    bool nvn_buffer_window{};
    std::bitset<MAX_NVN_BUFFERS> nvn_buffer_used{};

    bool requires_layer_emulation{};
    IR::Attribute emulated_layer{};

    u32 used_clip_distances{};

    boost::container::static_vector<ConstantBufferDescriptor, MAX_CBUFS>
        constant_buffer_descriptors;
    boost::container::static_vector<StorageBufferDescriptor, MAX_SSBOS> storage_buffers_descriptors;
    TextureBufferDescriptors texture_buffer_descriptors;
    ImageBufferDescriptors image_buffer_descriptors;
    TextureDescriptors texture_descriptors;
    ImageDescriptors image_descriptors;
    /// Texture pool view arrays the program indexes at run time; when not empty the program also
    /// reads the sampler pool and the pipeline needs the texture pool descriptor set.
    TexturePoolDescriptors texture_pool_descriptors;

    /// Handle programs: texture handles computed at draw time from constant and storage words.
    std::vector<std::vector<u32>> handle_programs;

    /// nvn-hle raw global memory (Profile::support_raw_global_memory), from Eglantine.
    /// Every field stays zero for the emulated GPU: only the nvn-hle storage pass
    /// (NvnGlobalMemoryToStorageBufferPass) and what reads `nvn_raw_global_memory` set them.
    /// The program was translated with HostTranslateInfo::nvn_raw_global_memory.
    bool nvn_raw_global_memory{};
    /// Untracked (raw) global memory instructions survive the storage buffer pass.
    /// `stores_raw_global_memory`: at least one of them writes. Only collected when
    /// `nvn_raw_global_memory` is set.
    bool uses_raw_global_memory{};
    bool stores_raw_global_memory{};
    /// The GlobalAtomic* opcodes (GlobalAtomicBit) in the program: with raw global memory the
    /// backend defines one address-to-memory atomic function for each. Only collected when
    /// `nvn_raw_global_memory` is set.
    u64 global_atomics{};
    /// The raw global memory address table is the storage buffer `raw_global_table_index` (the
    /// last descriptor), see HostTranslateInfo::nvn_raw_global_table_ssbo. Otherwise it is read
    /// from the driver bank (c0).
    bool raw_global_table_ssbo{};
    u32 raw_global_table_index{};
    /// Raw stores also write through to mirror targets in the driver bank table
    /// (HostTranslateInfo::raw_global_store_alias).
    bool raw_global_store_alias{};
    /// Misses of the raw global memory table (either place) are recorded in the coverage ring
    /// (HostTranslateInfo::raw_coverage), tagged with `raw_coverage_program` (the program hash the
    /// pipeline cache sets before emission).
    bool raw_coverage{};
    u64 raw_coverage_program{};
    /// The raw global memory translation reuses the invocation's last table entry
    /// (HostTranslateInfo::raw_global_last_hit).
    bool raw_global_last_hit{};
    /// Both translations per access, differences counted (HostTranslateInfo::
    /// raw_global_last_hit_verify; needs raw_global_last_hit and raw_coverage).
    bool raw_global_last_hit_verify{};

    /// True when the shader only runs correctly if a subgroup spans a whole guest warp.
    [[nodiscard]] bool UsesWarpOps() const noexcept {
        return uses_subgroup_vote || uses_subgroup_invocation_id || uses_subgroup_shuffles ||
               uses_subgroup_mask || uses_quad_shuffles || uses_fswzadd;
    }
};

template <typename Descriptors>
u32 NumDescriptors(const Descriptors& descriptors) {
    u32 num{};
    for (const auto& desc : descriptors) {
        num += desc.count;
    }
    return num;
}

/// Operations of a handle program, a stack machine over 32-bit words. Operands follow the
/// operation code in the program; values are popped right operand first.
enum class HandleOp : u32 {
    Immediate,     ///< value
    Cbuf,          ///< index, offset: the constant buffer word
    CbufDynamic,   ///< index: the word at the popped byte offset
    Storage,       ///< binding: the storage buffer word at the popped byte offset
    Add, Sub, Mul, And, Or, Xor, Shl, Shr, Sar, Neg, Not,
    UMin, UMax, SMin, SMax,
    SLt, SLe, SGt, SGe, ULt, ULe, UGt, UGe, Eq, Ne,
    LAnd, LOr, LXor, LNot,
    Select, ///< condition, true value, false value
    Tee,    ///< slot: keeps the top value in the slot (shared subexpressions)
    Load,   ///< slot: the value kept there
};

/// Runs a handle program; read_cbuf(index, offset) and read_storage(binding, offset) return the
/// words it reads. Returns nothing for a malformed program.
template <typename ReadCbuf, typename ReadStorage>
std::optional<u32> EvaluateHandleProgram(std::span<const u32> code, ReadCbuf&& read_cbuf,
                                         ReadStorage&& read_storage) {
    std::array<u32, 64> stack{};
    std::array<u32, 256> slots{};
    size_t size = 0;
    const auto pop = [&]() -> u32 { return size > 0 ? stack[--size] : 0; };
    const auto push = [&](u32 value) {
        if (size < stack.size()) stack[size++] = value;
    };
    for (size_t pc = 0; pc < code.size();) {
        const HandleOp op = static_cast<HandleOp>(code[pc++]);
        const auto operand = [&]() -> u32 { return pc < code.size() ? code[pc++] : 0; };
        if (op == HandleOp::Immediate) {
            push(operand());
            continue;
        }
        if (op == HandleOp::Cbuf) {
            const u32 index = operand();
            push(read_cbuf(index, operand()));
            continue;
        }
        if (op == HandleOp::CbufDynamic) {
            const u32 index = operand();
            push(read_cbuf(index, pop()));
            continue;
        }
        if (op == HandleOp::Storage) {
            const u32 binding = operand();
            push(read_storage(binding, pop()));
            continue;
        }
        if (op == HandleOp::Neg || op == HandleOp::Not || op == HandleOp::LNot) {
            const u32 a = pop();
            push(op == HandleOp::Neg ? 0U - a : op == HandleOp::Not ? ~a : (a == 0 ? 1U : 0U));
            continue;
        }
        if (op == HandleOp::Tee) {
            const u32 slot = operand();
            if (slot < slots.size() && size > 0) slots[slot] = stack[size - 1];
            continue;
        }
        if (op == HandleOp::Load) {
            const u32 slot = operand();
            push(slot < slots.size() ? slots[slot] : 0U);
            continue;
        }
        if (op == HandleOp::Select) {
            const u32 false_value = pop(), true_value = pop(), condition = pop();
            push(condition != 0 ? true_value : false_value);
            continue;
        }
        const u32 b = pop(), a = pop();
        const s32 sa = static_cast<s32>(a), sb = static_cast<s32>(b);
        switch (op) {
        case HandleOp::Add: push(a + b); break;
        case HandleOp::Sub: push(a - b); break;
        case HandleOp::Mul: push(a * b); break;
        case HandleOp::And: push(a & b); break;
        case HandleOp::Or: push(a | b); break;
        case HandleOp::Xor: push(a ^ b); break;
        case HandleOp::Shl: push(b >= 32 ? 0U : a << b); break;
        case HandleOp::Shr: push(b >= 32 ? 0U : a >> b); break;
        case HandleOp::Sar: push(static_cast<u32>(sa >> (b >= 32 ? 31 : b))); break;
        case HandleOp::UMin: push(a < b ? a : b); break;
        case HandleOp::UMax: push(a > b ? a : b); break;
        case HandleOp::SMin: push(static_cast<u32>(sa < sb ? sa : sb)); break;
        case HandleOp::SMax: push(static_cast<u32>(sa > sb ? sa : sb)); break;
        case HandleOp::SLt: push(sa < sb); break;
        case HandleOp::SLe: push(sa <= sb); break;
        case HandleOp::SGt: push(sa > sb); break;
        case HandleOp::SGe: push(sa >= sb); break;
        case HandleOp::ULt: push(a < b); break;
        case HandleOp::ULe: push(a <= b); break;
        case HandleOp::UGt: push(a > b); break;
        case HandleOp::UGe: push(a >= b); break;
        case HandleOp::Eq: push(a == b); break;
        case HandleOp::Ne: push(a != b); break;
        case HandleOp::LAnd: push(a != 0 && b != 0); break;
        case HandleOp::LOr: push(a != 0 || b != 0); break;
        case HandleOp::LXor: push((a != 0) != (b != 0)); break;
        default: return std::nullopt;
        }
    }
    if (size != 1) return std::nullopt;
    return stack[0];
}

} // namespace Shader
