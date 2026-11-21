// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {
namespace {
Id SubgroupScope(EmitContext& ctx) {
    return ctx.Const(static_cast<u32>(spv::Scope::Subgroup));
}

bool StageSupportsSubgroups(EmitContext& ctx) {
    return ctx.profile.SupportsSubgroupStage(ctx.stage);
}

Id GetThreadId(EmitContext& ctx) {
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.u32_zero_value;
    }
    return ctx.OpLoad(ctx.U32[1], ctx.subgroup_local_invocation_id);
}

Id WarpExtract(EmitContext& ctx, Id value) {
    const Id thread_id{GetThreadId(ctx)};
    const Id local_index{ctx.OpShiftRightArithmetic(ctx.U32[1], thread_id, ctx.Const(5U))};
    if (ctx.profile.has_broken_spirv_subgroup_mask_vector_extract_dynamic) {
        const Id c0_sel{ctx.OpSelect(ctx.U32[1], ctx.OpIEqual(ctx.U1, local_index, ctx.Const(0U)),
                                     ctx.OpCompositeExtract(ctx.U32[1], value, 0U), ctx.Const(0U))};
        const Id c1_sel{ctx.OpSelect(ctx.U32[1], ctx.OpIEqual(ctx.U1, local_index, ctx.Const(1U)),
                                     ctx.OpCompositeExtract(ctx.U32[1], value, 1U), ctx.Const(0U))};
        const Id c2_sel{ctx.OpSelect(ctx.U32[1], ctx.OpIEqual(ctx.U1, local_index, ctx.Const(2U)),
                                     ctx.OpCompositeExtract(ctx.U32[1], value, 2U), ctx.Const(0U))};
        const Id c3_sel{ctx.OpSelect(ctx.U32[1], ctx.OpIEqual(ctx.U1, local_index, ctx.Const(3U)),
                                     ctx.OpCompositeExtract(ctx.U32[1], value, 3U), ctx.Const(0U))};
        const Id c0_or_c1{ctx.OpBitwiseOr(ctx.U32[1], c0_sel, c1_sel)};
        const Id c2_or_c3{ctx.OpBitwiseOr(ctx.U32[1], c2_sel, c3_sel)};
        const Id c0_or_c1_or_c2_or_c3{ctx.OpBitwiseOr(ctx.U32[1], c0_or_c1, c2_or_c3)};
        return c0_or_c1_or_c2_or_c3;
    } else {
        return ctx.OpVectorExtractDynamic(ctx.U32[1], value, local_index);
    }
}

Id LoadMask(EmitContext& ctx, Id mask) {
    const Id value{ctx.OpLoad(ctx.U32[4], mask)};
    if (!ctx.profile.warp_size_potentially_larger_than_guest) {
        return ctx.OpCompositeExtract(ctx.U32[1], value, 0U);
    }
    return WarpExtract(ctx, value);
}

void SetInBoundsFlag(IR::Inst* inst, Id result) {
    IR::Inst* const in_bounds{inst->GetAssociatedPseudoOperation(IR::Opcode::GetInBoundsFromOp)};
    if (!in_bounds) {
        return;
    }
    in_bounds->SetDefinition(result);
    in_bounds->Invalidate();
}

Id ComputeMinThreadId(EmitContext& ctx, Id thread_id, Id segmentation_mask) {
    return ctx.OpBitwiseAnd(ctx.U32[1], thread_id, segmentation_mask);
}

Id ComputeMaxThreadId(EmitContext& ctx, Id min_thread_id, Id clamp, Id not_seg_mask) {
    return ctx.OpBitwiseOr(ctx.U32[1], min_thread_id,
                           ctx.OpBitwiseAnd(ctx.U32[1], clamp, not_seg_mask));
}

Id GetMaxThreadId(EmitContext& ctx, Id thread_id, Id clamp, Id segmentation_mask) {
    const Id not_seg_mask{ctx.OpNot(ctx.U32[1], segmentation_mask)};
    const Id min_thread_id{ComputeMinThreadId(ctx, thread_id, segmentation_mask)};
    return ComputeMaxThreadId(ctx, min_thread_id, clamp, not_seg_mask);
}

Id HostThreadId(EmitContext& ctx, Id thread_id) {
    if (!ctx.profile.warp_size_potentially_larger_than_guest) {
        return thread_id;
    }
    const Id partition_idx{ctx.OpShiftRightLogical(ctx.U32[1], GetThreadId(ctx), ctx.Const(5u))};
    const Id partition_base{ctx.OpShiftLeftLogical(ctx.U32[1], partition_idx, ctx.Const(5u))};
    return ctx.OpIAdd(ctx.U32[1], thread_id, partition_base);
}

Id GuestLane(EmitContext& ctx, Id index) {
    return ctx.OpBitwiseAnd(ctx.U32[1], index, ctx.Const(31U));
}

Id ShuffleAbsolute(EmitContext& ctx, Id value, Id src_thread_id) {
    if (!ctx.profile.has_broken_spirv_subgroup_shuffle) {
        return ctx.OpGroupNonUniformShuffle(ctx.U32[1], SubgroupScope(ctx), value, src_thread_id);
    }
    Id result{ctx.u32_zero_value};
    for (u32 lane = 0; lane < ctx.profile.max_subgroup_size; ++lane) {
        const Id read{
            ctx.OpGroupNonUniformBroadcast(ctx.U32[1], SubgroupScope(ctx), value, ctx.Const(lane))};
        const Id matches{ctx.OpIEqual(ctx.U1, src_thread_id, ctx.Const(lane))};
        result = ctx.OpSelect(ctx.U32[1], matches, read, result);
    }
    return result;
}

Id ShuffleRelative(EmitContext& ctx, Id value, Id delta, Id src_thread_id, spv::Op op) {
    if (!ctx.profile.support_shuffle_relative) {
        return ShuffleAbsolute(ctx, value, HostThreadId(ctx, src_thread_id));
    }
    const Id result{ctx.OpGroupNonUniformShuffleXor(ctx.U32[1], SubgroupScope(ctx), value, delta)};
    ctx.opcode_rewrites.emplace_back(result.value, op);
    return result;
}

Id BroadcastLane(EmitContext& ctx, Id value, u32 lane) {
    Id result{
        ctx.OpGroupNonUniformBroadcast(ctx.U32[1], SubgroupScope(ctx), value, ctx.Const(lane))};
    if (!ctx.profile.warp_size_potentially_larger_than_guest) {
        return result;
    }
    const Id partition_idx{ctx.OpShiftRightLogical(ctx.U32[1], GetThreadId(ctx), ctx.Const(5u))};
    for (u32 base = 32; base < ctx.profile.max_subgroup_size; base += 32) {
        const Id read{ctx.OpGroupNonUniformBroadcast(ctx.U32[1], SubgroupScope(ctx), value,
                                                     ctx.Const(base + lane))};
        const Id matches{ctx.OpIEqual(ctx.U1, partition_idx, ctx.Const(base >> 5))};
        result = ctx.OpSelect(ctx.U32[1], matches, read, result);
    }
    return result;
}
} // Anonymous namespace

Id EmitLaneId(EmitContext& ctx) {
    const Id id{GetThreadId(ctx)};
    if (!ctx.profile.warp_size_potentially_larger_than_guest) {
        return id;
    }
    return ctx.OpBitwiseAnd(ctx.U32[1], id, ctx.Const(31U));
}

Id EmitVoteAll(EmitContext& ctx, Id pred) {
    if (!StageSupportsSubgroups(ctx)) {
        return pred;
    }
    if (!ctx.profile.warp_size_potentially_larger_than_guest) {
        return ctx.OpGroupNonUniformAll(ctx.U1, SubgroupScope(ctx), pred);
    }
    const Id mask_ballot{
        ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), ctx.true_value)};
    const Id active_mask{WarpExtract(ctx, mask_ballot)};
    const Id ballot{
        WarpExtract(ctx, ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), pred))};
    const Id lhs{ctx.OpBitwiseAnd(ctx.U32[1], ballot, active_mask)};
    return ctx.OpIEqual(ctx.U1, lhs, active_mask);
}

Id EmitVoteAny(EmitContext& ctx, Id pred) {
    if (!StageSupportsSubgroups(ctx)) {
        return pred;
    }
    if (!ctx.profile.warp_size_potentially_larger_than_guest) {
        return ctx.OpGroupNonUniformAny(ctx.U1, SubgroupScope(ctx), pred);
    }
    const Id mask_ballot{
        ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), ctx.true_value)};
    const Id active_mask{WarpExtract(ctx, mask_ballot)};
    const Id ballot{
        WarpExtract(ctx, ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), pred))};
    const Id lhs{ctx.OpBitwiseAnd(ctx.U32[1], ballot, active_mask)};
    return ctx.OpINotEqual(ctx.U1, lhs, ctx.u32_zero_value);
}

Id EmitVoteEqual(EmitContext& ctx, Id pred) {
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.true_value;
    }
    if (!ctx.profile.warp_size_potentially_larger_than_guest) {
        return ctx.OpGroupNonUniformAllEqual(ctx.U1, SubgroupScope(ctx), pred);
    }
    const Id mask_ballot{
        ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), ctx.true_value)};
    const Id active_mask{WarpExtract(ctx, mask_ballot)};
    const Id ballot{
        WarpExtract(ctx, ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), pred))};
    const Id lhs{ctx.OpBitwiseXor(ctx.U32[1], ballot, active_mask)};
    return ctx.OpLogicalOr(ctx.U1, ctx.OpIEqual(ctx.U1, lhs, ctx.u32_zero_value),
                           ctx.OpIEqual(ctx.U1, lhs, active_mask));
}

Id EmitSubgroupBallot(EmitContext& ctx, Id pred) {
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.OpSelect(ctx.U32[1], pred, ctx.Const(1u), ctx.u32_zero_value);
    }
    if (ctx.profile.UseSubgroupBallotReduction()) {
        ctx.AddCapability(spv::Capability::GroupNonUniformArithmetic);
        const Id bit{ctx.OpShiftLeftLogical(ctx.U32[1], ctx.Const(1u), GetThreadId(ctx))};
        const Id contribution{ctx.OpSelect(ctx.U32[1], pred, bit, ctx.u32_zero_value)};
        const Id result{ctx.OpGroupNonUniformShuffleXor(ctx.U32[1], SubgroupScope(ctx),
                                                       contribution, ctx.u32_zero_value)};
        ctx.opcode_rewrites.emplace_back(result.value, spv::Op::OpGroupNonUniformBitwiseOr);
        return result;
    }
    const Id ballot{ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), pred)};
    if (!ctx.profile.warp_size_potentially_larger_than_guest) {
        return ctx.OpCompositeExtract(ctx.U32[1], ballot, 0U);
    }
    return WarpExtract(ctx, ballot);
}

Id EmitSubgroupEqMask(EmitContext& ctx) {
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.Const(1u);
    }
    return LoadMask(ctx, ctx.subgroup_mask_eq);
}

Id EmitSubgroupLtMask(EmitContext& ctx) {
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.u32_zero_value;
    }
    return LoadMask(ctx, ctx.subgroup_mask_lt);
}

Id EmitSubgroupLeMask(EmitContext& ctx) {
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.Const(1u);
    }
    return LoadMask(ctx, ctx.subgroup_mask_le);
}

Id EmitSubgroupGtMask(EmitContext& ctx) {
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.u32_zero_value;
    }
    return LoadMask(ctx, ctx.subgroup_mask_gt);
}

Id EmitSubgroupGeMask(EmitContext& ctx) {
    if (!StageSupportsSubgroups(ctx)) {
        return ctx.Const(1u);
    }
    return LoadMask(ctx, ctx.subgroup_mask_ge);
}

Id EmitShuffleIndex(EmitContext& ctx, IR::Inst* inst, Id value, Id index, Id clamp,
                    Id segmentation_mask) {
    const Id not_seg_mask{ctx.OpNot(ctx.U32[1], segmentation_mask)};
    const Id thread_id{EmitLaneId(ctx)};
    const Id min_thread_id{ComputeMinThreadId(ctx, thread_id, segmentation_mask)};
    const Id max_thread_id{ComputeMaxThreadId(ctx, min_thread_id, clamp, not_seg_mask)};

    const Id lhs{ctx.OpBitwiseAnd(ctx.U32[1], GuestLane(ctx, index), not_seg_mask)};
    const Id src_thread_id{ctx.OpBitwiseOr(ctx.U32[1], lhs, min_thread_id)};
    const Id in_range{ctx.OpSLessThanEqual(ctx.U1, src_thread_id, max_thread_id)};

    SetInBoundsFlag(inst, in_range);
    if (!StageSupportsSubgroups(ctx)) {
        return value;
    }
    const IR::Value lane{inst->Arg(1).Resolve()};
    const IR::Value segment{inst->Arg(3).Resolve()};
    if (lane.IsImmediate() && segment.IsImmediate() && segment.U32() == 0) {
        return ctx.OpSelect(ctx.U32[1], in_range, BroadcastLane(ctx, value, lane.U32() & 31),
                            value);
    }
    const Id shuffled{ShuffleAbsolute(ctx, value, HostThreadId(ctx, src_thread_id))};
    return ctx.OpSelect(ctx.U32[1], in_range, shuffled, value);
}

Id EmitShuffleUp(EmitContext& ctx, IR::Inst* inst, Id value, Id index, Id clamp,
                 Id segmentation_mask) {
    if (!StageSupportsSubgroups(ctx)) {
        SetInBoundsFlag(inst, ctx.false_value);
        return value;
    }
    const Id delta{GuestLane(ctx, index)};
    const Id thread_id{EmitLaneId(ctx)};
    const Id max_thread_id{GetMaxThreadId(ctx, thread_id, clamp, segmentation_mask)};
    const Id src_thread_id{ctx.OpISub(ctx.U32[1], thread_id, delta)};
    const Id in_range{ctx.OpSGreaterThanEqual(ctx.U1, src_thread_id, max_thread_id)};

    SetInBoundsFlag(inst, in_range);
    const Id shuffled{ShuffleRelative(ctx, value, delta, src_thread_id,
                                      spv::Op::OpGroupNonUniformShuffleUp)};
    return ctx.OpSelect(ctx.U32[1], in_range, shuffled, value);
}

Id EmitShuffleDown(EmitContext& ctx, IR::Inst* inst, Id value, Id index, Id clamp,
                   Id segmentation_mask) {
    if (!StageSupportsSubgroups(ctx)) {
        SetInBoundsFlag(inst, ctx.false_value);
        return value;
    }
    const Id delta{GuestLane(ctx, index)};
    const Id thread_id{EmitLaneId(ctx)};
    const Id max_thread_id{GetMaxThreadId(ctx, thread_id, clamp, segmentation_mask)};
    const Id src_thread_id{ctx.OpIAdd(ctx.U32[1], thread_id, delta)};
    const Id in_range{ctx.OpSLessThanEqual(ctx.U1, src_thread_id, max_thread_id)};

    SetInBoundsFlag(inst, in_range);
    const Id shuffled{ShuffleRelative(ctx, value, delta, src_thread_id,
                                      spv::Op::OpGroupNonUniformShuffleDown)};
    return ctx.OpSelect(ctx.U32[1], in_range, shuffled, value);
}

Id EmitShuffleButterfly(EmitContext& ctx, IR::Inst* inst, Id value, Id index, Id clamp,
                        Id segmentation_mask) {
    if (!StageSupportsSubgroups(ctx)) {
        SetInBoundsFlag(inst, ctx.false_value);
        return value;
    }
    const Id mask{GuestLane(ctx, index)};
    const Id thread_id{EmitLaneId(ctx)};
    const Id max_thread_id{GetMaxThreadId(ctx, thread_id, clamp, segmentation_mask)};
    const Id src_thread_id{ctx.OpBitwiseXor(ctx.U32[1], thread_id, mask)};
    const Id in_range{ctx.OpSLessThanEqual(ctx.U1, src_thread_id, max_thread_id)};

    SetInBoundsFlag(inst, in_range);
    const Id shuffled{ctx.OpGroupNonUniformShuffleXor(ctx.U32[1], SubgroupScope(ctx), value, mask)};
    return ctx.OpSelect(ctx.U32[1], in_range, shuffled, value);
}

Id EmitQuadBroadcast(EmitContext& ctx, Id value, Id lane) {
    if (ctx.profile.support_quad_shuffles) {
        return ctx.OpGroupNonUniformQuadBroadcast(ctx.U32[1], SubgroupScope(ctx), value, lane);
    }
    const Id base{ctx.OpBitwiseAnd(ctx.U32[1], GetThreadId(ctx), ctx.Const(~3u))};
    const Id local_lane{ctx.OpBitwiseAnd(ctx.U32[1], lane, ctx.Const(3u))};
    const Id src_thread_id{ctx.OpBitwiseOr(ctx.U32[1], base, local_lane)};
    return ShuffleAbsolute(ctx, value, src_thread_id);
}

Id EmitQuadSwap(EmitContext& ctx, Id value, Id direction) {
    if (ctx.profile.support_quad_shuffles) {
        const Id result{
            ctx.OpGroupNonUniformQuadBroadcast(ctx.U32[1], SubgroupScope(ctx), value, direction)};
        ctx.opcode_rewrites.emplace_back(result.value, spv::Op::OpGroupNonUniformQuadSwap);
        return result;
    }
    const Id xor_mask{ctx.OpIAdd(ctx.U32[1], direction, ctx.Const(1u))};
    return ctx.OpGroupNonUniformShuffleXor(ctx.U32[1], SubgroupScope(ctx), value, xor_mask);
}

Id EmitFSwizzleAdd(EmitContext& ctx, Id op_a, Id op_b, Id swizzle) {
    const Id three{ctx.Const(3U)};
    Id mask{GetThreadId(ctx)};
    mask = ctx.OpBitwiseAnd(ctx.U32[1], mask, three);
    mask = ctx.OpShiftLeftLogical(ctx.U32[1], mask, ctx.Const(1U));
    mask = ctx.OpShiftRightLogical(ctx.U32[1], swizzle, mask);
    mask = ctx.OpBitwiseAnd(ctx.U32[1], mask, three);

    const Id modifier_a{ctx.OpVectorExtractDynamic(ctx.F32[1], ctx.fswzadd_lut_a, mask)};
    const Id modifier_b{ctx.OpVectorExtractDynamic(ctx.F32[1], ctx.fswzadd_lut_b, mask)};

    const Id result_a{ctx.OpFMul(ctx.F32[1], op_a, modifier_a)};
    const Id result_b{ctx.OpFMul(ctx.F32[1], op_b, modifier_b)};
    return ctx.OpFAdd(ctx.F32[1], result_a, result_b);
}

Id EmitDPdxFine(EmitContext& ctx, Id op_a) {
    return ctx.OpDPdxFine(ctx.F32[1], op_a);
}

Id EmitDPdyFine(EmitContext& ctx, Id op_a) {
    return ctx.OpDPdyFine(ctx.F32[1], op_a);
}

Id EmitDPdxCoarse(EmitContext& ctx, Id op_a) {
    return ctx.OpDPdxCoarse(ctx.F32[1], op_a);
}

Id EmitDPdyCoarse(EmitContext& ctx, Id op_a) {
    return ctx.OpDPdyCoarse(ctx.F32[1], op_a);
}

namespace {
// HMMA.16816.F16 with shuffles: each lane gathers its rows of A and columns of B and accumulates
// in single precision, rounding to half after every eight products (the reference lowering).
Id HalfMatrixMultiplyAddShuffle(EmitContext& ctx, Id a, Id b, Id c) {
    const auto word = [&](Id composite, u32 index) {
        return ctx.OpCompositeExtract(ctx.U32[1], composite, index);
    };
    const std::array<Id, 4> a_words{word(a, 0), word(a, 1), word(a, 2), word(a, 3)};
    const std::array<Id, 2> b_words{word(b, 0), word(b, 1)};
    const Id lane{EmitLaneId(ctx)};
    const bool subgroups{StageSupportsSubgroups(ctx)};
    const auto shuffle = [&](Id value, Id source_lane) {
        return subgroups ? ShuffleAbsolute(ctx, value, HostThreadId(ctx, source_lane)) : value;
    };
    const auto unpack = [&](Id packed) { return ctx.OpUnpackHalf2x16(ctx.F32[2], packed); };
    const auto half = [&](Id pair, u32 index) {
        return ctx.OpCompositeExtract(ctx.F32[1], pair, index);
    };
    const Id c0{unpack(word(c, 0))};
    const Id c1{unpack(word(c, 1))};
    std::array<Id, 4> acc{half(c0, 0), half(c0, 1), half(c1, 0), half(c1, 1)};
    // Rounds a single-precision value to the nearest half (ties to even) and back. Not written
    // as unpack(pack(value)): drivers may eliminate that round trip. Same as Eglantine's software
    // HMMA (e88024e6b0, emit_spirv_tensor.cpp). Half subnormals lie on the exact 2^-24 grid.
    const auto round_half = [&](Id value) {
        const Id u32_type{ctx.U32[1]};
        const Id f32_type{ctx.F32[1]};
        const auto and_imm = [&](Id x, u32 mask) {
            return ctx.OpBitwiseAnd(u32_type, x, ctx.Const(mask));
        };
        const Id bits{ctx.OpBitcast(u32_type, value)};
        const Id magnitude{and_imm(bits, 0x7fffffffU)};
        const Id tie{and_imm(ctx.OpShiftRightLogical(u32_type, magnitude, ctx.Const(13U)), 1U)};
        const Id rounded{and_imm(
            ctx.OpIAdd(u32_type, ctx.OpIAdd(u32_type, magnitude, ctx.Const(0xfffU)), tie),
            ~0x1fffU)};
        const Id normal{ctx.OpSelect(u32_type,
                                     ctx.OpUGreaterThanEqual(ctx.U1, rounded, ctx.Const(0x47800000U)),
                                     ctx.Const(0x7f800000U), rounded)};
        const Id scaled{
            ctx.OpFMul(f32_type, ctx.OpBitcast(f32_type, magnitude), ctx.Const(16777216.0f))};
        const Id subnormal{ctx.OpBitcast(
            u32_type, ctx.OpFMul(f32_type, ctx.OpRoundEven(f32_type, scaled), ctx.Const(0x1p-24f)))};
        const Id finite{ctx.OpSelect(
            u32_type, ctx.OpUGreaterThanEqual(ctx.U1, magnitude, ctx.Const(0x38800000U)), normal,
            subnormal)};
        const Id result{ctx.OpSelect(
            u32_type, ctx.OpUGreaterThanEqual(ctx.U1, magnitude, ctx.Const(0x7f800000U)),
            magnitude, finite)};
        return ctx.OpBitcast(f32_type,
                             ctx.OpBitwiseOr(u32_type, result, and_imm(bits, 0x80000000U)));
    };
    const auto round_pair = [&](Id& x, Id& y) {
        x = round_half(x);
        y = round_half(y);
    };
    const Id row_base{ctx.OpBitwiseAnd(ctx.U32[1], lane, ctx.Const(0x1cU))};
    const Id column_base{ctx.OpShiftLeftLogical(
        ctx.U32[1], ctx.OpBitwiseAnd(ctx.U32[1], lane, ctx.Const(3U)), ctx.Const(3U))};
    for (u32 pair = 0; pair < 8; ++pair) {
        const Id step{ctx.Const(pair & 3)};
        const Id a_lane{ctx.OpIAdd(ctx.U32[1], row_base, step)};
        const Id b_lane{ctx.OpIAdd(ctx.U32[1], column_base, step)};
        const Id a_row0{unpack(shuffle(a_words[pair >= 4 ? 2 : 0], a_lane))};
        const Id a_row1{unpack(shuffle(a_words[pair >= 4 ? 3 : 1], a_lane))};
        const Id b_col0{unpack(shuffle(b_words[pair / 4], b_lane))};
        const Id b_col1{
            unpack(shuffle(b_words[pair / 4], ctx.OpIAdd(ctx.U32[1], b_lane, ctx.Const(4U))))};
        for (u32 h = 0; h < 2; ++h) {
            const Id b0{half(b_col0, h)};
            const Id b1{half(b_col1, h)};
            const Id r0{half(a_row0, h)};
            const Id r1{half(a_row1, h)};
            acc[0] = ctx.OpFma(ctx.F32[1], r0, b0, acc[0]);
            acc[1] = ctx.OpFma(ctx.F32[1], r0, b1, acc[1]);
            acc[2] = ctx.OpFma(ctx.F32[1], r1, b0, acc[2]);
            acc[3] = ctx.OpFma(ctx.F32[1], r1, b1, acc[3]);
        }
        if ((pair & 3) == 3 && pair + 1 < 8) {
            round_pair(acc[0], acc[1]);
            round_pair(acc[2], acc[3]);
        }
    }
    const Id d0{ctx.OpPackHalf2x16(ctx.U32[1], ctx.OpCompositeConstruct(ctx.F32[2], acc[0], acc[1]))};
    const Id d1{ctx.OpPackHalf2x16(ctx.U32[1], ctx.OpCompositeConstruct(ctx.F32[2], acc[2], acc[3]))};
    return ctx.OpCompositeConstruct(ctx.U32[2], d0, d1);
}

// The same on the host's matrix units: every lane stores its fragment words into its subgroup's
// scratch in the layout the fragments define, the subgroup loads them as cooperative matrices,
// multiplies, stores the accumulator and every lane reads its two result words back.
//   A 16x16 row-major, 8 words per row:  a0 row r k 2q, a1 row r+8, a2 k+8, a3 row r+8 k+8
//   B 16x8 column-major, 8 words/column: b0 column r k 2q, b1 k+8
//   C/D 16x8 row-major, 4 words per row: c0 row r columns 2q, c1 row r+8      (r = lane/4, q = lane%4)
Id HalfMatrixMultiplyAddCooperative(EmitContext& ctx, Id a, Id b, Id c) {
    const Id lane{EmitLaneId(ctx)};
    const Id subgroup{ctx.OpLoad(ctx.U32[1], ctx.coopmat_subgroup_id)};
    const Id base{ctx.OpShiftLeftLogical(ctx.U32[1], subgroup, ctx.Const(8U))};
    const Id row{ctx.OpShiftRightLogical(ctx.U32[1], lane, ctx.Const(2U))};
    const Id quad{ctx.OpBitwiseAnd(ctx.U32[1], lane, ctx.Const(3U))};
    const auto pointer = [&](Id index) {
        if (ctx.coopmat_scratch_in_block) {
            return ctx.OpAccessChain(ctx.coopmat_word_pointer, ctx.coopmat_scratch,
                                     ctx.u32_zero_value, index);
        }
        return ctx.OpAccessChain(ctx.coopmat_word_pointer, ctx.coopmat_scratch, index);
    };
    // Index of word (row, quad + column_word) of a matrix at `offset` with `words` words per row.
    const auto slot = [&](u32 offset, u32 words, u32 row_add, u32 column_add) {
        const Id r{row_add != 0 ? ctx.OpIAdd(ctx.U32[1], row, ctx.Const(row_add)) : row};
        const Id in_matrix{ctx.OpIAdd(ctx.U32[1], ctx.OpIMul(ctx.U32[1], r, ctx.Const(words)),
                                      ctx.OpIAdd(ctx.U32[1], quad, ctx.Const(column_add)))};
        return ctx.OpIAdd(ctx.U32[1], ctx.OpIAdd(ctx.U32[1], base, ctx.Const(offset)), in_matrix);
    };
    const auto word = [&](Id composite, u32 index) {
        return ctx.OpCompositeExtract(ctx.U32[1], composite, index);
    };
    const auto barrier = [&] {
        const u32 semantics{static_cast<u32>(spv::MemorySemanticsMask::AcquireRelease |
                                             spv::MemorySemanticsMask::WorkgroupMemory)};
        ctx.OpControlBarrier(ctx.coopmat_scope, ctx.coopmat_scope, ctx.Const(semantics));
    };
    ctx.OpStore(pointer(slot(0, 8, 0, 0)), word(a, 0));
    ctx.OpStore(pointer(slot(0, 8, 8, 0)), word(a, 1));
    ctx.OpStore(pointer(slot(0, 8, 0, 4)), word(a, 2));
    ctx.OpStore(pointer(slot(0, 8, 8, 4)), word(a, 3));
    ctx.OpStore(pointer(slot(128, 8, 0, 0)), word(b, 0));
    ctx.OpStore(pointer(slot(128, 8, 0, 4)), word(b, 1));
    ctx.OpStore(pointer(slot(192, 4, 0, 0)), word(c, 0));
    ctx.OpStore(pointer(slot(192, 4, 8, 0)), word(c, 1));
    barrier();
    const Id row_major{ctx.Const(0U)};
    const Id column_major{ctx.Const(1U)};
    const auto matrix_at = [&](u32 offset) {
        return pointer(ctx.OpIAdd(ctx.U32[1], base, ctx.Const(offset)));
    };
    const auto load = [&](size_t type, u32 offset, Id layout, u32 stride) {
        const std::array operands{matrix_at(offset), layout, ctx.Const(stride)};
        return ctx.OpExtInst(ctx.coopmat_types[type], ctx.coopmat_glsl, CoopMarker::Load, operands);
    };
    const Id matrix_a{load(0, 0, row_major, 8)};
    const Id matrix_b{load(1, 128, column_major, 8)};
    const Id matrix_c{load(2, 192, row_major, 4)};
    const std::array mul_operands{matrix_a, matrix_b, matrix_c};
    const Id matrix_d{
        ctx.OpExtInst(ctx.coopmat_types[2], ctx.coopmat_glsl, CoopMarker::MulAdd, mul_operands)};
    const std::array store_operands{matrix_at(192), matrix_d, row_major, ctx.Const(4U)};
    ctx.OpExtInst(ctx.void_id, ctx.coopmat_glsl, CoopMarker::Store, store_operands);
    barrier();
    const Id d0{ctx.OpLoad(ctx.U32[1], pointer(slot(192, 4, 0, 0)))};
    const Id d1{ctx.OpLoad(ctx.U32[1], pointer(slot(192, 4, 8, 0)))};
    barrier(); // The next multiply's stores must not overtake these reads
    return ctx.OpCompositeConstruct(ctx.U32[2], d0, d1);
}
} // Anonymous namespace

// The body exists once per module. Inlined at each of the hundreds of call sites of a network
// program it was most of the driver compile time (e27336d3, 768 HMMA: 19 s with shuffles inline).
void DefineHalfMatrixMultiplyAddFunction(EmitContext& ctx) {
    const Id function_type{ctx.TypeFunction(ctx.U32[2], ctx.U32[4], ctx.U32[2], ctx.U32[2])};
    ctx.hmma_func =
        ctx.OpFunction(ctx.U32[2], spv::FunctionControlMask::DontInline, function_type);
    const Id a{ctx.OpFunctionParameter(ctx.U32[4])};
    const Id b{ctx.OpFunctionParameter(ctx.U32[2])};
    const Id c{ctx.OpFunctionParameter(ctx.U32[2])};
    ctx.AddLabel();
    ctx.OpReturnValue(ctx.uses_cooperative_matrix ? HalfMatrixMultiplyAddCooperative(ctx, a, b, c)
                                                  : HalfMatrixMultiplyAddShuffle(ctx, a, b, c));
    ctx.OpFunctionEnd();
}

Id EmitHalfMatrixMultiplyAdd16816(EmitContext& ctx, Id a, Id b, Id c) {
    const std::array arguments{a, b, c};
    return ctx.OpFunctionCall(ctx.U32[2], ctx.hmma_func, arguments);
}

} // namespace Shader::Backend::SPIRV
