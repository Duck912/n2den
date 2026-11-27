// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {
namespace {
void MemoryBarrier(EmitContext& ctx, spv::Scope scope) {
    const auto semantics{
        spv::MemorySemanticsMask::AcquireRelease | spv::MemorySemanticsMask::UniformMemory |
        spv::MemorySemanticsMask::WorkgroupMemory | spv::MemorySemanticsMask::AtomicCounterMemory |
        spv::MemorySemanticsMask::ImageMemory};
    ctx.OpMemoryBarrier(ctx.Const(static_cast<u32>(scope)), ctx.Const(static_cast<u32>(semantics)));
}
} // Anonymous namespace

void EmitBarrier(EmitContext& ctx) {
    const auto execution{spv::Scope::Workgroup};
    const auto memory{spv::Scope::Workgroup};
    // Diagnostic (Kirby Air Riders list freeze investigation): EDEN_NX2_BARRIER_ALL_MEMORY=1: the workgroup barrier also orders
    // storage buffer and image memory (as BAR.SYNC orders every memory access of the CTA).
    static const bool all_memory = [] {
        const char* env = std::getenv("EDEN_NX2_BARRIER_ALL_MEMORY");
        return env != nullptr && env[0] == '1';
    }();
    const auto memory_semantics{
        all_memory ? (spv::MemorySemanticsMask::AcquireRelease |
                      spv::MemorySemanticsMask::WorkgroupMemory |
                      spv::MemorySemanticsMask::UniformMemory |
                      spv::MemorySemanticsMask::ImageMemory)
                   : (spv::MemorySemanticsMask::AcquireRelease |
                      spv::MemorySemanticsMask::WorkgroupMemory)};
    ctx.OpControlBarrier(ctx.Const(static_cast<u32>(execution)),
                         ctx.Const(static_cast<u32>(memory)),
                         ctx.Const(static_cast<u32>(memory_semantics)));
}

void EmitWorkgroupMemoryBarrier(EmitContext& ctx) {
    MemoryBarrier(ctx, spv::Scope::Workgroup);
}

void EmitDeviceMemoryBarrier(EmitContext& ctx) {
    MemoryBarrier(ctx, spv::Scope::Device);
}

} // namespace Shader::Backend::SPIRV
