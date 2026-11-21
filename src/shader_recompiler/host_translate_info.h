// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/common_types.h"

namespace Shader {

// Try to keep entries here to a minimum
// They can accidentally change the cached information in a shader

/// How texture handles that are not constant buffer words are bound.
enum class TexturePoolMode : u32 {
    Off,      ///< Per-draw descriptors only (buffer-backed arrays, handle programs)
    Fallback, ///< Texture pool lookup at run time for the handles nothing per draw can express
    All,      ///< Texture pool lookup at run time for every handle read from memory
};

/// Misc information about the host
struct HostTranslateInfo {
    static constexpr u32 DEFAULT_DESCRIPTOR_LIMIT = 1024;

    u64 min_ssbo_alignment{};            ///< Minimum alignment supported by the device for SSBOs
    u32 max_per_stage_descriptor_sampled_images{}; ///< maximum sampled descriptors per stage
    u32 max_per_stage_resources{};                 ///< maximum resources per stage
    u32 max_per_stage_descriptor_storage_buffers{}; ///< maximum storage buffers per stage
    u32 max_descriptor_set_samplers{};
    u32 max_descriptor_set_uniform_buffers{};
    u32 max_descriptor_set_uniform_buffers_dynamic{};
    u32 max_descriptor_set_storage_buffers{};
    u32 max_descriptor_set_storage_buffers_dynamic{};
    u32 max_descriptor_set_sampled_images{};
    u32 max_descriptor_set_storage_images{};
    u32 max_descriptor_set_input_attachements{};
    bool support_float64{};      ///< True when the device supports 64-bit floats
    bool support_float16{};      ///< True when the device supports 16-bit floats
    bool support_int64{};        ///< True when the device supports 64-bit integers
    bool needs_demote_reorder{}; ///< True when the device needs DemoteToHelperInvocation reordered
    bool support_snorm_render_buffer{};  ///< True when the device supports SNORM render buffers
    bool support_viewport_index_layer{}; ///< True when the device supports gl_Layer in VS
    bool support_viewport_mask{};
    bool support_geometry_shader_passthrough{}; ///< True when the device supports geometry
                                                ///< passthrough shaders
    bool support_conditional_barrier{}; ///< True when the device supports barriers in conditional
                                        ///< control flow
    bool single_lane_geometry_subgroups{};
    bool support_buffer_texture_handles{};
    /// Set by a renderer that binds the texture pool descriptor set (shader_info.h).
    TexturePoolMode texture_pool_mode{TexturePoolMode::Off};
    /// nvn-hle only (the emulated GPU leaves it false): a texture handle read per fragment from
    /// memory, which no per-draw binding can express (Ocarina of Time's shadow-map handles in its
    /// light records), gets a placeholder handle program (handle 0, which names no texture, so the
    /// sink binds its dummy) instead of rejecting the program. The draws run with that texture
    /// wrong, as on Eglantine; the texture pool in the sink is the real fix.
    bool unresolved_handle_placeholder{};
    /// nvn-hle only (from Eglantine; only nvn-hle's sink sets these, the emulated GPU
    /// leaves them all false): a global memory access whose address comes out of a memory load is
    /// never tied to a storage buffer (the search stops at loads); such accesses stay raw global
    /// memory (Profile::support_raw_global_memory).
    bool nvn_raw_global_memory{};
    /// With `nvn_raw_global_memory`: addresses that do not come out of a memory load are still
    /// tracked without the driver bank bias (any constant buffer word is a storage buffer
    /// descriptor); only what that fails to track is raw. The nvn-hle driver bank has none, it
    /// leaves this off and treats everything outside the biased window as raw.
    bool nvn_raw_global_unbiased_tracking{};
    /// With `nvn_raw_global_memory`: the guest -> device address table is a storage buffer of the
    /// program (its last descriptor, cbuf_index == RAW_GLOBAL_TABLE_CBUF) that the host binds,
    /// instead of living in the driver bank (c0).
    bool nvn_raw_global_table_ssbo{};
    /// With `nvn_raw_global_memory`: the raw coverage diagnostic (EDEN_NX2_RAW_COVERAGE=1): a
    /// raw access the table does not hold appends a record to the miss ring named by the table
    /// header (the table storage buffer, or c0 for the driver bank table; raw_coverage_layout.h).
    /// Off: the module is the one without the diagnostic.
    bool raw_coverage{};
    /// With `nvn_raw_global_memory`: every invocation remembers the table entry of its last
    /// translation and maps an address inside it without the binary search
    /// (EDEN_NX2_TRY_RAWGLOBAL_FAST=1, VideoCommon::TranslateRawGlobalAddressLastHit is the
    /// reference). Same results: the entries never overlap and do not change while the program
    /// runs.
    bool raw_global_last_hit{};
    /// With `raw_global_last_hit` and `raw_coverage` (EDEN_NX2_RAW_LASTHIT_VERIFY=1, measurement
    /// only): every translation computes both the last-hit result and the search result, returns
    /// the search result and counts the differences in the coverage ring
    /// (raw_coverage_layout.h, RING_LASTHIT_MISMATCH_WORD).
    bool raw_global_last_hit_verify{};
    /// With `nvn_raw_global_memory` and the driver bank table: every raw store (32/64/128-bit
    /// and the 8/16-bit field inserts) in an alias range writes to its mirror target too, and
    /// every raw atomic repeats its operation there. Targets are read-only mirrors or imported
    /// guest storage backing (EDEN_NX2_TRY_RO_MIRROR_EXACT /
    /// EDEN_NX2_TRY_STORAGE_RAW_WRITE_THROUGH).
    bool raw_global_store_alias{};
    /// Leave out the RescalingPass even while the resolution setting scales: a host whose draws
    /// never scale a target (nvn-hle: the game renders at the sizes it asks the driver for) gets
    /// the same SPIR-V whatever the resolution setting is. Off: the emulated GPU, which rescales.
    bool skip_rescaling_pass{};

    void ApplyDescriptorLimitPolicy() noexcept {
        if (min_ssbo_alignment == 0) {
            min_ssbo_alignment = 1;
        }
        ApplyDescriptorLimitFallback(max_per_stage_descriptor_sampled_images);
        ApplyDescriptorLimitFallback(max_per_stage_resources);
        ApplyDescriptorLimitFallback(max_per_stage_descriptor_storage_buffers);
        ApplyDescriptorLimitFallback(max_descriptor_set_samplers);
        ApplyDescriptorLimitFallback(max_descriptor_set_uniform_buffers);
        ApplyDescriptorLimitFallback(max_descriptor_set_uniform_buffers_dynamic);
        ApplyDescriptorLimitFallback(max_descriptor_set_storage_buffers);
        ApplyDescriptorLimitFallback(max_descriptor_set_storage_buffers_dynamic);
        ApplyDescriptorLimitFallback(max_descriptor_set_sampled_images);
        ApplyDescriptorLimitFallback(max_descriptor_set_storage_images);
        ApplyDescriptorLimitFallback(max_descriptor_set_input_attachements);
    }

private:
    static void ApplyDescriptorLimitFallback(u32& limit) noexcept {
        if (limit == 0) {
            limit = DEFAULT_DESCRIPTOR_LIMIT;
        }
    }
};

} // namespace Shader
