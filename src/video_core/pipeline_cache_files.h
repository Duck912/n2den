// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include <string_view>

#include "common/common_types.h"

namespace VideoCommon {

// NVN-HLE's disk caches: the NVN pipeline records and the driver's VkPipelineCache data. Bump
// NVN_PIPELINE_CACHE_VERSION when the record format changes (nvn_pipeline_disk.h).
inline constexpr u32 NVN_PIPELINE_CACHE_VERSION = 18;
inline constexpr u32 NVN_DRIVER_CACHE_VERSION = 1;

inline constexpr std::array<char, 8> NVN_PIPELINE_CACHE_MAGIC{'N', 'V', 'N', 'P',
                                                               'I', 'P', 'E', 'L'};
inline constexpr std::array<char, 8> NVN_DRIVER_CACHE_MAGIC{'N', 'V', 'N', 'V',
                                                             'K', 'C', 'H', '1'};
inline constexpr std::string_view NVN_DRIVER_CACHE_FILE_NAME = "nvn_pipelines.vkcache";

} // namespace VideoCommon
