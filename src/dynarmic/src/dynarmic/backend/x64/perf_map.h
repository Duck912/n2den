// SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <bit>

namespace Dynarmic::Backend::X64 {

#if defined(__linux__) && !defined(__ANDROID__)
namespace detail {
void PerfMapRegister(const void* start, const void* end, std::string_view friendly_name);
}  // namespace detail
template<typename T>
void PerfMapRegister(T start, const void* end, std::string_view friendly_name) noexcept {
    detail::PerfMapRegister(std::bit_cast<const void*>(start), end, friendly_name);
}
void PerfMapClear();
#else
// Resolve to no-op (compiler thinks fmt has side effects)
template<typename T> inline void PerfMapRegister(T, const void*, std::string_view) noexcept {}
inline void PerfMapClear() noexcept {}
#endif

/// The A64::HotMap side table (interface/A64/a64.h). Callers check HotMapEnabled() first.
bool HotMapEnabled();
/// Records one block of the code space [space, space + space_size) whose blocks start at code_begin.
void HotMapAdd(const void* space, const void* code_begin, std::size_t space_size, const void* host,
               std::size_t host_size, std::uint64_t guest_pc, std::uint64_t guest_end);
/// The code space's whole cache was cleared: its host code will be reused.
void HotMapClear(const void* space);
/// The code space is going away.
void HotMapForget(const void* space);

}  // namespace Dynarmic::Backend::X64
