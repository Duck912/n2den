// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <string>
#include <vector>

namespace Common {

/// Where this process's memory is, for finding what holds RAM (Windows; empty elsewhere).
/// Walks the address space (VirtualQuery) and asks the working set which committed pages are in
/// RAM (QueryWorkingSetEx). The lines report, in MiB:
///  - committed and resident bytes by kind: private, mapped (sections: the guest's RAM is one),
///    image (executables and DLLs);
///  - the heaps' committed total (C++ new/malloc, the driver's heaps) and the largest heaps;
///  - the largest allocations by resident bytes, with their protection (executable = JIT code);
///  - what grew most since the previous call (per allocation), the answer to "what grows".
/// Takes ~0.1-1 s on a process with tens of GiB committed; meant for a call per minute.
[[nodiscard]] std::vector<std::string> ProcessMemoryCensus();

/// Diagnostics (EDEN_MEMORY_TRACK_ALLOC=1, Windows): routes VirtualAlloc and
/// NtAllocateVirtualMemory in the import tables of the loaded modules through a counter of the
/// commits of 1 MiB and more per calling module, which ProcessMemoryCensus then reports. Modules
/// loaded later (the Vulkan driver) are patched at the next census. No-op when the variable is off.
void TrackLargeAllocations();

} // namespace Common
