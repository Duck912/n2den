// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <memory>
#include <string>
#include "common/common_types.h"
namespace Core { class System; }
namespace Core::Nx2Debug {
class N2DenService {
public:
    N2DenService(Core::System& system, u64 program_id, std::string title);
    ~N2DenService();
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
std::unique_ptr<N2DenService> MaybeStartN2DenService(Core::System& system, u64 program_id,
                                                  const std::string& title);
} // namespace Core::Nx2Debug
