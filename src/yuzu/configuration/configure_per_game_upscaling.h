// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "common/common_types.h"
#include "yuzu/configuration/configuration_shared.h"

namespace Core {
class System;
}

namespace ConfigurationShared {
class Builder;
}

// The per-game dialog's Upscaling page: the NVN-HLE upscaling settings that work for this game
// (file_sys/nvn_game_upscaling.h). Upscaler for every game; DLSS quality and Display resolution
// only for the games they do something for. Each row can follow the global setting or keep its own
// value for this game.
class ConfigurePerGameUpscaling : public ConfigurationShared::Tab {
public:
    explicit ConfigurePerGameUpscaling(Core::System& system, u64 title_id,
                                       std::shared_ptr<std::vector<ConfigurationShared::Tab*>> group,
                                       const ConfigurationShared::Builder& builder,
                                       QWidget* parent = nullptr);
    ~ConfigurePerGameUpscaling() override;

    void ApplyConfiguration() override;
    void SetConfiguration() override;

private:
    Core::System& system;
    std::vector<std::function<void(bool)>> apply_funcs;
};
