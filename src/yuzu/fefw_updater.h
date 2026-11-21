// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include <string>

#include "common/common_types.h"

class QWidget;

/// Compatibility interface for the retired portable updater.
/// Update downloads are handled manually through the project releases page.
namespace FefwUpdater {

struct Release {
    std::string tag;       ///< "v1.6.4"
    std::string name;      ///< Release title
    std::string html_url;  ///< Release page
    std::string asset_url; ///< The portable zip's download address
    std::string sha256;    ///< Hex, from the asset's digest
    u64 asset_size{};
};

/// No background lookup is performed; returns no release.
std::optional<Release> CheckForUpdate();

/// The retired installer never launches and returns false.
bool OfferUpdate(QWidget* parent, const Release& release);

} // namespace FefwUpdater
