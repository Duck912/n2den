// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "yuzu/fefw_updater.h"

namespace FefwUpdater {

std::optional<Release> CheckForUpdate() {
    // Updates are downloaded manually from the project's releases page.
    return std::nullopt;
}

bool OfferUpdate(QWidget*, const Release&) {
    // The legacy portable installer is intentionally disabled for this fork.
    return false;
}

} // namespace FefwUpdater
