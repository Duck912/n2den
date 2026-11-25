// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <QLabel>
#include <QVBoxLayout>

#include "common/settings.h"
#include "core/core.h"
#include "core/file_sys/nvn_game_upscaling.h"
#include "yuzu/configuration/configure_per_game_upscaling.h"
#include "yuzu/configuration/shared_widget.h"

namespace {
QLabel* Note(const QString& text, QWidget* parent) {
    auto* label = new QLabel(text, parent);
    label->setWordWrap(true);
    label->setStyleSheet(QStringLiteral("color: palette(mid);"));
    return label;
}
} // namespace

ConfigurePerGameUpscaling::ConfigurePerGameUpscaling(
    Core::System& system_, u64 title_id,
    std::shared_ptr<std::vector<ConfigurationShared::Tab*>> group,
    const ConfigurationShared::Builder& builder, QWidget* parent)
    : Tab(group, parent), system{system_} {
    namespace Game = FileSys::NvnGameUpscaling;
    auto* layout = new QVBoxLayout(this);
    layout->addWidget(Note(tr("Used when the GPU backend is NVN-HLE (Switch 2 games). Each "
                              "setting can follow Settings or keep its own value for this game. "
                              "Changes take effect the next time the game starts."),
                           this));

    const auto add = [&](Settings::BasicSetting* setting) {
        ConfigurationShared::Widget* widget = builder.BuildWidget(setting, apply_funcs);
        if (widget == nullptr) {
            return;
        }
        if (!widget->Valid()) {
            widget->deleteLater();
            return;
        }
        layout->addWidget(widget);
    };
    add(&Settings::values.nvn_upscaler);
    if (Game::DlssQualityApplies(title_id)) {
        add(&Settings::values.nvn_dlss_quality);
        layout->addWidget(Note(tr("DLSS quality is used with the upscaler at NVIDIA DLSS / DLAA."),
                               this));
    }
    if (Game::DisplayResolutionApplies(title_id)) {
        add(&Settings::values.display_resolution);
    }
    if (!Game::DlssQualityApplies(title_id) || !Game::DisplayResolutionApplies(title_id)) {
        layout->addWidget(
            Note(!Game::DlssQualityApplies(title_id) && !Game::DisplayResolutionApplies(title_id)
                     ? tr("DLSS quality and Display resolution are not shown: they do nothing "
                          "for this game.")
                 : !Game::DlssQualityApplies(title_id)
                     ? tr("DLSS quality is not shown: it does nothing for this game.")
                     : tr("Display resolution is not shown: this game does not ask for the "
                          "display size."),
                 this));
    }
    layout->addStretch(1);
}

ConfigurePerGameUpscaling::~ConfigurePerGameUpscaling() = default;

void ConfigurePerGameUpscaling::SetConfiguration() {}

void ConfigurePerGameUpscaling::ApplyConfiguration() {
    const bool powered_on = system.IsPoweredOn();
    for (const auto& func : apply_funcs) {
        func(powered_on);
    }
}
