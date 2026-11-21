// SPDX-FileCopyrightText: 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/settings.h"
#include "common/scope_exit.h"
#include <QFile>
#include <QScopedValueRollback>
#include "core/core.h"
#include "frontend_common/config.h"
#include "hid_core/frontend/emulated_controller.h"
#include "hid_core/hid_core.h"
#include "ui_configure_input_per_game.h"
#include "yuzu/configuration/configure_input_per_game.h"
#include "yuzu/configuration/input_profiles.h"

ConfigureInputPerGame::ConfigureInputPerGame(Core::System& system_, QtConfig* config_,
                                             QWidget* parent)
    : QWidget(parent), ui(std::make_unique<Ui::ConfigureInputPerGame>()),
      profiles(std::make_unique<InputProfiles>()), system{system_}, config{config_} {
    ui->setupUi(this);
    const std::array labels = {
        ui->label_player_1, ui->label_player_2, ui->label_player_3, ui->label_player_4,
        ui->label_player_5, ui->label_player_6, ui->label_player_7, ui->label_player_8,
    };
    profile_comboboxes = {
        ui->profile_player_1, ui->profile_player_2, ui->profile_player_3, ui->profile_player_4,
        ui->profile_player_5, ui->profile_player_6, ui->profile_player_7, ui->profile_player_8,
    };

    if (config) {
        Settings::values.players.SetGlobal(false);
    }

    const auto& profile_names = profiles->GetInputProfileNames();
    const auto populate_profiles = [this, &profile_names](size_t player_index) {
        const auto previous_profile = config
                                          ? Settings::values.players.GetValue()[player_index].profile_name
                                          : std::string{};

        auto* const player_combobox = profile_comboboxes[player_index];
        player_combobox->addItem(tr("Use global input configuration"));

        for (size_t index = 0; index < profile_names.size(); ++index) {
            const auto& profile_name = profile_names[index];
            player_combobox->addItem(QString::fromStdString(profile_name),
                                    QString::fromStdString(profile_name));
            if (profile_name == previous_profile) {
                // offset by 1 since the first element is the global config
                player_combobox->setCurrentIndex(static_cast<int>(index + 1));
            }
        }
    };
    for (size_t index = 0; index < profile_comboboxes.size(); ++index) {
        labels[index]->setText(tr("Player %1 profile").arg(index + 1));
        populate_profiles(index);
        connect(profile_comboboxes[index], &QComboBox::activated, this,
                &ConfigureInputPerGame::ConfigurationChanged);
    }

    if (config) {
        LoadConfiguration();
    }
}

void ConfigureInputPerGame::LoadProfilesFromFile(const QString& path) {
    // Read the INI directly: constructing a per-game QtConfig would activate its overrides.
    CSimpleIniA saved;
    saved.SetUnicode();
    QFile file(path);
    if (file.open(QIODevice::ReadOnly)) {
        const auto contents = file.readAll();
        saved.LoadData(contents.constData(), static_cast<size_t>(contents.size()));
    }
    for (size_t i = 0; i < profile_comboboxes.size(); ++i) {
        const auto key = "player_" + std::to_string(i) + "_profile_name";
        const QString name = QString::fromUtf8(saved.GetValue("Controls", key.c_str(), ""))
                                 .remove(QLatin1Char('"'));
        auto* combo = profile_comboboxes[i];
        int selected = name.isEmpty() ? 0 : combo->findData(name);
        if (selected < 0) {
            combo->addItem(name, name);
            selected = combo->count() - 1;
        }
        combo->setCurrentIndex(selected);
    }
}

bool ConfigureInputPerGame::SaveForGame(const std::string& config_name) {
    if (system.IsPoweredOn()) {
        return false;
    }
    const bool configuring_global = Settings::IsConfiguringGlobal();
    const bool input_global = Settings::values.players.UsingGlobal();
    Settings::SetConfiguringGlobal(false);
    // InputSetting has its own global/custom switch, separate from the setting linkage.
    Settings::values.players.SetGlobal(false);
    SCOPE_EXIT {
        Settings::RestoreGlobalState(false);
        Settings::values.players.SetGlobal(input_global);
        Settings::SetConfiguringGlobal(configuring_global);
        system.HIDCore().ReloadInputDevices();
    };
    QtConfig target(config_name, Config::ConfigType::PerGameConfig);
    QScopedValueRollback<QtConfig*> guard(config, &target);
    ApplyConfiguration();
    target.SaveAllValues();
    return true;
}

void ConfigureInputPerGame::SetCompactLayout() {
    setObjectName(QStringLiteral("gameInputProfiles"));
    const std::array<QBoxLayout*, 8> rows = {
        ui->input_profile_layout_1, ui->input_profile_layout_2,
        ui->input_profile_layout_3, ui->input_profile_layout_4,
        ui->input_profile_layout_5, ui->input_profile_layout_6,
        ui->input_profile_layout_7, ui->input_profile_layout_8,
    };
    for (auto* row : rows) {
        row->setDirection(QBoxLayout::TopToBottom);
        row->setSpacing(3);
    }
    ui->groupBox->setTitle({});
}

void ConfigureInputPerGame::ApplyConfiguration() {
    LoadConfiguration();
    SaveConfiguration();
}

void ConfigureInputPerGame::LoadConfiguration() {
    static constexpr size_t HANDHELD_INDEX = 8;

    auto& hid_core = system.HIDCore();
    for (size_t player_index = 0; player_index < profile_comboboxes.size(); ++player_index) {
        Settings::values.players.SetGlobal(false);

        auto* emulated_controller = hid_core.GetEmulatedControllerByIndex(player_index);
        auto* const player_combobox = profile_comboboxes[player_index];

        const auto selection_index = player_combobox->currentIndex();
        if (selection_index == 0) {
            Settings::values.players.GetValue()[player_index].profile_name = "";
            if (player_index == 0) {
                Settings::values.players.GetValue()[HANDHELD_INDEX] = {};
            }
            Settings::values.players.SetGlobal(true);
            emulated_controller->ReloadFromSettings();
            continue;
        }
        const auto profile_name = player_combobox->itemText(selection_index).toStdString();
        if (profile_name.empty()) {
            continue;
        }
        auto& player = Settings::values.players.GetValue()[player_index];
        player.profile_name = profile_name;
        // Read from the profile into the custom player settings
        profiles->LoadProfile(profile_name, player_index);
        // Make sure the controller is connected
        player.connected = true;

        emulated_controller->ReloadFromSettings();

        if (player_index > 0) {
            continue;
        }
        // Handle Handheld cases
        auto& handheld_player = Settings::values.players.GetValue()[HANDHELD_INDEX];
        auto* handheld_controller = hid_core.GetEmulatedController(Core::HID::NpadIdType::Handheld);
        if (player.controller_type == Settings::ControllerType::Handheld) {
            handheld_player = player;
        } else {
            handheld_player = {};
        }
        handheld_controller->ReloadFromSettings();
    }
}

void ConfigureInputPerGame::SaveConfiguration() {
    Settings::values.players.SetGlobal(false);

    // Clear all controls from the config in case the user reverted back to globals
    config->ClearControlPlayerValues();
    for (size_t index = 0; index < Settings::values.players.GetValue().size(); ++index) {
        config->SaveQtControlPlayerValues(index);
    }
}
