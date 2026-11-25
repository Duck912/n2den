// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>

#include <QPersistentModelIndex>
#include <QSplitter>
#include <QTimer>
#include <QWidget>

#include "common/common_types.h"

namespace Settings {
class BasicSetting;
}

class ConfigureInputPerGame;
class ConfigurePerGameAddons;
class QComboBox;
class QLabel;
class QModelIndex;
class QTabWidget;
namespace Core {
class System;
}

class GameDetailsPanel : public QWidget {
    Q_OBJECT
public:
    explicit GameDetailsPanel(Core::System& system, QWidget* parent = nullptr);
    void SetGame(const QModelIndex& index);
    void Reload();
    void SaveChanges();

signals:
    void ConfigurationSaved();

private:
    void LoadGame();
    void ClearTabs();
    QWidget* BuildUpscalingPage();
    void SaveUpscaling();

    Core::System& system;
    QString game_path;
    QString config_name;
    QString game_name;
    QString play_time;
    u64 title_id = 0;
    QLabel* title;
    QLabel* icon;
    QLabel* status;
    QTabWidget* tabs;
    ConfigurePerGameAddons* addons = nullptr;
    ConfigureInputPerGame* input = nullptr;
    bool addons_changed = false;
    bool input_changed = false;
    struct UpscalingRow {
        Settings::BasicSetting* setting;
        QComboBox* box;
    };
    std::vector<UpscalingRow> upscaling_rows;
    bool upscaling_changed = false;
    int selected_tab = 0;
    QTimer deferred_reload;
    QPersistentModelIndex deferred_selection;
    bool selection_pending = false;
    bool reload_pending = false;
};

// Keeps the selected-game pane with the library when emulation replaces the game list.
class GameLibrary : public QSplitter {
public:
    GameLibrary(QWidget* list, QWidget* details, QWidget* parent = nullptr);

protected:
    bool eventFilter(QObject* object, QEvent* event) override;

private:
    QWidget* game_list;
};
