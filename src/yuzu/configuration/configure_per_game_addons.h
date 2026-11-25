// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2016 Citra Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <vector>

#include <QList>
#include <QWidget>

#include "common/settings.h"
#include "core/file_sys/vfs/vfs_types.h"

namespace Core {
class System;
}

class QGraphicsScene;
class QStandardItem;
class QStandardItemModel;
class QTreeView;
class QVBoxLayout;

namespace Ui {
class ConfigurePerGameAddons;
}

class ConfigurePerGameAddons : public QWidget {
    Q_OBJECT

public:
    enum PatchData { NUMERIC_VERSION = Qt::UserRole, PATCH_LOCATION };

    explicit ConfigurePerGameAddons(Core::System& system_, QWidget* parent = nullptr);
    ~ConfigurePerGameAddons() override;

    /// Save all button configurations to settings file
    void ApplyConfiguration();

    void LoadFromFile(FileSys::VirtualFile file_);

    void SetTitleId(u64 id);
    void SetCompactLayout();

signals:
    void ConfigurationChanged();

public slots:
    void InstallMods(const QStringList& mods);
    void InstallModPath(const QString& path, const QString& fallbackName = {});

    void InstallModFolder();
    void InstallModZip();

    void AddonDeleteRequested(QList<QModelIndex> selected);

protected:
    void showContextMenu(const QPoint& pos);

private:
    void changeEvent(QEvent* event) override;
    void RetranslateUI();

    void LoadConfiguration();
    void OnItemChanged(QStandardItem* item);

    std::unique_ptr<Ui::ConfigurePerGameAddons> ui;
    FileSys::VirtualFile file;
    u64 title_id;

    QVBoxLayout* layout;
    QTreeView* tree_view;
    QStandardItemModel* item_model;

    std::vector<QList<QStandardItem*>> list_items;
    // Built-in add-ons of this game (n2den's own patches, on/off in Settings::values): shown in
    // the list like the mod folders, saved to their settings instead of the disabled list.
    struct BuiltinAddon {
        QStandardItem* item;
        Settings::SwitchableSetting<bool>* setting;
    };
    std::vector<BuiltinAddon> builtin_items;
    std::vector<QStandardItem*> update_items;

    Core::System& system;
    bool loading_configuration = false;
};
