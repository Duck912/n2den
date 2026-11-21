// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <QApplication>
#include <QBoxLayout>
#include <QDir>
#include <QEvent>
#include <QFileInfo>
#include <QFormLayout>
#include <QLabel>
#include <QModelIndex>
#include <QScrollArea>
#include <QTabWidget>

#include "common/fs/path_util.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/file_sys/control_metadata.h"
#include "core/file_sys/patch_manager.h"
#include "core/loader/loader.h"
#include "hid_core/hid_core.h"
#include "qt_common/config/qt_config.h"
#include "qt_common/game_list/game_list_p.h"
#include "qt_common/qt_common.h"
#include "yuzu/configuration/configure_input_per_game.h"
#include "yuzu/configuration/configure_per_game_addons.h"
#include "yuzu/game/game_details.h"
#include "yuzu/util/util.h"

namespace {
QScrollArea* ScrollPage(QWidget* content) {
    auto* scroll = new QScrollArea;
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setWidgetResizable(true);
    scroll->setWidget(content);
    return scroll;
}

void Detail(QFormLayout* form, const QString& name, const QString& text) {
    auto* value = new QLabel(text.isEmpty() ? QStringLiteral("—") : text);
    value->setWordWrap(true);
    value->setTextFormat(Qt::PlainText);
    value->setTextInteractionFlags(Qt::TextSelectableByMouse);
    form->addRow(name, value);
}
} // namespace

GameDetailsPanel::GameDetailsPanel(Core::System& system_, QWidget* parent)
    : QWidget(parent), system(system_) {
    setObjectName(QStringLiteral("gameDetailsPanel"));
    setMinimumWidth(320);
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(10, 8, 8, 8);
    layout->setSpacing(10);
    auto* heading = new QHBoxLayout;
    icon = new QLabel;
    icon->setFixedSize(64, 64);
    icon->hide();
    title = new QLabel(tr("Game details"));
    title->setObjectName(QStringLiteral("gameDetailsTitle"));
    title->setTextFormat(Qt::PlainText);
    title->setWordWrap(true);
    heading->addWidget(icon);
    heading->addWidget(title, 1);
    layout->addLayout(heading);
    tabs = new QTabWidget;
    tabs->setObjectName(QStringLiteral("gameDetailsTabs"));
    layout->addWidget(tabs, 1);
    status = new QLabel(tr("Select a game to see its details and configuration."));
    status->setObjectName(QStringLiteral("gameDetailsStatus"));
    status->setWordWrap(true);
    layout->addWidget(status);
    tabs->hide();
    deferred_reload.setInterval(100);
    connect(&deferred_reload, &QTimer::timeout, this, [this] {
        if (QApplication::activeModalWidget()) {
            return;
        }
        deferred_reload.stop();
        if (selection_pending) {
            const QModelIndex selection = deferred_selection;
            selection_pending = false;
            SetGame(selection);
        }
        if (reload_pending) {
            reload_pending = false;
            Reload();
        }
    });
}

void GameDetailsPanel::ClearTabs() {
    while (tabs->count()) {
        QWidget* page = tabs->widget(0);
        tabs->removeTab(0);
        delete page;
    }
    addons = nullptr;
    input = nullptr;
}

void GameDetailsPanel::SetGame(const QModelIndex& index) {
    // A filesystem refresh can arrive inside a mod-import dialog's nested event loop.
    // Keep its editor alive until the dialog and its calling slot have returned.
    if (QApplication::activeModalWidget()) {
        deferred_selection = index;
        selection_pending = true;
        deferred_reload.start();
        return;
    }
    selection_pending = false;
    const QModelIndex game = index.isValid() ? index.siblingAtColumn(0) : QModelIndex{};
    const bool is_game =
        game.data(GameListItem::TypeRole).value<GameListItemType>() == GameListItemType::Game;
    const QString path = is_game ? game.data(GameListItemPath::FullPathRole).toString() : QString{};
    const u64 selected_id = is_game ? game.data(GameListItemPath::ProgramIdRole).toULongLong() : 0;
    if (path == game_path && selected_id == title_id) {
        return;
    }
    SaveChanges();
    game_path = path;
    title_id = selected_id;
    game_name = game.data(GameListItemPath::TitleRole).toString();
    if (game_name.trimmed().isEmpty()) {
        game_name = QFileInfo(path).completeBaseName();
    }
    const QPixmap picture = game.data(Qt::DecorationRole).value<QPixmap>();
    icon->setPixmap(picture.isNull()
                        ? QPixmap{}
                        : picture.scaled(64, 64, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    icon->setVisible(!picture.isNull());
    play_time = game.data(Qt::ToolTipRole).toString().section(QLatin1Char('\n'), 0, 0);
    if (play_time.contains(QLatin1Char(':'))) {
        play_time = play_time.section(QLatin1Char(':'), 1).trimmed();
    }
    LoadGame();
}

void GameDetailsPanel::Reload() {
    if (QApplication::activeModalWidget()) {
        reload_pending = true;
        deferred_reload.start();
        return;
    }
    reload_pending = false;
    SaveChanges();
    LoadGame();
}

void GameDetailsPanel::LoadGame() {
    if (tabs->currentIndex() >= 0) {
        selected_tab = tabs->currentIndex();
    }
    ClearTabs();
    addons_changed = false;
    input_changed = false;
    title->setText(game_path.isEmpty() ? tr("Game details") : game_name);
    tabs->setVisible(!game_path.isEmpty());
    if (game_path.isEmpty()) {
        icon->hide();
        status->setText(tr("Select a game to see its details and configuration."));
        return;
    }
    const auto file = Core::GetGameFileFromPath(QtCommon::vfs, game_path.toStdString());
    const auto loader = file ? Loader::GetLoader(system, file) : nullptr;
    if (!loader) {
        status->setText(tr("This game's file could not be opened."));
        return;
    }
    config_name = title_id == 0
                      ? QFileInfo(game_path).fileName()
                      : QStringLiteral("%1").arg(title_id, 16, 16, QLatin1Char('0')).toUpper();

    auto* details = new QWidget;
    auto* form = new QFormLayout(details);
    form->setRowWrapPolicy(QFormLayout::WrapLongRows);
    form->setVerticalSpacing(12);
    const FileSys::PatchManager patches{title_id, system.GetFileSystemController(),
                                        system.GetContentProvider()};
    const auto control = patches.GetControlMetadata();
    QString developer;
    QString version = QStringLiteral("1.0.0");
    if (control.first) {
        const QString name = QString::fromStdString(control.first->GetApplicationName());
        if (!name.isEmpty()) {
            title->setText(name);
        }
        developer = QString::fromStdString(control.first->GetDeveloperName());
        version = QString::fromStdString(control.first->GetVersionString());
    } else {
        FileSys::NACP nacp;
        if (loader->ReadControlData(nacp) == Loader::ResultStatus::Success) {
            developer = QString::fromStdString(nacp.GetDeveloperName());
        }
    }
    Detail(form, tr("Developer"), developer);
    Detail(form, tr("Version"), version);
    Detail(form, tr("Play time"), play_time);
    Detail(form, tr("Title ID"),
           QStringLiteral("%1").arg(title_id, 16, 16, QLatin1Char('0')).toUpper());
    Detail(form, tr("Format"),
           QString::fromStdString(Loader::GetFileTypeString(loader->GetFileType())));
    Detail(form, tr("Size"), QtCommon::ReadableByteSize(file->GetSize()));
    Detail(form, tr("File"), QFileInfo(game_path).fileName());
    Detail(form, tr("Location"), game_path);
    tabs->addTab(ScrollPage(details), tr("Details"));

    addons = new ConfigurePerGameAddons(system);
    addons->SetCompactLayout();
    addons->SetTitleId(title_id);
    addons->LoadFromFile(file);
    tabs->addTab(addons, tr("Add-Ons"));

    // No per-game settings are activated while the library is being browsed.
    input = new ConfigureInputPerGame(system, nullptr);
    input->SetCompactLayout();
    const QString config_dir =
        QString::fromStdString(Common::FS::GetEdenPathString(Common::FS::EdenPath::ConfigDir));
    input->LoadProfilesFromFile(
        QDir(config_dir).filePath(QStringLiteral("custom/%1.ini").arg(config_name)));
    tabs->addTab(ScrollPage(input), tr("Input Profiles"));
    if (selected_tab >= 0 && selected_tab < tabs->count()) {
        tabs->setCurrentIndex(selected_tab);
    }
    status->setText(tr("Changes are saved automatically for this game."));
    connect(addons, &ConfigurePerGameAddons::ConfigurationChanged, this, [this] {
        addons_changed = true;
        SaveChanges();
    });
    connect(input, &ConfigureInputPerGame::ConfigurationChanged, this, [this] {
        input_changed = true;
        SaveChanges();
    });
}

void GameDetailsPanel::SaveChanges() {
    if ((!addons_changed && !input_changed) || system.IsPoweredOn() ||
        QApplication::activeModalWidget()) {
        return;
    }
    if (addons_changed && addons) {
        addons->ApplyConfiguration();
    }
    if (input_changed && input) {
        if (!input->SaveForGame(config_name.toStdString())) {
            return;
        }
    }
    addons_changed = false;
    input_changed = false;
    emit ConfigurationSaved();
}

GameLibrary::GameLibrary(QWidget* list, QWidget* details, QWidget* parent)
    : QSplitter(Qt::Horizontal, parent), game_list(list) {
    setObjectName(QStringLiteral("gameLibrary"));
    setChildrenCollapsible(false);
    setHandleWidth(6);
    addWidget(list);
    addWidget(details);
    setStretchFactor(0, 1);
    setStretchFactor(1, 0);
    setSizes({860, 340});
    list->installEventFilter(this);
    setVisible(list->isVisibleTo(parent));
}

bool GameLibrary::eventFilter(QObject* object, QEvent* event) {
    if (object == game_list) {
        if (event->type() == QEvent::ShowToParent) {
            show();
        } else if (event->type() == QEvent::HideToParent) {
            hide();
        }
    }
    return QSplitter::eventFilter(object, event);
}
