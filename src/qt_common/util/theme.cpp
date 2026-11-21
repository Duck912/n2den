// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <QApplication>
#include <QFile>
#include <QIcon>
#include <QStyle>
#include <QStyleFactory>

#include "qt_common/util/theme.h"

namespace QtCommon::Theme {

QString Normalize(const QString& name) {
    if (name == QStringLiteral("winter_fade") || name == QStringLiteral("default") ||
        name == QStringLiteral("colorful")) {
        return QStringLiteral("winter_fade");
    }
    return QStringLiteral("dark_frost");
}

QPalette Palette(bool dark) {
    QPalette palette;
    const auto gray = [](int value) { return QColor(value, value, value); };
    palette.setColor(QPalette::Window, gray(dark ? 32 : 242));
    palette.setColor(QPalette::WindowText, gray(dark ? 238 : 40));
    palette.setColor(QPalette::Base, gray(dark ? 24 : 255));
    palette.setColor(QPalette::AlternateBase, gray(dark ? 40 : 248));
    palette.setColor(QPalette::Text, gray(dark ? 238 : 40));
    palette.setColor(QPalette::Button, gray(dark ? 48 : 232));
    palette.setColor(QPalette::ButtonText, gray(dark ? 238 : 40));
    palette.setColor(QPalette::Light, gray(dark ? 92 : 255));
    palette.setColor(QPalette::Midlight, gray(dark ? 64 : 224));
    palette.setColor(QPalette::Mid, gray(dark ? 72 : 200));
    palette.setColor(QPalette::Dark, gray(dark ? 16 : 160));
    palette.setColor(QPalette::Shadow, gray(dark ? 8 : 112));
    palette.setColor(QPalette::Highlight, gray(dark ? 204 : 212));
#if QT_VERSION >= QT_VERSION_CHECK(6, 6, 0)
    palette.setColor(QPalette::Accent, gray(dark ? 204 : 212));
#endif
    palette.setColor(QPalette::HighlightedText, gray(24));
    palette.setColor(QPalette::BrightText, gray(dark ? 255 : 0));
    palette.setColor(QPalette::Link, gray(dark ? 224 : 56));
    palette.setColor(QPalette::LinkVisited, gray(dark ? 192 : 88));
    palette.setColor(QPalette::ToolTipBase, gray(dark ? 48 : 255));
    palette.setColor(QPalette::ToolTipText, gray(dark ? 238 : 40));
    palette.setColor(QPalette::PlaceholderText, gray(dark ? 160 : 112));
    for (const auto role : {QPalette::WindowText, QPalette::Text, QPalette::ButtonText}) {
        palette.setColor(QPalette::Disabled, role, gray(dark ? 120 : 144));
    }
    palette.setColor(QPalette::Disabled, QPalette::Highlight, gray(dark ? 64 : 224));
    palette.setColor(QPalette::Disabled, QPalette::HighlightedText, gray(dark ? 160 : 112));
    return palette;
}

bool Apply(const QString& name) {
    const QString theme = Normalize(name);
    QFile stylesheet(QStringLiteral(":/%1/style.qss").arg(theme));
    if (!stylesheet.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return false;
    }
    const QString contents =
        QString::fromUtf8(stylesheet.readAll()).replace(QStringLiteral("@theme@"), theme);
    qApp->setStyleSheet({});
    // Fusion consistently uses the selected palette, including native-drawn indicators and arrows.
    if (qApp->style()->objectName() != QStringLiteral("fusion")) {
        qApp->setStyle(QStyleFactory::create(QStringLiteral("Fusion")));
    }
    qApp->setPalette(Palette(theme == QStringLiteral("dark_frost")));
    QIcon::setThemeSearchPaths({QStringLiteral(":/icons")});
    QIcon::setThemeName(theme);
    qApp->setStyleSheet(contents);
    return true;
}

} // namespace QtCommon::Theme
