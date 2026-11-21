// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <QPalette>
#include <QString>

namespace QtCommon::Theme {

// Maps saved legacy themes to the corresponding light or dark palette.
QString Normalize(const QString& name);

QPalette Palette(bool dark);

// Applies the palette, embedded stylesheet and icon theme to the application.
bool Apply(const QString& name);

} // namespace QtCommon::Theme
