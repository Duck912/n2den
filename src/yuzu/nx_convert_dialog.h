// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <thread>

#include <QDialog>
#include <QString>

class QCheckBox;
class QComboBox;
class QLineEdit;
class QPlainTextEdit;
class QProgressBar;
class QPushButton;

struct nxformat_cancel;

/// Tools > Convert to NX: packs an unpacked game folder or a NUP file into an NX archive with the
/// nxformat library (https://github.com/fab918/nxformat). The source is only read.
class NxConvertDialog : public QDialog {
    Q_OBJECT

public:
    explicit NxConvertDialog(QWidget* parent = nullptr);
    ~NxConvertDialog() override;

protected:
    void closeEvent(QCloseEvent* event) override;
    void reject() override;

private:
    void BrowseSourceFile();
    void BrowseSourceFolder();
    void BrowseOutput();
    void SuggestOutput();
    void UpdateButtons();
    void Start();
    void RequestCancel();
    void OnLine(bool is_error, const QString& line);
    void OnFinished(int code, const QString& error);

    static void LineCallback(void* user, int is_error, const char* line);

    QLineEdit* source_edit{};
    QLineEdit* output_edit{};
    QComboBox* platform_combo{};
    QCheckBox* keep_source_check{};
    QPushButton* source_file_button{};
    QPushButton* source_folder_button{};
    QPushButton* output_button{};
    QProgressBar* progress_bar{};
    QPlainTextEdit* log_view{};
    QPushButton* convert_button{};
    QPushButton* cancel_button{};
    QPushButton* close_button{};

    std::thread worker;
    nxformat_cancel* cancel_handle{};
    bool running{};
    bool close_when_done{};
    quint64 total_bytes{};
    /// Set for a run whose source is deleted once the new archive has been verified.
    bool delete_source{};
    QString running_source;
};
