// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <string>
#include <vector>

#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QVBoxLayout>

#include <nxformat.h>

#include "yuzu/nx_convert_dialog.h"

namespace {

constexpr int ProgressSteps = 1000;

QString ResultText(int code) {
    switch (code) {
    case NXFORMAT_OK:
        return QObject::tr("Done.");
    case NXFORMAT_CANCELLED:
        return QObject::tr("Cancelled. No partial file was left.");
    default:
        return QObject::tr("Failed.");
    }
}

} // namespace

NxConvertDialog::NxConvertDialog(QWidget* parent) : QDialog{parent} {
    setWindowTitle(tr("Convert to NX"));
    setMinimumWidth(640);

    source_edit = new QLineEdit;
    source_edit->setPlaceholderText(tr("An unpacked game folder or a .nup file"));
    source_file_button = new QPushButton(tr("NUP File..."));
    source_folder_button = new QPushButton(tr("Folder..."));
    auto* source_row = new QHBoxLayout;
    source_row->addWidget(source_edit, 1);
    source_row->addWidget(source_file_button);
    source_row->addWidget(source_folder_button);

    output_edit = new QLineEdit;
    output_edit->setPlaceholderText(tr("The .nx file to create"));
    output_button = new QPushButton(tr("Browse..."));
    auto* output_row = new QHBoxLayout;
    output_row->addWidget(output_edit, 1);
    output_row->addWidget(output_button);

    // Neither a NUP file nor a loose folder says which console it is for, so the packer asks.
    platform_combo = new QComboBox;
    platform_combo->addItem(tr("Switch 2"), QStringLiteral("switch2"));
    platform_combo->addItem(tr("Switch"), QStringLiteral("switch1"));

    auto* form = new QFormLayout;
    form->addRow(tr("Source:"), source_row);
    form->addRow(tr("Output:"), output_row);
    form->addRow(tr("Platform:"), platform_combo);

    keep_source_check = new QCheckBox(tr("Keep the original files after converting"));
    keep_source_check->setChecked(true);
    keep_source_check->setToolTip(
        tr("When off, the source is deleted after the new NX file has been written and verified."));
    form->addRow(QString{}, keep_source_check);

    progress_bar = new QProgressBar;
    progress_bar->setRange(0, ProgressSteps);
    progress_bar->setValue(0);

    log_view = new QPlainTextEdit;
    log_view->setReadOnly(true);
    log_view->setMaximumBlockCount(2000);
    log_view->setMinimumHeight(160);

    convert_button = new QPushButton(tr("Convert"));
    convert_button->setDefault(true);
    cancel_button = new QPushButton(tr("Cancel"));
    close_button = new QPushButton(tr("Close"));
    auto* buttons = new QHBoxLayout;
    buttons->addWidget(new QLabel(tr("nxformat %1").arg(QString::fromUtf8(nxformat_version()))));
    buttons->addStretch(1);
    buttons->addWidget(convert_button);
    buttons->addWidget(cancel_button);
    buttons->addWidget(close_button);

    auto* layout = new QVBoxLayout(this);
    layout->addLayout(form);
    layout->addWidget(progress_bar);
    layout->addWidget(log_view, 1);
    layout->addLayout(buttons);

    connect(source_file_button, &QPushButton::clicked, this, &NxConvertDialog::BrowseSourceFile);
    connect(source_folder_button, &QPushButton::clicked, this,
            &NxConvertDialog::BrowseSourceFolder);
    connect(output_button, &QPushButton::clicked, this, &NxConvertDialog::BrowseOutput);
    connect(source_edit, &QLineEdit::textChanged, this, &NxConvertDialog::UpdateButtons);
    connect(output_edit, &QLineEdit::textChanged, this, &NxConvertDialog::UpdateButtons);
    connect(convert_button, &QPushButton::clicked, this, &NxConvertDialog::Start);
    connect(cancel_button, &QPushButton::clicked, this, &NxConvertDialog::RequestCancel);
    connect(close_button, &QPushButton::clicked, this, &NxConvertDialog::close);

    UpdateButtons();
}

NxConvertDialog::~NxConvertDialog() {
    if (cancel_handle != nullptr) {
        nxformat_cancel_set(cancel_handle);
    }
    if (worker.joinable()) {
        worker.join();
    }
    if (cancel_handle != nullptr) {
        nxformat_cancel_free(cancel_handle);
    }
}

void NxConvertDialog::BrowseSourceFile() {
    const QString file = QFileDialog::getOpenFileName(this, tr("Select NUP File"), {},
                                                      tr("NUP files (*.nup);;All files (*)"));
    if (!file.isEmpty()) {
        source_edit->setText(QDir::toNativeSeparators(file));
        SuggestOutput();
    }
}

void NxConvertDialog::BrowseSourceFolder() {
    const QString folder = QFileDialog::getExistingDirectory(this, tr("Select Unpacked Game"));
    if (!folder.isEmpty()) {
        source_edit->setText(QDir::toNativeSeparators(folder));
        SuggestOutput();
    }
}

void NxConvertDialog::BrowseOutput() {
    const QString file = QFileDialog::getSaveFileName(this, tr("Save NX File"),
                                                      output_edit->text(),
                                                      tr("NX archives (*.nx)"));
    if (!file.isEmpty()) {
        output_edit->setText(QDir::toNativeSeparators(file));
    }
}

void NxConvertDialog::SuggestOutput() {
    if (!output_edit->text().isEmpty()) {
        return;
    }
    const QFileInfo source{source_edit->text()};
    const QString name = source.isDir() ? source.fileName() : source.completeBaseName();
    if (name.isEmpty()) {
        return;
    }
    output_edit->setText(
        QDir::toNativeSeparators(source.absoluteDir().filePath(name + QStringLiteral(".nx"))));
}

void NxConvertDialog::UpdateButtons() {
    const bool ready = !source_edit->text().trimmed().isEmpty() &&
                       !output_edit->text().trimmed().isEmpty();
    convert_button->setEnabled(!running && ready);
    cancel_button->setEnabled(running);
    close_button->setEnabled(!running);
    for (QWidget* widget : std::array<QWidget*, 7>{source_edit, output_edit, platform_combo,
                                                   source_file_button, source_folder_button,
                                                   output_button, keep_source_check}) {
        widget->setEnabled(!running);
    }
}

void NxConvertDialog::Start() {
    const QString source = QDir::fromNativeSeparators(source_edit->text().trimmed());
    QString output = QDir::fromNativeSeparators(output_edit->text().trimmed());
    const QFileInfo source_info{source};
    if (!source_info.exists()) {
        QMessageBox::warning(this, windowTitle(), tr("The source does not exist."));
        return;
    }
    if (!output.endsWith(QStringLiteral(".nx"), Qt::CaseInsensitive)) {
        output += QStringLiteral(".nx");
        output_edit->setText(QDir::toNativeSeparators(output));
    }
    if (QFileInfo::exists(output)) {
        QMessageBox::warning(this, windowTitle(),
                             tr("The output file already exists. Choose another name."));
        return;
    }

    delete_source = !keep_source_check->isChecked();
    if (delete_source) {
        const auto answer = QMessageBox::warning(
            this, windowTitle(),
            tr("The original will be deleted for good once the NX file has been written and "
               "verified:\n\n%1\n\nContinue?")
                .arg(QDir::toNativeSeparators(source)),
            QMessageBox::Yes | QMessageBox::Cancel, QMessageBox::Cancel);
        if (answer != QMessageBox::Yes) {
            return;
        }
    }
    running_source = source;

    // The packer takes the nxconvert command line: pack-nup for a NUP file, pack-tree for a folder.
    std::vector<std::string> args;
    args.emplace_back(source_info.isDir() ? "pack-tree" : "pack-nup");
    args.push_back(source.toStdString());
    args.push_back(output.toStdString());
    args.emplace_back("--platform");
    args.push_back(platform_combo->currentData().toString().toStdString());
    args.emplace_back("--frontend");
    // No review files beside the output; the log below shows what the packer decided.
    args.emplace_back("--no-trail");

    if (worker.joinable()) {
        worker.join();
    }
    if (cancel_handle != nullptr) {
        nxformat_cancel_free(cancel_handle);
    }
    cancel_handle = nxformat_cancel_new();
    total_bytes = 0;
    progress_bar->setValue(0);
    log_view->clear();
    running = true;
    UpdateButtons();

    worker = std::thread([this, args = std::move(args), cancel = cancel_handle,
                          verify = delete_source, output = output.toStdString()] {
        std::vector<const char*> argv;
        argv.reserve(args.size());
        for (const auto& arg : args) {
            argv.push_back(arg.c_str());
        }
        std::array<char, 1024> error{};
        int code = nxformat_run(argv.data(), argv.size(), &NxConvertDialog::LineCallback, this,
                                cancel, error.data(), error.size());
        // Before the source may be deleted, read the whole new archive back and check it.
        if (code == NXFORMAT_OK && verify) {
            LineCallback(this, 0, "Verifying the new archive before deleting the original...");
            const std::array<const char*, 2> verify_argv{"verify", output.c_str()};
            code = nxformat_run(verify_argv.data(), verify_argv.size(),
                                &NxConvertDialog::LineCallback, this, cancel, error.data(),
                                error.size());
        }
        const QString message = QString::fromUtf8(error.data());
        QMetaObject::invokeMethod(
            this, [this, code, message] { OnFinished(code, message); }, Qt::QueuedConnection);
    });
}

void NxConvertDialog::RequestCancel() {
    if (running && cancel_handle != nullptr) {
        nxformat_cancel_set(cancel_handle);
        cancel_button->setEnabled(false);
        log_view->appendPlainText(tr("Cancelling..."));
    }
}

void NxConvertDialog::LineCallback(void* user, int is_error, const char* line) {
    // Called on a packer thread; the text is only valid during the call.
    auto* dialog = static_cast<NxConvertDialog*>(user);
    const QString text = QString::fromUtf8(line);
    QMetaObject::invokeMethod(
        dialog, [dialog, is_error, text] { dialog->OnLine(is_error != 0, text); },
        Qt::QueuedConnection);
}

void NxConvertDialog::OnLine(bool is_error, const QString& line) {
    // Progress protocol: each "@TOTAL\tN" adds N bytes to the expected total, "@PROGRESS\tN" is the
    // bytes done since the start.
    if (line.startsWith(QLatin1Char('@'))) {
        const QStringList fields = line.split(QLatin1Char('\t'));
        if (fields.size() >= 2 && fields[0] == QStringLiteral("@TOTAL")) {
            total_bytes += fields[1].toULongLong();
        } else if (fields.size() >= 2 && fields[0] == QStringLiteral("@PROGRESS") &&
                   total_bytes > 0) {
            const quint64 done = qMin(fields[1].toULongLong(), total_bytes);
            progress_bar->setValue(static_cast<int>(done * ProgressSteps / total_bytes));
        }
        return;
    }
    // Diagnostic lines already start with "ERROR:" or "WARNING:".
    Q_UNUSED(is_error)
    log_view->appendPlainText(line);
}

void NxConvertDialog::OnFinished(int code, const QString& error) {
    if (worker.joinable()) {
        worker.join();
    }
    running = false;
    if (code == NXFORMAT_OK) {
        progress_bar->setValue(ProgressSteps);
    }
    // The error text usually repeats the last diagnostic line.
    if (!error.isEmpty() && !log_view->toPlainText().contains(error)) {
        log_view->appendPlainText(error);
    }
    log_view->appendPlainText(ResultText(code));
    if (code == NXFORMAT_OK && delete_source && !running_source.isEmpty()) {
        const QFileInfo source{running_source};
        const bool removed = source.isDir() ? QDir{running_source}.removeRecursively()
                                            : QFile::remove(running_source);
        log_view->appendPlainText(
            removed ? tr("Deleted the original: %1").arg(QDir::toNativeSeparators(running_source))
                    : tr("Could not delete the original: %1")
                          .arg(QDir::toNativeSeparators(running_source)));
    } else if (delete_source) {
        log_view->appendPlainText(tr("The original was kept."));
    }
    delete_source = false;
    running_source.clear();
    UpdateButtons();
    if (close_when_done) {
        QDialog::reject();
    }
}

void NxConvertDialog::closeEvent(QCloseEvent* event) {
    if (!running) {
        event->accept();
        return;
    }
    // Stop the packer first; the dialog closes when it reports back.
    close_when_done = true;
    RequestCancel();
    event->ignore();
}

void NxConvertDialog::reject() {
    if (running) {
        close_when_done = true;
        RequestCancel();
        return;
    }
    QDialog::reject();
}
