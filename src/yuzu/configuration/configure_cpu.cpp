// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <memory>
#include <typeinfo>
#include <vector>
#include <QComboBox>
#include "common/common_types.h"
#include "common/settings.h"
#include "common/settings_enums.h"
#include "configuration/shared_widget.h"
#include "core/core.h"
#include "ui_configure_cpu.h"
#include "yuzu/configuration/configuration_shared.h"
#include "yuzu/configuration/configure_cpu.h"

ConfigureCpu::ConfigureCpu(const Core::System& system_,
                           std::shared_ptr<std::vector<ConfigurationShared::Tab*>> group_,
                           const ConfigurationShared::Builder& builder, QWidget* parent)
    : Tab(group_, parent), ui{std::make_unique<Ui::ConfigureCpu>()}, system{system_},
      combobox_translations(builder.ComboboxTranslations()) {
    ui->setupUi(this);

    Setup(builder);

    SetConfiguration();

    connect(accuracy_combobox, qOverload<int>(&QComboBox::currentIndexChanged), this,
            &ConfigureCpu::UpdateGroup);

    connect(backend_combobox, qOverload<int>(&QComboBox::currentIndexChanged), this,
            &ConfigureCpu::UpdateGroup);

#ifdef HAS_NCE
    ui->backend_group->setVisible(true);
#endif
}

ConfigureCpu::~ConfigureCpu() = default;

void ConfigureCpu::SetConfiguration() {}
void ConfigureCpu::Setup(const ConfigurationShared::Builder& builder) {
    auto* accuracy_layout = ui->widget_accuracy->layout();
    auto* backend_layout = ui->widget_backend->layout();
    auto* unsafe_layout = ui->unsafe_widget->layout();
    std::map<u32, QWidget*> unsafe_hold{};

    std::vector<Settings::BasicSetting*> settings;
    const auto push = [&](Settings::Category category) {
        for (const auto setting : Settings::values.linkage.by_category[category]) {
            settings.push_back(setting);
        }
    };

    push(Settings::Category::Cpu);
    push(Settings::Category::CpuUnsafe);

    for (const auto setting : settings) {
        auto* widget = builder.BuildWidget(setting, apply_funcs);

        if (widget == nullptr) {
            continue;
        }
        if (!widget->Valid()) {
            widget->deleteLater();
            continue;
        }

        if (setting->Id() == Settings::values.cpu_accuracy.Id()) {
            // Keep track of cpu_accuracy combobox to display/hide the unsafe settings
            accuracy_layout->addWidget(widget);
            accuracy_combobox = widget->combobox;
            // n2den shows only the JIT disk cache on this page; accuracy keeps its saved value.
            ui->widget_accuracy->setVisible(false);
        } else if (setting->Id() == Settings::values.cpu_backend.Id()) {
            backend_layout->addWidget(widget);
            backend_combobox = widget->combobox;
        } else if (setting->Id() == Settings::values.cpu_jit_cache.Id()) {
            // In General: the backend group is only shown on hosts with NCE.
            ui->general_layout->addWidget(widget);
            jit_cache_combobox = widget->combobox;
        } else if (setting->Id() == Settings::values.cpu_jit_precompile.Id()) {
            // Directly under the JIT disk cache (registered right after it in settings.h).
            ui->general_layout->addWidget(widget);
            jit_precompile_widget = widget;
        } else if (setting->Id() == Settings::values.cpu_ticks.Id()) {
            ui->general_layout->addWidget(widget);
            widget->setVisible(false);  // hidden like accuracy; keeps its saved value
        } else {
            // Presently, all other settings here are unsafe checkboxes
            unsafe_hold.insert({setting->Id(), widget});
        }
    }

    for (const auto& [label, widget] : unsafe_hold) {
        unsafe_layout->addWidget(widget);
    }

    if (jit_cache_combobox != nullptr) {
        connect(jit_cache_combobox, qOverload<int>(&QComboBox::currentIndexChanged), this,
                &ConfigureCpu::UpdateGroup);
    }

    UpdateGroup();
}

void ConfigureCpu::UpdateGroup() {
    // Precompiling stores generated x64 code, which only the "Translations + x64 code" cache keeps; NCE runs
    // without the JIT and its cache.
    if (jit_precompile_widget != nullptr) {
        const bool cache_keeps_code =
            jit_cache_combobox == nullptr ||
            jit_cache_combobox->currentIndex() == static_cast<int>(Settings::JitCache::TranslationsAndCode);
        const bool uses_jit = backend_combobox == nullptr ||
                              backend_combobox->currentIndex() != static_cast<int>(Settings::CpuBackend::Nce);
        jit_precompile_widget->setEnabled(cache_keeps_code && uses_jit);
    }

    const u32 accuracy = accuracy_combobox->currentIndex();
    const u32 backend = backend_combobox->currentIndex();
    // TODO(crueter): see if this works on NCE
    ui->unsafe_group->setVisible(accuracy == (u32)Settings::CpuAccuracy::Unsafe &&
                                 backend == (u32)Settings::CpuBackend::Dynarmic);
}

void ConfigureCpu::ApplyConfiguration() {
    const bool is_powered_on = system.IsPoweredOn();
    for (const auto& apply_func : apply_funcs) {
        apply_func(is_powered_on);
    }
}

void ConfigureCpu::changeEvent(QEvent* event) {
    if (event->type() == QEvent::LanguageChange) {
        RetranslateUI();
    }

    QWidget::changeEvent(event);
}

void ConfigureCpu::RetranslateUI() {
    ui->retranslateUi(this);
}
