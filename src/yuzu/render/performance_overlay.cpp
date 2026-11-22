// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "yuzu/render/performance_overlay.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <numeric>

#include <QApplication>
#include <QContextMenuEvent>
#include <QMenu>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>

#include "common/emu_counters.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/hle/kernel/k_process.h"
#include "qt_common/config/shared_translation.h"
#include "qt_common/config/uisettings.h"
#include "qt_common/qt_common.h"
#include "video_core/shader_notify.h"
#include "yuzu/main_window.h"

namespace {

// Look: dark translucent panel, bold coloured labels, white values, small dim units.
const QColor PanelColor{12, 13, 16, 178};
const QColor PanelBorder{255, 255, 255, 22};
const QColor SeparatorColor{255, 255, 255, 28};
const QColor ValueColor{244, 245, 247};
const QColor DimColor{150, 155, 164};
const QColor WarnColor{255, 196, 87};
const QColor ShadowColor{0, 0, 0, 170};

const QColor GpuColor{118, 224, 122};
const QColor VramColor{168, 230, 120};
const QColor CpuColor{96, 180, 255};
const QColor RamColor{255, 138, 216};
const QColor FpsColor{255, 181, 71};
const QColor FrameColor{255, 205, 120};
const QColor EmuColor{255, 150, 95};
const QColor ShaderColor{196, 160, 255};
const QColor GuestColor{242, 160, 200};
const QColor JitColor{120, 214, 230};
const QColor HostThreadColor{186, 196, 210};

/// A thread this busy has no room left: whatever it does is what holds the frame back.
constexpr double ThreadSaturated = 90.0;

const QColor GraphLine{124, 240, 150};
const QColor GraphSpike{255, 107, 90};
const QColor GraphBackground{0, 0, 0, 80};
const QColor GraphGuide{255, 255, 255, 38};
const QColor TickOn{255, 181, 71};
const QColor TickBorder{255, 255, 255, 110};
const QColor TickBorderHover{255, 255, 255, 200};
const QColor GripColor{255, 255, 255, 80};

constexpr int PadX = 10;
constexpr int PadY = 7;
constexpr int RowGap = 1;
constexpr int LabelGap = 12;
constexpr int ColumnGap = 12;
constexpr int PrefixGap = 4;
constexpr int UnitGap = 3;
constexpr int SectionGap = 9;
constexpr int GraphGap = 6;
constexpr int SimpleGraphHeight = 38;
constexpr int AdvancedGraphHeight = 64;
constexpr int SimpleMinContent = 190;
constexpr int AdvancedMinContent = 270;
constexpr int TickBox = 9;
constexpr int TickGap = 4;
constexpr int GripSize = 11;
constexpr float MinScale = 0.75f;
constexpr float MaxScale = 2.5f;

/// About 20 seconds at 60 FPS: enough frames for a meaningful 0.1% low.
constexpr size_t HistoryFrames = 1200;
constexpr int FramePollMs = 33;
constexpr float FrameMs60 = 1000.0f / 60.0f;

constexpr double MiB = 1024.0 * 1024.0;
constexpr double GiB = 1024.0 * MiB;

QString Number(std::optional<double> value, int decimals) {
    if (!value || !std::isfinite(*value)) {
        return QStringLiteral("--");
    }
    return QString::number(*value, 'f', decimals);
}

std::optional<double> ToMiB(std::optional<u64> bytes) {
    if (!bytes) {
        return std::nullopt;
    }
    return static_cast<double>(*bytes) / MiB;
}

std::optional<double> ToGiB(std::optional<u64> bytes) {
    if (!bytes) {
        return std::nullopt;
    }
    return static_cast<double>(*bytes) / GiB;
}

std::optional<double> Finite(double value) {
    if (!std::isfinite(value)) {
        return std::nullopt;
    }
    return value;
}

} // namespace

PerformanceOverlay::PerformanceOverlay(MainWindow* parent)
    : QWidget(parent), m_mainWindow{parent}, m_monitor{std::make_unique<SystemMonitor>()} {
    setWindowFlags(Qt::Tool | Qt::FramelessWindowHint | Qt::WindowDoesNotAcceptFocus |
                   Qt::NoDropShadowWindowHint);
    setAttribute(Qt::WA_TranslucentBackground);
    setAttribute(Qt::WA_ShowWithoutActivating);
    setAttribute(Qt::WA_NoSystemBackground);
    setFocusPolicy(Qt::NoFocus);
    setCursor(Qt::SizeAllCursor);
    setMouseTracking(true); // cursor feedback over the tick and the resize corner

    const QString family = QApplication::font().family();
    m_labelFont = QFont(family);
    m_labelFont.setPixelSize(13);
    m_labelFont.setWeight(QFont::Bold);
    m_valueFont = QFont(family);
    m_valueFont.setPixelSize(13);
    m_valueFont.setWeight(QFont::DemiBold);
    m_smallFont = QFont(family);
    m_smallFont.setPixelSize(10);
    m_smallFont.setWeight(QFont::Medium);
    // The FPS reading: readable at a glance over a busy scene.
    m_bigFont = QFont(family);
    m_bigFont.setPixelSize(28);
    m_bigFont.setWeight(QFont::Bold);
    for (QFont* font : {&m_labelFont, &m_valueFont, &m_smallFont, &m_bigFont}) {
        font->setStyleStrategy(QFont::PreferAntialias);
#if QT_VERSION >= QT_VERSION_CHECK(6, 7, 0)
        // Digits of equal width keep columns from twitching as values change.
        font->setFeature(QFont::Tag("tnum"), 1);
#endif
    }

    m_advanced = UISettings::values.perf_overlay_advanced.GetValue();
    m_showThreads = UISettings::values.perf_overlay_threads.GetValue();
    m_scale = std::clamp(UISettings::values.perf_overlay_scale.GetValue() / 100.0f, MinScale,
                         MaxScale);
    m_offset = QPoint(UISettings::values.perf_overlay_x.GetValue(),
                      UISettings::values.perf_overlay_y.GetValue());

    m_frameTimer.setInterval(FramePollMs);
    m_frameTimer.setTimerType(Qt::PreciseTimer);
    connect(&m_frameTimer, &QTimer::timeout, this, &PerformanceOverlay::pollFrames);

    connect(m_mainWindow, &MainWindow::positionChanged, this, &PerformanceOverlay::resetPosition);
    connect(m_mainWindow, &MainWindow::statsUpdated, this, &PerformanceOverlay::updateStats);

    rebuildRows();
}

PerformanceOverlay::~PerformanceOverlay() = default;

void PerformanceOverlay::SetAdvanced(bool advanced) {
    if (m_advanced == advanced) {
        return;
    }
    m_advanced = advanced;
    UISettings::values.perf_overlay_advanced = advanced;
    // Column widths only ever grow within a mode; start over for the other one.
    m_labelWidth = 0;
    m_prefixWidths.clear();
    m_valueWidths.clear();
    m_unitWidths.clear();
    rebuildRows();
}

void PerformanceOverlay::setScale(float new_scale) {
    // Never larger than the window it sits in.
    float max_scale = MaxScale;
    if (m_baseSize.width() > 0 && m_baseSize.height() > 0) {
        max_scale = std::min({max_scale,
                              static_cast<float>(m_mainWindow->width()) / m_baseSize.width(),
                              static_cast<float>(m_mainWindow->height()) / m_baseSize.height()});
    }
    m_scale = std::clamp(new_scale, MinScale, std::max(MinScale, max_scale));
    applySize();
}

void PerformanceOverlay::applySize() {
    const float s = scale();
    const QSize target(static_cast<int>(std::ceil(m_baseSize.width() * s)),
                       static_cast<int>(std::ceil(m_baseSize.height() * s)));
    if (size() != target) {
        setFixedSize(target);
        resetPosition();
    }
    update();
}

void PerformanceOverlay::resetPosition(const QPoint&) {
    if (!m_mainWindow) {
        return;
    }
    // Anchored to the top-left of the main window's client area and kept inside it.
    const QPoint anchor = m_mainWindow->mapToGlobal(QPoint(0, 0));
    const int max_x = std::max(0, m_mainWindow->width() - width());
    const int max_y = std::max(0, m_mainWindow->height() - height());
    const QPoint offset(std::clamp(m_offset.x(), 0, max_x), std::clamp(m_offset.y(), 0, max_y));
    move(anchor + offset);
}

int PerformanceOverlay::graphCapacity() const {
    // One sample per pixel, newest on the right.
    return std::max(m_graphRect.width(), 1);
}

void PerformanceOverlay::pollFrames() {
    if (!QtCommon::system || !QtCommon::system->IsPoweredOn() ||
        QtCommon::system->IsShuttingDown()) {
        return;
    }
    m_newFrames.clear();
    QtCommon::system->GetPerfStats().ReadFrameTimes(m_frameCursor, m_newFrames);
    if (m_newFrames.empty()) {
        return;
    }
    m_frames.insert(m_frames.end(), m_newFrames.begin(), m_newFrames.end());
    while (m_frames.size() > HistoryFrames) {
        m_frames.pop_front();
    }

    // Vertical scale: the visible peak plus headroom, snapped to half-frames at 60 Hz. It grows
    // at once for a spike and eases back down afterwards.
    const size_t visible = std::min(m_frames.size(), static_cast<size_t>(graphCapacity()));
    const float peak =
        *std::max_element(m_frames.end() - static_cast<std::ptrdiff_t>(visible), m_frames.end());
    constexpr float Step = FrameMs60 / 2.0f;
    const float target = std::ceil(std::max(20.0f, peak * 1.15f) / Step) * Step;
    if (target > m_graphScaleMs) {
        m_graphScaleMs = target;
    } else {
        m_graphScaleMs += (target - m_graphScaleMs) * 0.06f;
    }
    update();
}

PerformanceOverlay::FrameStats PerformanceOverlay::computeFrameStats() const {
    FrameStats stats;
    if (m_frames.empty()) {
        return stats;
    }

    // Current: the mean interval over the last half second, like the FPS counter.
    double recent_sum = 0.0;
    size_t recent_count = 0;
    for (auto it = m_frames.rbegin(); it != m_frames.rend() && recent_sum < 500.0; ++it) {
        recent_sum += *it;
        ++recent_count;
    }
    stats.current_ms = recent_sum / static_cast<double>(recent_count);

    // Min / avg / max over what the graph shows.
    const size_t visible = std::min(m_frames.size(), static_cast<size_t>(graphCapacity()));
    const auto first = m_frames.end() - static_cast<std::ptrdiff_t>(visible);
    const auto [min_it, max_it] = std::minmax_element(first, m_frames.end());
    stats.min_ms = *min_it;
    stats.max_ms = *max_it;
    stats.avg_ms = std::accumulate(first, m_frames.end(), 0.0) / static_cast<double>(visible);

    // Lows over the whole history: the 99th and 99.9th percentile frame time as FPS.
    std::vector<float> sorted(m_frames.begin(), m_frames.end());
    std::sort(sorted.begin(), sorted.end());
    const auto percentile_fps = [&](double fraction) -> std::optional<double> {
        const size_t n = sorted.size();
        const size_t index = std::min(n - 1, static_cast<size_t>(std::ceil(fraction * n)) - 1);
        if (sorted[index] <= 0.0f) {
            return std::nullopt;
        }
        return 1000.0 / sorted[index];
    };
    if (sorted.size() >= 100) {
        stats.low_1_fps = percentile_fps(0.99);
    }
    if (sorted.size() >= 1000) {
        stats.low_01_fps = percentile_fps(0.999);
    }
    return stats;
}

void PerformanceOverlay::updateStats(const Core::PerfStatsResults& results,
                                     const VideoCore::ShaderNotify& shaders) {
    m_results = results;
    m_shadersBuilding = shaders.ShadersPending();
    m_system = m_monitor->Get();
    updateRates();

    m_guestUsed.reset();
    m_guestTotal.reset();
    if (m_advanced && QtCommon::system && QtCommon::system->IsPoweredOn()) {
        if (const auto* process = QtCommon::system->ApplicationProcess()) {
            auto& kernel = QtCommon::system->Kernel();
            m_guestUsed = process->GetUsedUserPhysicalMemorySize(kernel);
            m_guestTotal = process->GetTotalUserPhysicalMemorySize(kernel);
        }
    }

    rebuildRows();
}

void PerformanceOverlay::updateRates() {
    namespace Counters = Common::EmuCounters;
    const auto now = std::chrono::steady_clock::now();
    const u64 jit_blocks = Counters::jit_blocks.Get();
    const u64 jit_invalidations = Counters::jit_invalidations.Get();
    const u64 draws = Counters::draws.Get();
    const u64 timing_work_ns = Counters::timing_work_ns.Get();

    const double seconds = std::chrono::duration<double>(now - m_ratesTime).count();
    const bool first = m_ratesTime == std::chrono::steady_clock::time_point{};
    if (first || seconds <= 0.0 || seconds > 5.0) {
        // Nothing to compare with yet, or the overlay was hidden for a while.
        m_jitBlocksPerSecond.reset();
        m_jitInvalidationsPerSecond.reset();
        m_drawsPerSecond.reset();
        m_timingWorkPercent.reset();
    } else {
        m_timingWorkPercent =
            100.0 * static_cast<double>(timing_work_ns - m_lastTimingWorkNs) / (seconds * 1e9);
        m_jitBlocksPerSecond = static_cast<double>(jit_blocks - m_lastJitBlocks) / seconds;
        m_jitInvalidationsPerSecond =
            static_cast<double>(jit_invalidations - m_lastJitInvalidations) / seconds;
        m_drawsPerSecond = static_cast<double>(draws - m_lastDraws) / seconds;
    }
    m_ratesTime = now;
    m_lastJitBlocks = jit_blocks;
    m_lastJitInvalidations = jit_invalidations;
    m_lastDraws = draws;
    m_lastTimingWorkNs = timing_work_ns;
}

void PerformanceOverlay::addThreadRows() {
    if (m_system.threads.empty()) {
        return;
    }
    // The timing thread waits for its next event in a way the OS counts as busy, so its CPU time
    // reads near 100% whatever it does. Show the time it spent running events instead.
    std::vector<ThreadUsage> threads = m_system.threads;
    if (m_timingWorkPercent) {
        for (ThreadUsage& thread : threads) {
            if (thread.name == "HostTiming") {
                thread.percent = std::min(thread.percent, *m_timingWorkPercent);
            }
        }
    }

    // Guest cores by index; everything else summed into what it works on.
    std::map<int, double> cores;
    enum Group {
        Front, Back, Replay, Hash, Complete, Vulkan, Present, Shaders, Decode,
        Audio, Timing, Ui, Net, Other, Count
    };
    std::array<std::optional<double>, Count> groups;
    const auto add = [&](Group group, double percent) {
        groups[group] = groups[group].value_or(0.0) + percent;
    };
    for (const ThreadUsage& thread : threads) {
        const QString name = QString::fromStdString(thread.name);
        if (name.startsWith(QLatin1String("CPUCore_"))) {
            cores[name.mid(8).toInt()] += thread.percent;
        } else if (name == QLatin1String("CPUThread")) {
            cores[0] += thread.percent;
        } else if (name == QLatin1String("GPU")) {
            add(Front, thread.percent);
        } else if (name == QLatin1String("GPUBack")) {
            add(Back, thread.percent);
        } else if (name == QLatin1String("NvnReplay")) {
            // NVN-HLE: the thread that replays the game's draw calls on Vulkan.
            add(Replay, thread.percent);
        } else if (name == QLatin1String("NvnPrehash") || name == QLatin1String("NvnTexPrehash")) {
            add(Hash, thread.percent);
        } else if (name == QLatin1String("NvnComplete") || name == QLatin1String("NvnBacking") ||
                   name == QLatin1String("NvnScreenshot")) {
            add(Complete, thread.percent);
        } else if (name.startsWith(QLatin1String("NvnPipeline"))) {
            add(Shaders, thread.percent);
        } else if (name == QLatin1String("NvnDecode") ||
                   (name.startsWith(QLatin1String("Nvn")) && name.endsWith(QLatin1String("Init")))) {
            add(Decode, thread.percent);
        } else if (name.startsWith(QLatin1String("bsdsocket"))) {
            add(Net, thread.percent);
        } else if (name == QLatin1String("VulkanWorker")) {
            add(Vulkan, thread.percent);
        } else if (name == QLatin1String("VulkanPresent")) {
            add(Present, thread.percent);
        } else if (name.startsWith(QLatin1String("VkPipeline"))) {
            add(Shaders, thread.percent);
        } else if (name.startsWith(QLatin1String("Texture")) ||
                   name == QLatin1String("ImageTranscode")) {
            add(Decode, thread.percent);
        } else if (name.startsWith(QLatin1String("DSP_")) ||
                   name.startsWith(QLatin1String("Audio"))) {
            add(Audio, thread.percent);
        } else if (name == QLatin1String("HostTiming") || name == QLatin1String("VSyncThread")) {
            add(Timing, thread.percent);
        } else if (name == QLatin1String("UI")) {
            add(Ui, thread.percent);
        } else {
            add(Other, thread.percent);
        }
    }

    const QString percent = QStringLiteral("%");
    const size_t first_row = m_rows.size();
    // Up to three readings to a row; rows after the first of a group carry no label.
    const auto add_rows = [&](const QString& label, const QColor& color,
                              const std::vector<std::pair<QString, double>>& readings) {
        for (size_t i = 0; i < readings.size(); i += 3) {
            Row row{i == 0 ? label : QString{}, color, {}};
            for (size_t j = i; j < std::min(i + 3, readings.size()); ++j) {
                const double value = readings[j].second;
                row.cells.push_back(Cell{readings[j].first, Number(value, 0), percent,
                                         value >= ThreadSaturated ? WarnColor : ValueColor});
            }
            m_rows.push_back(std::move(row));
        }
    };
    const auto collect = [&](std::initializer_list<std::pair<Group, QString>> wanted) {
        std::vector<std::pair<QString, double>> readings;
        for (const auto& [group, caption] : wanted) {
            if (groups[group]) {
                readings.emplace_back(caption, *groups[group]);
            }
        }
        return readings;
    };

    std::vector<std::pair<QString, double>> core_readings;
    for (const auto& [index, value] : cores) {
        core_readings.emplace_back(QString::number(index), value);
    }
    add_rows(tr("CORES"), CpuColor, core_readings);
    add_rows(tr("RENDER"), GpuColor,
             collect({{Front, tr("gpu")},
                      {Back, tr("back")},
                      {Replay, tr("replay")},
                      {Hash, tr("hash")},
                      {Complete, tr("complete")},
                      {Vulkan, tr("vk")},
                      {Present, tr("present")},
                      {Shaders, tr("shaders")},
                      {Decode, tr("decode")}}));
    std::vector<std::pair<QString, double>> host_readings =
        collect({{Audio, tr("audio")}, {Timing, tr("timing")}, {Ui, tr("ui")}, {Net, tr("net")},
                 {Other, tr("other")}});
    if (m_system.cpu_kernel) {
        // Time inside the OS (page faults, system calls) out of all the CPU time the emulator used.
        host_readings.emplace_back(tr("kernel"), *m_system.cpu_kernel);
    }
    add_rows(tr("HOST"), HostThreadColor, host_readings);
    // The kernel share is not a thread and is not a sign of saturation.
    if (m_system.cpu_kernel && !m_rows.empty() && !m_rows.back().cells.empty()) {
        m_rows.back().cells.back().color = ValueColor;
    }

    // The three busiest single threads by name, whatever they are.
    std::vector<const ThreadUsage*> busiest;
    for (const ThreadUsage& thread : threads) {
        if (thread.percent >= 1.0) {
            busiest.push_back(&thread);
        }
    }
    std::sort(busiest.begin(), busiest.end(),
              [](const ThreadUsage* a, const ThreadUsage* b) { return a->percent > b->percent; });
    if (busiest.size() > 3) {
        busiest.resize(3);
    }
    if (!busiest.empty()) {
        QStringList parts;
        for (const ThreadUsage* thread : busiest) {
            parts << QStringLiteral("%1 %2")
                         .arg(thread->name.empty() ? tr("unnamed")
                                                   : QString::fromStdString(thread->name))
                         .arg(thread->percent, 0, 'f', 0);
        }
        Row hot{tr("HOT"), WarnColor, {}};
        hot.note = parts.join(QStringLiteral("  ·  "));
        m_rows.push_back(std::move(hot));
    }

    if (m_rows.size() > first_row) {
        m_rows[first_row].separator_before = true;
    }
}

QString PerformanceOverlay::apiLabel() const {
    switch (Settings::values.renderer_backend.GetValue()) {
    case Settings::RendererBackend::Vulkan:
        return QStringLiteral("VULKAN");
    case Settings::RendererBackend::OpenGL_GLSL:
    case Settings::RendererBackend::OpenGL_GLASM:
    case Settings::RendererBackend::OpenGL_SPIRV:
        return QStringLiteral("OPENGL");
    case Settings::RendererBackend::Null:
    default:
        return QStringLiteral("NULL");
    }
}

QString PerformanceOverlay::configSummary() const {
    QStringList parts;

    if (Settings::values.cpu_backend.GetValue() == Settings::CpuBackend::Nce) {
        parts << tr("NCE");
    } else {
        QString cpu = tr("NX2");
        switch (Settings::values.cpu_jit_cache.GetValue()) {
        case Settings::JitCache::Off:
            break;
        case Settings::JitCache::Translations:
            cpu += QStringLiteral("·C");
            break;
        case Settings::JitCache::TranslationsAndCode:
            cpu += QStringLiteral("·x64");
            break;
        }
        parts << cpu;
    }

    const auto accuracy =
        ConfigurationShared::gpu_accuracy_texts_map.find(Settings::values.gpu_accuracy.GetValue());
    if (accuracy != ConfigurationShared::gpu_accuracy_texts_map.end()) {
        parts << accuracy->second;
    }

    parts << tr("%1x").arg(Settings::values.resolution_info.OutputFactor());
    if (const char* upscaler = Settings::ActiveUpscaler()) {
        parts << QString::fromUtf8(upscaler);
    }
    parts << (Settings::IsDockedMode() ? tr("Docked") : tr("Handheld"));
    if (!Settings::values.use_multi_core.GetValue()) {
        parts << tr("Single-core");
    }
    return parts.join(QStringLiteral("  ·  "));
}

void PerformanceOverlay::rebuildRows() {
    const FrameStats frames = computeFrameStats();
    const SystemStats& sys = m_system;

    const std::optional<double> fps =
        m_results ? Finite(m_results->average_game_fps) : std::nullopt;
    const std::optional<double> speed =
        m_results ? Finite(m_results->emulation_speed * 100.0) : std::nullopt;
    const std::optional<double> emu_busy_ms =
        m_results ? Finite(m_results->frametime * 1000.0) : std::nullopt;

    const auto cell = [](QString prefix, QString value, QString unit,
                         QColor color = ValueColor) {
        return Cell{std::move(prefix), std::move(value), std::move(unit), color};
    };

    m_rows.clear();
    const QString MB = tr("MB");
    const QString GB = tr("GB");
    const QString percent = QStringLiteral("%");

    if (!m_advanced) {
        // RivaTuner's classic layout: GPU, CPU, then the API line with FPS and frame time.
        const auto vram = sys.vram_process ? sys.vram_process : sys.vram_used;
        m_rows.push_back({tr("GPU"), GpuColor,
                          {cell({}, Number(sys.gpu_usage, 0), percent),
                           cell({}, Number(ToMiB(vram), 0), MB)}});
        m_rows.push_back({tr("CPU"), CpuColor,
                          {cell({}, Number(sys.cpu_process, 0), percent),
                           cell({}, Number(ToMiB(sys.ram_process), 0), MB)}});
        Row api{apiLabel(), FpsColor,
                {cell({}, Number(fps, 0), tr("FPS")),
                 cell({}, Number(frames.current_ms, 1), tr("ms"))}};
        api.big = true;
        m_rows.push_back(std::move(api));
        m_header.clear();
        m_footer.clear();
        relayout();
        return;
    }

    // ---- Host -----------------------------------------------------------------------------
    {
        Row gpu{tr("GPU"), GpuColor, {cell({}, Number(sys.gpu_usage, 0), percent)}};
        if (sys.gpu_temp_c) {
            gpu.cells.push_back(cell({}, Number(sys.gpu_temp_c, 0), QStringLiteral("°C")));
        }
        if (sys.gpu_clock_mhz) {
            gpu.cells.push_back(
                cell({}, Number(static_cast<double>(*sys.gpu_clock_mhz), 0), tr("MHz")));
        }
        if (sys.gpu_power_w) {
            gpu.cells.push_back(cell({}, Number(sys.gpu_power_w, 0), tr("W")));
        }
        m_rows.push_back(std::move(gpu));
    }
    {
        Row vram{tr("VRAM"), VramColor, {cell(tr("app"), Number(ToMiB(sys.vram_process), 0), MB)}};
        if (sys.vram_used || sys.vram_total) {
            vram.cells.push_back(cell(tr("sys"), Number(ToGiB(sys.vram_used), 1),
                                      sys.vram_total ? QStringLiteral("/ %1 %2")
                                                           .arg(Number(ToGiB(sys.vram_total), 1), GB)
                                                     : GB));
        }
        m_rows.push_back(std::move(vram));
    }
    m_rows.push_back({tr("CPU"), CpuColor,
                      {cell(tr("app"), Number(sys.cpu_process, 0), percent),
                       cell(tr("sys"), Number(sys.cpu_system, 0), percent)}});
    m_rows.push_back(
        {tr("RAM"), RamColor,
         {cell(tr("app"), Number(ToMiB(sys.ram_process), 0), MB),
          cell(tr("sys"), Number(ToGiB(sys.ram_used), 1),
               QStringLiteral("/ %1 %2").arg(Number(ToGiB(sys.ram_total), 1), GB))}});

    // ---- Emulator -------------------------------------------------------------------------
    const bool slow = speed && *speed < 97.0 && Settings::values.use_speed_limit.GetValue();
    Row api{apiLabel(), FpsColor,
            {cell({}, Number(fps, 0), tr("FPS")), cell({}, Number(frames.current_ms, 1), tr("ms")),
             cell(tr("speed"), Number(speed, 0), percent, slow ? WarnColor : ValueColor)}};
    api.separator_before = true;
    api.big = true;
    m_rows.push_back(std::move(api));
    m_rows.push_back({tr("FRAME"), FrameColor,
                      {cell(tr("min"), Number(frames.min_ms, 1), {}),
                       cell(tr("avg"), Number(frames.avg_ms, 1), {}),
                       cell(tr("max"), Number(frames.max_ms, 1), tr("ms"))}});
    m_rows.push_back({tr("LOWS"), FrameColor,
                      {cell(tr("1%"), Number(frames.low_1_fps, 0), tr("FPS")),
                       cell(tr("0.1%"), Number(frames.low_01_fps, 0), tr("FPS"))}});
    m_rows.push_back({tr("EMU"), EmuColor,
                      {cell(tr("busy"), Number(emu_busy_ms, 1), tr("ms")),
                       cell(tr("shaders"), QString::number(std::max(m_shadersBuilding, 0)), {},
                            m_shadersBuilding > 0 ? WarnColor : ValueColor)}});
    m_rows.push_back({tr("GUEST"), GuestColor,
                      {cell({}, Number(ToGiB(m_guestUsed), 2), GB),
                       cell(tr("of"), Number(ToGiB(m_guestTotal), 2), GB)}});
    {
        // Draws per presented frame; code the JIT compiles and is told to throw away, per second.
        std::optional<double> draws_per_frame;
        if (m_drawsPerSecond && fps && *fps >= 1.0) {
            draws_per_frame = *m_drawsPerSecond / *fps;
        }
        const QString per_second = tr("/s");
        m_rows.push_back({tr("DRAWS"), GpuColor,
                          {cell({}, Number(draws_per_frame, 0), tr("/frame")),
                           cell({}, Number(m_drawsPerSecond, 0), per_second)}});
        m_rows.push_back({tr("JIT"), JitColor,
                          {cell(tr("built"), Number(m_jitBlocksPerSecond, 0), per_second),
                           cell(tr("flush"), Number(m_jitInvalidationsPerSecond, 0), per_second)}});
    }

    if (m_showThreads) {
        addThreadRows();
    }

    m_header = sys.gpu_name.empty() ? QString{} : QString::fromStdString(sys.gpu_name);
    m_footer = configSummary();
    relayout();
}

void PerformanceOverlay::relayout() {
    const QFontMetrics label_fm(m_labelFont);
    const QFontMetrics value_fm(m_valueFont);
    const QFontMetrics small_fm(m_smallFont);
    const QFontMetrics big_fm(m_bigFont);

    // Widths only grow, so the panel does not breathe as numbers change.
    size_t columns = 0;
    for (const Row& row : m_rows) {
        m_labelWidth = std::max(m_labelWidth, label_fm.horizontalAdvance(row.label));
        if (!row.big) {
            columns = std::max(columns, row.cells.size());
        }
    }
    m_prefixWidths.resize(std::max(m_prefixWidths.size(), columns), 0);
    m_valueWidths.resize(std::max(m_valueWidths.size(), columns), 0);
    m_unitWidths.resize(std::max(m_unitWidths.size(), columns), 0);
    for (const Row& row : m_rows) {
        if (row.big) {
            continue; // laid out on its own line (bigRowWidth)
        }
        for (size_t i = 0; i < row.cells.size(); ++i) {
            const Cell& c = row.cells[i];
            if (!c.prefix.isEmpty()) {
                m_prefixWidths[i] =
                    std::max(m_prefixWidths[i], small_fm.horizontalAdvance(c.prefix) + PrefixGap);
            }
            // Reserve room for a typical 3-digit reading so the first samples don't jump.
            m_valueWidths[i] = std::max({m_valueWidths[i], value_fm.horizontalAdvance(c.value),
                                         value_fm.horizontalAdvance(QStringLiteral("000"))});
            if (!c.unit.isEmpty()) {
                m_unitWidths[i] =
                    std::max(m_unitWidths[i], small_fm.horizontalAdvance(c.unit) + UnitGap);
            }
        }
    }

    int content_width = m_labelWidth + LabelGap;
    for (size_t i = 0; i < m_valueWidths.size(); ++i) {
        content_width += m_prefixWidths[i] + m_valueWidths[i] + m_unitWidths[i];
        if (i + 1 < m_valueWidths.size()) {
            content_width += ColumnGap;
        }
    }
    content_width = std::max(content_width, m_advanced ? AdvancedMinContent : SimpleMinContent);
    // A big row: label, the large value (room for three digits), its unit, then the other cells.
    for (const Row& row : m_rows) {
        if (!row.big || row.cells.empty()) {
            continue;
        }
        int width = m_labelWidth + LabelGap +
                    std::max(big_fm.horizontalAdvance(row.cells[0].value),
                             big_fm.horizontalAdvance(QStringLiteral("000"))) +
                    UnitGap + small_fm.horizontalAdvance(row.cells[0].unit);
        for (size_t i = 1; i < row.cells.size(); ++i) {
            const Cell& c = row.cells[i];
            width += ColumnGap + (c.prefix.isEmpty() ? 0 : small_fm.horizontalAdvance(c.prefix) + PrefixGap) +
                     std::max(value_fm.horizontalAdvance(c.value),
                              value_fm.horizontalAdvance(QStringLiteral("00.0"))) +
                     (c.unit.isEmpty() ? 0 : UnitGap + small_fm.horizontalAdvance(c.unit));
        }
        content_width = std::max(content_width, width);
    }

    const int row_height = std::max(label_fm.height(), value_fm.height());
    const int big_row_height = std::max(row_height, big_fm.ascent() + small_fm.descent() + 2);
    int y = PadY;
    // Header line: the GPU name (Advanced) and the "Advanced" tick on the right, in both modes.
    m_headerY = y;
    const int caption_width = small_fm.horizontalAdvance(tr("ADVANCED"));
    const int tick_width = TickBox + TickGap + caption_width;
    const int tick_left = PadX + content_width - tick_width;
    m_tickRect = QRect(tick_left - 3, y - 2, tick_width + 6, small_fm.height() + 4);
    y += small_fm.height() + 4;
    for (Row& row : m_rows) {
        if (row.separator_before) {
            y += SectionGap;
        }
        row.y = y;
        row.height = row.big ? big_row_height : row_height;
        y += row.height + RowGap;
    }
    y += GraphGap - RowGap;
    const int graph_height = m_advanced ? AdvancedGraphHeight : SimpleGraphHeight;
    m_graphRect = QRect(PadX, y, content_width, graph_height);
    y += graph_height;
    if (m_advanced) {
        y += 5;
        m_footerY = y;
        y += small_fm.height();
    } else {
        m_footerY = -1;
    }
    y += PadY;

    m_baseSize = QSize(content_width + 2 * PadX, y);
    m_gripRect = m_advanced ? QRect(m_baseSize.width() - GripSize - 2,
                                    m_baseSize.height() - GripSize - 2, GripSize, GripSize)
                            : QRect();
    applySize();
}

void PerformanceOverlay::paintText(QPainter& painter, int x, int baseline, const QString& text,
                                   const QColor& color) const {
    painter.setPen(ShadowColor);
    painter.drawText(x + 1, baseline + 1, text);
    painter.setPen(color);
    painter.drawText(x, baseline, text);
}

void PerformanceOverlay::paintGraph(QPainter& painter, const QRect& rect) const {
    painter.setPen(Qt::NoPen);
    painter.setBrush(GraphBackground);
    painter.drawRoundedRect(rect, 3.0, 3.0);

    const double scale = std::max(1.0f, m_graphScaleMs);
    const auto y_of = [&](double ms) {
        return rect.bottom() - std::min(ms / scale, 1.0) * (rect.height() - 2);
    };

    // Guides at one and two frames of 60 Hz.
    painter.setFont(m_smallFont);
    const QFontMetrics small_fm(m_smallFont);
    for (const double guide : {static_cast<double>(FrameMs60), FrameMs60 * 2.0}) {
        if (guide >= scale * 0.97) {
            continue;
        }
        const double gy = y_of(guide);
        painter.setPen(QPen(GraphGuide, 1, Qt::DashLine));
        painter.drawLine(QPointF(rect.left() + 1, gy), QPointF(rect.right() - 1, gy));
        if (m_advanced) {
            painter.setPen(DimColor);
            painter.drawText(QPointF(rect.left() + 4, gy - 2), QString::number(guide, 'f', 1));
        }
    }
    if (m_advanced) {
        const QString top = tr("%1 ms").arg(scale, 0, 'f', 1);
        painter.setPen(DimColor);
        painter.drawText(rect.right() - 4 - small_fm.horizontalAdvance(top),
                         rect.top() + small_fm.ascent() + 2, top);
    }

    const size_t capacity = static_cast<size_t>(std::max(rect.width() - 2, 2));
    const size_t count = std::min(m_frames.size(), capacity);
    if (count < 2) {
        return;
    }
    const auto first = m_frames.end() - static_cast<std::ptrdiff_t>(count);

    // Spikes: well above the typical frame of the visible window.
    std::vector<float> window(first, m_frames.end());
    std::nth_element(window.begin(), window.begin() + window.size() / 2, window.end());
    const float median = window[window.size() / 2];
    const float spike = std::max(median * 1.6f, median + 4.0f);

    const double right = rect.right() - 1;
    const auto x_of = [&](size_t i) { return right - static_cast<double>(count - 1 - i); };

    QPainterPath line;
    for (size_t i = 0; i < count; ++i) {
        const QPointF p(x_of(i), y_of(first[i]));
        i == 0 ? line.moveTo(p) : line.lineTo(p);
    }

    QPainterPath fill = line;
    fill.lineTo(x_of(count - 1), rect.bottom());
    fill.lineTo(x_of(0), rect.bottom());
    fill.closeSubpath();
    QLinearGradient gradient(rect.topLeft(), rect.bottomLeft());
    QColor fill_top = GraphLine;
    fill_top.setAlpha(70);
    QColor fill_bottom = GraphLine;
    fill_bottom.setAlpha(8);
    gradient.setColorAt(0.0, fill_top);
    gradient.setColorAt(1.0, fill_bottom);
    painter.setPen(Qt::NoPen);
    painter.setBrush(gradient);
    painter.drawPath(fill);

    painter.setBrush(Qt::NoBrush);
    painter.setPen(QPen(GraphLine, 1.4, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter.drawPath(line);

    painter.setPen(QPen(GraphSpike, 1.6, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    for (size_t i = 1; i < count; ++i) {
        if (first[i] > spike) {
            painter.drawLine(QPointF(x_of(i - 1), y_of(first[i - 1])),
                             QPointF(x_of(i), y_of(first[i])));
            if (i + 1 < count) {
                painter.drawLine(QPointF(x_of(i), y_of(first[i])),
                                 QPointF(x_of(i + 1), y_of(first[i + 1])));
            }
        }
    }
}

void PerformanceOverlay::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setRenderHint(QPainter::TextAntialiasing);
    // Everything is laid out at 1:1 and scaled here, so text and graph grow together.
    const qreal s = scale();
    painter.scale(s, s);

    const QRectF panel(QPointF(0, 0), QSizeF(m_baseSize));
    painter.setPen(QPen(PanelBorder, 1.0 / s));
    painter.setBrush(PanelColor);
    painter.drawRoundedRect(panel.adjusted(0.5 / s, 0.5 / s, -0.5 / s, -0.5 / s), 6.0, 6.0);

    const QFontMetrics label_fm(m_labelFont);
    const QFontMetrics value_fm(m_valueFont);
    const QFontMetrics small_fm(m_smallFont);
    const int content_width = m_baseSize.width() - 2 * PadX;
    const int ascent = std::max(label_fm.ascent(), value_fm.ascent());

    // Header: GPU name on the left (Advanced), the mode tick on the right.
    const int header_baseline = m_headerY + small_fm.ascent();
    painter.setFont(m_smallFont);
    if (m_advanced) {
        const QString header = m_header.isEmpty() ? apiLabel() : m_header;
        paintText(painter, PadX, header_baseline,
                  small_fm.elidedText(header.toUpper(), Qt::ElideRight,
                                      std::max(0, m_tickRect.left() - PadX - 6)),
                  DimColor);
    }
    {
        const int box_x = m_tickRect.left() + 3;
        const QRectF box(box_x + 0.5, header_baseline - TickBox + 0.5, TickBox - 1, TickBox - 1);
        if (m_advanced) {
            painter.setPen(Qt::NoPen);
            painter.setBrush(TickOn);
            painter.drawRoundedRect(box.adjusted(-0.5, -0.5, 0.5, 0.5), 2.0, 2.0);
            QPainterPath check;
            check.moveTo(box.left() + 1.8, box.center().y() + 0.2);
            check.lineTo(box.left() + 3.6, box.bottom() - 1.8);
            check.lineTo(box.right() - 1.5, box.top() + 1.8);
            painter.setPen(QPen(QColor(20, 20, 20), 1.5, Qt::SolidLine, Qt::RoundCap,
                                Qt::RoundJoin));
            painter.setBrush(Qt::NoBrush);
            painter.drawPath(check);
        } else {
            painter.setPen(QPen(m_hoverTick ? TickBorderHover : TickBorder, 1.0));
            painter.setBrush(QColor(0, 0, 0, 60));
            painter.drawRoundedRect(box, 2.0, 2.0);
        }
        paintText(painter, box_x + TickBox + TickGap, header_baseline, tr("ADVANCED"),
                  m_advanced || m_hoverTick ? ValueColor : DimColor);
    }

    for (const Row& row : m_rows) {
        if (row.separator_before) {
            const int sy = row.y - SectionGap / 2 - 1;
            painter.setPen(SeparatorColor);
            painter.drawLine(PadX, sy, PadX + content_width, sy);
        }
        // A big row shares one baseline at the bottom of its line.
        const int baseline = row.big ? row.y + row.height - small_fm.descent() - 1 : row.y + ascent;
        painter.setFont(m_labelFont);
        paintText(painter, PadX, baseline, row.label, row.label_color);

        int x = PadX + m_labelWidth + LabelGap;
        if (row.big && !row.cells.empty()) {
            const QFontMetrics big_fm(m_bigFont);
            const Cell& lead = row.cells[0];
            const int lead_width = std::max(big_fm.horizontalAdvance(lead.value),
                                            big_fm.horizontalAdvance(QStringLiteral("000")));
            painter.setFont(m_bigFont);
            paintText(painter, x + lead_width - big_fm.horizontalAdvance(lead.value), baseline,
                      lead.value, lead.color);
            x += lead_width + UnitGap;
            painter.setFont(m_smallFont);
            paintText(painter, x, baseline, lead.unit, DimColor);
            x += small_fm.horizontalAdvance(lead.unit);
            for (size_t i = 1; i < row.cells.size(); ++i) {
                const Cell& c = row.cells[i];
                x += ColumnGap;
                if (!c.prefix.isEmpty()) {
                    painter.setFont(m_smallFont);
                    paintText(painter, x, baseline, c.prefix, DimColor);
                    x += small_fm.horizontalAdvance(c.prefix) + PrefixGap;
                }
                const int value_width = std::max(value_fm.horizontalAdvance(c.value),
                                                 value_fm.horizontalAdvance(QStringLiteral("00.0")));
                painter.setFont(m_valueFont);
                paintText(painter, x + value_width - value_fm.horizontalAdvance(c.value), baseline,
                          c.value, c.color);
                x += value_width;
                if (!c.unit.isEmpty()) {
                    painter.setFont(m_smallFont);
                    paintText(painter, x + UnitGap, baseline, c.unit, DimColor);
                    x += UnitGap + small_fm.horizontalAdvance(c.unit);
                }
            }
            continue;
        }
        if (!row.note.isEmpty()) {
            painter.setFont(m_smallFont);
            paintText(painter, x, baseline,
                      small_fm.elidedText(row.note, Qt::ElideRight,
                                          std::max(0, PadX + content_width - x)),
                      ValueColor);
            continue;
        }
        for (size_t i = 0; i < m_valueWidths.size(); ++i) {
            if (i < row.cells.size()) {
                const Cell& c = row.cells[i];
                if (!c.prefix.isEmpty()) {
                    painter.setFont(m_smallFont);
                    paintText(painter, x, baseline, c.prefix, DimColor);
                }
                const int value_x = x + m_prefixWidths[i];
                painter.setFont(m_valueFont);
                paintText(painter,
                          value_x + m_valueWidths[i] - value_fm.horizontalAdvance(c.value),
                          baseline, c.value, c.color);
                if (!c.unit.isEmpty()) {
                    painter.setFont(m_smallFont);
                    paintText(painter, value_x + m_valueWidths[i] + UnitGap, baseline, c.unit,
                              DimColor);
                }
            }
            x += m_prefixWidths[i] + m_valueWidths[i] + m_unitWidths[i] + ColumnGap;
        }
    }

    paintGraph(painter, m_graphRect);

    if (m_footerY >= 0 && !m_footer.isEmpty()) {
        painter.setFont(m_smallFont);
        paintText(painter, PadX, m_footerY + small_fm.ascent(),
                  small_fm.elidedText(m_footer, Qt::ElideRight, content_width - GripSize - 4),
                  DimColor);
    }

    // Resize corner: three short diagonal strokes.
    if (!m_gripRect.isNull()) {
        painter.setPen(QPen(GripColor, 1.2, Qt::SolidLine, Qt::RoundCap));
        const QPointF corner(m_gripRect.right() + 0.5, m_gripRect.bottom() + 0.5);
        for (int i = 1; i <= 3; ++i) {
            const qreal d = i * 3.5;
            painter.drawLine(QPointF(corner.x() - d, corner.y()),
                             QPointF(corner.x(), corner.y() - d));
        }
    }
}

void PerformanceOverlay::updateCursor(const QPointF& widget_pos) {
    const QPoint p = toLayout(widget_pos).toPoint();
    const bool over_tick = m_tickRect.contains(p);
    if (over_tick != m_hoverTick) {
        m_hoverTick = over_tick;
        update();
    }
    if (over_tick) {
        setCursor(Qt::PointingHandCursor);
    } else if (!m_gripRect.isNull() && m_gripRect.adjusted(-3, -3, 2, 2).contains(p)) {
        setCursor(Qt::SizeFDiagCursor);
    } else {
        setCursor(Qt::SizeAllCursor);
    }
}

void PerformanceOverlay::mousePressEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton) {
        return;
    }
    const QPoint p = toLayout(event->position()).toPoint();
    if (m_tickRect.contains(p)) {
        SetAdvanced(!m_advanced);
        updateCursor(event->position());
        return;
    }
    if (!m_gripRect.isNull() && m_gripRect.adjusted(-3, -3, 2, 2).contains(p)) {
        m_resizing = true;
        m_resizeStart = event->globalPosition().toPoint();
        m_resizeStartScale = m_scale;
        return;
    }
    m_dragging = true;
    m_dragStart = event->globalPosition().toPoint() - pos();
}

void PerformanceOverlay::mouseMoveEvent(QMouseEvent* event) {
    if (m_resizing && (event->buttons() & Qt::LeftButton)) {
        // The corner follows the mouse: average the scale implied by each axis.
        const QPoint delta = event->globalPosition().toPoint() - m_resizeStart;
        const float start_w = m_baseSize.width() * m_resizeStartScale;
        const float start_h = m_baseSize.height() * m_resizeStartScale;
        const float by_width = (start_w + delta.x()) / m_baseSize.width();
        const float by_height = (start_h + delta.y()) / m_baseSize.height();
        setScale((by_width + by_height) / 2.0f);
        return;
    }
    if (m_dragging && (event->buttons() & Qt::LeftButton)) {
        move(event->globalPosition().toPoint() - m_dragStart);
        m_offset = pos() - m_mainWindow->mapToGlobal(QPoint(0, 0));
        return;
    }
    updateCursor(event->position());
}

void PerformanceOverlay::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton) {
        return;
    }
    if (m_resizing) {
        m_resizing = false;
        UISettings::values.perf_overlay_scale = static_cast<s32>(std::lround(m_scale * 100.0f));
    } else if (m_dragging) {
        m_dragging = false;
        resetPosition(); // snap back inside the window if dragged past an edge
        m_offset = pos() - m_mainWindow->mapToGlobal(QPoint(0, 0));
        UISettings::values.perf_overlay_x = m_offset.x();
        UISettings::values.perf_overlay_y = m_offset.y();
    }
}

void PerformanceOverlay::leaveEvent(QEvent* event) {
    QWidget::leaveEvent(event);
    if (m_hoverTick) {
        m_hoverTick = false;
        update();
    }
}

void PerformanceOverlay::contextMenuEvent(QContextMenuEvent* event) {
    QMenu menu(this);
    QAction* advanced = menu.addAction(tr("Advanced"));
    advanced->setCheckable(true);
    advanced->setChecked(m_advanced);
    QAction* show_threads = menu.addAction(tr("Thread Usage"));
    show_threads->setCheckable(true);
    show_threads->setChecked(m_showThreads);
    show_threads->setEnabled(m_advanced);
    QAction* reset_size = menu.addAction(tr("Reset Size"));
    reset_size->setEnabled(m_advanced && m_scale != 1.0f);
    menu.addSeparator();
    QAction* reset = menu.addAction(tr("Reset Position"));
    QAction* hide_overlay = menu.addAction(tr("Hide Overlay"));

    QAction* chosen = menu.exec(event->globalPos());
    if (chosen == advanced) {
        SetAdvanced(!m_advanced);
    } else if (chosen == show_threads) {
        m_showThreads = !m_showThreads;
        UISettings::values.perf_overlay_threads = m_showThreads;
        rebuildRows();
    } else if (chosen == reset_size) {
        setScale(1.0f);
        UISettings::values.perf_overlay_scale = 100;
    } else if (chosen == reset) {
        m_offset = QPoint(UISettings::values.perf_overlay_x.GetDefault(),
                          UISettings::values.perf_overlay_y.GetDefault());
        UISettings::values.perf_overlay_x = m_offset.x();
        UISettings::values.perf_overlay_y = m_offset.y();
        resetPosition();
    } else if (chosen == hide_overlay) {
        hide();
        emit closed();
    }
}

void PerformanceOverlay::showEvent(QShowEvent* event) {
    QWidget::showEvent(event);
    resetPosition();
    m_monitor->SetActive(true);
    m_frameTimer.start();
}

void PerformanceOverlay::hideEvent(QHideEvent* event) {
    QWidget::hideEvent(event);
    m_frameTimer.stop();
    m_monitor->SetActive(false);
}

void PerformanceOverlay::closeEvent(QCloseEvent*) {
    emit closed();
}
