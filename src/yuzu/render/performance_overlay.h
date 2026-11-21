// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <deque>
#include <memory>
#include <chrono>
#include <optional>
#include <vector>

#include <QColor>
#include <QFont>
#include <QTimer>
#include <QWidget>

#include "common/common_types.h"
#include "core/perf_stats.h"
#include "yuzu/render/system_monitor.h"

namespace VideoCore {
class ShaderNotify;
}

class MainWindow;

/// On-screen statistics in the style of RivaTuner's overlay: a compact Simple mode (GPU, CPU,
/// FPS and a frame time graph) and an Advanced mode that adds frame time statistics, 1% lows,
/// emulator state, host memory and the CPU use of each emulator thread. Drag to move; the "Advanced" tick in its top-right corner
/// switches mode, and the Advanced panel scales by dragging its bottom-right corner.
class PerformanceOverlay : public QWidget {
    Q_OBJECT

public:
    explicit PerformanceOverlay(MainWindow* parent = nullptr);
    ~PerformanceOverlay() override;

    [[nodiscard]] bool IsAdvanced() const {
        return m_advanced;
    }
    void SetAdvanced(bool advanced);

signals:
    void closed();

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void leaveEvent(QEvent* event) override;
    void contextMenuEvent(QContextMenuEvent* event) override;
    void showEvent(QShowEvent* event) override;
    void hideEvent(QHideEvent* event) override;
    void closeEvent(QCloseEvent* event) override;

private:
    struct Cell {
        QString prefix; ///< Small dim text before the value ("avg", "1%")
        QString value;  ///< Right-aligned within its column
        QString unit;   ///< Small dim text after the value ("ms", "FPS")
        QColor color;
    };
    struct Row {
        QString label;
        QColor label_color;
        std::vector<Cell> cells;
        QString note; ///< Small dim text in place of cells
        bool separator_before = false;
        int y = 0;
    };
    struct FrameStats {
        std::optional<double> current_ms;
        std::optional<double> min_ms;
        std::optional<double> avg_ms;
        std::optional<double> max_ms;
        std::optional<double> low_1_fps;
        std::optional<double> low_01_fps;
    };

    void resetPosition(const QPoint& = {});
    void updateStats(const Core::PerfStatsResults& results, const VideoCore::ShaderNotify& shaders);
    void pollFrames();
    void rebuildRows();
    void addThreadRows();
    void updateRates();
    void relayout();
    void applySize();
    void setScale(float scale);
    void updateCursor(const QPointF& pos);

    /// The panel is laid out at 1:1 and drawn scaled; only the Advanced panel can be resized.
    [[nodiscard]] float scale() const {
        return m_advanced ? m_scale : 1.0f;
    }
    [[nodiscard]] QPointF toLayout(const QPointF& widget_pos) const {
        return widget_pos / scale();
    }

    [[nodiscard]] FrameStats computeFrameStats() const;
    [[nodiscard]] QString apiLabel() const;
    [[nodiscard]] QString configSummary() const;
    [[nodiscard]] int graphCapacity() const;

    void paintText(QPainter& painter, int x, int baseline, const QString& text,
                   const QColor& color) const;
    void paintGraph(QPainter& painter, const QRect& rect) const;

    MainWindow* m_mainWindow = nullptr;
    std::unique_ptr<SystemMonitor> m_monitor;
    QTimer m_frameTimer;

    bool m_advanced = false;
    QPoint m_offset;
    QPoint m_dragStart;
    bool m_dragging = false;

    // Scaling (Advanced only)
    float m_scale = 1.0f;
    bool m_resizing = false;
    QPoint m_resizeStart;
    float m_resizeStartScale = 1.0f;
    bool m_hoverTick = false;

    // Per-frame intervals in ms, oldest first
    u64 m_frameCursor = 0;
    std::vector<float> m_newFrames;
    std::deque<float> m_frames;
    float m_graphScaleMs = 33.3f;

    // Latest readings
    std::optional<Core::PerfStatsResults> m_results;
    int m_shadersBuilding = 0;
    std::optional<u64> m_guestUsed;
    std::optional<u64> m_guestTotal;
    SystemStats m_system;

    // Emulator event rates, from Common::EmuCounters
    bool m_showThreads = true;
    std::chrono::steady_clock::time_point m_ratesTime{};
    u64 m_lastJitBlocks = 0;
    u64 m_lastJitInvalidations = 0;
    u64 m_lastDraws = 0;
    u64 m_lastTimingWorkNs = 0;
    std::optional<double> m_jitBlocksPerSecond;
    std::optional<double> m_jitInvalidationsPerSecond;
    std::optional<double> m_drawsPerSecond;
    std::optional<double> m_timingWorkPercent; ///< Of one core, without its waiting

    // Layout
    QFont m_labelFont;
    QFont m_valueFont;
    QFont m_smallFont;
    std::vector<Row> m_rows;
    QString m_header;
    QString m_footer;
    int m_headerY = -1;
    int m_footerY = -1;
    QRect m_graphRect;
    QRect m_tickRect; ///< The "Advanced" tick box and its caption
    QRect m_gripRect; ///< Resize corner, Advanced only
    QSize m_baseSize; ///< Panel size before scaling
    int m_labelWidth = 0;
    std::vector<int> m_prefixWidths;
    std::vector<int> m_valueWidths;
    std::vector<int> m_unitWidths;
};
