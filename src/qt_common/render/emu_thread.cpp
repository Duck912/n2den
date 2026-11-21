// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <cstdlib>
#include <string_view>

#include <qdebug.h>
#include "common/settings.h"
#include "core/core.h"
#include "core/cpu_manager.h"
#include "emu_thread.h"
#include "qt_common/qt_common.h"
#include "video_core/gpu.h"
#include "video_core/load_progress.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/renderer_base.h"

namespace {

/// The emulated GPU's disk pipeline cache serves nothing while NVN-HLE renders the game: the
/// guest's NVN calls go to the renderer's NVN sink (RendererBase::NvnSink, non-null only when the
/// replay context exists; a device or context failure leaves the emulated GPU in charge and the
/// cache loads as before), and its own pipeline cache loads separately. Loading a title's
/// legacy cache there only rebuilds pipelines nothing binds (a minute at boot for a large
/// `vulkan.bin` against a cold driver cache). EDEN_NX2_NVN_SKIP_LEGACY_CACHE=0 loads it anyway.
bool LoadLegacyDiskCache() {
    if (!Settings::values.use_disk_shader_cache.GetValue()) {
        return false;
    }
    if (QtCommon::system->Renderer().NvnSink() == nullptr) {
        return true;
    }
    const char* const value = std::getenv("EDEN_NX2_NVN_SKIP_LEGACY_CACHE");
    if (value != nullptr && std::string_view{value} == "0") {
        LOG_INFO(Frontend, "NVN-HLE: emulated GPU disk shader cache loaded "
                           "(EDEN_NX2_NVN_SKIP_LEGACY_CACHE=0)");
        return true;
    }
    LOG_INFO(Frontend, "NVN-HLE: emulated GPU disk shader cache not loaded (unused under NVN-HLE; "
                       "EDEN_NX2_NVN_SKIP_LEGACY_CACHE=0 loads it)");
    return false;
}

} // Anonymous namespace

EmuThread::EmuThread() {}

EmuThread::~EmuThread() = default;

void EmuThread::run() {
    Common::SetCurrentThreadName("EmuControlThread");

    auto& gpu = QtCommon::system->GPU();
    auto stop_token = m_stop_source.get_token();

    QtCommon::system->RegisterHostThread();

    // Main process has been loaded. Make the context current to this thread and begin GPU and CPU
    // execution.
    gpu.ObtainContext();

    emit LoadProgress(VideoCore::LoadCallbackStage::Prepare, 0, 0);
    // nvn-hle: the pipeline cache's prepare step and boot rebuild run on its own threads once the guest names its
    // title (or alongside the CPU code build below when the cache is opened early); they report to the loading
    // screen too. While the CPU code is built its progress has the screen (no flicker between the two).
    std::atomic<bool> building_cpu_code{true};
    VideoCore::LoadProgressSink* const nvn_progress = QtCommon::system->Renderer().NvnLoadProgress();
    if (nvn_progress != nullptr) {
        nvn_progress->Set([this, &building_cpu_code](VideoCore::LoadCallbackStage stage, std::size_t value,
                                                    std::size_t total) {
            if (!building_cpu_code.load(std::memory_order_relaxed)) {
                emit LoadProgress(stage, value, total);
            }
        });
    }
    // First boot after the emulator executable changed: the CPU JIT cache is rebuilt from the program's manifest
    // before the guest runs (nothing to do otherwise).
    QtCommon::system->PrebuildJitCache([this](std::size_t value, std::size_t total) {
        emit LoadProgress(VideoCore::LoadCallbackStage::BuildCpuCode, value, total);
    });
    building_cpu_code.store(false, std::memory_order_relaxed);
    if (LoadLegacyDiskCache()) {
        QtCommon::system->Renderer().ReadRasterizer()->LoadDiskResources(
            QtCommon::system->GetApplicationProcessProgramID(), stop_token,
            [this](VideoCore::LoadCallbackStage stage, std::size_t value, std::size_t total) {
                emit LoadProgress(stage, value, total);
            });
    }
    emit LoadProgress(VideoCore::LoadCallbackStage::Complete, 0, 0);

    gpu.ReleaseContext();
    gpu.Start();

    QtCommon::system->GetCpuManager().OnGpuReady();

    if (QtCommon::system->DebuggerEnabled()) {
        QtCommon::system->InitializeDebugger();
    }

    while (!stop_token.stop_requested()) {
        std::unique_lock lk{m_should_run_mutex};
        if (m_should_run) {
            QtCommon::system->Run();
            m_stopped.Reset();

            m_should_run_cv.wait(lk, stop_token, [&] {
                return !m_should_run || m_pending_shader_cache_title.has_value();
            });

            if (m_should_run && m_pending_shader_cache_title.has_value()) {
                const u64 program_id = *m_pending_shader_cache_title;
                m_pending_shader_cache_title.reset();

                lk.unlock();
                this->ReloadDiskShaderCache(program_id);
                lk.lock();
            }
        } else {
            QtCommon::system->Pause();
            m_stopped.Set();

            EmulationPaused(lk);
            m_should_run_cv.wait(lk, stop_token, [&] { return m_should_run; });
            EmulationResumed(lk);
        }
    }

    if (nvn_progress != nullptr) {
        nvn_progress->Clear(); // the callback refers to this thread's locals
    }

    // Shutdown the main emulated process
    QtCommon::system->DetachDebugger();
    QtCommon::system->ShutdownMainProcess();
}

void EmuThread::ReloadDiskShaderCache(u64 program_id) {
    if (!LoadLegacyDiskCache()) {
        return;
    }

    LOG_INFO(Frontend, "Reloading disk shader cache for {:016X}", program_id);

    auto& system = *QtCommon::system;
    auto& gpu = system.GPU();

    system.Pause();
    gpu.WaitForIdle();
    gpu.ObtainContext();

    emit ShaderCacheReloadStarted();
    emit LoadProgress(VideoCore::LoadCallbackStage::Prepare, 0, 0);
    system.Renderer().ReadRasterizer()->LoadDiskResources(program_id, m_stop_source.get_token(),
        [this](VideoCore::LoadCallbackStage stage, std::size_t value, std::size_t total) {
            emit LoadProgress(stage, value, total);
        });
    emit LoadProgress(VideoCore::LoadCallbackStage::Complete, 0, 0);
    emit ShaderCacheReloadFinished();

    gpu.ReleaseContext();
}

// Unlock while emitting signals so that the main thread can
// continue pumping events.

void EmuThread::EmulationPaused(std::unique_lock<std::mutex>& lk) {
    lk.unlock();
    emit DebugModeEntered();
    lk.lock();
}

void EmuThread::EmulationResumed(std::unique_lock<std::mutex>& lk) {
    lk.unlock();
    emit DebugModeLeft();
    lk.lock();
}
