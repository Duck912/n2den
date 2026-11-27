// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 n2den Contributors
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// Disk cache of the NVN pipeline cache: record posting, the driver VkPipelineCache blob and the
// background loader that rebuilds the cached pipelines.

#include <algorithm>
#include <cstddef>
#include <fstream>

#include <fmt/format.h>

#include "common/frame_stats.h"
#include "common/logging.h"
#include "core/debugger/nx2_debug_tools.h"
#include "video_core/load_progress.h"
#include "video_core/nvn/sink.h"
#include "video_core/renderer_vulkan/nvn/nvn_debug.h"
#include "video_core/renderer_vulkan/nvn/nvn_format.h"
#include "video_core/renderer_vulkan/nvn/nvn_pipeline_impl.h"

namespace Vulkan::Nvn {

// ================================================================================================
// disk cache
// ================================================================================================

void PipelineCache::Impl::PostBytes(std::vector<u8> bytes) {
    writer.Post([this, bytes = std::move(bytes)] {
        if (appender.Append(bytes)) {
            ++stat_disk_written;
        } else {
            LOG_ERROR(Render_Vulkan, "[nvn] pipeline cache: appending to {} failed",
                      disk_path.string());
        }
    });
}

void PipelineCache::Impl::PostShader(const StageArtifact& artifact) {
    {
        std::scoped_lock lock{written_mutex};
        if (!written_shaders.insert(artifact.record->id).second) {
            return;
        }
    }
    writer.Post([this, record = artifact.record] {
        if (appender.Append(Disk::EncodeShader(*record))) {
            ++stat_disk_written;
        } else {
            LOG_ERROR(Render_Vulkan, "[nvn] pipeline cache: appending to {} failed",
                      disk_path.string());
        }
    });
}

namespace {
/// Frame statistics and trace span of one pipeline rebuilt by the preload.
void NotePreloadTime(std::chrono::steady_clock::time_point start, bool compute) {
    const auto end = std::chrono::steady_clock::now();
    Common::FrameStats::AddStall(
        Common::FrameStats::Cause::PipePreload,
        static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()));
    Common::FrameStats::AddCount(Common::FrameStats::Counter::PipePreload);
    if (Core::Nx2Debug::SchedTrace::enabled.load(std::memory_order_relaxed)) {
        Core::Nx2Debug::SchedTrace::RecordHostSpan(
            Core::Nx2Debug::SchedTrace::HostSpanKind::PipelineCompile, start, end,
            2u | (compute ? 0x10u : 0u));
    }
}

Disk::ContentId SignatureOf(const std::vector<u8>& encoded) {
    const u128 hash =
        Common::CityHash128(reinterpret_cast<const char*>(encoded.data()), encoded.size());
    return {hash[0], hash[1]};
}
} // Anonymous namespace

void PipelineCache::Impl::PostGraphicsRecord(const Disk::GraphicsPipelineRecord& record) {
    std::vector<u8> bytes = Disk::EncodeGraphicsPipeline(record);
    {
        std::scoped_lock lock{written_mutex};
        if (!written_pipelines.insert(SignatureOf(bytes)).second) {
            return;
        }
    }
    PostBytes(std::move(bytes));
}

void PipelineCache::Impl::PostComputeRecord(const Disk::ComputePipelineRecord& record) {
    std::vector<u8> bytes = Disk::EncodeComputePipeline(record);
    {
        std::scoped_lock lock{written_mutex};
        if (!written_pipelines.insert(SignatureOf(bytes)).second) {
            return;
        }
    }
    PostBytes(std::move(bytes));
}

std::vector<u8> PipelineCache::Impl::ReadVkCacheBlob() const {
    std::ifstream file(vk_cache_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return {};
    }
    const auto end = file.tellg();
    constexpr size_t HEADER = VK_CACHE_MAGIC.size() + sizeof(u32) + sizeof(u64) + sizeof(u32);
    if (end < 0 || static_cast<size_t>(end) < HEADER) {
        return {};
    }
    file.seekg(0);
    std::array<char, 8> magic{};
    u32 version{};
    u64 size{};
    u32 crc{};
    file.read(magic.data(), magic.size());
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    file.read(reinterpret_cast<char*>(&crc), sizeof(crc));
    if (!file || magic != VK_CACHE_MAGIC || version != VK_CACHE_VERSION ||
        size != static_cast<u64>(end) - HEADER) {
        LOG_WARNING(Render_Vulkan, "[nvn] driver pipeline cache {} discarded (header)",
                    vk_cache_path.string());
        return {};
    }
    std::vector<u8> data(static_cast<size_t>(size));
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!file || Disk::Crc32(data) != crc) {
        LOG_WARNING(Render_Vulkan, "[nvn] driver pipeline cache {} discarded (checksum)",
                    vk_cache_path.string());
        return {};
    }
    return data;
}

void PipelineCache::Impl::WriteVkCache() {
    if (vk_cache_path.empty()) {
        return; // use_vulkan_driver_pipeline_cache is off: nothing is saved
    }
    size_t size = 0;
    if (vk_cache.Read(&size, nullptr) != VK_SUCCESS || size == 0) {
        return;
    }
    std::vector<u8> data(size);
    const VkResult result = vk_cache.Read(&size, data.data());
    if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
        return;
    }
    data.resize(size);
    std::vector<u8> file;
    file.reserve(VK_CACHE_MAGIC.size() + 16 + data.size());
    const auto put = [&file](const void* bytes, size_t count) {
        const auto* const first = static_cast<const u8*>(bytes);
        file.insert(file.end(), first, first + count);
    };
    const u32 version = VK_CACHE_VERSION;
    const u64 payload_size = data.size();
    const u32 crc = Disk::Crc32(data);
    put(VK_CACHE_MAGIC.data(), VK_CACHE_MAGIC.size());
    put(&version, sizeof(version));
    put(&payload_size, sizeof(payload_size));
    put(&crc, sizeof(crc));
    put(data.data(), data.size());
    if (Disk::WriteFileAtomic(vk_cache_path, file)) {
        new_since_vk_write = 0;
        LOG_INFO(Render_Vulkan, "[nvn] driver pipeline cache written: {} bytes", data.size());
    } else {
        LOG_ERROR(Render_Vulkan, "[nvn] driver pipeline cache {} could not be written",
                  vk_cache_path.string());
    }
}

void PipelineCache::Impl::MaybeWriteVkCache() {
    if (vk_cache_path.empty()) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const std::chrono::steady_clock::time_point last{
        std::chrono::steady_clock::duration{last_vk_write_ticks.load()}};
    if (now - last < VK_CACHE_WRITE_INTERVAL) {
        return;
    }
    last_vk_write_ticks = now.time_since_epoch().count();
    writer.Post([this] { WriteVkCache(); });
}

void PipelineCache::Impl::NotePreloadStep() {
    const size_t done = ++preload_done;
    if (!preload_reporting) {
        return;
    }
    const size_t total = preload_total.load();
    constexpr u64 MIN_INTERVAL_NS = 33'000'000; // the loading screen redraws at ~30 Hz at best
    const u64 now = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count());
    u64 last = preload_report_ticks.load();
    if (done < total) {
        if (now - last < MIN_INTERVAL_NS ||
            !preload_report_ticks.compare_exchange_strong(last, now)) {
            return;
        }
    }
    options.load_progress->Report(VideoCore::LoadCallbackStage::BuildPipelines, done, total);
}

void PipelineCache::Impl::PreloadGraphics(
    const Disk::GraphicsPipelineRecord& record,
    const std::unordered_map<Disk::ContentId, StageArtifact, Disk::ContentIdHash>& shaders) {
    GraphicsKey key;
    std::array<StageArtifact, GRAPHICS_STAGE_COUNT> artifacts;
    for (u32 stage = 0; stage < GRAPHICS_STAGE_COUNT; ++stage) {
        if (record.shaders[stage].IsZero()) {
            continue;
        }
        const auto it = shaders.find(record.shaders[stage]);
        if (it == shaders.end() || it->second.record->stage != stage) {
            NoteRow("disk cache: pipeline record refers to a missing shader record");
            NotePreloadStep();
            return;
        }
        artifacts[stage] = it->second;
        key.hashes[stage] = it->second.record->program_hash;
    }
    key.state = record.state;
    key.targets = record.targets;
    key.Finalize();
    if (lazy_pipelines) {
        {
            std::scoped_lock lock{pipeline_mutex};
            deferred_graphics[key].push_back(DeferredGraphics{record, std::move(artifacts)});
        }
        ++stat_deferred;
        NotePreloadStep();
        return;
    }
    QueueGraphicsBuild(key, std::move(artifacts), record, true);
}

void PipelineCache::Impl::QueueGraphicsBuild(const GraphicsKey& key,
                                             std::array<StageArtifact, GRAPHICS_STAGE_COUNT> artifacts,
                                             const Disk::GraphicsPipelineRecord& record,
                                             bool preload) {
    std::shared_ptr<PendingKey> pk;
    {
        std::scoped_lock lock{pipeline_mutex};
        auto& slot = pending_graphics[key];
        if (!slot) {
            slot = std::make_shared<PendingKey>();
        }
        pk = slot;
        ++pk->outstanding;
    }
    workers->QueueWork([this, key, pk, artifacts = std::move(artifacts), record, preload] {
        std::shared_ptr<GraphicsPipeline> pipeline;
        if (!stopping.load()) {
            std::array<const StageArtifact*, GRAPHICS_STAGE_COUNT> stages{};
            for (u32 stage = 0; stage < GRAPHICS_STAGE_COUNT; ++stage) {
                if (artifacts[stage].record) {
                    stages[stage] = &artifacts[stage];
                }
            }
            const auto start = std::chrono::steady_clock::now();
            pipeline = AssembleGraphics(record.state, record.targets, stages);
            if (preload) {
                NotePreloadTime(start, false);
            }
        }
        {
            std::scoped_lock lock{pipeline_mutex};
            if (pipeline) {
                auto& variants = graphics[key];
                if (pipeline->log) {
                    pipeline->log->from_disk = true;
                    pipeline->log->variant = static_cast<u32>(variants.size());
                }
                variants.push_back(pipeline);
            }
            if (--pk->outstanding == 0) {
                pending_graphics.erase(key);
            }
        }
        {
            std::scoped_lock lock{pk->mutex};
        }
        pk->cv.notify_all();
        if (pipeline) {
            ++(preload ? stat_loaded : stat_materialized);
        }
        if (preload) {
            NotePreloadStep();
        }
    });
}

void PipelineCache::Impl::PreloadCompute(
    const Disk::ComputePipelineRecord& record,
    const std::unordered_map<Disk::ContentId, StageArtifact, Disk::ContentIdHash>& shaders) {
    const auto it = shaders.find(record.shader);
    if (it == shaders.end() || it->second.record->stage != STAGE_COMPUTE) {
        NoteRow("disk cache: compute pipeline record refers to a missing shader record");
        NotePreloadStep();
        return;
    }
    ComputeKey key;
    key.hash = it->second.record->program_hash;
    key.environment_flags = record.environment_flags;
    key.local_memory_size = record.local_memory_size;
    key.shared_memory_size = record.shared_memory_size;
    if (lazy_pipelines) {
        {
            std::scoped_lock lock{pipeline_mutex};
            deferred_compute[key].push_back(DeferredCompute{it->second, record.compute_layout});
        }
        ++stat_deferred;
        NotePreloadStep();
        return;
    }
    QueueComputeBuild(key, it->second, record.compute_layout, true);
}

void PipelineCache::Impl::QueueComputeBuild(const ComputeKey& key, StageArtifact artifact,
                                            VC::ComputeDriverLayout layout, bool preload) {
    std::shared_ptr<PendingKey> pk;
    {
        std::scoped_lock lock{pipeline_mutex};
        auto& slot = pending_compute[key];
        if (!slot) {
            slot = std::make_shared<PendingKey>();
        }
        pk = slot;
        ++pk->outstanding;
    }
    workers->QueueWork([this, key, pk, artifact = std::move(artifact), layout, preload] {
        std::shared_ptr<ComputePipeline> pipeline;
        if (!stopping.load()) {
            const auto start = std::chrono::steady_clock::now();
            pipeline = AssembleCompute(artifact, layout);
            if (preload) {
                NotePreloadTime(start, true);
            }
        }
        {
            std::scoped_lock lock{pipeline_mutex};
            if (pipeline) {
                auto& variants = compute[key];
                if (pipeline->log) {
                    pipeline->log->from_disk = true;
                    pipeline->log->variant = static_cast<u32>(variants.size());
                }
                variants.push_back(pipeline);
            }
            if (--pk->outstanding == 0) {
                pending_compute.erase(key);
            }
        }
        {
            std::scoped_lock lock{pk->mutex};
        }
        pk->cv.notify_all();
        if (pipeline) {
            ++(preload ? stat_loaded : stat_materialized);
        }
        if (preload) {
            NotePreloadStep();
        }
    });
}

void PipelineCache::Impl::MaterializeDeferredGraphics(const GraphicsKey& key) {
    std::vector<DeferredGraphics> entries;
    {
        std::scoped_lock lock{pipeline_mutex};
        if (deferred_graphics.empty()) {
            return;
        }
        const auto it = deferred_graphics.find(key);
        if (it == deferred_graphics.end()) {
            return;
        }
        entries = std::move(it->second);
        deferred_graphics.erase(it);
    }
    for (DeferredGraphics& entry : entries) {
        QueueGraphicsBuild(key, std::move(entry.artifacts), entry.record, false);
    }
}

void PipelineCache::Impl::MaterializeDeferredCompute(const ComputeKey& key) {
    std::vector<DeferredCompute> entries;
    {
        std::scoped_lock lock{pipeline_mutex};
        if (deferred_compute.empty()) {
            return;
        }
        const auto it = deferred_compute.find(key);
        if (it == deferred_compute.end()) {
            return;
        }
        entries = std::move(it->second);
        deferred_compute.erase(it);
    }
    for (DeferredCompute& entry : entries) {
        QueueComputeBuild(key, std::move(entry.artifact), entry.layout, false);
    }
}

void PipelineCache::Impl::RunLoader(std::stop_token stop) {
    Common::SetCurrentThreadName("NvnPipelineLoad");
    const auto mark_queued = [this] {
        {
            std::scoped_lock lock{loader_mutex};
            loader_queued = true;
        }
        loader_cv.notify_all();
    };
    const auto finish = [this, &mark_queued] {
        mark_queued();
        {
            std::scoped_lock lock{loader_mutex};
            loader_done = true;
        }
        loader_cv.notify_all();
    };
    try {
        Disk::Contents contents;
        const Disk::LoadResult result = Disk::LoadFile(disk_path, disk_header, contents);
        switch (result.status) {
        case Disk::LoadStatus::Missing:
            break;
        case Disk::LoadStatus::Rejected:
            ++stat_disk_rejected;
            contents = {};
            LOG_WARNING(Render_Vulkan, "[nvn] pipeline cache {} rejected ({}); starting fresh",
                        disk_path.string(), result.reason);
            break;
        case Disk::LoadStatus::Loaded:
            if (result.HasBadTail()) {
                LOG_WARNING(Render_Vulkan,
                            "[nvn] pipeline cache {}: {} after {} valid records; the rest is "
                            "dropped",
                            disk_path.string(), result.reason, result.records);
            }
            break;
        }
        PrepareSideFiles(); // before the writer is Ready: its appends need the files open
        const bool prepared = Disk::PrepareFile(disk_path, disk_header, result) &&
                              appender.Open(disk_path);
        if (!prepared) {
            LOG_ERROR(Render_Vulkan, "[nvn] pipeline cache {} cannot be written; disk cache off",
                      disk_path.string());
            writer.SetState(DiskWriter::State::Disabled);
        } else {
            Disk::PruneCacheFiles(options.directory, disk_path, MAX_CACHE_FILES);
            writer.SetState(DiskWriter::State::Ready);
        }

        std::unordered_map<Disk::ContentId, StageArtifact, Disk::ContentIdHash> shaders;
        shaders.reserve(contents.shaders.size());
        for (Disk::ShaderRecord& record : contents.shaders) {
            StageArtifact artifact;
            if (record.has_program_constants) {
                artifact.constants =
                    std::make_shared<const std::vector<u32>>(std::move(record.program_constants));
                record.program_constants.clear();
            }
            const Disk::ContentId id = record.id;
            artifact.record = std::make_shared<const Disk::ShaderRecord>(std::move(record));
            shaders[id] = std::move(artifact);
        }
        {
            std::scoped_lock lock{written_mutex};
            for (const auto& [id, artifact] : shaders) {
                written_shaders.insert(id);
            }
        }
        // Records written before the state was canonical (or by another writer) differ only in
        // bits the pipeline ignores: fold them so they dedupe and match the keys of the draws.
        for (Disk::GraphicsPipelineRecord& record : contents.graphics) {
            Disk::CanonicalizeState(record.state, vertex_input_mode);
        }
        const auto preload_start = std::chrono::steady_clock::now();
        if (options.preload && workers && !stop.stop_requested()) {
            const size_t total = contents.graphics.size() + contents.compute.size();
            preload_total = total;
            // Reported even with no records: the boot state list (LoadStateList below) adds its
            // entries to the total as it queues them, and a never-played machine has only those.
            if (options.load_progress != nullptr) {
                preload_reporting = true;
                options.load_progress->Report(VideoCore::LoadCallbackStage::BuildPipelines, 0,
                                              total);
            }
            // Compute first: a dispatch never skips, so a compute key still in the queue stalls the
            // frame that needs it, while most draws may be skipped until their pipeline is built.
            for (const Disk::ComputePipelineRecord& record : contents.compute) {
                if (stop.stop_requested()) {
                    break;
                }
                {
                    std::scoped_lock lock{written_mutex};
                    if (!written_pipelines.insert(SignatureOf(Disk::EncodeComputePipeline(record)))
                             .second) {
                        NotePreloadStep();
                        continue;
                    }
                }
                PreloadCompute(record, shaders);
            }
            for (const Disk::GraphicsPipelineRecord& record : contents.graphics) {
                if (stop.stop_requested()) {
                    break;
                }
                {
                    std::scoped_lock lock{written_mutex};
                    if (!written_pipelines.insert(SignatureOf(Disk::EncodeGraphicsPipeline(record)))
                             .second) {
                        NotePreloadStep();
                        continue;
                    }
                }
                PreloadGraphics(record, shaders);
            }
            // The portable state list: entries the records did not cover are translated from the
            // shader store and built on the same workers (compute first); the wait below covers
            // them too.
            LoadStateList(stop);
            // Every cached key is now pending: draws that missed so far may look again.
            mark_queued();
            workers->WaitForRequests(stop);
            if (GplEnabled() && !stop.stop_requested()) {
                // Background: the loading screen and the loader's completion do not wait for it.
                QueueWarmShaderParts(contents.graphics, shaders);
            }
        } else {
            {
                std::scoped_lock lock{written_mutex};
                for (const auto& record : contents.graphics) {
                    written_pipelines.insert(SignatureOf(Disk::EncodeGraphicsPipeline(record)));
                }
                for (const auto& record : contents.compute) {
                    written_pipelines.insert(SignatureOf(Disk::EncodeComputePipeline(record)));
                }
            }
            LoadStateList(stop); // no workers: only remembers the entries (nothing is built)
            mark_queued();
        }
        if (preload_reporting.exchange(false)) {
            // Whatever ended the preload (finished, stopped), the screen moves on; later builds
            // (state list entries whose stages the game creates in play) are not reported.
            options.load_progress->Report(VideoCore::LoadCallbackStage::Complete, 0, 0);
        }
        if (!stop.stop_requested()) {
            if (result.status == Disk::LoadStatus::Loaded) {
                LOG_INFO(Render_Vulkan,
                         "[nvn] pipeline cache preload: {} shader, {} graphics, {} compute "
                         "records read, {} deferred to their first use (lazy pipelines {}), {} ms",
                         contents.shaders.size(), contents.graphics.size(), contents.compute.size(),
                         stat_deferred.load(), lazy_pipelines ? "on" : "off",
                         std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - preload_start)
                             .count());
            }
            LogCounts();
        }
    } catch (const std::exception& e) {
        LOG_ERROR(Render_Vulkan, "[nvn] pipeline cache load failed: {}", e.what());
        writer.SetState(DiskWriter::State::Disabled);
        std::scoped_lock lock{list_mutex};
        list_ready = true;
    }
    finish();
}

// ================================================================================================
// portable state list, local shader store, never-dispatched compute set
// ================================================================================================

void PipelineCache::Impl::QueueBuild(BuildPriority priority, std::function<void()> job) {
    Common::ThreadWorker* const pool = workers ? workers.get() : async_workers.get();
    if (pool == nullptr || stopping.load()) {
        return;
    }
    {
        std::scoped_lock lock{build_mutex};
        build_queues[static_cast<size_t>(priority)].push_back(std::move(job));
    }
    // One runner per job; each runner takes the most urgent job queued when it starts.
    pool->QueueWork([this] { RunNextBuild(); });
}

void PipelineCache::Impl::RunNextBuild() {
    std::function<void()> job;
    {
        std::scoped_lock lock{build_mutex};
        for (auto& queue : build_queues) {
            if (!queue.empty()) {
                job = std::move(queue.front());
                queue.pop_front();
                break;
            }
        }
    }
    if (job) {
        job();
    }
}

void PipelineCache::Impl::PrepareSideFiles() {
    if (list_enabled) {
        auto list = std::make_unique<Disk::StateList>();
        const Disk::SideFileResult result = Disk::LoadStateList(list_path, options.title_id, *list);
        if (result.status == Disk::LoadStatus::Rejected) {
            // Another version, another title or a damaged header: leave the file alone (it may
            // be an import for another build) and record nothing this session.
            LOG_WARNING(Render_Vulkan, "[nvn] state list {} ignored ({}); not written this session",
                        list_path.filename().string(), result.reason);
            list->graphics.clear();
            list->compute.clear();
        } else {
            if (!result.reason.empty()) {
                LOG_WARNING(Render_Vulkan,
                            "[nvn] state list {}: {} after {} records; the rest is dropped",
                            list_path.filename().string(), result.reason, result.records);
            }
            if (Disk::PrepareSideFile(list_path, Disk::EncodeStateListHeader(options.title_id),
                                      result) &&
                list_appender.Open(list_path)) {
                list_writable = true;
            } else {
                LOG_ERROR(Render_Vulkan, "[nvn] state list {} cannot be written",
                          list_path.string());
            }
            // What the own file holds is not appended again (imported entries are, once used:
            // the own file is the export).
            std::scoped_lock lock{list_mutex};
            for (const auto& entry : list->graphics) {
                list_signatures.insert(SignatureOf(Disk::EncodeStateGraphics(entry)));
            }
            for (const auto& entry : list->compute) {
                list_signatures.insert(SignatureOf(Disk::EncodeStateCompute(entry)));
            }
            for (const auto& record : list->unused) {
                list_signatures.insert(SignatureOf(Disk::EncodeStateComputeUnused(record)));
            }
        }
        // EDEN_NX2_NVN_STATE_LIST_IMPORT: one more list, read only (a list from elsewhere).
        if (const char* import = std::getenv("EDEN_NX2_NVN_STATE_LIST_IMPORT");
            import != nullptr && *import != '\0') {
            Disk::StateList imported;
            const Disk::SideFileResult import_result =
                Disk::LoadStateList(std::filesystem::path{import}, options.title_id, imported);
            LOG_INFO(Render_Vulkan, "[nvn] state list import {}: {} graphics, {} compute{}{}",
                     import, imported.graphics.size(), imported.compute.size(),
                     import_result.reason.empty() ? "" : "; ", import_result.reason);
            for (auto& entry : imported.graphics) {
                list->graphics.push_back(std::move(entry));
            }
            for (auto& entry : imported.compute) {
                list->compute.push_back(std::move(entry));
            }
            list->unused.insert(list->unused.end(), imported.unused.begin(), imported.unused.end());
        }
        boot_list = std::move(list);
    }
    if (store_enabled) {
        auto store = std::make_unique<std::vector<Disk::StoredStage>>();
        const Disk::SideFileResult result = Disk::LoadShaderStore(store_path, *store);
        if (result.status == Disk::LoadStatus::Rejected && result.reason != "damaged header") {
            LOG_WARNING(Render_Vulkan, "[nvn] shader store {} ignored ({}); not written",
                        store_path.filename().string(), result.reason);
            store->clear();
        } else {
            if (!result.reason.empty()) {
                LOG_WARNING(Render_Vulkan, "[nvn] shader store {}: {} after {} records",
                            store_path.filename().string(), result.reason, result.records);
            }
            if (result.status == Disk::LoadStatus::Rejected) {
                store->clear(); // a damaged header: the file is rebuilt from scratch
            }
            if (Disk::PrepareSideFile(store_path, Disk::EncodeShaderStoreHeader(), result) &&
                store_appender.Open(store_path)) {
                store_writable = true;
            } else {
                LOG_ERROR(Render_Vulkan, "[nvn] shader store {} cannot be written",
                          store_path.string());
            }
        }
        std::scoped_lock lock{list_mutex};
        for (const Disk::StoredStage& stage : *store) {
            stored_hashes.insert(stage.hash);
        }
        boot_store = std::move(store);
    }
    PreparePackageStages();
}

void PipelineCache::Impl::PreparePackageStages() {
    if (prepare_mode == PrepareMode::Off || !boot_list || !boot_store || !store_writable.load()) {
        return;
    }
    std::unordered_set<u64> needed;
    for (const Disk::StateListGraphics& entry : boot_list->graphics) {
        for (const u64 hash : entry.stages) {
            if (hash != 0) {
                needed.insert(hash);
            }
        }
    }
    for (const Disk::StateListCompute& entry : boot_list->compute) {
        if (entry.program != 0) {
            needed.insert(entry.program);
        }
    }
    const size_t listed = needed.size();
    std::unordered_set<u64> stored;
    {
        std::scoped_lock lock{list_mutex};
        stored = stored_hashes;
    }
    std::erase_if(needed, [&stored](u64 hash) { return stored.contains(hash); });
    const size_t missing = needed.size();
    // Stages an earlier scan looked for and did not find (made at run time, or from another
    // package): not looked for again; the store gets them when the game creates them.
    std::vector<u64> absent_before;
    if (Disk::LoadHashSet(scan_path, absent_before)) {
        const std::unordered_set<u64> absent(absent_before.begin(), absent_before.end());
        std::erase_if(needed, [&absent](u64 hash) { return absent.contains(hash); });
    }
    if (needed.empty()) {
        if (missing != 0) {
            LOG_INFO(Render_Vulkan,
                     "[nvn] prepare: {} listed stages are missing from the shader store and were "
                     "not found in the package by an earlier scan; not scanned again",
                     missing);
        }
        return;
    }
    const VideoCommon::Nvn::PackageStageSource source =
        VideoCommon::Nvn::GetPackageStageSource(options.title_id);
    if (!source) {
        LOG_INFO(Render_Vulkan,
                 "[nvn] prepare: {} listed stages are missing from the shader store and no "
                 "package reader is installed; they arrive when the game creates its programs",
                 needed.size());
        return;
    }
    const bool all = prepare_mode == PrepareMode::All;
    LOG_INFO(Render_Vulkan,
             "[nvn] prepare: {} of the {} stages the state list references are not in the shader "
             "store; reading them from the title's package{}",
             needed.size(), listed, all ? " (storing every stage found)" : "");
    const auto start = std::chrono::steady_clock::now();
    std::unordered_set<u64> found;
    u64 seen = 0;
    u64 written = 0;
    u64 written_bytes = 0;
    bool append_failed = false;
    auto last_note = start;
    // Loading screen: needed stages found so far of those needed (at most ~30 reports a second).
    auto last_report = start;
    const auto report = [&](bool force) {
        if (options.load_progress == nullptr) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (!force && now - last_report < std::chrono::milliseconds{33}) {
            return;
        }
        last_report = now;
        options.load_progress->Report(VideoCore::LoadCallbackStage::PrepareShaders, found.size(),
                                      needed.size());
    };
    report(true);
    std::string summary;
    try {
        summary = source([&](u32 stage, std::span<const u8> control, std::span<const u8> code) {
            ++seen;
            const u64 hash = HashProgramStage(control, code);
            const bool wanted = needed.contains(hash);
            if (wanted) {
                found.insert(hash);
                report(false);
            }
            if ((wanted || all) && stored.insert(hash).second) {
                Disk::StoredStage record = Disk::MakeStoredStage(hash, stage, control, code);
                const std::vector<u8> bytes = Disk::EncodeStoredStage(record);
                if (!store_appender.Append(bytes)) {
                    append_failed = true;
                    return false;
                }
                ++written;
                written_bytes += bytes.size();
                boot_store->push_back(std::move(record));
                std::scoped_lock lock{list_mutex};
                stored_hashes.insert(hash);
            }
            const auto now = std::chrono::steady_clock::now();
            if (now - last_note > std::chrono::seconds{5}) {
                last_note = now;
                LOG_INFO(Render_Vulkan,
                         "[nvn] prepare: {} package stages seen, {} of {} needed found, {} written",
                         seen, found.size(), needed.size(), written);
            }
            // Everything the list needs is there: the rest of the package is not read.
            return all || found.size() < needed.size();
        });
    } catch (const std::exception& e) {
        summary = fmt::format("failed: {}", e.what());
        append_failed = true; // the scan did not finish: nothing is remembered as absent
    }
    report(true);
    // A scan that read no stage at all (no RomFS) proves nothing about absence either.
    const bool complete = !append_failed && seen != 0;
    std::vector<u64> absent = absent_before;
    for (const u64 hash : needed) {
        if (!found.contains(hash)) {
            absent.push_back(hash);
        }
    }
    if (complete && absent.size() != absent_before.size()) {
        std::ranges::sort(absent);
        const auto [first, last] = std::ranges::unique(absent);
        absent.erase(first, last);
        (void)Disk::WriteFileAtomic(scan_path, Disk::EncodeHashSet(absent));
    }
    LOG_INFO(Render_Vulkan,
             "[nvn] prepare: {} ms; {} of the {} needed stages found in the package, {} not ({}), "
             "{} stages written to the shader store ({:.1f} MB){}; package: {}",
             std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - start)
                 .count(),
             found.size(), needed.size(), needed.size() - found.size(),
             found.size() == needed.size() ? "none missing"
             : complete                    ? "remembered in nvn_package_scan.bin"
                                           : "scan incomplete",
             written, static_cast<double>(written_bytes) / 1e6,
             append_failed ? "; scan stopped early (store not writable or an error)" : "", summary);
}

void PipelineCache::Impl::LoadStateList(std::stop_token stop) {
    std::unique_ptr<Disk::StateList> list = std::move(boot_list);
    std::unique_ptr<std::vector<Disk::StoredStage>> store = std::move(boot_store);
    std::scoped_lock lock{list_mutex};
    if (!list || stop.stop_requested()) {
        list_ready = true;
        return;
    }
    // Compute authority (see PrepareCompute): which programs the list names, and what its
    // creation records say about the others.
    for (const Disk::StateListCompute& entry : list->compute) {
        if (entry.program != 0) {
            list_compute_programs.insert(entry.program);
        }
    }
    for (const Disk::StateListUnused& record : list->unused) {
        if (record.program != 0 && !list_compute_programs.contains(record.program)) {
            u32& sessions = list_unused_compute[record.program];
            sessions = std::max(sessions, record.sessions);
        }
    }
    list_has_unused_records = !list->unused.empty();
    list_compute_authority =
        list_authority_enabled && list_compute_programs.size() >= list_authority_min;
    {
        const size_t seen_long = static_cast<size_t>(std::ranges::count_if(
            list_unused_compute, [](const auto& entry) { return entry.second >= 1; }));
        LOG_INFO(Render_Vulkan,
                 "[nvn] state list compute authority {}: {} compute programs named, {} more seen "
                 "created and not dispatched ({} long enough to tell); {}",
                 list_compute_authority ? "on" : "off", list_compute_programs.size(),
                 list_unused_compute.size(), seen_long,
                 !list_authority_enabled
                     ? std::string{"off by EDEN_NX2_NVN_LIST_COMPUTE_AUTHORITY=0"}
                 : !list_compute_authority
                     ? fmt::format("fewer than {} named (EDEN_NX2_NVN_LIST_COMPUTE_MIN): compute "
                                   "programs compile at creation as without a list",
                                   list_authority_min)
                 : list_has_unused_records
                     ? std::string{"programs seen undispatched compile at first dispatch, "
                                   "unknown ones at creation"}
                     : std::string{"no creation records: every program not named compiles at "
                                   "first dispatch"});
    }
    std::unordered_map<u64, const Disk::StoredStage*> store_index;
    if (store) {
        for (const Disk::StoredStage& stage : *store) {
            store_index.emplace(stage.hash, &stage);
        }
    }
    std::unordered_map<u64, SharedProgram> decoded;
    u64 damaged = 0;
    const auto from_store = [&](u64 hash) -> SharedProgram {
        if (const auto it = decoded.find(hash); it != decoded.end()) {
            return it->second;
        }
        SharedProgram result;
        if (const auto it = store_index.find(hash); it != store_index.end()) {
            auto program = std::make_shared<OwnedProgram>();
            if (Disk::DecodeStoredStage(*it->second, program->control, program->code)) {
                result = std::move(program);
            } else {
                ++damaged;
            }
        }
        decoded.emplace(hash, result);
        return result;
    };
    const bool can_build = options.preload && workers != nullptr;
    const u64 covered_before = list_covered.load();
    u64 queued = 0;
    u64 duplicates = 0;
    std::unordered_set<u64> missing_stages;
    std::unordered_set<Disk::ContentId, Disk::ContentIdHash> seen;
    const auto first_time = [&](const std::vector<u8>& encoded) {
        if (seen.insert(SignatureOf(encoded)).second) {
            return true;
        }
        ++duplicates;
        return false;
    };
    // Compute first: a dispatch never skips, so a compute key still queued stalls its frame.
    for (const Disk::StateListCompute& entry : list->compute) {
        if (entry.program == 0 || !first_time(Disk::EncodeStateCompute(entry))) {
            continue;
        }
        ++list_loaded_compute;
        listed_stage_hashes.insert(entry.program);
        if (!can_build) {
            continue;
        }
        if (ListEntryCovered(entry)) {
            ++list_covered;
            continue;
        }
        if (SharedProgram program = from_store(entry.program)) {
            queued += QueueListCompute(entry, std::move(program), ListSource::Boot) ? 1 : 0;
            continue;
        }
        const u32 id = static_cast<u32>(waiting.size());
        waiting.push_back({true, static_cast<u32>(waiting_compute.size()), 1});
        waiting_compute.push_back(entry);
        waiting_by_stage[entry.program].push_back(id);
        missing_stages.insert(entry.program);
    }
    for (const Disk::StateListGraphics& entry : list->graphics) {
        if (!first_time(Disk::EncodeStateGraphics(entry))) {
            continue;
        }
        ++list_loaded_graphics;
        for (const u64 hash : entry.stages) {
            if (hash != 0) {
                listed_stage_hashes.insert(hash);
            }
        }
        if (!can_build) {
            continue;
        }
        if (ListEntryCovered(entry)) {
            ++list_covered;
            continue;
        }
        std::array<SharedProgram, GRAPHICS_STAGE_COUNT> programs;
        u32 missing = 0;
        for (u32 stage = 0; stage < GRAPHICS_STAGE_COUNT; ++stage) {
            if (entry.stages[stage] != 0) {
                programs[stage] = from_store(entry.stages[stage]);
                missing += programs[stage] ? 0 : 1;
            }
        }
        if (missing == 0) {
            queued += QueueListGraphics(entry, programs, ListSource::Boot) ? 1 : 0;
            continue;
        }
        const u32 id = static_cast<u32>(waiting.size());
        waiting.push_back({false, static_cast<u32>(waiting_graphics.size()), missing});
        waiting_graphics.push_back(entry);
        for (u32 stage = 0; stage < GRAPHICS_STAGE_COUNT; ++stage) {
            const u64 hash = entry.stages[stage];
            if (hash == 0) {
                continue;
            }
            if (programs[stage]) {
                Available& slot = available[hash];
                slot.program = programs[stage];
                ++slot.refs;
            } else {
                waiting_by_stage[hash].push_back(id);
                missing_stages.insert(hash);
            }
        }
    }
    list_ready = true;
    LOG_INFO(Render_Vulkan,
             "[nvn] state list: {} graphics + {} compute entries ({} duplicates); {} queued at boot "
             "from the shader store ({} stages in it, {} damaged), {} covered by the records, {} "
             "waiting for {} stages to be created{}",
             list_loaded_graphics.load(), list_loaded_compute.load(), duplicates, queued,
             store_index.size(), damaged, list_covered.load() - covered_before, waiting.size(),
             missing_stages.size(), can_build ? "" : " (no preload workers: nothing is built)");
}

bool PipelineCache::Impl::MakeListRequest(const Disk::StateListGraphics& entry,
                                          GraphicsRequest& request, GraphicsKey& key) const {
    for (u32 stage = 0; stage < GRAPHICS_STAGE_COUNT; ++stage) {
        request.stages[stage].hash = entry.stages[stage];
    }
    // This machine's attachment formats, chosen exactly as the draw chooses them (FormatOf in
    // nvn_draw.cpp resolves the NVNformat of the target through ResolveHostFormat).
    const auto map = [this](s32 nvn, s32& out) {
        if (nvn == 0) {
            out = 0;
            return true;
        }
        const auto host = ResolveHostFormat(device, nvn);
        if (!host) {
            return false;
        }
        out = static_cast<s32>(host->format);
        return true;
    };
    bool mapped = map(entry.nvn_depth, request.targets.depth);
    for (u32 slot = 0; slot < MAX_COLOR_TARGETS; ++slot) {
        mapped = mapped && map(entry.nvn_color[slot], request.targets.color[slot]);
    }
    if (!mapped) {
        return false;
    }
    request.targets.samples = entry.samples;
    request.nvn_color = entry.nvn_color;
    request.nvn_depth = entry.nvn_depth;
    request.state = entry.state;
    // The key form, as GetGraphics makes it from a draw (this machine's vertex input mode).
    MakeKeyState(request.state, vertex_input_mode, must_emulate_scaled_formats);
    for (u32 stage = 0; stage < GRAPHICS_STAGE_COUNT; ++stage) {
        key.hashes[stage] = request.stages[stage].hash;
    }
    key.state = request.state;
    key.try_switch_hash = TrySwitchesHash();
    key.targets = request.targets;
    key.Finalize();
    return true;
}

ComputeKey PipelineCache::Impl::MakeListComputeKey(const Disk::StateListCompute& entry) const {
    ComputeKey key;
    key.try_switch_hash = TrySwitchesHash();
    key.hash = entry.program;
    key.environment_flags = entry.environment_flags & (ENV_LOCAL_MEMORY | ENV_SHARED_MEMORY);
    key.local_memory_size =
        (key.environment_flags & ENV_LOCAL_MEMORY) != 0 ? entry.local_memory_size : 0;
    key.shared_memory_size =
        (key.environment_flags & ENV_SHARED_MEMORY) != 0 ? entry.shared_memory_size : 0;
    return key;
}

bool PipelineCache::Impl::ListEntryCovered(const Disk::StateListGraphics& entry) {
    GraphicsRequest request;
    GraphicsKey key;
    if (!MakeListRequest(entry, request, key)) {
        return false; // counted as unmappable when it is queued
    }
    const ListOracle oracle{entry.queries};
    std::scoped_lock lock{pipeline_mutex};
    if (const auto it = graphics.find(key); it != graphics.end()) {
        for (const auto& candidate : it->second) {
            if (candidate->QueriesMatch(oracle)) {
                return true;
            }
        }
    }
    const auto it = pending_graphics.find(key);
    return it != pending_graphics.end() && !it->second->list_build;
}

bool PipelineCache::Impl::ListEntryCovered(const Disk::StateListCompute& entry) {
    const ComputeKey key = MakeListComputeKey(entry);
    const ListOracle oracle{entry.queries};
    std::scoped_lock lock{pipeline_mutex};
    if (const auto it = compute.find(key); it != compute.end()) {
        for (const auto& candidate : it->second) {
            if (candidate->QueriesMatch(oracle)) {
                return true;
            }
        }
    }
    const auto it = pending_compute.find(key);
    return it != pending_compute.end() && !it->second->list_build;
}

bool PipelineCache::Impl::QueueListGraphics(
    const Disk::StateListGraphics& entry,
    const std::array<SharedProgram, GRAPHICS_STAGE_COUNT>& programs, ListSource source) {
    GraphicsRequest request;
    GraphicsKey key;
    if (!MakeListRequest(entry, request, key)) {
        ++list_unmappable;
        return false;
    }
    for (u32 stage = 0; stage < GRAPHICS_STAGE_COUNT; ++stage) {
        if (entry.stages[stage] != 0 && programs[stage]) {
            request.stages[stage] = {entry.stages[stage], &programs[stage]->control,
                                     &programs[stage]->code};
        }
    }
    auto oracle = std::make_shared<ListOracle>(entry.queries);
    const u64 answers = Disk::HashQueries(entry.queries);
    std::shared_ptr<PendingKey> claim;
    bool had_variants = false;
    bool joined = false;
    {
        std::scoped_lock lock{pipeline_mutex};
        if (deferred_graphics.contains(key)) {
            ++list_covered; // a cached record of the key, built at its first use (lazy pipelines)
            return true;
        }
        if (const auto it = graphics.find(key); it != graphics.end()) {
            for (const auto& candidate : it->second) {
                if (candidate->QueriesMatch(*oracle)) {
                    ++list_covered;
                    return true;
                }
            }
            had_variants = !it->second.empty();
        }
        if (const auto it = pending_graphics.find(key); it != pending_graphics.end()) {
            if (!it->second->list_build) {
                ++list_covered; // the records' preload or a run-time compile builds it
                return true;
            }
            if (std::ranges::find(it->second->list_answers, answers) !=
                it->second->list_answers.end()) {
                // The same key and answers as an entry already queued (in the Dynamic vertex
                // input mode, another vertex layout of the same type classes): built once.
                ++list_same_key;
                return true;
            }
            // Another variant (texture answers) of a key a list build already claimed.
            claim = it->second;
            ++claim->outstanding;
            claim->list_answers.push_back(answers);
            joined = true;
        } else {
            claim = std::make_shared<PendingKey>();
            claim->outstanding = 1;
            claim->list_build = true;
            claim->list_answers.push_back(answers);
            pending_graphics.emplace(key, claim);
        }
    }
    const bool boot = source == ListSource::Boot;
    if (boot) {
        ++list_boot_queued;
        ++preload_total;
    } else {
        ++list_creation_queued;
    }
    QueueBuild(BuildPriority::ListGraphics,
               [this, request, programs, oracle, key, claim, had_variants, joined,
                boot]() mutable {
                   (void)programs; // owns the bytes `request.stages` points into
                   request.oracle = oracle.get();
                   std::shared_ptr<GraphicsPipeline> pipeline;
                   bool exists = false;
                   if (joined) {
                       std::scoped_lock lock{pipeline_mutex};
                       if (const auto it = graphics.find(key); it != graphics.end()) {
                           exists = std::ranges::any_of(it->second, [&](const auto& candidate) {
                               return candidate->QueriesMatch(*oracle);
                           });
                       }
                   }
                   if (exists) {
                       ++list_covered;
                   } else if (!stopping.load()) {
                       const auto start = std::chrono::steady_clock::now();
                       try {
                           pipeline = CompileGraphics(request, key, had_variants, true);
                       } catch (const std::exception& e) {
                           NoteRow(fmt::format("pipeline: state list build failed: {}",
                                               FirstLine(e.what())));
                       }
                       if (boot) {
                           NotePreloadTime(start, false);
                       }
                   }
                   if (!pipeline && !exists) {
                       ++list_build_failed;
                   }
                   if (pipeline && vertex_input_mode == VertexInputMode::Dynamic) {
                       RecordSkippedStates(*claim, request, *pipeline);
                   }
                   ReleasePending(pending_graphics, key, claim);
                   if (boot) {
                       NotePreloadStep();
                   }
               });
    return true;
}

bool PipelineCache::Impl::QueueListCompute(const Disk::StateListCompute& entry,
                                           SharedProgram program, ListSource source) {
    const ComputeKey key = MakeListComputeKey(entry);
    auto oracle = std::make_shared<ListOracle>(entry.queries);
    std::shared_ptr<PendingKey> claim;
    bool had_variants = false;
    bool joined = false;
    {
        std::scoped_lock lock{pipeline_mutex};
        if (deferred_compute.contains(key)) {
            ++list_covered; // a cached record of the key, built at its first use (lazy pipelines)
            return true;
        }
        if (const auto it = compute.find(key); it != compute.end()) {
            for (const auto& candidate : it->second) {
                if (candidate->QueriesMatch(*oracle)) {
                    ++list_covered;
                    return true;
                }
            }
            had_variants = !it->second.empty();
        }
        if (const auto it = pending_compute.find(key); it != pending_compute.end()) {
            if (!it->second->list_build) {
                ++list_covered;
                return true;
            }
            claim = it->second; // another variant of a key a list build claimed
            ++claim->outstanding;
            joined = true;
        } else {
            claim = std::make_shared<PendingKey>();
            claim->outstanding = 1;
            claim->list_build = true;
            // A dispatch that finds the build not started yet takes it over (GetCompute)
            // instead of waiting behind the queue.
            claim->at_creation = true;
            pending_compute.emplace(key, claim);
        }
    }
    const bool boot = source == ListSource::Boot;
    if (boot) {
        ++list_boot_queued;
        ++preload_total;
    } else {
        ++list_creation_queued;
    }
    QueueBuild(BuildPriority::ListCompute,
               [this, key, claim, program, oracle, had_variants, joined, boot] {
                   // The first build of the key may be taken over by a dispatch; a joined
                   // variant build is not (it holds its own `outstanding`).
                   if (!joined && claim->taken.exchange(true)) {
                       if (boot) {
                           NotePreloadStep();
                       }
                       return; // a dispatch took it over and releases the claim
                   }
                   std::shared_ptr<ComputePipeline> pipeline;
                   bool exists = false;
                   if (joined) {
                       std::scoped_lock lock{pipeline_mutex};
                       if (const auto it = compute.find(key); it != compute.end()) {
                           exists = std::ranges::any_of(it->second, [&](const auto& candidate) {
                               return candidate->QueriesMatch(*oracle);
                           });
                       }
                   }
                   if (exists) {
                       ++list_covered;
                   } else if (!stopping.load()) {
                       ComputeRequest request;
                       request.program = {key.hash, &program->control, &program->code};
                       request.oracle = oracle.get();
                       if ((key.environment_flags & ENV_LOCAL_MEMORY) != 0) {
                           request.environment.compute_local_memory_size = key.local_memory_size;
                       }
                       if ((key.environment_flags & ENV_SHARED_MEMORY) != 0) {
                           request.environment.compute_shared_memory_size = key.shared_memory_size;
                       }
                       const auto start = std::chrono::steady_clock::now();
                       try {
                           pipeline = CompileCompute(request, key, had_variants, false, true);
                       } catch (const std::exception& e) {
                           NoteRow(fmt::format("pipeline: state list build failed: {}",
                                               FirstLine(e.what())));
                       }
                       if (boot) {
                           NotePreloadTime(start, true);
                       }
                   }
                   if (!pipeline && !exists) {
                       ++list_build_failed;
                   }
                   ReleasePending(pending_compute, key, claim);
                   if (boot) {
                       NotePreloadStep();
                   }
               });
    return true;
}

void PipelineCache::Impl::NoteProgram(const std::array<StageProgram, STAGE_COUNT>& stages,
                                      std::shared_ptr<const void> owner) {
    (void)owner; // every byte this keeps is copied below
    if (!DiskEnabled() || stopping.load()) {
        return;
    }
    if (store_enabled && store_writable.load()) {
        for (u32 stage = 0; stage < STAGE_COUNT; ++stage) {
            const StageProgram& program = stages[stage];
            if (program.hash == 0 || program.control == nullptr || program.code == nullptr) {
                continue;
            }
            {
                std::scoped_lock lock{list_mutex};
                if (!stored_hashes.insert(program.hash).second) {
                    continue;
                }
            }
            writer.Post([this, hash = program.hash, stage, control = *program.control,
                         code = *program.code] {
                if (store_appender.Append(Disk::EncodeStoredStage(hash, stage, control, code))) {
                    ++store_appended;
                }
            });
        }
    }
    if (!list_enabled) {
        return;
    }
    std::scoped_lock lock{list_mutex};
    if (!list_ready || waiting_by_stage.empty()) {
        return;
    }
    std::vector<u32> ready;
    for (u32 stage = 0; stage < STAGE_COUNT; ++stage) {
        const StageProgram& program = stages[stage];
        if (program.hash == 0 || program.control == nullptr || program.code == nullptr) {
            continue;
        }
        const auto it = waiting_by_stage.find(program.hash);
        if (it == waiting_by_stage.end()) {
            continue;
        }
        const std::vector<u32> ids = std::move(it->second);
        waiting_by_stage.erase(it);
        auto copy = std::make_shared<OwnedProgram>();
        copy->control = *program.control;
        copy->code = *program.code;
        Available& slot = available[program.hash];
        slot.program = std::move(copy);
        slot.refs += static_cast<u32>(ids.size());
        for (const u32 id : ids) {
            if (--waiting[id].missing == 0) {
                ready.push_back(id);
            }
        }
    }
    // Compute first (see LoadStateList).
    std::ranges::stable_sort(ready, [this](u32 a, u32 b) {
        return waiting[a].compute && !waiting[b].compute;
    });
    const auto take = [this](u64 hash) -> SharedProgram {
        const auto it = available.find(hash);
        if (it == available.end()) {
            return {};
        }
        SharedProgram program = it->second.program;
        if (--it->second.refs == 0) {
            available.erase(it);
        }
        return program;
    };
    for (const u32 id : ready) {
        const Waiting& entry = waiting[id];
        if (entry.compute) {
            const Disk::StateListCompute& listed = waiting_compute[entry.index];
            if (SharedProgram program = take(listed.program)) {
                (void)QueueListCompute(listed, std::move(program), ListSource::Creation);
            }
            continue;
        }
        const Disk::StateListGraphics& listed = waiting_graphics[entry.index];
        std::array<SharedProgram, GRAPHICS_STAGE_COUNT> programs;
        bool complete = true;
        for (u32 stage = 0; stage < GRAPHICS_STAGE_COUNT; ++stage) {
            if (listed.stages[stage] != 0) {
                programs[stage] = take(listed.stages[stage]);
                complete = complete && programs[stage] != nullptr;
            }
        }
        if (complete) {
            (void)QueueListGraphics(listed, programs, ListSource::Creation);
        }
    }
}

void PipelineCache::Impl::PostListRecord(std::vector<u8> bytes) {
    {
        std::scoped_lock lock{list_mutex};
        if (!list_signatures.insert(SignatureOf(bytes)).second) {
            return;
        }
    }
    writer.Post([this, bytes = std::move(bytes)] {
        if (list_appender.Append(bytes)) {
            ++list_appended;
        }
    });
}

namespace {
double MsSince(s64 ticks) {
    const auto built = std::chrono::steady_clock::time_point{std::chrono::steady_clock::duration{ticks}};
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - built)
        .count();
}
} // Anonymous namespace

void PipelineCache::Impl::NoteGraphicsUse(const GraphicsRequest& request,
                                          const GraphicsPipeline& pipeline) {
    if (pipeline.from_list) {
        ++list_used_graphics;
        lead_graphics_ms.push_back(MsSince(pipeline.built_ticks));
    }
    if (vertex_input_mode == VertexInputMode::Dynamic) {
        NoteDynamicLayout(request, pipeline);
    } else {
        RecordGraphics(request, pipeline);
    }
}

void PipelineCache::Impl::NoteDynamicLayout(const GraphicsRequest& request,
                                            const GraphicsPipeline& pipeline) {
    if (!DiskEnabled() || !list_enabled || !list_writable.load(std::memory_order_relaxed)) {
        return;
    }
    // Same pipeline = same key, so the rest of the Static form is the same: the layout words
    // tell entries apart.
    const u64 layout = StaticLayoutHash(request.state);
    if (std::ranges::find(pipeline.listed_layouts, layout) != pipeline.listed_layouts.end()) {
        return;
    }
    pipeline.listed_layouts.push_back(layout);
    RecordGraphics(request, pipeline);
}

u64 PipelineCache::Impl::StaticLayoutHash(const GraphicsFixedState& draw_state) {
    // Attributes, strides and divisors are contiguous in GraphicsFixedState.
    GraphicsFixedState state = draw_state;
    MakeKeyState(state, VertexInputMode::Static, false);
    static_assert(offsetof(GraphicsFixedState, stream_stride) ==
                  offsetof(GraphicsFixedState, attributes) + sizeof(state.attributes));
    static_assert(offsetof(GraphicsFixedState, stream_divisor) ==
                  offsetof(GraphicsFixedState, stream_stride) + sizeof(state.stream_stride));
    return Common::CityHash64(
        reinterpret_cast<const char*>(state.attributes.data()),
        sizeof(state.attributes) + sizeof(state.stream_stride) + sizeof(state.stream_divisor));
}

void PipelineCache::Impl::NoteSkippedLayout(PendingKey& pending,
                                            const GraphicsFixedState& draw_state) {
    if (!DiskEnabled() || !list_enabled || !list_writable.load(std::memory_order_relaxed)) {
        return;
    }
    const u64 layout = StaticLayoutHash(draw_state);
    std::scoped_lock lock{pipeline_mutex};
    if (pending.skipped_layouts.size() >= PendingKey::MAX_SKIPPED_STATES ||
        std::ranges::find(pending.skipped_layouts, layout) != pending.skipped_layouts.end()) {
        return;
    }
    pending.skipped_layouts.push_back(layout);
    pending.skipped_states.push_back(draw_state);
}

void PipelineCache::Impl::RecordSkippedStates(PendingKey& pending, const GraphicsRequest& request,
                                              const GraphicsPipeline& pipeline) {
    std::vector<GraphicsFixedState> states;
    {
        std::scoped_lock lock{pipeline_mutex};
        states.swap(pending.skipped_states);
    }
    for (const GraphicsFixedState& state : states) {
        RecordGraphics(request, pipeline, &state);
    }
}

void PipelineCache::Impl::RecordGraphics(const GraphicsRequest& request,
                                         const GraphicsPipeline& pipeline,
                                         const GraphicsFixedState* draw_state) {
    if (!DiskEnabled() || !list_enabled || !list_writable.load()) {
        return;
    }
    Disk::StateListGraphics entry;
    for (u32 stage = 0; stage < GRAPHICS_STAGE_COUNT; ++stage) {
        entry.stages[stage] = request.stages[stage].hash;
    }
    entry.state = draw_state != nullptr ? *draw_state : request.state;
    // The Static key form: a function of the guest words alone (any machine derives its own key).
    MakeKeyState(entry.state, VertexInputMode::Static, false);
    entry.nvn_color = request.nvn_color;
    entry.nvn_depth = request.nvn_depth;
    entry.samples = request.targets.samples;
    // Only what another machine can rebuild exactly: each NVN format must give back the VkFormat
    // the draw used (ResolveHostFormat, as the rebuild maps it).
    const auto maps_back = [this](s32 nvn, s32 used) {
        if (nvn == 0) {
            return used == 0;
        }
        const auto host = ResolveHostFormat(device, nvn);
        return host && static_cast<s32>(host->format) == used;
    };
    bool portable = maps_back(entry.nvn_depth, request.targets.depth);
    for (u32 slot = 0; slot < MAX_COLOR_TARGETS; ++slot) {
        portable = portable && maps_back(entry.nvn_color[slot], request.targets.color[slot]);
    }
    if (!portable) {
        if (++list_unportable <= 8) {
            LOG_WARNING(Render_Vulkan,
                        "[nvn] state list: a pipeline of stages {:016x} {:016x} is not recorded: "
                        "its target formats do not map back from their NVN formats",
                        entry.stages[0], entry.stages[1]);
        }
        return;
    }
    entry.queries = pipeline.queries;
    PostListRecord(Disk::EncodeStateGraphics(entry));
}

void PipelineCache::Impl::NoteComputeUse(const ComputeRequest& request,
                                         const ComputePipeline& pipeline) {
    NoteGuestDispatch(request.program.hash);
    if (pipeline.from_list) {
        const u64 used = ++list_used_compute;
        const double lead = MsSince(pipeline.built_ticks);
        lead_compute_ms.push_back(lead);
        if (used <= 256) {
            LOG_INFO(Render_Vulkan,
                     "[nvn] state list: compute {:016x} first dispatched {:.0f} ms after its build",
                     request.program.hash, lead);
        }
    }
    if (!DiskEnabled() || !list_enabled || !list_writable.load()) {
        return;
    }
    Disk::StateListCompute entry;
    entry.program = request.program.hash;
    if (request.environment.compute_local_memory_size) {
        entry.environment_flags |= ENV_LOCAL_MEMORY;
        entry.local_memory_size = *request.environment.compute_local_memory_size;
    }
    if (request.environment.compute_shared_memory_size) {
        entry.environment_flags |= ENV_SHARED_MEMORY;
        entry.shared_memory_size = *request.environment.compute_shared_memory_size;
    }
    entry.queries = pipeline.queries;
    PostListRecord(Disk::EncodeStateCompute(entry));
}

void PipelineCache::Impl::NoteGuestDispatch(u64 hash) {
    if (dispatched_compute.insert(hash).second && unused_enabled) {
        // Dispatched after all: no longer "unused" (a later creation this session compiles
        // ahead again, and the set written at close drops it).
        std::scoped_lock lock{pipeline_mutex};
        unused_compute.erase(hash);
    }
}

void PipelineCache::Impl::SaveUnusedCompute() {
    if (!unused_enabled || created_compute.empty()) {
        return;
    }
    // Earlier sessions' counts (minus what was dispatched since); a program this session created
    // and did not dispatch counts one more session only if it lived long enough undispatched to
    // tell (a short session that ends before a first dispatch does not mark it).
    const auto now = std::chrono::steady_clock::now();
    const auto min_age = std::chrono::seconds{unused_min_age_s};
    std::map<u64, u32> next;
    {
        std::scoped_lock lock{pipeline_mutex};
        next.insert(unused_compute.begin(), unused_compute.end());
    }
    u64 counted = 0;
    u64 too_young = 0;
    for (const auto& [hash, created] : created_compute) {
        if (dispatched_compute.contains(hash)) {
            continue;
        }
        if (now - created < min_age) {
            ++too_young;
            continue;
        }
        ++next[hash];
        ++counted;
    }
    for (const u64 hash : dispatched_compute) {
        next.erase(hash);
    }
    std::vector<Disk::UnusedCompute> entries;
    entries.reserve(next.size());
    for (const auto& [hash, sessions] : next) {
        entries.push_back({hash, sessions});
    }
    if (Disk::WriteFileAtomic(unused_path, Disk::EncodeUnusedSet(entries))) {
        const size_t skipping = static_cast<size_t>(std::ranges::count_if(
            entries, [this](const Disk::UnusedCompute& e) { return e.sessions >= unused_min_sessions; }));
        LOG_INFO(Render_Vulkan,
                 "[nvn] never-dispatched compute set: {} programs, {} of them skip the "
                 "creation-time compile next time ({} created, {} dispatched this session; {} "
                 "counted, {} created less than {} s before the close not counted)",
                 entries.size(), skipping, created_compute.size(), dispatched_compute.size(),
                 counted, too_young, unused_min_age_s);
    }
}

void PipelineCache::Impl::RecordUnusedComputeInList() {
    if (!list_enabled || !list_writable.load() || created_compute.empty()) {
        return;
    }
    // Hashes only: the portable list learns which programs the title creates and leaves
    // undispatched, so that a machine importing it can tell them from unknown programs. The count
    // is the most any earlier session (this machine's set or the list) saw, plus this session if
    // the program lived long enough undispatched to tell; capped, so a program costs at most a
    // few records over the life of the list.
    constexpr u32 MAX_LISTED_SESSIONS = 4;
    const auto now = std::chrono::steady_clock::now();
    const auto min_age = std::chrono::seconds{unused_min_age_s};
    u64 appended = 0;
    u64 seen = 0;
    for (const auto& [hash, created] : created_compute) {
        if (hash == 0 || dispatched_compute.contains(hash) || list_compute_programs.contains(hash)) {
            continue;
        }
        ++seen;
        u32 before = 0;
        bool known = false;
        if (const auto it = list_unused_compute.find(hash); it != list_unused_compute.end()) {
            before = it->second;
            known = true;
        }
        {
            std::scoped_lock lock{pipeline_mutex};
            if (const auto it = unused_compute.find(hash); it != unused_compute.end()) {
                before = std::max(before, it->second);
            }
        }
        const u32 sessions =
            std::min(MAX_LISTED_SESSIONS, before + (now - created >= min_age ? 1u : 0u));
        if (known && sessions <= list_unused_compute[hash]) {
            continue;
        }
        const std::vector<u8> bytes = Disk::EncodeStateComputeUnused({hash, sessions});
        {
            std::scoped_lock lock{list_mutex};
            if (!list_signatures.insert(SignatureOf(bytes)).second) {
                continue;
            }
        }
        if (!list_appender.Append(bytes)) {
            break;
        }
        ++appended;
    }
    LOG_INFO(Render_Vulkan,
             "[nvn] state list: {} compute programs created and not dispatched this session; {} "
             "creation records appended",
             seen, appended);
}

void PipelineCache::Impl::LogListCounts() const {
    u64 unused_graphics = 0;
    u64 unused_compute_built = 0;
    // Cache log: which listed graphics pipelines this session never drew (a list made from
    // several sessions holds variants a single session may or may not meet).
    std::string unused_detail;
    {
        std::scoped_lock lock{pipeline_mutex};
        for (const auto& [key, variants] : graphics) {
            for (const auto& pipeline : variants) {
                if (!pipeline->from_list || pipeline->used.load()) {
                    continue;
                }
                if (++unused_graphics <= 32 && Debug::Get().cache_log) {
                    unused_detail += fmt::format(" [{:x} {:x} color0 {} depth {}]", key.hashes[0],
                                                 key.hashes[1], key.targets.color[0],
                                                 key.targets.depth);
                }
            }
        }
        for (const auto& [key, variants] : compute) {
            for (const auto& pipeline : variants) {
                unused_compute_built += pipeline->from_list && !pipeline->used.load() ? 1 : 0;
            }
        }
    }
    const auto summary = [](std::vector<double> values) {
        if (values.empty()) {
            return std::string{"none"};
        }
        std::ranges::sort(values);
        return fmt::format("n={} min {:.0f} median {:.0f} max {:.0f} ms", values.size(),
                           values.front(), values[values.size() / 2], values.back());
    };
    LOG_INFO(Render_Vulkan,
             "[nvn] state list counts: loaded {} graphics + {} compute; queued {} at boot, {} at "
             "program creation; covered by records/pending {}; same key as another entry {}; "
             "unmappable {}; built {}, failed {}; "
             "used {} graphics + {} compute, built but never used {} graphics + {} compute; "
             "run-time compiles: graphics {} (listed programs {}), compute at dispatch {} "
             "(listed programs {}); recorded {} new entries, {} not portable; shader store +{} "
             "stages",
             list_loaded_graphics.load(), list_loaded_compute.load(), list_boot_queued.load(),
             list_creation_queued.load(), list_covered.load(), list_same_key.load(),
             list_unmappable.load(),
             list_built.load(), list_build_failed.load(), list_used_graphics.load(),
             list_used_compute.load(), unused_graphics, unused_compute_built,
             runtime_graphics.load(), runtime_graphics_listed.load(), runtime_compute.load(),
             runtime_compute_listed.load(), list_appended.load(), list_unportable.load(),
             store_appended.load());
    if (!unused_detail.empty()) {
        LOG_INFO(Render_Vulkan, "[nvn] state list: graphics built but never used (first 32):{}",
                 unused_detail);
    }
    LOG_INFO(Render_Vulkan, "[nvn] state list lead time (build done -> first use): graphics {}; "
                            "compute {}",
             summary(lead_graphics_ms), summary(lead_compute_ms));
}

} // namespace Vulkan::Nvn
