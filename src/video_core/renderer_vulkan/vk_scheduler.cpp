// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <map>
#include <vector>
#include <algorithm>
#ifdef _WIN32
#include <windows.h>
#endif
#include <cstdlib>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include <fmt/format.h>

#include "video_core/renderer_vulkan/vk_query_cache.h"
#include "video_core/sync_counters.h"

#include "common/settings.h"
#include "video_core/gpu_written_pages.h"
#include "common/thread.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "video_core/renderer_vulkan/vk_command_pool.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace VideoCommon {
void DebugCaptureCallers(u64* out, u32 count) {
    for (u32 i = 0; i < count; ++i) {
        out[i] = 0;
    }
#ifdef _WIN32
    void* frames[16]{};
    const USHORT n = RtlCaptureStackBackTrace(2, (std::min)(count, 16U), frames, nullptr);
    const u64 base = reinterpret_cast<u64>(GetModuleHandleW(nullptr));
    for (USHORT i = 0; i < n; ++i) {
        out[i] = reinterpret_cast<u64>(frames[i]) - base + 0x140000000ULL;
    }
#endif
}
} // namespace VideoCommon

namespace Vulkan {

void Scheduler::CommandChunk::ExecuteAll(vk::CommandBuffer cmdbuf,
                                         vk::CommandBuffer upload_cmdbuf) {
    auto command = first;
    while (command != nullptr) {
        auto next = command->GetNext();
        command->Execute(cmdbuf, upload_cmdbuf);
        command->~Command();
        command = next;
    }
    submit = false;
    command_offset = 0;
    first = nullptr;
    last = nullptr;
}

Scheduler::Scheduler(const Device& device_, StateTracker& state_tracker_)
    : device{device_}, state_tracker{state_tracker_},
      master_semaphore{std::make_unique<MasterSemaphore>(device)},
      command_pool{std::make_unique<CommandPool>(*master_semaphore, device)} {

    AcquireNewChunk();
    AllocateWorkerCommandBuffer();
    worker_thread = std::jthread([this](std::stop_token token) { WorkerThread(token); });
}

Scheduler::~Scheduler() = default;

u64 Scheduler::Flush(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore) {
    // When flushing, we only send data to the worker thread; no waiting is necessary.
    const u64 signal_value = SubmitExecution(signal_semaphore, wait_semaphore);
    AllocateNewContext();
    return signal_value;
}

void Scheduler::Finish(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore) {
    const VideoCommon::BackStats::WaitTimer wait_timer{VideoCommon::BackStats::shared.finish,
                                                       StatsQueued(), StatsRetained()};
#ifdef _WIN32
    // [DEBUG] EDEN_NX2_FINISH_CALLERS=1: every 4000 Finish calls, the most frequent callers as
    // return addresses (RVA + 0x140000000, for llvm-symbolizer --obj=eden.exe), 3 frames each.
    static const bool tally_callers = std::getenv("EDEN_NX2_FINISH_CALLERS") != nullptr;
    if (tally_callers) {
        static std::mutex tally_mutex;
        static std::map<std::array<u64, 7>, u32> tally;
        static u32 calls = 0;
        void* frames[8]{};
        const USHORT n = RtlCaptureStackBackTrace(1, 8, frames, nullptr);
        const u64 base = reinterpret_cast<u64>(GetModuleHandleW(nullptr));
        std::array<u64, 7> key{};
        for (USHORT i = 0; i < n && i < 7; ++i) {
            key[i] = reinterpret_cast<u64>(frames[i]) - base + 0x140000000ULL;
        }
        std::scoped_lock lock{tally_mutex};
        ++tally[key];
        if (++calls % 4000 == 0) {
            std::vector<std::pair<u32, std::array<u64, 7>>> top;
            for (const auto& [k, v] : tally) {
                top.emplace_back(v, k);
            }
            std::ranges::sort(top, std::greater{});
            std::string text;
            for (size_t i = 0; i < top.size() && i < 8; ++i) {
                text += fmt::format(" | {} x", top[i].first);
                for (const u64 frame : top[i].second) {
                    text += fmt::format(" {:#x}", frame);
                }
            }
            LOG_INFO(Render_Vulkan, "Finish callers (last 4000):{}", text);
            tally.clear();
        }
    }
#endif
    // When finishing, we need to wait for the submission to have executed on the device.
    VideoCommon::SyncCounters::Add(VideoCommon::SyncCounters::Finishes);
    const u64 presubmit_tick = CurrentTick();
    SubmitExecution(signal_semaphore, wait_semaphore);
    Wait(presubmit_tick);
    AllocateNewContext();
}

void Scheduler::WaitWorker() {
    const VideoCommon::BackStats::WaitTimer wait_timer{VideoCommon::BackStats::shared.wait_worker,
                                                       StatsQueued(), StatsRetained()};
    DispatchWork();

    // Ensure the queue is drained.
    {
        std::unique_lock ql{queue_mutex};
        event_cv.wait(ql, [this] { return work_queue.empty(); });
    }

    // Now wait for execution to finish.
    std::scoped_lock el{execution_mutex};
}

void Scheduler::DispatchWork() {
    if (chunk && !chunk->Empty()) {
        // Waking the worker is a system call: skip it while the worker is still busy, it pops
        // the queue before waiting again. (Eglantine a99484f3.)
        bool wake;
        {
            std::scoped_lock ql{queue_mutex};
            if (VideoCommon::BackStats::Enabled()) {
                namespace BackStats = VideoCommon::BackStats;
                const u64 queued = work_queue.size();
                BackStats::Add(BackStats::shared.dispatches, 1);
                BackStats::Add(BackStats::shared.dispatch_queue_sum, queued);
                BackStats::Max(BackStats::shared.dispatch_queue_max, queued);
            }
            work_queue.push(std::move(chunk));
            wake = worker_waiting;
        }
        if (wake) {
            event_cv.notify_all();
        }
        AcquireNewChunk();
    }
}

void Scheduler::BeginRenderPassImpl(const Framebuffer* framebuffer, VkRenderPass renderpass,
                                    const VkClearValue* clear_values, u32 clear_value_count) {
    const VkFramebuffer framebuffer_handle = framebuffer->Handle();
    const VkExtent2D render_area = framebuffer->RenderArea();
    VideoCommon::SyncCounters::Add(VideoCommon::SyncCounters::RenderPassBegins);
    state.renderpass = renderpass;
    state.framebuffer = framebuffer_handle;
    state.render_area = render_area;

    if (GPU::Logging::IsActive() && Settings::values.gpu_log_vulkan_calls.GetValue()) {
        const std::string render_pass_info =
            fmt::format("renderArea={}x{}, numImages={}", render_area.width, render_area.height,
                        framebuffer->NumImages());
        GPU::Logging::GPULogger::GetInstance().LogRenderPassBegin(render_pass_info);
    }

    std::array<VkClearValue, 9> values{};
    for (u32 i = 0; i < clear_value_count && i < values.size(); ++i) {
        values[i] = clear_values[i];
    }
    Record([renderpass, framebuffer_handle, render_area, values, clear_value_count](
               vk::CommandBuffer cmdbuf) {
        const VkRenderPassBeginInfo renderpass_bi{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = nullptr,
            .renderPass = renderpass,
            .framebuffer = framebuffer_handle,
            .renderArea =
                {
                    .offset = {.x = 0, .y = 0},
                    .extent = render_area,
                },
            .clearValueCount = clear_value_count,
            .pClearValues = clear_value_count != 0 ? values.data() : nullptr,
        };
        cmdbuf.BeginRenderPass(renderpass_bi, VK_SUBPASS_CONTENTS_INLINE);
    });
    num_renderpass_images = framebuffer->NumImages();
    renderpass_images = framebuffer->Images();
    renderpass_image_ranges = framebuffer->ImageRanges();
    framebuffer->MarkResolveShadowsUpToDate();
}

void Scheduler::RealizeDeferredClear() {
    if (deferred_clear.framebuffer == nullptr) {
        return;
    }
    const DeferredClear dc = deferred_clear;
    deferred_clear = {};

    std::array<VkClearValue, 9> clear_values{};
    u32 count = 0;
    const RenderPassKey& base = dc.framebuffer->RenderPassKeyBase();
    for (u32 slot = 0; slot < 8; ++slot) {
        if (base.color_formats[slot] == VideoCore::Surface::PixelFormat::Invalid) {
            continue;
        }
        clear_values[count++] = dc.color_values[slot];
    }
    if (base.depth_format != VideoCore::Surface::PixelFormat::Invalid) {
        clear_values[count++] = dc.depth_stencil_value;
    }
    const u32 color_discard_mask =
        dc.framebuffer->DiscardsMsaaColor() ? dc.color_clear_mask : 0u;
    const bool depth_stencil_discard =
        dc.depth_stencil && dc.framebuffer->DiscardsMsaaDepthStencil();
    const VkRenderPass renderpass = dc.framebuffer->RenderPassVariant(
        dc.color_clear_mask, dc.depth_stencil, color_discard_mask, depth_stencil_discard);
    EndRenderPass();
    BeginRenderPassImpl(dc.framebuffer, renderpass, clear_values.data(), count);
}

bool Scheduler::DeferColorClear(const Framebuffer* framebuffer, u32 rt_slot,
                                const VkClearValue& value) {
    if (IsRenderPassActive()) {
        return false;
    }
    if (deferred_clear.framebuffer != nullptr && deferred_clear.framebuffer != framebuffer) {
        const VideoCommon::SyncCounters::ScopedCause cause{VideoCommon::SyncCounters::CauseDeferredClear};
        RealizeDeferredClear();
        EndRenderPass();
    }
    deferred_clear.framebuffer = framebuffer;
    deferred_clear.color_clear_mask |= 1u << rt_slot;
    deferred_clear.color_values[rt_slot] = value;
    return true;
}

bool Scheduler::DeferDepthStencilClear(const Framebuffer* framebuffer, const VkClearValue& value) {
    if (IsRenderPassActive()) {
        return false;
    }
    if (deferred_clear.framebuffer != nullptr && deferred_clear.framebuffer != framebuffer) {
        const VideoCommon::SyncCounters::ScopedCause cause{VideoCommon::SyncCounters::CauseDeferredClear};
        RealizeDeferredClear();
        EndRenderPass();
    }
    deferred_clear.framebuffer = framebuffer;
    deferred_clear.depth_stencil = true;
    deferred_clear.depth_stencil_value = value;
    return true;
}

void Scheduler::FlushDeferredClear() {
    if (deferred_clear.framebuffer == nullptr) {
        return;
    }
    const VideoCommon::SyncCounters::ScopedCause cause{VideoCommon::SyncCounters::CauseDeferredClear};
    RealizeDeferredClear();
    EndRenderPass();
}

void Scheduler::RequestRenderpass(const Framebuffer* framebuffer) {
    if (deferred_clear.framebuffer == framebuffer) {
        RealizeDeferredClear();
        return;
    }
    const VkRenderPass renderpass = framebuffer->RenderPass();
    const VkFramebuffer framebuffer_handle = framebuffer->Handle();
    const VkExtent2D render_area = framebuffer->RenderArea();
    if (renderpass == state.renderpass && framebuffer_handle == state.framebuffer &&
        render_area.width == state.render_area.width &&
        render_area.height == state.render_area.height) {
        return;
    }
    // Ends any active pass and realizes a deferred clear
    const VideoCommon::SyncCounters::ScopedCause cause{VideoCommon::SyncCounters::CauseFramebuffer};
    EndRenderPass();
    BeginRenderPassImpl(framebuffer, renderpass, nullptr, 0);
}

void Scheduler::RequestOutsideRenderPassOperationContext() {
    EndRenderPass();
}

bool Scheduler::UpdateGraphicsPipeline(GraphicsPipeline* pipeline) {
    if (state.graphics_pipeline == pipeline) {
        if (pipeline && pipeline->UsesExtendedDynamicState() &&
            state.needs_state_enable_refresh) {
            state_tracker.InvalidateStateEnableFlag();
            state.needs_state_enable_refresh = false;
        }
        return false;
    }

    state.graphics_pipeline = pipeline;

    if (!pipeline) {
        return true;
    }

    if (!pipeline->UsesExtendedDynamicState()) {
        state.needs_state_enable_refresh = true;
    } else if (state.needs_state_enable_refresh) {
        state_tracker.InvalidateStateEnableFlag();
        state.needs_state_enable_refresh = false;
    }

    return true;
}

bool Scheduler::UpdateRescaling(bool is_rescaling) {
    if (state.rescaling_defined && is_rescaling == state.is_rescaling) {
        return false;
    }
    state.rescaling_defined = true;
    state.is_rescaling = is_rescaling;
    return true;
}

bool Scheduler::UpdateDescriptorBufferChunk(u32 descriptor_chunk) {
    if (state.descriptor_buffer_bound && descriptor_chunk == state.descriptor_buffer_chunk) {
        return false;
    }
    state.descriptor_buffer_bound = true;
    state.descriptor_buffer_chunk = descriptor_chunk;
    return true;
}

void Scheduler::WorkerThread(std::stop_token stop_token) {
    Common::SetCurrentThreadName("VulkanWorker");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();

    const auto TryPopQueue{[this](auto& work) -> bool {
        if (work_queue.empty()) {
            return false;
        }

        work = std::move(work_queue.front());
        work_queue.pop();
        event_cv.notify_all();
        return true;
    }};

    u64 idle_start{}; ///< Statistics: when the worker last ran out of work
    while (!stop_token.stop_requested()) {
        std::unique_ptr<CommandChunk> work;

        {
            std::unique_lock lk{queue_mutex};

            // Wait for work.
            event_cv.wait(lk, stop_token, [&] {
                worker_waiting = !TryPopQueue(work);
                return !worker_waiting;
            });
            worker_waiting = false;

            // If we've been asked to stop, we're done.
            if (stop_token.stop_requested()) {
                return;
            }

            // Exchange lock ownership so that we take the execution lock before
            // the queue lock goes out of scope. This allows us to force execution
            // to complete in the next step.
            void(std::exchange(lk, std::unique_lock{execution_mutex}));

            // Perform the work, tracking whether the chunk was a submission
            // before executing.
            const bool has_submit = work->HasSubmit();
            const u64 busy_start = VideoCommon::BackStats::Enabled() ? VideoCommon::BackStats::Stamp() : 0;
            if (busy_start != 0) {
                namespace BackStats = VideoCommon::BackStats;
                if (idle_start != 0) {
                    BackStats::Add(BackStats::shared.worker_idle, busy_start - idle_start);
                }
                const u64 frame = BackStats::shared.frame.load(std::memory_order_relaxed);
                BackStats::Max(BackStats::shared.worker_age_max,
                               frame > work->FirstFrame() ? frame - work->FirstFrame() : 0);
            }
            work->ExecuteAll(current_cmdbuf, current_upload_cmdbuf);
            if (busy_start != 0) {
                namespace BackStats = VideoCommon::BackStats;
                idle_start = BackStats::Stamp();
                BackStats::Add(BackStats::shared.worker_busy, idle_start - busy_start);
                BackStats::Add(BackStats::shared.worker_chunks, 1);
            }

            // If the chunk was a submission, reallocate the command buffer.
            if (has_submit) {
                AllocateWorkerCommandBuffer();
            }
        }

        {
            std::scoped_lock rl{reserve_mutex};

            // Recycle the chunk back to the reserve.
            chunk_reserve.emplace_back(std::move(work));
        }
    }
}

void Scheduler::AllocateWorkerCommandBuffer() {
    current_cmdbuf = vk::CommandBuffer(command_pool->Commit(), device.GetDispatchLoader());
    current_cmdbuf.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    current_upload_cmdbuf = vk::CommandBuffer(command_pool->Commit(), device.GetDispatchLoader());
    current_upload_cmdbuf.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
}

u64 Scheduler::SubmitExecution(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore) {
    VideoCommon::SyncCounters::Add(VideoCommon::SyncCounters::Submits);
    {
        const VideoCommon::SyncCounters::ScopedCause cause{VideoCommon::SyncCounters::CauseSubmit};
        EndPendingOperations();
    }
    InvalidateState();

    const u64 signal_value = master_semaphore->NextTick();
    const u64 recorded = VideoCommon::BackStats::Enabled() ? VideoCommon::BackStats::Stamp() : 0;
    RecordWithUploadBuffer([signal_semaphore, wait_semaphore, signal_value, recorded,
                            this](vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf) {
        if (recorded != 0) {
            namespace BackStats = VideoCommon::BackStats;
            const u64 latency = BackStats::Stamp() - recorded;
            BackStats::Add(BackStats::shared.submits, 1);
            BackStats::Add(BackStats::shared.submit_latency, latency);
            BackStats::Max(BackStats::shared.submit_latency_max, latency);
        }
        static constexpr VkMemoryBarrier WRITE_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        upload_cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, WRITE_BARRIER);
        upload_cmdbuf.End();
        cmdbuf.End();

        if (on_submit) {
            on_submit();
        }

        std::scoped_lock lock{submit_mutex};
        switch (const VkResult result = master_semaphore->SubmitQueue(
                    cmdbuf, upload_cmdbuf, signal_semaphore, wait_semaphore, signal_value)) {
        case VK_SUCCESS:
            // Log successful queue submission
            if (GPU::Logging::IsActive() &&
                Settings::values.gpu_log_vulkan_calls.GetValue()) {
                GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
                    "vkQueueSubmit", "", VK_SUCCESS);
            }
            break;
        case VK_ERROR_DEVICE_LOST:
            device.ReportLoss();
            [[fallthrough]];
        default:
            vk::Check(result);
            break;
        }
    });
    chunk->MarkSubmit();
    DispatchWork();
    return signal_value;
}

void Scheduler::AllocateNewContext() {
    // Enable counters once again. These are disabled when a command buffer is finished.
}

void Scheduler::InvalidateState() {
    state.graphics_pipeline = nullptr;
    state.rescaling_defined = false;
    state.descriptor_buffer_bound = false;
    state_tracker.InvalidateCommandBufferState();
}

void Scheduler::EndPendingOperations() {
    query_cache->CounterReset(VideoCommon::QueryType::ZPassPixelCount64);
    EndRenderPass();
}

void Scheduler::EndRenderPass()
    {
        RealizeDeferredClear();
        if (!state.renderpass) {
            return;
        }

        query_cache->CounterClose(VideoCommon::QueryType::StreamingByteCount);

        // Log render pass end
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_vulkan_calls.GetValue()) {
            GPU::Logging::GPULogger::GetInstance().LogRenderPassEnd();
        }

        query_cache->CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, false);
        query_cache->NotifySegment(false);

        // The transform-feedback output barrier only after a pass in which transform feedback was
        // begun (NoteTransformFeedbackActive, from the query cache's vkCmdBeginTransformFeedbackEXT);
        // it used to follow every pass on devices with the extension.
        // EDEN_NX2_XFB_BARRIER_ALWAYS=1 restores that.
        static const bool xfb_barrier_always = [] {
            const char* const value = std::getenv("EDEN_NX2_XFB_BARRIER_ALWAYS");
            return value != nullptr && value[0] != '\0' && value[0] != '0';
        }();
        const bool has_transform_feedback =
            device.IsExtTransformFeedbackSupported() && (xfb_barrier_always || xfb_in_pass);
        xfb_in_pass = false;
        if (VideoCommon::SyncCounters::Enabled()) {
            VideoCommon::SyncCounters::NoteRenderPassEnd();
            if (has_transform_feedback) {
                VideoCommon::SyncCounters::Add(VideoCommon::SyncCounters::XfbBarriers);
            }
        }

        Record([num_images = num_renderpass_images,
                       images = renderpass_images,
                       ranges = renderpass_image_ranges,
                       has_transform_feedback](
                          vk::CommandBuffer cmdbuf) {
            std::array<VkImageMemoryBarrier, 9> barriers;
            for (size_t i = 0; i < num_images; ++i) {
                const VkImageSubresourceRange& range = ranges[i];
                const bool is_color = (range.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) != 0;
                const bool is_depth_stencil = (range.aspectMask
                                              & (VK_IMAGE_ASPECT_DEPTH_BIT
                                                 | VK_IMAGE_ASPECT_STENCIL_BIT)) !=0;

                VkAccessFlags src_access = 0;

                if (is_color)
                    src_access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                else if (is_depth_stencil)
                    src_access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                else
                    src_access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                  | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

                barriers[i] = VkImageMemoryBarrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = src_access,
                        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                                         | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                                         | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                         | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                                         | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = images[i],
                        .subresourceRange = range,
                };
            }
            cmdbuf.EndRenderPass();
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, vk::PIPELINE_STAGE_GRAPHICS_COMPUTE,
                                   0, nullptr, nullptr, vk::Span(barriers.data(), num_images));
            if (has_transform_feedback) {
                static constexpr VkMemoryBarrier XFB_OUTPUT_BARRIER{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT,
                    .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                };
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
                                       VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       0, XFB_OUTPUT_BARRIER);
            }
        });

        state.renderpass = VkRenderPass{};
        num_renderpass_images = 0;
    }


u64 Scheduler::StatsQueued() {
    if (!VideoCommon::BackStats::Enabled()) {
        return 0;
    }
    std::scoped_lock ql{queue_mutex};
    return work_queue.size();
}

void Scheduler::AcquireNewChunk() {
    std::scoped_lock rl{reserve_mutex};

    if (chunk_reserve.empty()) {
        // If we don't have anything reserved, we need to make a new chunk.
        chunk = std::make_unique<CommandChunk>();
    } else {
        // Otherwise, we can just take from the reserve.
        chunk = std::move(chunk_reserve.back());
        chunk_reserve.pop_back();
    }
}

} // namespace Vulkan
