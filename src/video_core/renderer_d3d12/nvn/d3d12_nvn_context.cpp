// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <variant>

#include <d3d12.h>
#include <fmt/format.h>
#include <wrl/client.h>

#include "common/logging.h"
#include "common/thread.h"
#include "core/frontend/emu_window.h"
#include "video_core/nvn/record.h"
#include "video_core/nvn/timestamp_mode.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_d3d12/d3d12_device.h"
#include "video_core/renderer_d3d12/d3d12_swapchain.h"
#include "video_core/renderer_d3d12/nvn/d3d12_nvn_context.h"

namespace D3D12 {

namespace VC = VideoCommon::Nvn;
using Microsoft::WRL::ComPtr;

namespace {

/// Counters of the skeleton, one unsupported-TSV row each ("d3d12 skeleton: <name>").
enum Counter : size_t {
    Presents,
    PresentsHidden,
    Submissions,
    Lists,
    Records,
    Draws,
    Dispatches,
    Copies,
    Clears,
    SyncOps,
    ReportValues,
    ReportCounters,
    OtherOps,
    Pools,
    Textures,
    Samplers,
    Programs,
    Destroys,
    MapVirtuals,
    CpuWrites,
    HostWrites,
    EventCpuWrites,
    TexturesWritten,
    TextureFlushes,
    HolderReports,
    DroppedAfterDeviceLoss,
    BlackScreenshots,
    CounterCount,
};

constexpr std::array COUNTER_NAMES{
    "presents",
    "presents skipped (window hidden)",
    "submissions",
    "lists not executed",
    "records not executed",
    "draws not executed",
    "dispatches not executed",
    "copies not executed",
    "clears not executed",
    "sync and event records not executed",
    "report values written at completion",
    "counter reports written at completion (counter 0)",
    "other records not executed",
    "CreatePool not executed",
    "CreateTexture not executed",
    "CreateSampler not executed",
    "CreateProgram not executed",
    "Destroy not executed",
    "MapVirtual not executed",
    "NotifyCpuWrite not executed",
    "NotifyHostWrite not executed",
    "NotifyEventCpu not executed",
    "NotifyTextureWritten not executed",
    "FlushTextureToGuest not executed",
    "NoteTextureHolders not executed",
    "dropped_after_device_loss",
    "screenshots answered black",
};
static_assert(COUNTER_NAMES.size() == CounterCount);

constexpr auto STATUS_PERIOD = std::chrono::seconds{10};

/// Saturated colour cycling through the hues, one step per present.
std::array<float, 4> PresentColour(u64 count) {
    const float hue = static_cast<float>(count % 360) / 60.0f;
    const int sector = static_cast<int>(hue);
    const float f = hue - static_cast<float>(sector);
    const float v = 0.8f;
    const float p = 0.15f;
    const float q = v - (v - p) * f;
    const float t = p + (v - p) * f;
    switch (sector) {
    case 0:
        return {v, t, p, 1.0f};
    case 1:
        return {q, v, p, 1.0f};
    case 2:
        return {p, v, t, 1.0f};
    case 3:
        return {p, q, v, 1.0f};
    case 4:
        return {t, p, v, 1.0f};
    default:
        return {v, p, q, 1.0f};
    }
}

/// NVNcounterType values with a time in the report (renderer_vulkan/nvn/nvn_counters.h).
constexpr u32 COUNTER_TIMESTAMP = 0x0;
constexpr u32 COUNTER_TIMESTAMP_TOP = 0x10;

/// A guest report the GPU would write, written at completion in record order instead.
struct ReportWrite {
    u64 address;
    u32 value;   ///< ReportValue: the 32-bit value
    bool counter; ///< ReportCounter: NVNcounterData { u64 counter = 0; u64 timestamp; }
};

/// The timestamp field of a counter report written now, as the Vulkan sink's CounterEngine
/// formats it: the guest GPU clock in legacy mode, else the current guest time (the skeleton has
/// no GPU time to map; the guest clock is monotonic).
u64 ReportTimestampField() {
    if (VC::CurrentTimestampMode() == VC::TimestampMode::Legacy) {
        return VC::LegacyReportField();
    }
    return VC::GuestNanosecondsToReportField(VC::GuestClockNanoseconds());
}

/// A Submit or Present the GPU has to pass before its completion callback runs.
struct Flight {
    u64 seq{};
    u64 fence_value{}; ///< 0: nothing on the GPU (device lost), completes at once
    std::vector<ReportWrite> reports;
};

using Command = std::variant<VC::Submission, VC::PresentRequest>;

class ScopedEvent {
public:
    ScopedEvent() : handle{CreateEventW(nullptr, FALSE, FALSE, nullptr)} {
        if (handle == nullptr) {
            throw std::runtime_error{"CreateEventW failed"};
        }
    }
    ~ScopedEvent() {
        CloseHandle(handle);
    }
    ScopedEvent(const ScopedEvent&) = delete;
    ScopedEvent& operator=(const ScopedEvent&) = delete;
    [[nodiscard]] HANDLE Get() const {
        return handle;
    }

private:
    HANDLE handle;
};

} // Anonymous namespace

struct NvnContext::Impl {
    Impl(Device& device_, SwapChain& swap_chain_, Core::Frontend::EmuWindow& render_window_,
         VideoCore::RendererSettings& renderer_settings_)
        : device{device_}, swap_chain{swap_chain_}, render_window{render_window_},
          renderer_settings{renderer_settings_} {
        ID3D12Device* const d3d = device.Get();
        if (FAILED(d3d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) {
            throw std::runtime_error{"CreateFence failed"};
        }
        for (auto& allocator : allocators) {
            if (FAILED(d3d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                   IID_PPV_ARGS(&allocator)))) {
                throw std::runtime_error{"CreateCommandAllocator failed"};
            }
        }
        if (FAILED(d3d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocators[0].Get(),
                                          nullptr, IID_PPV_ARGS(&command_list)))) {
            throw std::runtime_error{"CreateCommandList failed"};
        }
        command_list->Close();
        last_status = std::chrono::steady_clock::now();
        replay_thread = std::thread{[this] { ReplayLoop(); }};
        try {
            completion_thread = std::thread{[this] { CompletionLoop(); }};
        } catch (...) {
            // The destructor does not run for a throwing constructor: stop the replay thread
            // here, or its std::thread would terminate the process.
            {
                std::scoped_lock lock{queue_mutex};
                stopping.store(true, std::memory_order_release);
            }
            queue_cv.notify_all();
            replay_thread.join();
            throw;
        }
    }

    ~Impl() {
        {
            std::scoped_lock lock{queue_mutex, flight_mutex};
            stopping.store(true, std::memory_order_release);
        }
        queue_cv.notify_all();
        flight_cv.notify_all();
        replay_thread.join();
        completion_thread.join();
        // The back buffers and allocators must be idle before the swap chain and lists go.
        WaitIdle();
    }

    void Count(Counter counter, u64 amount = 1) {
        counters[counter].fetch_add(amount, std::memory_order_relaxed);
    }
    [[nodiscard]] u64 Value(Counter counter) const {
        return counters[counter].load(std::memory_order_relaxed);
    }

    void Enqueue(Command&& command) {
        {
            std::scoped_lock lock{queue_mutex};
            queue.push_back(std::move(command));
            ++commands_enqueued;
        }
        queue_cv.notify_one();
    }

    // ---- replay thread -------------------------------------------------------------------------

    void ReplayLoop() {
        Common::SetCurrentThreadName("NvnD3D12Replay");
        while (true) {
            std::optional<Command> command;
            {
                std::unique_lock lock{queue_mutex};
                queue_cv.wait_for(lock, std::chrono::seconds{1}, [this] {
                    return stopping.load(std::memory_order_acquire) || !queue.empty();
                });
                if (stopping.load(std::memory_order_acquire)) {
                    return;
                }
                if (!queue.empty()) {
                    command.emplace(std::move(queue.front()));
                    queue.pop_front();
                }
            }
            if (command) {
                if (auto* submission = std::get_if<VC::Submission>(&*command)) {
                    ExecuteSubmit(*submission);
                } else {
                    ExecutePresent(std::get<VC::PresentRequest>(*command));
                }
                commands_dispatched.fetch_add(1, std::memory_order_release);
            }
            LogStatusIfDue();
        }
    }

    void ExecuteSubmit(const VC::Submission& submission) {
        Count(Submissions);
        Flight flight{.seq = submission.seq};
        for (const std::vector<u64>& list : submission.lists) {
            Count(Lists);
            VC::RecordReader reader{list.data(), list.size()};
            while (const VC::RecordHeader* header = reader.Next()) {
                Count(Records);
                switch (VC::RecordOp(header)) {
                case VC::Op::Draw:
                case VC::Op::DrawIndexed:
                case VC::Op::DrawTexture:
                case VC::Op::DrawTransformFeedback:
                case VC::Op::DrawMeshTasks:
                    Count(Draws);
                    break;
                case VC::Op::Dispatch:
                    Count(Dispatches);
                    break;
                case VC::Op::CopyBufferToBuffer:
                case VC::Op::CopyBufferToTexture:
                case VC::Op::CopyTextureToBuffer:
                case VC::Op::CopyTextureToTexture:
                    Count(Copies);
                    break;
                case VC::Op::ClearTexture:
                case VC::Op::ClearBuffer:
                case VC::Op::ClearColor:
                case VC::Op::ClearDepthStencil:
                    Count(Clears);
                    break;
                case VC::Op::FenceSync:
                case VC::Op::SignalEvent:
                case VC::Op::WaitSync:
                case VC::Op::WaitEvent:
                    Count(SyncOps);
                    break;
                case VC::Op::ReportValue:
                    // A semaphore-style release the game may poll: written at completion so a
                    // CPU wait on it ends (the Vulkan sink writes it on the GPU timeline).
                    if (VC::RecordPayloadSize(header) >= sizeof(VC::ReportValuePayload)) {
                        const auto* payload = VC::RecordPayload<VC::ReportValuePayload>(header);
                        flight.reports.push_back({payload->address, payload->value, false});
                    }
                    Count(OtherOps);
                    break;
                case VC::Op::ReportCounter:
                    // Counters (samples passed, ...) read 0, timestamps a plausible time, as the
                    // Vulkan sink writes the 16-byte NVNcounterData on the GPU timeline.
                    if (VC::RecordPayloadSize(header) >= sizeof(VC::ReportCounterPayload)) {
                        const auto* payload =
                            VC::RecordPayload<VC::ReportCounterPayload>(header);
                        flight.reports.push_back({payload->address, 0, true});
                    }
                    Count(OtherOps);
                    break;
                default:
                    Count(OtherOps);
                    break;
                }
            }
        }
        flight.fence_value = SignalQueue("Signal (submit)");
        if (flight.fence_value == 0) {
            Count(DroppedAfterDeviceLoss);
        }
        PushFlight(std::move(flight));
    }

    void ExecutePresent(const VC::PresentRequest& request) {
        const u64 present_count = counters[Presents].fetch_add(1, std::memory_order_relaxed);
        Flight flight{.seq = request.seq};
        if (device.IsRemoved()) {
            Count(DroppedAfterDeviceLoss);
        } else if (!render_window.IsShown()) {
            Count(PresentsHidden);
        } else {
            DrawAndPresent(present_count, request.present_interval);
            render_window.OnFrameDisplayed();
        }
        AnswerScreenshot();
        flight.fence_value = SignalQueue("Signal (present)");
        PushFlight(std::move(flight));
    }

    void DrawAndPresent(u64 present_count, u32 interval) {
        const Layout::FramebufferLayout& layout = render_window.GetFramebufferLayout();
        if (layout.width != 0 && layout.height != 0 &&
            (layout.width != swap_chain.Width() || layout.height != swap_chain.Height())) {
            WaitIdle();
            if (!swap_chain.Resize(layout.width, layout.height)) {
                CheckLost();
                return;
            }
        }
        const u32 index = swap_chain.CurrentIndex();
        WaitFence(allocator_fence[index]);
        ID3D12Resource* const back_buffer = swap_chain.BackBuffer(index);
        if (back_buffer == nullptr) {
            return; // a failed resize left no buffers; the next present tries again
        }
        ID3D12CommandAllocator* const allocator = allocators[index].Get();
        if (!device.Check(allocator->Reset(), "ID3D12CommandAllocator::Reset") ||
            !device.Check(command_list->Reset(allocator, nullptr),
                          "ID3D12GraphicsCommandList::Reset")) {
            CheckLost();
            return;
        }
        D3D12_RESOURCE_BARRIER barrier{
            .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            .Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
            .Transition =
                {
                    .pResource = back_buffer,
                    .Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    .StateBefore = D3D12_RESOURCE_STATE_PRESENT,
                    .StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET,
                },
        };
        command_list->ResourceBarrier(1, &barrier);
        const std::array<float, 4> colour = PresentColour(present_count);
        command_list->ClearRenderTargetView(swap_chain.Rtv(index), colour.data(), 0, nullptr);
        std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        command_list->ResourceBarrier(1, &barrier);
        if (!device.Check(command_list->Close(), "ID3D12GraphicsCommandList::Close")) {
            CheckLost();
            return;
        }
        ID3D12CommandList* const lists[] = {command_list.Get()};
        device.DirectQueue()->ExecuteCommandLists(1, lists);
        if (!device.Check(swap_chain.Present(interval), "IDXGISwapChain::Present")) {
            CheckLost();
        }
        allocator_fence[index] = SignalQueue("Signal (frame)");
    }

    /// nvn-hle presents serve screenshot requests; the skeleton has no picture to read back, so
    /// it answers with a black frame instead of leaving the request pending forever.
    void AnswerScreenshot() {
        if (!renderer_settings.ClaimScreenshot()) {
            return;
        }
        const Layout::FramebufferLayout& layout = renderer_settings.screenshot_framebuffer_layout;
        if (renderer_settings.screenshot_bits != nullptr) {
            std::memset(renderer_settings.screenshot_bits, 0,
                        static_cast<size_t>(layout.width) * layout.height * 4);
        }
        Count(BlackScreenshots);
        renderer_settings.screenshot_complete_callback(false);
        renderer_settings.FinishScreenshot();
    }

    /// Signals the next fence value on the direct queue; 0 when the device is lost.
    u64 SignalQueue(const char* what) {
        if (device.IsRemoved()) {
            CheckLost(); // the completion thread may have seen the removal first
            return 0;
        }
        const u64 value = ++fence_value;
        if (!device.Check(device.DirectQueue()->Signal(fence.Get(), value), what)) {
            CheckLost();
            return 0;
        }
        return value;
    }

    /// Replay thread: blocks until the fence reaches `value` (or the device is lost).
    void WaitFence(u64 value) {
        if (value == 0 || fence->GetCompletedValue() >= value) {
            return;
        }
        if (FAILED(fence->SetEventOnCompletion(value, replay_event.Get()))) {
            return;
        }
        while (WaitForSingleObject(replay_event.Get(), 100) == WAIT_TIMEOUT) {
            if (fence->GetCompletedValue() == UINT64_MAX) { // removed device
                device.Check(DXGI_ERROR_DEVICE_REMOVED, "fence wait (replay)");
                CheckLost();
                return;
            }
        }
    }

    void WaitIdle() {
        WaitFence(SignalQueue("Signal (idle)"));
    }

    void PushFlight(Flight&& flight) {
        {
            std::scoped_lock lock{flight_mutex};
            flights.push_back(std::move(flight));
        }
        flight_cv.notify_one();
    }

    /// Once the device is gone: tell the owner once (it stops the emulation); every later
    /// submission completes without a GPU fence (DESIGN.md device-lost mode).
    void CheckLost() {
        // Not while stopping: ~Impl's idle wait runs on the destroying thread, which must not
        // call back into the owner.
        if (!device.IsRemoved() || stopping.load(std::memory_order_acquire) ||
            failure_reported.exchange(true)) {
            return;
        }
        std::scoped_lock lock{failure_mutex};
        if (failure_callback) {
            failure_callback("the D3D12 device was removed");
        }
    }

    void LogStatusIfDue() {
        const auto now = std::chrono::steady_clock::now();
        if (now - last_status < STATUS_PERIOD) {
            return;
        }
        last_status = now;
        LOG_INFO(Render,
                 "[nvn] d3d12 skeleton: presents {} submissions {}, not executed: draws {} "
                 "dispatches {} copies {} clears {} other records {}, creators: pools {} "
                 "textures {} samplers {} programs {} destroys {}",
                 Value(Presents), Value(Submissions), Value(Draws), Value(Dispatches),
                 Value(Copies), Value(Clears), Value(OtherOps) + Value(SyncOps), Value(Pools),
                 Value(Textures), Value(Samplers), Value(Programs), Value(Destroys));
    }

    // ---- completion thread ---------------------------------------------------------------------

    void CompletionLoop() {
        Common::SetCurrentThreadName("NvnD3D12Complete");
        while (true) {
            Flight flight;
            {
                std::unique_lock lock{flight_mutex};
                flight_cv.wait(lock, [this] {
                    return stopping.load(std::memory_order_acquire) || !flights.empty();
                });
                if (stopping.load(std::memory_order_acquire)) {
                    return;
                }
                flight = std::move(flights.front());
                flights.pop_front();
                ++in_flight_taken;
            }
            if (!WaitCompletion(flight.fence_value)) {
                return; // stopping
            }
            {
                // Held while the callback runs: SetCompletionCallback({}) waits for a running
                // callback, after which neither the callback nor guest memory is touched again.
                std::scoped_lock callback_lock{callback_mutex};
                if (completion_callback && !stopping.load(std::memory_order_acquire)) {
                    if (VC::GuestMemory* memory = guest.load(std::memory_order_acquire)) {
                        WriteReports(*memory, flight.reports);
                    }
                    completion_callback(flight.seq);
                }
            }
            {
                std::scoped_lock lock{wait_mutex};
                if (flight.seq > completed_seq.load(std::memory_order_relaxed)) {
                    completed_seq.store(flight.seq, std::memory_order_release);
                }
            }
            wait_cv.notify_all();
        }
    }

    void WriteReports(VC::GuestMemory& memory, const std::vector<ReportWrite>& reports) {
        for (const ReportWrite& report : reports) {
            if (report.counter) {
                const std::array<u64, 2> data{0, ReportTimestampField()};
                memory.Write(report.address, data.data(), sizeof(data));
                Count(ReportCounters);
            } else {
                memory.Write(report.address, &report.value, sizeof(report.value));
                Count(ReportValues);
            }
        }
    }

    /// False when the context is stopping before the fence was reached.
    bool WaitCompletion(u64 value) {
        if (value == 0) {
            return true;
        }
        while (true) {
            const u64 completed = fence->GetCompletedValue();
            if (completed >= value) {
                if (completed == UINT64_MAX) {
                    device.Check(DXGI_ERROR_DEVICE_REMOVED, "fence wait (completion)");
                }
                return true;
            }
            if (stopping.load(std::memory_order_acquire)) {
                return false;
            }
            if (FAILED(fence->SetEventOnCompletion(value, completion_event.Get()))) {
                return true;
            }
            WaitForSingleObject(completion_event.Get(), 100);
        }
    }

    Device& device;
    SwapChain& swap_chain;
    Core::Frontend::EmuWindow& render_window;
    VideoCore::RendererSettings& renderer_settings;

    ComPtr<ID3D12Fence> fence;
    std::array<ComPtr<ID3D12CommandAllocator>, SwapChain::BUFFER_COUNT> allocators;
    std::array<u64, SwapChain::BUFFER_COUNT> allocator_fence{};
    ComPtr<ID3D12GraphicsCommandList> command_list;
    u64 fence_value{}; ///< replay thread only
    ScopedEvent replay_event;
    ScopedEvent completion_event;

    std::array<std::atomic<u64>, CounterCount> counters{};
    std::chrono::steady_clock::time_point last_status;

    std::atomic<VC::GuestMemory*> guest{nullptr};
    std::atomic<u32> window_origin{};
    std::atomic<u32> depth_mode{};
    std::atomic<u64> title_id{};

    std::atomic<bool> stopping{false};
    std::atomic<bool> failure_reported{false};

    mutable std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::deque<Command> queue;
    u64 commands_enqueued{};
    std::atomic<u64> commands_dispatched{};

    mutable std::mutex flight_mutex;
    std::condition_variable flight_cv;
    std::deque<Flight> flights;
    u64 in_flight_taken{};

    std::mutex callback_mutex;
    std::function<void(u64)> completion_callback;
    std::mutex failure_mutex;
    std::function<void(std::string)> failure_callback;

    std::mutex wait_mutex;
    std::condition_variable wait_cv;
    std::atomic<u64> completed_seq{};

    std::thread replay_thread;
    std::thread completion_thread;
};

NvnContext::NvnContext(Device& device, SwapChain& swap_chain,
                       Core::Frontend::EmuWindow& render_window,
                       VideoCore::RendererSettings& renderer_settings)
    : impl{std::make_unique<Impl>(device, swap_chain, render_window, renderer_settings)} {
    LOG_INFO(Render, "[nvn] d3d12 skeleton sink created: nothing the game records is executed");
}

NvnContext::~NvnContext() = default;

void NvnContext::CreatePool(const VC::PoolDesc& desc) {
    static_cast<void>(desc);
    impl->Count(Pools);
}

void NvnContext::CreateTexture(const VC::TextureDesc& desc) {
    static_cast<void>(desc);
    impl->Count(Textures);
}

void NvnContext::CreateSampler(const VC::SamplerDesc& desc) {
    static_cast<void>(desc);
    impl->Count(Samplers);
}

void NvnContext::CreateProgram(const VC::ProgramDesc& desc) {
    static_cast<void>(desc);
    impl->Count(Programs);
}

void NvnContext::Destroy(u64 serial) {
    static_cast<void>(serial);
    impl->Count(Destroys);
}

void NvnContext::Submit(VC::Submission&& submission) {
    impl->Enqueue(Command{std::in_place_type<VC::Submission>, std::move(submission)});
}

void NvnContext::Present(const VC::PresentRequest& request) {
    impl->Enqueue(Command{std::in_place_type<VC::PresentRequest>, request});
}

bool NvnContext::Wait(u64 seq, std::chrono::nanoseconds timeout) {
    std::unique_lock lock{impl->wait_mutex};
    return impl->wait_cv.wait_for(lock, timeout, [this, seq] {
        return impl->completed_seq.load(std::memory_order_acquire) >= seq;
    });
}

void NvnContext::SetCompletionCallback(std::function<void(u64 seq)> callback) {
    std::scoped_lock lock{impl->callback_mutex};
    impl->completion_callback = std::move(callback);
}

void NvnContext::SetFailureCallback(std::function<void(std::string reason)> callback) {
    std::scoped_lock lock{impl->failure_mutex};
    impl->failure_callback = std::move(callback);
}

void NvnContext::SetGuestMemory(VC::GuestMemory* memory) {
    // Cleared by the gate's Detach after SetCompletionCallback({}), so no completion writes
    // through a stale pointer (the writes happen under callback_mutex with a live callback).
    impl->guest.store(memory, std::memory_order_release);
}

void NvnContext::SetDeviceModes(u32 window_origin, u32 depth_mode) {
    impl->window_origin.store(window_origin, std::memory_order_relaxed);
    impl->depth_mode.store(depth_mode, std::memory_order_relaxed);
}

void NvnContext::SetTitleId(u64 title_id) {
    impl->title_id.store(title_id, std::memory_order_relaxed);
    LOG_INFO(Render, "[nvn] d3d12 skeleton: title {:016X}", title_id);
}

void NvnContext::CollectStats(std::vector<VC::SinkStat>& out) const {
    for (size_t index = 0; index < CounterCount; ++index) {
        out.push_back({fmt::format("d3d12 skeleton: {}", COUNTER_NAMES[index]),
                       impl->Value(static_cast<Counter>(index))});
    }
    // Read by the runtime's status line.
    out.push_back({"ops_unsupported", impl->Value(Records)});
}

void NvnContext::MapVirtual(u64 pool_serial, std::vector<VC::VirtualMapping>&& mappings) {
    static_cast<void>(pool_serial);
    static_cast<void>(mappings);
    impl->Count(MapVirtuals);
}

void NvnContext::NotifyCpuWrite(u64 guest_address, u64 size) {
    static_cast<void>(guest_address);
    static_cast<void>(size);
    impl->Count(CpuWrites);
}

void NvnContext::NotifyHostWrite(u64 guest_address, u64 size) {
    static_cast<void>(guest_address);
    static_cast<void>(size);
    impl->Count(HostWrites);
}

void NvnContext::NotifyEventCpu(u64 address, VC::EventCpuChange change, u32 value) {
    // No event model: the skeleton never waits on an event, and the recorder writes the guest
    // words itself at completion.
    static_cast<void>(address);
    static_cast<void>(change);
    static_cast<void>(value);
    impl->Count(EventCpuWrites);
}

void NvnContext::NotifyTextureWritten(u64 texture_serial) {
    static_cast<void>(texture_serial);
    impl->Count(TexturesWritten);
}

bool NvnContext::FlushTextureToGuest(u64 texture_serial) {
    // The GPU never writes a texture here: the guest bytes are as current as they get.
    static_cast<void>(texture_serial);
    impl->Count(TextureFlushes);
    return true;
}

std::string NvnContext::DescribeState() const {
    size_t queued;
    u64 enqueued;
    {
        std::scoped_lock lock{impl->queue_mutex};
        queued = impl->queue.size();
        enqueued = impl->commands_enqueued;
    }
    size_t waiting;
    {
        std::scoped_lock lock{impl->flight_mutex};
        waiting = impl->flights.size();
    }
    return fmt::format("d3d12 skeleton: commands enqueued {} dispatched {} queued {}, flights "
                       "waiting {}, completed seq {}, device {}",
                       enqueued, impl->commands_dispatched.load(std::memory_order_acquire),
                       queued, waiting, impl->completed_seq.load(std::memory_order_acquire),
                       impl->device.IsRemoved() ? "removed" : "ok");
}

std::string NvnContext::DescribeMemory() const {
    return "d3d12 skeleton: no guest resources created";
}

void NvnContext::NoteTextureHolders(VC::TextureHolderReport&& report) {
    static_cast<void>(report);
    impl->Count(HolderReports);
}

} // namespace D3D12
