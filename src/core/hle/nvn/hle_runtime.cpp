// SPDX-FileCopyrightText: Copyright 2026 n2den Contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>
#include <cstring>
#include <map>
#include <string>
#include <limits>
#include <thread>

#include <fmt/format.h>

#include "common/boot_counter.h"
#include "common/frame_stats.h"
#include "common/game_presets.h"
#include "common/logging.h"
#include "common/memory_census.h"
#include "common/tl_ring.h"
#include "core/arm/debug.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_hardware_timer.h"
#include "core/hle/kernel/k_scheduler.h"
#include "core/hle/kernel/k_scoped_scheduler_lock_and_sleep.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/k_thread_queue.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/nvn/function_table.h"
#include "core/hle/nvn/hle_runtime.h"
#include "core/hle/nvn/package_shaders.h"
#include "core/file_sys/common_funcs.h"
#include "core/file_sys/romfs.h"
#include "core/hle/service/filesystem/filesystem.h"
#include "core/hle/service/filesystem/romfs_controller.h"
#include "core/hle/service/filesystem/save_data_controller.h"
#include "core/memory.h"
#include "core/memory/page_tracker.h"
#include "video_core/gpu.h"
#include "video_core/nvn/prerec_probe.h"
#include "video_core/nvn/sink.h"
#include "video_core/nvn/timestamp_mode.h"
#include "video_core/renderer_base.h"

namespace Core::NvnHle::Hle {

namespace {

/// The pipeline cache's prepare step (ahead-of-time pipelines, stage 3) reads the shader stages of
/// the title's own RomFS before the first frame when its state list needs them. Read only.
void InstallPackageStageSource(Core::System& system) {
    const Kernel::KProcess* const process = system.ApplicationProcess();
    if (process == nullptr) {
        return;
    }
    const u64 process_id = process->GetProcessId();
    VideoCommon::Nvn::SetPackageStageSource(
        system.GetApplicationProcessProgramID(),
        [&system, process_id](const VideoCommon::Nvn::PackageStageVisitor& visit) {
            u64 program_id{};
            std::shared_ptr<Service::FileSystem::SaveDataController> save_data;
            std::shared_ptr<Service::FileSystem::RomFsController> romfs;
            if (system.GetFileSystemController().OpenProcess(&program_id, &save_data, &romfs,
                                                             process_id)
                    .IsError() ||
                !romfs) {
                return std::string{"no RomFS registered for the process"};
            }
            const FileSys::VirtualFile file = romfs->OpenRomFSCurrentProcess();
            const FileSys::VirtualDir root = file ? FileSys::ExtractRomFS(file) : nullptr;
            if (!root) {
                return std::string{"RomFS not readable"};
            }
            std::string summary = ScanPackageShaders(root, visit).Summary();
            // Add-on packages the loader registered (Switch 2 Editions keep their Switch 2 data,
            // shaders included, in an add-on: Kirby and the Forgotten Land found 0 of 1183 stages
            // in the base RomFS alone).
            auto& fsc = system.GetFileSystemController();
            // Registered under the program's base id; the RomFS controller may report the
            // application id instead (an .nx whose application id differs from the program id).
            std::vector<u64> add_ons =
                fsc.ListLooseAddOnContent(FileSys::GetBaseTitleID(program_id));
            if (add_ons.empty()) {
                add_ons = fsc.ListLooseAddOnContent(
                    FileSys::GetBaseTitleID(system.GetApplicationProcessProgramID()));
            }
            for (const u64 add_on : add_ons) {
                const FileSys::VirtualFile add_on_file = fsc.OpenLooseAddOnContent(add_on);
                const FileSys::VirtualDir add_on_root =
                    add_on_file ? FileSys::ExtractRomFS(add_on_file) : nullptr;
                if (add_on_root) {
                    summary += fmt::format("; add-on {:016X}: {}", add_on,
                                           ScanPackageShaders(add_on_root, visit).Summary());
                }
            }
            return summary;
        });
}

/// Queue of a guest thread asleep for an NVN wait. The base class' EndWait / CancelWait make the
/// thread runnable again and cancel its timer task.
class NvnWaitQueue final : public Kernel::KThreadQueue {
public:
    explicit NvnWaitQueue(Kernel::KernelCore& kernel) : KThreadQueue(kernel) {}
};

/// EDEN_NX2_NVN_GPU_PROFILE is set (read once): the heartbeat also reports where the guest blocked.
bool GpuProfileOn() {
    static const bool on = [] {
        const char* value = std::getenv("EDEN_NX2_NVN_GPU_PROFILE");
        return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
    }();
    return on;
}

/// EDEN_NX2_NVN_TIMELINE is set (read once), like the replay's Debug::Timeline().
bool TimelineOn() {
    static const bool on = [] {
        const char* value = std::getenv("EDEN_NX2_NVN_TIMELINE");
        return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
    }();
    return on;
}

} // namespace

/// GuestMemory over the application process' memory. Reads and writes skip the rasterizer's
/// cache flushing: the HLE has no legacy GPU state on this memory.
class Runtime::ProcessMemory final : public VideoCommon::Nvn::GuestMemory {
public:
    /// `guest_view`, `guest_view_size`: the fastmem arena (guest VA 0 and its size), null when
    /// the host has none.
    ProcessMemory(Core::Memory::Memory& memory_, const std::atomic<bool>& stopped_,
                  const u8* guest_view_, u64 guest_view_size_)
        : memory{memory_}, stopped{stopped_}, guest_view{guest_view_},
          guest_view_size{guest_view_size_} {}

    bool Read(u64 va, void* dst, size_t size) override {
        return !stopped.load(std::memory_order_relaxed) && memory.ReadBlockUnsafe(va, dst, size);
    }
    bool Write(u64 va, const void* src, size_t size) override {
        if (stopped.load(std::memory_order_relaxed)) {
            return false;
        }
        // The stores of the HLE itself (object tags, results, write-back) skip the page tracker,
        // so the access trap of the import gate does not see them: they wait here.
        gate->HoldWrite(va, size);
        return memory.WriteBlockUntracked(va, src, size);
    }
    u8* Pointer(u64 va) override {
        return stopped.load(std::memory_order_relaxed) ? nullptr : memory.GetPointerSilent(va);
    }
    bool Stopped() const override {
        return stopped.load(std::memory_order_acquire);
    }
    const u8* GuestViewPointer(u64 va) override {
        // Where JIT code reads and writes guest `va` (Memory::Impl maps the arena at the guest
        // address). Only ever asked about, never dereferenced.
        return guest_view != nullptr && va < guest_view_size ? guest_view + va : nullptr;
    }

    u64 Track(u64 va, u64 size) override {
        return stopped.load(std::memory_order_relaxed) ? 0
                                                       : memory.PageTracker().Register(va, size, false);
    }
    void Untrack(u64 id) override {
        memory.PageTracker().Unregister(id);
    }
    void Arm(u64 id, u64 va, u64 size) override {
        memory.PageTracker().Arm(id, va, size);
    }
    void TakeWritten(u64 id, u64 va, u64 size, std::vector<std::pair<u64, u64>>& runs) override {
        memory.PageTracker().CollectDirty(id, va, size, runs);
    }
    bool AnyWritten(u64 id, u64 va, u64 size) override {
        return memory.PageTracker().AnyDirty(id, va, size);
    }
    std::shared_ptr<const std::atomic<u64>> WriteEpoch(u64 id) override {
        return memory.PageTracker().WriteEpoch(id);
    }
    bool SupportsAccessTraps() const override {
        return true;
    }
    void SetAccessCallback(u64 id, std::function<void(u64 va, u64 size)> callback) override {
        memory.PageTracker().SetAccessCallback(id, std::move(callback));
    }
    void ArmAccessTrap(u64 id, u64 va, u64 size) override {
        memory.PageTracker().ArmAccessTrap(id, va, size);
    }
    void DisarmAccessTrap(u64 id, u64 va, u64 size) override {
        memory.PageTracker().DisarmAccessTrap(id, va, size);
    }

    u64 BeginImportGate(u64 va, u64 size) override {
        if (size == 0 || stopped.load(std::memory_order_relaxed)) {
            return 0;
        }
        // The gate first (the HLE writes wait from here on), then the trap for the guest.
        const u64 token = gate->Open(va, size);
        Core::Memory::PageTracker& tracker = memory.PageTracker();
        const u64 track_id = tracker.Register(va, size, false);
        if (track_id != 0) {
            // Runs on the thread whose read or write hit a page of the range, before the access
            // lands, with no tracker lock held. It waits for the gate alone; the gate state
            // outlives the registration (a hit collected just before EndImportGate still calls).
            tracker.SetAccessCallback(track_id, [state = gate, token](u64, u64) {
                state->HoldAccess(token);
            });
            tracker.ArmAccessTrap(track_id, va, size);
        }
        gate->SetTrackId(token, track_id);
        return token;
    }
    void EndImportGate(u64 token) override {
        if (token == 0) {
            return;
        }
        // Waiters go first; the registration (and with it the page protection) after them.
        const u64 track_id = gate->Close(token);
        if (track_id != 0) {
            memory.PageTracker().Unregister(track_id);
        }
    }

private:
    /// The imports in flight (BeginImportGate .. EndImportGate) and who waits for them. A leaf:
    /// its mutex is never held across a call into anything else, and a waiter waits for nothing
    /// but the EndImportGate of the importing thread, which that thread reaches straight after
    /// its vkAllocateMemory call.
    class ImportGate {
    public:
        u64 Open(u64 va, u64 size) {
            std::scoped_lock lock{mutex};
            const u64 token = next_token++;
            active.push_back({token, va, va + size, 0, std::this_thread::get_id()});
            active_count.store(active.size(), std::memory_order_release);
            bytes += size;
            return token;
        }
        void SetTrackId(u64 token, u64 track_id) {
            std::scoped_lock lock{mutex};
            for (Active& entry : active) {
                if (entry.token == token) {
                    entry.track_id = track_id;
                }
            }
        }
        /// Returns the tracker registration of the gate (0: none).
        u64 Close(u64 token) {
            u64 track_id = 0;
            {
                std::scoped_lock lock{mutex};
                std::erase_if(active, [&](const Active& entry) {
                    if (entry.token != token) {
                        return false;
                    }
                    track_id = entry.track_id;
                    return true;
                });
                active_count.store(active.size(), std::memory_order_release);
                ++imports;
                if (track_id == 0) {
                    ++untracked;
                }
                // 1, 2, 4, ... and every 256th: enough to see the gate work without flooding.
                if (std::has_single_bit(imports) || imports % 256 == 0) {
                    LOG_INFO(Core,
                             "[nvn] import gate: {} imports gated ({} MiB, {} without a page "
                             "tracker registration); held {} guest accesses and {} HLE writes "
                             "for {:.1f} ms in all (longest {:.1f} ms), {} gave up after {} ms",
                             imports, bytes >> 20, untracked, held_accesses, held_writes,
                             static_cast<double>(held_ns) / 1e6,
                             static_cast<double>(longest_ns) / 1e6, timeouts, TIMEOUT_MS);
                }
            }
            released.notify_all();
            return track_id;
        }
        /// A guest access hit the trap of import `token`: wait until that import is over.
        void HoldAccess(u64 token) {
            if (active_count.load(std::memory_order_acquire) == 0) {
                return;
            }
            const std::thread::id self = std::this_thread::get_id();
            Hold(held_accesses, [&] {
                for (const Active& entry : active) {
                    if (entry.token == token && entry.importer != self) {
                        return true;
                    }
                }
                return false;
            });
        }
        /// The HLE is about to write [va, va + size): wait while an import covers any of it.
        void HoldWrite(u64 va, u64 size) {
            if (size == 0 || active_count.load(std::memory_order_acquire) == 0) [[likely]] {
                return;
            }
            const std::thread::id self = std::this_thread::get_id();
            Hold(held_writes, [&] {
                for (const Active& entry : active) {
                    if (va < entry.end && entry.begin < va + size && entry.importer != self) {
                        return true;
                    }
                }
                return false;
            });
        }

    private:
        /// A stuck import must not hang the guest for good: after this long a waiter goes on
        /// (and risks the lost store the gate is there to prevent).
        static constexpr u64 TIMEOUT_MS = 2000;

        struct Active {
            u64 token;
            u64 begin;
            u64 end;
            u64 track_id;
            std::thread::id importer; ///< never waits for its own gate
        };

        /// Waits while `blocked()` (called with `mutex` held) says so.
        template <typename Blocked>
        void Hold(u64& counter, Blocked&& blocked) {
            std::unique_lock lock{mutex};
            if (!blocked()) {
                return;
            }
            ++counter;
            const auto begin = std::chrono::steady_clock::now();
            const bool done = released.wait_for(lock, std::chrono::milliseconds{TIMEOUT_MS},
                                                [&] { return !blocked(); });
            const u64 ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                std::chrono::steady_clock::now() - begin)
                                                .count());
            held_ns += ns;
            longest_ns = (std::max)(longest_ns, ns);
            if (!done) {
                ++timeouts;
                LOG_WARNING(Core, "[nvn] import gate: a waiter gave up after {} ms", TIMEOUT_MS);
            }
        }

        std::mutex mutex;
        std::condition_variable released;
        std::vector<Active> active;
        std::atomic<size_t> active_count{};
        u64 next_token{1};
        // Counters, under `mutex`.
        u64 imports{};
        u64 untracked{};
        u64 bytes{};
        u64 held_accesses{};
        u64 held_writes{};
        u64 held_ns{};
        u64 longest_ns{};
        u64 timeouts{};
    };

    Core::Memory::Memory& memory;
    const std::atomic<bool>& stopped;
    const u8* guest_view;
    u64 guest_view_size;
    std::shared_ptr<ImportGate> gate{std::make_shared<ImportGate>()};
};

Call::Call(Runtime& runtime_, Core::System* system_, Kernel::Svc::ThreadContext& ctx_,
           VideoCommon::Nvn::GuestMemory& memory, VideoCommon::Nvn::Recorder& recorder)
    : runtime{runtime_}, system{system_}, ctx{ctx_}, mem{memory}, rec{recorder}, args{ctx_, memory},
      ret{ctx_, memory} {}

bool Call::Zero(u64 va, size_t size) {
    static constexpr std::array<u8, 256> zeros{};
    while (size != 0) {
        const size_t chunk = std::min(size, zeros.size());
        if (!mem.Write(va, zeros.data(), chunk)) {
            return false;
        }
        va += chunk;
        size -= chunk;
    }
    return true;
}

std::string Call::ReadString(u64 va, size_t max_length) {
    std::string text;
    for (size_t i = 0; i < max_length; ++i) {
        char c{};
        if (!mem.Read(va + i, &c, 1) || c == '\0') {
            break;
        }
        text.push_back(c);
    }
    return text;
}

u64 Call::CallNvn(std::string_view name, std::initializer_list<u64> int_args) {
    const Handler handler = Registry::Find(name);
    if (handler == nullptr) {
        return 0;
    }
    Kernel::Svc::ThreadContext nested_ctx = ctx;
    size_t index = 0;
    for (const u64 arg : int_args) {
        nested_ctx.r[index++] = arg;
    }
    Call nested{runtime, system, nested_ctx, mem, rec};
    handler(nested);
    return nested_ctx.r[0];
}

bool Call::WaitUntil(const std::function<bool()>& ready, std::chrono::nanoseconds timeout,
                     std::source_location where) {
    return runtime.WaitUntil(system, ready, timeout, where);
}

Runtime::Runtime() = default;

Runtime::~Runtime() {
    // Without a Shutdown the renderer may be gone already: never touch the sink from here.
    if (gate) {
        gate->Abandon();
    }
}

Runtime& Runtime::Instance() {
    static Runtime instance;
    return instance;
}

bool Runtime::Start(VideoCommon::Nvn::Sink& target,
                    std::unique_ptr<VideoCommon::Nvn::GuestMemory> guest_memory,
                    Kernel::KernelCore* kernel_) {
    if (recorder != nullptr || guest_memory == nullptr) {
        return false;
    }
    kernel = kernel_;
    memory = std::move(guest_memory);
    gate = std::make_unique<VideoCommon::Nvn::SinkGate>(target);
    recorder = std::make_unique<VideoCommon::Nvn::Recorder>(*memory, *gate);
    recorder->SetCompletionObserver([this](u64) { WakeWaiters(); });
    heartbeat = std::jthread{[this](std::stop_token token) { HeartbeatLoop(token); }};
    return true;
}

void Runtime::OpenPipelineCacheEarly(Core::System& system) {
    VideoCommon::Nvn::Sink* const sink = system.Renderer().NvnSink();
    if (sink == nullptr) {
        return;
    }
    // System::PrebuildJitCache runs twice per boot (the emu thread, then CpuManager::OnGpuReady):
    // the title is named once per process.
    const Kernel::KProcess* const process = system.ApplicationProcess();
    if (process == nullptr) {
        return;
    }
    static std::atomic<u64> opened_for_process{~u64{0}};
    if (opened_for_process.exchange(process->GetProcessId()) == process->GetProcessId()) {
        return;
    }
    const u64 title_id = system.GetApplicationProcessProgramID();
    InstallPackageStageSource(system);
    // The replay thread opens the cache and waits for its rebuild there; this thread goes on.
    sink->SetTitleId(title_id);
    LOG_INFO(Core, "[nvn] pipeline cache opened early for {:016X}: its prepare step and rebuild run "
                   "while the CPU code is built (EDEN_NX2_NVN_EARLY_CACHE=0 turns this off)",
             title_id);
}

bool Runtime::EnsureRecorder(Core::System& system) {
    if (created.load(std::memory_order_acquire)) {
        return recorder != nullptr;
    }
    std::scoped_lock lock{create_mutex};
    if (created.load(std::memory_order_relaxed)) {
        return recorder != nullptr;
    }
    VideoCommon::Nvn::Sink* const sink = system.Renderer().NvnSink();
    if (sink == nullptr) {
        LOG_ERROR(Core, "[nvn] HLE call without a replay sink; nothing is recorded");
        created.store(true, std::memory_order_release);
        return false;
    }
    const u64 title_id = system.GetApplicationProcessProgramID();
    InstallPackageStageSource(system);
    sink->SetTitleId(title_id);
    const Common::HostMemory& host_memory = system.DeviceMemory().buffer;
    if (!Start(*sink,
               std::make_unique<ProcessMemory>(system.ApplicationMemory(), stopped,
                                               host_memory.VirtualBasePointer(),
                                               static_cast<u64>(host_memory.VirtualSize())),
               &system.Kernel())) {
        created.store(true, std::memory_order_release);
        return false;
    }
    gate->SetFailureCallback(
        [this, &system](std::string reason) { OnSinkFailure(system, std::move(reason)); });
    {
        // Submit copies a list's words in runs between the records that need work (calls and
        // deferred records, indexed at EndRecording) instead of one record at a time. The output
        // is the same; EDEN_NX2_NVN_BATCH_COPIES=0 turns it off.
        const char* const value = std::getenv("EDEN_NX2_NVN_BATCH_COPIES");
        const bool batch = value == nullptr || std::strcmp(value, "0") != 0;
        recorder->SetCommandListBatching(batch);
        LOG_INFO(Core, "[nvn] command-list copies at submit: {}", batch ? "batched" : "per record");
    }
    {
        // Uniform snapshot at submit: on (strict) for Ocarina of Time, the variable overrides it
        // for any title (0 off, 2 raw-reachable pages left to the CPU-written test,
        // anything else strict).
        using VideoCommon::Nvn::UniformSnapshotMode;
        const char* const value = std::getenv("EDEN_NX2_NVN_UBO_SNAPSHOT");
        const bool forced = value != nullptr && *value != '\0';
        const u64 title = system.GetApplicationProcessProgramID();
        const char* default_reason = "default";
        UniformSnapshotMode mode = UniformSnapshotMode::Off;
        if (forced) {
            mode = std::strcmp(value, "0") == 0   ? UniformSnapshotMode::Off
                   : std::strcmp(value, "2") == 0 ? UniformSnapshotMode::CpuOnly
                                                  : UniformSnapshotMode::Strict;
        } else if (title == Settings::OcarinaOfTimeProgramId) {
            mode = UniformSnapshotMode::Strict;
            default_reason = "Ocarina of Time";
        } else if (title == Settings::MarioKartWorldProgramId) {
            // Exposure and other per-frame uniforms must reflect the bytes at guest submit.
            mode = UniformSnapshotMode::Strict;
            default_reason = "Mario Kart World";
        }
        recorder->SetUniformSnapshot(mode);
        LOG_INFO(Core, "[nvn] uniform snapshot at submit: {} ({})",
                 mode == UniformSnapshotMode::Strict    ? "on, strict"
                 : mode == UniformSnapshotMode::CpuOnly ? "on, CPU-written test only"
                                                        : "off",
                 forced ? fmt::format("EDEN_NX2_NVN_UBO_SNAPSHOT={}", value)
                        : default_reason);
    }
    {
        // Byte cap of the lists recorded at synthetic command memory. Dropping one is not
        // provably safe (the game holds its handle), but a game that takes its command memory
        // from the callback this path does not invoke records every list there, once each, and
        // nothing else ever frees them: 2.8 GiB after four minutes of an Unreal Engine title.
        // The default is high enough that lists a game builds once and keeps never reach it.
        // EDEN_NX2_NVN_SYNTH_LIST_MIB=<n> sets it, 0 turns it off.
        constexpr u64 DefaultCapMib = 1024;
        const char* const value = std::getenv("EDEN_NX2_NVN_SYNTH_LIST_MIB");
        const u64 mib = value != nullptr ? std::strtoull(value, nullptr, 10) : DefaultCapMib;
        synth_list_cap_mib.store(mib);
        if (mib != 0) {
            recorder->SetSyntheticListCap(mib << 20);
            LOG_INFO(Core,
                     "[nvn] synthetic command lists: byte cap {} MiB ({}), least recently used "
                     "ones idle for {} presents and called by no list are dropped",
                     mib, value != nullptr ? "EDEN_NX2_NVN_SYNTH_LIST_MIB" : "default",
                     VideoCommon::Nvn::DEFAULT_SYNTHETIC_LIST_IDLE_PRESENTS);
        }
    }
    // Legacy-mode report timestamps are the GPU emulation path's own (`GPU::GetTicks()`); real
    // reports and nvnDeviceGetCurrentTimestampInNanoseconds run on the guest's global clock.
    VideoCommon::Nvn::SetLegacyTickSource([&system] { return system.GPU().GetTicks(); });
    VideoCommon::Nvn::SetGuestClockSource(
        [&system] { return static_cast<u64>(system.CoreTiming().GetGlobalTimeNs().count()); });
    Bridge::Instance().SetUnsupportedProvider([this] {
        std::vector<Bridge::UnsupportedRow> rows;
        for (UnsupportedRow& row : UnsupportedRows()) {
            rows.push_back({std::move(row.where), std::move(row.reason), row.count});
        }
        return rows;
    });
    created.store(true, std::memory_order_release);
    LOG_INFO(Core, "[nvn] recorder created");
    return true;
}

void Runtime::OnSinkFailure(Core::System& system, std::string reason) {
    if (failed.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    LOG_CRITICAL(Core,
                 "[nvn] the host GPU failed ({}): the game's graphics work can no longer run; "
                 "stopping the emulation",
                 reason);
    // The replay completes every later submission without executing it, so no guest wait hangs
    // until the frontend has stopped the emulation.
    system.Exit();
}

void Runtime::Dispatch(Core::System& system, Kernel::Svc::ThreadContext& ctx, Handler handler,
                       bool observer) {
    if (stopped.load(std::memory_order_acquire) || !EnsureRecorder(system)) {
        if (!observer) {
            ctx.r[0] = 0;
            ctx.v[0] = {0, 0};
        }
        return;
    }
    if (VideoCommon::Nvn::PreRec::Active()) [[unlikely]] {
        VideoCommon::Nvn::PreRec::OnGameCall(reinterpret_cast<const void*>(handler),
                                             TlRingTid(&system));
    }
    Call call{*this, &system, ctx, *memory, *recorder};
    handler(call);
}

void Runtime::Run(Kernel::Svc::ThreadContext& ctx, VideoCommon::Nvn::GuestMemory& memory_,
                  VideoCommon::Nvn::Recorder& recorder_, Handler handler) {
    Call call{*this, nullptr, ctx, memory_, recorder_};
    handler(call);
}

void Runtime::Shutdown() {
    if (stopped.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    VideoCommon::Nvn::SetLegacyTickSource({});
    VideoCommon::Nvn::SetGuestClockSource({});
    VideoCommon::Nvn::SetPackageStageSource(0, {});
    Bridge::Instance().SetUnsupportedProvider({});
    if (recorder) {
        recorder->SetCompletionObserver({});
        recorder->FlushDestroys();
        const VideoCommon::Nvn::RecorderStats stats = recorder->Stats();
        LOG_INFO(Core,
                 "[nvn] recorder: submissions={} presents={} lists={} records={} dropped={} "
                 "tag_mismatch={} stale_serial={} callbacks_skipped={} bad_argument={}",
                 stats.submissions, stats.presents, stats.lists_submitted, stats.records_recorded,
                 stats.records_dropped, stats.tag_mismatch, stats.stale_serial,
                 stats.callbacks_skipped, stats.bad_argument);
    }
    if (heartbeat.joinable()) {
        heartbeat.request_stop();
        heartbeat.join();
    }
    WakeWaiters();
    // The renderer is destroyed after this (while a guest thread can still be inside an NVN call,
    // using the Recorder): from here on the Recorder's calls end at the gate. The completion
    // thread can no longer be in the Recorder's callback once Detach returned.
    if (gate) {
        gate->Detach();
    }
}

void Runtime::Release() {
    Shutdown();
    std::scoped_lock lock{create_mutex};
    recorder.reset();
    gate.reset();
    memory.reset();
    kernel = nullptr;
    ClearPoolTable();
    ClearTextureBindings();
}

void Runtime::HeartbeatLoop(std::stop_token token) {
    // A status line every 10 s: what the recorder and the replay have done, and what is waiting.
    u64 last_presents = ~0ull;
    u32 stalled_beats = 0;
    // EDEN_NX2_NVN_MEMORY_LOG=0 turns the `[nvn] memory:` line off.
    const char* const memory_log_value = std::getenv("EDEN_NX2_NVN_MEMORY_LOG");
    const bool memory_log = memory_log_value == nullptr || std::strcmp(memory_log_value, "0") != 0;
    // What the previous status line saw: per heartbeat (per boot), never carried into the next one.
    std::map<std::string, std::pair<u64, u64>> last_waits;
    std::map<std::string, u64> last_calls;
    // The memory census runs every `census_every` status lines (10 s each).
    const u64 census_every = [] {
        const char* const value = std::getenv("EDEN_MEMORY_CENSUS_SEC");
        const u64 seconds = value != nullptr ? std::strtoull(value, nullptr, 10) : 60;
        return seconds == 0 ? u64{0} : (std::max<u64>)(1, seconds / 10);
    }();
    u64 census_ticks = 0;
    Common::TrackLargeAllocations(); // EDEN_MEMORY_TRACK_ALLOC=1 only
    while (!token.stop_requested()) {
        for (int i = 0; i < 100 && !token.stop_requested(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds{100});
        }
        if (token.stop_requested() || recorder == nullptr) {
            break;
        }
        const VideoCommon::Nvn::RecorderStats s = recorder->Stats();
        u64 replay_ops_unsupported = 0;
        u64 replay_waits_unsatisfied = 0;
        if (gate != nullptr) {
            std::vector<VideoCommon::Nvn::SinkStat> stats;
            gate->CollectStats(stats);
            for (const auto& stat : stats) {
                if (stat.reason == "ops_unsupported") {
                    replay_ops_unsupported = stat.count;
                } else if (stat.reason == "waits_unsatisfied") {
                    replay_waits_unsatisfied = stat.count;
                }
            }
        }
        LOG_INFO(Core,
                 "[nvn] status: presents={}{} seq last={} completed={} submissions={} lists={} "
                 "dropped={} bad_argument={} tag_mismatch={} waits_unsatisfied={} waiters={} "
                 "waits={} wait_ms={} timeouts={} replay_ops_unsupported={}",
                 presents.load(), presents.load() == last_presents ? " (no progress)" : "",
                 recorder->LastSeq(), recorder->CompletedSeq(), s.submissions, s.lists_submitted,
                 s.records_dropped, s.bad_argument, s.tag_mismatch, replay_waits_unsatisfied,
                 waiter_count.load(), blocking_waits.load(), blocking_wait_ns.load() / 1'000'000,
                 wait_timeouts.load(), replay_ops_unsupported);
        if (GpuProfileOn()) {
            // Where the guest blocked since the last line: function: waits / total ms.
            // `last_waits` is the heartbeat's own: waits are reported as the change since the last line.
            std::string line;
            std::scoped_lock lock{wait_by_mutex};
            for (const auto& [name, now] : wait_by_function) {
                auto& before = last_waits[name];
                if (now.first != before.first) {
                    line += fmt::format(" {}: {} waits {:.1f} ms;", name, now.first - before.first,
                                        static_cast<double>(now.second - before.second) / 1e6);
                }
                before = now;
            }
            LOG_INFO(Core, "[nvn] gpuprof guest waits since last status:{}", line);
            const u64 n = sync_wait_n.exchange(0);
            const u64 after = sync_after_fence_sum.exchange(0);
            const u64 behind = sync_behind_sum.exchange(0);
            if (n != 0) {
                LOG_INFO(Core,
                         "[nvn] gpuprof SyncWait: {} blocking waits; on average {:.2f} submissions "
                         "were already submitted after the waited fence, {:.2f} submissions up to "
                         "the fence were still to complete",
                         n, static_cast<double>(after) / static_cast<double>(n),
                         static_cast<double>(behind) / static_cast<double>(n));
            }
        }
        {
            std::string slots;
            std::scoped_lock window_lock{window_mutex};
            for (const auto& [va, state] : windows) {
                slots += fmt::format(" window {:#x}: acquired_mask={:#x}/{} presents={}", va,
                                     state.acquired_mask, state.count, state.presents);
            }
            LOG_INFO(Core, "[nvn] windows:{}", slots);
        }
        if (Bridge::Instance().GetConfig().thread_dump && kernel != nullptr) {
            // Diagnostic: which NVN functions were called since the last line.
            // `last_calls`: the census rows are cumulative; print the change since the last line.
            std::string calls;
            for (const auto& row : Bridge::Instance().CensusRows()) {
                const u64 before = last_calls[row.name];
                last_calls[row.name] = row.calls;
                if (row.calls != before) {
                    calls += fmt::format(" {}={}", row.name, row.calls - before);
                }
            }
            LOG_INFO(Core, "[nvn] calls since last status:{}", calls);
        }
        if (gate != nullptr) {
            LOG_INFO(Core, "[nvn] {}", gate->DescribeState());
        }
        if (memory_log) {
            // Where the memory is: the recorder's lists and the textures only they (or a
            // re-initialised object) keep, then the replay's heaps, spills and holders.
            const VideoCommon::Nvn::TextureHolderCounts holders = recorder->PublishTextureHolders();
            LOG_INFO(Core,
                     "[nvn] memory: synthetic lists {} ({} MiB, cap {}, dropped by cap {} by "
                     "count {}); finalized textures alive for synthetic lists {}, guest lists "
                     "{}, retiring {}; re-initialised textures alive {} (reinitialized_objects "
                     "{}); {}",
                     s.synthetic_lists, s.synthetic_list_bytes >> 20,
                     synth_list_cap_mib.load() != 0
                         ? fmt::format("{} MiB", synth_list_cap_mib.load())
                         : std::string{"none"},
                     s.synthetic_cap_evictions, s.synthetic_evictions, holders.synthetic_lists,
                     holders.guest_lists, holders.retiring, holders.reinitialized,
                     s.reinitialized_objects, gate != nullptr ? gate->DescribeMemory() : "");
        }
        // The process's own memory by kind, its largest allocations and what grew, once a minute
        // (Common::ProcessMemoryCensus; EDEN_MEMORY_CENSUS_SEC, 0: off).
        if (census_every != 0 && ++census_ticks >= census_every) {
            census_ticks = 0;
            for (const std::string& line : Common::ProcessMemoryCensus()) {
                LOG_INFO(Core, "[mem] {}", line);
            }
        }
        // Two status lines in a row without a present: say once where every guest thread is, so a
        // game that hangs while it boots names what it waits for. Read without the scheduler lock
        // (this is not a guest thread); the threads of a stalled game do not move.
        if (presents.load() == last_presents && ++stalled_beats == 2 && kernel != nullptr) {
            if (Kernel::KProcess* const process = kernel->System().ApplicationProcess()) {
                static constexpr const char* reasons[] = {"running", "sleep", "ipc", "sync",
                                                          "condvar", "arbitration", "suspended"};
                for (Kernel::KThread& thread : process->GetThreadList()) {
                    const u32 reason = static_cast<u32>(thread.GetWaitReasonForDebugging());
                    std::string where;
                    size_t frames = 0;
                    for (const auto& entry : Core::GetBacktrace(&thread)) {
                        where += fmt::format(" {}+{:#x}", entry.module, entry.offset);
                        if (++frames == 8) {
                            break;
                        }
                    }
                    LOG_INFO(Core, "[nvn] stalled: thread {} \"{}\" {}:{}", thread.GetThreadId(),
                             Core::GetThreadName(&thread).value_or(""),
                             reason < std::size(reasons) ? reasons[reason] : "?", where);
                }
            }
        } else if (presents.load() != last_presents) {
            stalled_beats = 0;
        }
        last_presents = presents.load();
    }
}

void Runtime::ResetForNewProcess() {
    if (!stopped.load(std::memory_order_acquire)) {
        return;
    }
    std::scoped_lock lock{create_mutex};
    // What the previous boot left (a Release already freed it): its Recorder is destroyed here, the
    // previous system's guest threads are gone and its gate is detached.
    recorder.reset();
    gate.reset();
    memory.reset();
    kernel = nullptr;
    ClearPoolTable();
    ClearTextureBindings();
    created.store(false, std::memory_order_release);
    failed.store(false, std::memory_order_release);
    Common::BootCounter::ResetAll();
    {
        std::scoped_lock window_lock{window_mutex};
        windows.clear();
    }
    guest_request_producer = 0;
    next_debug_domain_id = 1;
    device.window_origin_mode = 0;
    device.depth_mode = 0;
    device.flags = 0;
    nv_initialized = false;
    nv_pool_size = 0;
    presents = 0;
    blocking_waits = 0;
    blocking_wait_ns = 0;
    wait_timeouts = 0;
    sync_wait_n = 0;
    sync_after_fence_sum = 0;
    sync_behind_sum = 0;
    {
        std::scoped_lock wait_lock{wait_by_mutex};
        wait_by_function.clear();
    }
    waiter_count = 0;
    waiters.clear();
    stopped.store(false, std::memory_order_release);
}

void Runtime::WakeWaiters() {
    if (kernel == nullptr || waiter_count.load(std::memory_order_seq_cst) == 0) {
        return;
    }
    Kernel::KScopedSchedulerLock lock{*kernel};
    for (Kernel::KThread* thread : waiters) {
        thread->EndWait(*kernel, ResultSuccess);
    }
    waiters.clear();
}

bool Runtime::GpuProfileEnabled() {
    return GpuProfileOn();
}

bool Runtime::TimelineEnabled() {
    return TimelineOn();
}

void Runtime::TimelineNote(const char* event, u64 seq, std::string_view detail) {
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    LOG_INFO(Core, "[nvn] tl {} {} {} {}", us, event, seq, detail);
}

u32 Runtime::TlRingTid(Core::System* system) {
    if (system == nullptr) {
        return 0;
    }
    Kernel::KThread* const thread = Kernel::GetCurrentThreadPointer(system->Kernel());
    return thread != nullptr ? static_cast<u32>(thread->GetThreadId()) : 0;
}

void Runtime::NoteSyncWait(u64 fence, u64 last, u64 completed, std::chrono::nanoseconds waited) {
    sync_wait_n.fetch_add(1, std::memory_order_relaxed);
    sync_after_fence_sum.fetch_add(last > fence ? last - fence : 0, std::memory_order_relaxed);
    sync_behind_sum.fetch_add(fence > completed ? fence - completed : 0, std::memory_order_relaxed);
    if (const u64 present = presents.load(std::memory_order_relaxed); present % 200 == 99) {
        LOG_INFO(Core,
                 "[nvn] gpuprof SyncWait during frame {}: waited {:.2f} ms for fence seq {} (newest "
                 "submitted {}, completed {})",
                 present + 1, static_cast<double>(waited.count()) / 1e6, fence, last, completed);
    }
}

bool Runtime::WaitUntil(Core::System* system, const std::function<bool()>& ready,
                        std::chrono::nanoseconds timeout, const std::source_location& where) {
    if (wait_hook) {
        return wait_hook(ready, timeout);
    }
    if (system == nullptr || kernel == nullptr) {
        // No kernel to sleep in: poll on the host.
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (!ready()) {
            if (timeout.count() >= 0 && std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::microseconds{100});
        }
        return true;
    }
    const auto begin = std::chrono::steady_clock::now();
    Kernel::KernelCore& kernel_ref = *kernel;
    Kernel::KThread* const thread = Kernel::GetCurrentThreadPointer(kernel_ref);
    const bool infinite = timeout.count() < 0;
    s64 deadline_tick = 0;
    if (!infinite) {
        const s64 now_tick = kernel_ref.HardwareTimer().GetTick();
        const s64 offset = std::max<s64>(timeout.count(), 1);
        deadline_tick = now_tick > std::numeric_limits<s64>::max() - offset - 2
                            ? std::numeric_limits<s64>::max()
                            : now_tick + offset + 2;
    }
    bool satisfied = false;
    // Infinite waits sleep in slices so a wait that never ends is reported.
    constexpr s64 SliceTicks = 2'000'000'000;
    int reports = 0;
    while (true) {
        if (ready()) {
            satisfied = true;
            break;
        }
        if (infinite && std::chrono::steady_clock::now() - begin > std::chrono::seconds{5 * (reports + 1)}) {
            ++reports;
            const VideoCommon::Nvn::Recorder* rec = recorder.get();
            LOG_WARNING(Core, "[nvn] {} has waited {} s (last seq {}, completed {})",
                        where.function_name(), 5 * reports, rec ? rec->LastSeq() : 0,
                        rec ? rec->CompletedSeq() : 0);
        }
        if (stopped.load(std::memory_order_acquire)) {
            break;
        }
        if (!infinite && kernel_ref.HardwareTimer().GetTick() >= deadline_tick) {
            break;
        }
        NvnWaitQueue queue{kernel_ref};
        {
            Kernel::KHardwareTimer* timer{};
            Kernel::KScopedSchedulerLockAndSleep sleep{
                kernel_ref, &timer, thread,
                infinite ? kernel_ref.HardwareTimer().GetTick() + SliceTicks : deadline_tick};
            if (thread->IsTerminationRequested()) {
                sleep.CancelSleep();
                break;
            }
            // The waiter is registered before the final check: a completion that lands after the
            // check cannot slip past WakeWaiters, which runs after CompletedSeq is updated.
            waiter_count.fetch_add(1, std::memory_order_seq_cst);
            // `ready` may have side effects (a window slot is claimed): it runs once per check, and
            // a success here is final (never re-evaluated by the loop).
            const bool ready_now = ready();
            if (ready_now || stopped.load(std::memory_order_acquire)) {
                waiter_count.fetch_sub(1, std::memory_order_seq_cst);
                sleep.CancelSleep();
                satisfied = ready_now;
                break;
            }
            waiters.push_back(thread);
            if (Common::TlRing::Armed()) [[unlikely]] {
                Common::TlRing::Note(Common::TlRing::GBlockB,
                                     reinterpret_cast<u64>(where.function_name()), 0,
                                     static_cast<u32>(thread->GetThreadId()));
            }
            queue.SetHardwareTimer(timer);
            thread->BeginWait(kernel_ref, std::addressof(queue));
            thread->SetWaitReasonForDebugging(Kernel::ThreadWaitReasonForDebugging::Sleep);
        }
        // The thread slept here (the scheduler lock is released and the host fiber switched) and
        // was woken by a completion or by the timer.
        if (Common::TlRing::Armed()) [[unlikely]] {
            Common::TlRing::Note(Common::TlRing::GBlockE,
                                 reinterpret_cast<u64>(where.function_name()), 0,
                                 static_cast<u32>(thread->GetThreadId()));
        }
        {
            Kernel::KScopedSchedulerLock lock{kernel_ref};
            if (const auto it = std::ranges::find(waiters, thread); it != waiters.end()) {
                waiters.erase(it);
            }
            waiter_count.fetch_sub(1, std::memory_order_seq_cst);
        }
    }
    if (Common::FrameStats::Enabled()) {
        Common::FrameStats::AddStall(
            std::strstr(where.function_name(), "AcquireTexture") != nullptr
                ? Common::FrameStats::Cause::GuestAcquireWait
                : Common::FrameStats::Cause::GuestSyncWait,
            static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now() - begin)
                                 .count()));
    }
    blocking_waits.fetch_add(1, std::memory_order_relaxed);
    blocking_wait_ns.fetch_add(
        static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - begin)
                             .count()),
        std::memory_order_relaxed);
    (std::strstr(where.function_name(), "AcquireTexture") != nullptr ? acquire_wait_ns : sync_wait_ns)
        .fetch_add(static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - begin)
                                        .count()),
                   std::memory_order_relaxed);
    if (GpuProfileOn()) {
        std::scoped_lock lock{wait_by_mutex};
        auto& row = wait_by_function[where.function_name()];
        ++row.first;
        row.second += static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - begin)
                                           .count());
    }
    if (!satisfied) {
        wait_timeouts.fetch_add(1, std::memory_order_relaxed);
    }
    return satisfied;
}

std::vector<UnsupportedRow> Runtime::UnsupportedRows() {
    std::vector<UnsupportedRow> rows;
    if (recorder) {
        const VideoCommon::Nvn::RecorderStats s = recorder->Stats();
        const auto add = [&rows](const char* reason, u64 count) {
            rows.push_back({"recorder", reason, count});
        };
        add("tag_mismatch", s.tag_mismatch);
        add("stale_serial", s.stale_serial);
        add("bad_guest_read", s.bad_guest_read);
        add("bad_guest_write", s.bad_guest_write);
        add("bad_argument", s.bad_argument);
        add("callbacks_skipped", s.callbacks_skipped);
        add("unknown_call_handle", s.unknown_call_handle);
        add("call_depth_exceeded", s.call_depth_exceeded);
        add("deferred_read_failed", s.deferred_read_failed);
        add("program_code_unavailable", s.program_code_unavailable);
        add("stage_fallback", s.stage_fallback);
        add("reinitialized_objects", s.reinitialized_objects);
        add("records_dropped", s.records_dropped);
    }
    if (gate != nullptr) {
        std::vector<VideoCommon::Nvn::SinkStat> stats;
        gate->CollectStats(stats);
        for (auto& stat : stats) {
            rows.push_back({"replay", std::move(stat.reason), stat.count});
        }
    }
    return rows;
}

void Registry::Dispatch(Core::System& system, Kernel::Svc::ThreadContext& ctx, Handler handler,
                        bool observer) {
    Runtime::Instance().Dispatch(system, ctx, handler, observer);
}

namespace {
std::mutex process_handlers_mutex;
std::map<std::string, Handler, std::less<>> process_handlers;

void PublishHandler(std::string_view name, Handler handler) {
    std::scoped_lock lock{process_handlers_mutex};
    process_handlers.insert_or_assign(std::string{name}, handler);
    VideoCommon::Nvn::PreRec::NameHandler(reinterpret_cast<const void*>(handler), name);
}
} // namespace

Handler Registry::Find(std::string_view name) {
    std::scoped_lock lock{process_handlers_mutex};
    const auto it = process_handlers.find(name);
    return it != process_handlers.end() ? it->second : nullptr;
}

void Registry::AddImpl(std::string_view name, Handler handler, HleFn bridge_fn) {
    const auto names = FunctionNames();
    const auto it = std::ranges::find(names, name);
    if (it == names.end() || handlers.contains(name)) {
        ++rejected;
        LOG_ERROR(Core, "[nvn] HLE registration of {} rejected ({})", name,
                  it == names.end() ? "not an NVN function" : "registered twice");
        return;
    }
    handlers.emplace(std::string{name}, handler);
    PublishHandler(name, handler);
    if (bridge != nullptr) {
        bridge->RegisterHle(static_cast<u32>(it - names.begin()), bridge_fn);
    }
}

void Registry::AddNvImpl(std::string_view name, Handler handler, HleFn bridge_fn, bool observer) {
    if (handlers.contains(name)) {
        ++rejected;
        LOG_ERROR(Core, "[nvn] HLE registration of {} rejected (registered twice)", name);
        return;
    }
    handlers.emplace(std::string{name}, handler);
    PublishHandler(name, handler);
    if (bridge != nullptr) {
        if (observer) {
            bridge->RegisterNvObserver(name, bridge_fn);
        } else {
            bridge->RegisterNvHook(name, bridge_fn);
        }
    }
}

void RegisterAll(Registry& registry) {
    RegisterDeviceFunctions(registry);
    RegisterQueueFunctions(registry);
    RegisterWindowFunctions(registry);
    RegisterNvFunctions(registry);
    RegisterNativeImageFunctions(registry);
    RegisterPoolFunctions(registry);
    RegisterBufferFunctions(registry);
    RegisterTextureFunctions(registry);
    RegisterSamplerFunctions(registry);
    RegisterProgramFunctions(registry);
    RegisterSyncFunctions(registry);
    RegisterEventFunctions(registry);
    RegisterDlssFunctions(registry);
    RegisterStateFunctions(registry);
    RegisterCommandBufferFunctions(registry);
    RegisterCommandBufferBindFunctions(registry);
}

void RegisterAll(Bridge& bridge) {
    Runtime::Instance().ResetForNewProcess();
    Registry registry{&bridge};
    RegisterAll(registry);
    LOG_INFO(Core, "[nvn] HLE bodies registered: {} ({} rejected)", registry.Count(),
             registry.Rejected());
}

} // namespace Core::NvnHle::Hle
