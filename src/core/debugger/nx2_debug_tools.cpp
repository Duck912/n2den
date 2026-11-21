// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

#include "common/demangle.h"
#include "common/fs/fs_util.h"
#include "common/fs/path_util.h"
#include "common/log_ring.h"
#include "common/logging.h"
#include "common/nx2_sampler.h"
#include "common/scm_rev.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/arm/debug.h"
#include "core/arm/symbols.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/debugger/nx2_debug_server.h"
#include "core/debugger/nx2_debug_tools.h"
#include "core/hardware_properties.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/k_scheduler.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/kernel/kernel.h"
#include "core/hle/nvn/bridge.h"
#include "core/memory.h"
#include "dynarmic/interface/A64/a64.h"
#include "hid_core/hid_core.h"
#include "video_core/gpu.h"
#include "video_core/nx2_debug_control.h"

// Frames presented by the running game: the emulated GPU path counts in Nx2Debug::present_count,
// nvn-hle in Core::g_nx2_present_count; only one of them runs for a game, so the sum is its count.
static u64 Nx2PresentCount() {
    return VideoCore::Nx2Debug::present_count.load(std::memory_order_relaxed) +
           Core::g_nx2_present_count.load(std::memory_order_relaxed);
}


namespace Core::Nx2Debug {

using nlohmann::json;
using namespace std::chrono_literals;

/// Lines the log ring keeps for log_tail.
constexpr size_t RING_LINES = 20000;
/// Bytes one read_memory or read_gpu_memory call returns at most.
constexpr u64 MAX_READ = 64 * 1024;
/// Bytes one write_memory call writes at most.
constexpr size_t MAX_WRITE = 4096;
/// Presented frames one input or input_sequence call holds at most (10 minutes at 60 fps).
constexpr u64 MAX_INPUT_FRAMES = 36000;
constexpr size_t MAX_INPUT_STEPS = 1000;

constexpr std::array<std::pair<std::string_view, HID::NpadButton>, 16> INPUT_BUTTONS{{
    {"A", HID::NpadButton::A},
    {"B", HID::NpadButton::B},
    {"X", HID::NpadButton::X},
    {"Y", HID::NpadButton::Y},
    {"L", HID::NpadButton::L},
    {"R", HID::NpadButton::R},
    {"ZL", HID::NpadButton::ZL},
    {"ZR", HID::NpadButton::ZR},
    {"Plus", HID::NpadButton::Plus},
    {"Minus", HID::NpadButton::Minus},
    {"Up", HID::NpadButton::Up},
    {"Down", HID::NpadButton::Down},
    {"Left", HID::NpadButton::Left},
    {"Right", HID::NpadButton::Right},
    {"StickL", HID::NpadButton::StickL},
    {"StickR", HID::NpadButton::StickR},
}};

struct ToolState {
    explicit ToolState(System& system_, std::string title_, u64 program_id_,
                       std::filesystem::path dump_directory_)
        : system{system_}, title{std::move(title_)}, program_id{program_id_},
          dump_directory{std::move(dump_directory_)} {}

    System& system;
    std::string title;
    u64 program_id;
    std::filesystem::path dump_directory;
    std::chrono::steady_clock::time_point start{std::chrono::steady_clock::now()};
    /// Names this session's captures apart from earlier ones in the same dump directory.
    s64 session{std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                    .count()};
    std::string log_filter{Settings::values.log_filter.GetValue()};
    /// EDEN_NX2_DEBUG_WRITE=1 at launch: write_memory may change guest memory.
    bool write_enabled{[] {
        const char* const value = std::getenv("EDEN_NX2_DEBUG_WRITE");
        return value != nullptr && std::string_view{value} == "1";
    }()};
    u64 captures{};
    u64 samples{};
    u64 gpu_reads{};
    std::atomic<bool> stopping{};
};

namespace SchedTrace {

std::atomic<bool> enabled{};

namespace {

/// Switches one core keeps per window: 40000 a second for the longest window (5 s).
constexpr size_t RING_CAPACITY = 200000;
constexpr size_t MAX_PRESENTS = 4096;

struct Ring {
    /// Allocated on first use and never freed: a switch that read `enabled` just before a window
    /// ended may still write into it.
    std::unique_ptr<Switch[]> events;
    std::atomic<u32> count{};
    std::atomic<u64> dropped{};
};

std::array<Ring, Core::Hardware::NUM_CPU_CORES> rings;

/// Spans one core keeps per window (40 bytes each, allocated on the first window with spans).
constexpr size_t SPAN_CAPACITY = 600000;

struct SpanRing {
    std::unique_ptr<Span[]> events;
    std::atomic<u32> count{};
    std::atomic<u64> dropped{};
};

std::array<SpanRing, Core::Hardware::NUM_CPU_CORES> span_rings;
std::mutex present_mutex;
std::vector<Present> presents;
u64 presents_dropped{};
/// Under present_mutex: steady_clock ns at each recorded present (maps host spans to ticks).
std::vector<s64> present_host_ns;
constexpr size_t MAX_HOST_SPANS = 65536;
std::vector<HostSpan> host_spans;
u64 host_spans_dropped{};

s64 SteadyNs(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}

u32 TraceId(u64 id) {
    return id <= (std::numeric_limits<u32>::max)() ? static_cast<u32>(id)
                                                   : (std::numeric_limits<u32>::max)();
}

} // Anonymous namespace

void RecordSwitch(s32 core, s64 ticks, u64 out_id, u64 in_id) {
    if (core < 0 || static_cast<size_t>(core) >= rings.size() ||
        !enabled.load(std::memory_order_acquire)) {
        return;
    }
    Ring& ring = rings[static_cast<size_t>(core)];
    const u32 index = ring.count.load(std::memory_order_relaxed);
    if (index >= RING_CAPACITY) {
        ring.dropped.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    ring.events[index] = {ticks, TraceId(out_id), TraceId(in_id)};
    ring.count.store(index + 1, std::memory_order_release);
}

std::atomic<u32> spans_enabled{};

void RecordSpan(s32 core, const Span& span) {
    if (core < 0 || static_cast<size_t>(core) >= span_rings.size() ||
        (spans_enabled.load(std::memory_order_acquire) &
         (1u << static_cast<u32>(span.kind))) == 0) {
        return;
    }
    SpanRing& ring = span_rings[static_cast<size_t>(core)];
    const u32 index = ring.count.load(std::memory_order_relaxed);
    if (index >= SPAN_CAPACITY) {
        ring.dropped.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    ring.events[index] = span;
    ring.count.store(index + 1, std::memory_order_release);
}

void RecordPresent(u32 kind, s64 ticks, u64 count, u64 thread_id) {
    std::scoped_lock lock{present_mutex};
    if (!enabled.load(std::memory_order_relaxed)) {
        return;
    }
    if (presents.size() >= MAX_PRESENTS) {
        ++presents_dropped;
        return;
    }
    presents.push_back({ticks, count, thread_id, kind});
    present_host_ns.push_back(SteadyNs(std::chrono::steady_clock::now()));
}

void RecordHostSpan(HostSpanKind kind, std::chrono::steady_clock::time_point start,
                    std::chrono::steady_clock::time_point end, u64 arg) {
    std::scoped_lock lock{present_mutex};
    if (!enabled.load(std::memory_order_relaxed)) {
        return;
    }
    if (host_spans.size() >= MAX_HOST_SPANS) {
        ++host_spans_dropped;
        return;
    }
    host_spans.push_back({SteadyNs(start), SteadyNs(end), arg, kind});
}

namespace {

/// Starts a window: only the debug server thread calls Start and Collect, one window at a time.
void Start(u32 span_kinds) {
    const bool spans = span_kinds != 0;
    for (Ring& ring : rings) {
        if (!ring.events) {
            ring.events = std::make_unique<Switch[]>(RING_CAPACITY);
        }
        ring.count.store(0, std::memory_order_relaxed);
        ring.dropped.store(0, std::memory_order_relaxed);
    }
    if (spans) {
        for (SpanRing& ring : span_rings) {
            if (!ring.events) {
                ring.events = std::make_unique<Span[]>(SPAN_CAPACITY);
            }
            ring.count.store(0, std::memory_order_relaxed);
            ring.dropped.store(0, std::memory_order_relaxed);
        }
    }
    {
        std::scoped_lock lock{present_mutex};
        presents.clear();
        presents.reserve(MAX_PRESENTS);
        presents_dropped = 0;
        present_host_ns.clear();
        host_spans.clear();
        host_spans_dropped = 0;
    }
    enabled.store(true, std::memory_order_release);
    if (spans) {
        spans_enabled.store(span_kinds, std::memory_order_release);
    }
}

struct Collected {
    std::array<std::vector<Switch>, Core::Hardware::NUM_CPU_CORES> switches;
    std::array<u64, Core::Hardware::NUM_CPU_CORES> dropped{};
    std::array<std::vector<Span>, Core::Hardware::NUM_CPU_CORES> spans;
    std::array<u64, Core::Hardware::NUM_CPU_CORES> spans_dropped{};
    std::vector<Present> presents;
    u64 presents_dropped{};
    std::vector<s64> present_host_ns;
    std::vector<HostSpan> host_spans;
    u64 host_spans_dropped{};
};

/// Ends the window and copies what it recorded. Slots below a ring's count are complete (the writer
/// publishes the count after the slot); a late writer only touches slots above it.
Collected Stop() {
    const bool spans = spans_enabled.load(std::memory_order_relaxed) != 0;
    enabled.store(false, std::memory_order_relaxed);
    spans_enabled.store(0, std::memory_order_relaxed);
    Collected out;
    for (size_t core = 0; core < rings.size(); ++core) {
        const u32 count = rings[core].count.load(std::memory_order_acquire);
        out.switches[core].assign(rings[core].events.get(), rings[core].events.get() + count);
        out.dropped[core] = rings[core].dropped.load(std::memory_order_relaxed);
        if (spans) {
            SpanRing& ring = span_rings[core];
            const u32 span_count = ring.count.load(std::memory_order_acquire);
            out.spans[core].assign(ring.events.get(), ring.events.get() + span_count);
            out.spans_dropped[core] = ring.dropped.load(std::memory_order_relaxed);
        }
    }
    std::scoped_lock lock{present_mutex};
    out.presents = presents;
    out.presents_dropped = presents_dropped;
    out.present_host_ns = present_host_ns;
    out.host_spans = host_spans;
    out.host_spans_dropped = host_spans_dropped;
    return out;
}

} // Anonymous namespace

} // namespace SchedTrace

namespace {

std::string Hex(u64 value) {
    return fmt::format("{:016x}", value);
}

json HexList(const std::vector<u64>& values) {
    json list = json::array();
    for (const u64 value : values) {
        list.push_back(Hex(value));
    }
    return list;
}

template <typename T>
T Param(const json& params, const char* name, T fallback, T low, T high) {
    const T value = params.value(name, fallback);
    return (std::clamp)(value, low, high);
}

std::string HexAddress(u64 value) {
    return fmt::format("{:#x}", value);
}

/// A path as UTF-8 with forward slashes, for results and logs.
std::string PathText(const std::filesystem::path& path) {
    return Common::FS::ToUTF8String(path.generic_u8string());
}

/// An address as a JSON number or a hex string (0x optional).
u64 AddressParam(const json& params, const char* name) {
    const json& value = params.at(name);
    if (value.is_number_unsigned()) {
        return value.get<u64>();
    }
    if (!value.is_string()) {
        throw std::runtime_error{
            fmt::format("{} must be a hex string or a non-negative number", name)};
    }
    const std::string text = value.get<std::string>();
    const std::string_view view{text};
    const std::string_view digits =
        view.starts_with("0x") || view.starts_with("0X") ? view.substr(2) : view;
    u64 address = 0;
    const auto [end, error] =
        std::from_chars(digits.data(), digits.data() + digits.size(), address, 16);
    if (digits.empty() || error != std::errc{} || end != digits.data() + digits.size()) {
        throw std::runtime_error{
            fmt::format("bad {} \"{}\": want hex, e.g. 0x8012345000", name, text)};
    }
    return address;
}

/// The size of a read: 1 to MAX_READ bytes, not wrapping past the end of the address space.
u64 ReadSize(const json& params, u64 address) {
    const json& value = params.at("size");
    if (!value.is_number_integer() || value.get<s64>() < 1 || value.get<u64>() > MAX_READ) {
        throw std::runtime_error{fmt::format("size must be 1 to {} bytes", MAX_READ)};
    }
    const u64 size = value.get<u64>();
    if (address + size < address) {
        throw std::runtime_error{"address + size wraps around"};
    }
    return size;
}

Kernel::KProcess& Application(ToolState& state) {
    Kernel::KProcess* const process = state.system.ApplicationProcess();
    if (process == nullptr) {
        throw std::runtime_error{"no application process is running"};
    }
    return *process;
}

/// A host pointer to the byte at address, or null when its page is not mapped. GetPointerSilent
/// reads the page entry as guest cores do, with no commit, log or GPU download; it masks the top
/// 16 bits, so addresses there would alias and are unmapped instead.
const u8* GuestByte(Memory::Memory& memory, u64 address) {
    return (address >> 48) == 0 ? memory.GetPointerSilent(address) : nullptr;
}

bool IsVulkan() {
    return Settings::values.renderer_backend.GetValue() == Settings::RendererBackend::Vulkan;
}

json Status(ToolState& state, const json& params) {
    const auto window =
        std::chrono::milliseconds{Param<int>(params, "fps_window_ms", 500, 0, 5000)};
    const u64 before = Nx2PresentCount();
    const auto begin = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - begin < window &&
           !state.stopping.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(20ms);
    }
    const u64 after = Nx2PresentCount();
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    return {
        {"title", state.title},
        {"program_id", fmt::format("{:016X}", state.program_id)},
        {"build", std::string{Common::g_build_fullname}},
        {"scm_rev", std::string{Common::g_scm_rev}},
        {"renderer", IsVulkan() ? "vulkan" : "other"},
        {"powered_on", state.system.IsPoweredOn()},
        {"paused", state.system.IsPaused()},
        {"presents", after},
        {"fps", seconds > 0.0 ? static_cast<double>(after - before) / seconds : 0.0},
        {"uptime_s",
         std::chrono::duration<double>(std::chrono::steady_clock::now() - state.start).count()},
        {"dump_dir", PathText(state.dump_directory)},
        {"log_filter", state.log_filter},
        {"rejected", HexList(VideoCore::Nx2Debug::RejectedHashes())},
        {"write_memory", state.write_enabled},
    };
}

json LogTail(ToolState&, const json& params) {
    const u64 after = params.value("after", u64{0});
    const size_t limit = Param<size_t>(params, "limit", 200, 1, 5000);
    const std::string contains = params.value("contains", std::string{});
    const std::vector<Common::Log::RingLine> lines = Common::Log::ReadRing(after);
    std::vector<const Common::Log::RingLine*> matched;
    for (const auto& line : lines) {
        if (contains.empty() || line.text.find(contains) != std::string::npos) {
            matched.push_back(&line);
        }
    }
    // From the start the newest lines; after a sequence number the oldest ones, to page forward.
    const bool more = matched.size() > limit;
    if (more) {
        if (after == 0) {
            matched.erase(matched.begin(), matched.end() - static_cast<std::ptrdiff_t>(limit));
        } else {
            matched.resize(limit);
        }
    }
    json text = json::array();
    for (const auto* line : matched) {
        text.push_back(line->text);
    }
    const u64 next = more && after != 0 ? matched.back()->sequence
                     : lines.empty()    ? after
                                        : lines.back().sequence;
    return {{"lines", std::move(text)}, {"next", next}, {"more", more}};
}

json CaptureFrame(ToolState& state, const json& params) {
    if (!IsVulkan()) {
        throw std::runtime_error{"frame capture needs the Vulkan renderer"};
    }
    const u64 id = ++state.captures;
    const auto directory =
        state.dump_directory / fmt::format("capture-{}-{:03}", state.session, id);
    std::filesystem::create_directories(directory);
    const bool sampled = params.value("sampled", false);
    const bool buffers = params.value("buffers", false);
    const int timeout = Param<int>(params, "timeout_s", 120, 1, 600);
    VideoCore::Nx2Debug::RequestCapture(
        {.id = id, .directory = directory, .sampled = sampled, .buffers = buffers});
    const auto finished = VideoCore::Nx2Debug::WaitCapture(id, std::chrono::seconds{timeout});
    if (!finished) {
        VideoCore::Nx2Debug::CancelCapture(id);
        throw std::runtime_error{fmt::format(
            "the capture did not finish within {} s (paused or loading?); if it started, its "
            "files are still being written to {}",
            timeout, PathText(directory))};
    }
    if (!finished->error.empty()) {
        throw std::runtime_error{finished->error};
    }
    const size_t keep = Param<size_t>(params, "manifest_lines", 200, 0, 5000);
    json entries = json::array();
    json kinds = json::object();
    json frame;
    json summary;
    size_t total = 0;
    std::ifstream manifest{directory / "manifest.jsonl"};
    for (std::string line; std::getline(manifest, line);) {
        json entry = json::parse(line, nullptr, false);
        if (entry.is_discarded()) {
            continue;
        }
        const std::string kind = entry.value("kind", std::string{});
        if (kind == "frame") {
            frame = std::move(entry);
        } else if (kind == "summary") {
            summary = std::move(entry);
        } else {
            kinds[kind] = kinds.value(kind, 0) + 1;
            if (total++ < keep) {
                entries.push_back(std::move(entry));
            }
        }
    }
    return {
        {"directory", PathText(directory)},
        {"sampled", sampled},
        {"buffers", buffers},
        {"frame", std::move(frame)},
        {"summary", std::move(summary)},
        {"kinds", std::move(kinds)},
        {"entries_total", total},
        {"entries", std::move(entries)},
        {"index_command",
         fmt::format("python3 -I tools/nx2/frame_dump_index.py \"{}\"", PathText(directory))},
    };
}

json SampleThreads(ToolState& state, const json& params) {
    const std::string threads = params.value("threads", std::string{"GPU,GPUBack"});
    const double seconds = Param<double>(params, "seconds", 10.0, 1.0, 60.0);
    std::filesystem::create_directories(state.dump_directory);
    const auto path =
        state.dump_directory / fmt::format("samples-{}-{:03}.txt", state.session, ++state.samples);
    std::string report = Nx2Sampler::Sample(threads.c_str(), seconds, PathText(path));
    report.resize((std::min)(report.size(), Param<size_t>(params, "max_chars", 60000, 0, 4000000)));
    return {{"file", PathText(path)}, {"report", std::move(report)}};
}

constexpr size_t GUEST_CORES = Core::Hardware::NUM_CPU_CORES;

struct GuestThreadSample {
    u64 id{};
    std::string name;
    s32 priority{};
    s32 base_priority{};
    s32 active_core{};
    s32 last_core{};
    u64 affinity{};
    std::string state;
    /// CPU time in CoreTiming clock ticks, including the running slice when on a core.
    s64 cpu_ticks{};
    /// The core it is running on at the snapshot, or -1.
    s32 running_on{-1};
};

struct GuestSnapshot {
    s64 ticks{};
    std::chrono::steady_clock::time_point time;
    std::vector<GuestThreadSample> threads;
    std::array<s64, GUEST_CORES> idle_ticks{};
    std::array<s64, GUEST_CORES> switch_age{};
    std::array<bool, GUEST_CORES> idle_now{};
};

/// The application's guest threads and the per-core idle threads' CPU time, read while the game
/// runs. A thread that is on a core has its time since the scheduler's last context switch added,
/// as svcGetInfo ThreadTickCount does. The scheduler fields are read without its lock, so a
/// context switch racing the read can count one slice twice or not at all.
GuestSnapshot SnapshotGuestThreads(ToolState& state) {
    Kernel::KProcess& process = Application(state);
    Kernel::KernelCore& kernel = state.system.Kernel();
    std::array<const Kernel::KThread*, GUEST_CORES> current{};
    std::array<s64, GUEST_CORES> last_switch{};
    for (size_t core = 0; core < GUEST_CORES; ++core) {
        const Kernel::KScheduler& scheduler = kernel.Scheduler(core);
        current[core] = scheduler.GetSchedulerCurrentThread();
        last_switch[core] = scheduler.GetLastContextSwitchTime();
    }
    GuestSnapshot snapshot;
    snapshot.time = std::chrono::steady_clock::now();
    snapshot.ticks = static_cast<s64>(state.system.CoreTiming().GetClockTicks());
    const auto running_core = [&](const Kernel::KThread* thread) -> s32 {
        for (size_t core = 0; core < GUEST_CORES; ++core) {
            if (current[core] == thread) {
                return static_cast<s32>(core);
            }
        }
        return -1;
    };
    for (size_t core = 0; core < GUEST_CORES; ++core) {
        snapshot.switch_age[core] = snapshot.ticks - last_switch[core];
        const Kernel::KThread* const idle = kernel.Scheduler(core).GetIdleThread();
        snapshot.idle_now[core] = current[core] == idle;
        snapshot.idle_ticks[core] =
            idle->GetCpuTime() +
            (snapshot.idle_now[core] ? (std::max)(s64{0}, snapshot.switch_age[core]) : s64{0});
    }
    // The list lock is a KLightLock: from this host thread it uses the kernel's dummy KThread
    // (priority 65, so a wait never boosts a guest owner), as the gdb stub does. Held only to copy.
    Kernel::KScopedLightLock lock{process.GetListLock()};
    for (const Kernel::KThread& thread : process.GetThreadList()) {
        GuestThreadSample sample;
        sample.id = thread.GetThreadId();
        sample.name = Core::GetThreadName(&thread).value_or(std::string{});
        sample.priority = thread.GetPriority();
        sample.base_priority = thread.GetBasePriority();
        sample.active_core = thread.GetActiveCore();
        sample.last_core = thread.GetCurrentCore();
        sample.affinity = thread.GetAffinityMask().GetAffinityMask();
        sample.state = Core::GetThreadState(&thread);
        sample.running_on = running_core(&thread);
        sample.cpu_ticks = thread.GetCpuTime();
        if (sample.running_on >= 0) {
            sample.cpu_ticks +=
                (std::max)(s64{0}, snapshot.switch_age[static_cast<size_t>(sample.running_on)]);
        }
        snapshot.threads.push_back(std::move(sample));
    }
    return snapshot;
}

json GuestThreads(ToolState& state, const json& params) {
    const double seconds = Param<double>(params, "seconds", 10.0, 1.0, 60.0);
    const GuestSnapshot first = SnapshotGuestThreads(state);
    const auto window = std::chrono::duration<double>(seconds);
    while (std::chrono::steady_clock::now() - first.time < window &&
           !state.stopping.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(20ms);
    }
    if (state.stopping.load(std::memory_order_relaxed)) {
        throw std::runtime_error{"the game is shutting down"};
    }
    const GuestSnapshot second = SnapshotGuestThreads(state);

    constexpr double TICKS_PER_MS = static_cast<double>(Core::Hardware::CNTFREQ) / 1000.0;
    const s64 window_ticks = second.ticks - first.ticks;
    const double window_ms = static_cast<double>(window_ticks) / TICKS_PER_MS;
    const auto percent = [&](s64 ticks) {
        return window_ticks > 0
                   ? 100.0 * static_cast<double>(ticks) / static_cast<double>(window_ticks)
                   : 0.0;
    };
    const auto ms = [&](s64 ticks) { return static_cast<double>(ticks) / TICKS_PER_MS; };

    std::unordered_map<u64, const GuestThreadSample*> before;
    for (const GuestThreadSample& sample : first.threads) {
        before.emplace(sample.id, &sample);
    }
    struct Row {
        const GuestThreadSample* sample;
        s64 ticks;
        const char* presence;
    };
    std::vector<Row> rows;
    for (const GuestThreadSample& sample : second.threads) {
        const auto it = before.find(sample.id);
        if (it == before.end()) {
            // Started during the window: all of its time is inside it.
            rows.push_back({&sample, sample.cpu_ticks, "appeared"});
        } else {
            rows.push_back({&sample, sample.cpu_ticks - it->second->cpu_ticks, "both"});
            before.erase(it);
        }
    }
    for (const auto& [id, sample] : before) {
        // Exited during the window: its time after the first snapshot is lost with it.
        rows.push_back({sample, 0, "disappeared"});
    }
    std::ranges::stable_sort(rows, [](const Row& a, const Row& b) { return a.ticks > b.ticks; });

    json threads = json::array();
    s64 total_ticks = 0;
    for (const Row& row : rows) {
        const GuestThreadSample& sample = *row.sample;
        total_ticks += row.ticks;
        json entry = {
            {"id", sample.id},
            {"name", sample.name.empty() ? fmt::format("thread {}", sample.id) : sample.name},
            {"priority", sample.priority},
            {"base_priority", sample.base_priority},
            {"active_core", sample.active_core},
            {"last_core", sample.last_core},
            {"running_on", sample.running_on},
            {"affinity", fmt::format("{:#x}", sample.affinity)},
            {"state", sample.state},
            {"cpu_ms", ms(row.ticks)},
            {"cpu_percent", percent(row.ticks)},
        };
        if (std::string_view{row.presence} != "both") {
            entry["presence"] = row.presence;
        }
        threads.push_back(std::move(entry));
    }

    json cores = json::array();
    s64 idle_total = 0;
    for (size_t core = 0; core < GUEST_CORES; ++core) {
        const s64 idle = second.idle_ticks[core] - first.idle_ticks[core];
        idle_total += idle;
        cores.push_back({
            {"core", core},
            {"idle_ms", ms(idle)},
            {"idle_percent", percent(idle)},
            {"idle_at_end", second.idle_now[core]},
            {"last_switch_age_ms", ms(second.switch_age[core])},
        });
    }
    return {
        {"window_ms", window_ms},
        {"window_host_ms",
         std::chrono::duration<double, std::milli>(second.time - first.time).count()},
        {"thread_count", second.threads.size()},
        {"total_cpu_ms", ms(total_ticks)},
        {"total_cpu_percent", percent(total_ticks)},
        {"idle_total_ms", ms(idle_total)},
        {"idle_total_percent", percent(idle_total)},
        {"threads", std::move(threads)},
        {"cores", std::move(cores)},
        {"note", "cpu_percent: 100 = one full guest core over the window; time is measured in "
                 "CoreTiming ticks (19.2 MHz). Idle time is the per-core idle threads' time. "
                 "Threads that disappeared report 0: their time after the first snapshot is "
                 "lost."},
    };
}

/// A guest code range: a loaded module or a function symbol.
struct GuestRange {
    u64 start;
    u64 end;
    std::string name;
};

const GuestRange* FindRange(const std::vector<GuestRange>& ranges, u64 address) {
    const auto next = std::ranges::upper_bound(ranges, address, {}, &GuestRange::start);
    if (next == ranges.begin()) {
        return nullptr;
    }
    const GuestRange& range = *std::prev(next);
    return address < range.end ? &range : nullptr;
}

/// How many of the largest entries, and how many bytes they span, reach each share of the total.
template <typename Entry, typename Samples, typename Bytes>
json CoverageCurve(const std::vector<Entry>& sorted, u64 total, Samples samples, Bytes bytes) {
    json curve = json::array();
    constexpr std::array<double, 5> SHARES{0.50, 0.80, 0.90, 0.95, 0.99};
    size_t share = 0;
    u64 cumulative = 0;
    u64 cumulative_bytes = 0;
    for (size_t i = 0; i < sorted.size() && share < SHARES.size(); ++i) {
        cumulative += samples(sorted[i]);
        cumulative_bytes += bytes(sorted[i]);
        while (share < SHARES.size() &&
               static_cast<double>(cumulative) >= SHARES[share] * static_cast<double>(total)) {
            curve.push_back({{"percent", SHARES[share] * 100.0},
                             {"count", i + 1},
                             {"guest_bytes", cumulative_bytes}});
            ++share;
        }
    }
    return curve;
}

json JitHot(ToolState& state, const json& params) {
    namespace HotMap = Dynarmic::A64::HotMap;
    if (!HotMap::Enabled()) {
        throw std::runtime_error{"the JIT host-code map is off: start the emulator with "
                                 "EDEN_JIT_HOT_MAP=1"};
    }
    const double seconds = Param<double>(params, "seconds", 20.0, 1.0, 60.0);
    const u32 period_us = Param<u32>(params, "period_us", 1000, 250, 100000);
    const size_t top = Param<size_t>(params, "top", 200, 1, 5000);
    Kernel::KProcess& process = Application(state);

    // The game's modules (rtld, main, subsdk*, sdk) and their exported function symbols.
    // FindModules names only modules whose read-only data carries a module path; this one also
    // takes every other executable Code / AliasCode region, named codeN in address order (the
    // loader maps rtld, main, subsdk*, sdk in that order).
    Loader::AppLoader::Modules found = Core::FindModules(&process);
    {
        auto& page_table = process.GetPageTable();
        u64 address = 0;
        size_t index = 0;
        for (;;) {
            Kernel::KMemoryInfo info{};
            Kernel::Svc::PageInfo page_info{};
            if (R_FAILED(page_table.QueryInfo(std::addressof(info), std::addressof(page_info),
                                              Kernel::KProcessAddress{address}))) {
                break;
            }
            const Kernel::Svc::MemoryInfo svc = info.GetSvcMemoryInfo();
            if (svc.permission == Kernel::Svc::MemoryPermission::ReadExecute &&
                (svc.state == Kernel::Svc::MemoryState::Code ||
                 svc.state == Kernel::Svc::MemoryState::AliasCode)) {
                found.try_emplace(svc.base_address, fmt::format("code{}", index));
                ++index;
            }
            const u64 next = svc.base_address + svc.size;
            if (next <= address) {
                break;
            }
            address = next;
        }
    }
    std::vector<GuestRange> modules;
    std::vector<GuestRange> functions;
    for (const auto& [base, name] : found) {
        const u64 end = GetInteger(Core::GetModuleEnd(&process, Kernel::KProcessAddress{base}));
        modules.push_back({base, (std::max)(end, base), name});
        for (const auto& [symbol, where] :
             Core::Symbols::GetSymbols(base, process.GetMemory(), process.Is64Bit())) {
            const auto [offset, size] = where;
            if (offset != 0 && size != 0) {
                functions.push_back({base + offset, base + offset + size, symbol});
            }
        }
    }
    std::ranges::sort(modules, {}, &GuestRange::start);
    std::ranges::sort(functions, {}, &GuestRange::start);
    // A module whose end could not be read (or that overlaps the next one) ends where the next
    // one starts.
    for (size_t i = 0; i < modules.size(); ++i) {
        const u64 next = i + 1 < modules.size() ? modules[i + 1].start : modules[i].end;
        if (modules[i].end <= modules[i].start || modules[i].end > next) {
            modules[i].end = next;
        }
    }

    struct ThreadCounts {
        std::array<u64, 4> kinds{};
        std::unordered_map<u64, u64> outside_rips;
    };
    struct BlockCounts {
        u64 samples{};
        u64 guest_end{};
        u32 host_size{};
    };
    std::vector<ThreadCounts> per_thread;
    std::unordered_map<u64, BlockCounts> blocks;
    std::vector<HotMap::Hit> hits;
    std::vector<u64> rips;
    const HotMap::Stats before = HotMap::GetStats();
    const Nx2Sampler::RipRun run = Nx2Sampler::SampleRips(
        "CPUCore_0,CPUCore_1,CPUCore_2,CPUCore_3,CPUCore_4,CPUCore_5", seconds, period_us,
        [&](std::span<const Nx2Sampler::RipSample> batch) {
            rips.resize(batch.size());
            hits.resize(batch.size());
            for (size_t i = 0; i < batch.size(); ++i) {
                rips[i] = batch[i].rip;
            }
            HotMap::Resolve(rips, hits);
            for (size_t i = 0; i < batch.size(); ++i) {
                if (per_thread.size() <= batch[i].thread) {
                    per_thread.resize(batch[i].thread + 1);
                }
                ThreadCounts& counts = per_thread[batch[i].thread];
                const HotMap::Hit& hit = hits[i];
                ++counts.kinds[static_cast<size_t>(hit.kind)];
                if (hit.kind == HotMap::Kind::Outside) {
                    ++counts.outside_rips[batch[i].rip];
                } else if (hit.kind == HotMap::Kind::Block) {
                    BlockCounts& block = blocks[hit.guest_pc];
                    ++block.samples;
                    block.guest_end = (std::max)(block.guest_end, hit.guest_end);
                    block.host_size = (std::max)(block.host_size, hit.host_size);
                }
            }
        },
        [&state] { return state.stopping.load(std::memory_order_relaxed); });
    const HotMap::Stats after = HotMap::GetStats();
    if (state.stopping.load(std::memory_order_relaxed)) {
        throw std::runtime_error{"the game is shutting down"};
    }
    per_thread.resize(run.threads.size());

    const auto share = [](u64 part, u64 whole) {
        return whole > 0 ? 100.0 * static_cast<double>(part) / static_cast<double>(whole) : 0.0;
    };
    constexpr std::array<const char*, 4> KIND_NAMES{"outside", "prelude", "between_blocks",
                                                    "block"};
    std::unordered_map<u64, std::string> module_names;
    const auto host_module = [&](u64 rip) -> const std::string& {
        // Cache per 64 KiB: modules are at least that aligned.
        auto [it, inserted] = module_names.try_emplace(rip >> 16);
        if (inserted) {
            it->second = Nx2Sampler::HostModuleName(rip);
            if (it->second.empty()) {
                it->second = "(no module)";
            }
        }
        return it->second;
    };
    json threads = json::array();
    std::array<u64, 4> total_kinds{};
    std::unordered_map<std::string, u64> total_outside;
    for (size_t i = 0; i < run.threads.size(); ++i) {
        const ThreadCounts& counts = per_thread[i];
        u64 samples = 0;
        json entry = {{"thread", run.threads[i]}};
        for (size_t kind = 0; kind < KIND_NAMES.size(); ++kind) {
            samples += counts.kinds[kind];
            total_kinds[kind] += counts.kinds[kind];
        }
        entry["samples"] = samples;
        for (size_t kind = 0; kind < KIND_NAMES.size(); ++kind) {
            entry[KIND_NAMES[kind]] = counts.kinds[kind];
            entry[std::string{KIND_NAMES[kind]} + "_percent"] = share(counts.kinds[kind], samples);
        }
        std::unordered_map<std::string, u64> outside;
        for (const auto& [rip, count] : counts.outside_rips) {
            outside[host_module(rip)] += count;
        }
        std::vector<std::pair<std::string, u64>> sorted_outside(outside.begin(), outside.end());
        std::ranges::sort(sorted_outside, std::greater{}, &std::pair<std::string, u64>::second);
        json modules_json = json::array();
        for (const auto& [name, count] : sorted_outside) {
            total_outside[name] += count;
            if (modules_json.size() < 8) {
                modules_json.push_back({{"module", name}, {"samples", count}});
            }
        }
        entry["outside_by_host_module"] = std::move(modules_json);
        threads.push_back(std::move(entry));
    }
    u64 total_samples = 0;
    for (const u64 count : total_kinds) {
        total_samples += count;
    }
    json total = {{"samples", total_samples}};
    for (size_t kind = 0; kind < KIND_NAMES.size(); ++kind) {
        total[KIND_NAMES[kind]] = total_kinds[kind];
        total[std::string{KIND_NAMES[kind]} + "_percent"] = share(total_kinds[kind], total_samples);
    }
    {
        std::vector<std::pair<std::string, u64>> sorted(total_outside.begin(), total_outside.end());
        std::ranges::sort(sorted, std::greater{}, &std::pair<std::string, u64>::second);
        json list = json::array();
        for (const auto& [name, count] : sorted) {
            list.push_back({{"module", name}, {"samples", count}});
        }
        total["outside_by_host_module"] = std::move(list);
    }

    // Blocks, largest first.
    struct BlockRow {
        u64 guest_pc;
        BlockCounts counts;
    };
    std::vector<BlockRow> rows;
    rows.reserve(blocks.size());
    u64 in_block = 0;
    for (const auto& [pc, counts] : blocks) {
        rows.push_back({pc, counts});
        in_block += counts.samples;
    }
    std::ranges::sort(rows, [](const BlockRow& a, const BlockRow& b) {
        return a.counts.samples != b.counts.samples ? a.counts.samples > b.counts.samples
                                                    : a.guest_pc < b.guest_pc;
    });
    const auto guest_size = [](const BlockRow& row) {
        return row.counts.guest_end > row.guest_pc ? row.counts.guest_end - row.guest_pc : u64{0};
    };
    const auto where = [&](u64 address) {
        json out = json::object();
        if (const GuestRange* const module = FindRange(modules, address)) {
            out["module"] = module->name;
            out["module_offset"] = HexAddress(address - module->start);
        }
        if (const GuestRange* const function = FindRange(functions, address)) {
            out["function"] = Common::DemangleSymbol(function->name);
            out["function_offset"] = HexAddress(address - function->start);
        }
        return out;
    };
    json block_list = json::array();
    json block_file = json::array();
    u64 cumulative = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        const BlockRow& row = rows[i];
        cumulative += row.counts.samples;
        json entry = {
            {"guest_pc", HexAddress(row.guest_pc)},
            {"guest_end", HexAddress(row.counts.guest_end)},
            {"guest_size", guest_size(row)},
            {"host_size", row.counts.host_size},
            {"samples", row.counts.samples},
            {"percent", share(row.counts.samples, in_block)},
            {"cumulative_percent", share(cumulative, in_block)},
        };
        if (i < top) {
            entry.update(where(row.guest_pc));
            block_list.push_back(entry);
        }
        block_file.push_back(std::move(entry));
    }
    const json block_curve = CoverageCurve(
        rows, in_block, [](const BlockRow& row) { return row.counts.samples; }, guest_size);

    // Guest 4 KiB pages (by block start), modules and functions.
    struct Group {
        std::string name;
        u64 samples{};
        u64 blocks{};
        u64 bytes{};
    };
    const auto grouped = [&](auto key_of, auto name_of, auto bytes_of) {
        std::unordered_map<u64, Group> groups;
        for (const BlockRow& row : rows) {
            const std::optional<u64> key = key_of(row.guest_pc);
            if (!key) {
                continue;
            }
            auto [it, inserted] = groups.try_emplace(*key);
            if (inserted) {
                it->second.name = name_of(*key);
                it->second.bytes = bytes_of(*key);
            }
            it->second.samples += row.counts.samples;
            ++it->second.blocks;
        }
        std::vector<Group> sorted;
        for (auto& [key, group] : groups) {
            sorted.push_back(std::move(group));
        }
        std::ranges::sort(sorted, [](const Group& a, const Group& b) {
            return a.samples != b.samples ? a.samples > b.samples : a.name < b.name;
        });
        return sorted;
    };
    const auto group_list = [&](const std::vector<Group>& groups, size_t limit) {
        json list = json::array();
        for (size_t i = 0; i < groups.size() && i < limit; ++i) {
            list.push_back({{"name", groups[i].name},
                            {"samples", groups[i].samples},
                            {"percent", share(groups[i].samples, in_block)},
                            {"blocks", groups[i].blocks},
                            {"bytes", groups[i].bytes}});
        }
        return list;
    };
    const auto group_samples = [](const Group& group) { return group.samples; };
    const auto group_bytes = [](const Group& group) { return group.bytes; };

    const std::vector<Group> pages = grouped(
        [](u64 pc) { return std::optional<u64>{pc >> 12}; },
        [&](u64 page) {
            const u64 address = page << 12;
            const GuestRange* const module = FindRange(modules, address);
            return module != nullptr
                       ? fmt::format("{}+{:#x}", module->name, address - module->start)
                       : HexAddress(address);
        },
        [](u64) { return u64{4096}; });
    const std::vector<Group> by_module = grouped(
        [&](u64 pc) -> std::optional<u64> {
            const GuestRange* const module = FindRange(modules, pc);
            return module != nullptr ? module->start : (std::numeric_limits<u64>::max)();
        },
        [&](u64 base) {
            const GuestRange* const module = FindRange(modules, base);
            return module != nullptr ? fmt::format("{} ({:#x})", module->name, base)
                                     : std::string{"(no module)"};
        },
        [&](u64 base) {
            const GuestRange* const module = FindRange(modules, base);
            return module != nullptr ? module->end - module->start : u64{0};
        });
    const std::vector<Group> by_function = grouped(
        [&](u64 pc) -> std::optional<u64> {
            const GuestRange* const function = FindRange(functions, pc);
            return function != nullptr ? std::optional<u64>{function->start} : std::nullopt;
        },
        [&](u64 start) {
            const GuestRange* const function = FindRange(functions, start);
            return fmt::format("{} ({:#x})", Common::DemangleSymbol(function->name), start);
        },
        [&](u64 start) {
            const GuestRange* const function = FindRange(functions, start);
            return function->end - function->start;
        });
    u64 in_functions = 0;
    for (const Group& group : by_function) {
        in_functions += group.samples;
    }

    json module_ranges = json::array();
    for (const GuestRange& module : modules) {
        module_ranges.push_back({{"name", module.name},
                                 {"start", HexAddress(module.start)},
                                 {"end", HexAddress(module.end)}});
    }
    json result = {
        {"seconds", run.seconds},
        {"period_us", period_us},
        {"passes", run.passes},
        {"rate_hz_per_thread", run.seconds > 0.0 ? static_cast<double>(run.passes) / run.seconds
                                                 : 0.0},
        {"missed", run.missed},
        {"map", {{"spaces", after.spaces},
                 {"blocks", after.blocks},
                 {"clears_during_window", after.clears - before.clears}}},
        {"threads", std::move(threads)},
        {"total", std::move(total)},
        {"in_block_samples", in_block},
        {"distinct_blocks", rows.size()},
        {"block_curve", block_curve},
        {"blocks", std::move(block_list)},
        {"page_curve", CoverageCurve(pages, in_block, group_samples, group_bytes)},
        {"pages", group_list(pages, 50)},
        {"modules", group_list(by_module, 64)},
        {"module_ranges", std::move(module_ranges)},
        {"function_symbols", functions.size()},
        {"in_function_samples", in_functions},
        {"in_function_percent", share(in_functions, in_block)},
        {"function_curve", CoverageCurve(by_function, in_functions, group_samples, group_bytes)},
        {"functions", group_list(by_function, 100)},
        {"note", "Samples of the CPUCore_N host threads' instruction pointer. block: in a JIT "
                 "block's host code (guest_pc = the block's location PC, guest_end = one past "
                 "its last instruction); prelude / between_blocks: in a code space outside any "
                 "block (dispatcher, fast dispatch, memory thunks); outside: emulator code, the "
                 "OS or waiting. Pages and modules group blocks by their start PC; functions "
                 "come from the modules' exported symbols only. Curves: how many of the largest "
                 "entries, spanning how many guest bytes, reach that share of in-block samples. "
                 "clears_during_window > 0: some samples may be attributed to newer code."},
    };
    std::filesystem::create_directories(state.dump_directory);
    const auto path =
        state.dump_directory / fmt::format("jit_hot-{}-{:03}.json", state.session, ++state.samples);
    {
        json file = result;
        file["blocks"] = std::move(block_file);
        file["pages"] = group_list(pages, pages.size());
        file["functions"] = group_list(by_function, by_function.size());
        std::ofstream output{path};
        output << file.dump(1);
        if (!output) {
            throw std::runtime_error{fmt::format("cannot write {}", PathText(path))};
        }
    }
    result["file"] = PathText(path);
    return result;
}

json SchedTraceTool(ToolState& state, const json& params) {
    const double seconds = Param<double>(params, "seconds", 1.0, 0.1, 5.0);
    const bool events_in_result = params.value("events_in_result", true);
    // spans: true records every kind; span_kinds (bit 1 << kind) picks some: 6 = HLE and SVC only.
    const u32 span_kinds =
        params.value("spans", false) ? Param<u32>(params, "span_kinds", 7, 1, 7) : 0;
    const bool spans = span_kinds != 0;
    const GuestSnapshot first = SnapshotGuestThreads(state);
    Kernel::KernelCore& kernel = state.system.Kernel();
    json idle_ids = json::array();
    for (size_t core = 0; core < GUEST_CORES; ++core) {
        idle_ids.push_back(kernel.Scheduler(core).GetIdleThread()->GetThreadId());
    }
    SchedTrace::Start(span_kinds);
    const s64 start_ticks = static_cast<s64>(state.system.CoreTiming().GetClockTicks());
    const auto begin = std::chrono::steady_clock::now();
    const auto window = std::chrono::duration<double>(seconds);
    while (std::chrono::steady_clock::now() - begin < window &&
           !state.stopping.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(5ms);
    }
    const s64 end_ticks = static_cast<s64>(state.system.CoreTiming().GetClockTicks());
    const SchedTrace::Collected collected = SchedTrace::Stop();
    if (state.stopping.load(std::memory_order_relaxed)) {
        throw std::runtime_error{"the game is shutting down"};
    }
    const GuestSnapshot second = SnapshotGuestThreads(state);

    json names = json::object();
    for (const GuestSnapshot* snapshot : {&first, &second}) {
        for (const GuestThreadSample& sample : snapshot->threads) {
            names[std::to_string(sample.id)] =
                sample.name.empty() ? fmt::format("thread {}", sample.id) : sample.name;
        }
    }
    names["0"] = "(idle)";
    json cores = json::array();
    u64 total_events = 0;
    u64 total_dropped = 0;
    std::set<u16> hle_slots;
    for (size_t core = 0; core < GUEST_CORES; ++core) {
        json events = json::array();
        for (const SchedTrace::Switch& event : collected.switches[core]) {
            events.push_back(json::array({event.ticks - start_ticks, event.out_id, event.in_id}));
        }
        total_events += collected.switches[core].size();
        total_dropped += collected.dropped[core];
        json core_entry = {{"core", core},
                           {"switches", collected.switches[core].size()},
                           {"dropped", collected.dropped[core]},
                           {"events", std::move(events)}};
        if (spans) {
            json span_list = json::array();
            for (const SchedTrace::Span& span : collected.spans[core]) {
                const u16 kind = static_cast<u16>(span.kind);
                span_list.push_back(json::array({span.start - start_ticks, span.end - start_ticks,
                                                 span.thread_id, kind, span.id, span.arg0,
                                                 span.arg1}));
                if (span.kind == SchedTrace::SpanKind::Hle) {
                    hle_slots.insert(span.id);
                }
            }
            core_entry["spans_count"] = collected.spans[core].size();
            core_entry["spans_dropped"] = collected.spans_dropped[core];
            core_entry["spans"] = std::move(span_list);
        }
        cores.push_back(std::move(core_entry));
    }
    json slot_names = json::object();
    for (const u16 slot : hle_slots) {
        slot_names[std::to_string(slot)] = std::string{NvnHle::Bridge::Instance().SlotName(slot)};
    }
    json presents = json::array();
    for (const SchedTrace::Present& present : collected.presents) {
        presents.push_back(json::array(
            {present.ticks - start_ticks, present.count, present.thread_id, present.kind}));
    }
    // Host spans: steady_clock ns -> ticks from start_ticks, by a least-squares line through the
    // presents (recorded with both clocks); without two presents, by the window start.
    double slope = static_cast<double>(Core::Hardware::CNTFREQ) / 1e9;
    double intercept = -static_cast<double>(SchedTrace::SteadyNs(begin)) * slope;
    double max_residual_ticks = 0;
    const size_t pairs = (std::min)(collected.presents.size(), collected.present_host_ns.size());
    if (pairs >= 2) {
        double mx = 0, my = 0;
        for (size_t i = 0; i < pairs; ++i) {
            mx += static_cast<double>(collected.present_host_ns[i]);
            my += static_cast<double>(collected.presents[i].ticks - start_ticks);
        }
        mx /= static_cast<double>(pairs);
        my /= static_cast<double>(pairs);
        double sxy = 0, sxx = 0;
        for (size_t i = 0; i < pairs; ++i) {
            const double dx = static_cast<double>(collected.present_host_ns[i]) - mx;
            sxy += dx * (static_cast<double>(collected.presents[i].ticks - start_ticks) - my);
            sxx += dx * dx;
        }
        if (sxx > 0) {
            slope = sxy / sxx;
            intercept = my - slope * mx;
        }
        for (size_t i = 0; i < pairs; ++i) {
            const double fit =
                intercept + slope * static_cast<double>(collected.present_host_ns[i]);
            max_residual_ticks = (std::max)(
                max_residual_ticks,
                std::abs(fit - static_cast<double>(collected.presents[i].ticks - start_ticks)));
        }
    }
    json host_spans = json::array();
    for (const SchedTrace::HostSpan& span : collected.host_spans) {
        host_spans.push_back(json::array(
            {static_cast<s64>(intercept + slope * static_cast<double>(span.start_ns)),
             static_cast<s64>(intercept + slope * static_cast<double>(span.end_ns)),
             static_cast<u32>(span.kind), span.arg}));
    }
    json result = {
        {"seconds", seconds},
        {"ticks_per_second", Core::Hardware::CNTFREQ},
        {"start_ticks", start_ticks},
        {"window_ticks", end_ticks - start_ticks},
        {"switches", total_events},
        {"dropped", total_dropped},
        {"presents_dropped", collected.presents_dropped},
        {"idle_thread_ids", std::move(idle_ids)},
        {"threads", std::move(names)},
        {"presents", std::move(presents)},
        {"cores", std::move(cores)},
        {"note", "Times are CoreTiming ticks (19.2 MHz) from start_ticks. cores[].events: "
                 "[ticks, thread switched out, thread switched in]; 0 is the core's idle thread. "
                 "presents: [ticks, present count, guest thread id, kind]; kind 0 = the frame "
                 "was counted (nvn-hle window present), 1 = present pacing returned. Only the "
                 "nvn-hle present path records presents. With spans: cores[].spans: [start, end, "
                 "guest thread id, kind, id, arg0, arg1], kind 0 = JIT run (id 0), 1 = nvn-hle "
                 "bridge call (id = thunk slot, hle_slot_names), 2 = kernel SVC (id = SVC number, "
                 "args x0 x1; WaitSynchronization: first handle, count). A span is filed under "
                 "the core it ended on; spans are only in the file."},
    };
    if (spans) {
        result["hle_slot_names"] = std::move(slot_names);
    }
    // Host spans (renderer threads): [start, end, kind, arg]; kinds 0 replay idle, 1 replay step
    // (arg seq), 2 present blit (arg seq), 3 completion thread GPU wait (arg seq).
    result["host_spans_count"] = host_spans.size();
    result["host_spans_dropped"] = collected.host_spans_dropped;
    result["host_clock_max_residual_us"] =
        max_residual_ticks * 1e6 / static_cast<double>(Core::Hardware::CNTFREQ);
    result["host_spans"] = std::move(host_spans);
    std::filesystem::create_directories(state.dump_directory);
    const auto path = state.dump_directory /
                      fmt::format("sched_trace-{}-{:03}.json", state.session, ++state.samples);
    {
        std::ofstream output{path};
        output << result.dump();
        if (!output) {
            throw std::runtime_error{fmt::format("cannot write {}", PathText(path))};
        }
    }
    result.erase("host_spans");
    for (json& core : result["cores"]) {
        core.erase("spans");
        if (!events_in_result) {
            core.erase("events");
        }
    }
    result["file"] = PathText(path);
    return result;
}

json ReadMemory(ToolState& state, const json& params) {
    const u64 address = AddressParam(params, "address");
    const u64 size = ReadSize(params, address);
    const std::string as = params.value("as", std::string{"hex"});
    static_cast<void>(FormatMemory(std::span<const u8>{}, as));
    std::vector<u8> bytes(size);
    if (const auto unmapped = ReadGuestMemory(Application(state).GetMemory(), address, bytes)) {
        throw std::runtime_error{
            fmt::format("guest address {:#x} is not mapped (reading {} bytes at {:#x})", *unmapped,
                        size, address)};
    }
    json result = {{"address", HexAddress(address)}, {"size", size}};
    result.update(FormatMemory(bytes, as));
    return result;
}

/// For experiments only: off unless the emulator was started with EDEN_NX2_DEBUG_WRITE=1.
json WriteMemory(ToolState& state, const json& params) {
    if (!state.write_enabled) {
        throw std::runtime_error{"write_memory is off: start the emulator with "
                                 "EDEN_NX2_DEBUG_WRITE=1 to allow it"};
    }
    const u64 address = AddressParam(params, "address");
    std::vector<u8> bytes;
    if (params.contains("f32")) {
        for (const json& value : params.at("f32")) {
            const u32 bits = std::bit_cast<u32>(value.get<float>());
            for (int i = 0; i < 4; ++i) {
                bytes.push_back(static_cast<u8>(bits >> (i * 8)));
            }
        }
    } else {
        bytes = ParseHexBytes(params.at("hex").get<std::string>());
    }
    if (bytes.empty() || bytes.size() > MAX_WRITE) {
        throw std::runtime_error{fmt::format("write 1 to {} bytes", MAX_WRITE)};
    }
    if (address + bytes.size() < address) {
        throw std::runtime_error{"address + size wraps around"};
    }
    std::vector<u8> before(bytes.size());
    if (const auto unmapped = ReadGuestMemory(Application(state).GetMemory(), address, before)) {
        throw std::runtime_error{fmt::format("guest address {:#x} is not mapped", *unmapped)};
    }
    Application(state).GetMemory().WriteBlock(address, bytes.data(), bytes.size());
    LOG_INFO(Debug, "Debug server wrote {} bytes at {:#x}", bytes.size(), address);
    json result = {{"address", HexAddress(address)}, {"size", bytes.size()}};
    result["before"] = FormatMemory(before, "hex").at("hex");
    return result;
}

json SearchMemory(ToolState& state, const json& params) {
    const std::vector<u8> pattern = ParseHexBytes(params.at("pattern").get<std::string>());
    if (pattern.size() > 256) {
        throw std::runtime_error{"the pattern is longer than 256 bytes"};
    }
    Kernel::KProcess& process = Application(state);
    const std::string regions = params.value("regions", std::string{"heap"});
    u64 region_start = 0;
    u64 region_size = 0;
    if (regions == "heap") {
        region_start = GetInteger(process.GetPageTable().GetHeapRegionStart());
        region_size = process.GetPageTable().GetHeapRegionSize();
    } else if (regions == "all") {
        region_start = GetInteger(process.GetPageTable().GetAddressSpaceStart());
        region_size = process.GetPageTable().GetAddressSpaceSize();
    } else {
        throw std::runtime_error{fmt::format("regions \"{}\": want heap or all", regions)};
    }
    const u64 start = params.contains("start")
                          ? (std::max)(region_start, AddressParam(params, "start"))
                          : region_start;
    const u64 end = params.contains("end")
                        ? (std::min)(region_start + region_size, AddressParam(params, "end"))
                        : region_start + region_size;
    const size_t max_hits = Param<size_t>(params, "max_hits", 64, 1, 256);
    const int budget = Param<int>(params, "budget_ms", 2000, 1, 10000);
    const auto begin = std::chrono::steady_clock::now();
    const GuestSearch search =
        SearchGuestMemory(process.GetMemory(), pattern, start, (std::max)(start, end), max_hits,
                          begin + std::chrono::milliseconds{budget});
    json hits = json::array();
    for (const u64 hit : search.hits) {
        hits.push_back(HexAddress(hit));
    }
    json result = {
        {"hits", std::move(hits)},
        {"complete", search.complete},
        {"start", HexAddress(start)},
        {"end", HexAddress((std::max)(start, end))},
        {"scanned_bytes", search.scanned},
        {"elapsed_ms",
         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
             .count()},
    };
    if (!search.complete) {
        result["next"] = HexAddress(search.next);
    }
    return result;
}

json ReadGpuMemory(ToolState& state, const json& params) {
    if (!IsVulkan()) {
        throw std::runtime_error{"GPU memory reads need the Vulkan renderer"};
    }
    const u64 address = AddressParam(params, "gpu_address");
    const u64 size = ReadSize(params, address);
    const std::string as = params.value("as", std::string{"hex"});
    static_cast<void>(FormatMemory(std::span<const u8>{}, as));
    const u64 id = ++state.gpu_reads;
    const int timeout = Param<int>(params, "timeout_s", 5, 1, 60);
    VideoCore::Nx2Debug::RequestGpuRead({.id = id, .gpu_address = address, .size = size});
    const auto read = VideoCore::Nx2Debug::WaitGpuRead(id, std::chrono::seconds{timeout});
    if (!read) {
        VideoCore::Nx2Debug::CancelGpuRead(id);
        throw std::runtime_error{
            fmt::format("no frame was presented within {} s (paused or loading?)", timeout)};
    }
    if (!read->error.empty()) {
        throw std::runtime_error{read->error};
    }
    json result = {{"gpu_address", HexAddress(address)},
                   {"size", size},
                   {"address_space", read->address_space}};
    result.update(FormatMemory(read->bytes, as));
    return result;
}

/// Holds the steps on player 1 (or the handheld controller when player 1 is not connected) from
/// the next present on, and returns once they are done. The injector stops by itself after the
/// last step; leaving this function by any path removes it.
json RunInput(ToolState& state, const std::vector<InputStep>& steps, const json& params,
              const ClientGone& client_gone) {
    u64 total = 0;
    for (const InputStep& step : steps) {
        total += step.frames;
    }
    if (total > MAX_INPUT_FRAMES) {
        throw std::runtime_error{
            fmt::format("{} frames in total; at most {} per call", total, MAX_INPUT_FRAMES)};
    }
    auto& hid = state.system.HIDCore();
    HID::EmulatedController* controller = hid.GetEmulatedController(HID::NpadIdType::Player1);
    std::string name = "player1";
    if (!controller->IsConnected()) {
        controller = hid.GetEmulatedController(HID::NpadIdType::Handheld);
        name = "handheld";
    }
    if (!controller->IsConnected()) {
        throw std::runtime_error{"neither player 1 nor the handheld controller is connected"};
    }
    const int timeout = Param<int>(params, "timeout_s", static_cast<int>(total / 20 + 10), 1, 900);
    const u64 start = Nx2PresentCount() + 1;
    const auto script = std::make_shared<const std::vector<InputStep>>(steps);
    controller->SetInputInjector([script, start] {
        return InputAt(*script, start,
                       Nx2PresentCount());
    });
    // "async": return at once; the script releases the controller by itself after its last frame
    // (InputAt), so screenshots can be taken while it plays.
    if (params.contains("async") && params.at("async").is_boolean() && params.at("async").get<bool>()) {
        return {{"controller", name}, {"steps", steps.size()}, {"frames", total},
                {"first_present", start}, {"async", true}};
    }
    SCOPE_EXIT {
        controller->SetInputInjector({});
    };
    const auto begin = std::chrono::steady_clock::now();
    const auto deadline = begin + std::chrono::seconds{timeout};
    for (;;) {
        const u64 now = Nx2PresentCount();
        if (now >= start + total) {
            break;
        }
        if (state.stopping.load(std::memory_order_relaxed)) {
            throw std::runtime_error{"the game is shutting down; input released"};
        }
        if (client_gone()) {
            LOG_INFO(Debug, "Debug server client left during input; input released");
            throw std::runtime_error{"the client disconnected; input released"};
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error{fmt::format(
                "{} of {} frames were presented within {} s (paused or loading?); input released",
                now >= start ? now - start : 0, total, timeout)};
        }
        std::this_thread::sleep_for(2ms);
    }
    return {
        {"controller", name},
        {"steps", steps.size()},
        {"frames", total},
        {"first_present", start},
        {"elapsed_s",
         std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count()},
    };
}

json Input(ToolState& state, const json& params, const ClientGone& client_gone) {
    json step = params;
    step.erase("timeout_s");
    step.erase("async");
    return RunInput(state, {ParseInputStep(step)}, params, client_gone);
}

json InputSequence(ToolState& state, const json& params, const ClientGone& client_gone) {
    const json& list = params.at("steps");
    if (!list.is_array() || list.empty() || list.size() > MAX_INPUT_STEPS) {
        throw std::runtime_error{fmt::format("steps must be an array of 1 to {}", MAX_INPUT_STEPS)};
    }
    std::vector<InputStep> steps;
    for (size_t index = 0; index < list.size(); ++index) {
        try {
            steps.push_back(ParseInputStep(list[index]));
        } catch (const std::exception& error) {
            throw std::runtime_error{fmt::format("step {}: {}", index, error.what())};
        }
    }
    return RunInput(state, steps, params, client_gone);
}

Handlers MakeToolHandlers(const std::shared_ptr<ToolState>& state) {
    const auto handler = [&state](json (*tool)(ToolState&, const json&)) -> Handler {
        return
            [state, tool](const json& params, const ClientGone&) { return tool(*state, params); };
    };
    const auto waiting =
        [&state](json (*tool)(ToolState&, const json&, const ClientGone&)) -> Handler {
        return [state, tool](const json& params, const ClientGone& client_gone) {
            return tool(*state, params, client_gone);
        };
    };
    return {
        {"status", handler(Status)},
        {"log_tail", handler(LogTail)},
        {"capture_frame", handler(CaptureFrame)},
        {"sample", handler(SampleThreads)},
        {"guest_threads", handler(GuestThreads)},
        {"jit_hot", handler(JitHot)},
        {"sched_trace", handler(SchedTraceTool)},
        {"read_memory", handler(ReadMemory)},
        {"search_memory", handler(SearchMemory)},
        {"write_memory", handler(WriteMemory)},
        {"read_gpu_memory", handler(ReadGpuMemory)},
        {"input", waiting(Input)},
        {"input_sequence", waiting(InputSequence)},
    };
}

} // Anonymous namespace

u64 ParseShaderHash(std::string_view text) {
    const std::string_view digits =
        text.starts_with("0x") || text.starts_with("0X") ? text.substr(2) : text;
    u64 value = 0;
    const auto [end, error] =
        std::from_chars(digits.data(), digits.data() + digits.size(), value, 16);
    if (digits.size() != 16 || error != std::errc{} || end != digits.data() + digits.size()) {
        throw std::runtime_error{fmt::format(
            "bad hash \"{}\": want all 16 hex digits of a native program key (shader dump file "
            "name) or shader hash (frame dump \"shaders\"); a shorter one would match nothing",
            text)};
    }
    return value;
}

std::vector<u8> ParseHexBytes(std::string_view text) {
    std::string digits;
    const std::string_view body =
        text.starts_with("0x") || text.starts_with("0X") ? text.substr(2) : text;
    for (const char c : body) {
        if (c != ' ') {
            digits.push_back(c);
        }
    }
    std::vector<u8> bytes(digits.size() / 2);
    for (size_t i = 0; i < bytes.size(); ++i) {
        const auto [end, error] =
            std::from_chars(digits.data() + i * 2, digits.data() + i * 2 + 2, bytes[i], 16);
        if (error != std::errc{} || end != digits.data() + i * 2 + 2) {
            bytes.clear();
            break;
        }
    }
    if (bytes.empty() || digits.size() % 2 != 0) {
        throw std::runtime_error{fmt::format(
            "bad pattern \"{}\": want hex byte pairs in memory order (little endian: 1.0f is "
            "0000803f)",
            text)};
    }
    return bytes;
}

json FormatMemory(std::span<const u8> bytes, std::string_view as) {
    if (as == "hex") {
        std::string hex;
        hex.reserve(bytes.size() * 2);
        for (const u8 byte : bytes) {
            fmt::format_to(std::back_inserter(hex), "{:02x}", byte);
        }
        return {{"hex", std::move(hex)}};
    }
    const size_t element = as == "u32" || as == "f32" ? 4 : as == "u64" ? 8 : 0;
    if (element == 0) {
        throw std::runtime_error{fmt::format("as \"{}\": want hex, u32, u64 or f32", as)};
    }
    if (bytes.size() % element != 0) {
        throw std::runtime_error{fmt::format("size must be a multiple of {} for {}", element, as)};
    }
    json values = json::array();
    for (size_t offset = 0; offset < bytes.size(); offset += element) {
        if (element == 8) {
            u64 value;
            std::memcpy(&value, bytes.data() + offset, sizeof(value));
            values.push_back(HexAddress(value));
            continue;
        }
        u32 value;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        if (as == "u32") {
            values.push_back(value);
            continue;
        }
        const float number = std::bit_cast<float>(value);
        if (std::isnan(number)) {
            values.push_back("nan");
        } else if (std::isinf(number)) {
            values.push_back(number < 0 ? "-inf" : "inf");
        } else {
            values.push_back(number);
        }
    }
    return {{std::string{as}, std::move(values)}};
}

std::optional<u64> ReadGuestMemory(Memory::Memory& memory, u64 address, std::span<u8> out) {
    for (size_t offset = 0; offset < out.size();) {
        const u64 current = address + offset;
        const size_t length = (std::min)(static_cast<size_t>(Memory::YUZU_PAGESIZE -
                                                             (current & Memory::YUZU_PAGEMASK)),
                                         out.size() - offset);
        const u8* const source = GuestByte(memory, current);
        if (source == nullptr) {
            return current;
        }
        std::memcpy(out.data() + offset, source, length);
        offset += length;
    }
    return std::nullopt;
}

GuestSearch SearchGuestMemory(Memory::Memory& memory, std::span<const u8> pattern, u64 start,
                              u64 end, size_t max_hits,
                              std::chrono::steady_clock::time_point deadline) {
    GuestSearch search{.hits = {}, .next = end, .complete = true, .scanned = 0};
    // The bytes of a run of mapped pages that a match starting there could still need.
    std::vector<u8> window;
    window.reserve(Memory::YUZU_PAGESIZE + pattern.size());
    for (u64 address = start; address < end;) {
        if (std::chrono::steady_clock::now() >= deadline) {
            search.next = address - window.size();
            search.complete = false;
            return search;
        }
        const u64 page_end = (std::min)((address | Memory::YUZU_PAGEMASK) + 1, end);
        const u8* const source = GuestByte(memory, address);
        if (source == nullptr) {
            window.clear();
            address = page_end;
            continue;
        }
        const u64 window_start = address - window.size();
        const size_t length = page_end - address;
        window.insert(window.end(), source, source + length);
        search.scanned += length;
        for (size_t offset = 0; offset + pattern.size() <= window.size(); ++offset) {
            const void* const found = std::memchr(window.data() + offset, pattern[0],
                                                  window.size() - pattern.size() + 1 - offset);
            if (found == nullptr) {
                break;
            }
            offset = static_cast<size_t>(static_cast<const u8*>(found) - window.data());
            if (std::memcmp(window.data() + offset, pattern.data(), pattern.size()) != 0) {
                continue;
            }
            search.hits.push_back(window_start + offset);
            if (search.hits.size() >= max_hits) {
                search.next = window_start + offset + 1;
                search.complete = false;
                return search;
            }
        }
        window.erase(window.begin(),
                     window.end() - static_cast<std::ptrdiff_t>(
                                        (std::min)(pattern.size() - 1, window.size())));
        address = page_end;
    }
    return search;
}

InputStep ParseInputStep(const json& step) {
    if (!step.is_object()) {
        throw std::runtime_error{"a step is an object"};
    }
    for (const auto& [key, value] : step.items()) {
        if (key != "buttons" && key != "left_stick" && key != "right_stick" && key != "frames") {
            throw std::runtime_error{fmt::format(
                "unknown key \"{}\": want buttons, left_stick, right_stick, frames", key)};
        }
    }
    InputStep result{.input = {}, .frames = 0};
    const json& frames = step.at("frames");
    if (!frames.is_number_integer() || frames.get<s64>() < 1 ||
        frames.get<u64>() > MAX_INPUT_FRAMES) {
        throw std::runtime_error{fmt::format("frames must be 1 to {}", MAX_INPUT_FRAMES)};
    }
    result.frames = frames.get<u64>();
    for (const json& button : step.value("buttons", json::array())) {
        const std::string text = button.is_string() ? button.get<std::string>() : button.dump();
        const auto it = std::ranges::find(INPUT_BUTTONS, std::string_view{text},
                                          &std::pair<std::string_view, HID::NpadButton>::first);
        if (it == INPUT_BUTTONS.end()) {
            std::string names;
            for (const auto& [known, _] : INPUT_BUTTONS) {
                names += names.empty() ? "" : ", ";
                names += known;
            }
            throw std::runtime_error{fmt::format("unknown button {}: want one of {}", text, names)};
        }
        result.input.buttons |= it->second;
    }
    const auto stick = [&step](const char* key) -> std::optional<HID::AnalogStickState> {
        if (!step.contains(key)) {
            return std::nullopt;
        }
        const json& value = step.at(key);
        const auto axis = [&](size_t index) {
            if (!value.is_array() || value.size() != 2 || !value[index].is_number() ||
                std::abs(value[index].get<double>()) > 1.0) {
                throw std::runtime_error{fmt::format("{} must be [x, y] with both in -1..1", key)};
            }
            return static_cast<s32>(std::lround(value[index].get<double>() * 0x7fff));
        };
        return HID::AnalogStickState{.x = axis(0), .y = axis(1)};
    };
    result.input.left = stick("left_stick");
    result.input.right = stick("right_stick");
    return result;
}

std::optional<HID::InjectedInput> InputAt(std::span<const InputStep> steps, u64 start, u64 now) {
    if (now < start) {
        return std::nullopt;
    }
    u64 end = start;
    for (const InputStep& step : steps) {
        end += step.frames;
        if (now < end) {
            return step.input;
        }
    }
    return std::nullopt;
}

Tools::Tools(System& system, std::string title, u64 program_id, std::filesystem::path directory)
    : state{
          std::make_shared<ToolState>(system, std::move(title), program_id, std::move(directory))} {
    if (const char* directory_override = std::getenv("EDEN_NX2_DEBUG_DIR");
        directory_override && *directory_override) {
        state->dump_directory = directory_override;
    }
    VideoCore::Nx2Debug::BeginSession();
    Common::Log::SetRingCapacity(RING_LINES);
}
void Tools::NoteLogFilter(std::string filter) {
    state->log_filter = std::move(filter);
}
void Tools::Stop() {
    state->stopping.store(true, std::memory_order_relaxed);
    VideoCore::Nx2Debug::EndSession();
}
Tools::~Tools() {
    Stop();
    Common::Log::SetRingCapacity(0);
}
Handlers Tools::MakeHandlers() const {
    return MakeToolHandlers(state);
}
} // namespace Core::Nx2Debug
