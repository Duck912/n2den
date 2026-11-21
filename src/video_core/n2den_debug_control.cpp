// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
#include <thread>
#include <atomic>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include "video_core/n2den_debug_control.h"

namespace VideoCore::N2DenDebug {
namespace {
using nlohmann::json;
constexpr size_t MaxPrograms = 8192;
constexpr size_t MaxBytes = 128 * 1024 * 1024;
constexpr size_t MaxDraws = 8192;
struct Program {
    u64 address{};
    u32 stage{};
    std::vector<u8> bytes;
    std::set<u64> hashes;
    std::string frontend{"captured"};
    std::string error;
    u32 error_offset{};
    std::string native_error; ///< Native front end refusal, kept when the lowering succeeds
    u32 native_error_offset{};
};
struct Draw {
    u64 pipeline;
    std::vector<u64> hashes;
    bool compute;
    bool rejected;
};
std::atomic_bool active{false};
std::mutex mutex;
std::map<u64, Program> programs;
std::map<u64, u64> addresses;
std::map<u64, u64> native_keys;
std::set<u64> rejected_shaders, rejected_pipelines;
std::vector<Draw> current_draws, last_draws;
u64 frames{}, attempts{}, skipped{}, generation{}, applied_generation{}, frame_attempts{}, last_attempts{};
size_t stored_bytes{}, dropped_programs{};
constexpr size_t MaxComputeStates = 4096;
std::map<u64, json> compute_states;
std::set<u64> watched_compute, rebuild_compute;
std::mutex reader_mutex;
GpuReader gpu_reader;
std::map<u64, FrontendOverride> frontend_overrides;
std::set<u64> pending_invalidations;
std::set<u64> dump_requests;     // native keys
std::set<u64> dumped_hashes;     // host hashes already dumped for a request
std::map<u64, json> dumps;       // key -> {hash -> entry}

// Activity timeline (own lock: hot paths must not contend with the inventory lock).
using Clock = std::chrono::steady_clock;
struct ActivityEvent {
    std::string label;
    u64 detail;
    double start_ms;
    double duration_ms;
    size_t thread;
};
std::mutex activity_mutex;
const Clock::time_point activity_epoch = Clock::now();
std::deque<ActivityEvent> activity_events;
std::map<size_t, std::pair<std::string, Clock::time_point>> activity_active; // per thread
std::map<std::pair<u32, u64>, std::pair<u64, Clock::time_point>> channel_waits;
std::deque<double> frame_times_ms;
constexpr size_t MaxActivityEvents = 4096;
constexpr double ActivityThresholdMs = 2.0;
double SinceEpoch(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(t - activity_epoch).count();
}
size_t ThreadTag() {
    return std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xffff;
}
void PushActivity(ActivityEvent event) {
    if (activity_events.size() >= MaxActivityEvents) activity_events.pop_front();
    activity_events.push_back(std::move(event));
}

// Host shader hashes for an ID that is either a hash or a native program key.
std::vector<u64> ResolveHashes(u64 id) {
    std::vector<u64> out{id};
    if (const auto it = programs.find(id); it != programs.end()) {
        out.insert(out.end(), it->second.hashes.begin(), it->second.hashes.end());
    }
    return out;
}

std::string Hex(u64 value) { return fmt::format("{:016x}", value); }
json HexList(const std::set<u64>& values) {
    auto out = json::array();
    for (const auto value : values) out.push_back(Hex(value));
    return out;
}
json ProgramJson(u64 key, const Program& program) {
    return {{"key", Hex(key)}, {"program_base", Hex(program.address)}, {"stage", program.stage},
            {"instruction_bytes", program.bytes.size()}, {"shader_hashes", HexList(program.hashes)},
            {"frontend", program.frontend}, {"error", program.error},
            {"error_offset", program.error_offset}, {"native_error", program.native_error},
            {"native_error_offset", program.native_error_offset}};
}
} // namespace

bool Enabled() noexcept { return active.load(std::memory_order_relaxed); }
void SetActive(bool enabled) {
    active.store(false, std::memory_order_relaxed);
    std::scoped_lock lock{mutex};
    programs.clear(); addresses.clear(); native_keys.clear();
    rejected_shaders.clear(); rejected_pipelines.clear();
    current_draws.clear(); last_draws.clear();
    frames = attempts = skipped = generation = applied_generation = frame_attempts = last_attempts = 0;
    stored_bytes = dropped_programs = 0;
    compute_states.clear(); watched_compute.clear(); rebuild_compute.clear();
    frontend_overrides.clear(); pending_invalidations.clear();
    dump_requests.clear(); dumped_hashes.clear(); dumps.clear();
    {
        std::scoped_lock activity_lock{activity_mutex};
        activity_events.clear(); activity_active.clear(); channel_waits.clear(); frame_times_ms.clear();
    }
    active.store(enabled, std::memory_order_relaxed);
}
void RecordProgram(u64 key, u64 address, u32 stage, std::span<const u8> bytes) {
    if (!Enabled()) return;
    std::scoped_lock lock{mutex};
    if (!Enabled()) return;
    if (programs.contains(key)) { addresses[address] = key; return; }
    if (programs.size() >= MaxPrograms || bytes.size() > MaxBytes - stored_bytes) {
        ++dropped_programs;
        return;
    }
    auto& p = programs[key];
    p.address = address;
    p.stage = stage;
    p.bytes.assign(bytes.begin(), bytes.end());
    stored_bytes += bytes.size();
    // Bound metadata as well as raw code when guest addresses are reused or churn.
    if (addresses.size() >= MaxPrograms * 4) addresses.clear();
    addresses[address] = key;
}
std::vector<u64> NativeKeys(std::span<const u64> hashes) {
    std::scoped_lock lock{mutex};
    std::vector<u64> keys;
    for (u64 hash : hashes) {
        const auto it = native_keys.find(hash);
        keys.push_back(it == native_keys.end() ? 0 : it->second);
    }
    return keys;
}
u64 AssociateHash(u64 hash, u64 address) {
    if (!Enabled()) return hash;
    std::scoped_lock lock{mutex};
    const auto it = addresses.find(address);
    if (it != addresses.end() && programs.contains(it->second)) {
        if (native_keys.size() >= MaxPrograms * 4) native_keys.clear();
        native_keys[hash] = it->second;
        programs.at(it->second).hashes.insert(hash);
    }
    return hash;
}
void NoteResult(u64 key, std::string_view frontend, std::string_view error, u32 error_offset) {
    if (!Enabled()) return;
    std::scoped_lock lock{mutex};
    const auto it = programs.find(key);
    if (it == programs.end()) return;
    it->second.frontend = frontend;
    it->second.error = std::string{error.substr(0, 4096)};
    it->second.error_offset = error_offset;
}
void NoteNativeReject(u64 key, std::string_view error, u32 error_offset) {
    if (!Enabled()) return;
    std::scoped_lock lock{mutex};
    const auto it = programs.find(key);
    if (it == programs.end()) return;
    it->second.native_error = std::string{error.substr(0, 4096)};
    it->second.native_error_offset = error_offset;
}
bool RecordDraw(u64 pipeline, std::span<const u64> shaders, bool compute) {
    if (!Enabled()) return false;
    std::scoped_lock lock{mutex};
    if (!Enabled()) return false;
    bool reject = rejected_pipelines.contains(pipeline);
    for (const auto hash : shaders) {
        if (hash == 0) continue;
        reject |= rejected_shaders.contains(hash);
        const auto it = native_keys.find(hash);
        if (it != native_keys.end()) reject |= rejected_shaders.contains(it->second);
    }
    ++attempts;
    ++frame_attempts;
    if (reject) ++skipped;
    applied_generation = generation;
    if (current_draws.size() < MaxDraws) {
        current_draws.push_back({pipeline, {shaders.begin(), shaders.end()}, compute, reject});
    }
    return reject;
}
void FrameEnd() {
    if (!Enabled()) return;
    std::scoped_lock lock{mutex};
    if (!Enabled()) return;
    last_draws.swap(current_draws);
    current_draws.clear();
    last_attempts = frame_attempts;
    frame_attempts = 0;
    ++frames;
    std::scoped_lock activity_lock{activity_mutex};
    if (frame_times_ms.size() >= 4096) frame_times_ms.pop_front();
    frame_times_ms.push_back(SinceEpoch(Clock::now()));
}
json Status() {
    std::scoped_lock lock{mutex};
    return {{"enabled", Enabled()}, {"renderer_frames", frames}, {"draw_attempts", attempts},
            {"skipped", skipped}, {"programs", programs.size()}, {"stored_bytes", stored_bytes},
            {"dropped_programs", dropped_programs}, {"generation", generation},
            {"applied_generation", applied_generation}, {"rejected_shaders", HexList(rejected_shaders)},
            {"rejected_pipelines", HexList(rejected_pipelines)}};
}
json Programs(size_t offset, size_t limit, bool failed_only) {
    std::scoped_lock lock{mutex};
    auto entries = json::array();
    size_t total = 0;
    for (const auto& [key, program] : programs) {
        if (failed_only && program.error.empty()) continue;
        if (total >= offset && entries.size() < std::min(limit, size_t{256}))
            entries.push_back(ProgramJson(key, program));
        ++total;
    }
    const auto count = entries.size();
    return {{"shaders", std::move(entries)}, {"total", total}, {"offset", offset},
            {"has_more", offset < total && count < total - offset}};
}
json Draws(size_t offset, size_t limit) {
    std::scoped_lock lock{mutex};
    auto entries = json::array();
    const auto end = offset < last_draws.size()
        ? offset + std::min(std::min(limit, size_t{256}), last_draws.size() - offset) : offset;
    for (size_t i = offset; i < end; ++i) {
        const auto& draw = last_draws[i];
        auto shaders = json::array();
        for (const auto hash : draw.hashes) {
            if (!hash) continue;
            const auto key = native_keys.find(hash);
            shaders.push_back({{"hash", Hex(hash)},
                               {"key", key == native_keys.end() ? json(nullptr) : json(Hex(key->second))}});
        }
        entries.push_back({{"index", i}, {"pipeline", Hex(draw.pipeline)},
                           {"kind", draw.compute ? "compute" : "graphics"},
                           {"rejected", draw.rejected}, {"shaders", std::move(shaders)}});
    }
    const auto count = entries.size();
    return {{"frame", frames}, {"draws", std::move(entries)}, {"total", last_draws.size()},
            {"attempts_in_frame", last_attempts}, {"truncated", last_attempts > last_draws.size()},
            {"offset", offset}, {"has_more", offset < last_draws.size() && count < last_draws.size() - offset},
            {"note", "Attempted pipeline-backed work, not confirmed GPU execution; pipeline IDs group draws."}};
}
json SetRejected(const std::vector<u64>& hashes, bool pipelines) {
    if (hashes.size() > 256) throw std::runtime_error{"at most 256 rejection IDs"};
    std::scoped_lock lock{mutex};
    auto& target = pipelines ? rejected_pipelines : rejected_shaders;
    target = {hashes.begin(), hashes.end()};
    ++generation;
    return {{"rejected", HexList(target)}, {"generation", generation},
            {"note", "Applies to subsequently prepared work; queued GPU work may still complete. Clearing does not reconstruct skipped compute outputs."}};
}
std::vector<u8> ProgramBytes(u64 key) {
    std::scoped_lock lock{mutex};
    const auto it = programs.find(key);
    if (it == programs.end()) throw std::runtime_error{"unknown native program key"};
    return it->second.bytes;
}


void NoteCompute(u64 hash, const json& update) {
    if (!Enabled()) return;
    std::scoped_lock lock{mutex};
    if (!compute_states.contains(hash) && compute_states.size() >= MaxComputeStates) return;
    auto& state = compute_states[hash];
    state["hash"] = Hex(hash);
    if (const auto it = native_keys.find(hash); it != native_keys.end()) state["key"] = Hex(it->second);
    for (const auto& [name, value] : update.items()) state[name] = value;
}
json ComputeState(u64 id) {
    std::scoped_lock lock{mutex};
    auto out = json::array();
    for (const u64 hash : ResolveHashes(id)) {
        if (const auto it = compute_states.find(hash); it != compute_states.end()) out.push_back(it->second);
    }
    if (out.empty()) throw std::runtime_error{"no compute state for that hash or key (not dispatched yet?)"};
    return {{"states", std::move(out)}};
}
bool WatchedCompute(u64 hash) {
    if (!Enabled()) return false;
    std::scoped_lock lock{mutex};
    if (watched_compute.contains(hash)) return true;
    // First dispatch of a program loaded from the disk cache: no snapshot yet.
    const auto it = compute_states.find(hash);
    return it == compute_states.end() || !it->second.contains("cbufs");
}
json SetWatchedCompute(const std::vector<u64>& ids) {
    std::scoped_lock lock{mutex};
    watched_compute.clear();
    for (const u64 id : ids) for (const u64 hash : ResolveHashes(id)) watched_compute.insert(hash);
    return {{"watched", HexList(watched_compute)},
            {"note", "Watched programs refresh their cbuf/slot snapshot on every dispatch."}};
}
bool TakeComputeRebuild(u64 hash) {
    if (!Enabled()) return false;
    std::scoped_lock lock{mutex};
    return rebuild_compute.erase(hash) != 0;
}
json RequestComputeRebuild(const std::vector<u64>& ids) {
    std::scoped_lock lock{mutex};
    for (const u64 id : ids) for (const u64 hash : ResolveHashes(id)) rebuild_compute.insert(hash);
    return {{"pending", HexList(rebuild_compute)},
            {"note", "Each pipeline is retranslated against live state at its next dispatch; read compute_state after."}};
}
void SetGpuReader(GpuReader reader) {
    std::scoped_lock lock{reader_mutex};
    gpu_reader = std::move(reader);
}
json ReadGpu(u64 address, size_t size) {
    if (size == 0 || size > 64 * 1024) throw std::runtime_error{"size must be 1..65536"};
    std::vector<u8> bytes(size);
    {
        std::scoped_lock lock{reader_mutex};
        if (!gpu_reader) throw std::runtime_error{"no renderer attached"};
        if (!gpu_reader(address, bytes)) throw std::runtime_error{"range not fully mapped in the current channel"};
    }
    auto words = json::array();
    for (size_t i = 0; i + 4 <= size; i += 4) {
        u32 word;
        std::memcpy(&word, bytes.data() + i, 4);
        words.push_back(fmt::format("{:08x}", word));
    }
    std::string hex;
    hex.reserve(size * 2);
    for (const u8 b : bytes) hex += fmt::format("{:02x}", b);
    return {{"address", Hex(address)}, {"size", size}, {"u32", std::move(words)}, {"hex", std::move(hex)}};
}

FrontendOverride ForcedFrontend(u64 key) {
    if (!Enabled()) return FrontendOverride::None;
    std::scoped_lock lock{mutex};
    const auto it = frontend_overrides.find(key);
    return it == frontend_overrides.end() ? FrontendOverride::None : it->second;
}
static json FrontendsUnlocked() {
    auto out = json::object();
    for (const auto& [key, mode] : frontend_overrides) {
        out[Hex(key)] = mode == FrontendOverride::Native ? "native" : "lower";
    }
    return {{"overrides", std::move(out)}};
}
json SetFrontend(const std::vector<u64>& keys, FrontendOverride mode) {
    std::scoped_lock lock{mutex};
    auto invalidated = json::array();
    for (const u64 key : keys) {
        if (mode == FrontendOverride::None) {
            frontend_overrides.erase(key);
        } else {
            frontend_overrides[key] = mode;
        }
        for (const u64 hash : ResolveHashes(key)) {
            pending_invalidations.insert(hash);
            invalidated.push_back(Hex(hash));
        }
    }
    return {{"overrides", FrontendsUnlocked()["overrides"]}, {"invalidated_hashes", std::move(invalidated)},
            {"note", "Applied when the GPU thread next looks the programs up; read 'shaders' for the new frontend."}};
}
json Frontends() {
    std::scoped_lock lock{mutex};
    return FrontendsUnlocked();
}
std::vector<u64> TakeInvalidations() {
    if (!Enabled()) return {};
    std::scoped_lock lock{mutex};
    std::vector<u64> out{pending_invalidations.begin(), pending_invalidations.end()};
    pending_invalidations.clear();
    return out;
}

json RequestDump(const std::vector<u64>& keys) {
    std::scoped_lock lock{mutex};
    for (const u64 key : keys) {
        dump_requests.insert(key);
        for (const u64 hash : ResolveHashes(key)) dumped_hashes.erase(hash);
    }
    return {{"requested", HexList(dump_requests)},
            {"note", "Each program is dumped the next time a pipeline using it is translated or drawn; read 'dumps'."}};
}
std::optional<u64> PendingDump(u64 hash) {
    if (!Enabled()) return std::nullopt;
    std::scoped_lock lock{mutex};
    if (dump_requests.empty() || dumped_hashes.contains(hash)) return std::nullopt;
    if (dump_requests.contains(hash)) return hash;
    const auto it = native_keys.find(hash);
    if (it != native_keys.end() && dump_requests.contains(it->second)) return it->second;
    return std::nullopt;
}
bool AnyPendingDump(std::span<const u64> hashes) {
    for (const u64 hash : hashes) {
        if (hash != 0 && PendingDump(hash)) return true;
    }
    return false;
}
void NoteDump(u64 key, u64 hash, const json& entry) {
    std::scoped_lock lock{mutex};
    dumped_hashes.insert(hash);
    dumps[key][Hex(hash)] = entry;
}
json Dumps() {
    std::scoped_lock lock{mutex};
    auto out = json::object();
    for (const auto& [key, entries] : dumps) out[Hex(key)] = entries;
    return {{"requested", HexList(dump_requests)}, {"dumps", std::move(out)}};
}

ActivityScope::ActivityScope(const char* label_, u64 detail_)
    : label{label_}, detail{detail_}, armed{Enabled()} {
    if (!armed) return;
    start = Clock::now();
    std::scoped_lock lock{activity_mutex};
    activity_active[ThreadTag()] = {label, start};
}
ActivityScope::~ActivityScope() {
    if (!armed) return;
    const auto end = Clock::now();
    std::scoped_lock lock{activity_mutex};
    activity_active.erase(ThreadTag());
    const double ms = std::chrono::duration<double, std::milli>(end - start).count();
    if (ms >= ActivityThresholdMs) PushActivity({label, detail, SinceEpoch(start), ms, ThreadTag()});
}
void NoteActivity(const char* label, u64 detail, Clock::time_point start_, Clock::time_point end) {
    if (!Enabled()) return;
    const double ms = std::chrono::duration<double, std::milli>(end - start_).count();
    if (ms < ActivityThresholdMs) return;
    std::scoped_lock lock{activity_mutex};
    PushActivity({label, detail, SinceEpoch(start_), ms, ThreadTag()});
}
void ChannelWaitBegin(u32 channel, u64 address, u64 required) {
    if (!Enabled()) return;
    std::scoped_lock lock{activity_mutex};
    channel_waits[{channel, address}] = {required, Clock::now()};
}
void ChannelWaitEnd(u32 channel, u64 address) {
    if (!Enabled()) return;
    const auto end = Clock::now();
    std::scoped_lock lock{activity_mutex};
    const auto it = channel_waits.find({channel, address});
    if (it == channel_waits.end()) return;
    const double ms = std::chrono::duration<double, std::milli>(end - it->second.second).count();
    if (ms >= ActivityThresholdMs) {
        PushActivity({fmt::format("channel {} semaphore {:#x} >= {}", channel, address, it->second.first),
                      address, SinceEpoch(it->second.second), ms, ThreadTag()});
    }
    channel_waits.erase(it);
}
json Activity(double min_ms, size_t limit, bool clear) {
    std::scoped_lock lock{activity_mutex};
    const auto now = Clock::now();
    auto active_now = json::array();
    for (const auto& [thread, entry] : activity_active) {
        active_now.push_back({{"label", entry.first}, {"thread", thread},
                              {"ms", std::chrono::duration<double, std::milli>(now - entry.second).count()}});
    }
    for (const auto& [key, entry] : channel_waits) {
        active_now.push_back({{"label", fmt::format("channel {} semaphore {:#x} >= {}", key.first, key.second, entry.first)},
                              {"ms", std::chrono::duration<double, std::milli>(now - entry.second).count()}});
    }
    auto events = json::array();
    for (auto it = activity_events.rbegin(); it != activity_events.rend() && events.size() < limit; ++it) {
        if (it->duration_ms < min_ms) continue;
        events.push_back({{"label", it->label}, {"detail", Hex(it->detail)}, {"start_ms", it->start_ms},
                          {"ms", it->duration_ms}, {"thread", it->thread}});
    }
    auto slow_frames = json::array();
    for (size_t i = 1; i < frame_times_ms.size(); ++i) {
        const double gap = frame_times_ms[i] - frame_times_ms[i - 1];
        if (gap >= std::max(min_ms, 50.0)) slow_frames.push_back({{"end_ms", frame_times_ms[i]}, {"ms", gap}});
    }
    json result{{"now_ms", SinceEpoch(now)}, {"active", std::move(active_now)},
                {"events", std::move(events)}, {"slow_frames", std::move(slow_frames)},
                {"frames_recorded", frame_times_ms.size()},
                {"note", "Times are ms since the debug server started; events newest first. gpu_fence_wait = the GPU "
                         "thread waiting for host GPU work to finish (GPU busy)."}};
    if (clear) {
        activity_events.clear();
        frame_times_ms.clear();
    }
    return result;
}
namespace {
struct QmdEvent {
    u64 seq;
    u64 frame;
    u64 qmd;
    u64 space;
    u64 program; // 0 when no descriptor was read
    std::array<u32, 3> grid;
    s32 value;
    std::string event;
};
constexpr size_t MaxQmdEvents = 1 << 16;
std::deque<QmdEvent> qmd_events;
u64 qmd_seq{};
struct QmdProgramStats {
    std::map<std::string, u64> events;
    std::array<u32, 3> last_grid{};
    u64 last_frame{};
};
std::map<u64, QmdProgramStats> qmd_programs; // by program address, whole session
std::map<u64, u64> qmd_program_of;            // QMD address -> program address (last load)
} // namespace

void RecordQmd(std::string_view event, u64 qmd_address, u64 address_space, s32 value,
               const u32* qmd) {
    if (!Enabled()) return;
    std::scoped_lock lock{mutex};
    QmdEvent entry{.seq = ++qmd_seq, .frame = frames, .qmd = qmd_address, .space = address_space,
                   .program = 0, .grid = {}, .value = value, .event = std::string{event}};
    if (qmd) {
        entry.program = (u64{qmd[49] & 0x1ffff} << 32) | qmd[48];
        entry.grid = {qmd[12], qmd[13] & 0xffff, qmd[14] & 0xffff};
        if (qmd_program_of.size() >= 65536) qmd_program_of.clear();
        qmd_program_of[qmd_address] = entry.program;
    } else if (const auto it = qmd_program_of.find(qmd_address); it != qmd_program_of.end()) {
        entry.program = it->second;
    }
    if (entry.program != 0 && (qmd_programs.size() < 8192 || qmd_programs.contains(entry.program))) {
        auto& stats = qmd_programs[entry.program];
        ++stats.events[entry.event];
        if (qmd) stats.last_grid = entry.grid;
        stats.last_frame = frames;
    }
    if (qmd_events.size() >= MaxQmdEvents) qmd_events.pop_front();
    qmd_events.push_back(std::move(entry));
}

json QmdTrace(size_t limit, bool last_frame_only, bool clear) {
    std::scoped_lock lock{mutex};
    const auto key_of = [](u64 program) -> json {
        // Program addresses in a QMD point at the program header; RecordProgram stores the
        // program base the shader environment read, which is the same address.
        const auto it = addresses.find(program);
        if (it == addresses.end()) return nullptr;
        return fmt::format("{:016x}", it->second);
    };
    json summary = json::array();
    for (const auto& [program, stats] : qmd_programs) {
        summary.push_back({{"program", fmt::format("{:#x}", program)}, {"key", key_of(program)},
                           {"events", stats.events},
                           {"last_grid", stats.last_grid}, {"last_frame", stats.last_frame}});
    }
    // QMDs whose last event leaves them waiting (loaded and scheduled with dependencies left).
    std::map<u64, const QmdEvent*> last;
    for (const auto& e : qmd_events) last[e.qmd] = &e;
    json waiting = json::array();
    for (const auto& [qmd, e] : last) {
        if ((e->event == "load" || e->event == "count") && e->value > 0) {
            waiting.push_back({{"qmd", fmt::format("{:#x}", qmd)},
                               {"program", fmt::format("{:#x}", e->program)},
                               {"key", key_of(e->program)}, {"event", e->event},
                               {"dependencies", e->value}, {"frame", e->frame}});
        }
    }
    json events = json::array();
    const u64 want_frame = frames > 0 ? frames - 1 : 0;
    for (const auto& e : qmd_events) {
        if (last_frame_only && e.frame != want_frame) continue;
        if (events.size() >= limit) break;
        json row{{"seq", e.seq}, {"frame", e.frame}, {"qmd", fmt::format("{:#x}", e.qmd)},
                 {"event", e.event}, {"value", e.value}};
        if (e.program) {
            row["program"] = fmt::format("{:#x}", e.program);
            row["key"] = key_of(e.program);
        }
        if (e.event == "load" || e.event == "run" || e.event == "zero_grid" || e.event == "skip") {
            row["grid"] = e.grid;
        }
        events.push_back(std::move(row));
    }
    json result{{"frame", frames}, {"programs", std::move(summary)}, {"waiting", std::move(waiting)},
                {"events", std::move(events)}, {"stored_events", qmd_events.size()}};
    if (clear) {
        qmd_events.clear();
        qmd_programs.clear();
    }
    return result;
}

} // namespace VideoCore::N2DenDebug
