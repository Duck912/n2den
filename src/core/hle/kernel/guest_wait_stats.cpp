// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "common/logging.h"
#include "core/hle/kernel/guest_wait_stats.h"
#include "core/hle/kernel/k_synchronization_object.h"

namespace Kernel::GuestWaitStats {
namespace {
using Clock = std::chrono::steady_clock;

constexpr size_t MaxCauses = 8192;
constexpr auto ReportEvery = std::chrono::seconds{4};

struct Key {
    u64 thread;
    Kind kind;
    u32 sub;
    u64 value;
    const char* type_name;
    bool operator<(const Key& other) const {
        return std::tie(thread, kind, sub, value) <
               std::tie(other.thread, other.kind, other.sub, other.value);
    }
};

struct Totals {
    u64 count{};
    u64 ns{};
};

std::mutex names_mutex;
std::unordered_map<u64, std::string> names;

std::mutex stats_mutex;
std::map<Key, Totals> causes;
std::map<std::pair<std::string, u32>, Totals> service_commands;
u64 overflow_ns{};
Clock::time_point window_start = Clock::now();

std::string Describe(const Key& key) {
    switch (key.kind) {
    case Kind::Sync:
    case Kind::SyncTimeout: {
        std::string name;
        {
            std::scoped_lock lock{names_mutex};
            if (const auto it = names.find(key.value); it != names.end()) {
                name = it->second;
            }
        }
        if (name.empty()) {
            name = fmt::format("{}@{:x}", key.type_name ? key.type_name : "object", key.value & 0xffffff);
        }
        return fmt::format("{}{}{}", key.kind == Kind::SyncTimeout ? "timeout " : "wait ", name,
                           key.sub > 1 ? fmt::format(" (of {})", key.sub) : std::string{});
    }
    case Kind::Arbiter:
        return fmt::format("arbiter{} {:#x}", key.sub, key.value);
    case Kind::CondVar:
        return fmt::format("condvar {:#x}", key.value);
    case Kind::Mutex:
        return fmt::format("mutex {:#x}", key.value);
    case Kind::Sleep:
        return fmt::format("sleep {}us", key.value);
    case Kind::Ipc: {
        std::scoped_lock lock{names_mutex};
        const auto it = names.find(key.value);
        return fmt::format("ipc {}", it != names.end() ? it->second : fmt::format("session@{:x}", key.value & 0xffffff));
    }
    }
    return "?";
}

void Report(Clock::time_point now) {
    const double seconds = std::chrono::duration<double>(now - window_start).count();
    window_start = now;
    std::map<u64, Totals> threads;
    for (const auto& [key, totals] : causes) {
        auto& thread = threads[key.thread];
        thread.count += totals.count;
        thread.ns += totals.ns;
    }
    // The threads that block least run most: the game's main and render threads.
    std::vector<std::pair<u64, Totals>> sorted(threads.begin(), threads.end());
    std::ranges::sort(sorted, [](const auto& a, const auto& b) { return a.second.ns < b.second.ns; });
    const auto ms_per_s = [seconds](u64 ns) { return static_cast<double>(ns) / 1e6 / seconds; };
    std::string text;
    for (size_t shown = 0; shown < sorted.size() && shown < 10; ++shown) {
        const auto& [thread_id, total] = sorted[shown];
        std::vector<std::pair<const Key*, Totals>> top;
        for (const auto& [key, totals] : causes) {
            if (key.thread == thread_id) {
                top.emplace_back(&key, totals);
            }
        }
        std::ranges::sort(top, [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
        text += fmt::format("\n  thread {}: blocked {:.0f} ms/s, {:.0f} calls/s:", thread_id,
                            ms_per_s(total.ns), static_cast<double>(total.count) / seconds);
        for (size_t i = 0; i < top.size() && i < 6; ++i) {
            const Totals& totals = top[i].second;
            text += fmt::format(" {} {:.1f} ms/s ({:.0f}/s, {:.2f} ms each);", Describe(*top[i].first),
                                ms_per_s(totals.ns), static_cast<double>(totals.count) / seconds,
                                static_cast<double>(totals.ns) / 1e6 / static_cast<double>(totals.count));
        }
    }
    std::vector<std::pair<const std::pair<std::string, u32>*, Totals>> commands;
    for (const auto& [key, totals] : service_commands) {
        commands.emplace_back(&key, totals);
    }
    std::ranges::sort(commands, [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
    text += "\n  service commands:";
    for (size_t i = 0; i < commands.size() && i < 8; ++i) {
        const Totals& totals = commands[i].second;
        text += fmt::format(" {}#{} {:.1f} ms/s ({:.0f}/s, {:.2f} ms each);", commands[i].first->first,
                            commands[i].first->second, ms_per_s(totals.ns),
                            static_cast<double>(totals.count) / seconds,
                            static_cast<double>(totals.ns) / 1e6 / static_cast<double>(totals.count));
    }
    LOG_INFO(Kernel, "Guest waits over {:.1f} s, {} threads, least blocked first{}:{}", seconds,
             threads.size(), overflow_ns ? fmt::format(" ({:.0f} ms/s not kept)", ms_per_s(overflow_ns)) : "",
             text);
    causes.clear();
    service_commands.clear();
    overflow_ns = 0;
}
} // Anonymous namespace

bool Enabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_GUEST_WAIT_STATS");
        return value && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

void NameObject(const void* object, std::string_view name) {
    if (!Enabled() || object == nullptr) {
        return;
    }
    std::scoped_lock lock{names_mutex};
    names.insert_or_assign(reinterpret_cast<u64>(object), std::string{name});
}

void ForgetObject(const void* object) {
    if (!Enabled()) {
        return;
    }
    std::scoped_lock lock{names_mutex};
    names.erase(reinterpret_cast<u64>(object));
}

void NoteServiceCommand(std::string_view service, u32 command, u64 ns) {
    std::scoped_lock lock{stats_mutex};
    if (service_commands.size() >= MaxCauses) {
        return;
    }
    auto& totals = service_commands[{std::string{service}, command}];
    ++totals.count;
    totals.ns += ns;
}

Scope::Scope(u64 thread_id_) : thread_id{thread_id_}, armed{Enabled()} {
    if (armed) {
        start = Clock::now();
    }
}

void Scope::SetObject(Kind kind_, KSynchronizationObject* object, u32 count) {
    kind = kind_;
    value = reinterpret_cast<u64>(object);
    sub = count;
    type_name = object ? object->GetTypeName() : nullptr;
}

Scope::~Scope() {
    if (!armed) {
        return;
    }
    const auto now = Clock::now();
    const u64 ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count());
    std::scoped_lock lock{stats_mutex};
    const Key key{thread_id, kind, sub, value, type_name};
    auto it = causes.find(key);
    if (it == causes.end()) {
        if (causes.size() >= MaxCauses) {
            overflow_ns += ns;
        } else {
            it = causes.emplace(key, Totals{}).first;
        }
    }
    if (it != causes.end()) {
        ++it->second.count;
        it->second.ns += ns;
    }
    if (now - window_start >= ReportEvery) {
        Report(now);
    }
}

} // namespace Kernel::GuestWaitStats
