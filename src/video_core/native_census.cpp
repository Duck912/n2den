// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include "common/logging.h"
#include "video_core/native_census.h"

namespace VideoCore::NativeCensus {
namespace {
using nlohmann::json;

constexpr size_t MaxRows = 1024;
constexpr size_t MaxSeen = 1 << 16;
constexpr size_t MaxReason = 256;

constexpr std::array<const char*, static_cast<size_t>(Site::Count)> SiteNames{
    "native", "lowering", "build", "collected"};
constexpr std::array<const char*, static_cast<size_t>(Drop::Count)> DropNames{
    "graphics_rejected", "compute_rejected", "pipeline_translating", "pipeline_building"};

using RowKey = std::tuple<Site, u32, std::string>; // site, opcode, reason
struct Row {
    u64 programs{};
    u64 events{};
    u32 stage_mask{};
    u64 example{};
    std::string example_words; ///< Instruction words of the example, when the reason had them
};

std::mutex mutex;
std::map<RowKey, Row> rows;
std::set<std::tuple<Site, u64, u32, std::string>> seen;
u64 overflow_events{};
std::array<std::atomic<u64>, static_cast<size_t>(Drop::Count)> drops{};
thread_local bool failure_noted{};

// Build-site messages carry addresses and offsets; masking long numbers groups them by cause.
std::string Normalize(std::string_view what) {
    std::string out;
    out.reserve(std::min(what.size(), MaxReason));
    for (size_t i = 0; i < what.size() && out.size() < MaxReason;) {
        if (what[i] == '0' && i + 1 < what.size() && (what[i + 1] == 'x' || what[i + 1] == 'X')) {
            size_t end = i + 2;
            while (end < what.size() && std::isxdigit(static_cast<unsigned char>(what[end]))) ++end;
            if (end - i - 2 >= 4) {
                out += "0x#";
                i = end;
                continue;
            }
        }
        if (std::isdigit(static_cast<unsigned char>(what[i]))) {
            size_t end = i;
            while (end < what.size() && std::isdigit(static_cast<unsigned char>(what[end]))) ++end;
            if (end - i >= 5) {
                out += '#';
                i = end;
                continue;
            }
            out.append(what.substr(i, end - i));
            i = end;
            continue;
        }
        out += what[i++];
    }
    return out;
}

// Native refusals end with the instruction's words ("reason [00027a02 06001300 ...]"); they are
// split off so that one cause is one row, and the first example's words are kept.
std::pair<std::string, std::string> SplitWords(std::string_view reason) {
    const size_t open = reason.rfind(" [");
    if (open == std::string_view::npos || reason.empty() || reason.back() != ']') {
        return {std::string{reason}, {}};
    }
    const std::string_view words = reason.substr(open + 2, reason.size() - open - 3);
    const bool hex_words = !words.empty() && std::ranges::all_of(words, [](char c) {
        return c == ' ' || std::isxdigit(static_cast<unsigned char>(c));
    });
    if (!hex_words) {
        return {std::string{reason}, {}};
    }
    return {std::string{reason.substr(0, open)}, std::string{words}};
}

void Record(Site site, u64 program, u32 stage, u32 opcode, std::string reason,
            std::string words = {}) {
    std::scoped_lock lock{mutex};
    auto it = rows.find(RowKey{site, opcode, reason});
    if (it == rows.end()) {
        if (rows.size() >= MaxRows) {
            ++overflow_events;
            return;
        }
        it = rows.emplace(RowKey{site, opcode, reason}, Row{}).first;
        it->second.example = program;
        it->second.example_words = std::move(words);
    }
    Row& row = it->second;
    ++row.events;
    if (stage < 32) row.stage_mask |= 1U << stage;
    // Once the set is full, every event counts as a new program: an overcount, never a miss.
    if (seen.size() >= MaxSeen || seen.emplace(site, program, opcode, std::move(reason)).second) {
        ++row.programs;
    }
}
} // Anonymous namespace

void NoteReject(Site site, u64 program, u32 stage, u32 opcode, std::string_view reason,
                bool thrown) {
    failure_noted = thrown;
    auto [text, words] = SplitWords(reason);
    if (text.size() > MaxReason) text.resize(MaxReason);
    Record(site, program, stage, opcode, std::move(text), std::move(words));
}

void NoteBuildFailure(u64 program, std::string_view what) {
    if (std::exchange(failure_noted, false)) return;
    Record(Site::Build, program, NoStage, NoOpcode, Normalize(what));
}

void NoteDrop(Drop drop) {
    drops[static_cast<size_t>(drop)].fetch_add(1, std::memory_order_relaxed);
}

bool CollectAll() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NATIVE_CENSUS");
        return value && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

json Snapshot() {
    auto drop_json = json::object();
    for (size_t i = 0; i < drops.size(); ++i) {
        drop_json[DropNames[i]] = drops[i].load(std::memory_order_relaxed);
    }
    std::scoped_lock lock{mutex};
    std::vector<std::pair<const RowKey*, const Row*>> sorted;
    sorted.reserve(rows.size());
    for (const auto& [key, row] : rows) sorted.emplace_back(&key, &row);
    std::ranges::sort(sorted, [](const auto& a, const auto& b) {
        if (std::get<0>(*a.first) != std::get<0>(*b.first)) {
            return std::get<0>(*a.first) < std::get<0>(*b.first);
        }
        return a.second->programs > b.second->programs;
    });
    auto entries = json::array();
    for (const auto& [key, row] : sorted) {
        const auto& [site, opcode, reason] = *key;
        entries.push_back({{"site", SiteNames[static_cast<size_t>(site)]},
                           {"opcode", opcode == NoOpcode ? json(nullptr) : json(fmt::format("{:#05x}", opcode))},
                           {"reason", reason},
                           {"programs", row->programs},
                           {"events", row->events},
                           {"stage_mask", row->stage_mask},
                           {"example", fmt::format("{:016x}", row->example)},
                           {"example_words", row->example_words}});
    }
    return {{"rejections", std::move(entries)}, {"drops", std::move(drop_json)},
            {"overflow_events", overflow_events}, {"collect_all", CollectAll()}};
}

void LogSummary() {
    const json snapshot = Snapshot();
    const auto& entries = snapshot["rejections"];
    const auto& drop_json = snapshot["drops"];
    bool any_drop = false;
    for (const auto& [name, value] : drop_json.items()) any_drop |= value.get<u64>() != 0;
    if (entries.empty() && !any_drop) return;
    LOG_INFO(HW_GPU, "Native census: {} rejection rows, drops: {}", entries.size(), drop_json.dump());
    for (const auto& entry : entries) {
        LOG_INFO(HW_GPU, "  {:<9} op {:>5}: {:>5} programs, {:>7} events, stages {:#06x}, e.g. {}: {}",
                 entry["site"].get<std::string>(),
                 entry["opcode"].is_null() ? std::string{"-"} : entry["opcode"].get<std::string>(),
                 entry["programs"].get<u64>(), entry["events"].get<u64>(),
                 entry["stage_mask"].get<u32>(), entry["example"].get<std::string>(),
                 entry["reason"].get<std::string>());
    }
}

void Reset() {
    std::scoped_lock lock{mutex};
    rows.clear();
    seen.clear();
    overflow_events = 0;
    for (auto& drop : drops) drop.store(0, std::memory_order_relaxed);
    failure_noted = false;
}

} // namespace VideoCore::NativeCensus
