// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
#include <string>
#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include "video_core/native_census.h"

namespace C = VideoCore::NativeCensus;

namespace {
const nlohmann::json* FindRow(const nlohmann::json& snapshot, std::string_view site,
                              std::string_view reason) {
    for (const auto& row : snapshot.at("rejections")) {
        if (row.at("site").get<std::string>() == site && row.at("reason").get<std::string>() == reason) {
            return &row;
        }
    }
    return nullptr;
}
} // Anonymous namespace

TEST_CASE("Native census counts programs once and events every time", "[native-census]") {
    C::Reset();
    C::NoteReject(C::Site::Native, 0xaaaa, 1, 0x28, "unsupported native opcode 0x28");
    C::NoteReject(C::Site::Native, 0xaaaa, 1, 0x28, "unsupported native opcode 0x28");
    C::NoteReject(C::Site::Native, 0xbbbb, 4, 0x28, "unsupported native opcode 0x28");
    const auto snapshot = C::Snapshot();
    const auto* row = FindRow(snapshot, "native", "unsupported native opcode 0x28");
    REQUIRE(row != nullptr);
    CHECK(row->at("programs") == 2);
    CHECK(row->at("events") == 3);
    CHECK(row->at("stage_mask") == ((1U << 1) | (1U << 4)));
    CHECK(row->at("opcode") == "0x028");
}

TEST_CASE("Native census groups refusals that differ only in their instruction words", "[native-census]") {
    C::Reset();
    C::NoteReject(C::Site::Native, 0x1, 0, 0xa02, "native constant buffer address [00027a02 06001300 00000f00 000fe200]");
    C::NoteReject(C::Site::Native, 0x2, 0, 0xa02, "native constant buffer address [00107a02 06001300 00000f00 000fe200]");
    const auto snapshot = C::Snapshot();
    const auto* row = FindRow(snapshot, "native", "native constant buffer address");
    REQUIRE(row != nullptr);
    CHECK(row->at("programs") == 2);
    CHECK(row->at("example_words").get<std::string>() == "00027a02 06001300 00000f00 000fe200");
    CHECK(snapshot.at("rejections").size() == 1);
}

TEST_CASE("Native census does not count a thrown lowering failure twice", "[native-census]") {
    C::Reset();
    C::NoteReject(C::Site::Lowering, 0xcccc, 0, 0x925, "unsupported opcode", true);
    C::NoteBuildFailure(0x1000, "Native shader at 0x80001000, instruction 0x40: unsupported opcode");
    C::NoteBuildFailure(0x2000, "Unsupported native graphics shader stage 1");
    C::NoteBuildFailure(0x3000, "Native compute trap at 0x80012340 failed");
    const auto snapshot = C::Snapshot();
    CHECK(FindRow(snapshot, "lowering", "unsupported opcode") != nullptr);
    CHECK(FindRow(snapshot, "build", "Unsupported native graphics shader stage 1") != nullptr);
    // Long addresses are masked so failures group by cause.
    CHECK(FindRow(snapshot, "build", "Native compute trap at 0x# failed") != nullptr);
    CHECK(snapshot.at("rejections").size() == 3);
}

TEST_CASE("Native census counts dropped work", "[native-census]") {
    C::Reset();
    C::NoteDrop(C::Drop::GraphicsRejected);
    C::NoteDrop(C::Drop::GraphicsRejected);
    C::NoteDrop(C::Drop::PipelineBuilding);
    const auto drops = C::Snapshot().at("drops");
    CHECK(drops.at("graphics_rejected") == 2);
    CHECK(drops.at("pipeline_building") == 1);
    CHECK(drops.at("compute_rejected") == 0);
    C::Reset();
}
