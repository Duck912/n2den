// SPDX-License-Identifier: GPL-3.0-or-later
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include "video_core/n2den_debug_control.h"

namespace D = VideoCore::N2DenDebug;

TEST_CASE("n2Den diagnostics are inert unless activated", "[n2den-debug]") {
    D::SetActive(false);
    const std::array<u64, 1> shaders{0x1111};
    CHECK_FALSE(D::RecordDraw(0x2222, shaders, false));
    CHECK(D::Status().at("draw_attempts") == 0);
}

TEST_CASE("n2Den rejection matches host hashes and native keys and clears", "[n2den-debug]") {
    D::SetActive(true);
    const std::array<u8, 16> bytes{};
    const std::array<u64, 1> hashes{0x1234};
    D::RecordProgram(0x5678, 0x1000, 6, bytes);
    CHECK(D::AssociateHash(0x1234, 0x1000) == 0x1234);
    CHECK_FALSE(D::RecordDraw(0x2222, hashes, true));
    D::SetRejected({0x5678}, false);
    CHECK(D::RecordDraw(0x2222, hashes, true));
    D::SetRejected({}, false);
    CHECK_FALSE(D::RecordDraw(0x2222, hashes, true));
    D::SetRejected({0x1234}, false);
    CHECK(D::RecordDraw(0x2222, hashes, false));
    D::SetRejected({}, false);
    D::SetRejected({0x2222}, true);
    CHECK(D::RecordDraw(0x2222, hashes, false));
    CHECK_FALSE(D::RecordDraw(0x9999, hashes, false));
    D::SetRejected({}, true);
    CHECK_FALSE(D::RecordDraw(0x2222, hashes, false));
    D::FrameEnd();
    const auto draws = D::Draws(0, 256);
    CHECK(draws.at("total") == 7);
    CHECK(draws.at("draws").size() == 7);
    const auto programs = D::Programs(0, 256, false);
    REQUIRE(programs.at("total") == 1);
    CHECK(programs.at("shaders").at(0).at("key") == "0000000000005678");
    CHECK(D::ProgramBytes(0x5678).size() == bytes.size());
    CHECK_THROWS(D::ProgramBytes(0x9999));
    D::SetActive(false);
}

TEST_CASE("n2Den failed programs retain their reason and snapshots paginate", "[n2den-debug]") {
    D::SetActive(true);
    const std::array<u8, 16> bytes{};
    D::RecordProgram(0xabcd, 0x1000, 6, bytes);
    D::NoteResult(0xabcd, "failed", "unsupported opcode", 32);
    D::RecordProgram(0xbeef, 0x2000, 5, bytes);
    D::NoteResult(0xbeef, "lower", "", 0);
    const auto failed = D::Programs(0, 1, true);
    CHECK(failed.at("total") == 1);
    CHECK(failed.at("shaders").at(0).at("error_offset") == 32);
    CHECK(D::Programs(1, 1, false).at("shaders").size() == 1);
    CHECK(D::Programs(100, 1, false).at("shaders").empty());
    D::SetActive(false);
}
