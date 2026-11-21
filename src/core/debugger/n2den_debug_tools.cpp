// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later
// Transport and game-frame screenshot technique adapted from the nx2 debug server (nx2_debug_server.h).
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <algorithm>
#include <memory>
#include <vector>
#include <stdexcept>
#include <thread>
#include <fmt/format.h>
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/n2den_debug_buttons.h"
#include "common/scm_rev.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/stb.h"
#include "core/core.h"
#include "core/debugger/n2den_debug_tools.h"
#include "core/debugger/nx2_debug_server.h"
#include "core/debugger/nx2_debug_tools.h"
#include "core/frontend/framebuffer_layout.h"
#include "core/hle/service/am/service/native_video_decoder.h"
#include "video_core/gpu.h"
#include "video_core/n2den_debug_control.h"
#include "video_core/native_census.h"
#include "video_core/renderer_base.h"

namespace Core::Nx2Debug {
namespace {
using nlohmann::json;
namespace D = VideoCore::N2DenDebug;
using namespace std::chrono_literals;

u64 ParseHash(const json& value) {
    if (!value.is_string()) throw std::runtime_error{"ID must be exactly 16 hexadecimal digits"};
    const auto text = value.get<std::string>();
    u64 hash{};
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), hash, 16);
    if (text.size() != 16 || ec != std::errc{} || end != text.data() + text.size())
        throw std::runtime_error{"ID must be exactly 16 hexadecimal digits (no 0x prefix)"};
    return hash;
}
size_t Bound(const json& params, const char* name, size_t fallback, size_t minimum, size_t maximum) {
    if (!params.contains(name)) return fallback;
    const auto& value = params.at(name);
    if ((!value.is_number_unsigned() && !value.is_number_integer()) ||
        (value.is_number_integer() && !value.is_number_unsigned() && value.get<s64>() < 0))
        throw std::runtime_error{fmt::format("{} must be a nonnegative integer", name)};
    const auto result = value.get<u64>();
    if (result < minimum || result > maximum)
        throw std::runtime_error{fmt::format("{} outside {}..{}", name, minimum, maximum)};
    return static_cast<size_t>(result);
}
json Reject(const json& params, bool pipelines) {
    const char* name = pipelines ? "pipelines" : "hashes";
    const auto& values = params.at(name);
    if (!values.is_array() || values.size() > 256)
        throw std::runtime_error{"expected an array of at most 256 IDs"};
    std::vector<u64> hashes;
    for (const auto& value : values) hashes.push_back(ParseHash(value));
    return D::SetRejected(hashes, pipelines);
}
std::vector<u64> ParseIds(const json& values) {
    if (!values.is_array() || values.size() > 256)
        throw std::runtime_error{"expected an array of at most 256 IDs"};
    std::vector<u64> ids;
    for (const auto& value : values) ids.push_back(ParseHash(value));
    return ids;
}
void AppendPng(void* context, void* data, int size) {
    auto& out = *static_cast<std::vector<u8>*>(context);
    const auto* bytes = static_cast<const u8*>(data);
    out.insert(out.end(), bytes, bytes + size);
}
} // namespace

struct N2DenService::Impl {
    Core::System& system;
    const u64 program_id;
    const std::string title;
    const std::filesystem::path dump_dir;
    std::atomic_bool stopping{false};
    std::unique_ptr<Tools> upstream_tools;
    std::unique_ptr<Server> server;

    Impl(Core::System& system_, u64 id, std::string title_)
        : system{system_}, program_id{id}, title{std::move(title_)},
          dump_dir{Common::FS::GetEdenPath(Common::FS::EdenPath::DumpDir) /
                   "n2den-debug" / fmt::format("{:016x}", id)} {
        const char* config = std::getenv("EDEN_NX2_DEBUG_SERVER");
        std::string error;
        const auto endpoint = ParseEndpoint(config ? config : "", error);
        if (!endpoint || endpoint->port == 0)
            throw std::runtime_error{error.empty() ? "debug port must be 1..65535" : error};
        upstream_tools = std::make_unique<Tools>(system, title, program_id, dump_dir);
        Handlers handlers = upstream_tools->MakeHandlers();
        handlers["status"] =
            [this, upstream_status = handlers.at("status")](const json& p, const ClientGone& gone) {
                auto params = p;
                if (!params.contains("fps_window_ms"))
                    params["fps_window_ms"] = 0;
                auto status = upstream_status(params, gone);
                status.update(D::Status());
                status["rejected"] = status["rejected_shaders"];
                status["title"] = title;
                status["program_id"] = fmt::format("{:016x}", program_id);
                status["build"] = std::string{Common::g_build_fullname};
                status["scm_rev"] = std::string{Common::g_scm_rev};
                status["dump_dir"] = dump_dir.generic_string();
                return status;
            };
        handlers["shaders"] = [](const json& p, const ClientGone&) {
            auto result = D::Programs(Bound(p, "offset", 0, 0, 1000000), Bound(p, "limit", 64, 1, 256),
                                      p.value("failed_only", false));
            result["census"] = VideoCore::NativeCensus::Snapshot();
            return result;
        };
        handlers["draws"] = [](const json& p, const ClientGone&) {
            return D::Draws(Bound(p, "offset", 0, 0, 1000000), Bound(p, "limit", 64, 1, 256));
        };
        handlers["reject_shaders"] = [](const json& p, const ClientGone&) { return Reject(p, false); };
        handlers["reject_draws"] = [](const json& p, const ClientGone&) { return Reject(p, true); };
        handlers["dump_shader"] = [this](const json& p, const ClientGone&) {
            const u64 key = ParseHash(p.at("key"));
            const auto bytes = D::ProgramBytes(key);
            std::filesystem::create_directories(dump_dir);
            const auto path = dump_dir / fmt::format("{:016x}.bin", key);
            std::ofstream file{path, std::ios::binary | std::ios::trunc};
            file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
            file.close();
            if (!file) throw std::runtime_error{"cannot write shader dump"};
            return json{{"key", fmt::format("{:016x}", key)}, {"path", path.generic_string()},
                        {"bytes", bytes.size()}, {"format", "raw 16-byte Ampere instructions; no shader header or resource bindings"}};
        };
        handlers["screenshot"] = [this](const json& p, const ClientGone& gone) { return Screenshot(p, gone); };
        handlers["flash_burst"] = [this](const json& p, const ClientGone& gone) { return FlashBurst(p, gone); };
        // [DEBUG] compute inspection and live controls (read-only except rebuild/log filter).
        handlers["compute_state"] = [](const json& p, const ClientGone&) {
            return D::ComputeState(ParseHash(p.at("id")));
        };
        handlers["watch_compute"] = [](const json& p, const ClientGone&) {
            return D::SetWatchedCompute(ParseIds(p.at("ids")));
        };
        handlers["rebuild_compute"] = [](const json& p, const ClientGone&) {
            return D::RequestComputeRebuild(ParseIds(p.at("ids")));
        };
        handlers["read_gpu"] = [](const json& p, const ClientGone&) {
            const auto& value = p.at("address");
            if (!value.is_string()) throw std::runtime_error{"address must be a hex string"};
            auto text = value.get<std::string>();
            if (text.starts_with("0x")) text = text.substr(2);
            u64 address{};
            const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), address, 16);
            if (text.empty() || ec != std::errc{} || end != text.data() + text.size())
                throw std::runtime_error{"address must be a hex string"};
            return D::ReadGpu(address, Bound(p, "size", 256, 1, 65536));
        };
        // A marker in the log where the player saw something (a flash, a hitch): markkey.py sends one
        // per key press, so what the emulator logged just before each mark can be read afterwards.
        handlers["mark"] = [](const json& p, const ClientGone&) {
            static std::atomic<u32> marks{0};
            const u32 n = ++marks;
            const std::string label = p.value("label", std::string{});
            LOG_INFO(Core, "USER MARK {} {}", n, label);
            return json{{"mark", n}};
        };
        handlers["log_filter"] = [this](const json& p, const ClientGone&) {
            const auto filter_text = p.at("filter").get<std::string>();
            Common::Log::Filter filter;
            filter.ParseFilterString(Settings::values.log_filter.GetValue());
            filter.ParseFilterString(filter_text);
            Common::Log::SetGlobalFilter(filter);
            upstream_tools->NoteLogFilter(Settings::values.log_filter.GetValue() + " " +
                                          filter_text);
            return json{{"base", Settings::values.log_filter.GetValue()}, {"applied", filter_text}};
        };
        // Live native/lowering switch per program, live IR/SPIR-V dumps and the activity timeline.
        handlers["set_frontend"] = [](const json& p, const ClientGone&) {
            const auto mode_text = p.at("mode").get<std::string>();
            D::FrontendOverride mode;
            if (mode_text == "native") mode = D::FrontendOverride::Native;
            else if (mode_text == "lower") mode = D::FrontendOverride::Lower;
            else if (mode_text == "clear") mode = D::FrontendOverride::None;
            else throw std::runtime_error{"mode must be native, lower or clear"};
            return D::SetFrontend(ParseIds(p.at("keys")), mode);
        };
        handlers["frontends"] = [](const json&, const ClientGone&) { return D::Frontends(); };
        handlers["dump_ir"] = [](const json& p, const ClientGone&) { return D::RequestDump(ParseIds(p.at("keys"))); };
        handlers["dumps"] = [](const json&, const ClientGone&) { return D::Dumps(); };
        handlers["qmd_trace"] = [](const json& p, const ClientGone&) {
            return D::QmdTrace(Bound(p, "limit", 512, 1, 65536), p.value("last_frame", true),
                               p.value("clear", false));
        };
        handlers["gpu_activity"] = [](const json& p, const ClientGone&) {
            const double min_ms = p.contains("min_ms") ? p.at("min_ms").get<double>() : 5.0;
            return D::Activity(min_ms, Bound(p, "limit", 200, 1, 4096), p.value("clear", false));
        };
        handlers["video_state"] = [](const json& p, const ClientGone&) {
            return Service::AM::NativeVideoDebugState(Bound(p, "trace", 40, 1, 256));
        };
        handlers["video_inject"] = [](const json& p, const ClientGone&) {
            return Service::AM::NativeVideoDebugInject(p.value("serial", u64{0}), p.at("what").get<std::string>());
        };
        handlers["skip_draws"] = [](const json& p, const ClientGone&) { return Reject(p, true); };
        handlers["press_a"] = [this](const json& p, const ClientGone& gone) {
            const auto duration = Bound(p, "hold_ms", 120, 30, 500);
            SCOPE_EXIT { Common::N2DenDebugButtons::ReleaseA(); };
            Common::N2DenDebugButtons::PulseA(static_cast<u32>(duration));
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{duration};
            while (std::chrono::steady_clock::now() < deadline) {
                if (stopping.load() || gone()) throw std::runtime_error{"A pulse cancelled"};
                std::this_thread::sleep_for(5ms);
            }
            return json{{"button", "A"}, {"hold_ms", duration}, {"released", true}};
        };
        server = std::make_unique<Server>(*endpoint, std::move(handlers));
        D::SetActive(true);
        LOG_INFO(Core, "n2Den diagnostic server listening at {}:{} (diagnostics + A pulse, runtime only)", endpoint->host, server->Port());
    }
    ~Impl() {
        // End waiting screenshot requests before stopping the transport and before GPU teardown.
        stopping.store(true);
        Common::N2DenDebugButtons::ReleaseA();
        upstream_tools->Stop();
        server.reset();
        D::SetActive(false);
    }
    json Screenshot(const json& params, const ClientGone& gone) {
        auto& renderer = system.GPU().Renderer();
        if (renderer.IsScreenshotPending()) throw std::runtime_error{"a screenshot is already pending"};
        const auto timeout = Bound(params, "timeout_s", 5, 1, 15);
        const auto layout = Layout::DefaultFrameLayout(1280, 720);
        const auto pixels = std::make_shared<std::vector<u8>>(size_t{layout.width} * layout.height * 4);
        const auto done = std::make_shared<std::promise<bool>>();
        auto future = done->get_future();
        renderer.RequestScreenshot(pixels->data(), [pixels, done](bool invert) { done->set_value(invert); }, layout);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{timeout};
        while (future.wait_for(20ms) != std::future_status::ready) {
            if (stopping.load() || gone()) throw std::runtime_error{"screenshot cancelled: shutdown or client disconnected"};
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error{"no rendered frame before timeout (paused or loading?)"};
        }
        const bool invert = future.get();
        std::vector<u8> rgb(size_t{layout.width} * layout.height * 3);
        for (u32 y = 0; y < layout.height; ++y) {
            const auto source_y = invert ? layout.height - 1 - y : y;
            const u8* src = pixels->data() + size_t{source_y} * layout.width * 4;
            u8* dst = rgb.data() + size_t{y} * layout.width * 3;
            for (u32 x = 0; x < layout.width; ++x, src += 4, dst += 3) {
                dst[0] = src[2]; dst[1] = src[1]; dst[2] = src[0];
            }
        }
        std::vector<u8> png;
        if (!stbi_write_png_to_func(AppendPng, &png, static_cast<int>(layout.width),
                                    static_cast<int>(layout.height), 3, rgb.data(),
                                    static_cast<int>(layout.width * 3)))
            throw std::runtime_error{"PNG encoding failed"};
        return {{"width", layout.width}, {"height", layout.height}, {"png_base64", Base64(png)}};
    }

    /// [DEBUG] Flash hunt: captures presented frames back to back (small size, no PNG per frame) and
    /// counts "transient" near-white pixels: min(R,G,B) > 245 in frame i but in neither neighbour. A
    /// one-frame white material is a blob of them. Saves the frames with the most (and the frame
    /// before) as PNG under params.dir and returns per-frame counts plus how many presents it saw.
    json FlashBurst(const json& params, const ClientGone& gone) {
        auto& renderer = system.GPU().Renderer();
        const size_t count = Bound(params, "frames", 600, 3, 20000);
        const u32 width = static_cast<u32>(Bound(params, "width", 640, 160, 1920));
        const size_t keep = Bound(params, "keep", 12, 0, 200);
        const std::filesystem::path dir{params.value("dir", std::string{})};
        if (dir.empty()) throw std::runtime_error{"flash_burst needs 'dir'"};
        std::filesystem::create_directories(dir);
        const auto layout = Layout::DefaultFrameLayout(width, width * 9 / 16);
        const size_t w = layout.width, h = layout.height, px = w * h;
        struct Frame { std::vector<u8> bgra; bool invert; size_t index; };
        auto grab = [&](size_t index) {
            auto frame = std::make_shared<Frame>(Frame{std::vector<u8>(px * 4), false, index});
            const auto done = std::make_shared<std::promise<bool>>();
            auto future = done->get_future();
            while (renderer.IsScreenshotPending()) std::this_thread::sleep_for(1ms);
            renderer.RequestScreenshot(frame->bgra.data(), [done](bool invert) { done->set_value(invert); }, layout);
            const auto deadline = std::chrono::steady_clock::now() + 10s;
            while (future.wait_for(5ms) != std::future_status::ready) {
                if (stopping.load() || gone()) throw std::runtime_error{"flash_burst cancelled"};
                if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error{"no frame within 10 s"};
            }
            frame->invert = future.get();
            return frame;
        };
        auto white = [&](const Frame& f) {
            std::vector<u8> mask(px);
            for (size_t i = 0; i < px; ++i) {
                const u8* p = f.bgra.data() + i * 4;
                mask[i] = (std::min)({p[0], p[1], p[2]}) > 245;
            }
            return mask;
        };
        auto save = [&](const Frame& f, const std::string& name) {
            std::vector<u8> rgb(px * 3);
            for (size_t y = 0; y < h; ++y) {
                const size_t sy = f.invert ? h - 1 - y : y;
                const u8* src = f.bgra.data() + sy * w * 4;
                u8* dst = rgb.data() + y * w * 3;
                for (size_t x = 0; x < w; ++x, src += 4, dst += 3) { dst[0] = src[2]; dst[1] = src[1]; dst[2] = src[0]; }
            }
            const auto path = dir / name;
            std::vector<u8> png;
            stbi_write_png_to_func(AppendPng, &png, static_cast<int>(w), static_cast<int>(h), 3, rgb.data(),
                                   static_cast<int>(w * 3));
            std::ofstream file{path, std::ios::binary | std::ios::trunc};
            file.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
        };
        const auto frames_before = D::Status().value("renderer_frames", u64{0});
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<std::shared_ptr<Frame>> ring;
        std::vector<std::vector<u8>> masks;
        std::vector<u32> transient;
        struct Best { u32 count; std::shared_ptr<Frame> frame, before; };
        std::vector<Best> best;
        for (size_t i = 0; i < count; ++i) {
            ring.push_back(grab(i));
            masks.push_back(white(*ring.back()));
            if (ring.size() < 3) continue;
            u32 t = 0;
            for (size_t k = 0; k < px; ++k) t += masks[1][k] && !masks[0][k] && !masks[2][k];
            transient.push_back(t);
            if (keep && (best.size() < keep || t > best.back().count)) {
                best.push_back({t, ring[1], ring[0]});
                std::sort(best.begin(), best.end(), [](const Best& a, const Best& b) { return a.count > b.count; });
                if (best.size() > keep) best.pop_back();
            }
            ring.erase(ring.begin()); masks.erase(masks.begin());
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const auto frames_after = D::Status().value("renderer_frames", u64{0});
        json saved = json::array();
        for (const Best& b : best) {
            const auto name = fmt::format("{:05}-t{}.png", b.frame->index, b.count);
            save(*b.frame, name);
            save(*b.before, fmt::format("{:05}-t{}-before.png", b.frame->index, b.count));
            saved.push_back(name);
        }
        return {{"captures", count}, {"seconds", seconds}, {"presents", frames_after - frames_before},
                {"width", w}, {"height", h}, {"transient", transient}, {"saved", saved},
                {"dir", dir.generic_string()}};
    }
};
N2DenService::N2DenService(Core::System& system, u64 id, std::string title)
    : impl{std::make_unique<Impl>(system, id, std::move(title))} {}
N2DenService::~N2DenService() = default;
std::unique_ptr<N2DenService> MaybeStartN2DenService(Core::System& system, u64 id, const std::string& title) {
    const char* enabled = std::getenv("EDEN_NX2_DEBUG_SERVER");
    if (!enabled || !*enabled || std::string_view{enabled} == "0") return {};
    try {
        return std::make_unique<N2DenService>(system, id, title);
    } catch (const std::exception& error) {
        LOG_ERROR(Core, "n2Den diagnostic server disabled: {}", error.what());
        return {};
    }
}
} // namespace Core::Nx2Debug
