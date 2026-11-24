// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <version>
#if __cpp_lib_chrono >= 201907L
#include <chrono>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#endif
#include <compare>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <optional>
#include <string_view>
#include <type_traits>
#include <deque>
#include <atomic>
#include <fmt/core.h>

#include "common/settings_enums.h"
#include "common/assert.h"
#include "common/fs/fs_util.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/time_zone.h"

#if defined(__linux__ ) && defined(ARCHITECTURE_arm64)
#include <unistd.h>
#endif

namespace Settings {

// Clang 14 and earlier have errors when explicitly instantiating these classes
#ifndef CANNOT_EXPLICITLY_INSTANTIATE
#define SETTING(TYPE, RANGED) template class Setting<TYPE, RANGED>
#define SWITCHABLE(TYPE, RANGED) template class SwitchableSetting<TYPE, RANGED>

SETTING(AppletMode, false);
SETTING(AudioEngine, false);
SETTING(bool, false);
SETTING(int, false);
SETTING(std::string, false);
SETTING(u16, false);
SWITCHABLE(AnisotropyMode, true);
SWITCHABLE(AntiAliasing, false);
SWITCHABLE(AspectRatio, true);
SWITCHABLE(AstcDecodeMode, true);
SWITCHABLE(AstcRecompression, true);
SWITCHABLE(AudioMode, true);
SWITCHABLE(CpuBackend, true);
SWITCHABLE(CpuAccuracy, true);
SWITCHABLE(FullscreenMode, true);
SWITCHABLE(GpuAccuracy, true);
SWITCHABLE(Language, true);
SWITCHABLE(MemoryLayout, true);
SWITCHABLE(NvdecEmulation, false);
SWITCHABLE(Region, true);
SWITCHABLE(RendererBackend, true);
SWITCHABLE(ScalingFilter, false);
SWITCHABLE(SpirvOptimizeMode, true);
SWITCHABLE(TimeZone, true);
SWITCHABLE(Upscaler, true);
SWITCHABLE(UpscalerQuality, true);
SETTING(VSyncMode, true);
SWITCHABLE(bool, false);
SWITCHABLE(int, false);
SWITCHABLE(int, true);
SWITCHABLE(s64, false);
SWITCHABLE(u16, true);
SWITCHABLE(u32, false);
SWITCHABLE(u8, false);
SWITCHABLE(u8, true);

// Used in UISettings
// TODO see if we can move this to uisettings.cpp
SWITCHABLE(ConfirmStop, true);

#undef SETTING
#undef SWITCHABLE
#endif

Values values;

std::string GetTimeZoneString(TimeZone time_zone) {
    const auto time_zone_index = static_cast<std::size_t>(time_zone);
    ASSERT(time_zone_index < Common::TimeZone::GetTimeZoneStrings().size());

    std::string location_name;
    if (time_zone_index == 0) { // Auto
#if __cpp_lib_chrono >= 201907L && !defined(MINGW)
        // Disabled for MinGW -- tzdb always returns Etc/UTC
        try {
            const struct std::chrono::tzdb& time_zone_data = std::chrono::get_tzdb();
            const std::chrono::time_zone* current_zone = time_zone_data.current_zone();
            std::string_view current_zone_name = current_zone->name();
            location_name = current_zone_name;
        } catch (std::runtime_error& runtime_error) {
            // VCRUNTIME will throw a runtime_error if the operating system's selected time zone
            // cannot be found
            location_name = Common::TimeZone::FindSystemTimeZone();
            LOG_WARNING(Common,
                        "Error occurred when trying to determine system time zone:\n{}\nFalling "
                        "back to hour offset \"{}\"",
                        runtime_error.what(), location_name);
        }
#else
        location_name = Common::TimeZone::FindSystemTimeZone();
#endif
    } else {
        location_name = Common::TimeZone::GetTimeZoneStrings()[time_zone_index];
    }
    return location_name;
}

void LogSettings() {
    std::deque<std::string> settings_list;
    for (auto& [category, settings] : values.linkage.by_category) {
        for (const auto& setting : settings) {
            // Hide the token secret, for security reasons.
            if (setting->Id() != values.eden_token.Id()) {
                auto const is_default = setting->ToString() == setting->DefaultToString();
                auto const name = fmt::format(
                    "{:c}{:c} {}.{}",
                    is_default ? '-' : 'M',
                    setting->UsingGlobal() ? '-' : 'C', TranslateCategory(category),
                    setting->GetLabel());
                if (is_default)
                    settings_list.push_back(fmt::format("{}: {}", name, setting->Canonicalize()));
                else
                    settings_list.push_front(fmt::format("{}: {}", name, setting->Canonicalize()));
            }
        }
    }
    LOG_INFO(Config, "n2den Configuration:");
    for (auto const& e : settings_list)
        LOG_INFO(Config, "{}", e);
#define LOG_PATH(NAME) \
    LOG_INFO(Config, #NAME ": {}", Common::FS::PathToUTF8String(Common::FS::GetEdenPath(Common::FS::EdenPath::NAME)))
    LOG_PATH(CacheDir);
    LOG_PATH(ConfigDir);
    LOG_PATH(LoadDir);
    LOG_PATH(NANDDir);
    LOG_PATH(SaveDir);
    LOG_PATH(SDMCDir);
#undef LOG_PATH
}

bool GetDebugKnobAt(u8 i) {
    return (values.debug_knobs.GetValue() & (1 << (i & 0xF))) != 0;
}

void UpdateGPUAccuracy() {
    values.current_gpu_accuracy = values.gpu_accuracy.GetValue();
}

bool IsGPULevelHigh() {
    return values.current_gpu_accuracy == GpuAccuracy::High;
}

bool IsDMALevelDefault() {
    return values.dma_accuracy.GetValue() == DmaAccuracy::Default;
}

bool IsDMALevelSafe() {
    return values.dma_accuracy.GetValue() == DmaAccuracy::Safe;
}

bool IsGPUFenceBehaviorDefault() {
    return values.gpu_fence_behavior.GetValue() == GpuFenceBehavior::Default;
}

bool IsGPUFenceBehaviorBalanced() {
    return values.gpu_fence_behavior.GetValue() == GpuFenceBehavior::Balanced;
}

bool IsGPUFenceBehaviorAccurate() {
    return values.gpu_fence_behavior.GetValue() == GpuFenceBehavior::Accurate;
}

bool IsFastmemEnabled() {
    if (values.cpu_accuracy.GetValue() == Settings::CpuAccuracy::Debugging)
        return bool(values.cpuopt_fastmem);
    else if (values.cpu_accuracy.GetValue() == CpuAccuracy::Unsafe)
        return bool(values.cpuopt_unsafe_host_mmu);
#if defined(__linux__) && defined(ARCHITECTURE_arm64)
    // Only 4kb systems support host MMU right now
    // TODO: Support this
    return getpagesize() == 4096;
#elif !defined(__APPLE__) && !defined(__ANDROID__) && !defined(_WIN32) && !defined(__linux__) && !defined(__FreeBSD__)
    return false;
#else
    return true;
#endif
}

static bool is_nce_enabled = false;

void SetNceEnabled(bool is_39bit) {
    const bool is_nce_selected = values.cpu_backend.GetValue() == CpuBackend::Nce;
    if (is_nce_selected && !IsFastmemEnabled()) {
        LOG_WARNING(Common, "Fastmem is required to natively execute code in a performant manner, "
                            "falling back to Dynarmic");
    }
    if (is_nce_selected && !is_39bit) {
        LOG_WARNING(
            Common,
            "Program does not utilize 39-bit address space, unable to natively execute code");
    }
    is_nce_enabled = IsFastmemEnabled() && is_nce_selected && is_39bit;
}

bool IsNceEnabled() {
    return is_nce_enabled;
}

static u64 current_program_id = 0;

void SetCurrentProgramID(u64 program_id) {
    current_program_id = program_id;
}

u32 ForcedSwapInterval() {
    static const u32 from_env = [] {
        const char* const value = std::getenv("EDEN_NX2_SWAP_INTERVAL");
        return value ? (std::min)(static_cast<u32>(std::strtoul(value, nullptr, 0)), 4U) : 0U;
    }();
    if (from_env != 0) {
        return from_env;
    }
    constexpr u64 DonkeyKongBananza = 0x040074A01BF12000ULL;
    if (values.dk_single_interval.GetValue() &&
        (GetCurrentProgramID() & ~u64{0xFFF}) == DonkeyKongBananza) {
        return 1;
    }
    return 0;
}

u64 GetCurrentProgramID() {
    return current_program_id;
}

bool IsDockedMode() {
    return values.use_docked_mode.GetValue() == Settings::ConsoleMode::Docked;
}

float Volume() {
    if (values.audio_muted) {
        return 0.0f;
    }
    return values.volume.GetValue() / static_cast<f32>(values.volume.GetDefault());
}

const char* TranslateCategory(Category category) {
    switch (category) {
    case Category::Android:
        return "Android";
    case Category::Audio:
        return "Audio";
    case Category::Core:
        return "Core";
    case Category::Cpu:
    case Category::CpuDebug:
    case Category::CpuUnsafe:
        return "Cpu";
    case Category::Overlay:
        return "Overlay";
    case Category::Renderer:
    case Category::RendererAdvanced:
    case Category::RendererHacks:
    case Category::RendererDebug:
    case Category::RendererExtensions:
        return "Renderer";
    case Category::System:
    case Category::SystemAudio:
        return "System";
    case Category::DataStorage:
        return "Data Storage";
    case Category::Debugging:
    case Category::DebuggingGraphics:
        return "Debugging";
    case Category::GpuDriver:
        return "GpuDriver";
    case Category::LibraryApplet:
        return "LibraryApplet";
    case Category::Miscellaneous:
        return "Miscellaneous";
    case Category::Network:
        return "Network";
    case Category::WebService:
        return "WebService";
    case Category::AddOns:
        return "DisabledAddOns";
    case Category::Controls:
        return "Controls";
    case Category::Ui:
    case Category::UiGeneral:
        return "UI";
    case Category::UiAudio:
        return "UiAudio";
    case Category::UiLayout:
        return "UILayout";
    case Category::UiGameList:
        return "UIGameList";
    case Category::Screenshots:
        return "Screenshots";
    case Category::Shortcuts:
        return "Shortcuts";
    case Category::Multiplayer:
        return "Multiplayer";
    case Category::Services:
        return "Services";
    case Category::Paths:
        return "Paths";
    case Category::MaxEnum:
        break;
    }
    return "Miscellaneous";
}

void TranslateResolutionInfo(ResolutionSetup setup, ResolutionScalingInfo& info) {
    info.downscale = false;
    switch (setup) {
    case ResolutionSetup::Res1_4X:
        info.up_scale = 1;
        info.down_shift = 2;
        info.downscale = true;
        break;
    case ResolutionSetup::Res1_2X:
        info.up_scale = 1;
        info.down_shift = 1;
        info.downscale = true;
        break;
    case ResolutionSetup::Res3_4X:
        info.up_scale = 3;
        info.down_shift = 2;
        info.downscale = true;
        break;
    case ResolutionSetup::Res1X:
        info.up_scale = 1;
        info.down_shift = 0;
        break;
    case ResolutionSetup::Res3_2X:
        info.up_scale = 3;
        info.down_shift = 1;
        break;
    case ResolutionSetup::Res5_4X:
        info.up_scale = 5;
        info.down_shift = 2;
        break;
    case ResolutionSetup::Res2X:
        info.up_scale = 2;
        info.down_shift = 0;
        break;
    case ResolutionSetup::Res3X:
        info.up_scale = 3;
        info.down_shift = 0;
        break;
    case ResolutionSetup::Res4X:
        info.up_scale = 4;
        info.down_shift = 0;
        break;
    case ResolutionSetup::Res5X:
        info.up_scale = 5;
        info.down_shift = 0;
        break;
    case ResolutionSetup::Res6X:
        info.up_scale = 6;
        info.down_shift = 0;
        break;
    case ResolutionSetup::Res7X:
        info.up_scale = 7;
        info.down_shift = 0;
        break;
    case ResolutionSetup::Res8X:
        info.up_scale = 8;
        info.down_shift = 0;
        break;
    default:
        UNREACHABLE();
    }
    info.up_factor = static_cast<f32>(info.up_scale) / (1U << info.down_shift);
    info.down_factor = static_cast<f32>(1U << info.down_shift) / info.up_scale;
    info.active = info.up_scale != 1 || info.down_shift != 0;
}

namespace {
std::atomic<AstcRecompression> effective_astc_recompression{AstcRecompression::Auto};
}

namespace {
std::atomic<bool> hdr_output_active{false};
std::atomic<const char*> active_upscaler{nullptr};
}

const char* ActiveUpscaler() {
    return active_upscaler.load(std::memory_order_relaxed);
}

void SetActiveUpscaler(const char* name) {
    active_upscaler.store(name, std::memory_order_relaxed);
}

bool HdrOutputActive() {
    return hdr_output_active.load(std::memory_order_relaxed);
}

void SetHdrOutputActive(bool active) {
    if (hdr_output_active.exchange(active, std::memory_order_relaxed) != active) {
        LOG_INFO(Common, "HDR output: {}", active ? "on (HDR10)" : "off (SDR)");
    }
}

bool GpuBcnTranscodeAllowed() {
    static const bool allowed = [] {
        const char* value = std::getenv("EDEN_GPU_BCN_TRANSCODE");
        return value == nullptr || value[0] != '0';
    }();
    return allowed && values.accelerate_astc.GetValue() == AstcDecodeMode::Gpu;
}

void ResolveAstcRecompression(u64 memory_limit_bytes, bool gpu_bcn_transcode) {
    AstcRecompression setting = values.astc_recompression.GetValue();
    // EDEN_N2DEN_ASTC_RECOMPRESSION=uncompressed|bc1|bc3|auto|smart overrides the setting for one
    // run (A/B comparisons, scripted runs), whatever the config holds.
    if (const char* const value = std::getenv("EDEN_N2DEN_ASTC_RECOMPRESSION")) {
        const std::string_view name{value};
        if (name == "uncompressed") {
            setting = AstcRecompression::Uncompressed;
        } else if (name == "bc1") {
            setting = AstcRecompression::Bc1;
        } else if (name == "bc3") {
            setting = AstcRecompression::Bc3;
        } else if (name == "auto") {
            setting = AstcRecompression::Auto;
        } else if (name == "smart") {
            setting = AstcRecompression::Smart;
        }
    }
    AstcRecompression effective = setting;
    if (setting == AstcRecompression::Smart && gpu_bcn_transcode) {
        // Each texture in the BC format its blocks need (Eglantine's smart BC stand-ins):
        // luminance as BC4, luminance+alpha as BC5, opaque colour as BC1, the rest as BC3.
        LOG_INFO(Common, "ASTC textures: smart BC (BC1/BC3/BC4/BC5 per texture), re-compressed on "
                         "the GPU ({} MiB of video memory may be used)",
                 memory_limit_bytes >> 20);
        effective_astc_recompression.store(effective, std::memory_order_relaxed);
        return;
    }
    if (setting == AstcRecompression::Smart) {
        // Smart needs the GPU encoder; without it, decide as Auto does.
        LOG_INFO(Common, "ASTC textures: smart BC needs the GPU re-compression, deciding as Auto");
        setting = AstcRecompression::Auto;
        effective = setting;
    }
    if (setting == AstcRecompression::Auto && gpu_bcn_transcode) {
        // The GPU decodes ASTC and re-compresses it to BC3 itself: a quarter of the memory of
        // uncompressed, without the CPU stalls that BC3 used to cost.
        effective = AstcRecompression::Bc3;
        LOG_INFO(Common, "ASTC textures: BC3, re-compressed on the GPU (Auto; {} MiB of video "
                         "memory may be used)",
                 memory_limit_bytes >> 20);
    } else if (setting == AstcRecompression::Auto) {
        // Uncompressed textures are decoded on the GPU but take 4x the memory of BC3 (decoded on
        // the CPU, which stalls draws while a battle loads its textures). Fortune's Weave at 1x
        // with the ultrawide patch had 13.5 GB in use within a minute uncompressed: they only fit
        // where the emulator may use about 15 GB (20 GB cards and up), and not at 1.5x and above
        // or with the ultrawide patch above 1x.
        static constexpr u64 UncompressedMinBytes = (29ULL << 30) / 2; // 14.5 GiB
        const bool high_resolution =
            values.resolution_setup.GetValue() >= ResolutionSetup::Res3_2X;
        const bool ultrawide = values.fefw_ultrawide.GetValue();
        const bool ultrawide_scaled =
            ultrawide && values.resolution_setup.GetValue() > ResolutionSetup::Res1X;
        const bool enough_memory = memory_limit_bytes >= UncompressedMinBytes; // 0: unknown
        effective = high_resolution || ultrawide_scaled || !enough_memory
                        ? AstcRecompression::Bc3
                        : AstcRecompression::Uncompressed;
        LOG_INFO(Common, "ASTC textures: {} (Auto; resolution {}1.5x, ultrawide {}, {} MiB of video "
                         "memory may be used, uncompressed needs {} MiB)",
                 effective == AstcRecompression::Bc3 ? "BC3" : "uncompressed",
                 high_resolution ? "at or above " : "below ", ultrawide ? "on" : "off",
                 memory_limit_bytes >> 20, UncompressedMinBytes >> 20);
    } else {
        LOG_INFO(Common, "ASTC textures: {} (set by the user)",
                 CanonicalizeEnum(setting));
    }
    effective_astc_recompression.store(effective, std::memory_order_relaxed);
}

// -1: no override; else the AstcRecompression value nvn-hle set for this boot.
static std::atomic<int> astc_recompression_override{-1};

void SetAstcRecompressionOverride(std::optional<AstcRecompression> value) {
    astc_recompression_override.store(value ? static_cast<int>(*value) : -1,
                                      std::memory_order_relaxed);
}

AstcRecompression EffectiveAstcRecompression() {
    if (const int forced = astc_recompression_override.load(std::memory_order_relaxed);
        forced >= 0) [[unlikely]] {
        return static_cast<AstcRecompression>(forced);
    }
    AstcRecompression effective = effective_astc_recompression.load(std::memory_order_relaxed);
    if (effective == AstcRecompression::Auto) {
        // Not resolved by a renderer yet (OpenGL): decide without the video memory size.
        ResolveAstcRecompression(0);
        effective = effective_astc_recompression.load(std::memory_order_relaxed);
    }
    return effective;
}

void UpdateRescalingInfo() {
    const auto setup = values.resolution_setup.GetValue();
    auto& info = values.resolution_info;
    TranslateResolutionInfo(setup, info);
    info.scene_only = false;
    // Upscaler quality modes (1X only for now): the game draws its scene at half the picture's
    // size (Performance), so draw the scene alone bigger: Native 2x (1440p), Ultra Quality 1.5x
    // (1920x1080), Quality 1.3125x (1680x945), Balanced 1.25x (1600x900). Scales with small
    // denominators: the emulator rounds the game's pixel-exact reads to its own pixel grid, and
    // 1.16x (74/64) rounded unevenly enough to show. The upscaled picture and the UI stay at
    // 1440p.
    if (setup != ResolutionSetup::Res1X || values.upscaler.GetValue() == Upscaler::GameDlss) {
        return;
    }
    // nvn-hle has no scene-only scale: its rescale (nvn_rescale.h) reads this info as the
    // Resolution setting and gives every eligible render target, depth and the picture included,
    // a second image at the scaled size (ScalesHeight / SCENE_MAX_HEIGHT are not looked at). At
    // "1X" that cost 0.9 GiB of video memory and more in Ocarina of Time (96 scaled twins, the
    // shadow atlas among the eligible targets). EDEN_NVN_SCENE_SCALE=1 keeps the old behaviour
    // for comparison.
    if (ResolveNvnHle().hle) {
        const char* const keep = std::getenv("EDEN_NVN_SCENE_SCALE");
        if (keep == nullptr || keep[0] == '\0' || keep[0] == '0') {
            LOG_INFO(Common, "Upscaler quality {}: no scene scale under nvn-hle, the game draws at "
                             "its own size (EDEN_NVN_SCENE_SCALE=1 scales every render target)",
                     CanonicalizeEnum(values.upscaler_quality.GetValue()));
            return;
        }
    }
    u32 up_scale = 1;
    u32 down_shift = 0;
    switch (values.upscaler_quality.GetValue()) {
    case UpscalerQuality::Native:
        up_scale = 2;
        break;
    case UpscalerQuality::UltraQuality:
        up_scale = 3;
        down_shift = 1;
        break;
    case UpscalerQuality::Quality:
        up_scale = 21;
        down_shift = 4;
        break;
    case UpscalerQuality::Balanced:
        up_scale = 5;
        down_shift = 2;
        break;
    case UpscalerQuality::Performance:
    default:
        return;
    }
    info.up_scale = up_scale;
    info.down_shift = down_shift;
    info.up_factor = static_cast<f32>(up_scale) / static_cast<f32>(1U << down_shift);
    info.down_factor = static_cast<f32>(1U << down_shift) / static_cast<f32>(up_scale);
    info.active = true;
    info.scene_only = true;
    LOG_INFO(Common, "Upscaler quality {}: the 3D scene is drawn at {:.3f}x, the picture at 1x",
             CanonicalizeEnum(values.upscaler_quality.GetValue()), info.up_factor);
}

u32 FrameGenMultiplier() {
    return std::clamp(values.frame_gen_multiplier.GetValue(), MIN_FRAME_GEN_MULTIPLIER,
                      MAX_FRAME_GEN_MULTIPLIER);
}

size_t FrameGenGenerations() {
    if (!values.frame_gen.GetValue()) {
        return 0;
    }
    return FrameGenMultiplier() - 1;
}

size_t FrameGenMaxGenerations() {
    if (!values.frame_gen.GetValue()) {
        return 0;
    }
    if (values.frame_gen_target_rate.GetValue() != 0) {
        return MAX_FRAME_GEN_MULTIPLIER - 1;
    }
    return FrameGenMultiplier() - 1;
}

void RestoreGlobalState(bool is_powered_on) {
    // If a game is running, DO NOT restore the global settings state
    if (is_powered_on) {
        return;
    }

    for (const auto& reset : values.linkage.restore_functions) {
        reset();
    }

    // Reset per-game flags
    values.use_squashed_iterated_blend = false;
    values.fefw_ultrawide_active = false;
}

static bool configuring_global = true;

bool IsConfiguringGlobal() {
    return configuring_global;
}

void SetConfiguringGlobal(bool is_global) {
    configuring_global = is_global;
}

u16 SpeedLimit() {
    switch (SpeedMode(values.current_speed_mode)) {
    case SpeedMode::Standard:
        return values.speed_limit.GetValue();
    case SpeedMode::Turbo:
        return values.turbo_speed_limit.GetValue();
    case SpeedMode::Slow:
        return values.slow_speed_limit.GetValue();
    default:
        UNIMPLEMENTED();
    }

    return 100;
}

void SetSpeedMode(const SpeedMode& mode) {
    values.current_speed_mode.SetValue(mode);

    switch (mode) {
    case SpeedMode::Turbo:
    case SpeedMode::Slow:
        values.use_speed_limit.SetValue(true);
        break;
    case SpeedMode::Standard:
    default:
        break;
    }
}

void ToggleStandardMode() {
    values.use_speed_limit.SetValue(!values.use_speed_limit.GetValue());
    SetSpeedMode(SpeedMode::Standard);
}

void ToggleTurboMode() {
    if (values.current_speed_mode.GetValue() != SpeedMode::Turbo)
        SetSpeedMode(SpeedMode::Turbo);
    else
        SetSpeedMode(SpeedMode::Standard);
}

void ToggleSlowMode() {
    if (values.current_speed_mode.GetValue() != SpeedMode::Slow)
        SetSpeedMode(SpeedMode::Slow);
    else
        SetSpeedMode(SpeedMode::Standard);
}

bool IsOpenGL() {
    const auto backend = Settings::values.renderer_backend.GetValue();
    switch (backend) {
    case RendererBackend::OpenGL_GLSL:
    case RendererBackend::OpenGL_GLASM:
    case RendererBackend::OpenGL_SPIRV:
        return true;
    default:
        return false;
    }
}


static bool current_program_is_switch2 = false;

bool DockedDisplayIs4K() {
    if (const char* const value = std::getenv("EDEN_NX2_DOCKED_DEFAULT_4K"); value && value[0] != '\0') {
        return value[0] == '1';
    }
    return CurrentProgramIsSwitch2();
}

void SetCurrentProgramIsSwitch2(bool is_switch2) {
    current_program_is_switch2 = is_switch2;
}

bool CurrentProgramIsSwitch2() {
    return current_program_is_switch2;
}

NvnHleChoice ChooseNvnHle(const char* environment, GraphicsPath setting, bool switch2_program) {
    if (environment != nullptr) {
        const bool off = std::string_view{environment} == "0";
        return {!off, off ? NvnHleReason::EnvironmentOff : NvnHleReason::EnvironmentOn};
    }
    if (setting != GraphicsPath::Hle) {
        return {false, NvnHleReason::SettingEmulated};
    }
    if (!switch2_program) {
        return {false, NvnHleReason::NotSwitch2};
    }
    return {true, NvnHleReason::SettingHle};
}

NvnHleChoice ResolveNvnHle() {
    return ChooseNvnHle(std::getenv("EDEN_NVN_HLE"), values.graphics_path.GetValue(),
                        current_program_is_switch2);
}

const char* NvnHleReasonText(NvnHleReason reason) {
    switch (reason) {
    case NvnHleReason::EnvironmentOn:
        return "EDEN_NVN_HLE is set";
    case NvnHleReason::EnvironmentOff:
        return "EDEN_NVN_HLE=0";
    case NvnHleReason::SettingHle:
        return "Graphics path setting: NVN-HLE";
    case NvnHleReason::SettingEmulated:
        return "Graphics path setting: Emulated GPU";
    case NvnHleReason::NotSwitch2:
        return "the program is not a Switch 2 one (NVN-HLE targets the Switch 2 NVN SDK)";
    }
    return "";
}

namespace {

// An EDEN_NVN_* on/off override: std::nullopt when unset or empty, else whether it is "1".
std::optional<bool> NvnEnvironmentSwitch(const char* name) {
    const char* const value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return std::nullopt;
    }
    return std::string_view{value} == "1";
}

} // Anonymous namespace

bool NvnSinkDlss() {
    static const std::optional<bool> environment = NvnEnvironmentSwitch("EDEN_NVN_DLSS");
    if (environment.has_value()) {
        return *environment;
    }
    return values.nvn_upscaler.GetValue() == NvnUpscaler::Dlss;
}

bool NvnReplaceSameSizeNetwork() {
    static const std::optional<bool> environment =
        NvnEnvironmentSwitch("EDEN_NVN_REPLACE_SAME_SIZE_NETWORK");
    if (environment.has_value()) {
        return *environment;
    }
    return values.nvn_upscaler.GetValue() != NvnUpscaler::Auto;
}

bool NvnDlssTemporalRequested() {
    static const std::optional<bool> environment = NvnEnvironmentSwitch("EDEN_NVN_DLSS_TEMPORAL");
    if (environment.has_value()) {
        return *environment;
    }
    return values.nvn_dlss_temporal.GetValue() &&
           values.nvn_upscaler.GetValue() == NvnUpscaler::Dlss;
}

bool NvnBilinearUpscale() {
    return values.nvn_upscaler.GetValue() == NvnUpscaler::Copy;
}

NvnDlssQuality NvnDlssSceneQuality() {
    if (!ResolveNvnHle().hle || !NvnSinkDlss()) {
        return NvnDlssQuality::Native;
    }
    return values.nvn_dlss_quality.GetValue();
}

NvnRenderer NvnRendererChoice() {
    const char* const environment = std::getenv("EDEN_NVN_RENDERER");
    if (environment != nullptr && environment[0] != '\0') {
        return std::string_view{environment} == "d3d12" ? NvnRenderer::D3D12 : NvnRenderer::Vulkan;
    }
    return values.nvn_renderer.GetValue();
}

u32 FrameRateCap() {
    switch (values.frame_pacing_mode.GetValue()) {
    case FramePacingMode::Target_Auto:
        return 0;
    case FramePacingMode::Target_30:
        return 30;
    case FramePacingMode::Target_60:
        return 60;
    case FramePacingMode::Target_90:
        return 90;
    case FramePacingMode::Target_120:
        return 120;
    }
    return 0;
}

} // namespace Settings
