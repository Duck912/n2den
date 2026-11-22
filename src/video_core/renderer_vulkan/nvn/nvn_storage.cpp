// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 n2den Contributors
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

// Texture storages (the VkImages behind NVN textures), virtual memory pools and the coherency of
// both with guest memory: the storage cache, lazy image creation, MapVirtual, CPU write
// notifications and the ReadTexels flush. See DESIGN.md "Memory and coherency".

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_set>
#include <vector>

#include <fmt/format.h>

#include "common/boot_counter.h"
#include "common/cityhash.h"
#include "common/logging.h"
#include "video_core/nvn/prerec_probe.h"
#include "video_core/nx2.h"
#include "video_core/renderer_vulkan/nvn/nvn_replay_phases.h"
#include "video_core/renderer_vulkan/nvn/nvn_impl.h"
#include "video_core/renderer_vulkan/nvn/nvn_draw.h"

#include "video_core/renderer_vulkan/nvn/nvn_sparse.h"
#include "video_core/renderer_vulkan/nvn/nvn_upload.h"
#include "video_core/textures/decoders.h"
#ifdef _WIN32
#include <windows.h>
#endif

namespace Vulkan::Nvn {
namespace {

u32 Mip(u32 size, u32 level) {
    return (std::max)(size >> level, 1u);
}

/// The host's memory the process can still take: physical RAM that is free or reclaimable, and
/// commit (RAM plus page file) not yet charged. `known` false where there is no cheap query.
struct HostMemoryLeft {
    u64 ram{};
    u64 commit{};
    bool known{false};
};

HostMemoryLeft QueryHostMemoryLeft() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return {status.ullAvailPhys, status.ullAvailPageFile, true};
    }
#endif
    return {};
}

/// An environment variable in MiB as bytes (`fallback_mib` when unset or empty).
u64 EnvMiB(const char* name, u64 fallback_mib) {
    const char* const value = std::getenv(name);
    const u64 mib =
        value != nullptr && value[0] != 0 ? std::strtoull(value, nullptr, 10) : fallback_mib;
    return mib << 20;
}

/// EDEN_NVN_RELEASE_SUPERSEDED_PRESENTS: presents without a use after which a superseded
/// GPU-written storage gives its image back, live textures or not (0: never; see
/// Context::Impl::ReleaseSupersededImages).
u64 ReleaseSupersededAge() {
    static const u64 age = [] {
        const char* const value = std::getenv("EDEN_NVN_RELEASE_SUPERSEDED_PRESENTS");
        return value != nullptr && value[0] != 0 ? std::strtoull(value, nullptr, 10) : u64{60};
    }();
    return age;
}

/// Percent of the driver's memory budget (VK_EXT_memory_budget) the device may use before the
/// storage cache sheds what it can rebuild (EDEN_NVN_DEVICE_PRESSURE_PERCENT, 50-100). Waiting for
/// the full budget leaves no room for the rest of the system: a 16 GB card filled to 95% during an
/// area load while the cache still held gigabytes of textures nothing used.
u64 DevicePressurePercent() {
    static const u64 percent = [] {
        const char* env = std::getenv("EDEN_NVN_DEVICE_PRESSURE_PERCENT");
        const u64 value = env ? std::strtoull(env, nullptr, 10) : 85;
        return std::clamp<u64>(value, 50, 100);
    }();
    return percent;
}

/// Bytes the device uses over its pressure line (0 when under it or without a budget).
u64 DevicePressureExcess(const Device& device, u64* budget_out = nullptr) {
    const u64 budget = device.GetDeviceMemoryBudget();
    if (budget_out) {
        *budget_out = budget;
    }
    if (budget == 0) {
        return 0;
    }
    const u64 line = budget / 100 * DevicePressurePercent();
    const u64 used = device.GetDeviceMemoryUsage() + device.GetDeviceMemorySlack();
    return used > line ? used - line : 0;
}

/// Bytes of a storage's span the storages that superseded it may leave uncovered when an aged
/// storage is judged: the padding between textures of other sizes placed over it.
u64 SupersededGap(const Storage& s) {
    return (std::min)(u64{64} << 10, s.layout.size / 32);
}

} // Anonymous namespace

bool Context::Impl::PollMemoryGuard() {
    static const u64 min_free = EnvMiB("EDEN_NVN_MIN_FREE_RAM_MIB", 2048);
    static const u64 abort_free = EnvMiB("EDEN_NVN_ABORT_FREE_RAM_MIB", 0);
    MemoryGuard& guard = memory_guard;
    const auto now = std::chrono::steady_clock::now();
    if (now < guard.next_poll) {
        return guard.refusing;
    }
    guard.next_poll = now + std::chrono::milliseconds(200);
    // The device over its budget: the driver pages video memory into system RAM (Windows shows
    // it as shared GPU memory). The trim pass sheds what the cache can rebuild; it runs at the
    // next image or present instead of up to 30 presents later.
    const u64 budget = device.GetDeviceMemoryBudget();
    const u64 used =
        budget != 0 ? device.GetDeviceMemoryUsage() + device.GetDeviceMemorySlack() : 0;
    const u64 excess = used > budget ? used - budget : 0;
    // The trim starts at the pressure line, before the driver's budget is reached.
    const u64 pressure_excess = DevicePressureExcess(device);
    if (pressure_excess != 0 && !trim_legacy) {
        trim_device_excess = (std::max)(trim_device_excess, pressure_excess);
        trim_floor_bytes = 0;
        trim_next_present = 0;
    }
    if (excess != 0) {
        ++guard.device_over_polls;
        if (!trim_legacy) {
            trim_device_excess = (std::max)(trim_device_excess, excess);
            trim_floor_bytes = 0;
            trim_next_present = 0;
        }
    }
    const HostMemoryLeft host = QueryHostMemoryLeft();
    if (!host.known) {
        guard.refusing = false;
        return false;
    }
    const u64 left = (std::min)(host.ram, host.commit);
    if (abort_free != 0 && left < abort_free) {
        LOG_CRITICAL(Render_Vulkan,
                     "[nvn] memory guard: {} MiB of RAM and {} MiB of commit left, under "
                     "EDEN_NVN_ABORT_FREE_RAM_MIB ({} MiB); device {}/{} MiB, storage cache {} "
                     "MiB. Exiting before the machine runs out of memory",
                     host.ram >> 20, host.commit >> 20, abort_free >> 20, used >> 20, budget >> 20,
                     storage_cache_bytes >> 20);
        Common::Log::Stop();
        std::_Exit(3);
    }
    // Refusing ends with some room above the floor, or when the device is back in its budget.
    const u64 floor = guard.refusing ? min_free + min_free / 4 : min_free;
    const bool host_low = min_free != 0 && left < floor;
    const bool refuse = host_low && excess != 0;
    if (refuse != guard.refusing || (refuse && now >= guard.next_log)) {
        guard.next_log = now + std::chrono::seconds(5);
        if (refuse) {
            LOG_ERROR(Render_Vulkan,
                      "[nvn] memory guard: {} MiB of RAM and {} MiB of commit left (floor {} MiB) "
                      "and the device is {} MiB over its budget ({}/{} MiB, storage cache {} "
                      "MiB): no new texture images until memory is back, {} refused so far "
                      "(EDEN_NVN_MIN_FREE_RAM_MIB, 0 turns the guard off)",
                      host.ram >> 20, host.commit >> 20, min_free >> 20, excess >> 20, used >> 20,
                      budget >> 20, storage_cache_bytes >> 20, guard.refused);
        } else {
            LOG_WARNING(Render_Vulkan,
                        "[nvn] memory guard: memory is back ({} MiB of RAM, {} MiB of commit "
                        "left, device {}/{} MiB): texture images are made again, {} were refused",
                        host.ram >> 20, host.commit >> 20, used >> 20, budget >> 20,
                        guard.refused);
        }
    }
    if (host_low && !refuse && !guard.host_low_logged) {
        // Not the replay's video memory: said once, nothing refused.
        guard.host_low_logged = true;
        LOG_WARNING(Render_Vulkan,
                    "[nvn] memory guard: {} MiB of RAM and {} MiB of commit left (floor {} MiB) "
                    "while the device is within its budget ({}/{} MiB): nothing is refused",
                    host.ram >> 20, host.commit >> 20, min_free >> 20, used >> 20, budget >> 20);
    }
    guard.refusing = refuse;
    return refuse;
}

u64 Context::Impl::PollTick() {
    const auto now = std::chrono::steady_clock::now();
    const u64 presents = stats.presents.load(std::memory_order_relaxed);
    if (presents != poll_presents_seen || poll_last_change == std::chrono::steady_clock::time_point{}) {
        poll_presents_seen = presents;
        poll_last_change = now;
    }
    const u64 idle =
        static_cast<u64>(std::chrono::duration_cast<std::chrono::milliseconds>(now - poll_last_change)
                             .count()) /
        16;
    poll_tick = (std::max)(poll_tick, presents + idle);
    return poll_tick;
}

void Context::Impl::DetectGuestChange(Storage& s, u64 seq) {
    // A storage the GPU wrote is watched too (a decoder writing a video plane the game cleared
    // first): its baseline is the guest signature at the first check after the GPU write, so a
    // GPU-only change never looks like a CPU one.
    if (s.sparse || s.force_upload || s.samples != VK_SAMPLE_COUNT_1_BIT ||
        s.origin == StorageOrigin::None || s.checked_seq == seq) {
        return;
    }
    s.checked_seq = seq;
    const u64 tick = PollTick();
    if (s.tex_gate.slot) [[unlikely]] {
        s.tex_gate.slot->last_use_tick.store(tick, std::memory_order_relaxed);
    }
    if (!s.force_check && tick < s.next_check_tick) {
        return;
    }
    const bool forced = s.force_check;
    s.force_check = false;
    const auto pool_it = pools.find(s.pool_serial);
    if (pool_it == pools.end() || s.pool_offset >= pool_it->second->size) {
        return;
    }
    const Pool& pool = *pool_it->second;
    const bool watching = s.origin == StorageOrigin::Gpu;
    if (watching && ((pool.flags & PoolFlag::CpuNoAccess) != 0 || pool.IsVirtual())) {
        return;
    }
    if (watching && s.cpu_writeback) {
        return; // its guest bytes are our write-back of its GPU contents (Storage::cpu_writeback)
    }
    const u64 size = (std::min)(s.layout.size, pool.size - s.pool_offset);
    if (size == 0) {
        return;
    }
    // A texture the CPU keeps writing parts of (a glyph atlas) shows new content only in a few
    // lines the signature may not sample: small uncompressed ones and ones that changed before
    // are compared in full at every check. Block-compressed data is rewritten whole (a streamed
    // tile), so its signature plus the scheduled full compare are enough.
    constexpr u64 SMALL_STORAGE = u64{4} << 20;
    const bool compressed = s.desc.format != 0 && FormatBlockInfo(s.desc.format).width > 1;
    const bool every_check = forced || s.dynamic || (size <= SMALL_STORAGE && !compressed);
    bool full = every_check || tick >= s.next_full_tick;
    if (full && !every_check) {
        // The scheduled full compares of the rest share a byte budget per present (a 2 GB texture
        // set has ~20 MB due each present otherwise): one that does not fit stays due and is
        // looked at again at its next check.
        if (full_hash_tick != tick) {
            full_hash_tick = tick;
            full_hash_bytes = 0;
        }
        if (full_hash_bytes >= FULL_HASH_BYTES_PER_PRESENT) {
            full = false;
        } else {
            full_hash_bytes += size;
        }
    }
    // EDEN_NVN_TEX_PREHASH: a check proven to find the bytes unchanged takes today's
    // "unchanged" branch without hashing (nvn_tex_prehash.h; verify mode hashes anyway).
    bool gate_skip = false;
    TexPrehash::Record gate_record;
    if (tex_prehash && !pool.IsVirtual()) [[unlikely]] {
        if (watching) {
            if (s.tex_gate.slot) {
                s.tex_gate.slot->arm_gen.store(0, std::memory_order_release);
            }
        } else {
            gate_skip = TexGateDecide(s, *pool_it->second, size, full, tick, gate_record);
            if (!gate_skip) {
                TexGateArm(s, *pool_it->second, size, tick);
            }
        }
    }
    if (gate_skip && !tex_prehash->Verify()) {
        if (full) {
            stats.storage_full_hashes.fetch_add(1, std::memory_order_relaxed);
            s.next_full_tick = tick + FULL_VERIFY_PRESENTS;
        }
        stats.cpu_checks.fetch_add(1, std::memory_order_relaxed);
        ++s.unchanged_checks;
        u64 backoff = CheckBackoff(s.unchanged_checks);
        if (size <= SMALL_STORAGE) {
            backoff = (std::min)(backoff, u64{3}); // a new glyph shows within a few frames
        }
        s.next_check_tick = tick + 1 + (s.dynamic ? (std::min)(backoff, u64{1}) : backoff);
        if (s.tex_gate.slot) {
            s.tex_gate.slot->next_check_tick.store(s.next_check_tick, std::memory_order_relaxed);
            s.tex_gate.slot->next_full_tick.store(s.next_full_tick, std::memory_order_relaxed);
        }
        return;
    }
    // Verify mode with a skip: an untracked byte flip (offset 0: in every signature and full
    // hash) before today's check reads the bytes, which it must see (nvn_tex_prehash.h).
    bool injected = false;
    if (gate_skip) [[unlikely]] {
        VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
        injected = memory != nullptr &&
                   tex_prehash->BeginInject(memory->Pointer(pool.address + s.pool_offset));
    }
    // A virtual pool's storage is gathered from the physical pools: not copied whole for a
    // signature alone.
    const bool signature_only = watching || !full;
    const bool gathered_lines = pool.IsVirtual() && signature_only;
    std::vector<u8> scratch;
    const auto guest_begin = std::chrono::steady_clock::now();
    const u8* data = nullptr;
    if (!gathered_lines) {
        // EDEN_NX2_NVN_PREREC_PROBE: the lines (and the full hash) are noted below, not the range.
        const VideoCommon::Nvn::PreRec::SuppressTexBytes lines_only;
        data = GuestBytes(pool, s.pool_offset, size, scratch);
    }
    if (!data && !gathered_lines) {
        if (injected) {
            tex_prehash->EndInject(false);
        }
        return;
    }
    if (data != nullptr && !pool.IsVirtual()) {
        VideoCommon::Nvn::PreRec::NoteLines(VideoCommon::Nvn::PreRec::Cls::TexSig,
                                            pool.address + s.pool_offset, size, pool.serial);
    }
    const auto begin = std::chrono::steady_clock::now();
    const u64 guest_ns = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(begin - guest_begin).count());
    const auto signature = [&]() -> u64 {
        if (data) {
            return SampleGuestBytes(data, size, mario_kart_world);
        }
        return SampleGuestBytesBy(size, [&](u64 offset, u64 count, u8* destination) {
            std::memset(destination, 0, static_cast<size_t>(count));
            GatherVirtual(pool, s.pool_offset + offset, count, destination);
        }, mario_kart_world);
    };
    bool changed = false;
    bool verify_signature_differs = false;
    u64 hashed = (std::min)(size, u64{16384});
    if (gate_skip) [[unlikely]] {
        // Verify: today's check, compared; the decision and bookkeeping are the skip's.
        const u64 signature_now = signature();
        verify_signature_differs = signature_now != s.sample_hash;
        bool today_changed = verify_signature_differs;
        if (!today_changed && full) {
            today_changed = HashGuestBytes(data, size) != s.guest_hash;
            hashed += size;
        }
        ++tex_prehash->window.verified;
        VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
        if (injected) {
            tex_prehash->EndInject(today_changed);
        } else if (today_changed) {
            const TexPrehash::DisagreementInfo info{
                .pool_serial = pool.serial,
                .address = pool.address + s.pool_offset,
                .size = size,
                .width = s.extent.width,
                .height = s.extent.height,
                .format = s.desc.format,
                .full = full,
                .record = &gate_record,
                .epoch_now = split_raw_epoch.load(std::memory_order_acquire),
                .completed_now = completed_seq.load(std::memory_order_acquire),
                .raw_submitted = raw_flights_submitted.load(std::memory_order_acquire),
                .block_seq = s.tex_gate.slot ? s.tex_gate.slot->block_seq.load() : 0,
                .stopped = memory == nullptr || memory->Stopped(),
                .signature_differs = verify_signature_differs,
            };
            tex_prehash->NoteDisagreement(info);
        }
        if (full) {
            stats.storage_full_hashes.fetch_add(1, std::memory_order_relaxed);
            stats.storage_full_hash_bytes.fetch_add(size, std::memory_order_relaxed);
            s.next_full_tick = tick + FULL_VERIFY_PRESENTS;
        }
    } else if (watching) {
        const u64 sample = signature();
        changed = s.watch_valid && sample != s.sample_hash;
        s.sample_hash = sample;
        s.watch_valid = true;
        if (data != nullptr && RaValidating()) [[unlikely]] {
            RaNoteTexCheck(pool, pool.address + s.pool_offset, s.pool_offset, size, sample,
                           false, 0);
        }
    } else {
        const u64 sample = signature();
        changed = sample != s.sample_hash;
        u64 full_hash = 0;
        const bool full_taken = !changed && full;
        if (full_taken) {
            if (!pool.IsVirtual()) {
                VideoCommon::Nvn::PreRec::NoteRead(VideoCommon::Nvn::PreRec::Cls::TexFull,
                                                   pool.address + s.pool_offset, size,
                                                   pool.serial);
            }
            full_hash = HashGuestBytes(data, size);
            changed = full_hash != s.guest_hash;
            hashed += size;
            stats.storage_full_hashes.fetch_add(1, std::memory_order_relaxed);
            stats.storage_full_hash_bytes.fetch_add(size, std::memory_order_relaxed);
            s.next_full_tick = tick + FULL_VERIFY_PRESENTS;
        }
        if (data != nullptr && RaValidating()) [[unlikely]] {
            RaNoteTexCheck(pool, pool.address + s.pool_offset, s.pool_offset, size, sample,
                           full_taken, full_hash);
        }
    }
    const u64 check_ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                              std::chrono::steady_clock::now() - begin)
                                              .count());
    stats.hash_ns.fetch_add(check_ns, std::memory_order_relaxed);
    stats.hash_bytes.fetch_add(hashed, std::memory_order_relaxed);
    if (s.desc.format != 0 && FormatBlockInfo(s.desc.format).width > 1) {
        stats.hash_bytes_compressed.fetch_add(hashed, std::memory_order_relaxed);
    }
    if (s.dynamic) {
        stats.hash_bytes_dynamic.fetch_add(hashed, std::memory_order_relaxed);
    }
    stats.cpu_checks.fetch_add(1, std::memory_order_relaxed);
    if (Debug::Get().upload_log && check_ns + guest_ns > 1'000'000) {
        LOG_INFO(Render_Vulkan,
                 "[nvn] coherency check: present {} {}x{} fmt {:#x} size {:.2f} MB origin {} pool "
                 "{} full {} forced {} changed {}: hash {:.2f} ms, guest read {:.2f} ms",
                 stats.presents.load(std::memory_order_relaxed), s.extent.width, s.extent.height,
                 s.desc.format, size / 1048576.0, static_cast<int>(s.origin),
                 pool.IsVirtual() ? "virtual" : (pool.HasHost() ? "host" : "nohost"), full, forced,
                 changed, check_ns / 1e6, guest_ns / 1e6);
    }
    if (!changed) {
        ++s.unchanged_checks;
        u64 backoff = CheckBackoff(s.unchanged_checks);
        if (size <= SMALL_STORAGE) {
            backoff = (std::min)(backoff, u64{3}); // a new glyph shows within a few frames
        }
        s.next_check_tick = tick + 1 + (s.dynamic ? (std::min)(backoff, u64{1}) : backoff);
        if (s.tex_gate.slot) {
            s.tex_gate.slot->next_check_tick.store(s.next_check_tick, std::memory_order_relaxed);
            s.tex_gate.slot->next_full_tick.store(s.next_full_tick, std::memory_order_relaxed);
        }
        return;
    }
    stats.cpu_changes.fetch_add(1, std::memory_order_relaxed);
    s.dynamic = true;
    s.force_upload = true; // makes the upload treat it as a later one
}

bool Context::Impl::StorageNeedsCommands(Storage& s, bool will_overwrite, u64 seq,
                                         bool scaled_ok) {
    const ReplayPhases::Scope all_phase{ReplayPhases::Phase::NeedsAll};
    if (!EnsureImage(s)) {
        return false;
    }
    if (s.scale_owner != nullptr) {
        return false; // a twin has no guest bytes: its owner decides what is current
    }
    if (will_overwrite) {
        return false;
    }
    if (!scaled_ok && Rescale::NativeIsStale(s.scale_state)) {
        return true; // the guest-sized image has to be made from the twin
    }
    if (s.sparse) {
        VerifySparse(s);
        return s.sparse->HasWork();
    }
    if (!s.contents_valid || s.force_upload) {
        return true;
    }
    {
        const ReplayPhases::Scope detect_phase{ReplayPhases::Phase::NeedsDetect};
        DetectGuestChange(s, seq);
    }
    if (s.force_upload) {
        return true;
    }
    const PoolIndex* index_ptr = static_cast<const PoolIndex*>(s.pool_index);
    if (index_ptr == nullptr || s.pool_index_epoch != storage_index_epoch) {
        const auto index_it = storage_index.find(s.pool_serial);
        if (index_it == storage_index.end()) {
            return false;
        }
        index_ptr = &index_it->second;
        s.pool_index = index_ptr;
        s.pool_index_epoch = storage_index_epoch;
    }
    const PoolIndex& index = *index_ptr;
    if (s.pool_gen_seen == index.write_gen) {
        return false;
    }
    const ReplayPhases::Scope scan_phase{ReplayPhases::Phase::NeedsScan};
    const u64 begin = s.pool_offset;
    const u64 end = begin + s.layout.size;
    const u64 scan_from = begin > index.max_size ? begin - index.max_size : 0;
    for (auto it = index.spans.lower_bound(scan_from); it != index.spans.end() && it->first < end;
         ++it) {
        const Storage& other = *it->second;
        if (&other != &s && (other.image || Rescale::NativeIsStale(other.scale_state)) &&
            other.gpu_write_gen > s.sync_gen &&
            it->first + other.layout.size > begin) {
            return true;
        }
    }
    return false;
}

Tegra::Texture::ASTC::Channels Context::Impl::ClassifyStandin(const Storage& s, const u8* data,
                                                              u64 size, bool& empty) {
    const BlockInfo block = FormatBlockInfo(s.desc.format);
    const auto begin = std::chrono::steady_clock::now();
    const Tegra::Texture::ASTC::Channels channels = Tegra::Texture::ASTC::ClassifyChannels(
        std::span<const u8>{data, static_cast<size_t>(size)}, block.width, block.height, &empty);
    const u64 ns = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                        std::chrono::steady_clock::now() - begin)
                                        .count());
    const size_t bucket = empty ? 4 : static_cast<size_t>(channels);
    for (SmartBcCounts* counts : {&smart_total, &smart_window}) {
        counts->classes[bucket].fetch_add(1, std::memory_order_relaxed);
        counts->scan_bytes.fetch_add(size, std::memory_order_relaxed);
        counts->scan_ns.fetch_add(ns, std::memory_order_relaxed);
    }
    return channels;
}

void Context::Impl::ChooseSmartStandin(Storage& s) {
    s.standin_empty = false;
    s.standin_fresh = false;
    s.standin_class = Tegra::Texture::ASTC::Channels::Rgba;
    const auto pool_it = pools.find(s.pool_serial);
    if (pool_it == pools.end()) {
        return;
    }
    const Pool& pool = *pool_it->second;
    if ((!pool.pending.load(std::memory_order_acquire) && pool.backing == Pool::Backing::None) ||
        s.pool_offset >= pool.size) {
        return;
    }
    const u64 size = (std::min)(s.layout.size, pool.size - s.pool_offset);
    std::vector<u8> scratch;
    const u8* data = GuestBytes(pool, s.pool_offset, size, scratch);
    if (data == nullptr) {
        return;
    }
    bool empty = false;
    const Tegra::Texture::ASTC::Channels channels = ClassifyStandin(s, data, size, empty);
    s.standin_fresh = true;
    if (empty) {
        s.standin_empty = true; // not loaded yet: BC3 until a whole upload brings blocks
        return;
    }
    s.standin_class = channels;
    SetStandin(s, SmartStandinFor(channels,
                                  VideoCore::Surface::IsPixelFormatSRGB(s.host.pixel_format) ||
                                      s.standin_srgb_view,
                                  smart_bc));
}

void Context::Impl::SetStandin(Storage& s, AstcStandin standin) {
    s.host = ToBcStandin(s.host, standin);
    const u32 block_bytes = BcStandinBlockBytes(s.host);
    u64 bytes = 0;
    for (u32 level = 0; level < s.levels; ++level) {
        bytes += BcBlocksSize((std::max)(s.extent.width >> level, 1u),
                              (std::max)(s.extent.height >> level, 1u), 1, block_bytes);
    }
    s.vram_bytes = bytes * s.layers;
}

bool Context::Impl::EnsureImage(Storage& s) {
    if (s.image) {
        return true;
    }
    if (s.image_failed) {
        return false;
    }
    // The machine is running out of memory under the replay's video memory: no new image (not
    // `image_failed`: it is made at a later use, once memory is back).
    if (PollMemoryGuard()) {
        ++memory_guard.refused;
        return false;
    }
    // The first image of a smart stand-in (Recorder::RefitStandin chooses for later ones).
    if (s.smart_standin && !s.standin_pinned && s.origin == StorageOrigin::None) {
        ChooseSmartStandin(s);
    }
    try {
        const VkImageCreateInfo image_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = s.create_flags,
            .imageType = s.type,
            .format = s.host.format,
            .extent = s.extent,
            .mipLevels = s.levels,
            .arrayLayers = s.layers,
            .samples = s.samples,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = s.usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        if (s.sparse) {
            // Tile-backed: a sparse residency image with no memory; the tiles the pool maps get
            // theirs (SparseBacking::Prepare).
            const char* reason = "the device cannot create a sparse residency image";
            s.image = allocator->CreateUnbackedImage(image_info);
            if (s.image && s.sparse->Attach(*this, s, reason)) {
                s.current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
                stats.storages_created.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
            LOG_WARNING(Render_Vulkan,
                        "[nvn] sparse texture {}x{}x{} ({} levels, format {:#x}) is not tile "
                        "backed: {}; it is an ordinary image",
                        s.extent.width, s.extent.height, s.layers, s.levels, s.desc.format, reason);
            s.image = vk::Image{};
            s.sparse.reset();
            // The whole-storage paths start from scratch.
            s.contents_valid = false;
            s.origin = StorageOrigin::None;
        }
        // EDEN_NX2_NVN_STORAGE_DEDICATED=1: render-target-capable storages get their own allocation
        // (boot-crash A/B: a ROP write fault inside a fresh render target placed in a shared block).
        static const bool dedicated_targets = [] {
            const char* v = std::getenv("EDEN_NX2_NVN_STORAGE_DEDICATED");
            return v != nullptr && *v != '\0' && v[0] != '0';
        }();
        const bool dedicated = dedicated_targets && s.host.attachable;
        s.image = allocator->CreateImage(image_info, dedicated, AllocationKind::StorageImage);
    } catch (const vk::Exception& exception) {
        s.image_failed = true;
        stats.textures_failed.fetch_add(1, std::memory_order_relaxed);
        LOG_ERROR(Render_Vulkan, "[nvn] texture storage ({}x{}x{}, {} levels, format {:#x}) image failed: {}",
                  s.extent.width, s.extent.height, s.extent.depth, s.levels, s.desc.format,
                  exception.what());
        return false;
    }
    s.current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    s.last_used_present = stats.presents.load(std::memory_order_relaxed);
    static const bool boot_diag = [] {
        const char* v = std::getenv("EDEN_NX2_NVN_BOOT_DIAG");
        return v != nullptr && *v != '\0' && v[0] != '0';
    }();
    if (boot_diag && s.last_used_present < 3) {
        // EDEN_NX2_NVN_BOOT_DIAG=1 (Aftermath names a faulting resource only by size and age):
        // every image created in the first frames with its memory size.
        LOG_INFO(Render_Vulkan, "[nvn] boot image {:#x}: {}x{}x{} L{} A{} fmt {} usage {:#x} mem {} bytes",
                 reinterpret_cast<uintptr_t>(*s.image), s.extent.width, s.extent.height,
                 s.extent.depth, s.levels, s.layers, static_cast<int>(s.host.format), s.usage,
                 device.GetLogical().GetImageMemoryRequirements(*s.image).size);
    }
    stats.storages_created.fetch_add(1, std::memory_order_relaxed);
    if (s.cached) {
        storage_cache_bytes += s.vram_bytes;
        TrimStorageCache();
        stats.storage_cache_bytes.store(storage_cache_bytes, std::memory_order_relaxed);
    }
    return true;
}

const u8* Context::Impl::GuestBytes(const Pool& pool, u64 offset, u64 size,
                                    std::vector<u8>& scratch) const {
    if (size == 0 || offset > pool.size || size > pool.size - offset) {
        return nullptr;
    }
    VC::GuestMemory* memory = guest.load(std::memory_order_acquire);
    if (!pool.IsVirtual()) {
        VC::PreRec::NoteRead(VC::PreRec::Cls::TexBytes, pool.address + offset, size, pool.serial);
        const u8* out = nullptr;
        if (pool.HasHost()) {
            out = pool.HostRead(VC::PreRec::Cls::TexBytes, offset, size);
        } else if (const u8* pointer = GuestPointer(pool.address + offset, size)) {
            out = pointer;
        } else {
            scratch.resize(static_cast<size_t>(size));
            if (memory && memory->Read(pool.address + offset, scratch.data(), size)) {
                out = scratch.data();
            }
        }
        if (RaValidating() && !VC::PreRec::TexBytesSuppressed()) [[unlikely]] {
            // Record-ahead stage 2: texture bytes read ahead (an upload): hashed again at the
            // release. A coherency check notes its own hashes instead.
            RaNoteTexBytes(pool, offset, size, out);
        }
        return out;
    }
    // A virtual pool: gather the mapped parts from the physical pools, zero elsewhere.
    scratch.assign(static_cast<size_t>(size), 0);
    GatherVirtual(pool, offset, size, scratch.data());
    return scratch.data();
}

void Context::Impl::GatherVirtual(const Pool& pool, u64 offset, u64 size, u8* out) const {
    VC::GuestMemory* memory = guest.load(std::memory_order_acquire);
    auto it = pool.virtual_runs.upper_bound(offset);
    if (it != pool.virtual_runs.begin()) {
        --it;
    }
    for (; it != pool.virtual_runs.end() && it->first < offset + size; ++it) {
        const u64 run_begin = it->first;
        const Pool::VirtualRun& run = it->second;
        const u64 begin = (std::max)(run_begin, offset);
        const u64 end = (std::min)(run_begin + run.size, offset + size);
        if (begin >= end) {
            continue;
        }
        const auto physical = pools.find(run.physical_serial);
        if (physical == pools.end()) {
            continue;
        }
        const Pool& source = *physical->second;
        const u64 source_offset = run.physical_offset + (begin - run_begin);
        if (source_offset > source.size || end - begin > source.size - source_offset) {
            continue;
        }
        u8* const target = out + (begin - offset);
        VC::PreRec::NoteRead(VC::PreRec::Cls::TexGather, source.address + source_offset,
                             end - begin, source.serial);
        if (const u8* pointer =
                source.HasHost()
                    ? source.HostRead(VC::PreRec::Cls::TexGather, source_offset, end - begin)
                    : GuestPointer(source.address + source_offset, end - begin)) {
            std::memcpy(target, pointer, static_cast<size_t>(end - begin));
        } else if (memory) {
            memory->Read(source.address + source_offset, target, end - begin);
        }
    }
}

bool Context::Impl::ResolveVirtualRange(const Pool& virtual_pool, u64 offset, u64 size,
                                        Pool*& physical, u64& physical_offset) const {
    const auto& runs = virtual_pool.virtual_runs;
    auto it = runs.upper_bound(offset);
    if (it == runs.begin()) {
        return false;
    }
    --it;
    u64 cursor = offset;
    const u64 end = offset + (std::max)(size, u64{1});
    Pool* first = nullptr;
    u64 first_offset = 0;
    for (; it != runs.end() && cursor < end; ++it) {
        const u64 run_begin = it->first;
        const Pool::VirtualRun& run = it->second;
        if (cursor < run_begin || cursor >= run_begin + run.size) {
            return false; // a hole
        }
        const auto found = pools.find(run.physical_serial);
        if (found == pools.end()) {
            return false;
        }
        Pool* const pool = found->second.get();
        const u64 at = run.physical_offset + (cursor - run_begin);
        if (!first) {
            first = pool;
            first_offset = at;
        } else if (pool != first || at != first_offset + (cursor - offset)) {
            return false; // scattered over the physical pools: one buffer cannot cover it
        }
        cursor = run_begin + run.size;
    }
    if (cursor < end || !first) {
        return false;
    }
    physical = first;
    physical_offset = first_offset;
    return true;
}

std::string Context::Impl::DescribeUnresolved(u64 address, u64 size) const {
    auto it = pools_by_address.upper_bound(address);
    if (it != pools_by_address.begin()) {
        --it;
        const Pool& pool = *it->second;
        if (address >= pool.address && address - pool.address <= pool.size) {
            if (pool.IsVirtual()) {
                return fmt::format("range {:#x}+{:#x} of virtual pool {:#x} is unmapped or is "
                                   "backed by scattered physical ranges",
                                   address - pool.address, size, pool.address);
            }
            return fmt::format("range {:#x}+{:#x} leaves pool {:#x} ({:#x} bytes)",
                               address - pool.address, size, pool.address, pool.size);
        }
    }
    return fmt::format("address {:#x} (+{:#x}) is not inside a memory pool", address, size);
}

namespace {

/// A mapping of the virtual pool over a range, in the form the old and the new state of a remap
/// are compared in.
struct RunView {
    u64 start;
    u64 size;
    u64 serial;
    u64 physical;
    bool operator==(const RunView&) const = default;
};

/// The runs of [begin, end), cut to it, neighbours that continue each other merged.
std::vector<RunView> SnapshotRuns(
    const std::map<u64, Pool::VirtualRun>& runs, u64 begin, u64 end,
    const std::unordered_map<u64, std::shared_ptr<Pool>>* physical_pools) {
    std::vector<RunView> out;
    auto it = runs.upper_bound(begin);
    if (it != runs.begin()) {
        --it;
    }
    for (; it != runs.end() && it->first < end; ++it) {
        const u64 run_end = it->first + it->second.size;
        const u64 from = (std::max)(it->first, begin);
        const u64 to = (std::min)(run_end, end);
        if (from >= to) {
            continue;
        }
        RunView view{from, to - from, it->second.physical_serial,
                     it->second.physical_offset + (from - it->first)};
        if (physical_pools != nullptr) {
            const auto physical = physical_pools->find(view.serial);
            if (physical != physical_pools->end() && physical->second->address != 0 &&
                (physical->second->flags & PoolFlag::Virtual) == 0) {
                view.physical += physical->second->address;
                view.serial = 0; // canonical guest backing, independent of the pool object
            }
        }
        if (!out.empty() && out.back().start + out.back().size == view.start &&
            out.back().serial == view.serial &&
            out.back().physical + out.back().size == view.physical) {
            out.back().size += view.size;
        } else {
            out.push_back(view);
        }
    }
    return out;
}

} // Anonymous namespace

bool Context::Impl::VirtualRangeMapped(const Pool& virtual_pool, u64 offset, u64 size) {
    const auto& runs = virtual_pool.virtual_runs;
    auto it = runs.upper_bound(offset);
    if (it != runs.begin()) {
        const auto previous = std::prev(it);
        if (previous->first + previous->second.size > offset) {
            return true;
        }
    }
    return it != runs.end() && it->first < offset + size;
}

void Context::Impl::MapVirtualNow(u64 pool_serial, const std::vector<VC::VirtualMapping>& mappings) {
    ++virtual_map_generation; // a virtual pool's runs may change below
    const auto found = pools.find(pool_serial);
    if (found == pools.end() || !found->second->IsVirtual()) {
        LOG_ERROR(Render_Vulkan, "[nvn] MapVirtual on {:#x}: not a virtual pool", pool_serial);
        return;
    }
    Pool& pool = *found->second;
    if (mario_kart_world && !mappings.empty()) {
        ++mkw_mapping_generation;
    }
    PoolIndex* const index = [&]() -> PoolIndex* {
        const auto it = storage_index.find(pool_serial);
        return it == storage_index.end() ? nullptr : &it->second;
    }();
    for (const VC::VirtualMapping& mapping : mappings) {
        if (mapping.size == 0 || mapping.virtual_offset > pool.size ||
            mapping.size > pool.size - mapping.virtual_offset) {
            continue;
        }
        const u64 begin = mapping.virtual_offset;
        const u64 end = begin + mapping.size;
        mkw_sparse_users_dirty = true;
        auto& runs = pool.virtual_runs;
        // As on the emulated GPU path, mapping the same backing through another pool
        // object must not replace a lightmap tile with an upload of older guest bytes.
        const auto* physical_pools = mario_kart_world ? &pools : nullptr;
        const std::vector<RunView> before = SnapshotRuns(runs, begin, end, physical_pools);
        // Cut the existing runs that overlap [begin, end).
        auto it = runs.lower_bound(begin);
        if (it != runs.begin()) {
            auto previous = std::prev(it);
            const u64 previous_end = previous->first + previous->second.size;
            if (previous_end > begin) {
                Pool::VirtualRun tail = previous->second;
                const u64 tail_begin = end;
                previous->second.size = begin - previous->first;
                if (previous_end > end) {
                    tail.physical_offset += tail_begin - previous->first;
                    tail.size = previous_end - tail_begin;
                    runs.emplace(tail_begin, tail);
                }
            }
        }
        it = runs.lower_bound(begin);
        while (it != runs.end() && it->first < end) {
            const u64 run_end = it->first + it->second.size;
            if (run_end > end) {
                Pool::VirtualRun tail = it->second;
                tail.physical_offset += end - it->first;
                tail.size = run_end - end;
                runs.emplace(end, tail);
            }
            it = runs.erase(it);
        }
        if (mapping.physical_pool_serial != 0) {
            runs.emplace(begin, Pool::VirtualRun{mapping.size, mapping.physical_pool_serial,
                                                 mapping.physical_offset});
        }
        // A remap that leaves the bytes where they were (the game maps what is already mapped)
        // changes nothing for the storages over it.
        if (!index || SnapshotRuns(runs, begin, end, physical_pools) == before) {
            continue;
        }
        // Storages placed over the remapped bytes read different bytes now.
        const u64 scan_from = begin > index->max_size ? begin - index->max_size : 0;
        for (auto s = index->spans.lower_bound(scan_from);
             s != index->spans.end() && s->first < end; ++s) {
            Storage& storage = *s->second;
            if (s->first + storage.layout.size <= begin) {
                continue;
            }
            if (storage.sparse) {
                // Tile by tile: only the tiles of the range are bound, unbound or uploaded again.
                const u64 from = (std::max)(begin, s->first) - s->first;
                const u64 to = (std::min)(end, s->first + storage.layout.size) - s->first;
                storage.sparse->Remap(from, to, [&](u64 offset, u64 size) {
                    return VirtualRangeMapped(pool, s->first + offset, size);
                });
                continue;
            }
            storage.contents_valid = false;
            storage.origin = StorageOrigin::None;
            storage.gpu_write_gen = 0; // no longer a source for overlapping textures
            storage.force_upload = false;
        }
    }
    stats.virtual_maps.fetch_add(1, std::memory_order_relaxed);
    if (stats.virtual_maps.load(std::memory_order_relaxed) <= 4) {
        u64 mapped = 0;
        for (const auto& [offset, run] : pool.virtual_runs) {
            mapped += run.size;
        }
        LOG_INFO(Render_Vulkan,
                 "[nvn] MapVirtual: virtual pool {:#x} ({:#x} bytes): {} request(s), now {} run(s), "
                 "{:#x} bytes mapped",
                 pool.address, pool.size, mappings.size(), pool.virtual_runs.size(), mapped);
    }
}

void Context::Impl::VerifySparse(Storage& s) {
    SparseBacking& sparse = *s.sparse;
    if (s.origin == StorageOrigin::Gpu) {
        return; // the GPU wrote it: the guest bytes are not what it holds
    }
    const auto pool_it = pools.find(s.pool_serial);
    if (pool_it == pools.end()) {
        return;
    }
    // At most this many tiles per present in all, and per call: a 64 KiB gather and a hash each.
    constexpr u32 TILES_PER_TICK = 64;
    constexpr u32 TILES_PER_CALL = 16;
    const u64 tick = PollTick();
    if (sparse_verify_tick != tick) {
        sparse_verify_tick = tick;
        sparse_verify_count = 0;
    }
    thread_local std::vector<u8> bytes;
    bytes.resize(static_cast<size_t>(SPARSE_TILE_BYTES));
    for (u32 n = 0; n < TILES_PER_CALL && sparse_verify_count < TILES_PER_TICK; ++n) {
        const u32 id = sparse.NextVerifyTile();
        if (id == ~0u) {
            break;
        }
        ++sparse_verify_count;
        std::fill(bytes.begin(), bytes.end(), u8{0});
        GatherVirtual(*pool_it->second, s.pool_offset + sparse.Layout().TileOffset(id),
                      SPARSE_TILE_BYTES, bytes.data());
        if (Common::CityHash64(reinterpret_cast<const char*>(bytes.data()), bytes.size()) !=
            sparse.TileHash(id)) {
            sparse.MarkTileDirty(id);
        }
    }
}

void Context::Impl::FlushedRuns(Pool& pool, u64 begin, u64 size,
                                std::vector<std::pair<u64, u64>>& runs) {
    constexpr u64 PAGE = MirrorPages::PAGE;
    PoolMirror& mirror = *pool.vram_mirror;
    VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    const u64 end = (std::min)(begin + size, pool.size);
    if (memory == nullptr) {
        runs.emplace_back(begin, end);
        return;
    }
    thread_local std::vector<std::pair<u64, u64>> written;
    written.clear();
    // The guest CPU's stores to pages the mirror holds are all in the tracker.
    memory->TakeWritten(mirror.track_id, pool.address + begin, end - begin, written);
    size_t next = 0;
    u64 kept = 0;
    u64 taken = 0;
    const auto push = [&](u64 from, u64 to) {
        from = (std::max)(from, begin);
        to = (std::min)(to, end);
        if (from >= to) {
            return;
        }
        if (!runs.empty() && runs.back().second == from) {
            runs.back().second = to;
        } else {
            runs.emplace_back(from, to);
        }
    };
    for (u64 page = begin / PAGE; page * PAGE < end; ++page) {
        const u64 offset = page * PAGE;
        while (next < written.size() && written[next].second <= pool.address + offset) {
            ++next;
        }
        const bool was_written = next < written.size() && written[next].first < pool.address + offset + PAGE;
        if (was_written || !mirror.pages.AllValid(offset, PAGE)) {
            push(offset, offset + PAGE);
            ++taken;
        } else {
            ++kept;
        }
    }
    stats.vram_flush_pages_kept.fetch_add(kept, std::memory_order_relaxed);
    stats.vram_flush_pages_taken.fetch_add(taken, std::memory_order_relaxed);
}

std::span<const u64> Context::Impl::MkwSparseVirtualUsers(u64 physical_serial) {
    if (mkw_sparse_users_dirty) {
        mkw_sparse_physical_users.clear();
        for (const u64 serial : sparse_virtual_pools) {
            const auto found = pools.find(serial);
            if (found == pools.end()) {
                continue;
            }
            for (const auto& [offset, run] : found->second->virtual_runs) {
                auto& users = mkw_sparse_physical_users[run.physical_serial];
                // sparse_virtual_pools is ordered: append each virtual pool only once.
                if (users.empty() || users.back() != serial) {
                    users.push_back(serial);
                }
            }
        }
        mkw_sparse_users_dirty = false;
    }
    const auto found = mkw_sparse_physical_users.find(physical_serial);
    return found != mkw_sparse_physical_users.end() ? std::span<const u64>{found->second}
                                                   : std::span<const u64>{};
}

void Context::Impl::NotifyCpuWriteNow(u64 address, u64 size, bool host_write) {
    auto pool_it = pools_by_address.upper_bound(address);
    if (pool_it == pools_by_address.begin()) {
        return;
    }
    --pool_it;
    Pool& pool = *pool_it->second;
    if (address < pool.address || address - pool.address >= pool.size) {
        return;
    }
    const u64 begin = address - pool.address;
    const u64 end = begin + size;
    if (ra.enabled && pool.vram_mirror && pool.vram_mirror->ra_log != nullptr) [[unlikely]] {
        // Record-ahead: a decision recorded ahead used the pool's mirror; its effects on the
        // mirror are applied at the release (frozen write view).
        RaVramDeferCpuWrite(pool, begin, size, host_write);
    } else {
        // What the mirror has to look at: the whole range, or with `flush_tracked` the pages of a
        // guest flush that the guest wrote or the mirror does not hold.
        thread_local std::vector<std::pair<u64, u64>> runs;
        runs.clear();
        CpuWriteMirrorRuns(pool, begin, size, host_write, runs);
        // Live storage shadows are normally authoritative until their segment ends. Refresh only
        // the CPU-written overlap before either a whole-shadow copy-back or a mirror word-merge.
        for (const auto& [run_begin, run_end] : runs) {
            for (auto& [owner_queue, draw] : draw_contexts) {
                static_cast<void>(owner_queue);
                draw->OnCpuWrite(pool, run_begin, run_end - run_begin);
            }
        }
        CpuWriteMirrorPageReset(pool, begin, size, host_write);
        // A pool mirror copies the range in again at its next use (the guest CPU's writes are seen
        // by the page tracker too; a host service's writes through a host pointer only through
        // this).
        if (pool.vram_mirror && pool.vram_mirror->monitor) {
            // EDEN_NX2_NVN_WB_MONITOR: how much of what the game declares written the page tracker
            // saw written (the rest is only a flush the guest made without a JIT store, or a host
            // service's write).
            MirrorMonitor& monitor = *pool.vram_mirror->monitor;
            VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
            const u64 last = (std::min)(end, pool.size);
            monitor.notify_calls.fetch_add(1, std::memory_order_relaxed);
            for (u64 page = begin / MirrorPages::PAGE; page * MirrorPages::PAGE < last; ++page) {
                const u64 offset = page * MirrorPages::PAGE;
                monitor.notify_pages.fetch_add(1, std::memory_order_relaxed);
                if (!pool.vram_mirror->pages.AllValid(offset, MirrorPages::PAGE)) {
                    monitor.notify_invalid.fetch_add(1, std::memory_order_relaxed);
                } else if (memory != nullptr &&
                           memory->AnyWritten(pool.vram_mirror->track_id, pool.address + offset,
                                              MirrorPages::PAGE)) {
                    monitor.notify_dirty.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        for (const auto& [run_begin, run_end] : runs) {
            MirrorInvalidate(pool, run_begin, run_end - run_begin);
        }
    }
    if (!sparse_virtual_pools.empty() && !pool.IsVirtual()) {
        // The sparse textures that map these bytes through a virtual pool read them again.
        const auto notify_virtual = [&](u64 serial) {
            stats.sparse_write_pool_checks.fetch_add(1, std::memory_order_relaxed);
            const auto virtual_it = pools.find(serial);
            const auto index_it = storage_index.find(serial);
            if (virtual_it == pools.end() || index_it == storage_index.end()) {
                return;
            }
            for (const auto& [run_begin, run] : virtual_it->second->virtual_runs) {
                if (run.physical_serial != pool.serial || run.physical_offset >= end ||
                    run.physical_offset + run.size <= begin) {
                    continue;
                }
                const u64 from = (std::max)(run.physical_offset, begin) - run.physical_offset;
                const u64 to = (std::min)(run.physical_offset + run.size, end) - run.physical_offset;
                const u64 virtual_from = run_begin + from;
                const u64 virtual_to = run_begin + to;
                const PoolIndex& virtual_index = index_it->second;
                const u64 scan_from = virtual_from > virtual_index.max_size
                                          ? virtual_from - virtual_index.max_size
                                          : 0;
                for (auto it = virtual_index.spans.lower_bound(scan_from);
                     it != virtual_index.spans.end() && it->first < virtual_to; ++it) {
                    Storage& storage = *it->second;
                    if (!storage.sparse || it->first + storage.layout.size <= virtual_from) {
                        continue;
                    }
                    storage.sparse->ContentChanged(
                        (std::max)(virtual_from, it->first) - it->first,
                        (std::min)(virtual_to, it->first + storage.layout.size) - it->first);
                }
            }
        };
        if (mario_kart_world) {
            for (const u64 serial : MkwSparseVirtualUsers(pool.serial)) {
                notify_virtual(serial);
            }
        } else {
            for (const u64 serial : sparse_virtual_pools) {
                notify_virtual(serial);
            }
        }
    }
    const auto index_it = storage_index.find(pool.serial);
    if (index_it == storage_index.end()) {
        return;
    }
    const PoolIndex& index = index_it->second;
    const u64 scan_from = begin > index.max_size ? begin - index.max_size : 0;
    for (auto it = index.spans.lower_bound(scan_from); it != index.spans.end() && it->first < end;
         ++it) {
        Storage& storage = *it->second;
        if (it->first + storage.layout.size <= begin) {
            continue;
        }
        if (storage.origin == StorageOrigin::Cpu) {
            // The bytes may have changed: look at them at the next use, whatever the back-off says.
            storage.force_check = true;
        }
        TexGateDisarmStorage(storage);
    }
}

void Context::Impl::CpuWriteMirrorRuns(Pool& pool, u64 begin, u64 size, bool host_write,
                                       std::vector<std::pair<u64, u64>>& runs) {
    if (flush_tracked && !host_write && pool.vram_mirror) {
        FlushedRuns(pool, begin, size, runs);
    } else {
        runs.emplace_back(begin, begin + size);
    }
}

void Context::Impl::CpuWriteMirrorPageReset(Pool& pool, u64 begin, u64 size, bool host_write) {
    const u64 end = begin + size;
    if (!pool.pending.load(std::memory_order_acquire) && !pool.mirror_pages.empty()) {
        // A mirror's pages in the range are compared with the guest's again at their next use,
        // even in a submission that already found them unchanged. Trusted pages stay trusted for a
        // guest flush with `flush_tracked` (the guest CPU's stores are in the page tracker), not
        // for a host service's write through a host pointer.
        const bool untrust = host_write || !flush_tracked;
        const u64 last = (std::min)(end, pool.size);
        for (u64 page = begin / Pool::MIRROR_PAGE; page * Pool::MIRROR_PAGE < last; ++page) {
            Pool::MirrorPage& mirror_page = pool.mirror_pages[static_cast<size_t>(page)];
            mirror_page.checked = 0;
            if (untrust) {
                mirror_page.trusted = false;
            }
        }
    }
}

void Context::Impl::NotifyHostWriteNow(u64 address, u64 size) {
    // What a CPU write gets: mirror pages compared again, sparse tiles and CPU-sourced storages
    // looked at again.
    NotifyCpuWriteNow(address, size, true);
    auto pool_it = pools_by_address.upper_bound(address);
    if (pool_it == pools_by_address.begin()) {
        return;
    }
    --pool_it;
    Pool& pool = *pool_it->second;
    if (address < pool.address || address - pool.address >= pool.size || pool.IsVirtual()) {
        return;
    }
    // Buffer fills need 4-byte aligned ranges; the extra bytes are the guest's own.
    const u64 begin = (address - pool.address) & ~u64{3};
    const u64 end = (std::min)((address - pool.address + size + 3) & ~u64{3}, pool.size);
    // A GPU-only pool's buffer is the GPU's copy of the memory (an imported one is the memory
    // itself, a mirror compares its pages at their next use): it takes the new bytes now. A pool
    // still waiting for its first upload reads them then.
    if (pool.backing == Pool::Backing::DeviceLocal && !pool.upload_pending &&
        !pool.pending.load(std::memory_order_acquire)) {
        const PoolUpload range{&pool, begin, end - begin};
        UploadPoolRanges({&range, 1}, true);
        stats.host_write_uploads.fetch_add(1, std::memory_order_relaxed);
    }
    // A texture inside the range holds the written bytes now, whatever the GPU wrote to it before
    // (a decoder frame over the plane the game cleared); one only partly inside keeps its image
    // (the rest of its bytes may be GPU contents the guest memory does not hold).
    const auto index_it = storage_index.find(pool.serial);
    if (index_it == storage_index.end()) {
        return;
    }
    const PoolIndex& index = index_it->second;
    for (auto it = index.spans.lower_bound(begin); it != index.spans.end() && it->first < end; ++it) {
        Storage& storage = *it->second;
        if (storage.sparse || it->first + storage.layout.size > end) {
            continue;
        }
        storage.force_upload = true;
        storage.force_check = true;
    }
}

void Context::Impl::NotifyTextureWrittenNow(u64 texture_serial) {
    const auto it = textures.find(texture_serial);
    if (it == textures.end() || !it->second->storage) {
        return;
    }
    Storage& storage = *it->second->storage;
    if (storage.sparse) {
        storage.sparse->MarkAllDirty(); // the game wrote texels: the tiles are uploaded again
        return;
    }
    storage.force_upload = true;
    storage.force_check = true;
    TexGateDisarmStorage(storage);
}

void Context::Impl::EvictPoolStorages(u64 pool_serial) {
    const auto index_it = storage_index.find(pool_serial);
    if (index_it == storage_index.end()) {
        return;
    }
    std::vector<Storage*> doomed;
    for (const auto& [begin, storage] : index_it->second.spans) {
        doomed.push_back(storage);
    }
    storage_index.erase(index_it);
    if (sparse_virtual_pools.erase(pool_serial) != 0) {
        mkw_sparse_users_dirty = true;
    }
    ++storage_index_epoch;
    for (auto it = storage_cache.begin(); it != storage_cache.end();) {
        if (it->first.pool_serial == pool_serial) {
            Storage& storage = *it->second;
            if (storage.image && !storage.sparse) {
                storage_cache_bytes -= (std::min)(storage_cache_bytes, storage.vram_bytes);
            }
            storage.cached = false;
            stats.storages_evicted.fetch_add(1, std::memory_order_relaxed);
            it = storage_cache.erase(it);
        } else {
            ++it;
        }
    }
    stats.storage_cache_bytes.store(storage_cache_bytes, std::memory_order_relaxed);
}

bool Context::Impl::StorageSuperseded(const Storage& s, u64 gap) const {
    // Only what the GPU wrote and what can take part in an overlap transfer is judged.
    if (s.origin != StorageOrigin::Gpu || s.gpu_write_gen == 0 || !s.image || s.sparse ||
        s.samples != VK_SAMPLE_COUNT_1_BIT || s.layout.size == 0) {
        return false;
    }
    const auto index_it = storage_index.find(s.pool_serial);
    if (index_it == storage_index.end()) {
        return false;
    }
    const PoolIndex& index = index_it->second;
    const u64 begin = s.pool_offset;
    const u64 end = begin + s.layout.size;
    const u64 scan_from = begin > index.max_size ? begin - index.max_size : 0;
    thread_local std::vector<std::pair<u64, u64>> covers;
    covers.clear();
    for (auto it = index.spans.lower_bound(scan_from); it != index.spans.end() && it->first < end;
         ++it) {
        const Storage& other = *it->second;
        const u64 other_end = it->first + other.layout.size;
        if (&other == &s || other_end <= begin || !other.image || other.sparse ||
            other.samples != VK_SAMPLE_COUNT_1_BIT || other.origin != StorageOrigin::Gpu ||
            other.gpu_write_gen <= s.gpu_write_gen) {
            continue;
        }
        // A pair whose contents no transfer could carry (ResolveOverlaps logged it) did not take
        // this storage's texels over.
        if (alias_logged.contains({reinterpret_cast<u64>(&s), reinterpret_cast<u64>(&other)})) {
            continue;
        }
        covers.emplace_back((std::max)(begin, it->first), (std::min)(end, other_end));
    }
    return gap != 0 ? RangesCoverNearly(covers, begin, end, gap) : RangesCover(covers, begin, end);
}

void Context::Impl::ReleaseStorageImage(Storage& s) {
    if (!s.image) {
        return;
    }
    // After every submission recorded so far (the ones that used it are among them).
    u64 after = (std::max)({completed_seq.load(std::memory_order_acquire),
                            replay_seq.load(std::memory_order_relaxed), s.last_used_seq});
    if (s.touched_seq != ~u64{0}) {
        after = (std::max)(after, s.touched_seq);
    }
    auto old_image = std::make_shared<vk::Image>(std::move(s.image));
    auto old_views = std::make_shared<std::vector<vk::ImageView>>(
        draw_resources ? draw_resources->TakeStorageViews(s) : std::vector<vk::ImageView>{});
    retired.push_back({after, [old_image, old_views]() mutable {
                           old_views->clear();
                           *old_image = vk::Image{};
                       }});
    s.image = vk::Image{};
    if (s.cached) {
        storage_cache_bytes -= (std::min)(storage_cache_bytes, s.vram_bytes);
    }
    // As a storage never used (MapVirtualNow resets the same fields of remapped storages): no
    // contents, no source for overlapping textures, nothing to watch in guest memory.
    s.current_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    s.contents_valid = false;
    s.origin = StorageOrigin::None;
    s.gpu_write_gen = 0;
    s.sync_gen = 0;
    s.force_upload = false;
    s.watch_valid = false;
    s.gpu_guest_sig_valid = false;
    s.gpu_takeover_armed = false;
    s.pool_gen_seen = ~u64{0};
    s.trim_checked_gen = ~u64{0};
}

void Context::Impl::ReleaseSupersededImages(u64 presents) {
    const u64 age = ReleaseSupersededAge();
    if (age == 0) {
        return;
    }
    constexpr size_t MAX_TESTS = 1024;
    struct Candidate {
        Storage* storage;
        long holders; ///< the cache's shared_ptr's use count
    };
    std::vector<Candidate> candidates;
    for (const auto& [key, storage] : storage_cache) {
        Storage& s = *storage;
        // Held by more than the cache (the idle ones are TrimStorageCache's), GPU-written, with
        // an ordinary image and no scaled twin.
        if (storage.use_count() == 1 || !s.image || s.sparse || s.scaled ||
            s.scale_owner != nullptr || s.origin != StorageOrigin::Gpu ||
            presents < s.last_used_present || presents - s.last_used_present < age) {
            continue;
        }
        candidates.push_back({&s, storage.use_count()});
    }
    if (candidates.empty()) {
        return;
    }
    // The references texture objects and their views hold. Any other holder (a submission being
    // recorded or in flight, a destroyed texture waiting for its last submission, a draw plan)
    // may still name the image: such a storage keeps it.
    std::unordered_map<const Storage*, long> texture_refs;
    for (const auto& [serial, texture] : textures) {
        if (texture && texture->storage) {
            ++texture_refs[texture->storage.get()];
        }
    }
    for (const auto& [key, texture] : target_aliases) {
        if (texture && texture->storage) {
            ++texture_refs[texture->storage.get()];
        }
    }
    if (draw_resources) {
        for (const auto& [serial, views] : draw_resources->texture_views) {
            if (views.storage) {
                ++texture_refs[views.storage.get()];
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.storage->last_used_seq < b.storage->last_used_seq;
    });
    size_t tests = 0;
    u64 released = 0;
    u64 released_bytes = 0;
    u64 busy = 0;
    for (const Candidate& candidate : candidates) {
        if (tests >= MAX_TESTS) {
            break;
        }
        Storage& s = *candidate.storage;
        const auto index = storage_index.find(s.pool_serial);
        const u64 gen = index != storage_index.end() ? index->second.write_gen : 0;
        if (s.trim_checked_gen == gen) {
            continue; // found live, and nothing was written in its pool since
        }
        ++tests;
        if (!StorageSuperseded(s, SupersededGap(s))) {
            s.trim_checked_gen = gen;
            continue;
        }
        const auto refs = texture_refs.find(&s);
        if (candidate.holders != 1 + (refs != texture_refs.end() ? refs->second : 0)) {
            ++busy;
            continue;
        }
        released_bytes += s.vram_bytes;
        ++released;
        ReleaseStorageImage(s);
    }
    if (released == 0) {
        return;
    }
    images_released += released;
    images_released_bytes += released_bytes;
    stats.storage_cache_bytes.store(storage_cache_bytes, std::memory_order_relaxed);
    if (images_released == released || presents >= images_released_logged_present + 300) {
        images_released_logged_present = presents;
        LOG_INFO(Render_Vulkan,
                 "[nvn] storage trim: present {}: {} superseded render target image(s) ({} MiB) "
                 "that live textures hold released after {} presents without a use ({} more "
                 "superseded but in use by a submission or a retiring texture; {} candidates, {} "
                 "tested); {} ({} MiB) so far, storage cache {} MiB "
                 "(EDEN_NVN_RELEASE_SUPERSEDED_PRESENTS, 0 off)",
                 presents, released, released_bytes >> 20, age, busy, candidates.size(), tests,
                 images_released, images_released_bytes >> 20, storage_cache_bytes >> 20);
    }
}

void Context::Impl::TrimStorageCache(bool at_present) {
    // A pass walks the whole cache: it runs when the cache has grown since the last one or after
    // a few presents, not at every storage created.
    constexpr u64 PASS_INTERVAL_PRESENTS = 30;
    // Superseded-storage tests per pass (each scans the storages over the tested one's bytes).
    constexpr size_t MAX_SUPERSEDED_TESTS = 1024;
    const u64 presents = stats.presents.load(std::memory_order_relaxed);
    const bool over_budget = storage_cache_bytes > storage_cache_budget;
    if (trim_legacy) {
        if (at_present || !over_budget ||
            (storage_cache_bytes < trim_floor_bytes && presents < trim_next_present)) {
            return;
        }
    } else if (at_present) {
        // The periodic pass: textures died, storages aged, the device budget moved. Strict
        // device-local allocations over the budget (EDEN_NX2_TRY_VMA_DEVICE_STRICT) bring it
        // forward: the driver pages until the cache sheds what it can rebuild.
        const u64 strict_over_budget = spills.over_budget_count.load(std::memory_order_relaxed);
        if (presents < trim_next_present && strict_over_budget == trim_seen_over_budget) {
            return;
        }
        trim_seen_over_budget = strict_over_budget;
    } else if ((!over_budget && trim_device_excess == 0) ||
               (storage_cache_bytes < trim_floor_bytes && presents < trim_next_present)) {
        return;
    }
    // The device over its pressure line (DevicePressurePercent of the VK_EXT_memory_budget budget;
    // the allocator's empty block space counts, the driver counts it too): what the cache can
    // rebuild goes until it is under it again, before the driver starts paging.
    u64 device_budget = 0;
    trim_device_excess = 0;
    if (!trim_legacy) {
        trim_device_excess = DevicePressureExcess(device, &device_budget);
        if (trim_device_excess != 0) {
            stats.storage_trim_pressure_passes.fetch_add(1, std::memory_order_relaxed);
        }
    }
    struct Candidate {
        u64 last_used;
        StorageKey key;
    };
    std::vector<Candidate> gpu_written;
    std::vector<Candidate> from_guest;
    for (const auto& [key, storage] : storage_cache) {
        // Only the cache holds it: no texture and no submission in flight uses it.
        if (storage.use_count() != 1 || !storage->image || storage->sparse) {
            continue;
        }
        if (storage->origin == StorageOrigin::Gpu) {
            gpu_written.push_back({storage->last_used_seq, key});
        } else {
            from_guest.push_back({storage->last_used_seq, key});
        }
    }
    const auto by_age = [](const Candidate& a, const Candidate& b) { return a.last_used < b.last_used; };
    const u64 budget_target = storage_cache_budget / 10 * 9;
    u64 target = over_budget ? budget_target : storage_cache_bytes;
    if (trim_device_excess != 0) {
        // The excess and a margin, so that the next pass does not find the device over at once.
        const u64 shed = trim_device_excess + device_budget / 32;
        target = (std::min)(target, storage_cache_bytes - (std::min)(storage_cache_bytes, shed));
    }
    enum class Reason { Budget, Superseded, Aged };
    const auto evict = [&](const StorageKey& key, Reason reason) {
        const auto it = storage_cache.find(key);
        if (it == storage_cache.end()) {
            return;
        }
        Storage& storage = *it->second;
        if (reason != Reason::Aged && storage_cache_bytes <= budget_target) {
            stats.storages_pressure_evicted.fetch_add(1, std::memory_order_relaxed);
        }
        storage_cache_bytes -= (std::min)(storage_cache_bytes, storage.vram_bytes);
        if (const auto index = storage_index.find(storage.pool_serial); index != storage_index.end()) {
            auto range = index->second.spans.equal_range(storage.pool_offset);
            for (auto s = range.first; s != range.second; ++s) {
                if (s->second == &storage) {
                    index->second.spans.erase(s);
                    break;
                }
            }
        }
        storage.cached = false;
        stats.storages_evicted.fetch_add(1, std::memory_order_relaxed);
        if (reason == Reason::Superseded) {
            stats.storages_superseded_evicted.fetch_add(1, std::memory_order_relaxed);
        } else if (reason == Reason::Aged) {
            stats.storages_aged_evicted.fetch_add(1, std::memory_order_relaxed);
        }
        storage_cache.erase(it);
    };
    size_t tests = 0;
    // Whether a later GPU write replaced every byte of a GPU-written storage (null: the pass is
    // out of tests, or nothing was written in its pool since it was found to be live).
    const auto superseded = [&](Storage& storage, u64 gap = 0) -> std::optional<bool> {
        const auto index = storage_index.find(storage.pool_serial);
        const u64 gen = index != storage_index.end() ? index->second.write_gen : 0;
        if (storage.trim_checked_gen == gen || tests >= MAX_SUPERSEDED_TESTS) {
            return std::nullopt;
        }
        ++tests;
        if (StorageSuperseded(storage, gap)) {
            return true;
        }
        storage.trim_checked_gen = gen;
        return false;
    };
    // Below the budget (the device budget's share) the storages of the last presents stay: the
    // transient targets a game re-creates every frame would otherwise be made again every frame.
    constexpr u64 PRESSURE_KEEP_PRESENTS = 4;
    const auto kept_under_budget = [&](const Storage& storage) {
        return storage_cache_bytes <= budget_target && presents >= storage.last_used_present &&
               presents - storage.last_used_present < PRESSURE_KEEP_PRESENTS;
    };
    // First the render targets the GPU wrote again since (the game re-uses the bytes for other
    // shapes; each shape keeps an image of its own): their texels live in the newer storages, so
    // they cost nothing but the image. Storages the GPU wrote that nothing replaced hold data no
    // guest bytes can rebuild: they are kept.
    std::sort(gpu_written.begin(), gpu_written.end(), by_age);
    for (const Candidate& candidate : gpu_written) {
        if (storage_cache_bytes <= target || tests >= MAX_SUPERSEDED_TESTS) {
            break;
        }
        const auto it = storage_cache.find(candidate.key);
        if (it != storage_cache.end() && !kept_under_budget(*it->second) &&
            superseded(*it->second).value_or(false)) {
            evict(candidate.key, Reason::Superseded);
        }
    }
    // Then the storages that hold guest bytes (the next use uploads them again).
    std::sort(from_guest.begin(), from_guest.end(), by_age);
    for (const Candidate& candidate : from_guest) {
        if (storage_cache_bytes <= target) {
            break;
        }
        const auto it = storage_cache.find(candidate.key);
        if (it != storage_cache.end() && !kept_under_budget(*it->second)) {
            evict(candidate.key, Reason::Budget);
        }
    }
    // Under the budget too: render targets nothing used for `trim_age_presents` that cost nothing
    // to rebuild. Ones that hold guest bytes (or nothing) upload them again; GPU-written ones go
    // only when superseded, never with texels no other storage holds.
    if (!trim_legacy && trim_age_presents != 0) {
        const auto aged = [&](const Storage& storage) {
            return presents >= storage.last_used_present &&
                   presents - storage.last_used_present >= trim_age_presents;
        };
        constexpr VkImageUsageFlags TARGET_USAGE =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        for (const Candidate& candidate : from_guest) {
            const auto it = storage_cache.find(candidate.key);
            if (it != storage_cache.end() && (it->second->usage & TARGET_USAGE) != 0 &&
                aged(*it->second)) {
                evict(candidate.key, Reason::Aged);
            }
        }
        for (const Candidate& candidate : gpu_written) {
            if (tests >= MAX_SUPERSEDED_TESTS) {
                break;
            }
            const auto it = storage_cache.find(candidate.key);
            if (it != storage_cache.end() && aged(*it->second) &&
                superseded(*it->second).value_or(false)) {
                stats.storages_superseded_evicted.fetch_add(1, std::memory_order_relaxed);
                evict(candidate.key, Reason::Aged);
            }
        }
    }
    // n2den: superseded GPU-written storages unused for EDEN_NVN_RELEASE_SUPERSEDED_PRESENTS go
    // under the budget too. The idle ones leave the cache here; the ones live textures hold give
    // their image back (ReleaseSupersededImages). The device shares its memory with the pools'
    // mirrors, so waiting for the cache budget leaves gigabytes of replaced targets in it.
    if (!trim_legacy && at_present) {
        if (const u64 release_age = ReleaseSupersededAge(); release_age != 0) {
            for (const Candidate& candidate : gpu_written) {
                if (tests >= MAX_SUPERSEDED_TESTS) {
                    break;
                }
                const auto it = storage_cache.find(candidate.key);
                if (it == storage_cache.end()) {
                    continue;
                }
                Storage& storage = *it->second;
                if (storage.scaled || presents < storage.last_used_present ||
                    presents - storage.last_used_present < release_age) {
                    continue;
                }
                if (superseded(storage, SupersededGap(storage)).value_or(false)) {
                    stats.storages_superseded_evicted.fetch_add(1, std::memory_order_relaxed);
                    evict(candidate.key, Reason::Aged);
                }
            }
            ReleaseSupersededImages(presents);
        }
    }
    trim_floor_bytes = storage_cache_bytes + (std::max)(storage_cache_budget / 32, u64{1});
    trim_next_present = presents + PASS_INTERVAL_PRESENTS;
    stats.storage_cache_bytes.store(storage_cache_bytes, std::memory_order_relaxed);
}

std::unordered_map<const Storage*, VC::TextureHolder> Context::Impl::StorageHolders() const {
    std::unordered_map<const Storage*, VC::TextureHolder> holders;
    const auto holder_of = [&](u64 serial) {
        const auto it = texture_holders.find(serial);
        return it != texture_holders.end() ? it->second : VC::TextureHolder::Game;
    };
    const auto note = [&](const Storage* storage, VC::TextureHolder holder) {
        const auto [it, inserted] = holders.try_emplace(storage, holder);
        if (!inserted && holder > it->second) {
            it->second = holder;
        }
    };
    for (const auto& [serial, texture] : textures) {
        if (texture && texture->storage) {
            note(texture->storage.get(), holder_of(serial));
        }
    }
    // A render-target alias goes with the texture it was made for.
    for (const auto& [key, texture] : target_aliases) {
        if (texture && texture->storage) {
            note(texture->storage.get(), holder_of(std::get<0>(key)));
        }
    }
    return holders;
}

void Context::Impl::NoteTextureHoldersNow(VC::TextureHolderReport& report) {
    texture_holders.clear();
    texture_holders.reserve(report.serials.size());
    for (const auto& [serial, holder] : report.serials) {
        texture_holders.emplace(serial, holder);
    }
    texture_holders_present = stats.presents.load(std::memory_order_relaxed);
    std::array<u64, VC::TEXTURE_HOLDER_KINDS> count{};
    std::array<u64, VC::TEXTURE_HOLDER_KINDS> bytes{};
    for (const auto& [storage, holder] : StorageHolders()) {
        if (!storage->image) {
            continue;
        }
        const size_t i = static_cast<size_t>(holder);
        ++count[i];
        bytes[i] += storage->vram_bytes;
        if (storage->scaled && storage->scaled->image) {
            bytes[i] += storage->scaled->vram_bytes;
        }
    }
    for (size_t i = 0; i < VC::TEXTURE_HOLDER_KINDS; ++i) {
        holder_storages[i].store(count[i], std::memory_order_relaxed);
        holder_storage_bytes[i].store(bytes[i], std::memory_order_relaxed);
    }
    holder_report_seen.store(true, std::memory_order_release);
}

void Context::Impl::LogStorageCensus() {
    std::unordered_set<const Storage*> seen;
    struct Kind {
        u64 count{};
        u64 bytes{};
    };
    Kind sparse, targets, depth, astc, other, gpu_written;
    std::array<Kind, ASTC_STANDIN_COUNT> astc_bc{}; // by AstcStandin
    u64 sparse_bound = 0;
    std::vector<const Storage*> largest;
    const auto add = [&](const Storage& s) {
        if (!s.image || !seen.insert(&s).second) {
            return;
        }
        Kind& kind = s.sparse ? sparse
                     : s.host.depth ? depth
                     : (s.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0 ? targets
                     : s.host.needs_transcode ? astc
                                              : other;
        ++kind.count;
        kind.bytes += s.vram_bytes;
        if (s.host.bc_standin) {
            Kind& bc = astc_bc[static_cast<size_t>(BcStandinOf(s.host))];
            ++bc.count;
            bc.bytes += s.vram_bytes;
        }
        if (s.sparse) {
            sparse_bound += s.sparse->BoundBytes();
        }
        if (s.origin == StorageOrigin::Gpu) {
            ++gpu_written.count;
            gpu_written.bytes += s.vram_bytes;
        }
        largest.push_back(&s);
    };
    for (const auto& [serial, texture] : textures) {
        if (texture && texture->storage) {
            add(*texture->storage);
        }
    }
    for (const auto& [key, storage] : storage_cache) {
        add(*storage);
    }
    const size_t top = (std::min)(largest.size(), size_t{16});
    std::partial_sort(largest.begin(), largest.begin() + top, largest.end(),
                      [](const Storage* a, const Storage* b) { return a->vram_bytes > b->vram_bytes; });
    const auto mib = [](u64 bytes) { return bytes >> 20; };
    // The BC stand-ins by format ("as BC3 <MiB> (<n>)", one entry per format held), else what
    // new ones become.
    std::string as_bc;
    for (const AstcStandin bc : {AstcStandin::Bc1, AstcStandin::Bc3, AstcStandin::Bc4,
                                 AstcStandin::Bc5, AstcStandin::Bc7}) {
        const Kind& held = astc_bc[static_cast<size_t>(bc)];
        if (held.count != 0) {
            as_bc += fmt::format("{}as {} {} ({})", as_bc.empty() ? "" : ", ", AstcStandinName(bc),
                                 mib(held.bytes), held.count);
        }
    }
    if (as_bc.empty()) {
        as_bc = fmt::format("as {} 0 (0)", AstcStandinName(astc_standin));
    }
    LOG_INFO(Render_Vulkan,
             "[nvn] storage census: {} images, estimated MiB (count): sparse {} ({}, bound {}), depth {} "
             "({}), colour targets {} ({}), compressed stand-ins {} ({}; {}), other {} "
             "({}); GPU-written {} ({}); cache {} MiB; sparse pages bound/reserved {}/{} MiB",
             seen.size(), mib(sparse.bytes), sparse.count, mib(sparse_bound), mib(depth.bytes),
             depth.count, mib(targets.bytes), targets.count, mib(astc.bytes), astc.count, as_bc,
             mib(other.bytes), other.count, mib(gpu_written.bytes), gpu_written.count,
             mib(storage_cache_bytes), sparse_pages ? mib(sparse_pages->UsedBytes()) : 0,
             sparse_pages ? mib(sparse_pages->ReservedBytes()) : 0);
    if (smart_bc.enabled) {
        const auto total = [](const std::atomic<u64>& value) {
            return value.load(std::memory_order_relaxed);
        };
        LOG_INFO(Render_Vulkan,
                 "[nvn] smart BC so far: {} scans (L {}, LA {}, RGB {}, RGBA {}, all zero {}), "
                 "{:.1f} MiB of ASTC in {:.1f} ms; {} re-classified to BC3, {} kept at BC3 by "
                 "copies",
                 total(smart_total.classes[0]) + total(smart_total.classes[1]) +
                     total(smart_total.classes[2]) + total(smart_total.classes[3]) +
                     total(smart_total.classes[4]),
                 total(smart_total.classes[0]), total(smart_total.classes[1]),
                 total(smart_total.classes[2]), total(smart_total.classes[3]),
                 total(smart_total.classes[4]),
                 static_cast<double>(total(smart_total.scan_bytes)) / 1048576.0,
                 static_cast<double>(total(smart_total.scan_ns)) / 1e6, total(smart_total.refits),
                 total(smart_total.pinned));
    }
    {
        // The GPU-written images only the cache holds (no texture, no submission in flight), and
        // how many of the first ones a later GPU write replaced (TrimStorageCache drops those
        // first when the cache is over budget).
        constexpr u64 TESTED_LIMIT = 1024;
        u64 idle = 0, idle_bytes = 0, tested = 0, superseded = 0, superseded_bytes = 0;
        for (const auto& [key, storage] : storage_cache) {
            if (storage.use_count() != 1 || storage->origin != StorageOrigin::Gpu ||
                !storage->image || storage->sparse) {
                continue;
            }
            ++idle;
            idle_bytes += storage->vram_bytes;
            if (tested < TESTED_LIMIT) {
                ++tested;
                if (StorageSuperseded(*storage)) {
                    ++superseded;
                    superseded_bytes += storage->vram_bytes;
                }
            }
        }
        size_t most_spans = 0;
        u64 widest = 0;
        for (const auto& [serial, index] : storage_index) {
            most_spans = (std::max)(most_spans, index.spans.size());
            widest = (std::max)(widest, index.max_size);
        }
        LOG_INFO(Render_Vulkan,
                 "[nvn] storage census: idle GPU-written {} ({} MiB), of the first {} tested {} "
                 "superseded ({} MiB); budget {} MiB, evicted {} ({} superseded); most storages in "
                 "one pool {}, widest {} MiB",
                 idle, mib(idle_bytes), tested, superseded, mib(superseded_bytes),
                 mib(storage_cache_budget), stats.storages_evicted.load(std::memory_order_relaxed),
                 stats.storages_superseded_evicted.load(std::memory_order_relaxed), most_spans,
                 mib(widest));
        // The GPU-written images that textures (or anything else) still hold: how many nothing
        // used for the release age, and how many of those newer GPU writes replaced, exactly and
        // with slivers allowed (what ReleaseSupersededImages releases).
        const u64 release_age = ReleaseSupersededAge();
        const u64 report_age = release_age != 0 ? release_age : 60;
        const u64 now = stats.presents.load(std::memory_order_relaxed);
        Kind held, held_aged, held_exact, held_nearly, idle_nearly;
        u64 held_tested = 0;
        u64 idle_tested = 0;
        for (const auto& [key, storage] : storage_cache) {
            const Storage& s = *storage;
            if (s.origin != StorageOrigin::Gpu || !s.image || s.sparse) {
                continue;
            }
            if (storage.use_count() == 1) {
                if (idle_tested < TESTED_LIMIT) {
                    ++idle_tested;
                    if (StorageSuperseded(s, SupersededGap(s))) {
                        ++idle_nearly.count;
                        idle_nearly.bytes += s.vram_bytes;
                    }
                }
                continue;
            }
            ++held.count;
            held.bytes += s.vram_bytes;
            if (now < s.last_used_present || now - s.last_used_present < report_age) {
                continue;
            }
            ++held_aged.count;
            held_aged.bytes += s.vram_bytes;
            if (held_tested >= TESTED_LIMIT) {
                continue;
            }
            ++held_tested;
            if (StorageSuperseded(s)) {
                ++held_exact.count;
                held_exact.bytes += s.vram_bytes;
            }
            if (StorageSuperseded(s, SupersededGap(s))) {
                ++held_nearly.count;
                held_nearly.bytes += s.vram_bytes;
            }
        }
        LOG_INFO(Render_Vulkan,
                 "[nvn] storage census: GPU-written images with holders {} ({} MiB); unused for "
                 "{} presents {} ({} MiB), of the first {} tested {} superseded ({} MiB), {} with "
                 "slivers allowed ({} MiB); idle superseded with slivers allowed {} ({} MiB); "
                 "images of held storages released so far {} ({} MiB), release age {} "
                 "(EDEN_NVN_RELEASE_SUPERSEDED_PRESENTS)",
                 held.count, mib(held.bytes), report_age, held_aged.count, mib(held_aged.bytes),
                 held_tested, held_exact.count, mib(held_exact.bytes), held_nearly.count,
                 mib(held_nearly.bytes), idle_nearly.count, mib(idle_nearly.bytes),
                 images_released, mib(images_released_bytes), release_age);
    }
    {
        // Why the cache cannot reach its budget: who holds the cached images (only the cache
        // holds the ones a trim may drop), where their bytes came from, how long ago the render
        // targets were used, and the twins of the Resolution setting (not in the cache's count).
        std::unordered_set<const Storage*> live;
        for (const auto& [serial, texture] : textures) {
            if (texture && texture->storage) {
                live.insert(texture->storage.get());
            }
        }
        for (const auto& [key, texture] : target_aliases) {
            if (texture && texture->storage) {
                live.insert(texture->storage.get());
            }
        }
        Kind held_live, held_other, idle_gpu, idle_rebuild, twins;
        std::array<Kind, 3> origin{};
        // Presents since the last use: < 60, < 300, < 1800, older.
        constexpr std::array<u64, 3> AGE_LIMITS{60, 300, 1800};
        std::array<Kind, 4> target_age{};
        std::array<Kind, 4> live_target_age{};
        std::array<Kind, 4> idle_gpu_target_age{};
        // The render targets textures hold, by the strongest holder of their storage (the
        // recorder's last report: the game's texture, one waiting for its submissions, one only
        // recorded lists keep, one the game re-initialised without a finalize).
        const std::unordered_map<const Storage*, VC::TextureHolder> holders = StorageHolders();
        std::array<std::array<Kind, 4>, VC::TEXTURE_HOLDER_KINDS> holder_target_age{};
        const u64 presents = stats.presents.load(std::memory_order_relaxed);
        const auto age_bucket = [&](const Storage& s) {
            const u64 age = presents > s.last_used_present ? presents - s.last_used_present : 0;
            size_t bucket = 0;
            while (bucket < AGE_LIMITS.size() && age >= AGE_LIMITS[bucket]) {
                ++bucket;
            }
            return bucket;
        };
        const auto count = [](Kind& kind, const Storage& s) {
            ++kind.count;
            kind.bytes += s.vram_bytes;
        };
        for (const Storage* s : seen) {
            if (s->scaled && s->scaled->image) {
                count(twins, *s->scaled);
            }
        }
        for (const auto& [key, storage] : storage_cache) {
            const Storage& s = *storage;
            if (!s.image || s.sparse) {
                continue;
            }
            count(origin[static_cast<size_t>(s.origin)], s);
            const bool idle = storage.use_count() == 1;
            if (live.contains(&s)) {
                count(held_live, s);
            } else if (!idle) {
                count(held_other, s);
            } else if (s.origin == StorageOrigin::Gpu) {
                count(idle_gpu, s);
            } else {
                count(idle_rebuild, s);
            }
            if ((s.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) == 0) {
                continue;
            }
            const size_t bucket = age_bucket(s);
            count(target_age[bucket], s);
            if (live.contains(&s)) {
                count(live_target_age[bucket], s);
                const auto holder = holders.find(&s);
                count(holder_target_age[static_cast<size_t>(
                          holder != holders.end() ? holder->second : VC::TextureHolder::Game)]
                                       [bucket],
                      s);
            } else if (idle && s.origin == StorageOrigin::Gpu) {
                count(idle_gpu_target_age[bucket], s);
            }
        }
        const auto ages = [&](const std::array<Kind, 4>& kinds) {
            return fmt::format("<60 {} ({}), <300 {} ({}), <1800 {} ({}), older {} ({})",
                               mib(kinds[0].bytes), kinds[0].count, mib(kinds[1].bytes),
                               kinds[1].count, mib(kinds[2].bytes), kinds[2].count,
                               mib(kinds[3].bytes), kinds[3].count);
        };
        LOG_INFO(Render_Vulkan,
                 "[nvn] storage census: cache holders MiB (count): live textures {} ({}), other "
                 "holders {} ({}), idle GPU-written {} ({}), idle rebuildable {} ({}); origin "
                 "none/cpu/gpu {}/{}/{} MiB; scaled twins {} ({}); {} live textures",
                 mib(held_live.bytes), held_live.count, mib(held_other.bytes), held_other.count,
                 mib(idle_gpu.bytes), idle_gpu.count, mib(idle_rebuild.bytes), idle_rebuild.count,
                 mib(origin[0].bytes), mib(origin[1].bytes), mib(origin[2].bytes),
                 mib(twins.bytes), twins.count, textures.size());
        LOG_INFO(Render_Vulkan,
                 "[nvn] storage census: render targets by presents since use MiB (count): {}; of "
                 "them held by live textures: {}; idle GPU-written: {}; trim {} age {} evicted "
                 "aged {} pressure {}, device over budget at the last pass {} MiB ({} passes)",
                 ages(target_age), ages(live_target_age), ages(idle_gpu_target_age),
                 trim_legacy ? "legacy" : "current", trim_age_presents,
                 stats.storages_aged_evicted.load(std::memory_order_relaxed),
                 stats.storages_pressure_evicted.load(std::memory_order_relaxed),
                 mib(trim_device_excess),
                 stats.storage_trim_pressure_passes.load(std::memory_order_relaxed));
        const auto by_holder = [&](VC::TextureHolder holder) {
            return ages(holder_target_age[static_cast<size_t>(holder)]);
        };
        LOG_INFO(Render_Vulkan,
                 "[nvn] storage census: render targets held by textures, by holder ({}), MiB "
                 "(count): game textures: {}; retiring: {}; guest lists: {}; synthetic lists: {}; "
                 "re-initialised: {}",
                 holder_report_seen.load(std::memory_order_relaxed)
                     ? fmt::format("recorder report of present {}", texture_holders_present)
                     : std::string{"no recorder report: all counted as game textures"},
                 by_holder(VC::TextureHolder::Game), by_holder(VC::TextureHolder::Retiring),
                 by_holder(VC::TextureHolder::GuestList),
                 by_holder(VC::TextureHolder::SyntheticList),
                 by_holder(VC::TextureHolder::Reinitialized));
    }
    for (size_t i = 0; i < top; ++i) {
        const Storage& s = *largest[i];
        LOG_INFO(Render_Vulkan,
                 "[nvn] storage census: #{} {}x{}x{} lv {} fmt {:#x} flags {:#x} {} MiB{}{}{}{} "
                 "pool {:#x} +{:#x} guest bytes {:#x}",
                 i + 1, s.extent.width, s.extent.height, s.layers, s.levels, s.desc.format,
                 s.desc.flags, mib(s.vram_bytes), s.sparse ? " sparse" : "",
                 s.host.needs_transcode ? " stand-in" : "",
                 s.host.bc_standin ? fmt::format(" {}", AstcStandinName(BcStandinOf(s.host)))
                                   : std::string{},
                 s.origin == StorageOrigin::Gpu ? " gpu-written" : "", s.pool_serial,
                 s.pool_offset, s.layout.size);
    }
    // The images by shape: which sizes and formats the count is made of, in how many pools and at
    // how many placements (pool offsets), and the guest bytes they stand for (a stand-in's image
    // against the game's compressed data; many placements of one target shape in few pools).
    using Shape = std::tuple<u32, u32, u32, u32, s32, bool>; // w, h, layers, levels, format, gpu
    struct ShapeUse {
        u64 count{};
        u64 bytes{};
        u64 guest_bytes{};
        u64 in_virtual{};
        std::set<u64> pools;
    };
    std::map<Shape, ShapeUse> shapes;
    for (const Storage* const storage : largest) {
        const Storage& s = *storage;
        ShapeUse& use = shapes[Shape{s.extent.width, s.extent.height, s.layers, s.levels,
                                     static_cast<s32>(s.desc.format),
                                     s.origin == StorageOrigin::Gpu}];
        ++use.count;
        use.bytes += s.vram_bytes;
        use.guest_bytes += s.layout.size;
        use.pools.insert(s.pool_serial);
        if (const auto pool = pools.find(s.pool_serial);
            pool != pools.end() && pool->second->IsVirtual()) {
            ++use.in_virtual;
        }
    }
    std::vector<const std::pair<const Shape, ShapeUse>*> by_bytes;
    by_bytes.reserve(shapes.size());
    for (const auto& entry : shapes) {
        by_bytes.push_back(&entry);
    }
    std::sort(by_bytes.begin(), by_bytes.end(),
              [](const auto* a, const auto* b) { return a->second.bytes > b->second.bytes; });
    for (size_t i = 0; i < (std::min)(by_bytes.size(), size_t{24}); ++i) {
        const Shape& shape = by_bytes[i]->first;
        const ShapeUse& use = by_bytes[i]->second;
        LOG_INFO(Render_Vulkan,
                 "[nvn] storage census: shape {}x{}x{} lv {} fmt {:#x}{}: {} images, {} MiB "
                 "(guest bytes {} MiB), {} pools, {} in virtual pools",
                 std::get<0>(shape), std::get<1>(shape), std::get<2>(shape), std::get<3>(shape),
                 std::get<4>(shape), std::get<5>(shape) ? " gpu-written" : "", use.count,
                 mib(use.bytes),
                 mib(use.guest_bytes), use.pools.size(), use.in_virtual);
    }
}

void Context::Impl::FlushTextureNow(CmdFlushTexture& command) {
    const auto finish = [&](bool ok) { command.done->set_value(ok); };
    const auto texture_it = textures.find(command.serial);
    if (texture_it == textures.end() || !texture_it->second->storage) {
        finish(false);
        return;
    }
    Storage& s = *texture_it->second->storage;
    // The game reads a texture the Resolution setting scaled: the guest-sized image first.
    if (s.origin == StorageOrigin::Gpu && Rescale::NativeIsStale(s.scale_state) &&
        !DownscaleNow(s)) {
        finish(false);
        return;
    }
    if (s.origin != StorageOrigin::Gpu || !s.image) {
        finish(true); // the guest bytes are what the texture holds
        return;
    }
    const auto pool_it = pools.find(s.pool_serial);
    if (pool_it == pools.end() || (pool_it->second->flags & PoolFlag::CpuNoAccess) != 0 ||
        pool_it->second->IsVirtual()) {
        finish(false);
        return;
    }
    Pool& pool = *pool_it->second;
    const bool plain = !s.host.needs_transcode && !s.host.guest_layout_differs &&
                       s.samples == VK_SAMPLE_COUNT_1_BIT && !s.host.depth && !s.host.stencil;
    const BlockInfo block = FormatBlockInfo(s.desc.format);
    if (!plain || !block.bytes || s.current_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        static Common::BootCounter logged;
        if (logged.First()) {
            LOG_WARNING(Render_Vulkan,
                        "[nvn] ReadTexels of a GPU-written texture (format {:#x}): readback of "
                        "depth/stencil, multisampled or compressed storage is not supported",
                        s.desc.format);
        }
        finish(false);
        return;
    }
    // One staging buffer for every layer and level, copied out in one submission.
    const bool linear = IsLinearStorage(s.desc);
    struct Piece {
        u32 level;
        u32 layer;
        u32 cols, rows, slices;
        u32 width, height;
        u64 buffer_offset;
        u64 guest_offset;
        u32 tile_h, tile_d;
    };
    std::vector<Piece> pieces;
    u64 total = 0;
    const u32 layer_count = linear ? 1 : s.layers;
    const u32 level_count = linear ? 1 : s.levels;
    for (u32 layer = 0; layer < layer_count; ++layer) {
        for (u32 level = 0; level < level_count; ++level) {
            Piece piece{};
            piece.level = level;
            piece.layer = layer;
            piece.width = Mip(s.extent.width, level);
            piece.height = Mip(s.extent.height, level);
            piece.slices = s.type == VK_IMAGE_TYPE_3D ? Mip(s.extent.depth, level) : 1u;
            piece.cols = (piece.width + block.width - 1) / block.width;
            piece.rows = (piece.height + block.height - 1) / block.height;
            piece.guest_offset = static_cast<u64>(layer) * s.layout.layer_size +
                                 (linear ? 0 : TextureLevelOffset(s.desc, level));
            TextureLevelTiles(s.desc, level, piece.tile_h, piece.tile_d);
            total = (total + 63) / 64 * 64;
            piece.buffer_offset = total;
            total += static_cast<u64>(piece.cols) * piece.rows * piece.slices * block.bytes;
            pieces.push_back(piece);
        }
    }
    vk::Buffer download;
    try {
        download = CreateHostBuffer(total, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryUsage::Download);
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "[nvn] ReadTexels readback buffer failed: {}", exception.what());
        finish(false);
        return;
    }
    const VkImageLayout previous_layout = s.current_layout;
    try {
        ImmediateSubmit([&](vk::CommandBuffer cmd) {
            const auto barrier = [&](VkImageLayout from, VkImageLayout to) {
                const VkImageMemoryBarrier image_barrier{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                    .oldLayout = from,
                    .newLayout = to,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = *s.image,
                    .subresourceRange{
                        .aspectMask = s.aspect,
                        .baseMipLevel = 0,
                        .levelCount = VK_REMAINING_MIP_LEVELS,
                        .baseArrayLayer = 0,
                        .layerCount = VK_REMAINING_ARRAY_LAYERS,
                    },
                };
                const VkMemoryBarrier memory{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                    .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                };
                cmd.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, memory, {},
                                    image_barrier);
            };
            barrier(previous_layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            std::vector<VkBufferImageCopy> regions;
            for (const Piece& piece : pieces) {
                regions.push_back({
                    .bufferOffset = piece.buffer_offset,
                    .bufferRowLength = 0,
                    .bufferImageHeight = 0,
                    .imageSubresource{
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .mipLevel = piece.level,
                        .baseArrayLayer = piece.layer,
                        .layerCount = 1,
                    },
                    .imageOffset = {0, 0, 0},
                    .imageExtent = {piece.width, piece.height, piece.slices},
                });
            }
            cmd.CopyImageToBuffer(*s.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, *download,
                                  regions);
            barrier(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, previous_layout);
            const VkMemoryBarrier host{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
            };
            cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                                host, {}, {});
        });
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "[nvn] ReadTexels readback failed: {}", exception.what());
        finish(false);
        return;
    }
    // Swizzle into guest memory (the whole storage is read-modify-written: GOB padding stays). A
    // storage that leaves its pool (the driver does not check) only has its inside written.
    if (s.pool_offset >= pool.size) {
        finish(false);
        return;
    }
    const u64 storage_size = (std::min)(s.layout.size, pool.size - s.pool_offset);
    VC::GuestMemory* memory = guest.load(std::memory_order_acquire);
    std::vector<u8> scratch;
    VC::PreRec::NoteRead(VC::PreRec::Cls::TexReadback, pool.address + s.pool_offset, storage_size,
                         pool.serial);
    u8* storage_bytes = pool.HostRead(VC::PreRec::Cls::TexReadback, s.pool_offset, storage_size);
    if (!storage_bytes) {
        scratch.resize(static_cast<size_t>(storage_size));
        if (!memory || !memory->Read(pool.address + s.pool_offset, scratch.data(), scratch.size())) {
            finish(false);
            return;
        }
        storage_bytes = scratch.data();
    }
    const std::span<const u8> mapped = download.Mapped();
    for (const Piece& piece : pieces) {
        const u64 linear_size = static_cast<u64>(piece.cols) * piece.rows * piece.slices * block.bytes;
        if (piece.guest_offset >= storage_size || piece.buffer_offset + linear_size > mapped.size()) {
            continue;
        }
        const std::span<const u8> source{mapped.data() + piece.buffer_offset,
                                         static_cast<size_t>(linear_size)};
        if (linear) {
            const u64 pitch = s.layout.layer_size / (std::max)(piece.height, 1u);
            const u64 row_bytes = static_cast<u64>(piece.cols) * block.bytes;
            for (u32 row = 0; row < piece.rows; ++row) {
                const u64 destination = piece.guest_offset + row * pitch;
                if (destination + row_bytes > storage_size) {
                    break;
                }
                std::memcpy(storage_bytes + destination, source.data() + row * row_bytes,
                            static_cast<size_t>(row_bytes));
            }
        } else {
            const u64 needed = Tegra::Texture::CalculateSize(true, block.bytes, piece.cols,
                                                             piece.rows, piece.slices,
                                                             piece.tile_h, piece.tile_d);
            const u64 available = storage_size - piece.guest_offset;
            std::vector<u8> padded;
            u8* target = storage_bytes + piece.guest_offset;
            if (available < needed) {
                padded.assign(static_cast<size_t>(needed), 0);
                std::memcpy(padded.data(), target, static_cast<size_t>(available));
                target = padded.data();
            }
            Tegra::Texture::SwizzleTexture(std::span<u8>{target, static_cast<size_t>(needed)},
                                           source, block.bytes, piece.cols, piece.rows,
                                           piece.slices, piece.tile_h, piece.tile_d, 0, true);
            if (!padded.empty()) {
                std::memcpy(storage_bytes + piece.guest_offset, padded.data(),
                            static_cast<size_t>(available));
            }
        }
    }
    if (!scratch.empty() && memory) {
        memory->Write(pool.address + s.pool_offset, scratch.data(), scratch.size());
    }
    // An untracked write: gated storages over it are disarmed (EDEN_NVN_TEX_PREHASH), a mirror
    // pool hashes these pages again at their next use.
    if (tex_prehash) {
        TexGateDisarm(pool, s.pool_offset, s.pool_offset + storage_size);
    }
    if (!pool.mirror_pages.empty()) {
        const u64 end = s.pool_offset + storage_size;
        for (u64 page = s.pool_offset / Pool::MIRROR_PAGE; page * Pool::MIRROR_PAGE < end; ++page) {
            pool.mirror_pages[static_cast<size_t>(page)].trusted = false;
        }
    }
    // The guest bytes are the texture's contents now: from here on it is a CPU-sourced storage.
    s.origin = StorageOrigin::Cpu;
    s.guest_hash = HashGuestBytes(storage_bytes, storage_size);
    s.sample_hash = SampleGuestBytes(storage_bytes, storage_size, mario_kart_world);
    s.unchanged_checks = 0;
    s.next_check_tick = 0;
    s.next_full_tick = PollTick() + FULL_VERIFY_PRESENTS;
    stats.texels_flushed.fetch_add(1, std::memory_order_relaxed);
    finish(true);
}


namespace {

/// The storage may be gated at all (nvn_tex_prehash.h (e)).
bool TexGateEligible(const Storage& s, const Pool& pool) {
    return s.origin == StorageOrigin::Cpu && !s.dynamic && !s.sparse && s.cached &&
           !s.tex_gate_off && !s.force_upload && !pool.IsVirtual() && s.scale_owner == nullptr;
}

} // Anonymous namespace

bool Context::Impl::TexGateDecide(Storage& s, Pool& pool, u64 size, bool full, u64 tick,
                                  TexPrehash::Record& record) {
    TexPrehash& gate = *tex_prehash;
    TexPrehash::Counts& w = gate.window;
    ++w.checks;
    {
        const u64 hint = gate.TickHint();
        ++w.tick_offset[tick == hint ? 0 : tick == hint + 1 ? 1 : 2];
    }
    if (!TexGateEligible(s, pool) || pool.tex_track == 0 || !pool.tex_write_epoch) {
        return false;
    }
    ++w.eligible;
    const std::shared_ptr<TexPrehashSlot>& slot = s.tex_gate.slot;
    if (!slot || slot->arm_gen.load(std::memory_order_relaxed) == 0) {
        ++w.not_armed;
        return false;
    }
    gate.NoteCheck(slot, tick);
    VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    if (memory == nullptr || memory->Stopped()) [[unlikely]] {
        ++w.stopped;
        return false;
    }
    // (a) No guest CPU write since the arm: the registration's write epoch has not moved since
    // it was read before the arm (every page turning dirty moves it), else the pages are asked.
    // A clean answer refreshes the epoch with the value read before the question: a page
    // dirtied after that read moves the epoch past it.
    const u64 epoch_now = pool.tex_write_epoch->load(std::memory_order_acquire);
    if (epoch_now != s.tex_arm_epoch) {
        ++w.any_written_calls;
        if (memory->AnyWritten(pool.tex_track, pool.address + s.pool_offset, size)) {
            ++w.dirty;
            return false;
        }
        s.tex_arm_epoch = epoch_now;
    }
    if (!TexPrehash::ReadRecord(*slot, record)) {
        ++w.no_record;
        return false;
    }
    // The worker read the bytes after this arm (it read arm_gen first).
    if (record.arm_gen != slot->arm_gen.load(std::memory_order_relaxed)) {
        ++w.arm_mismatch;
        return false;
    }
    // (c) No raw-storing submission completed after the worker's read.
    if (record.epoch != split_raw_epoch.load(std::memory_order_acquire)) {
        ++w.epoch_mismatch;
        return false;
    }
    // (b) Every recorded GPU write over the storage had completed before the worker's read.
    if (record.block_ok < slot->block_seq.load(std::memory_order_relaxed)) {
        ++w.blocked;
        return false;
    }
    if (full && !record.has_full) {
        ++w.no_full;
        return false;
    }
    if (record.signature != s.sample_hash || (full && record.full != s.guest_hash)) {
        ++w.said_changed;
        return false;
    }
    ++w.skips;
    if (full) {
        ++w.skip_full;
    }
    return true;
}

void Context::Impl::TexGateArm(Storage& s, Pool& pool, u64 size, u64 tick) {
    if (!TexGateEligible(s, pool)) {
        return;
    }
    TexPrehash& gate = *tex_prehash;
    TexPrehash::Counts& w = gate.window;
    VC::GuestMemory* const memory = guest.load(std::memory_order_acquire);
    if (memory == nullptr || memory->Stopped()) {
        return;
    }
    if (pool.tex_track == 0 && !pool.tex_track_tried) {
        pool.tex_track_tried = true;
        pool.tex_track = memory->Track(pool.address, pool.size);
        if (pool.tex_track != 0) {
            pool.tex_write_epoch = memory->WriteEpoch(pool.tex_track);
            if (!pool.tex_write_epoch) {
                memory->Untrack(pool.tex_track);
                pool.tex_track = 0;
            } else {
                pool.tex_faults_counted = pool.tex_write_epoch->load(std::memory_order_relaxed);
            }
        }
    }
    if (pool.tex_track == 0) {
        return;
    }
    const u64 address = pool.address + s.pool_offset;
    std::shared_ptr<TexPrehashSlot>& slot = s.tex_gate.slot;
    if (slot && slot->arm_gen.load(std::memory_order_relaxed) != 0) {
        // Armed: arm again only after a write.
        const u64 epoch_now = pool.tex_write_epoch->load(std::memory_order_acquire);
        if (epoch_now == s.tex_arm_epoch) {
            return;
        }
        if (!memory->AnyWritten(pool.tex_track, address, size)) {
            s.tex_arm_epoch = epoch_now;
            return;
        }
        ++w.rearms_after_write;
        if (++s.tex_write_rearms >= TexPrehash::MAX_WRITE_REARMS) {
            // The CPU keeps writing it: no more faults for it.
            s.tex_gate_off = true;
            slot->arm_gen.store(0, std::memory_order_release);
            ++w.gave_up;
            return;
        }
    }
    if (!slot) {
        slot = std::make_shared<TexPrehashSlot>(address, size, mario_kart_world);
        constexpr u64 SMALL_STORAGE = u64{4} << 20; // as DetectGuestChange
        const bool compressed = s.desc.format != 0 && FormatBlockInfo(s.desc.format).width > 1;
        slot->every_check.store(size <= SMALL_STORAGE && !compressed, std::memory_order_relaxed);
    }
    // Arm (clean + write-armed) the pages, the epoch read first. Arming may clean a dirty page
    // another gated storage over the same page relies on: unless nothing in the range was dirty
    // and nothing turned dirty meanwhile, those storages are disarmed.
    const u64 epoch_before = pool.tex_write_epoch->load(std::memory_order_acquire);
    const bool was_dirty = memory->AnyWritten(pool.tex_track, address, size);
    memory->Arm(pool.tex_track, address, size);
    const u64 epoch_after = pool.tex_write_epoch->load(std::memory_order_acquire);
    s.tex_arm_epoch = epoch_before;
    if (was_dirty || epoch_after != epoch_before) {
        constexpr u64 PAGE = 4096;
        const u64 page_begin = (address & ~(PAGE - 1)) > pool.address
                                   ? (address & ~(PAGE - 1)) - pool.address
                                   : 0;
        const u64 page_end = ((address + size + PAGE - 1) & ~(PAGE - 1)) - pool.address;
        const auto index_it = storage_index.find(pool.serial);
        if (index_it != storage_index.end()) {
            const PoolIndex& index = index_it->second;
            const u64 scan_from = page_begin > index.max_size ? page_begin - index.max_size : 0;
            for (auto it = index.spans.lower_bound(scan_from);
                 it != index.spans.end() && it->first < page_end; ++it) {
                Storage& other = *it->second;
                if (&other == &s || it->first + other.layout.size <= page_begin ||
                    !other.tex_gate.slot ||
                    other.tex_gate.slot->arm_gen.load(std::memory_order_relaxed) == 0) {
                    continue;
                }
                other.tex_gate.slot->arm_gen.store(0, std::memory_order_release);
                ++w.disarmed_by_overlap;
            }
        }
    }
    if (pool.tex_last_write_seq > slot->block_seq.load(std::memory_order_relaxed)) {
        slot->block_seq.store(pool.tex_last_write_seq, std::memory_order_release);
    }
    slot->next_check_tick.store(s.next_check_tick, std::memory_order_relaxed);
    slot->next_full_tick.store(s.next_full_tick, std::memory_order_relaxed);
    slot->arm_gen.store(gate.NextArmGen(), std::memory_order_release);
    ++w.arms;
    gate.NoteCheck(slot, tick);
}

void Context::Impl::TexGateNoteWrite(Pool& pool, u64 offset, u64 size, u64 seq) {
    if (seq > pool.tex_last_write_seq) {
        pool.tex_last_write_seq = seq;
    }
    if (pool.tex_track == 0) {
        return; // nothing armed in this pool yet: a later arm starts from tex_last_write_seq
    }
    const auto index_it = storage_index.find(pool.serial);
    if (index_it == storage_index.end()) {
        return;
    }
    const PoolIndex& index = index_it->second;
    const u64 end = offset + size;
    const u64 scan_from = offset > index.max_size ? offset - index.max_size : 0;
    for (auto it = index.spans.lower_bound(scan_from); it != index.spans.end() && it->first < end;
         ++it) {
        Storage& storage = *it->second;
        if (it->first + storage.layout.size <= offset || !storage.tex_gate.slot) {
            continue;
        }
        TexPrehashSlot& slot = *storage.tex_gate.slot;
        if (slot.block_seq.load(std::memory_order_relaxed) < seq) {
            slot.block_seq.store(seq, std::memory_order_release);
            ++tex_prehash->window.blocks_raised;
        }
    }
}

void Context::Impl::TexGateDisarmStorage(Storage& storage) {
    if (storage.tex_gate.slot &&
        storage.tex_gate.slot->arm_gen.load(std::memory_order_relaxed) != 0) {
        // A possibly untracked write (a host service's, the guest's WriteTexels): no skip before
        // today's check has armed it again.
        storage.tex_gate.slot->arm_gen.store(0, std::memory_order_release);
        if (tex_prehash) {
            ++tex_prehash->window.disarmed_by_host_write;
        }
    }
}

void Context::Impl::TexGateDisarm(Pool& pool, u64 begin, u64 end) {
    const auto index_it = storage_index.find(pool.serial);
    if (index_it == storage_index.end()) {
        return;
    }
    const PoolIndex& index = index_it->second;
    const u64 scan_from = begin > index.max_size ? begin - index.max_size : 0;
    for (auto it = index.spans.lower_bound(scan_from); it != index.spans.end() && it->first < end;
         ++it) {
        Storage& storage = *it->second;
        if (it->first + storage.layout.size <= begin || !storage.tex_gate.slot ||
            storage.tex_gate.slot->arm_gen.load(std::memory_order_relaxed) == 0) {
            continue;
        }
        storage.tex_gate.slot->arm_gen.store(0, std::memory_order_release);
        ++tex_prehash->window.disarmed_by_host_write;
    }
}

u64 Context::Impl::TexGateTakeFaults() {
    u64 faults = 0;
    for (auto& [serial, pool] : pools) {
        if (pool->tex_track != 0 && pool->tex_write_epoch) {
            const u64 now = pool->tex_write_epoch->load(std::memory_order_relaxed);
            faults += now - pool->tex_faults_counted;
            pool->tex_faults_counted = now;
        }
    }
    return faults;
}

} // namespace Vulkan::Nvn
