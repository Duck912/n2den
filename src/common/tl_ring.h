// SPDX-FileCopyrightText: Copyright 2026 n2den contributors
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include "common/common_types.h"

/// Frame timeline ring (EDEN_NX2_NVN_TLRING=<first present>:<last present>[:<path>]), measurement
/// only and off by default. The NVN-HLE game side, the replay thread, the completion thread and the
/// GPU profiler's per-command-buffer timestamps record events (rdtsc + two words) into one
/// preallocated array from the first present on; at the last present recording stops and a
/// background thread writes the array to <path> (default tlring.bin in the working directory).
/// No log line per event, no allocation per event.
/// Pipeline keys and shaders are untouched (the switch is read nowhere else).
namespace Common::TlRing {

enum Ev : u16 {
    // Game side (tid = guest thread id).
    GSubmitB = 1,   ///< nvnQueueSubmitCommands entered: a = list count
    GSubmitE = 2,   ///< ... returned: a = seq
    GFence = 3,     ///< nvnQueueFenceSync: a = seq
    GPresentB = 4,  ///< nvnQueuePresentTexture entered
    GPresentE = 5,  ///< ... recorder done: a = seq
    GPaceB = 6,     ///< speed limiter / pacing
    GPaceE = 7,
    GSyncWaitB = 10, ///< nvnSyncWait blocking: a = fence seq, b = last seq << 32 | completed
    GSyncWaitE = 11, ///< a = fence seq, b = completed at wake
    GBlockB = 12,    ///< Runtime::WaitUntil about to sleep: a = function name pointer
    GBlockE = 13,    ///< a = function name pointer, b = satisfied
    GWaitSync = 14,  ///< nvnQueueWaitSync: a = seq
    GPresentRet = 15, ///< nvnQueuePresentTexture returns
    // Recorder lock contention (any thread; tid = ring thread number).
    LockWait = 16, ///< sink_mutex not free: a = rdtsc at the attempt, b = source line
    // Replay thread.
    RIdleB = 20,     ///< waiting for a command: a = streams waiting
    RIdleE = 21,     ///< a = kind of the command taken (~0: none)
    RRecB = 22,      ///< submission recording (re)starts: a = seq, b = resumed
    RRecE = 23,      ///< submission finished: a = seq, b = draws so far
    RSuspend = 24,   ///< submission suspended on an unsatisfied event wait: a = seq
    RQSubB = 25,     ///< vkQueueSubmit path entered: a = seq, b = command buffers
    RQSubLocked = 26, ///< submit_mutex taken
    RQSubE = 27,     ///< vkQueueSubmit returned: a = seq, b = draws so far
    RPresB = 28,     ///< present: a = seq
    RPresAcq = 29,   ///< swapchain frame acquired
    RPresE = 30,     ///< present handed over
    RFlushWaitB = 31, ///< replay waits for the GPU (FlushAndWait): a = seq
    RFlushWaitE = 32,
    RDispB = 33,     ///< command dispatch: a = command kind
    RDispE = 34,
    RHandB = 35,     ///< present hand-over to the presenter
    RHandE = 36,
    // Completion thread.
    CWaitB = 40,  ///< about to wait for the flight's timeline value: a = seq
    CGpuDone = 41, ///< value reached: a = seq
    CCbB = 42,    ///< completion callback: a = seq
    CCbE = 43,
    CDone = 44,   ///< completed_seq published: a = seq
    CStep = 45,   ///< completion step done before the callback: a = seq, b = step (nvn_context.cpp)
    // GPU (written by the completion thread): tsc = rdtsc at the vkQueueSubmit that carried the
    // command buffer, a = GPU begin tick, b = GPU end tick, tid = seq, aux = queue.
    GpuSeg = 50,
    /// EDEN_NVN_REPLAY_PHASES + TLRING: replay phase ticks since the previous vkQueueSubmit
    /// (written at each submit): a = phase, b = rdtsc ticks, tid = seq.
    PhaseDelta = 51,
    Present = 60, ///< guest present count: a = count
};

struct Rec {
    u64 tsc;
    u64 a;
    u64 b;
    u32 tid;
    u16 ev;
    u16 aux;
};
static_assert(sizeof(Rec) == 32);

namespace Detail {
struct Config {
    u64 first{};
    u64 last{};
    std::string path{"tlring.bin"};
    bool on{};
};
inline Config Read() {
    Config c;
    const char* v = std::getenv("EDEN_NX2_NVN_TLRING");
    if (v == nullptr || *v == 0 || std::strcmp(v, "0") == 0) {
        return c;
    }
    unsigned long long a = 0, b = 0;
    int n = 0;
    if (std::sscanf(v, "%llu:%llu%n", &a, &b, &n) < 2 || b <= a) {
        return c;
    }
    c.first = a;
    c.last = b;
    if (v[n] == ':') {
        c.path = v + n + 1;
    }
    c.on = true;
    return c;
}
inline const Config config = Read();
/// 4M records (128 MiB) when on; nothing otherwise.
inline constexpr u64 Capacity = u64{1} << 22;
inline Rec* const buffer = config.on ? new Rec[Capacity] : nullptr;
inline std::atomic<u64> head{0};
inline std::atomic<bool> armed{false};
inline std::atomic<bool> dumped{false};
inline std::atomic<u32> next_thread{1};
inline std::atomic<u64> gpu_period_bits{0}; ///< double ns per GPU tick
inline u64 tsc0{};
inline s64 ns0{};
} // namespace Detail

[[nodiscard]] inline bool On() noexcept {
    return Detail::config.on;
}

[[nodiscard]] inline bool Armed() noexcept {
    return Detail::config.on && Detail::armed.load(std::memory_order_relaxed);
}

[[nodiscard]] inline u64 Now() noexcept {
    return __rdtsc();
}

[[nodiscard]] inline u32 ThreadNumber() noexcept {
    thread_local u32 number = Detail::next_thread.fetch_add(1, std::memory_order_relaxed);
    return number | 0x80000000u;
}

inline void Note(Ev ev, u64 a = 0, u64 b = 0, u32 tid = 0, u16 aux = 0, u64 tsc = 0) noexcept {
    if (!Armed()) [[likely]] {
        return;
    }
    const u64 index = Detail::head.fetch_add(1, std::memory_order_relaxed);
    if (index >= Detail::Capacity) {
        return;
    }
    Detail::buffer[index] =
        Rec{tsc != 0 ? tsc : Now(), a, b, tid != 0 ? tid : ThreadNumber(), ev, aux};
}

inline void SetGpuPeriod(double ns) noexcept {
    u64 bits;
    std::memcpy(&bits, &ns, sizeof(bits));
    Detail::gpu_period_bits.store(bits, std::memory_order_relaxed);
}

/// The guest present count (game thread, once per present): arms at `first`, stops and dumps at
/// `last` (a background thread writes the file; recording threads mid-write get 100 ms).
inline void OnPresent(u64 count) noexcept {
    if (!On()) [[likely]] {
        return;
    }
    if (count == Detail::config.first && !Detail::armed.load()) {
        Detail::tsc0 = Now();
        Detail::ns0 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();
        Detail::armed.store(true);
    }
    Note(Present, count);
    if (count == Detail::config.last && Detail::armed.exchange(false) &&
        !Detail::dumped.exchange(true)) {
        const u64 tsc1 = Now();
        const s64 ns1 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
        std::thread([tsc1, ns1] {
            std::this_thread::sleep_for(std::chrono::milliseconds{100});
            const u64 count_written =
                (std::min)(Detail::head.load(), Detail::Capacity);
            std::FILE* f = std::fopen(Detail::config.path.c_str(), "wb");
            if (f == nullptr) {
                return;
            }
            const u64 header[8] = {0x31474e4952544cull, count_written, Detail::tsc0,
                                   static_cast<u64>(Detail::ns0), tsc1, static_cast<u64>(ns1),
                                   Detail::gpu_period_bits.load(), Detail::head.load()};
            std::fwrite(header, sizeof(header), 1, f);
            std::fwrite(Detail::buffer, sizeof(Rec), count_written, f);
            // Names of the GBlock sites (function name pointers of string literals).
            std::vector<u64> seen;
            for (u64 i = 0; i < count_written; ++i) {
                const Rec& r = Detail::buffer[i];
                if ((r.ev == GBlockB) && std::find(seen.begin(), seen.end(), r.a) == seen.end()) {
                    seen.push_back(r.a);
                    std::fprintf(f, "%llu %s\n", static_cast<unsigned long long>(r.a),
                                 reinterpret_cast<const char*>(r.a));
                }
            }
            std::fclose(f);
        }).detach();
    }
}

/// std::lock_guard for the recorder's sink_mutex that records the wait when the lock is taken.
template <typename Mutex>
class TimedLock {
public:
    TimedLock(Mutex& m, u32 line) : mutex{m} {
        if (!Armed()) [[likely]] {
            mutex.lock();
            return;
        }
        if (mutex.try_lock()) {
            return;
        }
        const u64 begin = Now();
        mutex.lock();
        Note(LockWait, begin, line);
    }
    ~TimedLock() {
        mutex.unlock();
    }
    TimedLock(const TimedLock&) = delete;
    TimedLock& operator=(const TimedLock&) = delete;

private:
    Mutex& mutex;
};

} // namespace Common::TlRing
