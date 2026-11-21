// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 n2den Contributors
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <span>
#include <string_view>
#include <thread>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <xmmintrin.h>
#endif
#if defined(ARCHITECTURE_x86_64)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

#include "common/frame_stats.h"
#include "common/cityhash.h"
#include "common/logging.h"
#include "common/thread.h"
#include "common/uint128.h"
#include "video_core/renderer_vulkan/nvn/nvn_upload.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/decode_bc.h"
#include "video_core/textures/astc.h"
#include "video_core/textures/decoders.h"

namespace Vulkan::Nvn {

u64 GuestContentHash(const u8* data, u64 size); // (declared in nvn_impl.h)

namespace {

/// GuestContentHash: XXH3's long-input construction (64-byte stripes into eight 64-bit lanes,
/// acc[j ^ 1] += d, acc[j] += lo32(d ^ key) * hi32(d ^ key); a scramble after each 64 stripes; a
/// 128-bit multiply fold to merge; the XXH3 avalanche), with its own 96-word key from splitmix64.
/// Not XXH3-compatible and never needs to be: every value is only compared with another value of
/// the same function in the same process. Stripe n of a block uses keys[n .. n + 8), so equal
/// stripes at different offsets add differently (a moved or swapped stripe changes the hash).
namespace StripeHash {

constexpr u64 STRIPE = 64;
constexpr u64 BLOCK_STRIPES = 64;
constexpr u64 BLOCK = STRIPE * BLOCK_STRIPES;
constexpr size_t LAST_KEY = 72;     ///< keys of the last stripe
constexpr size_t SCRAMBLE_KEY = 80; ///< keys of the scramble after a block
constexpr size_t MERGE_KEY = 88;    ///< keys of the merge
constexpr u64 P32_1 = 0x9E3779B1ULL;
constexpr u64 P64_1 = 0x9E3779B185EBCA87ULL;

struct Keys {
    std::array<u64, 96> word{};
};

constexpr Keys MakeKeys() {
    Keys keys;
    u64 x = 0x4E564E2D48415348ULL;
    for (u64& word : keys.word) {
        x += 0x9E3779B97F4A7C15ULL;
        u64 z = x;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        word = z ^ (z >> 31);
    }
    return keys;
}

alignas(64) constexpr Keys KEYS = MakeKeys();

#if defined(ARCHITECTURE_x86_64)
/// The same as the SSE2 path of Accumulate, four lanes at a time (AVX2 shuffles and multiplies
/// within each 128-bit half, which is exactly two SSE2 lanes side by side).
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
inline void AccumulateAvx2(u64* acc, const u8* data, u64 count, const u64* key) {
    __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + 0));
    __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + 4));
    for (u64 n = 0; n < count; ++n, data += STRIPE, ++key) {
        const __m256i d0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data));
        const __m256i d1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + 32));
        const __m256i k0 =
            _mm256_xor_si256(d0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(key)));
        const __m256i k1 =
            _mm256_xor_si256(d1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(key + 4)));
        const __m256i p0 = _mm256_mul_epu32(k0, _mm256_shuffle_epi32(k0, _MM_SHUFFLE(0, 3, 0, 1)));
        const __m256i p1 = _mm256_mul_epu32(k1, _mm256_shuffle_epi32(k1, _MM_SHUFFLE(0, 3, 0, 1)));
        a0 = _mm256_add_epi64(
            a0, _mm256_add_epi64(p0, _mm256_shuffle_epi32(d0, _MM_SHUFFLE(1, 0, 3, 2))));
        a1 = _mm256_add_epi64(
            a1, _mm256_add_epi64(p1, _mm256_shuffle_epi32(d1, _MM_SHUFFLE(1, 0, 3, 2))));
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + 0), a0);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + 4), a1);
}

inline bool HasAvx2() {
    static const bool has = [] {
#if defined(_MSC_VER)
        int info[4];
        __cpuid(info, 0);
        if (info[0] < 7) {
            return false;
        }
        __cpuid(info, 1);
        const bool os_avx = (info[2] & (1 << 27)) != 0 && (info[2] & (1 << 28)) != 0 &&
                            (_xgetbv(0) & 6) == 6;
        __cpuidex(info, 7, 0);
        return os_avx && (info[1] & (1 << 5)) != 0;
#else
        return __builtin_cpu_supports("avx2");
#endif
    }();
    return has;
}
#endif

/// `count` consecutive stripes at `data`, the first one with the keys from `key` on.
inline void Accumulate(u64* acc, const u8* data, u64 count, const u64* key) {
#if defined(ARCHITECTURE_x86_64)
    if (count >= 4 && HasAvx2()) {
        AccumulateAvx2(acc, data, count, key);
        return;
    }
    __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(acc + 0));
    __m128i a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(acc + 2));
    __m128i a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(acc + 4));
    __m128i a3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(acc + 6));
    const auto lane = [](__m128i a, const u8* p, const u64* k) {
        const __m128i d = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        const __m128i dk = _mm_xor_si128(d, _mm_loadu_si128(reinterpret_cast<const __m128i*>(k)));
        const __m128i product = _mm_mul_epu32(dk, _mm_shuffle_epi32(dk, _MM_SHUFFLE(0, 3, 0, 1)));
        const __m128i swapped = _mm_shuffle_epi32(d, _MM_SHUFFLE(1, 0, 3, 2));
        return _mm_add_epi64(a, _mm_add_epi64(product, swapped));
    };
    for (u64 n = 0; n < count; ++n, data += STRIPE, ++key) {
        a0 = lane(a0, data + 0, key + 0);
        a1 = lane(a1, data + 16, key + 2);
        a2 = lane(a2, data + 32, key + 4);
        a3 = lane(a3, data + 48, key + 6);
    }
    _mm_storeu_si128(reinterpret_cast<__m128i*>(acc + 0), a0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(acc + 2), a1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(acc + 4), a2);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(acc + 6), a3);
#else
    for (u64 n = 0; n < count; ++n, data += STRIPE, ++key) {
        for (size_t j = 0; j < 8; ++j) {
            u64 d;
            std::memcpy(&d, data + j * 8, sizeof(d));
            const u64 dk = d ^ key[j];
            acc[j ^ 1] += d;
            acc[j] += (dk & 0xFFFFFFFFULL) * (dk >> 32);
        }
    }
#endif
}

inline void Scramble(u64* acc) {
    for (size_t j = 0; j < 8; ++j) {
        u64 a = acc[j];
        a ^= a >> 47;
        a ^= KEYS.word[SCRAMBLE_KEY + j];
        acc[j] = a * P32_1;
    }
}

inline u64 Fold(u64 a, u64 b) {
    const u128 product = Common::Multiply64Into128(a, b);
    return product[0] ^ product[1];
}

/// Takes the stripes of a `length`-byte input in order (Add for each one of its
/// StripeCount(length) stripes, then AddLast with its last 64 bytes, zero-padded when it is
/// shorter), so that a caller may hand over separate 64-byte pieces (sampled lines) and get the
/// hash of their concatenation.
class Hasher {
public:
    explicit Hasher(u64 length_)
        : length{length_}, stripes{StripeCount(length_)},
          scrambled_stripes{length_ == 0 ? 0 : (length_ - 1) / BLOCK * BLOCK_STRIPES} {}

    static constexpr u64 StripeCount(u64 length) {
        return length <= STRIPE ? 0 : (length - 1) / STRIPE;
    }

    /// `count` consecutive stripes (count may cross a block)
    void Add(const u8* data, u64 count) {
        while (count != 0) {
            const u64 in_block = taken % BLOCK_STRIPES;
            const u64 run = (std::min)(count, BLOCK_STRIPES - in_block);
            Accumulate(acc, data, run, KEYS.word.data() + in_block);
            taken += run;
            data += run * STRIPE;
            count -= run;
            if (taken % BLOCK_STRIPES == 0 && taken <= scrambled_stripes) {
                Scramble(acc);
            }
        }
    }

    [[nodiscard]] u64 AddLast(const u8* last) {
        Accumulate(acc, last, 1, KEYS.word.data() + LAST_KEY);
        u64 result = length * P64_1;
        for (size_t i = 0; i < 4; ++i) {
            result += Fold(acc[2 * i] ^ KEYS.word[MERGE_KEY + 2 * i],
                           acc[2 * i + 1] ^ KEYS.word[MERGE_KEY + 2 * i + 1]);
        }
        result ^= result >> 37;
        result *= 0x165667919E3779F9ULL;
        return result ^ (result >> 32);
    }

    [[nodiscard]] u64 Stripes() const {
        return stripes;
    }

private:
    alignas(16) u64 acc[8]{0xC2B2AE3DULL,         0x9E3779B185EBCA87ULL, 0xC2B2AE3D27D4EB4FULL,
                           0x165667B19E3779F9ULL, 0x85EBCA77C2B2AE63ULL, 0x85EBCA77ULL,
                           0x27D4EB2F165667C5ULL, 0x9E3779B1ULL};
    u64 length;
    u64 stripes;
    u64 scrambled_stripes;
    u64 taken{};
};

inline u64 Hash(const u8* data, u64 length) {
    Hasher hasher{length};
    hasher.Add(data, hasher.Stripes());
    if (length < STRIPE) {
        alignas(16) std::array<u8, STRIPE> last{};
        if (length != 0) {
            std::memcpy(last.data(), data, static_cast<size_t>(length));
        }
        return hasher.AddLast(last.data());
    }
    return hasher.AddLast(data + length - STRIPE);
}

} // namespace StripeHash

/// EDEN_NVN_PAGE_HASH=city: change detection hashes with CityHash64 again (A/B). Read once: a
/// value is only ever compared with one of the same function, so it never changes in a process.
bool CityGuestHash() {
    static const bool city = [] {
        const char* const value = std::getenv("EDEN_NVN_PAGE_HASH");
        const bool is_city = value != nullptr && std::string_view{value} == "city";
        LOG_INFO(Render_Vulkan, "[nvn] guest change detection hash: {}",
                 is_city ? "CityHash64 (EDEN_NVN_PAGE_HASH=city)" : "stripe (XXH3-style)");
        return is_city;
    }();
    return city;
}

using VideoCore::Surface::PixelFormat;

u32 Mip(u32 size, u32 level) {
    return (std::max)(size >> level, 1u);
}

u32 CeilDiv(u32 value, u32 divisor) {
    return (value + divisor - 1) / divisor;
}

VkImageAspectFlags AspectOf(const HostFormat& host) {
    if (host.depth) {
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    if (host.stencil) {
        return VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    return VK_IMAGE_ASPECT_COLOR_BIT;
}

/// A small persistent pool for ASTC / BCn decoding and big unswizzles. The replay thread takes part
/// in the work, so one decode never waits for a worker that is busy with another.
class DecodePool {
public:
    DecodePool() {
        const unsigned hardware = std::thread::hardware_concurrency();
        const unsigned count = std::clamp(hardware / 4, 2u, 6u);
        for (unsigned i = 0; i < count; ++i) {
            threads.emplace_back([this] {
                Common::SetCurrentThreadName("NvnDecode");
                Run();
            });
        }
    }

    ~DecodePool() {
        {
            std::scoped_lock lock{mutex};
            stopping = true;
        }
        condition.notify_all();
        for (std::thread& thread : threads) {
            thread.join();
        }
    }

    void Execute(size_t count, const std::function<void(size_t)>& fn) {
        if (count == 0) {
            return;
        }
        auto job = std::make_shared<Job>();
        job->fn = &fn;
        job->count = count;
        const size_t helpers = (std::min)(count - 1, threads.size());
        if (helpers > 0) {
            {
                std::scoped_lock lock{mutex};
                for (size_t i = 0; i < helpers; ++i) {
                    jobs.push_back(job);
                }
            }
            condition.notify_all();
        }
        Work(*job);
        // Every index was claimed; wait for the ones helpers are still working on.
        std::unique_lock lock{job->mutex};
        job->done_condition.wait(lock, [&] { return job->finished == job->count; });
    }

private:
    struct Job {
        const std::function<void(size_t)>* fn{};
        size_t count{};
        std::atomic<size_t> next{0};
        std::mutex mutex;
        std::condition_variable done_condition;
        size_t finished{0};
    };

    static void Work(Job& job) {
        for (;;) {
            const size_t index = job.next.fetch_add(1);
            if (index >= job.count) {
                return;
            }
            (*job.fn)(index);
            std::scoped_lock lock{job.mutex};
            if (++job.finished == job.count) {
                job.done_condition.notify_all();
            }
        }
    }

    void Run() {
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock lock{mutex};
                condition.wait(lock, [this] { return stopping || !jobs.empty(); });
                if (stopping) {
                    return;
                }
                job = std::move(jobs.front());
                jobs.pop_front();
            }
            Work(*job);
        }
    }

    std::mutex mutex;
    std::condition_variable condition;
    std::deque<std::shared_ptr<Job>> jobs;
    std::vector<std::thread> threads;
    bool stopping{false};
};

DecodePool& GetDecodePool() {
    static DecodePool pool;
    return pool;
}

/// A hint that `address` is read soon: the sampled lines of a large storage are each in another
/// page, so loading them one after the other waits for one cache (and TLB) miss at a time. Never
/// faults and changes no result.
inline void PrefetchLine(const u8* address) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_prefetch(reinterpret_cast<const char*>(address), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(address);
#else
    static_cast<void>(address);
#endif
}

} // Anonymous namespace

void RunParallel(size_t count, const std::function<void(size_t)>& fn) {
    if (count <= 1) {
        for (size_t i = 0; i < count; ++i) {
            fn(i);
        }
        return;
    }
    GetDecodePool().Execute(count, fn);
}

u64 GuestContentHash(const u8* data, u64 size) {
    if (CityGuestHash()) {
        return Common::CityHash64(reinterpret_cast<const char*>(data), static_cast<size_t>(size));
    }
    return StripeHash::Hash(data, size);
}

u64 HashGuestBytes(const u8* data, u64 size) {
    const Common::FrameStats::Scope stall{Common::FrameStats::Cause::CoherencyCheck};
    return GuestContentHash(data, size);
}

u64 SampleGuestBytes(const u8* data, u64 size, bool batch_lines) {
    const Common::FrameStats::Scope stall{Common::FrameStats::Cause::CoherencyCheck};
    constexpr u64 LINE = 64;
    constexpr u64 LINES = 256;
    if (size <= LINE * LINES) {
        return HashGuestBytes(data, size);
    }
    // Evenly spaced 64-byte lines, the first and the last one included.
    const u64 last_line = (size - LINE) / LINE;
    const auto line_offset = [last_line](u64 i) { return (last_line * i / (LINES - 1)) * LINE; };
    // The lines PREFETCH_AHEAD further on are asked for while one is hashed (a line may straddle
    // two cache lines when the data is not 64-byte aligned: its last byte is asked for too).
    constexpr u64 PREFETCH_AHEAD = 16;
    const auto prefetch = [&](u64 i) {
        if (i < LINES) {
            const u8* const line = data + line_offset(i);
            PrefetchLine(line);
            PrefetchLine(line + LINE - 1);
        }
    };
    for (u64 i = 0; i < PREFETCH_AHEAD; ++i) {
        prefetch(i);
    }
    if (!CityGuestHash()) {
        // The hash of the lines one after the other (with or without batch_lines), taken line by
        // line without a gathered copy.
        StripeHash::Hasher hasher{LINE * LINES};
        for (u64 i = 0; i + 1 < LINES; ++i) {
            prefetch(i + PREFETCH_AHEAD);
            hasher.Add(data + line_offset(i), 1);
        }
        return hasher.AddLast(data + line_offset(LINES - 1));
    }
    if (batch_lines) {
        std::array<u8, LINE * LINES> samples;
        for (u64 i = 0; i < LINES; ++i) {
            prefetch(i + PREFETCH_AHEAD);
            std::memcpy(samples.data() + i * LINE, data + line_offset(i), LINE);
        }
        return Common::CityHash64(reinterpret_cast<const char*>(samples.data()), samples.size());
    }
    u64 hash = 0;
    for (u64 i = 0; i < LINES; ++i) {
        prefetch(i + PREFETCH_AHEAD);
        hash = Common::CityHash64WithSeed(reinterpret_cast<const char*>(data + line_offset(i)),
                                          LINE, hash);
    }
    return hash;
}

u64 SampleGuestBytesBy(u64 size,
                       const std::function<void(u64 offset, u64 count, u8* destination)>& read,
                       bool batch_lines) {
    const Common::FrameStats::Scope stall{Common::FrameStats::Cause::CoherencyCheck};
    return SampleGuestBytesByQuiet(size, read, batch_lines);
}

u64 SampleGuestBytesByQuiet(
    u64 size, const std::function<void(u64 offset, u64 count, u8* destination)>& read,
    bool batch_lines) {
    constexpr u64 LINE = 64;
    constexpr u64 LINES = 256;
    if (size <= LINE * LINES) {
        std::array<u8, LINE * LINES> all;
        read(0, size, all.data());
        return GuestContentHash(all.data(), size);
    }
    // The same lines, in the same order, as SampleGuestBytes.
    const u64 last_line = (size - LINE) / LINE;
    if (!CityGuestHash()) {
        StripeHash::Hasher hasher{LINE * LINES};
        alignas(16) std::array<u8, LINE> line;
        for (u64 i = 0; i < LINES; ++i) {
            const u64 offset = (last_line * i / (LINES - 1)) * LINE;
            read(offset, LINE, line.data());
            if (i + 1 == LINES) {
                return hasher.AddLast(line.data());
            }
            hasher.Add(line.data(), 1);
        }
    }
    if (batch_lines) {
        std::array<u8, LINE * LINES> samples;
        for (u64 i = 0; i < LINES; ++i) {
            const u64 offset = (last_line * i / (LINES - 1)) * LINE;
            read(offset, LINE, samples.data() + i * LINE);
        }
        return Common::CityHash64(reinterpret_cast<const char*>(samples.data()), samples.size());
    }
    u64 hash = 0;
    std::array<u8, LINE> line;
    for (u64 i = 0; i < LINES; ++i) {
        const u64 offset = (last_line * i / (LINES - 1)) * LINE;
        read(offset, LINE, line.data());
        hash = Common::CityHash64WithSeed(reinterpret_cast<const char*>(line.data()), LINE, hash);
    }
    return hash;
}

u32 DecodedTexelBytes(const HostFormat& host) {
    return VideoCore::Surface::IsPixelFormatBCn(host.pixel_format)
               ? VideoCommon::ConvertedBytesPerBlock(host.pixel_format)
               : 4u;
}

void DecodeBlocks(const HostFormat& host, const BlockInfo& block, std::span<const u8> blocks,
                  u32 width, u32 height, u32 slices, std::span<u8> out) {
    if (VideoCore::Surface::IsPixelFormatASTC(host.pixel_format)) {
        Tegra::Texture::ASTC::Decompress(blocks, width, height, slices, block.width, block.height,
                                         out);
        return;
    }
    VideoCommon::BufferImageCopy copy{};
    copy.image_subresource.num_layers = 1;
    copy.image_extent = {width, height, slices};
    VideoCommon::DecompressBCn(blocks, out, copy, host.pixel_format);
}

UploadPlan PlanStorageUpload(const Storage& s, u64 guest_size) {
    UploadPlan plan;
    const BlockInfo block = FormatBlockInfo(s.desc.format);
    if (!block.bytes) {
        plan.unsupported = "the format has no guest layout";
        return plan;
    }
    if (s.host.guest_layout_differs) {
        plan.unsupported = "depth/stencil texel layout differs between guest and host";
        return plan;
    }
    plan.aspect = AspectOf(s.host);
    plan.linear = IsLinearStorage(s.desc);
    u32 converted_bytes = 0;
    if (s.host.needs_transcode) {
        if (VideoCore::Surface::IsPixelFormatASTC(s.host.pixel_format)) {
            plan.decode = UploadPlan::Decode::Astc;
            converted_bytes = 4;
        } else if (VideoCore::Surface::IsPixelFormatBCn(s.host.pixel_format)) {
            plan.decode = UploadPlan::Decode::Bcn;
            converted_bytes = VideoCommon::ConvertedBytesPerBlock(s.host.pixel_format);
        } else {
            plan.unsupported = "no decoder for the compressed format";
            return plan;
        }
    }
    const u32 layer_count = plan.linear ? 1 : s.layers;
    const u32 level_count = plan.linear ? 1 : s.levels;
    u64 total = 0;
    for (u32 layer = 0; layer < layer_count; ++layer) {
        for (u32 level = 0; level < level_count; ++level) {
            UploadPiece piece{};
            piece.level = level;
            piece.layer = layer;
            piece.width = Mip(s.extent.width, level);
            piece.height = Mip(s.extent.height, level);
            piece.depth = s.type == VK_IMAGE_TYPE_3D ? Mip(s.extent.depth, level) : 1u;
            piece.cols = CeilDiv(piece.width, block.width);
            piece.rows = CeilDiv(piece.height, block.height);
            piece.slices = piece.depth;
            piece.guest_offset = static_cast<u64>(layer) * s.layout.layer_size +
                                 (plan.linear ? 0 : TextureLevelOffset(s.desc, level));
            if (piece.guest_offset >= guest_size) {
                continue;
            }
            TextureLevelTiles(s.desc, level, piece.tile_h, piece.tile_d);
            switch (plan.decode) {
            case UploadPlan::Decode::None:
                piece.staging_size = static_cast<u64>(piece.cols) * block.bytes * piece.rows *
                                     piece.slices;
                break;
            case UploadPlan::Decode::Astc:
            case UploadPlan::Decode::Bcn:
                piece.staging_size = static_cast<u64>(piece.width) * piece.height * piece.slices *
                                     converted_bytes;
                break;
            }
            total = (total + 63) / 64 * 64;
            piece.staging_offset = total;
            total += piece.staging_size;
            plan.pieces.push_back(piece);
        }
    }
    plan.staging_size = total;
    return plan;
}

void FillStorageUpload(const Storage& s, const UploadPlan& plan, const u8* guest, u64 guest_size,
                       u8* staging, UploadStats* stats) {
    const BlockInfo block = FormatBlockInfo(s.desc.format);
    const auto begin = std::chrono::steady_clock::now();
    const auto fill = [&](size_t index) {
        const UploadPiece& piece = plan.pieces[index];
        u8* destination = staging + piece.staging_offset;
        const u64 block_bytes = static_cast<u64>(piece.cols) * block.bytes * piece.rows * piece.slices;
        std::vector<u8> blocks;
        u8* block_destination = destination;
        if (plan.decode != UploadPlan::Decode::None) {
            blocks.resize(static_cast<size_t>(block_bytes));
            block_destination = blocks.data();
        }
        if (plan.linear) {
            // Rows at the texture's pitch; the copy reads the pitch straight from the guest bytes.
            const u64 pitch = s.layout.layer_size / (std::max)(piece.height, 1u);
            std::memset(block_destination, 0, static_cast<size_t>(block_bytes));
            for (u32 row = 0; row < piece.rows; ++row) {
                const u64 source = piece.guest_offset + row * pitch;
                if (source >= guest_size) {
                    break;
                }
                const u64 row_bytes = static_cast<u64>(piece.cols) * block.bytes;
                std::memcpy(block_destination + row * row_bytes, guest + source,
                            static_cast<size_t>((std::min)(row_bytes, guest_size - source)));
            }
        }
        if (!plan.linear) {
            // The unswizzle reads whole GOBs: a storage that stops short of the last one is padded.
            const u64 needed = Tegra::Texture::CalculateSize(true, block.bytes, piece.cols,
                                                             piece.rows, piece.slices,
                                                             piece.tile_h, piece.tile_d);
            const u64 available = guest_size - piece.guest_offset;
            std::vector<u8> padded;
            std::span<const u8> input{guest + piece.guest_offset, static_cast<size_t>(available)};
            if (available < needed) {
                padded.assign(static_cast<size_t>(needed), 0);
                std::memcpy(padded.data(), input.data(), static_cast<size_t>(available));
                input = padded;
            }
            Tegra::Texture::UnswizzleTexture(
                std::span<u8>{block_destination, static_cast<size_t>(block_bytes)}, input,
                block.bytes, piece.cols, piece.rows, piece.slices, piece.tile_h, piece.tile_d, 0,
                true);
        }
        if (plan.decode != UploadPlan::Decode::None) {
            DecodeBlocks(s.host, block, blocks, piece.width, piece.height, piece.slices,
                         std::span<u8>{destination, static_cast<size_t>(piece.staging_size)});
        }
    };
    // Plain copies are memory-bound and small: only decodes and big unswizzles fan out.
    const bool heavy = plan.decode != UploadPlan::Decode::None || plan.staging_size >= (u64{1} << 20);
    if (heavy) {
        RunParallel(plan.pieces.size(), fill);
    } else {
        for (size_t i = 0; i < plan.pieces.size(); ++i) {
            fill(i);
        }
    }
    if (stats && plan.decode != UploadPlan::Decode::None) {
        stats->decode_ns += static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                 std::chrono::steady_clock::now() - begin)
                                                 .count());
        stats->decode_bytes += plan.staging_size;
    }
}

std::vector<VkBufferImageCopy> MakeUploadRegions(const Storage& s, const UploadPlan& plan,
                                                  u64 staging_base) {
    static_cast<void>(s);
    std::vector<VkBufferImageCopy> regions;
    regions.reserve(plan.pieces.size());
    for (const UploadPiece& piece : plan.pieces) {
        regions.push_back({
            .bufferOffset = staging_base + piece.staging_offset,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource{
                .aspectMask = plan.aspect,
                .mipLevel = piece.level,
                .baseArrayLayer = piece.layer,
                .layerCount = 1,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {piece.width, piece.height, piece.depth},
        });
    }
    return regions;
}

} // namespace Vulkan::Nvn
