// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstring>
#include <iterator>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "core/memory.h"
#include "video_core/texture_cache/types.h"

// Reuse of resolved texture banks between draws.
//
// Fortune's Weave binds a bank of ~1250 textures (a texture array indexed from a constant buffer)
// on most draws, and the bank is the same as on the previous draw most of the time. Resolving it
// (a descriptor table read, a view lookup and an image preparation per texture) dominated the GPU
// thread. A resolved bank is kept with what its resolution depended on, and reused while all of it
// is unchanged:
//  - the descriptor table (channel, address, limit) and the bytes of the entries the bank reads
//    (DescriptorRunSnapshot, compared on every reuse);
//  - the texture cache generation (TextureCache::bank_generation), bumped wherever images or
//    views are created, deleted or rescaled and wherever an image is marked as written by the CPU
//    (which the next preparation would upload);
//  - for views, images with aliases are still prepared on every reuse (alias synchronization
//    depends on render target writes, which happen on every draw), and the LRU ticks of the bank's
//    images are refreshed once per frame (what the per-draw preparation kept fresh).
// EDEN_NO_HEAP_REUSE disables it (compact_texture_table.h).

namespace VideoCommon {

/// A copy of the descriptor table entries a bank reads, as runs of nearby entries (small gaps are
/// covered too, so a dense bank is a few long runs). Compared on every reuse: the guest may rewrite
/// an entry without any other notice.
class DescriptorRunSnapshot {
public:
    /// indices: the entries to cover, any order, repeats allowed; those above limit are skipped
    /// (they resolve to null whatever they hold). Returns false when part of the range is not host
    /// memory (nothing is reused then). Memory: Tegra::MemoryManager (GetPointer).
    template <typename Memory>
    bool Capture(const Memory& gpu_memory, GPUVAddr table_addr,
                 std::span<const u32> indices, u32 limit, size_t entry_size) {
        Clear();
        table = table_addr;
        stride = entry_size;
        sorted.assign(indices.begin(), indices.end());
        std::ranges::sort(sorted);
        const auto unique_end = std::ranges::unique(sorted);
        sorted.erase(unique_end.begin(), unique_end.end());
        size_t offset = 0;
        for (const u32 index : sorted) {
            if (index > limit) {
                break;
            }
            if (!runs.empty() && index - (runs.back().first + runs.back().count) <= MAX_GAP) {
                const u32 grown = index + 1 - runs.back().first;
                offset += (grown - runs.back().count) * stride;
                runs.back().count = grown;
                continue;
            }
            runs.push_back(Run{.first = index, .count = 1, .offset = offset});
            offset += stride;
        }
        bytes.resize(offset);
        for (const Run& run : runs) {
            u8* const dest = bytes.data() + run.offset;
            if (!VisitHost(gpu_memory, table + u64{run.first} * stride, run.count * stride,
                           [dest](const u8* host, size_t size, size_t at) {
                               std::memcpy(dest + at, host, size);
                               return true;
                           })) {
                Clear();
                return false;
            }
        }
        valid = true;
        return true;
    }

    /// True if the guest entries still hold the captured bytes.
    template <typename Memory>
    [[nodiscard]] bool Matches(const Memory& gpu_memory) const {
        if (!valid) {
            return false;
        }
        for (const Run& run : runs) {
            const u8* const expected = bytes.data() + run.offset;
            if (!VisitHost(gpu_memory, table + u64{run.first} * stride, run.count * stride,
                           [expected](const u8* host, size_t size, size_t at) {
                               return std::memcmp(expected + at, host, size) == 0;
                           })) {
                return false;
            }
        }
        return true;
    }

    /// The captured bytes of an entry, or nullptr if it was not captured.
    [[nodiscard]] const u8* Entry(u32 index) const {
        const auto it = std::ranges::upper_bound(runs, index, {}, &Run::first);
        if (it == runs.begin()) {
            return nullptr;
        }
        const Run& run = *std::prev(it);
        if (index - run.first >= run.count) {
            return nullptr;
        }
        return bytes.data() + run.offset + size_t{index - run.first} * stride;
    }

    [[nodiscard]] size_t Bytes() const noexcept {
        return bytes.size();
    }

    void Clear() noexcept {
        runs.clear();
        bytes.clear();
        valid = false;
    }

private:
    /// Entries between two bank entries copied along instead of starting a new run.
    static constexpr u32 MAX_GAP = 8;

    struct Run {
        u32 first;
        u32 count;
        size_t offset;
    };

    /// Visits [addr, addr + size) as host memory, one device page at a time (contiguous within a
    /// page, as DescriptorTable::Read relies on). Stops and returns false when visit does or when a
    /// page has no host pointer.
    template <typename Memory, typename Visit>
    static bool VisitHost(const Memory& gpu_memory, GPUVAddr addr, size_t size,
                          Visit&& visit) {
        size_t done = 0;
        while (done < size) {
            const GPUVAddr at = addr + done;
            const size_t in_page = Core::Memory::YUZU_PAGESIZE - (at & Core::Memory::YUZU_PAGEMASK);
            const size_t chunk = (std::min)(in_page, size - done);
            const u8* const host = gpu_memory.GetPointer(at);
            if (!host || !visit(host, chunk, done)) {
                return false;
            }
            done += chunk;
        }
        return true;
    }

    GPUVAddr table{};
    size_t stride{};
    bool valid{};
    std::vector<Run> runs;
    std::vector<u8> bytes;
    std::vector<u32> sorted;
};

/// What a bank's resolution depended on, common to view and sampler banks.
struct TextureBankKey {
    u64 hash{};             ///< Of the indices (CompactTextureTable::tics_hash / tscs_hash)
    std::vector<u32> indices;
    const void* channel{};
    GPUVAddr table_addr{};
    u32 table_limit{};
    u64 stamp{};            ///< Unique per store; names this exact resolution (descriptor reuse)
    u64 last_use{};
    DescriptorRunSnapshot snapshot;

    [[nodiscard]] bool SameIndices(u64 other_hash, std::span<const u32> other) const noexcept {
        return stamp != 0 && hash == other_hash && indices.size() == other.size() &&
               std::equal(indices.begin(), indices.end(), other.begin());
    }
};

struct ViewBankEntry : TextureBankKey {
    std::vector<ImageViewId> ids;
    u64 generation{};
    u64 frame_tick{};
    std::vector<ImageId> images;       ///< Distinct images of the bank (LRU, verification)
    std::vector<ImageId> alias_images; ///< Images with aliases, prepared on every reuse
};

struct SamplerBankEntry : TextureBankKey {
    std::vector<SamplerId> ids;
};

/// The texture pool's marked elements (TexturePoolSet), resolved incrementally: see
/// TextureCache::ResolvePoolViews. The list of marked elements only grows, so an element keeps
/// its position.
struct PoolViewState {
    bool valid{};
    bool resolving{}; ///< ResolvePoolViews is looking elements up (its own lookups are not noted)
    // What every stored id depends on; a change resolves all the elements again.
    const void* channel{};
    GPUVAddr table_addr{};
    u32 table_limit{};
    u64 table_serial{};    ///< DescriptorTable::serial
    u64 view_generation{}; ///< TextureCache::view_generation
    // What the images' preparation depends on.
    u64 bank_generation{}; ///< TextureCache::bank_generation the images were last all prepared at
    u64 frame_tick{};      ///< Frame the images' LRU ticks were last refreshed in
    std::vector<u32> indices;         ///< TIC index by position
    std::vector<ImageViewId> ids;     ///< Resolved view by position
    std::vector<ImageId> images;      ///< The view's image by position (null: none to prepare)
    std::vector<u32> alias_positions; ///< Positions whose image has aliases (one per image)
    std::vector<u32> position_of;     ///< By TIC index: position + 1, or 0 if not marked
    std::vector<u32> redo;            ///< Positions another lookup saw a new entry for
    std::vector<u8> redo_pending;     ///< By position: is in redo
    std::vector<u32> changed;         ///< Positions whose id changed in the last resolve
    DescriptorRunSnapshot snapshot;   ///< The elements' entries, equal to the table's copies
    bool snapshot_valid{};
};

/// What TextureCache::ResolvePoolViews returns. The spans stay valid until the next call.
struct PoolViewsResult {
    std::span<const ImageViewId> ids; ///< By position in the indices given
    std::span<const u32> changed;     ///< Positions whose id differs from the previous call's
    bool all_changed{};               ///< Every position may differ (changed is not filled)
};

/// A few banks (a scene binds one or two; each pipeline stage binds the same one).
template <typename Entry>
struct TextureBankCache {
    static constexpr size_t SIZE = 8;
    std::vector<Entry> entries = std::vector<Entry>(SIZE);
    u64 use{};

    Entry* Find(u64 hash, std::span<const u32> indices) {
        for (Entry& entry : entries) {
            if (entry.SameIndices(hash, indices)) {
                return &entry;
            }
        }
        return nullptr;
    }

    /// The entry of these indices, or the least recently used one (for a new store).
    Entry& Slot(u64 hash, std::span<const u32> indices) {
        if (Entry* const found = Find(hash, indices)) {
            return *found;
        }
        return *std::ranges::min_element(entries, {}, &Entry::last_use);
    }

    void Touch(Entry& entry) {
        entry.last_use = ++use;
    }
};

} // namespace VideoCommon
