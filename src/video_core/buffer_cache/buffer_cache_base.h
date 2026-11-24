// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdlib>
#include <algorithm>
#include <array>
#include <bit>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <source_location>
#include <utility>
#include <span>
#include <unordered_map>
#include <vector>

#include "common/container/unordered_map.h"
#include <boost/container/static_vector.hpp>
#include <boost/container/small_vector.hpp>

#include "common/common_types.h"
#include "common/div_ceil.h"
#include "common/literals.h"
#include "common/lru_cache.h"
#include "common/range_sets.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/slot_vector.h"
#include "video_core/buffer_cache/buffer_base.h"
#include "video_core/buffer_cache/storage_range_groups.h"
#include "video_core/buffer_cache/virtual_range_cache.h"
#include "video_core/control/channel_state_cache.h"
#include "video_core/delayed_destruction_ring.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/types.h"

namespace VideoCommon {

using BufferId = Common::SlotId;

using VideoCore::Surface::PixelFormat;
using namespace Common::Literals;

#ifdef __APPLE__
constexpr u32 NUM_VERTEX_BUFFERS = 16;
#else
constexpr u32 NUM_VERTEX_BUFFERS = 32;
#endif
constexpr u32 NUM_TRANSFORM_FEEDBACK_BUFFERS = 4;
constexpr u32 NUM_GRAPHICS_UNIFORM_BUFFERS = 32;
constexpr u32 NUM_COMPUTE_UNIFORM_BUFFERS = 8;
// Native Switch 2 compute programs bind up to 18 global memory ranges (Donkey Kong Bananza),
// so 16 was too few. The tracking bitmasks are u32, which caps this at 32.
constexpr u32 NUM_STORAGE_BUFFERS = 32;
// Compute programs can take pointer table entries beyond 32 bindings (Shader::Info::MAX_SSBOS);
// their enable/written masks are u64.
constexpr u32 NUM_COMPUTE_STORAGE_BUFFERS = 64;
constexpr u32 NUM_TEXTURE_BUFFERS = 32;
constexpr u32 NUM_STAGES = 5;

static_assert(NUM_GRAPHICS_UNIFORM_BUFFERS <= 32, "fast bitmask must fit u32");
static_assert(NUM_STORAGE_BUFFERS <= 32, "fast bitmask must fit u32");

using UniformBufferSizes = std::array<std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES>;
using ComputeUniformBufferSizes = std::array<u32, NUM_COMPUTE_UNIFORM_BUFFERS>;

enum class ObtainBufferSynchronize : u32 {
    NoSynchronize = 0,
    FullSynchronize = 1,
    SynchronizeNoDirty = 2,
};

enum class ObtainBufferOperation : u32 {
    DoNothing = 0,
    MarkAsWritten = 1,
    DiscardWrite = 2,
    MarkQuery = 3,
};

static constexpr BufferId NULL_BUFFER_ID{0};
static constexpr u32 DEFAULT_SKIP_CACHE_SIZE = static_cast<u32>(4_KiB);

struct Binding {
    DAddr device_addr{};
    GPUVAddr gpu_addr{};
    u32 size{};
    BufferId buffer_id;
    u32 segment_first{};
    u32 segment_count{};
    u32 segment_offset{};
    u32 segment_size{};
    bool segment_written{};
    /// Storage bindings: bytes from device_addr the program can write (see
    /// Shader::StorageBufferDescriptor::write_extent); the rest is bound for reads only.
    u32 write_size{~0U};
    /// [DEBUG] storage bindings: cbuf index << 16 | cbuf offset, bit 31 when through a table.
    u32 dbg_cbuf{};
    /// Storage bindings: bytes from device_addr (aligned down) to the pointer the program uses.
    u32 pointer_offset{};
};

struct MultiRangeSegment {
    BufferId buffer_id;
    DAddr device_addr{};
    u32 size{};
};

struct TextureBufferBinding : Binding {
    PixelFormat format;
};

static constexpr Binding NULL_BINDING{
    .device_addr = 0,
    .size = 0,
    .buffer_id = NULL_BUFFER_ID,
};

template <typename Buffer>
struct HostBindings {
    boost::container::static_vector<Buffer*, NUM_VERTEX_BUFFERS> buffers;
    boost::container::static_vector<u64, NUM_VERTEX_BUFFERS> offsets;
    boost::container::static_vector<u64, NUM_VERTEX_BUFFERS> sizes;
    boost::container::static_vector<u64, NUM_VERTEX_BUFFERS> strides;
    u32 min_index{NUM_VERTEX_BUFFERS};
    u32 max_index{0};
};

class BufferCacheChannelInfo : public ChannelInfo {
public:
    BufferCacheChannelInfo() = delete;
    BufferCacheChannelInfo(Tegra::Control::ChannelState& state) noexcept : ChannelInfo(state) {}
    BufferCacheChannelInfo(const BufferCacheChannelInfo& state) = delete;
    BufferCacheChannelInfo& operator=(const BufferCacheChannelInfo&) = delete;

    Binding index_buffer;
    std::array<Binding, NUM_VERTEX_BUFFERS> vertex_buffers;
    std::array<std::array<Binding, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES> uniform_buffers;
    std::array<std::array<Binding, NUM_STORAGE_BUFFERS>, NUM_STAGES> storage_buffers;
    std::array<std::array<TextureBufferBinding, NUM_TEXTURE_BUFFERS>, NUM_STAGES> texture_buffers;
    std::array<Binding, NUM_TRANSFORM_FEEDBACK_BUFFERS> transform_feedback_buffers;
    Binding count_buffer_binding;
    Binding indirect_buffer_binding;

    std::array<Binding, NUM_COMPUTE_UNIFORM_BUFFERS> compute_uniform_buffers;
    std::array<Binding, NUM_COMPUTE_STORAGE_BUFFERS> compute_storage_buffers;
    std::array<TextureBufferBinding, NUM_TEXTURE_BUFFERS> compute_texture_buffers;

    std::array<u32, NUM_STAGES> enabled_uniform_buffer_masks{};
    u32 enabled_compute_uniform_buffer_mask = 0;

    const UniformBufferSizes* uniform_buffer_sizes{};
    const ComputeUniformBufferSizes* compute_uniform_buffer_sizes{};

    std::array<u32, NUM_STAGES> enabled_storage_buffers{};
    std::array<u32, NUM_STAGES> written_storage_buffers{};
    u64 enabled_compute_storage_buffers = 0;
    u64 written_compute_storage_buffers = 0;

    u32 total_graphics_storage_buffers = 0;
    u32 total_compute_storage_buffers = 0;

    std::array<u32, NUM_STAGES> enabled_texture_buffers{};
    std::array<u32, NUM_STAGES> written_texture_buffers{};
    std::array<u32, NUM_STAGES> image_texture_buffers{};
    u32 enabled_compute_texture_buffers = 0;
    u32 written_compute_texture_buffers = 0;
    u32 image_compute_texture_buffers = 0;

    std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS> uniform_cache_hits{};
    std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS> uniform_cache_shots{};

    u32 uniform_buffer_skip_cache_size = DEFAULT_SKIP_CACHE_SIZE;

    bool has_deleted_buffers = false;

    std::array<u32, NUM_STAGES> dirty_uniform_buffers{};
    std::array<u32, NUM_STAGES> fast_bound_uniform_buffers{};
    std::array<std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES>
        uniform_buffer_binding_sizes{};
};

template <class P>
class BufferCache : public VideoCommon::ChannelSetupCaches<BufferCacheChannelInfo> {
    // Page size for caching purposes.
    // This is unrelated to the CPU page size and it can be changed as it seems optimal.
#ifdef YUZU_LEGACY
    static constexpr u32 CACHING_PAGEBITS = 12;
#else
    static constexpr u32 CACHING_PAGEBITS = 16;
#endif
    static constexpr u64 CACHING_PAGESIZE = u64{1} << CACHING_PAGEBITS;

    static constexpr bool IS_OPENGL = P::IS_OPENGL;
    static constexpr bool HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS =
        P::HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS;
    static constexpr bool HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT =
        P::HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT;
    static constexpr bool NEEDS_BIND_UNIFORM_INDEX = P::NEEDS_BIND_UNIFORM_INDEX;
    static constexpr bool NEEDS_BIND_STORAGE_INDEX = P::NEEDS_BIND_STORAGE_INDEX;
    static constexpr bool USE_MEMORY_MAPS = P::USE_MEMORY_MAPS;
    static constexpr bool SEPARATE_IMAGE_BUFFERS_BINDINGS = P::SEPARATE_IMAGE_BUFFER_BINDINGS;
    static constexpr bool USE_MEMORY_MAPS_FOR_UPLOADS = P::USE_MEMORY_MAPS_FOR_UPLOADS;

#ifdef YUZU_LEGACY
    static constexpr s64 TARGET_THRESHOLD = 3_GiB;
#else
    static constexpr s64 TARGET_THRESHOLD = 4_GiB;
#endif

    static constexpr s64 DEFAULT_EXPECTED_MEMORY = 512_MiB;
    static constexpr s64 DEFAULT_CRITICAL_MEMORY = 1_GiB;

    // Debug Flags.

    static constexpr bool DISABLE_DOWNLOADS = true;

    using Maxwell = Tegra::Engines::Maxwell3D::Regs;

    using Runtime = typename P::Runtime;
    using Buffer = typename P::Buffer;
    using Async_Buffer = typename P::Async_Buffer;
    using MemoryTracker = typename P::MemoryTracker;

    struct OverlapResult {
        boost::container::small_vector<BufferId, 16> ids;
        DAddr begin;
        DAddr end;
        bool has_stream_leap = false;
    };

public:
    explicit BufferCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, Runtime& runtime_);

    ~BufferCache();

    void TickFrame();
    template <typename Func>
    void ForEachBoundBuffer(bool compute, Func&& func) {
        if (!channel_state)
            return;
        const auto visit = [&](u32 stage, bool storage, u32 index, bool written,
                               const Binding& binding, u32 size) {
            if (!binding.size || !size)
                return;
            const Buffer* host = binding.segment_count < 2 && binding.buffer_id
                                     ? &slot_buffers[binding.buffer_id]
                                     : nullptr;
            func(stage, storage, index, written, binding, (std::min)(binding.size, size), host,
                 host ? host->Offset(binding.device_addr) : 0);
        };
        const auto& state = *channel_state;
        if (compute) {
            ForEachEnabledBit(state.enabled_compute_uniform_buffer_mask, [&](u32 i) {
                visit(5, false, i, false, state.compute_uniform_buffers[i],
                      (*state.compute_uniform_buffer_sizes)[i]);
            });
            ForEachEnabledBit(state.enabled_compute_storage_buffers, [&](u32 i) {
                visit(5, true, i, (state.written_compute_storage_buffers >> i) & 1,
                      state.compute_storage_buffers[i], state.compute_storage_buffers[i].size);
            });
        } else {
            for (u32 stage = 0; stage < NUM_STAGES; ++stage) {
                ForEachEnabledBit(state.enabled_uniform_buffer_masks[stage], [&](u32 i) {
                    visit(stage, false, i, false, state.uniform_buffers[stage][i],
                          (*state.uniform_buffer_sizes)[stage][i]);
                });
                ForEachEnabledBit(state.enabled_storage_buffers[stage], [&](u32 i) {
                    visit(stage, true, i, (state.written_storage_buffers[stage] >> i) & 1,
                          state.storage_buffers[stage][i], state.storage_buffers[stage][i].size);
                });
            }
        }
    }

    bool BindMultiRangeStorage(const Binding& binding, bool is_written,
                               std::span<const MultiRangeSegment> pool);

    /// Virtual range lookups served from the cache and walked again since the last call.
    std::pair<u64, u64> TakeVirtualRangeCounts() noexcept {
        return virtual_ranges.TakeCounts();
    }

    void PrepareGroupedStorage(std::span<const std::pair<Binding*, bool>> bindings,
                               std::vector<MultiRangeSegment>& pool);

    void ResolveMultiRangeStorage(Binding& binding, bool is_written,
                                  std::vector<MultiRangeSegment>& pool);

    /// Resolves segments whose buffer was deleted by a join later in the same update.
    void RefreshMultiRangeSegments(std::vector<MultiRangeSegment>& pool);

    void UnmapGPUMemory(size_t as_id, GPUVAddr gpu_addr, size_t size);

    void WriteMemory(DAddr device_addr, u64 size);

    void CachedWriteMemory(DAddr device_addr, u64 size);

    bool OnCPUWrite(DAddr device_addr, u64 size);

    void DownloadMemory(DAddr device_addr, u64 size);

    using DownloadSnapshot =
        decltype(std::declval<Runtime&>().DownloadStagingBuffer(size_t{}, true));

    /// Records a copy of [device_addr, device_addr + size) into a new deferred download staging
    /// buffer without waiting for the GPU, and without changing what is marked GPU-modified. The
    /// copy is done once the scheduler tick current now has completed; the caller frees the
    /// staging buffer with runtime.FreeDeferredStagingBuffer. Empty if the range is not inside a
    /// single buffer (or the backend does not map download memory).
    [[nodiscard]] std::optional<DownloadSnapshot> RecordDownloadSnapshot(DAddr device_addr,
                                                                         u64 size);

    std::optional<VideoCore::RasterizerDownloadArea> GetFlushArea(DAddr device_addr, u64 size);

    bool InlineMemory(DAddr dest_address, size_t copy_size, std::span<const u8> inlined_buffer);

    void BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr, u32 size);

    void DisableGraphicsUniformBuffer(size_t stage, u32 index);

    void UpdateGraphicsBuffers(bool is_indexed);

    void UpdateComputeBuffers();

    void BindHostGeometryBuffers(bool is_indexed);

    void BindHostStageBuffers(size_t stage);

    void BindHostComputeBuffers();

    void SetUniformBuffersState(const std::array<u32, NUM_STAGES>& mask,
                                const UniformBufferSizes* sizes);

    void SetComputeUniformBufferState(u32 mask, const ComputeUniformBufferSizes* sizes);

    void UnbindGraphicsStorageBuffers(size_t stage);

    bool BindGraphicsStorageBuffer(size_t stage, size_t ssbo_index, u32 cbuf_index, u32 cbuf_offset,
                                   bool is_written, u32 indirect_offset = ~0U,
                                   u32 write_extent = ~0U);

    /// A range of a pointer table span (Shader::StorageBufferDescriptor::span_slot): the
    /// program stores through the pointers at table + entry_offset + i * stride for the draw's
    /// vertices i in [first, first + count), the table pointer being c[cbuf_index][cbuf_offset].
    /// Their targets are coalesced into at most max_ranges ranges (each reaching extent bytes
    /// past its last pointer) and range `slot` is bound. Returns the bound GPU base and size;
    /// a size of zero binds nothing.
    std::pair<GPUVAddr, u32> BindGraphicsTableSpan(size_t stage, size_t ssbo_index, u32 cbuf_index,
                                                   u32 cbuf_offset, u32 stride, u32 entry_offset,
                                                   u32 slot, u32 max_ranges, u32 extent,
                                                   bool is_written, u32 first, u32 count);

    void UnbindGraphicsTextureBuffers(size_t stage);

    void BindGraphicsTextureBuffer(size_t stage, size_t tbo_index, GPUVAddr gpu_addr, u32 size,
                                   PixelFormat format, bool is_written, bool is_image);

    void UnbindComputeStorageBuffers();

    void BindComputeStorageBuffer(size_t ssbo_index, u32 cbuf_index, u32 cbuf_offset,
                                  bool is_written, u32 indirect_offset = ~0U,
                                  u32 write_extent = ~0U);

    void UnbindComputeTextureBuffers();

    void BindComputeTextureBuffer(size_t tbo_index, GPUVAddr gpu_addr, u32 size, PixelFormat format,
                                  bool is_written, bool is_image);

    [[nodiscard]] std::pair<Buffer*, u32> ObtainBuffer(GPUVAddr gpu_addr, u32 size,
                                                       ObtainBufferSynchronize sync_info,
                                                       ObtainBufferOperation post_op);

    [[nodiscard]] std::pair<Buffer*, u32> ObtainCPUBuffer(DAddr gpu_addr, u32 size,
                                                          ObtainBufferSynchronize sync_info,
                                                          ObtainBufferOperation post_op);
    void FlushCachedWrites();

    /// Return true when there are uncommitted buffers to be downloaded
    [[nodiscard]] bool HasUncommittedFlushes() const noexcept;

    void AccumulateFlushes();

    /// Return true when the caller should wait for async downloads
    [[nodiscard]] bool ShouldWaitAsyncFlushes() const noexcept;

    /// Commit asynchronous downloads
    void CommitAsyncFlushes();
    void CommitAsyncFlushesHigh();

    /// Pop asynchronous downloads
    void PopAsyncFlushes();
    void PopAsyncBuffers();

    bool DMACopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount);

    bool DMAClear(GPUVAddr src_address, u64 amount, u32 value);

    /// Collects command-page writes while recording a compute dispatch or DMA copy.
    void Nx2CollectListCopies(bool collect) {
        nx2_collect_list_copies = collect;
        if (!collect) {
            nx2_list_copy_requests.clear();
        }
    }
    /// True when it recorded copies, which the GPU completes at `tick` (the caller submits).
    bool Nx2CopyListPages(u64 tick);
    enum class Nx2ListRead { Unknown, Copied, Synced };
    /// Front thread: Synced when guest memory holds the page `device_addr` lies in (flushed and
    /// written by no GPU work since), else Copied with `out` filled from the copy of its last
    /// write once the GPU is done with it. Unknown otherwise: the caller drains and flushes.
    Nx2ListRead Nx2ReadListCopy(DAddr device_addr, std::span<u8> out);
    /// Front thread, the split drained: the command list page was flushed to guest memory.
    void Nx2NoteListSynced(DAddr page);
    /// CPU and inline writes: they reach guest memory, so a synced page stays synced, but its copy
    /// no longer holds the page.
    void Nx2NoteListHostWrite(DAddr device_addr, u64 size);

    /// Return true when a CPU region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(DAddr addr, size_t size);

    /// True when the cache provably holds no GPU-written data in [addr, addr + size). Safe to call
    /// without `mutex` (it reads only the page filter's atomic bits); false means "unknown": ask
    /// IsRegionGpuModified under the lock.
    [[nodiscard]] bool ProvesNotGpuModified(DAddr addr, size_t size) const {
        return gpu_modified_ranges.ProvesDisjoint(addr, size);
    }

    /// The guest CPU reads [addr, addr + size) (a reactive flush asked for it): without
    /// AsyncBufferDownloads, GPU writes there are still copied back at fences from now on.
    void NoteCpuRead(DAddr addr, size_t size);

    /// Return true when a region is registered on the cache
    [[nodiscard]] bool IsRegionRegistered(DAddr addr, size_t size);

    /// Return true when a CPU region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(DAddr addr, size_t size);

    void SetDrawIndirect(
        const Tegra::Engines::Maxwell3D::DrawManager::IndirectParams* current_draw_indirect_) {
        current_draw_indirect = current_draw_indirect_;
    }

    [[nodiscard]] std::pair<Buffer*, u32> GetDrawIndirectCount();

    [[nodiscard]] std::pair<Buffer*, u32> GetDrawIndirectBuffer();

    template <typename Func>
    void BufferOperations(Func&& func) {
        do {
            channel_state->has_deleted_buffers = false;
            func();
        } while (channel_state->has_deleted_buffers);
    }

    std::recursive_mutex mutex;
    Runtime& runtime;
    bool any_buffer_uploaded = false;

private:
    template <typename Mask, typename Func>
    static void ForEachEnabledBit(Mask enabled_mask, Func&& func) {
        for (u32 index = 0; enabled_mask != 0; ++index, enabled_mask >>= 1) {
            const int disabled_bits = std::countr_zero(enabled_mask);
            index += disabled_bits;
            enabled_mask >>= disabled_bits;
            func(index);
        }
    }

    template <typename Func>
    void ForEachBufferInRange(DAddr device_addr, u64 size, Func&& func) {
        const u64 page_end = Common::DivCeil(device_addr + size, CACHING_PAGESIZE);
        for (u64 page = device_addr >> CACHING_PAGEBITS; page < page_end;) {
            const BufferId buffer_id = page_table[page];
            if (!buffer_id) {
                ++page;
                continue;
            }
            Buffer& buffer = slot_buffers[buffer_id];
            func(buffer_id, buffer);

            const DAddr end_addr = buffer.CpuAddr() + buffer.SizeBytes();
            page = Common::DivCeil(end_addr, CACHING_PAGESIZE);
        }
    }

    static bool IsRangeGranular(DAddr device_addr, size_t size) {
        return (device_addr & ~Core::DEVICE_PAGEMASK) ==
               ((device_addr + size) & ~Core::DEVICE_PAGEMASK);
    }

    void RunGarbageCollector();

    void BindHostIndexBuffer();

    void BindHostVertexBuffers();

    void BindHostDrawIndirectBuffers();

    void BindHostGraphicsUniformBuffers(size_t stage);

    void BindHostGraphicsUniformBuffer(size_t stage, u32 index, u32 binding_index, bool needs_bind);

    void BindHostGraphicsStorageBuffers(size_t stage);

    /// The bytes a written storage binding can store to (see Binding::write_size).
    [[nodiscard]] std::pair<DAddr, u32> StorageWriteRange(const Binding& binding, u32 size) const;

    void BindHostGraphicsTextureBuffers(size_t stage);

    void BindHostTransformFeedbackBuffers();

    void BindHostVertexBuffer(u32 index, Buffer& buffer, u32 offset, u32 size, u32 stride);

    void BindHostComputeUniformBuffers();

    void BindHostComputeStorageBuffers();

    void BindHostComputeTextureBuffers();

    void DoUpdateGraphicsBuffers(bool is_indexed);

    void DoUpdateComputeBuffers();

    void UpdateIndexBuffer();

    void UpdateVertexBuffers();

    void UpdateVertexBuffer(u32 index);

    void UpdateDrawIndirect();

    void UpdateUniformBuffers(size_t stage);

    void UpdateStorageBuffers(size_t stage);

    void UpdateTextureBuffers(size_t stage);

    void UpdateTransformFeedbackBuffers();

    void UpdateTransformFeedbackBuffer(u32 index);

    void UpdateComputeUniformBuffers();

    void UpdateComputeStorageBuffers();

    void UpdateComputeTextureBuffers();

    void MarkWrittenBuffer(BufferId buffer_id, DAddr device_addr, u32 size, bool needs_sync = false);
    void Nx2NoteListWrite(BufferId buffer_id, DAddr device_addr, u64 size);

    [[nodiscard]] BufferId FindBuffer(DAddr device_addr, u32 size, bool sparse_compatible);

    void SynchronizeBufferWrites(Buffer& buffer);

    [[nodiscard]] OverlapResult ResolveOverlaps(DAddr device_addr, u32 wanted_size);

    void JoinOverlap(BufferId new_buffer_id, BufferId overlap_id, bool accumulate_stream_score);

    [[nodiscard]] BufferId CreateBuffer(DAddr device_addr, u32 wanted_size,
                                        bool sparse_compatible);

    void Register(BufferId buffer_id);

    void Unregister(BufferId buffer_id);

    template <bool insert>
    void ChangeRegister(BufferId buffer_id);

    void TouchBuffer(Buffer& buffer, BufferId buffer_id) noexcept;

    bool SynchronizeBuffer(Buffer& buffer, DAddr device_addr, u32 size,
                           std::source_location caller = std::source_location::current());

    void UploadMemory(Buffer& buffer, u64 total_size_bytes, u64 largest_copy,
                      std::span<BufferCopy> copies);

    void ImmediateUploadMemory(Buffer& buffer, u64 largest_copy,
                               std::span<const BufferCopy> copies);

    void MappedUploadMemory(Buffer& buffer, u64 total_size_bytes, std::span<BufferCopy> copies);

    void DownloadBufferMemory(Buffer& buffer_id);

    void DownloadBufferMemory(Buffer& buffer_id, DAddr device_addr, u64 size);

    void DeleteBuffer(BufferId buffer_id, bool do_not_mark = false);

    [[nodiscard]] Binding StorageBufferBinding(GPUVAddr ssbo_addr, u32 cbuf_index,
                                               bool is_written, bool native_pointer,
                                               bool table_pointer = false) const;
    [[nodiscard]] Binding StorageBufferBindingUncached(GPUVAddr gpu_addr, u64 following_word,
                                                       u32 cbuf_index, bool is_written,
                                                       bool native_pointer,
                                                       bool table_pointer) const;

    [[nodiscard]] TextureBufferBinding GetTextureBufferBinding(GPUVAddr gpu_addr, u32 size,
                                                               PixelFormat format);

    [[nodiscard]] std::span<const u8> ImmediateBufferWithData(DAddr device_addr, size_t size);

    [[nodiscard]] std::span<u8> ImmediateBuffer(size_t wanted_capacity);

    [[nodiscard]] bool HasFastUniformBufferBound(size_t stage, u32 binding_index) const noexcept;

    [[nodiscard]] Binding& VertexBufferSlot(u32 index);

    [[nodiscard]] const Binding& VertexBufferSlot(u32 index) const;

    void UpdateVertexBufferSlot(u32 index, const Binding& binding);

    void ClearDownload(DAddr base_addr, u64 size);

    void InlineMemoryImplementation(DAddr dest_address, size_t copy_size,
                                    std::span<const u8> inlined_buffer);

    Tegra::MaxwellDeviceMemoryManager& device_memory;

    Common::SlotVector<Buffer> slot_buffers;
#ifdef YUZU_LEGACY
    static constexpr size_t TICKS_TO_DESTROY = 6;
#else
    static constexpr size_t TICKS_TO_DESTROY = 8;
#endif
    DelayedDestructionRing<Buffer, TICKS_TO_DESTROY> delayed_destruction_ring;

    const Tegra::Engines::Maxwell3D::DrawManager::IndirectParams* current_draw_indirect{};

    u32 last_index_count = 0;

    u32 enabled_vertex_buffers_mask = 0;
    u64 vertex_buffers_serial = 0;
    std::array<Binding, 32> v_buffer{};

    boost::container::small_vector<BufferCopy, 4> upload_copies;

    /// SynchronizeBuffer memo: a range synchronized at 'epoch' had no CPU or untracked bit left
    /// in it, so while memory_tracker reports its regions unchanged since then, a new sync would
    /// upload nothing. Direct mapped; EDEN_SYNC_MEMO=0 turns it off.
    struct SyncMemoEntry {
        DAddr device_addr = 0;
        u32 size = 0;
        u64 epoch = 0;
    };
    std::array<SyncMemoEntry, 256> sync_memo{};
    u64 sync_memo_hits = 0;
    u64 sync_memo_misses = 0;

    MemoryTracker memory_tracker;
    Common::RangeSet<DAddr> uncommitted_gpu_modified_ranges;
    Common::RangeSet<DAddr> gpu_modified_ranges;

    /// Whether every GPU-written range is copied back to guest memory at the next fence. Switch 2
    /// titles (native tiling) copy back only what their CPU reads (cpu_read_ranges): copying
    /// everything is a large part of each fence's cost, and the reactive flush still brings any
    /// other page the CPU reads. From PokeDinner c3be47da (Eden-NS2 15b48a6615), where copying
    /// everything dropped Pokemon Pokopia from 60 to about 42 frames a second.
    bool AsyncBufferDownloads() const {
        return gpu_memory == nullptr || !gpu_memory->UsesNativeTiling();
    }
    /// Buffer pages the guest CPU reads (NoteCpuRead).
    Common::RangeSet<DAddr> cpu_read_ranges;

    /// Queues the parts of a GPU write to [addr, addr + size) that are copied back at the next
    /// fence.
    void QueueDownload(DAddr addr, u64 size) {
        if (AsyncBufferDownloads()) {
            uncommitted_gpu_modified_ranges.Add(addr, size);
            return;
        }
        cpu_read_ranges.ForEachInRange(addr, size, [this](DAddr start, DAddr end) {
            uncommitted_gpu_modified_ranges.Add(start, end - start);
        });
    }
    std::deque<Common::RangeSet<DAddr>> committed_gpu_modified_ranges;

    // Async Buffers
    Common::OverlapRangeSet<DAddr> async_downloads;
    std::deque<std::optional<Async_Buffer>> async_buffers;
    std::deque<boost::container::small_vector<BufferCopy, 4>> pending_downloads;
    std::optional<Async_Buffer> current_buffer;

    std::deque<Async_Buffer> async_buffers_death_ring;

    size_t immediate_buffer_capacity = 0;
    Common::ScratchBuffer<u8> immediate_buffer_alloc;

    struct LRUItemParams {
        using ObjectType = BufferId;
        using TickType = u64;
    };
    Common::LeastRecentlyUsedCache<LRUItemParams> lru_cache;
    VirtualRangeCache virtual_ranges;
    // Scratch of the grouped storage path, kept between draws instead of allocated on each.
    std::vector<std::pair<Binding*, bool>> grouped_bindings;
    std::vector<StorageAddressRange> grouped_ranges;
    std::vector<size_t> grouped_order;
    std::vector<StorageRangeGroup> grouped_groups;
    std::vector<std::pair<VirtualSegments, bool>> grouped_physical;
    std::vector<MultiRangeSegment> graphics_segments;

    /// Raised whenever a buffer is registered or unregistered: while it holds, FindBuffer over a
    /// covered range returns the same id.
    u64 buffer_generation = 0;

    /// PrepareGroupedStorage memo, one for graphics and one for compute: the same bindings,
    /// mappings and buffers give the same groups. EDEN_GROUPED_MEMO=0 turns it off.
    struct GroupedMemo {
        struct Input {
            GPUVAddr gpu_addr;
            DAddr device_addr;
            u32 size;
            bool written;
            bool operator==(const Input&) const = default;
        };
        bool valid = false;
        size_t as_id = 0;
        size_t pool_base = 0;
        u64 buffer_generation = 0;
        u64 range_generation = 0;
        u64 map_serial = 0;
        std::vector<Input> inputs;
        std::vector<Binding> outputs;
        std::vector<MultiRangeSegment> segments;
    };
    std::array<GroupedMemo, 2> grouped_memo{};
    std::vector<MultiRangeSegment> compute_segments;
    u64 frame_tick = 0;
    u64 total_used_memory = 0;
    u64 minimum_memory = 0;
    u64 critical_memory = 0;
    BufferId inline_buffer_id;
#ifdef YUZU_LEGACY
    bool immediately_free = false;
#endif

    std::array<BufferId, ((1ULL << 34) >> CACHING_PAGEBITS)> page_table;
    Common::ScratchBuffer<u8> tmp_buffer;

    /// A command list page (Nx2CopyListPages): the GPU writes noted to it and the last copy.
    using Nx2Staging = decltype(std::declval<Runtime&>().DownloadStagingBuffer(size_t{}, true));
    struct Nx2ListPage {
        u64 writes{};      ///< GPU writes noted to the page
        u64 copy_writes{}; ///< `writes` when the copy was made: the copy is current while equal
        u64 copy_tick{};   ///< GPU tick that completes the copy
        /// `writes` when the front last flushed the page: guest memory holds it while equal
        u64 synced_writes{~u64{0}};
        u32 copy_begin{}; ///< Bytes of the page the copy holds (a buffer can end inside it)
        u32 copy_end{};
        /// Written outside a compute dispatch or a DMA copy (a draw's storage writes) at least
        /// once: the front can't tell from what it sent whether such a write is still queued, so
        /// the page's lists always drain the split instead of reading a copy.
        bool draw_written{};
        std::optional<Nx2Staging> copy;
    };
    std::mutex nx2_list_mutex; ///< nx2_list_pages: the split's front reads the copies
    std::unordered_map<DAddr, Nx2ListPage> nx2_list_pages;
    /// Copies replaced while the GPU may still write them: freed once it is past their tick.
    std::vector<std::pair<u64, Nx2Staging>> nx2_retired_list_copies;
    boost::container::small_vector<std::pair<BufferId, DAddr>, 8> nx2_list_copy_requests;
    bool nx2_collect_list_copies{};
};

} // namespace VideoCommon
