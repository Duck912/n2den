// SPDX-FileCopyrightText: Copyright (c) 2026 seedds (nxnative)
// SPDX-FileCopyrightText: Copyright 2026 n2den Contributors
// SPDX-License-Identifier: GPL-3.0-or-later AND MIT

#include "common/settings.h"
#include "video_core/nvn/shader/nvn_program_map.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include <fmt/format.h>

#include "common/alignment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/native.h"
#include "shader_recompiler/frontend/ampere/native_program_key.h"
#include "shader_recompiler/frontend/ampere/program_capture.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/program_header.h"

namespace VideoCommon::Nvn {
namespace {

u32 ReadU32(std::span<const u8> bytes, size_t offset) {
    u32 value;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

u32 ReadU16(std::span<const u8> bytes, size_t offset) {
    u16 value;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

// Sampler ids are 12 bits wide: keys 1..0xFFE name records, 0xFFF means "unknown" (overflow).
constexpr u32 UnknownHandleKey = 0xFFF;

} // namespace

std::optional<NvnControl> ParseControl(std::span<const u8> control, std::string& error) {
    if (control.size() < MinimumControlBytes) {
        error = fmt::format("control section is {} bytes, need at least {:#x}", control.size(),
                            MinimumControlBytes);
        return std::nullopt;
    }
    NvnControl result;
    result.magic = ReadU32(control, 0);
    result.version = ReadU32(control, 4);
    if (result.magic != ControlMagic) {
        error = fmt::format("bad control magic {:#x}", result.magic);
        return std::nullopt;
    }
    if (result.version != ControlVersion) {
        error = fmt::format("unsupported control version {:#x}", result.version);
        return std::nullopt;
    }
    result.own_length = ReadU32(control, 0x30);
    result.inst_bytes = ReadU32(control, 0x38);
    result.c1_offset = ReadU32(control, 0x40);
    result.code_bytes = ReadU32(control, 0x44);
    result.register_count = control[0x4c];
    result.stage_byte = control[0x56];
    for (size_t i = 0; i < 3; ++i) {
        result.thread_group[i] = ReadU32(control, 0x58 + i * 4);
    }
    result.local_memory_per_warp_bytes = ReadU32(control, 0x50);
    result.shared_memory_bytes = ReadU32(control, 0x64);
    if (result.stage_byte >= ShaderStageCount) {
        error = fmt::format("bad stage byte {:#x}", result.stage_byte);
        return std::nullopt;
    }
    if (result.c1_offset > result.code_bytes) {
        error = fmt::format("c1 offset {:#x} is past the code area ({:#x})", result.c1_offset,
                            result.code_bytes);
        return std::nullopt;
    }
    if (result.inst_bytes > result.code_bytes) {
        error = fmt::format("instruction bytes {:#x} exceed the code area ({:#x})",
                            result.inst_bytes, result.code_bytes);
        return std::nullopt;
    }
    if (result.c1_offset < result.inst_bytes) {
        error = fmt::format("c1 offset {:#x} is inside the instructions ({:#x} bytes)",
                            result.c1_offset, result.inst_bytes);
        return std::nullopt;
    }
    if (result.stage_byte == static_cast<u32>(ShaderStage::Compute)) {
        if (control.size() < ComputeControlBytes) {
            error = fmt::format("compute control section is {} bytes, need at least {:#x}",
                                control.size(), ComputeControlBytes);
            return std::nullopt;
        }
        result.has_compute_counts = true;
        std::memcpy(result.compute_raw.data(), control.data() + 0xa8, result.compute_raw.size());
        auto& layout = result.compute;
        layout.push_constant_words = control[0xa9];
        layout.uniform_count = control[0xb1];
        layout.storage_count = control[0xb2];
        layout.texture_count = control[0xb3];
        layout.image_count = control[0xb6];
        layout.declared_total = ReadU16(control, 0xba);
        layout.uniform_offset = Common::AlignUp(
            layout.push_constant_offset + layout.push_constant_words * 4, ComputeEntryTableAlign);
        layout.storage_offset = layout.uniform_offset + layout.uniform_count * 0x10;
        layout.texture_offset = layout.storage_offset + layout.storage_count * 0x10;
        layout.image_offset = layout.texture_offset + layout.texture_count * HandleStride;
        layout.end_offset = layout.image_offset + layout.image_count * HandleStride;
        layout.total_matches = layout.end_offset == layout.declared_total;
    }
    if (result.stage_byte == static_cast<u32>(ShaderStage::Task) ||
        result.stage_byte == static_cast<u32>(ShaderStage::Mesh)) {
        auto& mesh = result.mesh;
        mesh.threads = static_cast<u16>(ReadU16(control, 0x58));
        mesh.output_bytes = static_cast<u16>(ReadU16(control, 0x5a));
        mesh.local_bytes = static_cast<u16>(ReadU16(control, 0x5c));
        mesh.topology = control[0x5f];
        mesh.max_vertices = static_cast<u16>(ReadU16(control, 0x60));
        mesh.max_primitives = static_cast<u16>(ReadU16(control, 0x62));
        if (result.stage_byte == static_cast<u32>(ShaderStage::Mesh)) {
            constexpr size_t MeshControlBytes = 0x6a;
            if (control.size() < MeshControlBytes) {
                error = fmt::format("mesh control section is {} bytes, need at least {:#x}",
                                    control.size(), MeshControlBytes);
                return std::nullopt;
            }
            mesh.primitive_header_offset = ReadU32(control, 0x64);
            mesh.primitive_header_bytes = static_cast<u16>(ReadU16(control, 0x68));
        }
    }
    return result;
}

u32 MeshShaderA(const MeshControl& mesh) noexcept {
    const u32 topology = mesh.topology == 1 ? 0U : mesh.topology == 2 ? 1U : 4U;
    return topology | (u32{mesh.max_vertices} & 0x1ff) << 4 |
           (u32{mesh.max_primitives} & 0x3ff) << 13;
}

u32 MeshShaderB(const MeshControl& mesh) noexcept {
    return ((u32{mesh.local_bytes} * 2 + 0xfe) & 0x1ff00) | (u32{mesh.threads} & 0x3f) << 20;
}

u32 MeshInitShader(const MeshControl& task) noexcept {
    const u32 passed = task.output_bytes == 0 ? 1U : (u32{task.output_bytes} + 0x7f) >> 7;
    const u32 local = passed + ((u32{task.local_bytes} + 0x7f) >> 7);
    return (passed & 0x1ff) << 22 | (u32{task.threads} & 0x3f) | ((local << 12) & 0x1ff000);
}

std::array<u32, 3> MeshRegisters(const MeshControl* task, const MeshControl& mesh) noexcept {
    return {MeshShaderA(mesh), MeshShaderB(mesh), task != nullptr ? MeshInitShader(*task) : 0U};
}

std::optional<ShaderStage> StageFromSph(std::span<const u8> code) {
    if (code.size() < sizeof(u32)) {
        return std::nullopt;
    }
    return StageFromSphType((ReadU32(code, 0) >> 10) & 0xF);
}

std::shared_ptr<const PreparedNvnProgram> PrepareNvnProgram(const NvnProgramBinary& binary) {
    using Shader::Ampere::Instruction;
    using Capture = Shader::Ampere::ProgramCaptureState;

    auto result = std::make_shared<PreparedNvnProgram>();
    result->stage = binary.stage;
    const auto fail = [&](std::string message) {
        result->error = std::move(message);
        return std::shared_ptr<const PreparedNvnProgram>{result};
    };

    std::string control_error;
    const auto control = ParseControl(binary.control, control_error);
    if (!control) {
        return fail("control: " + control_error);
    }
    result->control = *control;
    const bool is_compute = binary.stage == ShaderStage::Compute;
    const bool needs_stage_byte = is_compute || binary.stage == ShaderStage::Task ||
                                  binary.stage == ShaderStage::Mesh;
    if (needs_stage_byte ? control->stage_byte != static_cast<u32>(binary.stage)
                         : control->stage_byte >= static_cast<u32>(ShaderStage::Compute)) {
        return fail(fmt::format("control: stage byte {} does not match the requested {} stage",
                                control->stage_byte, StageName(binary.stage)));
    }
    if (binary.code.size() != control->code_bytes) {
        return fail(fmt::format("control: code area is {:#x} bytes, control says {:#x}",
                                binary.code.size(), control->code_bytes));
    }
    const u32 header_bytes = is_compute ? 0 : GraphicsHeaderBytes;
    if (control->inst_bytes < header_bytes ||
        (control->inst_bytes - header_bytes) % sizeof(Instruction) != 0) {
        return fail(fmt::format("control: instruction bytes {:#x} are not {}whole instructions",
                                control->inst_bytes, is_compute ? "" : "header + "));
    }
    const u32 instruction_count = (control->inst_bytes - header_bytes) / sizeof(Instruction);
    if (instruction_count == 0) {
        return fail("control: program has no instructions");
    }
    if (!is_compute) {
        std::memcpy(result->sph.data(), binary.code.data(), SphBytes);
    }
    {
        const size_t c1_words = (control->code_bytes - control->c1_offset) / sizeof(u32);
        result->c1.resize(c1_words);
        if (c1_words != 0) {
            std::memcpy(result->c1.data(), binary.code.data() + control->c1_offset,
                        c1_words * sizeof(u32));
        }
    }
    if (binary.stage == ShaderStage::Mesh && control->mesh.primitive_header_bytes != 0) {
        // The geometry pipeline the SDK binds with the program (MeshControl): the hardware reads
        // that program's SPH, whose output map lists the per-primitive outputs.
        const u64 offset = control->mesh.primitive_header_offset;
        if (control->mesh.primitive_header_bytes < SphBytes ||
            offset + SphBytes > control->code_bytes) {
            return fail(fmt::format("control: per-primitive header of {:#x} bytes at +{:#x} is "
                                    "not a header inside the {:#x}-byte code area",
                                    control->mesh.primitive_header_bytes, offset,
                                    control->code_bytes));
        }
        auto& header = result->primitive_header.emplace();
        std::memcpy(header.data(), binary.code.data() + offset, SphBytes);
    }

    // Capture, as GenericEnvironment::LowerNativeCode does: instruction by instruction until the
    // program completes. The end of the program image plays the part of the end of the mapping:
    // the validation judges what was captured.
    static_assert(sizeof(Instruction) == 4 * sizeof(u32));
    const auto read_instruction = [&](u32 index) {
        Instruction instruction;
        std::memcpy(&instruction, binary.code.data() + header_bytes + index * sizeof(Instruction),
                    sizeof(instruction));
        return instruction;
    };
    // The driver's size footer: the last eight instructions of the instruction area are
    // `MOV RZ, 0x12345678` then `MOV RZ, <inst_bytes>`. It is believed only when the size names
    // exactly the place it sits at; the program ends before it.
    constexpr u32 FooterInstructions = 8;
    constexpr u32 FooterMagic = 0x12345678;
    const auto is_mov_rz = [](const Instruction& instruction, u32 immediate) {
        return instruction.words[0] == 0x00ff7802U && instruction.words[1] == immediate;
    };
    const u32 limit = std::min<u32>(instruction_count, Capture::MaximumInstructions);
    // Instructions before the footer (all of them when there is none): nothing past it is code.
    u32 program_limit = limit;
    std::vector<Instruction> native;
    Capture capture;
    bool complete = false;
    native.reserve(limit);
    for (u32 index = 0; index < limit; ++index) {
        const Instruction instruction = read_instruction(index);
        if (index != 0 && is_mov_rz(native.back(), FooterMagic) &&
            is_mov_rz(instruction, header_bytes + (index - 1 + FooterInstructions) *
                                                      static_cast<u32>(sizeof(Instruction)))) {
            native.pop_back();
            program_limit = index - 1;
            complete = true;
            break;
        }
        native.push_back(instruction);
        const auto status = capture.Observe(instruction, index);
        if (status == Capture::Result::InvalidTarget) {
            result->error_offset = index * static_cast<u32>(sizeof(Instruction));
            result->failed_instruction = index;
            result->error_words = instruction.words;
            return fail(fmt::format("capture: branch outside capture bounds at +{:#x}",
                                    result->error_offset));
        }
        if (status == Capture::Result::Complete) {
            complete = true;
            break;
        }
    }
    if (program_limit == limit && limit >= FooterInstructions) {
        // The capture completed before the footer: find it, so the tail read below stops there.
        const u32 first = limit - FooterInstructions;
        if (is_mov_rz(read_instruction(first), FooterMagic) &&
            is_mov_rz(read_instruction(first + 1), control->inst_bytes)) {
            program_limit = first;
        }
    }
    // The blocks a computed branch selects can lie past the first EXIT: read each branch's table
    // from the constant segment (c1, the only bank whose contents belong to the program) and
    // capture the tail through the furthest target. A table in another bank reads as zeros, which
    // are no targets; the validation below then refuses the branch.
    {
        const auto tables = Shader::Ampere::ResolveCaptureIndirectBranches(
            native, capture,
            [&](u32 index) -> std::optional<Instruction> {
                if (index >= program_limit) {
                    return std::nullopt;
                }
                return read_instruction(index);
            },
            [&](u32 bank, u32 offset) -> u32 {
                const size_t word = offset / sizeof(u32);
                if (bank != ProgramBank || offset % sizeof(u32) != 0 || word >= result->c1.size()) {
                    return 0;
                }
                return result->c1[word];
            });
        for (const auto& table : tables.tables) {
            result->indirect_tables.push_back({table.branch, table.targets});
        }
    }
    result->capture_complete = complete;
    result->native_key = Shader::Ampere::NativeProgramKey(native);
    result->instruction_count = static_cast<u32>(native.size());
    result->source_size =
        header_bytes + static_cast<u32>(native.size() * sizeof(Instruction));
    result->native = std::move(native);

    // The native Ampere front end decides, before any IR exists, whether it translates the
    // program. There is no other front end to fall back to: its reason is the program's error.
    try {
        NvnEnvironment env{result};
        const Shader::Ampere::ProgramInfo info =
            Shader::Ampere::ValidateProgram(env, result->native);
        result->texture_bound = info.texture_buffer;
    } catch (const Shader::Ampere::UnsupportedError& e) {
        result->error_offset = e.index * static_cast<u32>(sizeof(Instruction));
        if (e.index < result->native.size()) {
            result->failed_instruction = e.index;
            result->error_words = result->native[e.index].words;
        }
        const std::string reason = fmt::format("native: instruction {:#x}: {}",
                                               result->error_offset, e.what());
        return fail(complete ? reason
                             : "capture: program never completes within its instruction bytes; " +
                                   reason);
    }
    return result;
}

NvnEnvironment::NvnEnvironment(std::shared_ptr<const PreparedNvnProgram> prepared_,
                               NvnEnvironmentOptions options_)
    : prepared{std::move(prepared_)}, options{std::move(options_)} {
    const bool is_compute = prepared->stage == ShaderStage::Compute;
    stage = ToRecompilerStage(prepared->stage);
    is_proprietary_driver = true;
    static_assert(sizeof(sph) == SphBytes);
    std::memcpy(&sph, prepared->sph.data(), sizeof(sph));
    texture_bound = prepared->texture_bound;
    if (prepared->stage == ShaderStage::Geometry) {
        // The mask is the post-VTG skip mask of the Maxwell channel (a set bit skips an
        // attribute), which NVN does not carry; all-zero would make every attribute of the
        // attribute space (instance id, point coordinate, front face, ...) a pass-through input
        // and declare built-ins the stage cannot read. A pass-through geometry stage forwards the
        // position and the generic attributes.
        gp_passthrough_mask.fill(~0u);
        for (u32 attribute = 28; attribute < 160; ++attribute) {
            gp_passthrough_mask[attribute / 32] &= ~(1u << (attribute % 32));
        }
    }
    if (prepared->stage == ShaderStage::Task || prepared->stage == ShaderStage::Mesh) {
        // What the SDK writes to SET_MESH_SHADER_A/B and SET_MESH_INIT_SHADER for the pipeline,
        // and the per-primitive output map: the SPH of the pass-through geometry program the SDK
        // binds with a mesh program whose control section names one (none: no per-primitive
        // outputs, the geometry pipeline is disabled).
        mesh_registers = options.mesh_registers;
        if (prepared->primitive_header) {
            static_assert(sizeof(mesh_primitive_header) == SphBytes);
            std::memcpy(&mesh_primitive_header, prepared->primitive_header->data(), SphBytes);
        }
    }
    if (is_compute) {
        start_address = SphBytes;
        local_memory_size =
            options.compute_local_memory_size.value_or(prepared->control.LocalMemoryBytes());
        shared_memory_size =
            options.compute_shared_memory_size.value_or(prepared->control.shared_memory_bytes);
        workgroup_size = prepared->control.thread_group;
    } else {
        start_address = 0;
        local_memory_size =
            static_cast<u32>(sph.LocalMemorySize()) + sph.common3.shader_local_memory_crs_size;
    }
}

NvnEnvironment::~NvnEnvironment() = default;

u64 NvnEnvironment::ReadInstruction(u32 address) {
    throw Shader::LogicError("nvn-hle programs are native Ampere: no Maxwell instruction at {:#x}",
                             address);
}

std::span<const Shader::Ampere::Instruction> NvnEnvironment::NativeProgram() const {
    return prepared->native;
}

std::span<const u32> NvnEnvironment::NativeIndirectTargetsAt(u32 index) const {
    for (const NvnIndirectTable& table : prepared->indirect_tables) {
        if (table.branch == index) {
            return table.targets;
        }
    }
    return NativeIndirectTargets();
}

u32 NvnEnvironment::NativeSourceBase() const {
    return prepared->stage == ShaderStage::Compute ? 0 : GraphicsHeaderBytes;
}

bool NvnEnvironment::IsCbufStorageBufferPointer(u32 cbuf_index, u32 cbuf_offset) {
    if (cbuf_index != DriverBank) {
        return false;
    }
    if (prepared->stage != ShaderStage::Compute) {
        return StorageBufferSlot(prepared->stage, cbuf_offset).has_value();
    }
    const Shader::NvnBufferWindow window = NvnComputeBufferWindow();
    return window.active && cbuf_offset >= window.base &&
           (cbuf_offset - window.base) % StorageBufferStride == 0 &&
           (cbuf_offset - window.base) / StorageBufferStride < window.count;
}

bool NvnEnvironment::IsCbufIndirectStorageBufferPointer(u32, u32, u32) {
    return false;
}

u32 NvnEnvironment::ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) {
    if (cbuf_index == ProgramBank) {
        const auto& c1 = prepared->c1;
        const size_t word = cbuf_offset / sizeof(u32);
        const u32 shift = (cbuf_offset % sizeof(u32)) * 8;
        const u32 low = word < c1.size() ? c1[word] : 0;
        if (shift == 0) {
            return low;
        }
        const u32 high = word + 1 < c1.size() ? c1[word + 1] : 0;
        return (low >> shift) | (high << (32 - shift));
    }
    bool handle_word = false;
    if (cbuf_index == HandleBank) {
        handle_word = cbuf_offset >= CombinedHandleBase && cbuf_offset < HandleBankEnd;
    } else if (cbuf_index >= FirstUniformBank) {
        handle_word = true;
    } else if (cbuf_index == DriverBank && prepared->control.has_compute_counts) {
        const auto& layout = prepared->control.compute;
        handle_word =
            (cbuf_offset >= layout.texture_offset &&
             cbuf_offset < layout.texture_offset + layout.texture_count * HandleStride) ||
            (cbuf_offset >= layout.image_offset &&
             cbuf_offset < layout.image_offset + layout.image_count * HandleStride);
    }
    if (!handle_word) {
        return 0;
    }
    const u64 word_key = (u64{cbuf_index} << 32) | cbuf_offset;
    u32 key;
    if (const auto it = record_index.find(word_key); it != record_index.end()) {
        key = it->second;
    } else if (records.size() >= UnknownHandleKey - 1) {
        key_overflow = true;
        key = UnknownHandleKey; // never a record: RecordOf() yields nullptr for it
    } else {
        records.push_back({cbuf_index, cbuf_offset});
        key = static_cast<u32>(records.size());
        record_index.emplace(word_key, key);
    }
    // A separate sampler word holds `id << 20`, so it ORs cleanly with a texture word.
    if (cbuf_index == HandleBank && cbuf_offset >= SeparateSamplerBase) {
        return SeparateSamplerWord(key);
    }
    return key;
}

const NvnHandleRecord* NvnEnvironment::RecordOf(u32 raw_handle) const noexcept {
    const u32 key = TextureIdOf(raw_handle);
    if (key == 0 || key > records.size()) {
        return nullptr;
    }
    return &records[key - 1];
}

NvnTextureInfo NvnEnvironment::InfoOf(u32 raw_handle) const {
    if (options.texture_info) {
        if (const NvnHandleRecord* record = RecordOf(raw_handle)) {
            return options.texture_info(*record);
        }
    }
    return {};
}

Shader::TextureType NvnEnvironment::ReadTextureType(u32 raw_handle) {
    return InfoOf(raw_handle).type;
}

Shader::TexturePixelFormat NvnEnvironment::ReadTexturePixelFormat(u32 raw_handle) {
    return InfoOf(raw_handle).format;
}

bool NvnEnvironment::IsTexturePixelFormatInteger(u32 raw_handle) {
    return InfoOf(raw_handle).is_integer;
}

u32 NvnEnvironment::ReadViewportTransformState() {
    return 1;
}

u32 NvnEnvironment::TextureBoundBuffer() const {
    return texture_bound != ~0U ? texture_bound : HandleBank;
}

u32 NvnEnvironment::LocalMemorySize() const {
    return local_memory_size;
}

u32 NvnEnvironment::SharedMemorySize() const {
    return shared_memory_size;
}

std::array<u32, 3> NvnEnvironment::WorkgroupSize() const {
    return workgroup_size;
}

bool NvnEnvironment::HasHLEMacroState() const {
    // Vertex programs read the draw's base instance and task and mesh programs the draw index of
    // a multi-draw from words the draw macros write into the driver bank.
    return stage == Shader::Stage::VertexB || stage == Shader::Stage::Task ||
           stage == Shader::Stage::Mesh;
}

std::optional<Shader::ReplaceConstant> NvnEnvironment::GetReplaceConstBuffer(u32 bank,
                                                                              u32 offset) {
    if (bank != DriverBank) {
        return std::nullopt;
    }
    // The instanced draw macros write the base instance to the driver bank
    // (DriverBaseInstanceOffset); the host draw passes it as firstInstance (an indirect draw: the
    // record's baseInstance word), which Vulkan hands to the vertex stage as BaseInstance.
    if (stage == Shader::Stage::VertexB && offset == DriverBaseInstanceOffset) {
        return Shader::ReplaceConstant::BaseInstance;
    }
    // The mesh-task multi-draw macros write the draw index to the driver bank before every draw
    // (DriverDrawIdOffset); Vulkan hands it to the task and mesh stages as DrawIndex (0 for a
    // single draw).
    if ((stage == Shader::Stage::Task || stage == Shader::Stage::Mesh) &&
        offset == DriverDrawIdOffset) {
        return Shader::ReplaceConstant::DrawID;
    }
    return std::nullopt;
}

Shader::NvnBufferWindow NvnEnvironment::NvnComputeBufferWindow() const {
    if (prepared->stage != ShaderStage::Compute || !prepared->control.has_compute_counts) {
        return {};
    }
    const ComputeDriverLayout& layout = prepared->control.compute;
    // Uniform entries are followed by storage entries; cover both (and anything between them).
    u32 begin = ~0U;
    u32 end = 0;
    const auto add = [&](u32 offset, u32 count) {
        if (count != 0) {
            begin = (std::min)(begin, offset);
            end = (std::max)(end, offset + count * StorageBufferStride);
        }
    };
    add(layout.uniform_offset, layout.uniform_count);
    add(layout.storage_offset, layout.storage_count);
    if (end == 0) {
        return {0, 0, true};
    }
    return {begin, (end - begin) / StorageBufferStride, true};
}

void NvnEnvironment::Dump(u64, u64) {}

Shader::IR::Program TranslateNvnStage(NvnShaderPools& pools, NvnEnvironment& env,
                                      const Shader::HostTranslateInfo& host_info) {
    // As on the emulated GPU path: graphics CFG starts after the SPH (StartAddress is 0 there),
    // compute starts at StartAddress, which already is the end of the SPH. Either way 0x50, the
    // location of native instruction 0 (Shader::Ampere::LocationOf), which is where the CFG of a
    // native Ampere environment must start: Flow::CFG builds it from env.NativeProgram().
    const u32 cfg_offset = env.ShaderStage() == Shader::Stage::Compute
                               ? env.StartAddress()
                               : static_cast<u32>(env.StartAddress() + sizeof(Shader::ProgramHeader));
    Shader::Maxwell::Flow::CFG cfg(env, pools.flow_block, cfg_offset, false);
    return Shader::Maxwell::TranslateProgram(pools.inst, pools.block, env, cfg, host_info);
}

Shader::Profile MakeReferenceProfile() {
    Shader::Profile profile{};
    profile.supported_spirv = 0x00010600;
    profile.unified_descriptor_binding = true;
    profile.support_descriptor_aliasing = false;
    profile.support_int8 = true;
    profile.support_uniform_and_storage_buffer_8bit = true;
    profile.support_storage_buffer_8bit = true;
    profile.support_int16 = true;
    profile.support_uniform_and_storage_buffer_16bit = true;
    profile.support_storage_buffer_16bit = true;
    profile.support_int64 = true;
    profile.support_float_controls = true;
    profile.support_separate_denorm_behavior = true;
    profile.support_separate_rounding_mode = true;
    profile.support_fp16_denorm_preserve = true;
    profile.support_fp32_denorm_preserve = true;
    profile.support_fp16_denorm_flush = true;
    profile.support_fp32_denorm_flush = true;
    profile.support_fp16_signed_zero_nan_preserve = true;
    profile.support_fp32_signed_zero_nan_preserve = true;
    profile.support_fp64_signed_zero_nan_preserve = true;
    profile.support_explicit_workgroup_layout = false;
    profile.support_shader_quad_control = false;
    profile.support_quad_shuffles = true;
    profile.support_vote = true;
    profile.support_shuffle_relative = true;
    profile.supported_subgroup_stages = 0x7F;
    profile.support_viewport_index_layer_non_geometry = true;
    profile.support_typeless_image_loads = true;
    profile.support_demote_to_helper_invocation = true;
    profile.support_int64_atomics = true;
    profile.support_shared_int64_atomics = true;
    profile.support_derivative_control = true;
    profile.support_native_ndc = true;
    profile.support_scaled_attributes = true;
    profile.support_multi_viewport = true;
    profile.support_geometry_streams = true;
    profile.max_subgroup_size = 32;
    profile.min_ssbo_alignment = 16;
    profile.max_user_clip_distances = 8;
    return profile;
}

Shader::HostTranslateInfo MakeReferenceHostInfo() {
    Shader::HostTranslateInfo info{};
    info.min_ssbo_alignment = 16;
    info.support_float64 = true;
    info.support_float16 = true;
    info.support_int64 = true;
    info.support_snorm_render_buffer = true;
    info.support_viewport_index_layer = true;
    info.support_geometry_shader_passthrough = false; // matches the profile: no NV extension
    info.support_conditional_barrier = true;
    info.skip_rescaling_pass = !Settings::values.resolution_info.active; // host_translate_info.h
    info.ApplyDescriptorLimitPolicy();
    return info;
}

} // namespace VideoCommon::Nvn
