// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Shader stages from a title's own package (ahead-of-time pipelines, stage 3). The container
// layouts below are the ones Longshot's contributors documented for their ROM shader enumerator
// (nx2/bake_ds/romenum.py and nx2/shader_census.py); this is an independent reader of the same
// formats. A stage is the same two byte runs in the package and in guest memory at program
// creation (the control section and the code area it sizes), so the renderer's content hash of
// an enumerated stage equals the hash of the program the game creates from it.

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <zstd.h>

#include "common/logging.h"
#include "core/file_sys/vfs/vfs.h"
#include "core/hle/nvn/package_shaders.h"

namespace Core::NvnHle {

namespace {

constexpr u32 CONTROL_MAGIC = 0x98761234u;
constexpr u32 ZSTD_FRAME_MAGIC = 0xFD2FB528u;
constexpr u32 ZSTD_DICT_MAGIC = 0xEC30A437u;
/// Bytes of a control section the recorder copies (recorder.cpp ReadControlSection).
constexpr size_t CONTROL_COPY_BYTES = 0x100;
/// Shortest control section accepted (the NVN2 section is 0xd8 bytes in every container seen).
constexpr size_t CONTROL_MIN_BYTES = 0xd8;
constexpr u32 MAX_STAGE_INDEX = 7;
constexpr size_t MAX_FILE_BYTES = size_t{1} << 30;
constexpr size_t MAX_INFLATED_BYTES = size_t{768} << 20;
constexpr size_t HEAD_BYTES = 64;
constexpr size_t HEAD_INPUT_BYTES = size_t{64} << 10;
constexpr size_t HEAD_RETRY_INPUT_BYTES = size_t{160} << 10;
constexpr int MAX_DEPTH = 6;

template <typename T>
bool Get(std::span<const u8> data, size_t offset, T& out) {
    if (offset > data.size() || data.size() - offset < sizeof(T)) {
        return false;
    }
    std::memcpy(&out, data.data() + offset, sizeof(T));
    return true;
}

bool HasMagic(std::span<const u8> data, size_t offset, std::string_view magic) {
    return offset <= data.size() && data.size() - offset >= magic.size() &&
           std::memcmp(data.data() + offset, magic.data(), magic.size()) == 0;
}

u32 Word(std::span<const u8> data, size_t offset = 0) {
    u32 value = 0;
    (void)Get(data, offset, value);
    return value;
}

enum class Kind { None, Zstd, Sarc, Bnsh, Bahs, Ndsr, Ndve };

Kind Sniff(std::span<const u8> head) {
    if (head.size() < 4) {
        return Kind::None;
    }
    if (Word(head) == ZSTD_FRAME_MAGIC) {
        return Kind::Zstd;
    }
    if (HasMagic(head, 0, "SARC")) {
        return Kind::Sarc;
    }
    if (HasMagic(head, 0, "BNSH")) {
        return Kind::Bnsh;
    }
    if (HasMagic(head, 0, "BAHS")) {
        return Kind::Bahs;
    }
    if (HasMagic(head, 0, "NDSR")) {
        return Kind::Ndsr;
    }
    if (HasMagic(head, 0, "NDVE")) {
        return Kind::Ndve;
    }
    return Kind::None;
}

bool IsContainer(Kind kind) {
    return kind == Kind::Sarc || kind == Kind::Bnsh || kind == Kind::Bahs || kind == Kind::Ndsr ||
           kind == Kind::Ndve;
}

std::string Lower(std::string_view text) {
    std::string out{text};
    std::ranges::transform(out, out.begin(), [](char c) {
        return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c;
    });
    return out;
}

/// The name without a trailing ".zs" (lower case).
std::string Inner(std::string_view name) {
    std::string lower = Lower(name);
    if (lower.ends_with(".zs")) {
        lower.resize(lower.size() - 3);
    }
    return lower;
}

/// Names of files that hold BNSH containers somewhere inside rather than at offset 0 (shader
/// archives, effect and particle sets): searched for embedded BNSH.
bool HintsEmbeddedShaders(std::string_view inner) {
    static constexpr std::array<std::string_view, 5> hints{".bfsha", ".bnsh", ".vsetb", ".esetb",
                                                           ".ptcl"};
    return std::ranges::any_of(hints, [inner](std::string_view hint) {
        return inner.find(hint) != std::string_view::npos;
    });
}

class Inflater {
public:
    Inflater() : dctx{ZSTD_createDCtx()} {}
    ~Inflater() {
        for (auto& [id, dict] : dicts) {
            ZSTD_freeDDict(dict);
        }
        ZSTD_freeDCtx(dctx);
    }
    Inflater(const Inflater&) = delete;
    Inflater& operator=(const Inflater&) = delete;

    enum class Result { Ok, Failed, TooLarge };

    bool AddDictionary(std::span<const u8> bytes) {
        if (Word(bytes) != ZSTD_DICT_MAGIC) {
            return false;
        }
        const unsigned id = ZSTD_getDictID_fromDict(bytes.data(), bytes.size());
        if (id == 0 || dicts.contains(id)) {
            return false;
        }
        ZSTD_DDict* const dict = ZSTD_createDDict(bytes.data(), bytes.size());
        if (dict == nullptr) {
            return false;
        }
        dicts.emplace(id, dict);
        return true;
    }

    [[nodiscard]] size_t Dictionaries() const {
        return dicts.size();
    }

    /// Inflates the zstd frame(s) in `src` into `out`. `stop_after`: stop once that many bytes
    /// are out (a head); a head may also come from a truncated input.
    Result Inflate(std::span<const u8> src, std::vector<u8>& out, size_t stop_after = SIZE_MAX) {
        out.clear();
        if (dctx == nullptr) {
            return Result::Failed;
        }
        ZSTD_DCtx_reset(dctx, ZSTD_reset_session_and_parameters);
        const unsigned dict_id = ZSTD_getDictID_fromFrame(src.data(), src.size());
        if (dict_id != 0) {
            const auto it = dicts.find(dict_id);
            if (it == dicts.end()) {
                return Result::Failed;
            }
            ZSTD_DCtx_refDDict(dctx, it->second);
        }
        const bool head = stop_after != SIZE_MAX;
        const unsigned long long known = ZSTD_getFrameContentSize(src.data(), src.size());
        if (!head && known != ZSTD_CONTENTSIZE_UNKNOWN && known != ZSTD_CONTENTSIZE_ERROR) {
            if (known > MAX_INFLATED_BYTES) {
                return Result::TooLarge;
            }
            out.reserve(static_cast<size_t>(known));
        }
        ZSTD_inBuffer in{src.data(), src.size(), 0};
        constexpr size_t CHUNK = size_t{1} << 20;
        while (true) {
            const size_t old = out.size();
            const size_t want = std::min(CHUNK, stop_after - old);
            out.resize(old + want);
            ZSTD_outBuffer ob{out.data() + old, want, 0};
            const size_t ret = ZSTD_decompressStream(dctx, &ob, &in);
            out.resize(old + ob.pos);
            if (ZSTD_isError(ret)) {
                return head && !out.empty() ? Result::Ok : Result::Failed;
            }
            if (out.size() >= stop_after) {
                return Result::Ok;
            }
            if (out.size() > MAX_INFLATED_BYTES) {
                out.clear();
                return Result::TooLarge;
            }
            if (in.pos == in.size && ob.pos < want) {
                // Input used up and the output not full: done (ret 0), or a truncated frame.
                if (ret == 0 || head) {
                    return out.empty() && !head ? Result::Failed : Result::Ok;
                }
                return Result::Failed;
            }
        }
    }

private:
    ZSTD_DCtx* dctx{};
    std::unordered_map<unsigned, ZSTD_DDict*> dicts;
};

class Scanner {
public:
    Scanner(const PackageStageVisitor& visit_, PackageShaderScan& scan_)
        : visit{visit_}, scan{scan_} {}

    Inflater inflater;

    void LoadDictionaries(const FileSys::VirtualFile& file) {
        std::vector<u8> data;
        if (!ReadWhole(file, data)) {
            return;
        }
        std::vector<u8> inflated;
        if (Sniff(data) == Kind::Zstd) {
            if (inflater.Inflate(data, inflated) != Inflater::Result::Ok) {
                return;
            }
            scan.bytes_inflated += inflated.size();
            data = std::move(inflated);
        }
        if (Sniff(data) == Kind::Sarc) {
            ForEachSarcMember(data, [this](std::span<const u8> member, std::string_view) {
                inflater.AddDictionary(member);
            });
        } else {
            inflater.AddDictionary(data);
        }
        scan.dictionaries = inflater.Dictionaries();
    }

    void File(const FileSys::VirtualFile& file, const std::string& path) {
        if (stopped) {
            return;
        }
        const size_t size = file->GetSize();
        if (size < 8) {
            return;
        }
        const std::string lower_path = Lower(path);
        const bool traced = std::ranges::any_of(trace, [&lower_path](const std::string& part) {
            return lower_path.find(part) != std::string::npos;
        });
        const std::string inner = Inner(file->GetName());
        const bool hinted = HintsEmbeddedShaders(inner);
        std::vector<u8> head = file->ReadBytes(std::min(size, HEAD_INPUT_BYTES));
        scan.bytes_read += head.size();
        Kind kind = Sniff(head);
        std::vector<u8> content;
        const auto note = [&](std::string_view what) {
            if (traced) {
                const auto magic = [](std::span<const u8> bytes) {
                    std::string text;
                    for (size_t i = 0; i < std::min<size_t>(bytes.size(), 8); ++i) {
                        const char c = static_cast<char>(bytes[i]);
                        text += c >= 0x20 && c < 0x7f ? c : '.';
                    }
                    return text;
                };
                LOG_INFO(Core, "[nvn] package shader scan trace: {} ({} bytes): head '{}', inflated '{}': {}",
                         path, size, magic(head), magic(content), what);
            }
        };
        if (kind == Kind::Zstd && !hinted) {
            // Only the first bytes of the content decide whether the whole file is read. A zstd
            // block (up to 128 KiB compressed) gives no output until all of it is in: a head that
            // inflated to too little is read again with room for a whole block.
            Inflater::Result result = inflater.Inflate(head, content, HEAD_BYTES);
            if (result == Inflater::Result::Ok && content.size() < HEAD_BYTES &&
                head.size() < size) {
                head = file->ReadBytes(std::min(size, HEAD_RETRY_INPUT_BYTES));
                scan.bytes_read += head.size();
                result = inflater.Inflate(head, content, HEAD_BYTES);
            }
            if (result != Inflater::Result::Ok) {
                ++scan.zstd_undecodable;
                note("zstd head undecodable");
                return;
            }
            if (!IsContainer(Sniff(content))) {
                note("not a shader container");
                return;
            }
        } else if (kind != Kind::Zstd && !IsContainer(kind) && !hinted) {
            note("not a shader container");
            return;
        }
        std::vector<u8> data;
        if (head.size() == size) {
            data = std::move(head);
        } else {
            if (!traced) {
                head = {};
            }
            if (!ReadWhole(file, data)) {
                note("read failed");
                return;
            }
        }
        ++scan.files_parsed;
        const u64 before = scan.stages;
        Blob(data, file->GetName(), 0);
        note(fmt::format("parsed, {} stages", scan.stages - before));
    }

    /// EDEN_NX2_NVN_PREPARE_TRACE ("a;b"): log the files whose path contains one of the parts.
    std::vector<std::string> trace;
    bool stopped{};

private:
    const PackageStageVisitor& visit;
    PackageShaderScan& scan;

    bool ReadWhole(const FileSys::VirtualFile& file, std::vector<u8>& out) {
        const size_t size = file->GetSize();
        if (size == 0 || size > MAX_FILE_BYTES) {
            ++scan.skipped_files;
            return false;
        }
        out = file->ReadBytes(size);
        scan.bytes_read += out.size();
        if (out.size() != size) {
            ++scan.skipped_files;
            return false;
        }
        return true;
    }

    template <typename Fn>
    bool ForEachSarcMember(std::span<const u8> data, Fn&& fn) {
        u16 bom = 0;
        u32 start = 0;
        u16 count = 0;
        if (!Get(data, 0x06, bom) || bom != 0xFEFF || !Get(data, 0x0C, start) ||
            !HasMagic(data, 0x14, "SFAT") || !Get(data, 0x1A, count)) {
            return false;
        }
        const size_t names = 0x20 + size_t{count} * 16;
        const bool has_names = HasMagic(data, names, "SFNT");
        for (u32 k = 0; k < count; ++k) {
            u32 attr = 0;
            u32 begin = 0;
            u32 end = 0;
            const size_t node = 0x20 + size_t{k} * 16;
            if (!Get(data, node + 4, attr) || !Get(data, node + 8, begin) ||
                !Get(data, node + 12, end)) {
                return false;
            }
            const size_t first = size_t{start} + begin;
            const size_t last = size_t{start} + end;
            if (end < begin || last > data.size()) {
                ++scan.skipped_containers;
                continue;
            }
            std::string_view name;
            if (has_names && (attr >> 24) != 0) {
                const size_t at = names + 8 + size_t{attr & 0xFFFFFF} * 4;
                if (at < data.size()) {
                    const auto* text = reinterpret_cast<const char*>(data.data() + at);
                    name = std::string_view{text, strnlen(text, data.size() - at)};
                }
            }
            fn(data.subspan(first, last - first), name);
        }
        return true;
    }

    void Count(const char* kind) {
        ++scan.containers[kind];
    }

    /// Stage at control offset `c` with its code at `code` (both in `data`).
    bool Emit(std::span<const u8> data, size_t c, size_t code) {
        if (stopped || c > data.size() || data.size() - c < CONTROL_MIN_BYTES ||
            Word(data, c) != CONTROL_MAGIC) {
            return false;
        }
        u16 stage = 0;
        u32 code_size = 0;
        if (!Get(data, c + 0x56, stage) || stage > MAX_STAGE_INDEX || !Get(data, c + 0x44, code_size) ||
            code_size == 0 || code > data.size() || data.size() - code < code_size) {
            return false;
        }
        std::array<u8, CONTROL_COPY_BYTES> control{};
        std::memcpy(control.data(), data.data() + c, std::min(CONTROL_COPY_BYTES, data.size() - c));
        ++scan.stages;
        if (!visit(stage, control, data.subspan(code, code_size))) {
            stopped = true;
            scan.stopped = true;
        }
        return true;
    }

    bool Bnsh(std::span<const u8> data) {
        u32 size = 0;
        if (!HasMagic(data, 0, "BNSH") || data.size() < 0x88 || data[12] != 0xFF ||
            data[13] != 0xFE || !HasMagic(data, 96, "grsc") || !Get(data, 28, size)) {
            return false;
        }
        data = data.first(std::min<size_t>(size, data.size()));
        u32 count = 0;
        u64 variations = 0;
        if (!Get(data, 124, count) || !Get(data, 128, variations)) {
            return false;
        }
        Count("bnsh");
        for (u32 i = 0; i < count && !stopped; ++i) {
            u64 program = 0;
            if (variations > data.size() ||
                !Get(data, static_cast<size_t>(variations) + size_t{i} * 64 + 16, program)) {
                break;
            }
            if (program == 0 || program + 80 > data.size()) {
                continue;
            }
            for (size_t rel = 8; rel <= 72; rel += 8) {
                u64 offset = 0;
                if (!Get(data, static_cast<size_t>(program) + rel, offset) || offset == 0 ||
                    offset + 32 > data.size()) {
                    continue;
                }
                u64 control_offset = 0;
                u64 code_offset = 0;
                u32 code_size = 0;
                u32 control_size = 0;
                const size_t at = static_cast<size_t>(offset) + 8;
                if (!Get(data, at, control_offset) || !Get(data, at + 8, code_offset) ||
                    !Get(data, at + 16, code_size) || !Get(data, at + 20, control_size)) {
                    continue;
                }
                if (control_size < 0x60 || control_size > 0x1000 ||
                    control_offset + control_size > data.size() ||
                    code_offset + code_size > data.size()) {
                    continue;
                }
                Emit(data, static_cast<size_t>(control_offset), static_cast<size_t>(code_offset));
            }
        }
        return true;
    }

    /// Every BNSH inside `data` (shader archives, effect sets).
    void EmbeddedBnsh(std::span<const u8> data) {
        static constexpr std::string_view MAGIC{"BNSH"};
        size_t pos = 0;
        while (!stopped && pos + 100 <= data.size()) {
            const auto it = std::search(data.begin() + static_cast<std::ptrdiff_t>(pos), data.end(),
                                        MAGIC.begin(), MAGIC.end());
            if (it == data.end()) {
                return;
            }
            pos = static_cast<size_t>(it - data.begin());
            u32 size = 0;
            if (HasMagic(data, pos + 96, "grsc") && Get(data, pos + 28, size) && size >= 0x88) {
                const auto sub = data.subspan(pos, std::min<size_t>(size, data.size() - pos));
                if (Bnsh(sub)) {
                    pos += sub.size();
                    continue;
                }
                ++scan.skipped_containers;
            }
            pos += 4;
        }
    }

    bool Bahs(std::span<const u8> data) {
        u32 base = 0;
        u32 blob = 0;
        u32 count = 0;
        if (!Get(data, 0x14, base) || !Get(data, 0x1C, blob) || !Get(data, size_t{base} + 0x154, count)) {
            return false;
        }
        Count("bahs");
        size_t at = size_t{base} + 0x158;
        for (u32 k = 0; k < count && !stopped; ++k) {
            u32 step = 0;
            u64 offset = 0;
            s32 rel = 0;
            if (!Get(data, at, step) || step < 32 || !Get(data, at + 16, offset) ||
                !Get(data, at + 28, rel)) {
                return false;
            }
            const s64 control = static_cast<s64>(at) + 16 + rel;
            if (control < 0 || offset > data.size() ||
                !Emit(data, static_cast<size_t>(control), size_t{blob} + static_cast<size_t>(offset))) {
                return false;
            }
            at += step;
        }
        return true;
    }

    bool Ndsr(std::span<const u8> data, size_t base) {
        u32 count = 0;
        if (!HasMagic(data, base, "NDSR") || !Get(data, base + 0x0C, count) || count > (1u << 16)) {
            return false;
        }
        Count("ndsr");
        u32 prev_offset = 0;
        u32 prev_type = 0;
        for (u32 i = 0; i < count && !stopped; ++i) {
            u32 offset = 0;
            u32 type = 0;
            if (!Get(data, base + 0x14 + size_t{i} * 8, offset) ||
                !Get(data, base + 0x18 + size_t{i} * 8, type)) {
                return false;
            }
            if (i != 0) {
                // A control item (type 0, 3 or 6) followed by its code item (type + 1).
                const u32 kind = prev_type & 15;
                if ((kind == 0 || kind == 3 || kind == 6) && (type & 15) == kind + 1) {
                    Emit(data, base + prev_offset, base + offset);
                }
            }
            prev_offset = offset;
            prev_type = type;
        }
        return true;
    }

    bool Ndve(std::span<const u8> data) {
        Count("ndve");
        if (HasMagic(data, 0x28, "NDSR")) {
            Ndsr(data, 0x28);
        }
        // Stages outside the NDSR: a control section whose code is located by a small table
        // entry {u32 to_control, u32 to_code, u32 code_size, u32 0} before it.
        for (size_t c = 0; c + CONTROL_MIN_BYTES <= data.size() && !stopped; c += 4) {
            if (Word(data, c) != CONTROL_MAGIC) {
                continue;
            }
            u32 end = 0;
            u16 stage = 0;
            if (!Get(data, c + 0x44, end) || !Get(data, c + 0x56, stage) || stage > MAX_STAGE_INDEX) {
                continue;
            }
            for (size_t e = 8; e + 8 <= data.size(); e += 4) {
                if (Word(data, e) != end || Word(data, e + 4) != 0) {
                    continue;
                }
                const size_t entry = e - 8;
                const u32 to_control = Word(data, entry);
                const u32 to_code = Word(data, entry + 4);
                if (to_control == 0 || to_control > c || (c - to_control) % 4 != 0 ||
                    c + size_t{to_code} + end > data.size()) {
                    continue;
                }
                const size_t table = c - to_control;
                const u32 n = Word(data, table);
                const u32 blobs = Word(data, table + 4);
                const size_t first = table + 8 + size_t{blobs} * 8;
                if (n > 0 && n < 256 && blobs < 16 && entry >= first && (entry - first) % 16 == 0 &&
                    (entry - first) / 16 < n) {
                    Emit(data, c, c + to_code);
                    break;
                }
            }
        }
        return true;
    }

public:
    void Blob(std::span<const u8> data, std::string_view name, int depth) {
        if (stopped || depth > MAX_DEPTH) {
            return;
        }
        const std::string inner = Inner(name);
        const Kind kind = Sniff(data);
        switch (kind) {
        case Kind::Zstd: {
            const bool hinted = HintsEmbeddedShaders(inner);
            std::vector<u8> content;
            if (!hinted) {
                if (inflater.Inflate(data, content, HEAD_BYTES) != Inflater::Result::Ok ||
                    !IsContainer(Sniff(content))) {
                    return;
                }
            }
            const Inflater::Result result = inflater.Inflate(data, content);
            if (result != Inflater::Result::Ok) {
                ++scan.skipped_files;
                return;
            }
            scan.bytes_inflated += content.size();
            Blob(content, inner, depth + 1);
            return;
        }
        case Kind::Sarc:
            Count("sarc");
            if (!ForEachSarcMember(data, [this, depth](std::span<const u8> member, std::string_view n) {
                    Blob(member, n, depth + 1);
                })) {
                ++scan.skipped_containers;
            }
            return;
        case Kind::Bnsh:
            EmbeddedBnsh(data);
            return;
        case Kind::Bahs:
            if (!Bahs(data)) {
                ++scan.skipped_containers;
            }
            return;
        case Kind::Ndsr:
            if (!Ndsr(data, 0)) {
                ++scan.skipped_containers;
            }
            return;
        case Kind::Ndve:
            Ndve(data);
            return;
        case Kind::None:
            if (HintsEmbeddedShaders(inner)) {
                Count("embedded");
                EmbeddedBnsh(data);
            }
            return;
        }
    }
};

void Collect(const FileSys::VirtualDir& dir, const std::string& prefix,
             std::vector<std::pair<std::string, FileSys::VirtualFile>>& out, int depth) {
    if (!dir || depth > 64) {
        return;
    }
    for (const auto& file : dir->GetFiles()) {
        out.emplace_back(prefix + file->GetName(), file);
    }
    for (const auto& sub : dir->GetSubdirectories()) {
        Collect(sub, prefix + sub->GetName() + "/", out, depth + 1);
    }
}

} // namespace

std::string PackageShaderScan::Summary() const {
    std::string kinds;
    for (const auto& [kind, count] : containers) {
        kinds += fmt::format("{}{} {}", kinds.empty() ? "" : ", ", kind, count);
    }
    return fmt::format("{} files ({} read whole), {:.1f} MB read, {:.1f} MB inflated, {} stages, "
                       "containers: {}; {} files and {} containers skipped, {} zstd heads undecodable, "
                       "{} zstd dictionaries, {:.1f} s{}",
                       files, files_parsed, static_cast<double>(bytes_read) / 1e6,
                       static_cast<double>(bytes_inflated) / 1e6, stages,
                       kinds.empty() ? "none" : kinds, skipped_files, skipped_containers,
                       zstd_undecodable, dictionaries, seconds, stopped ? " (stopped)" : "");
}

PackageShaderScan ScanPackageShaders(const FileSys::VirtualDir& root,
                                     const PackageStageVisitor& visit) {
    PackageShaderScan scan;
    const auto start = std::chrono::steady_clock::now();
    std::vector<std::pair<std::string, FileSys::VirtualFile>> files;
    Collect(root, "", files, 0);
    scan.files = files.size();
    Scanner scanner{visit, scan};
    if (const char* trace = std::getenv("EDEN_NX2_NVN_PREPARE_TRACE"); trace != nullptr) {
        std::string_view parts{trace};
        while (!parts.empty()) {
            const size_t cut = parts.find(';');
            const std::string part = Lower(parts.substr(0, cut));
            if (!part.empty()) {
                scanner.trace.push_back(part);
            }
            parts = cut == std::string_view::npos ? std::string_view{} : parts.substr(cut + 1);
        }
    }
    // The title's zstd dictionaries first (Nintendo's "ZsDic" pack convention).
    for (const auto& [path, file] : files) {
        if (Lower(file->GetName()).find("zsdic") != std::string::npos) {
            scanner.LoadDictionaries(file);
        }
    }
    auto last_note = start;
    size_t done = 0;
    for (const auto& [path, file] : files) {
        try {
            scanner.File(file, path);
        } catch (const std::exception& e) {
            ++scan.skipped_files;
            LOG_WARNING(Core, "[nvn] package shader scan: {} skipped ({})", file->GetName(),
                        e.what());
        }
        ++done;
        if (scanner.stopped) {
            break;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - last_note > std::chrono::seconds{5}) {
            last_note = now;
            LOG_INFO(Core,
                     "[nvn] package shader scan: {}/{} files, {:.0f} MB read, {:.0f} MB inflated, "
                     "{} stages",
                     done, files.size(), static_cast<double>(scan.bytes_read) / 1e6,
                     static_cast<double>(scan.bytes_inflated) / 1e6, scan.stages);
        }
    }
    scan.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return scan;
}

} // namespace Core::NvnHle
