// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/arm/dynarmic/jit_prebuild_list.h"

#include <cstring>
#include <fstream>
#include <system_error>
#include <vector>

#include <fmt/format.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "core/arm/dynarmic/jit_block_cache.h"

namespace Core {

namespace {

constexpr u32 list_magic = 0x4C50324E;  // "N2PL"
constexpr u32 list_version = 1;

template<typename T>
T Get(const u8* p) {
    u64 v = 0;
    for (size_t i = 0; i < sizeof(T); ++i) {
        v |= u64{p[i]} << (8 * i);
    }
    return static_cast<T>(v);
}

/// Checks a list header against the file size and program id; empty if it is valid.
std::string CheckHeader(const u8* h, u64 file_size, u64 program_id) {
    if (Get<u32>(h) != list_magic) {
        return "not a prebuild list";
    }
    if (Get<u32>(h + 4) != list_version || Get<u32>(h + 8) != JitPrebuildList::entry_size) {
        return fmt::format("unknown version {} / entry size {}", Get<u32>(h + 4), Get<u32>(h + 8));
    }
    const u64 count = Get<u64>(h + 32);
    if (count > (file_size - JitPrebuildList::header_size) / JitPrebuildList::entry_size ||
        file_size != JitPrebuildList::header_size + count * JitPrebuildList::entry_size) {
        return fmt::format("size {} does not match {} entries", file_size, count);
    }
    if (Get<u64>(h + 16) != program_id) {
        return fmt::format("for program {:016X}", Get<u64>(h + 16));
    }
    return {};
}

}  // namespace

std::unique_ptr<JitPrebuildList> JitPrebuildList::Open(const std::filesystem::path& path, u64 program_id,
                                                       std::string* error) {
    const auto fail = [&](std::string why) -> std::unique_ptr<JitPrebuildList> {
        if (error) {
            *error = std::move(why);
        }
        return nullptr;
    };
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        return fail({});
    }
    const u64 size = std::filesystem::file_size(path, ec);
    if (ec || size < header_size) {
        return fail("too small");
    }
    auto list = std::unique_ptr<JitPrebuildList>(new JitPrebuildList());
#ifdef _WIN32
    HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return fail("cannot open");
    }
    list->file_handle = file;
    HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping) {
        return fail("cannot map");
    }
    list->mapping_handle = mapping;
    void* p = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (!p) {
        return fail("cannot map");
    }
#else
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return fail("cannot open");
    }
    list->file_descriptor = fd;
    void* p = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) {
        return fail("cannot map");
    }
#endif
    list->view = static_cast<const u8*>(p);
    list->view_size = size;
    if (std::string why = CheckHeader(list->view, size, program_id); !why.empty()) {
        return fail(std::move(why));
    }
    std::memcpy(list->header.data(), list->view, header_size);
    list->count = static_cast<size_t>(Get<u64>(list->view + 32));
    return list;
}

JitPrebuildList::~JitPrebuildList() {
#ifdef _WIN32
    if (view) {
        UnmapViewOfFile(view);
    }
    if (mapping_handle) {
        CloseHandle(static_cast<HANDLE>(mapping_handle));
    }
    if (file_handle) {
        CloseHandle(static_cast<HANDLE>(file_handle));
    }
#else
    if (view) {
        munmap(const_cast<u8*>(view), view_size);
    }
    if (file_descriptor >= 0) {
        close(file_descriptor);
    }
#endif
}

JitPrebuildList::Entry JitPrebuildList::At(size_t i) const {
    const u8* p = view + header_size + i * entry_size;
    return Entry{Get<u64>(p), Get<u64>(p + 8), Get<u64>(p + 16), Get<u16>(p + 24)};
}

bool JitPrebuildList::Import(const std::filesystem::path& source, const std::filesystem::path& destination,
                             u64 program_id, bool& changed) {
    changed = false;
    std::error_code ec;
    if (std::filesystem::equivalent(source, destination, ec)) {
        return true;
    }
    const auto read_header = [](const std::filesystem::path& p, std::array<u8, header_size>& h, u64& size) {
        std::error_code e;
        size = std::filesystem::file_size(p, e);
        if (e || size < header_size) {
            return false;
        }
        std::ifstream file(p, std::ios::binary);
        file.read(reinterpret_cast<char*>(h.data()), header_size);
        return static_cast<bool>(file);
    };
    std::array<u8, header_size> source_header{};
    u64 source_size = 0;
    if (!read_header(source, source_header, source_size) ||
        !CheckHeader(source_header.data(), source_size, program_id).empty()) {
        return false;
    }
    std::array<u8, header_size> existing{};
    u64 existing_size = 0;
    if (read_header(destination, existing, existing_size) && existing_size == source_size &&
        existing == source_header) {
        return true;
    }
    const auto temp = JitBlockCache::TempPathFor(destination);
    std::filesystem::copy_file(source, temp, std::filesystem::copy_options::overwrite_existing, ec);
    if (!ec) {
        JitBlockCache::ReplaceDurably(temp, destination, ec);
    }
    if (ec) {
        std::filesystem::remove(temp, ec);
        return false;
    }
    changed = true;
    return true;
}

} // namespace Core
