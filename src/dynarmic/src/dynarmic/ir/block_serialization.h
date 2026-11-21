// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include <span>
#include <vector>

#include "common/common_types.h"

namespace Dynarmic::IR {

class Block;

/// Serializes an optimized block: its locations, condition, cycle counts, terminal and every instruction with
/// its opcode, name and arguments. The format is compact (LEB128 varints) and only valid for the same
/// IR schema (see SchemaHash).
std::vector<u8> SerializeBlock(const Block& block);

/// Rebuilds a block serialized by SerializeBlock into `block`, which must be freshly reset. Returns false for a
/// malformed record (the block is then left in an unspecified state and must be reset before reuse).
bool DeserializeBlock(std::span<const u8> data, Block& block);

/// The location descriptor value a serialized block was translated for, read from its header without
/// deserializing it; nothing for a malformed or other-version record.
std::optional<u64> PeekBlockLocation(std::span<const u8> data);

/// A hash of the IR schema: every opcode's name, return type and argument types, plus the serialization format
/// version. Cached blocks are only valid for an identical schema.
u64 SchemaHash();

/// 64-bit hashing helpers shared by the block cache.
u64 HashBytes(const void* data, size_t size, u64 seed = 0);
u64 HashCombine(u64 a, u64 b);

}  // namespace Dynarmic::IR
