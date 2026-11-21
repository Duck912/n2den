#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later
"""Merge NVN-HLE portable state lists (nvn_state_list.<title>.bin) from several sessions.

A state list holds the pipelines one machine met while playing. A game does not draw exactly the
same pipelines every session: some are drawn only in some sessions (the same programs into
another set of render targets, another vertex layout, another texture type), depending on timing
inside the game. A list recorded in one session misses those, and a prepared first launch that
meets one compiles it at run time and skips its draws for a frame. A list made from several
sessions covers what any of them met.

The merge keeps every record of every input once (identical records are written once), in input
order. All inputs must have the same version, state layout and title; a damaged record ends that
input (the records before it are kept), as the emulator's reader does. The output is written to a
temporary file and renamed.

Usage: state_list_merge.py OUTPUT INPUT [INPUT ...]
"""

from __future__ import annotations

import os
import struct
import sys
import zlib

HEADER = struct.Struct("<8sIIQI")  # magic, version, layout, title, crc of the bytes before it
FRAME = struct.Struct("<IIII")  # tag, type, payload size, crc of (type, size, payload)
RECORD_TAG = 0x4352564E  # "NVRC"
MAX_RECORD_BYTES = 64 << 20
TYPE_NAMES = {16: "graphics", 17: "compute", 19: "compute seen unused"}


def crc32(data: bytes, crc: int = 0) -> int:
    # Same polynomial and conditioning as the emulator's Crc32 (zlib's CRC-32).
    return zlib.crc32(data, crc) & 0xFFFFFFFF


def read_list(path: str):
    data = open(path, "rb").read()
    if len(data) < HEADER.size:
        raise ValueError(f"{path}: too short for a state list header")
    header = data[: HEADER.size]
    magic, version, layout, title, crc = HEADER.unpack(header)
    if crc32(header[:-4]) != crc:
        raise ValueError(f"{path}: damaged header")
    records = []
    pos = HEADER.size
    note = ""
    while pos < len(data):
        if len(data) - pos < FRAME.size:
            note = "truncated record frame"
            break
        tag, rtype, size, rcrc = FRAME.unpack_from(data, pos)
        if tag != RECORD_TAG or size > MAX_RECORD_BYTES or size > len(data) - pos - FRAME.size:
            note = "bad record frame"
            break
        payload = data[pos + FRAME.size : pos + FRAME.size + size]
        if crc32(payload, crc32(struct.pack("<II", rtype, size))) != rcrc:
            note = "record checksum mismatch"
            break
        records.append((rtype, data[pos : pos + FRAME.size + size]))
        pos += FRAME.size + size
    return (magic, version, layout, title), header, records, note


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    output, inputs = argv[1], argv[2:]
    identity = None
    header_bytes = b""
    seen: set[bytes] = set()
    merged: list[bytes] = []
    for path in inputs:
        ident, header, records, note = read_list(path)
        if identity is None:
            identity, header_bytes = ident, header
        elif ident != identity:
            print(f"{path}: version/layout/title differ from the first input "
                  f"({ident[1]} {ident[2]:#x} {ident[3]:016x} vs {identity[1]} "
                  f"{identity[2]:#x} {identity[3]:016x}); not merged", file=sys.stderr)
            return 1
        added: dict[str, int] = {}
        for rtype, frame in records:
            if frame in seen:
                continue
            seen.add(frame)
            merged.append(frame)
            name = TYPE_NAMES.get(rtype, f"type {rtype}")
            added[name] = added.get(name, 0) + 1
        summary = ", ".join(f"{n} {k}" for k, n in sorted(added.items())) or "nothing new"
        print(f"{path}: {len(records)} records, added {summary}" + (f" ({note})" if note else ""))
    temp = output + ".tmp"
    with open(temp, "wb") as out:
        out.write(header_bytes)
        for frame in merged:
            out.write(frame)
    os.replace(temp, output)
    print(f"{output}: {len(merged)} records, {os.path.getsize(output)} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
