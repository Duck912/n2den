#!/usr/bin/env python3
"""Read the host GPU's compiled SASS for n2Den shaders out of NVIDIA's Vulkan driver cache.

  py -I nvcache_sass.py list <cache.bin>                          entry offset, size, NVuc sections
  py -I nvcache_sass.py find <cache.bin> <hex> [<hex> ...]         entries containing ALL byte patterns
  py -I nvcache_sass.py dump <cache.bin> <entry offset> <outdir> [SM120]
                                                                   sections + nvdisasm of the code section

<cache.bin>: %LOCALAPPDATA%/NVIDIA/GLCache/<id>/<id>/<hash>.bin (the Vulkan cache lives under GLCache; pick the
             big .bin whose mtime matches the last emulator run). DXCache only holds .nvph hints.
<hex>:       little-endian patterns, e.g. the guest kernel's distinctive immediates (0.99f -> a4707d3f,
             half pair -> e2a2e2a2, 1000.0f -> 00007a44). Several hits = several builds; highest offset = newest.

Format (driver 5xx/6xx, Windows): entries are zstd frames that all use ONE 2 MiB dictionary embedded in the same
file after an `_NVDICT_` marker (u32 size just before the dict magic 37a430ec). A decoded entry is
`NVDANVVM` + an NVuc container: word 2 low 16 bits = section count, table at word 8, 8 words per entry
(type, length, offset; layout from therontarigo/nvcachetools, ISC). Section type 0x0001 is the SASS
(SM120 on RTX 50xx); nvdisasm 13 no longer opens NVuc files directly, so extract it first.
Needs the `zstandard` module (pip install zstandard) and nvdisasm 13.x (set NVDISASM or put it on PATH).
"""
import os
import re
import struct
import subprocess
import sys

import zstandard

# nvdisasm 13.x (CUDA toolkit or the standalone cuda_nvdisasm archive): NVDISASM=<path>, else PATH.
ND = os.environ.get("NVDISASM", "nvdisasm")


def load(path):
    data = open(path, "rb").read()
    d = data.find(b"\x37\xa4\x30\xec")
    if d < 0:
        sys.exit("no zstd dictionary in file")
    size = struct.unpack_from("<I", data, d - 4)[0] or 0x200000
    return data, zstandard.ZstdDecompressor(dict_data=zstandard.ZstdCompressionDict(data[d:d + size]))


def entries(data, dec):
    for m in re.finditer(re.escape(b"\x28\xb5\x2f\xfd"), data):
        try:
            yield m.start(), dec.decompressobj().decompress(data[m.start():m.start() + (16 << 20)])
        except zstandard.ZstdError:
            continue


def sections(obj):
    base = obj.find(b"NVuc")
    if base < 0:
        return base, []
    count = struct.unpack_from("<I", obj, base + 8)[0] & 0xffff
    return base, [(i, *struct.unpack_from("<III", obj, base + 32 + 32 * i)) for i in range(count)]


def main():
    cmd, path = sys.argv[1], sys.argv[2]
    data, dec = load(path)
    if cmd == "list":
        for off, obj in entries(data, dec):
            _, secs = sections(obj)
            print(off, len(obj), " ".join(f"{t:04x}@{o:x}+{l:x}" for _, t, l, o in secs))
    elif cmd == "find":
        needles = [bytes.fromhex(h) for h in sys.argv[3:]]
        for off, obj in entries(data, dec):
            if all(n in obj for n in needles):
                print(off, len(obj))
    elif cmd == "dump":
        want, outdir = int(sys.argv[3]), sys.argv[4]
        arch = sys.argv[5] if len(sys.argv) > 5 else "SM120"
        os.makedirs(outdir, exist_ok=True)
        for off, obj in entries(data, dec):
            if off != want:
                continue
            base, secs = sections(obj)
            open(os.path.join(outdir, "object.bin"), "wb").write(obj)
            for i, t, length, o in secs:
                name = os.path.join(outdir, f"section{i}_{t:04x}.bin")
                open(name, "wb").write(obj[base + o:base + o + length])
                code = ""
                if t == 0x0001 and length:
                    r = subprocess.run([ND, "-b", arch, "-raw", name], capture_output=True, text=True)
                    open(name[:-4] + ".sass", "w").write(r.stdout)
                    code = f" -> {name[:-4]}.sass ({length // 16} insns)"
                print(f"section {i} type {t:04x} at {o:#x} len {length:#x}{code}")
            return
        sys.exit("entry not found")


if __name__ == "__main__":
    main()
