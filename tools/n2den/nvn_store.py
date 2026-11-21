#!/usr/bin/env python3
"""List / extract stages from an n2den NVN-HLE shader store (nvn_shader_store.bin).

usage: py -I tools/n2den/nvn_store.py <store> [list | extract <hash_hex> <outdir>]

file    := "NVNSHSTR" u32 version u32 crc
frame   := u32 tag u32 type u32 size u32 crc payload
payload := u64 hash u32 stage u32 control_size u32 code_size u32 csize zstd(control || code)

The extracted <hash>.code.bin disassembles with `nvdisasm -b SM86 -raw` from offset 0.
Stage 5 = compute. The store sits in the title's shader cache folder (nvn_shader_store.bin).
Extracted programs are game code: keep them local, never commit or upload them.
Needs the `zstandard` module (pip install zstandard).
"""
import os
import struct
import sys

import zstandard


def records(path):
    b = open(path, 'rb').read()
    assert b[:8] == b'NVNSHSTR', b[:8]
    off = 16
    while off + 16 <= len(b):
        _tag, _typ, size, _crc = struct.unpack_from('<4I', b, off)
        payload = b[off + 16: off + 16 + size]
        off += 16 + size
        if len(payload) < 24:
            break
        h, stage, cs, code, csz = struct.unpack_from('<QIIII', payload, 0)
        yield h, stage, cs, code, payload[24:24 + csz]


def main():
    path = sys.argv[1]
    cmd = sys.argv[2] if len(sys.argv) > 2 else 'list'
    recs = list(records(path))
    if cmd == 'list':
        for h, stage, cs, code, _ in recs:
            print(f'{h:016x} stage {stage} control {cs} code {code}')
        print(len(recs), 'records')
        return
    want = int(sys.argv[3], 16)
    out = sys.argv[4]
    os.makedirs(out, exist_ok=True)
    for h, stage, cs, code, comp in recs:
        if h == want:
            joined = zstandard.ZstdDecompressor().decompress(comp, max_output_size=cs + code)
            open(os.path.join(out, f'{h:016x}.control.bin'), 'wb').write(joined[:cs])
            open(os.path.join(out, f'{h:016x}.code.bin'), 'wb').write(joined[cs:])
            print(f'wrote {h:016x} stage {stage} control {cs} code {code}')


if __name__ == '__main__':
    main()
