#!/usr/bin/env python3
"""Scan NVN-HLE per-pass render-target dumps for NaN / Inf texels, in pass order.

Dumps come from EDEN_NX2_NVN_DUMP_TARGETS=<dir> EDEN_NX2_NVN_DUMP_FRAME=<n> (optionally _MIN_WIDTH): files
f<frame:05>_p<pass:03>_<tag>_t<serial>_d<draws>.tga plus .tga.raw = the host image bytes.
Bytes per texel come from raw size / (TGA width*height): 8 = RGBA16F, 16 = RGBA32F, 4 = treated as
B10G11R11_UFLOAT (host VkFormat 122, the usual HDR scene target; exponent all ones = Inf/NaN). Depth (z) and
other formats are listed without a verdict.
usage: py -I scan_nan.py <dump dir> [frame]
First line flagged "<-- NaN/Inf" is the pass whose draws first produced bad texels.
"""
import glob
import os
import struct
import sys

import numpy as np


def tga_size(path):
    h = open(path, "rb").read(18)
    return struct.unpack("<HH", h[12:16])


def scan(raw, w, h):
    data = open(raw, "rb").read()
    bpt = len(data) // max(w * h, 1)
    if bpt == 8:
        a = np.frombuffer(data[: w * h * 8], dtype=np.float16).astype(np.float32).reshape(h, w, 4)
    elif bpt == 16:
        a = np.frombuffer(data[: w * h * 16], dtype=np.float32).reshape(h, w, 4)
    elif bpt == 4:
        u = np.frombuffer(data[: w * h * 4], dtype=np.uint32).reshape(h, w)
        bad = (((u >> 6) & 0x1F) == 0x1F) | (((u >> 17) & 0x1F) == 0x1F) | (((u >> 27) & 0x1F) == 0x1F)
        return bpt, int(bad.sum()), None, None
    else:
        return bpt, None, None, None
    rgb = a[..., :3]
    nan = np.isnan(rgb).any(axis=2)
    inf = np.isinf(rgb).any(axis=2)
    finite = rgb[np.isfinite(rgb)]
    return bpt, int(nan.sum()), int(inf.sum()), (float(finite.max()) if finite.size else None)


def main():
    d = sys.argv[1]
    frame = sys.argv[2].zfill(5) if len(sys.argv) > 2 else "*"
    first_bad = None
    for tga in sorted(glob.glob(os.path.join(d, f"f{frame}_p*_*.tga"))):
        raw = tga + ".raw"
        parts = os.path.basename(tga).split("_")
        if not os.path.exists(raw) or parts[2].startswith("z"):
            continue
        w, h = tga_size(tga)
        bpt, nan, inf, mx = scan(raw, w, h)
        bad = (nan or 0) + (inf or 0)
        if bad and first_bad is None:
            first_bad = (parts[1], parts[2], parts[3])
        mxs = f"{mx:.4g}" if isinstance(mx, float) else "-"
        print(f"{parts[1]} {parts[2]} {parts[3]} {w}x{h} {bpt}B nan={nan} inf={inf} max={mxs}"
              + (" <-- NaN/Inf" if bad else ""))
    print("first NaN/Inf:", first_bad)


if __name__ == "__main__":
    main()
