# Reading GPU shader code (guest Ampere and host driver output)

Three small tools in `tools/n2den/` let you read the shader code involved in a rendering bug without a debugger.
All are Python 3 scripts. Run them with `py -I` (or `python3 -I`) from the repo root.

| Tool | Reads | Gives you |
|---|---|---|
| `nvn_store.py` | n2den's NVN-HLE shader store (`nvn_shader_store.bin` in the title's shader cache folder) | the game's own SM86 program for a program hash from the log (`[nvn]` draw/dispatch lines) |
| `nvcache_sass.py` | NVIDIA's Vulkan driver cache (Windows: `%LOCALAPPDATA%\NVIDIA\GLCache\<id>\<id>\*.bin`) | the host GPU's compiled SASS for a translated shader (what the driver really ran) |
| `scan_nan.py` | NVN-HLE per-pass render-target dumps | the first pass of a frame that writes NaN/Inf into a target |

Requirements: `pip install zstandard numpy`, and nvdisasm 13.x from the CUDA toolkit or NVIDIA's standalone
`cuda_nvdisasm` archive. `nvcache_sass.py` uses `NVDISASM=<path>` or finds `nvdisasm` on `PATH`.

Programs pulled out of a game or a driver cache are game code. Keep them local and never commit or upload them.

## Guest program: what the game asked for

```
py -I tools/n2den/nvn_store.py <shader cache>/nvn_shader_store.bin list
py -I tools/n2den/nvn_store.py <store> extract <program hash> out/
nvdisasm -b SM86 -raw out/<hash>.code.bin
```

Program hashes come from the NVN-HLE log, for example `EDEN_NX2_NVN_DRAWLOG` draw lines or
`EDEN_NX2_NVN_TRACE_DISPATCH`. Stage 5 is compute.

## Host program: what the driver compiled

1. Run the emulator to the scene so the driver writes the shader to its cache. The cache file is the big `.bin`
   whose modified time matches the run.
2. Find the shader by a byte pattern that is distinctive for it, usually a float or half immediate from the guest
   program, little-endian (0.99f -> `a4707d3f`, 1000.0f -> `00007a44`):
   `py -I tools/n2den/nvcache_sass.py find <cache.bin> a4707d3f 00007a44`
   Several hits mean several builds of the same shader; the highest offset is the newest.
3. Dump it: `py -I tools/n2den/nvcache_sass.py dump <cache.bin> <entry offset> out/ SM120` (use your GPU's arch;
   SM120 = RTX 50 series). This writes each NVuc section and the disassembly of section type 0x0001 (the code).

This answers "did the term I expect survive compilation" before you audit translation code. It works the same for
emulated-GPU and NVN-HLE shaders, so you can diff the two builds of one program.

The cache file format (driver 5xx/6xx on Windows) is described in the script's header: zstd entries that share one
dictionary embedded after an `_NVDICT_` marker, each holding an NVuc container. The section layout follows
therontarigo/nvcachetools (ISC).

## Black or splotchy frames: where NaN enters

```
EDEN_NX2_NVN_DUMP_TARGETS=<dir> EDEN_NX2_NVN_DUMP_FRAME=<frame> EDEN_NX2_NVN_DUMP_MIN_WIDTH=1000 <run the game>
py -I tools/n2den/scan_nan.py <dir> <frame>
```

The first line flagged `<-- NaN/Inf` is the pass whose draws first wrote bad texels. List that pass's draws and
bindings with `EDEN_NX2_NVN_DRAWLOG` in the same boot. If skipping the producing program
(`EDEN_NX2_NVN_SKIP_PROGRAMS=<hash>`) just moves the NaN to another draw, the shaders are fine and a shared input
is bad. Check the log for dropped clears (`unsupported ClearTexture`, `ClearTexture dropped`) first.

4-byte targets are read as B10G11R11_UFLOAT, so R16G16 targets (motion vectors) can show false positives.
Judge by the scene/HDR target.
