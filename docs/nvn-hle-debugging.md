# Debugging the NVN-HLE renderer (Vulkan)

How to find the cause of a hang, a wrong-looking frame or a random per-boot bug in the NVN-HLE renderer
(`src/video_core/renderer_vulkan/nvn/`). The Fast Fusion fixes are used as worked examples. Reading the shader
code itself (the game's program, the driver's compiled SASS, NaN scans) is covered in
[shader-inspection.md](shader-inspection.md).

All switches below are environment variables, off unless set. Logs go to the normal eden log.

## 1. Classify the bug first

- **Device lost / GPU hang:** the log says `the Vulkan device was lost`. Go to section 2.
- **Wrong image** (too bright, black, splotches, missing effect): go to section 3.
- **Random per boot:** many HLE bugs depend on memory layout and timing, so one boot proves nothing. Judge a
  switch or a fix by rates over several boots per setting ("washed 4 of 5 without, 0 of 4 with"). A cheap way
  to classify boots without watching them is the mean brightness of screenshots taken at fixed times after a
  known point (start of a race, a menu). Black frames sit near 0.0, a washed-out frame well above the normal
  value.

Compare against the emulated GPU path (NVN-HLE off) in the same scene: it gives reference values for
uniforms, exposure and render-target contents.

## 2. Hangs

1. `EDEN_NX2_NVN_CHECKPOINTS=1` puts a `VK_NV_device_diagnostic_checkpoints` marker before every replayed op.
   After a device loss the log lists the last checkpoints the GPU reached: frame, op number, op type, programs,
   counts and indirect address. The op in flight is the hang. A checkpoint from frame 1 in a late frame means
   a recorded command list that is replayed again.
2. `EDEN_NX2_NVN_SKIP_PROGRAMS=<hash>,<hash>` skips every draw and dispatch that uses these programs (check the
   `skipped` log line). Use it to confirm a suspect or bisect.
3. `EDEN_NX2_NVN_TRACE_DISPATCH=<hash>` logs the first 64 dispatches of a compute program: groups, indirect
   address, caller.
4. Heavy tracing changes timing and can hide a hang. Bisect a hang with logging off; turn tracing on only once
   the suspect is known.
5. Nsight Aftermath dumps often say "hung" with no active shader on NVN-HLE. Checkpoints are more useful.

Worked example (Fast Fusion, race start): the checkpoints stopped at an indexed indirect draw. A compute
shader writes its draw counts with atomics into a storage shadow, but the arguments were read from the pool's
read-only VRAM mirror, which held stale bytes, so the GPU drew garbage counts until the device was lost. Rule
now: an indirect argument range is served from a live shadow first, then from a current mirror, then from host
memory (`ResolveIndirectSource`).

## 3. Wrong images

### Find the pass that goes wrong

- `EDEN_NX2_NVN_DUMP_TARGETS=<dir>` with `EDEN_NX2_NVN_DUMP_FRAME=<n>` dumps every render target after each
  pass of frame `n` (`.tga` preview plus `.tga.raw` with the host image bytes). `EDEN_NX2_NVN_DUMP_MIN_WIDTH=N`
  keeps only targets at least `N` texels wide, which cuts a 4K frame's dump to the full-size targets.
- `EDEN_NX2_NVN_DRAWLOG=<n>` logs up to `n` draws of the frame chosen by `EDEN_NX2_NVN_DUMP_FRAME`, with their
  programs, targets, blend state and bindings (`draw+` lines list uniform, storage and texture bindings).
- Compare a good and a bad frame pass by pass (mean brightness per target is enough to see where they part).
- **At the moment the bug shows, without a frame number:** write `nvn-frame` into `nx2-request.txt` in the user
  folder (`%APPDATA%\eden`, e.g. `echo nvn-frame> "%APPDATA%\eden\nx2-request.txt"`). The next frame's targets
  of every pass go to `diag\<time>\` in the same file names as above, and its draws to the log. With
  `EDEN_NX2_NVN_OPLOG=1 EDEN_NX2_NVN_TRACE_ON_REQUEST=1` at launch the op/dispatch log covers that frame only.
- **A pass that reads a texture nothing rendered into this frame:** look for passes with no target
  (`fb=32768x32768`, empty `tg=`) just before it. They write through storage images; a `fallbacks=[...]` entry
  with `word=0x0` on such a draw means the image handle did not resolve and every store was dropped.

### If the shader and its state match but the output differs

The data it reads is wrong. Check, in this order:

1. **Uniforms computed on the CPU.** `EDEN_NX2_NVN_DUMP_DRAW_VS=<vertex program hash>` dumps the CPU view of the
   uniform and vertex buffers of every draw with that vertex program in the dump frame (draw numbers move
   between boots, program hashes don't). Compare the values with the emulated path. Before comparing, check in
   the shader which uniform block and offsets it actually reads.
2. **GPU results the game reads on the CPU.** A game can read a GPU-written texture straight through its CPU
   mapping without calling any NVN read or flush function. NVN-HLE renders into host images, so without a
   write-back the CPU sees stale bytes. Fast Fusion's washed-out race was this: the auto-exposure state goes
   into a 1x1 linear render target every frame, the CPU reads it to build the tone map's exposure, and stale
   bytes pinned the exposure to its clamp (4.0 instead of about 0.4). NVN-HLE now writes GPU-written linear
   textures in CPU-readable pools (up to 1 MiB) back to guest memory at the end of each submission.
3. **Feedback copies.** `EDEN_NX2_NVN_FEEDBACK_NO_PRUNE=1` redoes every feedback copy, to rule out a skipped one.

### Black frames, black or white splotches

These are usually NaN or Inf values that temporal passes (TAA, DLSS history, downsample chains read next frame)
spread in square blocks: NaN shows as black, Inf as white.

1. Dump the frame and run `tools/n2den/scan_nan.py` (see [shader-inspection.md](shader-inspection.md)). The first
   flagged pass is where bad values enter.
2. List that pass's draws with `EDEN_NX2_NVN_DRAWLOG`, then skip the producing program. If another draw becomes
   the first NaN producer, the shaders are fine and an input they share is bad. Compare their texture bindings.
3. Grep the log for dropped operations: `unsupported ClearTexture`, `ClearTexture dropped`. A texture the game
   clears but HLE didn't keeps whatever bytes the memory held, which can decode as NaN.
4. Not every splotch is NaN. `scan_nan.py` reads every 4-byte target as R11G11B10, so RGBA8 targets show false
   NaN counts: look at the images. Fire Emblem's map splotches were an AO depth pass whose storage image handle
   (`c2[0x120]`, in a bank other than the program's bound one) translated to `c0[0x0]`: its stores were dropped
   and the AO passes read leftover memory, which darkened the terrain in blocks.

Worked example (Fast Fusion, black race): the game clears 3D fog volumes one Z slice at a time. NVN-HLE rejected
slice views of 3D textures, so the clears were dropped, the volumes kept stale bytes, and the lighting wrote NaN
into the HDR scene target. Fixed in `Recorder::ClearTexture`, which now clears Z slices through a 2D-array view.

## 4. Memory-layout switches

Useful to make a random bug reproducible or to test a theory about where data lives:

| Variable | Effect |
| --- | --- |
| `EDEN_NVN_SPARSE_SPLIT_IMPORT=0` | Imports large pools whole instead of in sparse pieces. Changes which data sits where; it made Fast Fusion's washed-out race reproducible. |
| `EDEN_NX2_TRY_VTX_MIRROR=0` | No read-only VRAM mirror of pools (vertex, index and read-only storage data come from host memory). |
| `EDEN_NX2_NVN_NO_HOST_IMPORT=1` | Device-local copies instead of imported guest memory. Breaks data the CPU updates, so only for experiments. |

## 5. Behaviour switches added with the Fast Fusion fixes

Each fix has a switch to turn it off for A/B tests.

| Variable | Effect |
| --- | --- |
| `EDEN_NX2_TRY_TEX_WRITEBACK=0` | Turns off the per-submission write-back of GPU-written linear textures in CPU-readable pools. |
| `EDEN_NX2_NVN_TEX_WRITEBACK_MAX_KIB=N` | Size cap for that write-back (default 1024). |
| `EDEN_NX2_NVN_MIRROR_GPU_WRITTEN=0` | Lets read-only pool mirrors serve pages the GPU wrote again (the old behaviour; costs large copy-ins every pass). |

## 6. Diagnostic switches

| Variable | Effect |
| --- | --- |
| `EDEN_NX2_NVN_CHECKPOINTS=1` | Vulkan checkpoints on every replayed op; logs the last ones reached after a device loss. |
| `EDEN_NX2_NVN_SKIP_PROGRAMS=<hash>,...` | Skips every draw and dispatch using these programs. |
| `EDEN_NX2_NVN_TRACE_DISPATCH=<hash>` | Logs the first 64 dispatches of a compute program. |
| `EDEN_NX2_NVN_DUMP_TARGETS=<dir>`, `EDEN_NX2_NVN_DUMP_FRAME=<n>` | Dumps every render target after each pass of frame `n`. |
| `EDEN_NX2_NVN_DUMP_MIN_WIDTH=N` | Dumps only targets at least `N` texels wide. |
| `EDEN_NX2_NVN_DUMP_DRAW_VS=<hash>` | Dumps the CPU view of uniform and vertex buffers of every draw with this vertex program in the dump frame. |
| `EDEN_NX2_NVN_DRAWLOG=<n>` | Logs up to `n` draws of the selected frame (`EDEN_NX2_NVN_DUMP_FRAME`) with state and bindings. |
| `EDEN_NX2_NVN_FEEDBACK_NO_PRUNE=1` | Redoes every feedback copy. |
| `nvn-frame` in `nx2-request.txt` | Dumps every pass of the next frame and logs its draws (no frame number needed). |
| `EDEN_NX2_NVN_TRACE_ON_REQUEST=1` | Environment traces (`EDEN_NX2_NVN_OPLOG`, ...) cover only the frame of an `nvn-frame` request. |
| `EDEN_NX2_NVN_TARGET_FIRST_INIT=0` | Leaves a render target's new image undefined at its first use (old behaviour). |

The in-game DLSS (`nvnDlss*`) path also logs each distinct `nvnDlssNetworkBuilderGetOptimalSettings` question
and answer (method, output size, optimal/max/min render size). Games usually ask once per mode change.

## 7. Practical notes

- The log is buffered. A run that is killed (rather than closed) can lose its last lines: set `flush_line=true`
  in the config's `[Miscellaneous]` section while debugging, and back to `false` afterwards.
- Only one emulator instance at a time; a build cannot replace the executable while it runs.
- Programs and dumps taken from a game are the game's data. Keep them local.

## Open issues (Fast Fusion, NVN-HLE)

- Intermittent boot crash: device lost on the first submission, a few seconds after launch, before the menu.
  Not tied to HDR or to the fixes above.
- With PC DLSS, the game renders at the minimum size NVN-HLE reports (output / 3) from Balanced mode up, because
  `nvnDlssNetworkBuilderGetOptimalSettings` answers with PC ratios.
- The game's own DLSS network (CUDA kernels) has no runner under NVN-HLE; without PC DLSS it falls back to FSR1.
