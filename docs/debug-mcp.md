# Live debug tools

Start n2Den with `EDEN_NX2_DEBUG_SERVER=127.0.0.1:6545` and boot a game. Run
`py -I tools/n2den/debug_mcp.py` as a stdio MCP server, or make a single request:

```powershell
py -I tools/n2den/debug_mcp.py --call '{"method":"screenshot"}' --out shot.png
py -I tools/n2den/debug_mcp.py --call '{"method":"capture_frame","params":{"sampled":true,"buffers":true,"index":true}}'
```

The adapter accepts `--server`, `--catalog` and `--out`. The server address defaults to
`N2DEN_DEBUG_SERVER`, then `EDEN_NX2_DEBUG_SERVER`, then `127.0.0.1:6545`. Catalogs are
annotated by native program key, with both dictionary and list `programs` formats accepted.

One TCP server serves these 34 methods:

`status`, `screenshot`, `log_tail`, `log_filter`, `capture_frame`, `sample`, `guest_threads`,
`jit_hot`, `sched_trace`, `read_memory`, `search_memory`, `write_memory`, `read_gpu_memory`, `input`,
`input_sequence`, `shaders`, `draws`, `reject_shaders`, `reject_draws`, `dump_shader`,
`compute_state`, `watch_compute`, `rebuild_compute`, `read_gpu`, `set_frontend`,
`frontends`, `dump_ir`, `dumps`, `qmd_trace`, `gpu_activity`, `video_state`,
`video_inject`, `skip_draws`, `press_a`.

MCP exposes the same names plus `capture_image`. That adapter-local tool reads a PNG
only from a capture made by that adapter session. `capture_frame`'s adapter-only
`index` option invokes `frame_dump_index.py` to produce a contact sheet.

Method notes:

- `screenshot` returns a 1280 by 720 game image in `png_base64`; `timeout_s` is 1–15.
- `reject_shaders` replaces the entire set of 16-digit native keys or host hashes.
  It requires `hashes`; `[]` restores all shaders. Read `status.rejected_shaders`
  to inspect the set.
- `log_filter` overlays the configured filter and returns `base` and `applied`.
- `status` returns n2Den's counters and identity, present counts, timing,
  renderer, pause state, log filter and the write-memory flag. Its default FPS
  sampling window is zero; pass `fps_window_ms` to measure FPS.

Guest memory reads are limited to 64 KiB. Writes require `EDEN_NX2_DEBUG_WRITE=1`
at boot and return the previous bytes. Input scripts hold buttons and sticks for
presented frames, retain physical input, and release on completion, disconnect,
timeout or shutdown. With `"async": true`, `input` and `input_sequence` return at once and the
script releases the controller after its last frame, so screenshots can be taken while it plays.
`press_a` sends a pulse of the given number of milliseconds.
`sample` samples named Windows x64 host threads and writes a text report and
`.folded` stacks. Symbols depend on the available PDBs.
`guest_threads` (`seconds`, 1 to 60, default 10) reports guest CPU time per thread of the game
over the window from the kernel's own per-thread accounting, with per-core idle time; 100%
is one full guest core. It does not suspend anything.

`jit_hot` (`seconds`, 1 to 60, default 20; `period_us`, default 1000; `top`, default 200)
samples only the instruction pointer of the `CPUCore_N` host threads and resolves each sample
to the dynarmic block it is in, through a side table of host range to guest start and end PC
that every A64 JIT fills when it emits a block or loads one from the block cache. The table is
off unless n2Den starts with `EDEN_JIT_HOT_MAP=1` (it costs about 24 bytes per block). Samples
are classed as in a block, in a code space outside blocks (prelude: dispatcher and thunks;
between blocks) or outside JIT code (split by host module: emulator, OS, waiting). The result
has the top blocks, coverage curves (how many blocks and guest bytes reach 50, 80, 90, 95 and
99% of in-block samples), and the same by guest 4 KiB page, by loaded module and by exported
function symbol; the file `jit_hot-*.json` in the dump directory has the full lists.

`sched_trace` (`seconds`, 0.1 to 5, default 1) records every guest context switch per core
(CoreTiming ticks at 19.2 MHz, thread switched out, thread switched in; 0 is the idle thread)
into preallocated per-core rings, and every frame counted by nvn-hle's window present (kind 0)
and the end of its present pacing (kind 1). Off, it costs the scheduler one relaxed atomic load
per switch; on, no locks. It returns the events with thread names and writes
`sched_trace-*.json`; `py -I tools/n2den/sched_trace_report.py <file>` prints per-frame
timelines and a one-line summary per frame.

Frame capture and `read_gpu_memory` run on the Vulkan rasterizer thread between
frames. Tool output defaults to n2Den's debug dump directory; `EDEN_NX2_DEBUG_DIR`
overrides the directory for captures and sampling reports without changing other
dump paths.

Environment-driven capture also works without the TCP server:

- `EDEN_NX2_FRAME_DUMP=<directory>` enables a single frame dump.
- `EDEN_NX2_FRAME_DUMP_AT` accepts seconds after renderer creation, a frame count
  with an `f` suffix, or `file` to wait for `<directory>/capture.now`.
- `EDEN_NX2_FRAME_DUMP_SAMPLED=1` includes sampled textures and bound buffers.

The frame manifest has the fields `frame`, `pass`, `sampled`, `buffer`, `dispatch`,
`launch`, `summary`, and the presented image entry. It includes all nonzero
pipeline shader hashes, including task and mesh stages, and native keys where
known. Buffer bindings keep their GPU addresses. Bindings spanning host
allocations are read from the assembled buffer actually referenced by the Vulkan
descriptor. Captures have a staging budget and a size limit, store raw files
compressed and skip MSAA images.

Run the adapter tests with:

```powershell
py -I tools/n2den/test_debug_mcp.py
```
