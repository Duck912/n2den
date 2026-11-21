<!--
# SPDX-FileCopyrightText: Copyright 2025 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# SPDX-FileCopyrightText: 2018 yuzu Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later
-->
<!-- lang: en-GB -->

<h1 align="center">
  <img src="./dist/n2den.svg" alt="n2den" width="160"><br>
  n2den
</h1>

<h4 align="center">An experimental Nintendo Switch 2 recompulator, built on Eden.<br>
The focus is on Switch 2 games.</h4>

> **Development build.** n2den is a work in progress: expect missing graphics, crashes and per-game launch
> settings. For source provenance and test-build caveats, see
> [provenance, credits and test-build caveats](docs/N2DEN.md).

## What n2den is

n2den is a **recompulator**: part recompiler, part emulator. A game's CPU code and graphics pipelines are
compiled for the host before play starts, as a recompilation project would do, but the game still runs inside
n2den, which emulates the rest of the console. Nothing is turned into a standalone program.

Switch 2 games come first in n2den, and design decisions are made for them. The Switch 1 support inherited
from Eden is kept for decrypted dumps, but it is not where the work goes.

## Games

n2den runs decrypted dumps of games you own. It contains no decryption code and does not use console keys.

It loads `.nup` and `.nx` packages, unpacked ExeFS/RomFS folders, and NSO, NRO and KIP executables.
Encrypted content is not supported.

## Design

| Area | n2den |
|---|---|
| **CPU** | **Dynarmic NX2**, extended for the Switch 2's Cortex-A78C instructions, with a persistent JIT disk cache: Off, Translations, or Translations + x64 code. Compatible x64 blocks can be reused across runs on Windows. [CPU details](docs/dynarmic/A78C.md). |
| **Shaders** | A **native Ampere (SM86) front end** decodes the Switch 2 GPU's own instructions straight into the shader recompiler's IR, including task and mesh shaders. |
| **Graphics backend** | Two backends, chosen per game in the settings. **Emulated GPU** (default): the game's own NVN driver code runs unchanged and n2den emulates the hardware it talks to (native command channels and their semaphores, compute QMDs, GPU-written command lists, the Switch 2 graphics supervisor service). **NVN-HLE** (experimental): the game's NVN calls are recorded and replayed on Vulkan by a reimplementation of the driver, which removes the emulated hardware from the frame. |
| **Prepared launch** | Work that normally causes stutter during play is done before the game starts: the CPU code of the whole game can be compiled into the JIT disk cache at boot ("Precompile CPU code"), and under NVN-HLE the graphics pipelines a game is known to use are built at boot from a per-game state list, with the shaders taken from the game's own files. |
| **Bindless textures** | A run-time **texture pool** resolves texture handles that shaders load from storage buffers. |
| **GPU threading** | The GPU command stream is split over two host threads, with per-binding caches to keep the per-draw cost low for games that issue ~100,000 draws a second. |
| **Video** | Native AV1 / H.264 decoder service for in-game movies. |
| **Debugging** | A built-in debug server with MCP tools (shader dumps, frame capture, draw lists, profiler, scripted input) to drive and inspect a running game. |

The shader path matters most. Games such as The Legend of Zelda: Ocarina of Time (Switch 2) draw foliage,
grass and distant trees with mesh shaders and GPU-built object lists, which only render correctly when the
Ampere programs are translated as Ampere.

## Game progress

Which games boot, how far they get on each graphics backend and what is still wrong is tracked in
[docs/GameProgress.md](docs/GameProgress.md).

## Running Switch 2 games

Several features are still behind launch switches. The settings a game needs (for Ocarina of Time:
`EDEN_NATIVE_FRONTEND=auto`, `EDEN_NX2_BARRIER_STUB=1`, `EDEN_NX2_TEXTURE_POOL=fallback`,
`EDEN_N2DEN_KEEP_ZERO_DRAWS=1`, and Reactive Flushing on) are set by the test launchers; a plain launch of the
executable has none of them. Making them defaults is planned.


## Building

See the [General Build Guide](docs/Build.md) (Windows builds: MSVC with Ninja) and the
[Linux build and AppImage guide](docs/build/Linux-NS2.md). Development tooling is in the [tools directory](./tools);
the debug server is documented in [docs/debug-mcp.md](docs/debug-mcp.md).

## License

n2den is licensed under the GPLv3 (or any later version), like Eden. See [LICENSE.txt](LICENSE.txt).

## Credits

n2den stands on the work of the Eden project and everyone before it:

- Eden
- Ryujinx, Sudachi, Citron, Torzu, Suyu, Ryubing
- The Switch 2 emulation community; ported fixes are credited in the source comments next to the code they brought in.
- The Eglantine Community, whose work the NVN-HLE backend is ported from.
- Fortune's Eden and its contributors, including xDarkWindx and the portable-build maintainers, for the source snapshot used as n2den's base.
