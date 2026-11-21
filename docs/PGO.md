# ThinLTO and PGO builds

n2den can be built with clang and profile-guided optimization (PGO): an instrumented build counts
which branches and calls run while games play, and the final build is compiled with those counts,
so it inlines and lays out the code that is hot in practice. The approach (clang, ThinLTO and an
LLVM IR profile trained on gameplay) follows Eglantine's release builds. The profile is trained
on n2den itself, on our own games and settings.

Measured on 2026-11-14, Ocarina of Time, Kokiri Forest camera pan, frames ahead 1, two runs each:

| Build | fps |
|---|---|
| MSVC Release | 43.4, 43.8 |
| clang-cl + ThinLTO (no profile) | 51.0, 50.7 |

MSVC stays the development compiler. The clang build has no PGO-specific code paths; the CMake
options below only change compiler and linker flags.

## Options

| Option | Default | Meaning |
|---|---|---|
| `N2DEN_THINLTO` | `OFF` | Compile to LLVM bitcode and optimize across files at link time. |
| `N2DEN_THINLTO_JOBS` | `6` | ThinLTO backend jobs at link time (`0` = the linker's default, one per thread). Each job holds a lot of memory. |
| `N2DEN_PGO` | `OFF` | `GENERATE`: instrumented build. `USE`: optimize with `N2DEN_PGO_PROFILE`. |
| `N2DEN_PGO_PROFILE` | `pgo/n2den.profdata` | Merged profile for `USE`. Its hash is part of the compile flags, so a new profile rebuilds everything. |
| `N2DEN_PGO_PROFRAW_DIR` | `<build>/profraw` | Where a `GENERATE` build writes `.profraw` files. |

They need clang. On Windows that is clang-cl with lld-link and llvm-lib from an LLVM install
(tested with LLVM 22.1.8), against the same MSVC headers, Windows SDK and Qt `msvc2022_64` as the
MSVC build. Run CMake from a shell where `vcvars64.bat` has been called.

## Building

1. Instrumented build (no ThinLTO needed):

   ```
   cmake -S . -B build-pgo-gen -G Ninja -DCMAKE_BUILD_TYPE=Release ^
     -DCMAKE_C_COMPILER="C:/Program Files/LLVM/bin/clang-cl.exe" ^
     -DCMAKE_CXX_COMPILER="C:/Program Files/LLVM/bin/clang-cl.exe" ^
     -DCMAKE_LINKER="C:/Program Files/LLVM/bin/lld-link.exe" ^
     -DCMAKE_AR="C:/Program Files/LLVM/bin/llvm-lib.exe" ^
     -DCMAKE_RC_COMPILER="C:/Program Files/LLVM/bin/llvm-rc.exe" ^
     "-DCMAKE_C_FLAGS=-DWIN32 -D_WINDOWS -DNOMINMAX" "-DCMAKE_CXX_FLAGS=-EHsc -DWIN32 -D_WINDOWS -DNOMINMAX" ^
     -DN2DEN_PGO=GENERATE <your usual options: Qt prefix, CPM cache, ...>
   ninja -C build-pgo-gen yuzu
   ```

2. Train it (next section) and merge the profile into `pgo/n2den.profdata`.

3. Final build: the same configure line in a new build folder, with `-DN2DEN_THINLTO=ON
   -DN2DEN_PGO=USE` instead of `-DN2DEN_PGO=GENERATE`. A build with only `-DN2DEN_THINLTO=ON` is the
   clang build without a profile.

The executable is `bin/eden.exe` as usual. Give each build folder the same portable `user` folder
as your normal build (a directory junction works) so that all of them use the same settings,
saves and caches.

## Training

```
py -I tools/n2den/pgo_train.py --exe build-pgo-gen/bin/eden.exe --profraw build-pgo-gen/profraw ^
  --out runs/pgo-train --merge pgo/n2den.profdata ^
  pan:<scenes>/oot-kokiri.json pan:<scenes>/oot-field.json scene:<scenes>/fe-square.json scene:<scenes>/fe-title.json
```

`<scenes>` is wherever you keep your scene files; they point at your own game dumps and save
snapshots, so they are not part of the repository. Each scene runs in its own session through `pan_run.py` or `scene_run.py` (scene files are
described in `scene_run.py`). The counters are written only when the emulator exits normally,
which both scripts do by closing its window; a session that crashes or is killed contributes
nothing, and the tool stops. The instrumented build is noticeably slower; its frame rate during
training does not matter. Playing the instrumented build by hand and closing it from its window
also adds to the profile.

Append `@K=V,K=V` to a scene to set environment variables for that session only. Use it to cover
code a warm session never runs: `@EDEN_JIT_CACHE=0` makes the CPU translator run (otherwise it is
cold in the profile and optimized for size, which slows first runs), and `@EDEN_NVN_HLE=0` (plus
the game's usual launch switches) trains the emulated GPU path when your settings use NVN-HLE, or
the reverse. Set `EDEN_JIT_CACHE_DIR` to a separate folder while training so the instrumented build
does not rewrite your normal JIT cache. The scene scripts replace the game's save folder with the
scene's snapshot: back up your own save first.

The profile used for the numbers above came from four sessions: Ocarina of Time in Kokiri Forest
and Hyrule Field (camera pans) and Fire Emblem: Fortune's Weave's town square and title screen.

## Caveats

- Don't use a compiler cache (sccache, ccache) for the instrumented and PGO builds. A cached object
  can carry header dependencies from another checkout, so later header edits don't rebuild it, and
  the cache can return objects built with an older profile. Use a fresh build folder per variant.

- Code that the training never ran is treated as cold and optimized for size. Trained games gain
  the most; others gain less, and paths only an untrained game uses can come out slower.
- The profile goes stale as the code changes: functions edited since training lose their profile
  data (a warning, not an error) and are compiled as in the plain ThinLTO build. Retrain after
  large changes.
- clang only reads profiles written by its own LLVM version; retrain after upgrading LLVM.
- A new compiler can expose code that worked by luck under MSVC. The first clang build crashed in
  the Vulkan scheduler, whose command buffer was misaligned for the ASTC pass (fixed in `9cf03ab`).
  Check a clang build in the games you care about before relying on it.
