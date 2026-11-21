n2Den test build for Windows x64
================================

n2Den is a fork of the Eden emulator with native Switch 2 (NX2) shader
translation, tested on The Legend of Zelda: Ocarina of Time. This build was
made by GitHub Actions from commit @COMMIT@.

n2den runs decrypted dumps of games you own. It contains no decryption code
and does not use console keys. No games are included.


BUILT IN
--------
Ocarina of Time runs as tested with no launcher, environment variables or
setting changes:
- Native shader front end, with the lowering as the fallback for programs it
  rejects.
- Texture pool lookups for handles no draw expresses, and zero-count mesh draws.
- Reactive Flushing on by default (correct outdoor brightness).
- No one-frame flashes: occlusion-culled objects are drawn when depth-tested,
  and the present waits for its frame (about 5 ms a frame).


SETUP
-----
1. Extract into a new folder. The Visual C++ runtime is included.
2. For portable mode, create an empty folder named user beside n2Den.exe:
   settings, saves and caches then stay in that folder. Without it the
   build uses %APPDATA%\eden like any Eden build.
3. Add the folder with your decrypted game dumps (.nup or .nx packages, or
   unpacked ExeFS/RomFS folders), then start the game.

The first time a new area or effect appears, expect a short stutter while its
shaders build. Later runs reuse the shader cache.


SWITCHES (set before starting, for A/B tests)
---------------------------------------------
  EDEN_NATIVE_FRONTEND=lower             every shader through the lowering
  EDEN_NX2_TEXTURE_POOL=off              no texture pool
  EDEN_N2DEN_KEEP_ZERO_DRAWS=0           skip zero-count draws
  EDEN_NX2_BARRIER_STUB=0                lower barrier cancellation
  EDEN_N2DEN_OCCLUSION_PREDICATES=honor  keep the game's occlusion culling
  EDEN_NX2_FRAMES_AHEAD=-1               the present returns at once
                                         (faster; the flashes come back)

FSR 4 needs AMD's amd_fidelityfx_upscaler_vk.dll beside n2Den.exe; it is not
included.


REPORTING
---------
Send user\log\eden_log.txt (or %APPDATA%\eden\log\eden_log.txt without a user
folder) right after a problem, with your GPU and driver version. It is
overwritten on the next launch.


SOURCE
------
n2Den is derived from Eden and is licensed under the GNU GPL v3 or later (see
LICENSE.txt and the LICENSES folder). The source code is at
<n2den repository URL>.
