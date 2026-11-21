n2Den test build for Linux x86-64
=================================

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
1. Make the AppImage executable and run it:
       chmod +x n2den-linux-*.AppImage
       ./n2den-linux-*.AppImage
   If it does not start, your distribution may need FUSE 2 (often a package
   called fuse2 or libfuse2), or run it without FUSE:
       ./n2den-linux-*.AppImage --appimage-extract-and-run
2. Settings, saves and caches use ~/.config/eden and ~/.local/share/eden
   like any Eden build. Back them up first if you also use another Eden build.
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

Example: EDEN_NATIVE_FRONTEND=lower ./n2den-linux-*.AppImage


REPORTING
---------
Send ~/.local/share/eden/log/eden_log.txt right after a problem, with your GPU,
driver version and distribution. It is overwritten on the next launch.


SOURCE
------
n2Den is derived from Eden and is licensed under the GNU GPL v3 or later (see
LICENSE.txt). The source code is at <n2den repository URL>.
