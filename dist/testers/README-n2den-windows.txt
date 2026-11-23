n2Den for Windows x64
=====================

n2Den is an experimental Nintendo Switch 2 recompulator built on the Eden
emulator. This build was made from commit @COMMIT@.

n2den runs decrypted dumps of games you own. It contains no decryption code
and does not use console keys. No games are included.

Which games have been tested, and how far they get on each graphics backend,
is listed in docs/GameProgress.md in the source repository.


SETUP
-----
1. Extract into a new folder. The Visual C++ runtime is included.
2. For portable mode, create an empty folder named user beside n2Den.exe:
   settings, saves and caches then stay in that folder. Without it the
   build uses %APPDATA%\eden like any Eden build.
3. Add the folder with your decrypted game dumps (.nup or .nx packages, or
   unpacked ExeFS/RomFS folders), then start the game.

No launch switches or environment variables are needed. The graphics backend
(Emulated GPU, the default, or NVN-HLE) is chosen per game in the game's
properties. "Precompile CPU code" compiles a game's CPU code when it boots, so
play does not stop to translate it.

The first time a new area or effect appears, expect a short stutter while its
shaders build. Later runs reuse the shader cache.

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
https://github.com/Duck912/n2den.
