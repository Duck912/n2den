# Linux build and AppImage

This guide builds n2den's Qt GUI and command-line frontend on Linux x86-64.
It was tested on an Arch-based distribution with system Qt 6, FFmpeg and SDL3.
CMake 3.31 or newer and a recent C++20 compiler are required. Other distributions
may need newer development packages than their default repositories provide.
See [Dependencies](../Deps.md).

Commands below use Bash. If your terminal uses fish, run `bash` first. Keep the
source and build paths free of spaces: glslang cannot handle some spaced include
paths. n2den runs decrypted dumps of games you own. It contains no decryption
code and does not use console keys.

## 1. Dependencies on Arch-based distributions

```sh
sudo pacman -Syu --needed \
  base-devel cmake ninja git curl python patch unzip \
  boost catch2 fmt nlohmann-json lz4 zstd openssl zlib opus libusb libunwind \
  sdl3 ffmpeg libva qt6-base qt6-charts qt6-wayland \
  vulkan-headers vulkan-utility-libraries vulkan-icd-loader \
  glslang spirv-headers spirv-tools
```

Install the Vulkan driver appropriate for your GPU as well. CMake downloads
dependencies not supplied by the system into `.cache/cpm`; initial configuration
needs network access. `patchelf` is additionally needed for AppImage packaging.

## 2. Clone the repository

```sh
git clone <n2den repository URL> n2den
cd n2den
```

Do not put access tokens into clone URLs, build commands or shared archives.

## 3. Configure and build

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  "-DCMAKE_C_FLAGS=-ffile-prefix-map=$PWD=/builds/n2den/source" \
  "-DCMAKE_CXX_FLAGS=-ffile-prefix-map=$PWD=/builds/n2den/source" \
  -DENABLE_QT=ON -DYUZU_USE_BUNDLED_QT=OFF -DENABLE_QT_TRANSLATION=OFF \
  -DENABLE_WEB_SERVICE=OFF -DENABLE_UPDATE_CHECKER=OFF \
  -DENABLE_LTO=OFF -DENABLE_WERROR=OFF -DYUZU_DISABLE_LLVM=ON \
  -DYUZU_CMD=ON -DYUZU_ROOM_STANDALONE=OFF \
  -DBUILD_TESTING=OFF -DYUZU_TESTS=OFF -DDYNARMIC_TESTS=OFF
cmake --build build --target yuzu yuzu-cmd --parallel 8
```

The **CMake targets** are `yuzu` and `yuzu-cmd`; the resulting **executables** are
`build/bin/eden` and `build/bin/eden-cli`. Reduce the parallel job count if memory
usage is too high. The prefix maps keep this checkout's path out of compiler file
macros and debug information. They do not scrub copied settings, logs or archives.

To rebuild after source changes, repeat the `cmake --build` command.

## 4. Run and load the game

```sh
env QT_QPA_PLATFORM=xcb ./build/bin/eden
```

The xcb backend is the verified workaround for a Qt/Wayland presentation failure.
For an extracted game, choose **File > Load Folder** and select the folder that
contains `main` and `main.npdm` (usually `base/exefs`). This also avoids file-picker
filters that hide extensionless `main` files. A direct launch is:

```sh
env QT_QPA_PLATFORM=xcb ./build/bin/eden "/path/to/game/base/exefs/main"
```

For the command-line frontend:

```sh
env SDL_VIDEODRIVER=x11 ./build/bin/eden-cli "/path/to/game/base/exefs/main"
```

For Fire Emblem: Fortune's Weave, use these settings in **Emulation > Configure**:

| Setting | Value |
| --- | --- |
| Memory Layout | 12GB DRAM (Unsafe) |
| Console Mode | Handheld (1080p) or Docked (1440p) |
| GPU Mode | Fast |
| Force maximum clocks | On |
| Sync Memory Operations | On |
| ASTC Recompression | BC3 (Medium quality) |
| Asynchronous presentation | On |
| Asynchronous shader compilation | On |
| General > Debug > Enable Auto-Stub | On; check before launching the game |

Settings and data use `~/.config/eden` and `~/.local/share/eden`. Back up an existing
installation's settings and saves before testing another build. After a failure,
copy `~/.local/share/eden/log/eden_log.txt` before launching again.

## 5. Optional tests

```sh
cmake -S . -B build -DBUILD_TESTING=ON -DYUZU_TESTS=ON
cmake --build build --target tests --parallel 8
./build/bin/tests --reporter compact
```

The suite has known failures in the memory-tracker mock, shared timing state,
native-shader expectations and QMD-history expectations. Socket restrictions can
also cause controller/network test failures. Record the commit and complete test
output when reporting results.

## 6. Package an AppImage

The repository's [Build workflow](../../.github/workflows/build.yml) contains its
container-based packaging recipe. For a local build, first install `patchelf`:

```sh
sudo pacman -S --needed patchelf
```

For distribution to other PCs, use generic x86-64 runtime packages or an Arch
container. Optimized host packages, such as some distributions' x86-64-v4 glibc
builds, can add an AVX-512 requirement. Compiler prefix maps do not change runtime CPU requirements.
Other distributions and CPUs still need testing.

From the repository root after building, deploy into a **fresh** AppDir. The
packager otherwise reuses existing staged binaries, even after a rebuild.

```sh
stage_dir="$(mktemp -d /tmp/n2den-appimage.XXXXXX)"
curl -fL --retry 3 -o "$stage_dir/quick-sharun" \
  https://raw.githubusercontent.com/pkgforge-dev/Anylinux-AppImages/bacfcc005398d74e6454cdb8f494efe42e0bb60a/useful-tools/quick-sharun.sh
printf '%s  %s\n' \
  fa9dbe2532d568e1c1b8eeaba94d1c57109b776b5541d21b74d3f44b74faf4d3 \
  "$stage_dir/quick-sharun" | sha256sum -c -
chmod +x "$stage_dir/quick-sharun"

export APPDIR="$stage_dir/AppDir"
export OUTPATH="$PWD/artifacts"
export OUTNAME="n2den-linux-$(git rev-parse --short HEAD)-x86_64.AppImage"
export ICON="$PWD/dist/dev.eden_emu.eden.svg"
export DESKTOP="$PWD/dist/dev.eden_emu.eden.desktop"
export OPTIMIZE_LAUNCH=0 STRACE_MODE=0
export DEPLOY_OPENGL=1 DEPLOY_VULKAN=1
export DEPLOY_DATADIR=0 DEPLOY_LOCALE=0 DEPLOY_KF=0
export DEPLOY_QML=0 DEPLOY_QT_WEB_ENGINE=0
export ADD_HOOKS=wayland-is-broken.hook

"$stage_dir/quick-sharun" build/bin/eden build/bin/eden-cli
mkdir -p "$APPDIR/share/licenses/eden"
cp LICENSE.txt "$APPDIR/share/licenses/eden/"
cp -R LICENSES "$APPDIR/share/licenses/eden/"
"$stage_dir/quick-sharun" --make-appimage
cp dist/testers/README-n2den-linux.txt artifacts/README.txt
sha256sum "artifacts/$OUTNAME" > "artifacts/$OUTNAME.sha256"
```

This pins a known-good version of the packaging script. Its helper
downloads are checked by the script. Launch profiling is disabled so packaging
does not require a working FUSE device. If you change packaging-tool versions,
review their options and verify the new downloads.

Run the resulting image, including on a machine without FUSE:

```sh
./artifacts/n2den-linux-*.AppImage
./artifacts/n2den-linux-*.AppImage --appimage-extract-and-run
```

Before sharing, extract and audit the **final** AppImage for personal paths and
configuration. Include the corresponding source and dependency licenses. Build
source archives from tracked source files, omit Git metadata/build caches, and
normalize tar owners and zip metadata. Never include console keys, system files,
game dumps, saves, logs or game-derived captures/modules in a tester package.
