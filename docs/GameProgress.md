# Game progress

Last updated: 2026-11-24.

This page tracks how far each tested Switch 2 game gets in n2den, per graphics backend, and what is known to
be wrong. It only lists what has actually been run; a game that is not here has not been tested.

Frame rates are from one development machine and one scene per game. Read them as "before and after" figures
for n2den's own changes, not as what another machine will reach.

## Status words

| Status | Meaning |
|---|---|
| **Playable** | Can be played normally: boots, menus, movies and gameplay work, no known crashes in play. |
| **In-game** | Reaches gameplay or its opening 3D sequence and renders it correctly as far as checked. |
| **Boots** | Starts and shows a picture, with known rendering or stability faults. |
| **Unstable** | Can crash, or hang the graphics driver. Not recommended. |
| **Not tested** | Never run on this backend, or not since the code changed substantially. |

## Summary

| Game | Emulated GPU | NVN-HLE (Vulkan) | Prepared launch |
|---|---|---|---|
| The Legend of Zelda: Ocarina of Time (Switch 2) | In-game | In-game | Works |
| Fire Emblem: Fortune's Weave | In-game | In-game | CPU code only |
| Mario Kart World | Not tested | Unstable | Not usable |
| Fast Fusion | Playable | Playable | Works |
| Kirby and the Forgotten Land – Nintendo Switch 2 Edition + Star-Crossed World | Playable | Playable | Works |
| Nintendo Switch Sports Resort | Not tested | Boots | Not tested |
| Orbitals Deluxe Edition | Not usable | In-game, slow | Not tested |
| Donkey Kong Bananza | In-game | In-game | Not tested |
| Xenoblade Chronicles 2 – Nintendo Switch 2 Edition | Not tested | Boots | Not tested |

"Prepared launch" means the game's CPU code and graphics pipelines are built before play starts, so that
nothing has to be compiled while playing.

The NVN-HLE renderer for Direct3D 12 is not listed: it only clears the screen and shows the window.

## The Legend of Zelda: Ocarina of Time (Switch 2)

Test scene: Kokiri Forest, from a save.

- **Emulated GPU:** in-game, about 39–41 fps in the test scene.
- **NVN-HLE (Vulkan):** in-game, about 43 fps in the test scene. The frame is limited by the thread that
  replays the game's draw calls, not by the game's own CPU work.
- **Prepared launch:** with a pipeline list for the game, a first launch compiles nothing during play (no CPU
  code, no graphics or compute pipelines) and skips no draws. Boot takes about 28 s on a first launch and
  7–9 s afterwards, against about 13 s unprepared.
- **Upscaling:** the game's own upscaling network runs; NVIDIA DLSS / DLAA can replace it.

Known issues:

- Only one route (boot to Kokiri Forest) has been measured. Other areas will use pipelines the list does not
  contain yet; those compile on first use.
- The Kokiri load still loses 1.4–2.0 s to a few long frames (texture set-up and loading cached CPU code).

## Fire Emblem: Fortune's Weave

Test scene: the opening sequence, which plays without input.

- **NVN-HLE (Vulkan):** in-game. The opening 3D sequence renders correctly as far as checked, at about 56 fps
  on average, with no rejected shaders or pipelines.
- **Prepared launch:** the CPU code of the whole game is compiled at boot (about 2.5 million blocks in 6–7 s).
  There is no pipeline list for this game yet, so several hundred graphics pipelines still compile during
  the first run of the sequence, with visible pop-in while they do.
- **Upscaling:** the game uses its own upscaling network.
- **Fixed:** blocky splotches on the map on NVN-HLE. The ambient-occlusion pass writes through a storage image
  whose handle sits in the NVN handle bank (c2); the shader translator lost that handle and dropped every
  store, so a later pass read leftover memory.

Known issues:

- Some draws still read texture handles that name no texture, and a small number of operations per frame are
  not supported yet. No visible fault has been tied to them so far.
- A cold first boot has a few very long frames (the worst close to 0.9 s) while pipelines compile.

## Mario Kart World

Test scene: the race menu, whose background plays a 3D race without input.

- **NVN-HLE (Vulkan):** unstable. The game boots, reaches the race menu and plays the background race at about
  40–50 fps, with no rejected shaders. Some graphics are still missing, and many pipelines compile during play
  (draws are skipped until they are ready).
- **Fixed:** GPU memory used to grow without limit until the game crashed after about 3 minutes. The game
  maps more and more texture-atlas tiles onto the same memory, and n2den backed each mapping with its own
  memory; tiles that map the same bytes now share it, and atlas memory stays flat.
- **Known issues:** after several minutes in the menu scene the system can still run out of RAM: the game's own
  memory plus GPU memory close to the driver's budget, which the driver then moves into system RAM. ASTC
  textures are held as BC3, four times their size on the console. Rare graphics driver hangs have been seen
  (once on the title screen during a pipeline compile, once in the menu scene with an experimental change);
  their cause is not known.
- Not recommended for play yet. Run it only with plenty of free system RAM, and close the emulator if the
  graphics driver resets.

## Fast Fusion

- **Emulated GPU:** playable. The game's own DLSS upscaling is sharp in every mode, with no ghosting in Ultra
  Quality.
- **NVN-HLE (Vulkan):** playable with PC DLSS. Races render correctly across the game's quality modes. Fixed on
  the way: a GPU hang at race start, a washed-out race and a black or splotchy race (see
  [nvn-hle-debugging.md](nvn-hle-debugging.md)).
- **Prepared launch:** works. CPU code is prebuilt (941,676 blocks) and pipelines rebuild before the first
  frame from a state list once the game has been played on that machine (1,182 pipelines, none compiled during
  a race). Pipelines for a machine that has never run the game are not covered yet: the game keeps its shaders in its own `SHN!` files (`romfs/shaders/*.shdr`, code
  LZ4-compressed), which the prepare step can't read yet, so it finds 0 of the 419 stages a list needs.
- **Known issues on NVN-HLE:** a boot crash (device lost on the first submission, about one boot in five) is
  mitigated by giving feedback scratch images their own memory; still being monitored. Without PC DLSS the game
  falls back to FSR1, because its own DLSS network has no runner under NVN-HLE.

## Kirby and the Forgotten Land – Nintendo Switch 2 Edition + Star-Crossed World

Tested from a repacked `.nx` with the Star-Crossed World add-on; title screen, notice, difficulty select,
opening movie, first level and in-game movies.

- **Emulated GPU:** playable. Boots, renders the right way up and plays.
- **NVN-HLE (Vulkan):** playable. Title, menus, movies and gameplay render correctly; no crashes in play.
- **Prepared launch:** works. CPU code is prebuilt, and pipelines rebuild before the first frame once the game
  has been played on that machine. On a machine that has never run it, the prepare step finds only 37 of the
  1,183 stages a list needs in the game's files (the add-on is scanned too); the rest are built at run time.
- **Needed:** the game uses an older SDK (Generic_19_3_7), and its Switch 2 data lives in the add-on, so the
  add-on must be in the container. Packers that wrote its RomFS folders unlinked are handled.

## Nintendo Switch Sports Resort

Tested with update 1.1.0: boot, intro screens, opening movie and title screen.

- **NVN-HLE (Vulkan):** boots. The intro screens, the opening movie (VP9) and the title screen show; nothing
  past the title screen has been checked.
- **Fixed on the way:** a crash at boot on Switch 2 system data the game asks for but a PC does not have (an
  empty archive is now returned), a black picture with sound after the intro (the movie decoder refused the
  VP9 movie), and shaders rejected for constant bank 24 reads and a jump table.
- **Known issues:** one buffer set-up call fails about 20 s in; no visible effect has been tied to it.

## Orbitals Deluxe Edition

Tested with its update (version code 327680): boot, title screen, opening movie and the first minutes of play.
Unreal Engine 5.

- **NVN-HLE (Vulkan):** boots to the title screen with picture and sound at full speed; the opening movie
  plays; in-game runs at about 20 frames per second with frame-time spikes.
- **Emulated GPU:** stops after a few frames (a display timestamp query and a compute dispatch form are not
  implemented).
- **Fixed on the way:** a boot panic on the Switch 2 audio codec session, a stall after about 40 frames (the
  game's frame flip tracking reads a frame history only the system's own NVN driver filled), a pink and
  then black title screen (texture buffers, texture descriptors that outlive their texture object, texture
  and sampler handles stored apart, and layered rendering into a 3D colour table), no sound (the game mixes
  into a Switch 2 sound mixer stream that was not played), memory growing until the system ran out
  (recorded command lists were never released), and draws skipped for a bindless gradient texture sample.
- **Known issues:** characters draw in a flat colour and one light shaft is dithered (some texture handles
  resolve to no texture); the picture is limited by the graphics card in play (many copies of the depth
  buffer per frame); one freeze while loading a level (the game stopped on "zero-sized buffer").

## Donkey Kong Bananza

Tested to the first level.

- **Emulated GPU:** in-game. Reaches the first level and renders it correctly.
- **NVN-HLE (Vulkan):** in-game. Reaches the first level and renders it correctly.
- **Fixed on the way:** a graphics device loss while the level loads (a one-texel placeholder copied in the
  wrong format), a black floor on the emulated GPU (stores made through pointer tables the game builds on the
  GPU), and a long list of per-draw costs on both backends.
- **Speed:** below full speed on both backends. The NVN-HLE backend went from a slideshow to more than six
  times its earlier frame rate, and the emulated GPU gained about half again; there is still headroom to find.
- **Option:** the built-in add-on "Single-interval vsync for Donkey Kong Bananza" in the game's settings.

## Xenoblade Chronicles 2 – Nintendo Switch 2 Edition

Tested from a `.nup` container with the base game, update and add-ons: boot, intro and title screen.

- **NVN-HLE (Vulkan):** boots. Loads (about 30 s, most of it unpacking the game data from the container),
  then runs the intro and the title screen at full speed. The intro scene renders with correct colours, but
  it does not reach the screen yet: only the title text and menus show, over black.
- **Emulated GPU:** not tested past the loading screen.
- **Fixed on the way:** a crash at boot (the game reserves two very large address ranges for its memory
  allocator, which did not fit), a crash about 40 s in (the game reads its data through the base program's
  storage, which the container does not hold separately), and wrong colours in the intro (the game's
  colour-grading tables are stored in a different memory layout from the one n2den assumed).
- **Known issues:** the final upscale of the 3D scene to the screen is missing (black background).
