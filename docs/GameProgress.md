# Game progress

Last updated: 2026-11-18.

This page tracks how far each tested Switch 2 game gets in n2den, per graphics backend, and what is known to
be wrong. It only lists what has actually been run; a game that is not here has not been tested.

Frame rates are from one development machine and one scene per game. Read them as "before and after" figures
for n2den's own changes, not as what another machine will reach.

## Status words

| Status | Meaning |
|---|---|
| **In-game** | Reaches gameplay or its opening 3D sequence and renders it correctly as far as checked. |
| **Boots** | Starts and shows a picture, with known rendering or stability faults. |
| **Unstable** | Can crash, or hang the graphics driver. Not recommended. |
| **Not tested** | Never run on this backend, or not since the code changed substantially. |

## Summary

| Game | Emulated GPU | NVN-HLE (Vulkan) | Prepared launch |
|---|---|---|---|
| The Legend of Zelda: Ocarina of Time (Switch 2) | In-game | In-game | Works |
| Fire Emblem: Fortune's Weave | Not tested | In-game | CPU code only |
| Mario Kart World | Not tested | Unstable | Not usable |
| Fast Fusion | In-game | In-game | Not tested |

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

Known issues:

- Some draws still read texture handles that name no texture, and a small number of operations per frame are
  not supported yet. No visible fault has been tied to them so far.
- A cold first boot has a few very long frames (the worst close to 0.9 s) while pipelines compile.

## Mario Kart World

- **NVN-HLE (Vulkan):** unstable. The game boots to its title screen at 33–50 fps. On some title-screen
  backdrops one graphics pipeline, compiled during play, is followed within milliseconds by a graphics driver
  hang. Whether the cause is n2den's shader translation, the data that draw reads, or the driver is not known.
- Do not run this game for now: after the driver reset the emulator used to wait on the graphics card without
  a time limit.

## Fast Fusion

- **Emulated GPU:** in-game. The game's own DLSS upscaling is sharp in every mode, with no ghosting in Ultra
  Quality.
- **NVN-HLE (Vulkan):** in-game with PC DLSS. Races render correctly across the game's quality modes. Fixed on
  the way: a GPU hang at race start, a washed-out race and a black or splotchy race (see
  [nvn-hle-debugging.md](nvn-hle-debugging.md)).
- **Known issues on NVN-HLE:** an intermittent crash on boot (device lost on the first submission, before the
  menu). Without PC DLSS the game falls back to FSR1, because its own DLSS network has no runner under NVN-HLE.
