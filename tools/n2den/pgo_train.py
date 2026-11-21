#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Train a profile for an N2DEN_PGO=GENERATE build (docs/PGO.md).

  py -I tools/n2den/pgo_train.py --exe <instrumented eden.exe> --profraw <dir> --out <dir> \
      [--merge <n2den.profdata>] pan:<scene.json> scene:<scene.json> scene-noreload:<scene.json> ...

Each scene runs in its own session: pan:<scene> with pan_run.py (camera pan), scene:<scene> with
scene_run.py (boot script, warm-up and measurement window), scene-noreload:<scene> the same without
restoring a save first (for scenes that have no save snapshot). Append @K=V,K=V to a scene to set
environment variables for that session only, e.g. scene:fe.json@EDEN_NVN_HLE=0,EDEN_JIT_CACHE=0 to
train the emulated GPU path with a cold JIT cache. Both scripts close the emulator
through its window, which is what makes the profile runtime write its counters; a session that
crashes, is killed or writes no .profraw stops the training. With --merge the .profraw files are
merged with llvm-profdata at the end (--llvm-profdata, default: next to clang-cl, then PATH).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = {"pan": ("pan_run.py", []), "scene": ("scene_run.py", []),
         "scene-noreload": ("scene_run.py", ["--no-restore"])}


def find_profdata(explicit):
    if explicit:
        return explicit
    for candidate in (r"C:\Program Files\LLVM\bin\llvm-profdata.exe", shutil.which("llvm-profdata")):
        if candidate and os.path.exists(candidate):
            return candidate
    raise SystemExit("llvm-profdata not found; pass --llvm-profdata")


def emulator_running():
    if os.name != "nt":
        return False
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq eden.exe"], capture_output=True, text=True).stdout
    return "eden.exe" in out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--profraw", required=True, help="folder for the .profraw files")
    parser.add_argument("--out", required=True, help="folder for each session's run output")
    parser.add_argument("--merge", help="write the merged profile here")
    parser.add_argument("--llvm-profdata")
    parser.add_argument("scenes", nargs="+", help="pan:<scene.json>, scene:<scene.json> or scene-noreload:<scene.json>")
    args = parser.parse_args()
    os.makedirs(args.profraw, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    pattern = os.path.join(args.profraw, "n2den-%m-%p.profraw")
    for item in args.scenes:
        kind, _, scene = item.partition(":")
        session_env = []
        head, sep, tail = scene.rpartition("@")
        if sep and "=" in tail:
            scene = head
            session_env = [pair for pair in tail.split(",") if pair]
        if kind not in TOOLS or not scene:
            raise SystemExit(f"bad scene argument '{item}' (want pan:, scene: or scene-noreload:)")
        if emulator_running():
            raise SystemExit("an eden.exe is already running; close it first (it may share the user folder)")
        tool, extra = TOOLS[kind]
        name = f"{kind}-{os.path.splitext(os.path.basename(scene))[0]}"
        if session_env:
            name += "-" + "-".join(pair.replace("=", "") for pair in session_env).lower()
        out = os.path.join(args.out, name)
        before = len(glob.glob(os.path.join(args.profraw, "*.profraw")))
        started = time.time()
        with open(out + ".txt", "w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-I", os.path.join(HERE, tool), scene, "--exe", args.exe,
                            "--out", out, "--env", f"LLVM_PROFILE_FILE={pattern}",
                            *[arg for pair in session_env for arg in ("--env", pair)], *extra],
                           stdout=log, stderr=subprocess.STDOUT, check=False)
        after = len(glob.glob(os.path.join(args.profraw, "*.profraw")))
        try:
            with open(os.path.join(out, "run.json"), encoding="utf-8") as file:
                exit_kind = json.load(file).get("exit")
        except (OSError, ValueError):
            exit_kind = None
        print(f"{name}: {time.time() - started:.0f} s, exit {exit_kind}, .profraw files {before} -> {after}",
              flush=True)
        if exit_kind != "closed" or after <= before:
            raise SystemExit(f"training stopped at {name}; see {out}.txt")
    if args.merge:
        files = sorted(glob.glob(os.path.join(args.profraw, "*.profraw")))
        subprocess.run([find_profdata(args.llvm_profdata), "merge", "-o", args.merge, *files], check=True)
        print(f"merged {len(files)} files into {args.merge} ({os.path.getsize(args.merge) >> 20} MiB)")


if __name__ == "__main__":
    main()
