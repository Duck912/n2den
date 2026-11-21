#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pan the camera in a scene and count one-frame brightness outliers (EDEN_N2DEN_LUMA_TRACE=1).

  py -I tools/n2den/pan_run.py <scene.json> --exe <eden.exe> --out <dir> [--env K=V ...]
      [--cycles 8] [--pan-frames 150] [--speed 0.8]

Boots the scene like scene_run.py (save restored, boot script, warm-up), then turns the camera
with the right stick: right for --pan-frames presented frames, still for 20, left, still, repeated
--cycles times. fps is sampled every 5 s while it plays. The luma trace's report lines from the pan
are summed: frames read back, frames with both neighbours, and one-frame outliers with their sizes
(those in frames that skipped draws while their pipelines were made are counted as "skipped").
run.json holds the totals, the per-report lines and the fps samples; eden_log.txt the whole log.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from debug_mcp import Emulator, EmulatorError  # noqa: E402
from scene_run import close, data_dir, log_cursor, restore_save, run_boot, screenshot, wait_for_game  # noqa: E402

LUMA = re.compile(r"Luma trace: (\d+) frames \((\d+) with both neighbours\), mean ([\d.]+), "
                  r"one-frame outliers (\d+)(?:, (\d+) without skipped draws)?(.*)$")


def pan_steps(cycles, frames, speed):
    steps = []
    for _ in range(cycles):
        steps.append({"right_stick": [speed, 0.0], "frames": frames})
        steps.append({"right_stick": [0.0, 0.0], "frames": 20})
        steps.append({"right_stick": [-speed, 0.0], "frames": frames})
        steps.append({"right_stick": [0.0, 0.0], "frames": 20})
    return steps


def summarise(lines):
    """Outliers in frames that skipped draws for pipelines still being made are counted apart
    ("skipped"): their objects were missing, not drawn from stale data."""
    total = {"reports": 0, "frames": 0, "contiguous": 0, "outliers": 0, "skipped": 0, "sizes": []}
    for line in lines:
        match = LUMA.search(line)
        if not match:
            continue
        outliers = int(match.group(4))
        clean = int(match.group(5)) if match.group(5) is not None else outliers
        total["reports"] += 1
        total["frames"] += int(match.group(1))
        total["contiguous"] += int(match.group(2))
        total["outliers"] += clean
        total["skipped"] += outliers - clean
        for item in match.group(6).split():
            _, _, size = item.partition(":")
            try:
                total["sizes"].append(size if size.endswith("s") else float(size))
            except ValueError:
                pass
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("--exe", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--env", action="append", default=[], help="K=V, repeatable")
    parser.add_argument("--port", type=int, default=6545)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--pan-frames", type=int, default=150)
    parser.add_argument("--speed", type=float, default=0.8)
    args = parser.parse_args()
    with open(args.scene, encoding="utf-8") as file:
        scene = json.load(file)
    exe = os.path.abspath(args.exe)
    root = data_dir(exe)
    os.makedirs(args.out, exist_ok=True)
    restore_save(scene, root)
    env = dict(os.environ)
    env.update({"EDEN_NX2_DEBUG_SERVER": f"127.0.0.1:{args.port}", "EDEN_N2DEN_LUMA_TRACE": "1"})
    for item in args.env:
        key, _, value = item.partition("=")
        env[key] = value
    started = time.time()
    process = subprocess.Popen([exe, "-g", scene["game"]], env=env, cwd=os.path.dirname(exe))
    emu = Emulator(f"127.0.0.1:{args.port}")
    result = {"scene": args.scene, "env": args.env, "cycles": args.cycles,
              "pan_frames": args.pan_frames, "speed": args.speed}
    try:
        wait_for_game(emu, process, scene.get("boot_timeout_s", 300))
        result["boot_s"] = round(time.time() - started, 1)
        run_boot(emu, scene, args.out)
        time.sleep(scene.get("warmup_s", 20))
        screenshot(emu, args.out, "pan-start")
        cursor = log_cursor(emu)
        steps = pan_steps(args.cycles, args.pan_frames, args.speed)
        reply = emu.call("input_sequence", {"steps": steps, "async": True}, 30)
        end_present = reply["first_present"] + reply["frames"]
        fps = []
        deadline = time.time() + reply["frames"] / 10 + 60
        while time.time() < deadline:
            status = emu.call("status", {"fps_window_ms": 5000}, 20)
            fps.append(status.get("fps"))
            if status.get("presents", 0) >= end_present:
                break
        time.sleep(5)  # the luma trace reports every 4 s
        lines = emu.call("log_tail", {"after": cursor, "limit": 5000}, 30).get("lines", [])
        lines = [line for line in lines if isinstance(line, str)]
        result["fps"] = fps
        result["luma_lines"] = [line for line in lines if "Luma trace" in line]
        result["present_lines"] = [line for line in lines if "Present " in line]
        result["luma"] = summarise(result["luma_lines"])
    finally:
        result["exit"] = close(process)
        emu.close()
        log = os.path.join(root, "log", "eden_log.txt")
        if os.path.exists(log):
            shutil.copy(log, os.path.join(args.out, "eden_log.txt"))
        with open(os.path.join(args.out, "run.json"), "w", encoding="utf-8") as file:
            json.dump(result, file, indent=1)
    fps_values = [value for value in result.get("fps", []) if value]
    print(json.dumps({"luma": {k: v for k, v in result.get("luma", {}).items() if k != "sizes"},
                      "sizes": result.get("luma", {}).get("sizes"),
                      "fps_mean": round(sum(fps_values) / len(fps_values), 1) if fps_values else None,
                      "exit": result["exit"]}))


if __name__ == "__main__":
    main()
