#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run one scripted scene and collect performance statistics. There are no save states: a scene
starts from the game's own save, restored from a snapshot before every run, and reaches the spot
with a scripted input sequence.

  py -I tools/n2den/scene_run.py <scene.json> --exe <eden.exe> --out <dir> [--env K=V ...]
  py -I tools/n2den/scene_run.py <scene.json> --exe <eden.exe> --boot-only   (leave it running)
  py -I tools/n2den/scene_run.py <scene.json> --exe <eden.exe> --hold        (... once in the scene)

Scene file:
  {"game": "<path>", "title_id": "0400...", "user": "<profile id>", "save": "<snapshot dir>",
   "boot": [{"sleep": 20}, {"steps": [{"buttons": ["A"], "frames": 6}]}, {"screenshot": "title"}],
   "warmup_s": 20, "measure_s": 40}

The save snapshot directory holds the title's save folder (named by title id). The exe's data
directory is its portable "user" folder when there is one, else %APPDATA%/eden. Statistics:
EDEN_BACK_THREAD_STATS=1 report lines logged during the measurement window, fps sampled every
5 s, the native rejection census (debug server "shaders"), and the whole log.
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from debug_mcp import Emulator, EmulatorError  # noqa: E402


def data_dir(exe):
    portable = os.path.join(os.path.dirname(exe), "user")
    return portable if os.path.isdir(portable) else os.path.join(os.environ["APPDATA"], "eden")


def restore_save(scene, root):
    target = os.path.join(root, "nand", "user", "save", "0000000000000000", scene["user"],
                          scene["title_id"])
    source = os.path.join(scene["save"], scene["title_id"])
    if not os.path.isdir(source):
        raise SystemExit(f"no save snapshot at {source}")
    if os.path.isdir(target):
        shutil.rmtree(target)
    shutil.copytree(source, target)


def wait_for_game(emu, process, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"the emulator exited with {process.returncode} before presenting")
        try:
            status = emu.call("status", {}, 5)
            if status.get("presents", status.get("renderer_frames", 0)):
                return status
        except EmulatorError:
            pass
        time.sleep(1)
    raise SystemExit("no presented frame within the boot timeout")


def screenshot(emu, out, name):
    if not out:
        return
    try:
        shot = emu.call("screenshot", {"timeout_s": 15}, 30)
        with open(os.path.join(out, f"{name}.png"), "wb") as file:
            file.write(base64.b64decode(shot["png_base64"]))
    except EmulatorError as error:
        print(f"screenshot {name}: {error}", file=sys.stderr)


def run_boot(emu, scene, out):
    for entry in scene.get("boot", []):
        if "sleep" in entry:
            time.sleep(entry["sleep"])
        elif "steps" in entry:
            frames = sum(step["frames"] for step in entry["steps"])
            emu.call("input_sequence", {"steps": entry["steps"]}, 30 + frames / 10)
        elif "screenshot" in entry:
            screenshot(emu, out, entry["screenshot"])


def log_cursor(emu):
    return emu.call("log_tail", {"limit": 1}, 10).get("next", 0)


def close(process, timeout=120):
    # The first close stops emulation (the JIT cache is written then), a later one closes the
    # window. Set the frontend's confirmStop to 2 (never ask) or a dialog blocks both.
    deadline = time.time() + timeout
    while time.time() < deadline:
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"(Get-Process -Id {process.pid}).CloseMainWindow() | Out-Null"],
                       check=False, capture_output=True)
        try:
            process.wait(10)
            return "closed"
        except subprocess.TimeoutExpired:
            pass
    process.kill()
    process.wait(30)
    return "killed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("--exe", required=True)
    parser.add_argument("--out")
    parser.add_argument("--env", action="append", default=[], help="K=V, repeatable")
    parser.add_argument("--port", type=int, default=6545)
    parser.add_argument("--boot-only", action="store_true")
    parser.add_argument("--hold", action="store_true",
                        help="play the boot script and the warm-up, then leave the game running")
    parser.add_argument("--no-restore", action="store_true")
    args = parser.parse_args()
    with open(args.scene, encoding="utf-8") as file:
        scene = json.load(file)
    exe = os.path.abspath(args.exe)
    root = data_dir(exe)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    if not args.no_restore:
        restore_save(scene, root)
    env = dict(os.environ)
    env.update({"EDEN_NX2_DEBUG_SERVER": f"127.0.0.1:{args.port}", "EDEN_BACK_THREAD_STATS": "1",
                "EDEN_NATIVE_CENSUS": "1"})
    for item in args.env:
        key, _, value = item.partition("=")
        env[key] = value
    started = time.time()
    process = subprocess.Popen([exe, "-g", scene["game"]], env=env, cwd=os.path.dirname(exe))
    emu = Emulator(f"127.0.0.1:{args.port}")
    status = wait_for_game(emu, process, scene.get("boot_timeout_s", 300))
    boot_s = time.time() - started
    if args.boot_only:
        print(json.dumps({"pid": process.pid, "boot_s": round(boot_s, 1), "title": status.get("title")}))
        return
    result = {"scene": scene, "env": args.env, "boot_s": round(boot_s, 1)}
    if args.hold:
        run_boot(emu, scene, args.out)
        time.sleep(scene.get("warmup_s", 20))
        screenshot(emu, args.out, "hold")
        print(json.dumps({"pid": process.pid, "boot_s": round(boot_s, 1)}))
        return
    try:
        run_boot(emu, scene, args.out)
        time.sleep(scene.get("warmup_s", 20))
        screenshot(emu, args.out, "measure-start")
        cursor = log_cursor(emu)
        fps = []
        end = time.time() + scene.get("measure_s", 40)
        while time.time() < end:
            fps.append(emu.call("status", {"fps_window_ms": 5000}, 20).get("fps"))
        lines = emu.call("log_tail", {"after": cursor, "limit": 5000}, 30).get("lines", [])
        screenshot(emu, args.out, "measure-end")
        result["fps"] = fps
        result["stats_lines"] = [line for line in lines if isinstance(line, str) and any(
            key in line for key in ("Back thread", "Macro memory reads", "Split ", "Fences:",
                                    "Vulkan worker", "Cache locks", "Invalidation", "Guest CPU"))]
        result["census"] = emu.call("shaders", {"limit": 1}, 30).get("census")
        result["status"] = emu.call("status", {}, 10)
    finally:
        result["exit"] = close(process)
        emu.close()
        if args.out:
            log = os.path.join(root, "log", "eden_log.txt")
            if os.path.exists(log):
                shutil.copy(log, os.path.join(args.out, "eden_log.txt"))
            with open(os.path.join(args.out, "run.json"), "w", encoding="utf-8") as file:
                json.dump(result, file, indent=1)
    print(json.dumps({k: result.get(k) for k in ("boot_s", "fps", "exit")}))


if __name__ == "__main__":
    main()
