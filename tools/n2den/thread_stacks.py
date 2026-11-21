"""Dump every thread's call stack of a running (or frozen) n2Den/Eden process, then detach.

Uses cdb from the Windows SDK "Debugging Tools for Windows" in non-invasive mode (-pv): threads are
suspended only while the stacks are read and the process keeps running afterwards.

  python thread_stacks.py [--pid N | --name n2den.exe] [--symbols DIR] [--out FILE] [--frames 40]

Symbols: pass the directory holding the build's eden.pdb (a renamed exe still matches its PDB).
Microsoft system symbols are fetched into %LOCALAPPDATA%\\symcache.
"""
import argparse
import os
import re
import subprocess
import sys

CDB_CANDIDATES = [
    r"C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\cdb.exe",
    r"C:\Program Files\Windows Kits\10\Debuggers\x64\cdb.exe",
]


def find_pid(name_pattern: str) -> int:
    out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True).stdout
    rx = re.compile(name_pattern, re.I)
    hits = [(row.split('","')[0].strip('"'), int(row.split('","')[1])) for row in out.splitlines()
            if row.count('","') >= 2 and rx.search(row.split('","')[0])]
    if len(hits) != 1:
        sys.exit(f"expected one process matching {name_pattern!r}, found {hits}")
    return hits[0][1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int)
    ap.add_argument("--name", default=r"^(n2Den|eden).*\.exe$")
    ap.add_argument("--symbols", action="append", default=[])
    ap.add_argument("--out")
    ap.add_argument("--frames", type=int, default=40)
    args = ap.parse_args()

    cdb = next((c for c in CDB_CANDIDATES if os.path.exists(c)), None)
    if not cdb:
        sys.exit("cdb.exe not found: install the Windows SDK feature OptionId.WindowsDesktopDebuggers")
    pid = args.pid or find_pid(args.name)
    cache = os.path.join(os.environ.get("LOCALAPPDATA", "."), "symcache")
    sympath = ";".join(args.symbols + [f"srv*{cache}*https://msdl.microsoft.com/download/symbols"])
    commands = f".reload /f; ~*kn {args.frames}; qd"
    result = subprocess.run([cdb, "-pv", "-p", str(pid), "-y", sympath, "-c", commands],
                            capture_output=True, text=True, errors="replace", timeout=600)
    text = result.stdout
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    # Short summary: per thread, the first frames inside the emulator (not ntdll/kernelbase/ucrt).
    threads = re.split(r"\n(?=\s*[#.]?\s*\d+\s+Id: )", text)
    for block in threads:
        head = re.search(r"(\d+)\s+Id: ([0-9a-f]+)\.([0-9a-f]+)", block)
        if not head:
            continue
        frames = re.findall(r"^[0-9a-f]{2} (?:[0-9a-f`]+ [0-9a-f`]+|\(Inline Function\) [-`]+)\s+(\S+)", block, re.M)
        own = [f for f in frames if not re.match(r"(ntdll|KERNELBASE|kernel32|ucrtbase|msvcp|vcruntime|win32u|"
                                                  r"USER32|combase|nvoglv|nvgpucomp|nvwgf|Qt6\w+|msvcp\w*|std::)!?", f, re.I)]
        print(f"thread {head.group(1)} tid {int(head.group(3), 16)}: {frames[0] if frames else '?'}")
        for f in own[:6]:
            print(f"      {f}")
    if not args.out:
        print(text[-2000:])


if __name__ == "__main__":
    main()
