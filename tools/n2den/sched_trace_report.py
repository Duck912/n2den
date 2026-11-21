#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-frame timelines from a sched_trace result (debug server call or its dump file).

  py -I tools/n2den/sched_trace_report.py <sched_trace.json> [--frames N] [--all-threads]

For each frame (from one counted present to the next) prints, per thread of interest (the main
thread, ModuleSystemWorker*, BaseProcExecutor*, the presentation thread and any other thread that
ran more than 2% of the window), its run intervals in ms from the frame start as core:start-end,
and a one-line summary: main thread run time, worker run time (max over workers, and the sum),
executor and presentation run time, the time every core ran only its idle thread, the time at most
one core ran anything else, the average number of busy cores, and the order of the phases (each
group's first start and last end). Detailed frames also get one strip per group (0.5 ms a char). Ends with averages over all frames.

"Running" is the kernel's view: a guest thread is running from the switch that put it on a core
to the switch that took it off, including time it spends in HLE code on the host.
"""
import argparse
import json
import sys

IDLE = 0


def load(path):
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    if "result" in data and "cores" not in data:
        data = data["result"]
    if any("events" not in core for core in data["cores"]):
        raise SystemExit("this result has no events (events_in_result was false): use the file")
    return data


def run_intervals(data):
    """{thread id: [(core, start, end), ...]} in ticks from the window start, and per-core idle."""
    end_of_window = data["window_ticks"]
    by_thread = {}
    idle = []
    for core in data["cores"]:
        events = sorted(core["events"], key=lambda e: e[0])
        core_idle = []
        if not events:
            idle.append(core_idle)
            continue
        running = events[0][1]
        start = 0
        for ticks, out_id, in_id in events:
            # A dropped event would break the chain: trust the event's own "out" thread.
            del out_id
            if ticks > start:
                if running == IDLE:
                    core_idle.append((start, ticks))
                else:
                    by_thread.setdefault(running, []).append((core["core"], start, ticks))
            running = in_id
            start = ticks
        if end_of_window > start:
            if running == IDLE:
                core_idle.append((start, end_of_window))
            else:
                by_thread.setdefault(running, []).append((core["core"], start, end_of_window))
        idle.append(core_idle)
    return by_thread, idle


def all_idle(idle_per_core, cores, lo, hi):
    """Ticks in [lo, hi) during which at least `cores` cores ran their idle thread."""
    points = []
    for intervals in idle_per_core:
        for start, end in intervals:
            start, end = max(start, lo), min(end, hi)
            if end > start:
                points.append((start, 1))
                points.append((end, -1))
    points.sort()
    count = 0
    last = lo
    total = 0
    for ticks, delta in points:
        if count == cores:
            total += ticks - last
        count += delta
        last = ticks
    return total


def clip(intervals, lo, hi):
    out = []
    for core, start, end in intervals:
        start, end = max(start, lo), min(end, hi)
        if end > start:
            out.append((core, start, end))
    return out


def merge(intervals, gap):
    """Joins one thread's intervals separated by less than gap ticks (any core)."""
    out = []
    for core, start, end in sorted(intervals, key=lambda c: c[1]):
        if out and start - out[-1][2] < gap:
            out[-1] = (out[-1][0] if out[-1][0] == core else "*", out[-1][1], max(out[-1][2], end))
        else:
            out.append((core, start, end))
    return out


STRIP = " .:-=#"


def strip(intervals, lo, hi, bucket, width):
    """One character per bucket: how many of `width` threads ran in it, on average."""
    count = max(1, (hi - lo + bucket - 1) // bucket)
    busy = [0] * count
    for _, start, end in intervals:
        start, end = max(start, lo), min(end, hi)
        while start < end:
            index = (start - lo) // bucket
            stop = min(end, lo + (index + 1) * bucket)
            busy[index] += stop - start
            start = stop
    chars = []
    for value in busy:
        share = value / (bucket * width)
        chars.append(STRIP[min(len(STRIP) - 1, int(round(share * (len(STRIP) - 1))))]
                     if value else " ")
    return "".join(chars)


def group_of(name):
    if "MainThread" in name:
        return "main"
    if name.startswith("ModuleSystemWorker"):
        return "workers"
    if name.startswith("BaseProcExecutor"):
        return "executors"
    if name.startswith("Presentation"):
        return "presentation"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--frames", type=int, default=6, help="frames to print in detail")
    parser.add_argument("--all-threads", action="store_true",
                        help="detail every thread that ran in the frame")
    parser.add_argument("--merge-ms", type=float, default=0.2,
                        help="detail: join a thread's run intervals closer than this")
    parser.add_argument("--bucket-ms", type=float, default=0.5, help="strip resolution")
    args = parser.parse_args()
    data = load(args.trace)
    tps = data["ticks_per_second"]
    ms = lambda ticks: ticks * 1000.0 / tps  # noqa: E731
    names = {int(key): value for key, value in data["threads"].items()}
    name = lambda tid: names.get(tid, f"thread {tid}")  # noqa: E731
    cores = len(data["cores"])
    by_thread, idle = run_intervals(data)
    window = data["window_ticks"]

    total_run = {tid: sum(end - start for _, start, end in iv) for tid, iv in by_thread.items()}
    interest = set()
    for tid in by_thread:
        if group_of(name(tid)) or total_run[tid] > 0.02 * window:
            interest.add(tid)

    presents = sorted(p for p in data["presents"] if p[3] == 0)
    paced = sorted(p for p in data["presents"] if p[3] == 1)
    dropped = sum(core.get("dropped", 0) for core in data["cores"])
    print(f"window {ms(window):.1f} ms, {data['switches']} switches, dropped {dropped}, "
          f"{len(presents)} presents, {cores} cores")
    if dropped:
        print("WARNING: events were dropped; intervals after a drop on that core are wrong")
    print("\nwindow totals (ms run, % of window):")
    for tid in sorted(by_thread, key=lambda t: -total_run[t]):
        if total_run[tid] > 0.005 * window:
            print(f"  {name(tid):28s} {ms(total_run[tid]):8.1f} {100.0 * total_run[tid] / window:6.1f}%")
    idle_total = sum(end - start for core in idle for start, end in core)
    print(f"  {'(idle, all cores)':28s} {ms(idle_total):8.1f} "
          f"{100.0 * idle_total / window:6.1f}% of one core")

    if len(presents) < 2:
        print("\nfewer than two presents in the window: no frames")
        return 0
    groups = ("main", "workers", "executors", "presentation")
    sums = {key: 0.0 for key in ("frame", "main", "workers_max", "workers_sum", "executors_max",
                                 "presentation", "all_idle", "at_most_one_busy",
                                 "busy_cores", "pace")}
    frames = 0
    print()
    for index in range(len(presents) - 1):
        lo, hi = presents[index][0], presents[index + 1][0]
        frame = hi - lo
        detail = index < args.frames
        run = {}
        spans = {}
        for tid in by_thread:
            clipped = clip(by_thread[tid], lo, hi)
            if not clipped:
                continue
            run[tid] = sum(end - start for _, start, end in clipped)
            group = group_of(name(tid))
            if group:
                first, last = min(s for _, s, _ in clipped), max(e for _, _, e in clipped)
                old = spans.get(group)
                spans[group] = (min(first, old[0]), max(last, old[1])) if old else (first, last)
            if detail and (tid in interest or args.all_threads):
                pieces = " ".join(f"{core}:{ms(start - lo):.1f}-{ms(end - lo):.1f}"
                                  for core, start, end in merge(clipped, args.merge_ms * tps / 1000))
                print(f"    {name(tid):28s} {ms(run[tid]):5.1f} ms  {pieces}")
        by_group = {group: [run[tid] for tid in run if group_of(name(tid)) == group]
                    for group in groups}
        main_ms = ms(sum(by_group["main"]))
        workers_max = ms(max(by_group["workers"], default=0))
        workers_sum = ms(sum(by_group["workers"]))
        executors_max = ms(max(by_group["executors"], default=0))
        presentation = ms(sum(by_group["presentation"]))
        idle_ms = ms(all_idle(idle, cores, lo, hi))
        serial_ms = ms(all_idle(idle, cores - 1, lo, hi))
        busy_ms = ms(frame * cores - sum(min(e, hi) - max(s0, lo) for core in idle
                                         for s0, e in core if min(e, hi) > max(s0, lo)))
        pace = next((p[0] for p in paced if lo <= p[0] < hi), None)
        order = " > ".join(f"{group}[{ms(spans[group][0] - lo):.1f}-{ms(spans[group][1] - lo):.1f}]"
                           for group in sorted(spans, key=lambda g: spans[g][0]))
        pace_text = f" pace_done@{ms(pace - lo):.1f}" if pace is not None else ""
        print(f"frame {index:3d} @{ms(lo):7.1f}: {ms(frame):5.1f} ms | main {main_ms:5.1f} | "
              f"workers max {workers_max:5.1f} sum {workers_sum:5.1f} | exec max {executors_max:4.1f} | "
              f"pres {presentation:4.1f} | all-idle {idle_ms:4.1f} <=1 busy {serial_ms:4.1f} | "
              f"busy cores {busy_ms / ms(frame):3.1f}{pace_text} | {order}")
        if detail:
            bucket = max(1, int(args.bucket_ms * tps / 1000))
            print(f"    strips, one char per {args.bucket_ms} ms (' .:-=#' = share of the group's "
                  f"threads running):")
            for group in groups:
                members = [tid for tid in by_thread if group_of(name(tid)) == group]
                ivs = [iv for tid in members for iv in clip(by_thread[tid], lo, hi)]
                print(f"    {group:13s}|{strip(ivs, lo, hi, bucket, max(1, len(members)))}|")
            others = [iv for tid in by_thread if not group_of(name(tid))
                      for iv in clip(by_thread[tid], lo, hi)]
            print(f"    {'other':13s}|{strip(others, lo, hi, bucket, cores)}|")
            idle_ivs = [(0, s0, e0) for core in idle for s0, e0 in core]
            print(f"    {'idle (cores)':13s}|{strip(idle_ivs, lo, hi, bucket, cores)}|")
            print()
        frames += 1
        sums["frame"] += ms(frame)
        sums["main"] += main_ms
        sums["workers_max"] += workers_max
        sums["workers_sum"] += workers_sum
        sums["executors_max"] += executors_max
        sums["presentation"] += presentation
        sums["all_idle"] += idle_ms
        sums["at_most_one_busy"] += serial_ms
        sums["busy_cores"] += busy_ms / ms(frame)
    print(f"\naverages over {frames} frames (ms): " +
          ", ".join(f"{key} {value / frames:.2f}" for key, value in sums.items() if key != "pace"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
