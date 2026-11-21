#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Summarise scene_run.py results: one row per run, averaged over the statistics reports
(EDEN_BACK_THREAD_STATS=1, every 4 s) logged in its measurement window.

  py -I tools/n2den/scene_report.py <run dir> [<run dir> ...] [--buckets N] [--census N]
"""
import argparse
import json
import os
import re
import statistics

NUMBER = r"([0-9.]+)"


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def parse(lines):
    reports = {"draws_s": [], "busy": [], "us_draw": [], "spans": {}, "handles": {},
               "macro": [], "finish": [], "wait_tick": [], "drain_macro": []}
    for line in lines:
        text = line.split(":", 3)[-1] if "Render_Vulkan" in line else line
        if m := re.search(r"Back thread: " + NUMBER + r" draws/s; busy " + NUMBER + r"%", line):
            reports["draws_s"].append(float(m[1]))
            reports["busy"].append(float(m[2]))
            if u := re.search(r"\(" + NUMBER + r" us/draw", line):
                reports["us_draw"].append(float(u[1]))
        elif "Back thread draw spans" in line:
            for name, value in re.findall(r"([a-z ]+?) " + NUMBER + r"%", line.split("):", 1)[-1]):
                reports["spans"].setdefault(name.strip(" ,"), []).append(float(value))
        elif "Back thread handles by source" in line:
            if m := re.search(r"handle memo hits " + NUMBER + r"/s, misses " + NUMBER +
                              r"/s, verify mismatches ([0-9]+)", line):
                reports.setdefault("memo", []).append((float(m[1]), float(m[2]), int(m[3])))
            for name, value in re.findall(r"(handle programs|buffer arrays|constant-buffer arrays|"
                                          r"rest \(plain handles, samplers\)) " + NUMBER + "%", line):
                reports["handles"].setdefault(name, []).append(float(value))
        elif m := re.search(r"Macro memory reads: " + NUMBER + r"/s, " + NUMBER + r" words each, " +
                            NUMBER + r" us/s .*GPU-modified ranges " + NUMBER + r"/s, " + NUMBER +
                            r" us/s", line):
            reports["macro"].append(tuple(float(m[i]) for i in range(1, 6)))
        elif "Back thread waits:" in line:
            if m := re.search(r"Finish " + NUMBER + r"/s " + NUMBER + r" us/s", line):
                reports["finish"].append((float(m[1]), float(m[2])))
            if m := re.search(r"Wait\(tick, also inside Finish\) " + NUMBER + r"/s " + NUMBER +
                              r" us/s", line):
                reports["wait_tick"].append((float(m[1]), float(m[2])))
        elif "Split drains by caller" in line:
            if m := re.search(r"macro memory " + NUMBER + r"/s " + NUMBER + r" us/s", line):
                reports["drain_macro"].append((float(m[1]), float(m[2])))
    return reports


def pair_mean(pairs, index):
    return mean([p[index] for p in pairs])


def fmt(value, digits=1):
    return "-" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--buckets", type=int, default=8)
    parser.add_argument("--census", type=int, default=0)
    args = parser.parse_args()
    for run in args.runs:
        with open(os.path.join(run, "run.json"), encoding="utf-8") as file:
            result = json.load(file)
        r = parse(result.get("stats_lines", []))
        fps = mean(result.get("fps") or [])
        busy = mean(r["busy"])
        back_ms = busy / 100 * 1000 / fps if busy is not None and fps else None
        print(f"{os.path.basename(run)}: {fmt(fps)} fps, {fmt(mean(r['draws_s']), 0)} draws/s, "
              f"back busy {fmt(busy)}% = {fmt(back_ms)} ms/frame, {fmt(mean(r['us_draw']), 2)} us/draw "
              f"({len(r['busy'])} reports, exit {result.get('exit')})")
        spans = sorted(((mean(v), k) for k, v in r["spans"].items()), reverse=True)[:args.buckets]
        print("  spans: " + ", ".join(f"{k} {fmt(v)}%" for v, k in spans))
        if r["handles"]:
            print("  handles: " + ", ".join(f"{k} {fmt(mean(v), 2)}%" for k, v in r["handles"].items()))
        if r.get("memo"):
            print(f"  handle memo: hits {fmt(pair_mean(r['memo'], 0), 0)}/s, misses "
                  f"{fmt(pair_mean(r['memo'], 1), 0)}/s, mismatches {sum(m[2] for m in r['memo'])}")
        if r["macro"]:
            print(f"  macro reads {fmt(pair_mean(r['macro'], 0))}/s x {fmt(pair_mean(r['macro'], 1))} "
                  f"words, {fmt(pair_mean(r['macro'], 2), 0)} us/s; GPU-modified "
                  f"{fmt(pair_mean(r['macro'], 3))}/s {fmt(pair_mean(r['macro'], 4), 0)} us/s; "
                  f"drains {fmt(pair_mean(r['drain_macro'], 0))}/s {fmt(pair_mean(r['drain_macro'], 1), 0)} us/s; "
                  f"Finish {fmt(pair_mean(r['finish'], 0))}/s {fmt(pair_mean(r['finish'], 1), 0)} us/s; "
                  f"Wait(tick) {fmt(pair_mean(r['wait_tick'], 0))}/s {fmt(pair_mean(r['wait_tick'], 1), 0)} us/s")
        census = result.get("census") or {}
        if args.census and census:
            print(f"  census drops: {census.get('drops')}")
            for row in census.get("rejections", [])[:args.census]:
                print(f"    {row['site']:<9} {row['opcode'] or '-':>6} {row['programs']:>4} prog "
                      f"{row['events']:>6} ev  {row['reason'][:90]}")


if __name__ == "__main__":
    main()
