#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""MCP server (stdio) for n2Den's in-process debug server (docs/debug-mcp.md).

Run n2Den or eden-cli with EDEN_NX2_DEBUG_SERVER=127.0.0.1:6545, then register this
script as a stdio MCP server for the agent:  py -I tools/n2den/debug_mcp.py [--server 127.0.0.1:6545]
or run one request from a shell:
  py -I tools/n2den/debug_mcp.py --call '{"method":"screenshot","params":{}}' --out shot.png

Standard library only. Each tool call is forwarded as one JSON line to the emulator, which may
be restarted between calls: the adapter reconnects on demand.
"""

import argparse
import base64
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
DEFAULT_SERVER = "127.0.0.1:6545"
HERE = os.path.dirname(os.path.abspath(__file__))

BUTTONS = ["A", "B", "X", "Y", "L", "R", "ZL", "ZR", "Plus", "Minus", "Up", "Down", "Left", "Right",
           "StickL", "StickR"]
STICK = {"type": "array", "items": {"type": "number", "minimum": -1, "maximum": 1},
         "minItems": 2, "maxItems": 2, "description": "[x, y], up is +y"}

TOOLS = [
    {
        "name": "status",
        "description": "Game title, program id, build, renderer, paused state, presented frame "
                       "count and fps (measured over fps_window_ms), dump directory, live log "
                       "filter and the shader hashes rejected at runtime.",
        "inputSchema": {
            "type": "object",
            "properties": {"fps_window_ms": {"type": "integer", "minimum": 0, "maximum": 5000}},
        },
    },
    {
        "name": "screenshot",
        "description": "PNG of the next presented frame (16:9, default 720 lines). Fails when no "
                       "frame is presented within timeout_s (paused or loading).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "height": {"type": "integer", "minimum": 144, "maximum": 2160},
                "save": {"type": "boolean", "description": "Also write it into the dump directory."},
                "timeout_s": {"type": "integer", "minimum": 1, "maximum": 30},
            },
        },
    },
    {
        "name": "log_tail",
        "description": "Log lines kept in memory since the game booted. Without 'after', the newest "
                       "'limit' lines; with 'after' (the 'next' of an earlier call), the lines "
                       "logged since, oldest first. 'contains' keeps lines with that substring.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "after": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1, "maximum": 5000},
                "contains": {"type": "string"},
            },
        },
    },
    {
        "name": "log_filter",
        "description": "Replaces the live log filter, e.g. '*:Info HW.GPU:Debug Shader:Debug'. "
                       "Returns the previous one. Lines filtered out are never logged.",
        "inputSchema": {
            "type": "object",
            "properties": {"filter": {"type": "string"}},
            "required": ["filter"],
        },
    },
    {
        "name": "capture_frame",
        "description": "EDEN_NX2_FRAME_DUMP for the next frame, now: every pass colour target, "
                       "dispatch storage image and the present source as PNG and .raw.zst, plus "
                       "manifest.jsonl. 'sampled' also reads back every sampled texture with its "
                       "sampler state; 'buffers' records every draw's and dispatch's uniform and "
                       "storage buffers (kind: buffer, buffer_<hash>.bin.zst) and dispatch group "
                       "counts (kind: launch). Returns "
                       "the directory, summary, entry counts and the first manifest entries. "
                       "'index' builds index.html with frame_dump_index.py.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sampled": {"type": "boolean"},
                "buffers": {"type": "boolean"},
                "timeout_s": {"type": "integer", "minimum": 1, "maximum": 600},
                "manifest_lines": {"type": "integer", "minimum": 0, "maximum": 5000},
                "index": {"type": "boolean"},
            },
        },
    },
    {
        "name": "capture_image",
        "description": "One PNG of a capture made in this session, by entry name (the manifest "
                       "'name', without extension), e.g. '0042_pass007_d120-180_rt0'.",
        "inputSchema": {
            "type": "object",
            "properties": {"directory": {"type": "string"}, "name": {"type": "string"}},
            "required": ["directory", "name"],
        },
    },
    {
        "name": "reject_shaders",
        "description": "EDEN_NX2_REJECT_HASH at runtime: rejects (or with reject=false restores) "
                       "native programs by native program key (shader dump names) or shader hash "
                       "(frame dump 'shaders'); their cached pipelines are dropped at the next "
                       "frame end. With no hashes, lists the rejected ones.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "hashes": {"type": "array", "items": {"type": "string"}},
                "reject": {"type": "boolean"},
                "timeout_s": {"type": "integer", "minimum": 0, "maximum": 60},
            },
        },
    },
    {
        "name": "sample",
        "description": "Sampling profiler: pauses the named host threads about 1000 times a second "
                       "for the given seconds and returns where they were (self, self <- caller, "
                       "inclusive counts per function). Also writes the report and its folded "
                       "stacks to the dump directory. Windows only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "threads": {"type": "string",
                            "description": "Comma separated thread names; default GPU,GPUBack"},
                "seconds": {"type": "number", "minimum": 1, "maximum": 60},
                "max_chars": {"type": "integer", "minimum": 0},
            },
        },
    },
    {
        "name": "guest_threads",
        "description": "Guest CPU time per thread of the game: snapshots the kernel's per-thread "
                       "CPU time, waits the given seconds and returns each thread's milliseconds "
                       "and percent of one guest core in the window (largest first) with name, "
                       "priority, cores, affinity and state; threads that started or exited in "
                       "the window are flagged. Also returns per-core idle time. Read only; the "
                       "game keeps running.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "seconds": {"type": "number", "minimum": 1, "maximum": 60},
            },
        },
    },
    {
        "name": "jit_hot",
        "description": "Where the JIT time goes, by guest code: samples the instruction pointer "
                       "of the six CPUCore_N host threads about 1000 times a second (no stack "
                       "walk) and resolves each sample to the guest block whose host code it is "
                       "in. Returns per-core and total shares of in-block / dispatcher (prelude, "
                       "between blocks) / outside JIT code, the top blocks (guest PC, size, "
                       "module), coverage curves (blocks and guest bytes for 50-99% of in-block "
                       "samples) and the same by guest 4 KiB page, module and exported function; "
                       "writes the full lists to the dump directory. Needs EDEN_JIT_HOT_MAP=1 at "
                       "launch. Windows x64 only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "seconds": {"type": "number", "minimum": 1, "maximum": 60},
                "period_us": {"type": "integer", "minimum": 250, "maximum": 100000},
                "top": {"type": "integer", "minimum": 1, "maximum": 5000},
            },
        },
    },
    {
        "name": "sched_trace",
        "description": "Who runs when: records every guest context switch on every core "
                       "(CoreTiming ticks, thread out, thread in; 0 = idle) and every frame "
                       "presented through nvn-hle for the given seconds, and returns them with "
                       "thread names. Also writes them to the dump directory; "
                       "tools/n2den/sched_trace_report.py turns the file into per-frame "
                       "timelines. Read only; the game keeps running.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "seconds": {"type": "number", "minimum": 0.1, "maximum": 5},
                "events_in_result": {"type": "boolean"},
            },
        },
    },
    {
        "name": "read_memory",
        "description": "Reads up to 64 KiB of the game's guest virtual memory, read only and "
                       "while it runs. 'as' formats it: hex (default), u32, u64 (hex strings) or "
                       "f32. An unmapped page fails with the first unmapped address. Addresses "
                       "are hex strings ('0x8012345000') or numbers.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "address": {"type": ["string", "integer"]},
                "size": {"type": "integer", "minimum": 1, "maximum": 65536},
                "as": {"type": "string", "enum": ["hex", "u32", "u64", "f32"]},
            },
            "required": ["address", "size"],
        },
    },
    {
        "name": "search_memory",
        "description": "Finds a byte pattern (hex pairs in memory order: little endian, so 1.0f "
                       "is '0000803f') in the mapped guest memory of the heap region (default) or "
                       "the whole address space, optionally narrowed by start and end. Returns up "
                       "to max_hits (at most 256) addresses; it stops after budget_ms and then "
                       "returns 'next', the start for the following call.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "start": {"type": ["string", "integer"]},
                "end": {"type": ["string", "integer"]},
                "max_hits": {"type": "integer", "minimum": 1, "maximum": 256},
                "regions": {"type": "string", "enum": ["heap", "all"]},
                "budget_ms": {"type": "integer", "minimum": 1, "maximum": 10000},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "write_memory",
        "description": "For experiments only: writes 1 to 4096 bytes of the game's guest memory "
                       "and returns what was there before. Refused unless the emulator was "
                       "started with EDEN_NX2_DEBUG_WRITE=1 (status shows write_memory). Give "
                       "'hex' (byte pairs in memory order) or 'f32' (floats, little endian). "
                       "Restore the 'before' bytes when done.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "address": {"type": ["string", "integer"]},
                "hex": {"type": "string"},
                "f32": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["address"],
        },
    },
    {
        "name": "read_gpu_memory",
        "description": "Reads up to 64 KiB of GPU virtual memory (e.g. query report and semaphore "
                       "buffers the game reads back) at the next frame end, as the GPU sees it, "
                       "without flushing GPU caches. Tries the bound channel's address space "
                       "first; 'as' as for read_memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "gpu_address": {"type": ["string", "integer"]},
                "size": {"type": "integer", "minimum": 1, "maximum": 65536},
                "as": {"type": "string", "enum": ["hex", "u32", "u64", "f32"]},
                "timeout_s": {"type": "integer", "minimum": 1, "maximum": 60},
            },
            "required": ["gpu_address", "size"],
        },
    },
    {
        "name": "input",
        "description": "Holds buttons and sticks on player 1 (or the handheld controller) for "
                       "'frames' presented frames from the next present on, then releases; returns "
                       "when done. Buttons are added to the owner's; a given stick replaces the "
                       "owner's. Fails and releases when frames stop (paused or loading) for "
                       "timeout_s (default frames / 20 + 10).",
        "inputSchema": {
            "type": "object",
            "properties": {**{
                "buttons": {"type": "array", "items": {"type": "string", "enum": BUTTONS}},
                "left_stick": STICK,
                "right_stick": STICK,
                "frames": {"type": "integer", "minimum": 1, "maximum": 36000},
            },
                           "timeout_s": {"type": "integer", "minimum": 1, "maximum": 900}},
            "required": ["frames"],
        },
    },
    {
        "name": "input_sequence",
        "description": "Runs input steps back to back, each for exactly its 'frames' presented "
                       "frames, so a camera pan or walk replays identically for an A/B test. A "
                       "step without buttons or sticks leaves the owner's input alone. At most "
                       "1000 steps and 36000 frames.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "steps": {"type": "array", "minItems": 1, "maxItems": 1000,
                          "items": {"type": "object", "properties": {
                "buttons": {"type": "array", "items": {"type": "string", "enum": BUTTONS}},
                "left_stick": STICK,
                "right_stick": STICK,
                "frames": {"type": "integer", "minimum": 1, "maximum": 36000},
            },
                                    "required": ["frames"]}},
                "timeout_s": {"type": "integer", "minimum": 1, "maximum": 900},
            },
            "required": ["steps"],
        },
    },
]

UPSTREAM_TOOLS = TOOLS

def tool(name, description, properties=None, required=None):
    schema = {"type": "object", "properties": properties or {}, "additionalProperties": False}
    if required:
        schema["required"] = required
    return {"name": name, "description": description, "inputSchema": schema}


HASHES = {"type": "array", "items": {"type": "string", "pattern": "^[0-9a-fA-F]{16}$"}, "maxItems": 256}
TOOLS = [
    tool("status", "Read boot identity, diagnostic counters and active rejection settings."),
    tool("screenshot", "Capture the game's next rendered frame, not the desktop.",
         {"timeout_s": {"type": "integer", "minimum": 1, "maximum": 15}}),
    tool("shaders", "List captured native program keys, translator errors and host shader hashes.",
         {"offset": {"type": "integer", "minimum": 0},
          "limit": {"type": "integer", "minimum": 1, "maximum": 256},
          "failed_only": {"type": "boolean"}}),
    tool("draws", "List the latest completed renderer-frame's attempted draws/dispatches; not a GPU replay.",
         {"offset": {"type": "integer", "minimum": 0},
          "limit": {"type": "integer", "minimum": 1, "maximum": 256}}),
    tool("dump_shader", "Save captured Ampere instruction bytes and metadata locally; not bound resources.",
         {"key": {"type": "string", "pattern": "^[0-9a-fA-F]{16}$"}}, ["key"]),
    tool("reject_shaders", "Replace the runtime rejection set with native keys/host shader hashes. Empty list restores draws. Skipped compute writes may persist until reload.",
         {"hashes": HASHES}, ["hashes"]),
    tool("reject_draws", "Replace the runtime rejection set of pipeline groups. Empty list clears it. Not a single draw ordinal.",
         {"pipelines": HASHES}, ["pipelines"]),
]
ID = {"type": "string", "pattern": "^[0-9a-fA-F]{16}$"}
TOOLS += [
    tool("compute_state", "Per-compute snapshot by host shader hash or native key: translation error, storage "
         "bindings, untracked global accesses with address IR trees, pointer/null slots, and the last "
         "snapshotted dispatch's enabled cbuf banks (address, size, first 64 words).",
         {"id": ID}, ["id"]),
    tool("watch_compute", "Replace the set of computes whose cbuf/slot snapshot refreshes on every dispatch.",
         {"ids": HASHES}, ["ids"]),
    tool("rebuild_compute", "Drop these compute pipelines so each retranslates against live state at its next "
         "dispatch; read compute_state afterwards.", {"ids": HASHES}, ["ids"]),
    tool("read_gpu", "Read guest GPU virtual memory (current channel), read-only, up to 64 KiB.",
         {"address": {"type": "string", "pattern": "^(0x)?[0-9a-fA-F]{1,16}$"},
          "size": {"type": "integer", "minimum": 1, "maximum": 65536}}, ["address"]),
    tool("log_filter", "Apply a log filter on top of the configured one at runtime, e.g. 'Shader:Debug'.",
         {"filter": {"type": "string", "maxLength": 512}}, ["filter"]),
]
TOOLS += [
    tool("set_frontend", "Switch native programs between the direct Ampere front end ('native') and the Maxwell "
         "lowering ('lower') while the game runs, or 'clear' the override. Their shader cache entries are dropped "
         "and the next draw/dispatch retranslates them; check 'shaders' for the new frontend.",
         {"keys": HASHES, "mode": {"type": "string", "enum": ["native", "lower", "clear"]}}, ["keys", "mode"]),
    tool("frontends", "List active per-program frontend overrides."),
    tool("dump_ir", "Dump the live IR (after translation passes, with binding summary) and SPIR-V of these "
         "programs the next time they are drawn/dispatched, using the real environment (header, cbufs, BRX "
         "tables). Read paths with 'dumps'. Combine with set_frontend to get both translations.",
         {"keys": HASHES}, ["keys"]),
    tool("dumps", "List dump_ir results: per key and host hash, frontend, stage, IR and SPIR-V paths."),
    tool("gpu_activity", "Timeline of long waits: GPU fence waits (GPU busy), guest channel semaphore waits, "
         "pipeline translation/driver builds, waits for pipeline builds, and slow frames; plus what is "
         "waiting right now.",
         {"min_ms": {"type": "number", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 4096},
          "clear": {"type": "boolean"}}),
    tool("video_state", "Native video decoders (AV1/H.264 movies): registered images with their fences and the "
         "syncpoint value, pending inputs, decoded frames, unfetched outputs, EOS flags, decode errors, and the "
         "last IPC calls the game made (identical repeats collapsed with a count).",
         {"trace": {"type": "integer", "minimum": 1, "maximum": 256}}),
    tool("video_inject", "Experiment on a stuck native video decoder: on the next release of an image the decoder "
         "already holds, 'eos' writes an end-of-stream output to it, 'fence' reaches its fence without an output; "
         "'refuse' answers InvalidState; 'flush_return' hands every held image back (fences reached) as a real flush would; 'none' disarms. One-shot, except refuse_while_flushed / accept_while_flushed (sticky: refuse idle releases between Flush and Start). serial from video_state (0 = first decoder).",
         {"what": {"type": "string", "enum": ["eos", "fence", "refuse", "flush_return", "none", "refuse_while_flushed", "accept_while_flushed"]}, "serial": {"type": "integer", "minimum": 0}},
         ["what"]),
    tool("skip_draws", "Alias of reject_draws: skip every draw of these pipeline IDs (from 'draws'); empty list "
         "restores them. Pair with screenshot to see what a pipeline paints.", {"pipelines": HASHES}, ["pipelines"]),
]
TOOL_NAMES = {entry["name"] for entry in TOOLS}
TOOLS.append(tool("press_a", "Momentarily press guest A on player one/handheld, then release; preserves physical input and mappings.",
                  {"hold_ms": {"type": "integer", "minimum": 30, "maximum": 500}}))
TOOL_NAMES.add("press_a")
TOOLS.append(tool("qmd_trace", "Read native QMD launch and completion events.",
                  {"limit": {"type": "integer", "minimum": 1, "maximum": 65536},
                   "last_frame": {"type": "boolean"}, "clear": {"type": "boolean"}}))
TOOL_NAMES.add("qmd_trace")
for entry in TOOLS:
    if entry["name"] == "status":
        entry["inputSchema"]["properties"]["fps_window_ms"] = {
            "type": "integer", "minimum": 0, "maximum": 5000}


TOOLS += [t for t in UPSTREAM_TOOLS if t["name"] not in TOOL_NAMES]
TOOL_NAMES = {t["name"] for t in TOOLS}


class EmulatorError(Exception):
    pass


class Emulator:
    """One JSON line per request to the debug server, reconnecting when the game restarted."""

    def __init__(self, address):
        host, _, port = address.rpartition(":")
        self.host = host.strip("[]") or "127.0.0.1"
        self.port = int(port)
        self.sock = None
        self.buffer = b""
        self.next_id = 1

    def close(self):
        if self.sock is not None:
            self.sock.close()
        self.sock = None
        self.buffer = b""

    def call(self, method, params, timeout):
        for attempt in (0, 1):
            try:
                if self.sock is None:
                    self.sock = socket.create_connection((self.host, self.port), timeout=5)
                self.sock.settimeout(timeout)
                request_id = self.next_id
                self.next_id += 1
                line = json.dumps({"id": request_id, "method": method, "params": params})
                self.sock.sendall(line.encode() + b"\n")
                while b"\n" not in self.buffer:
                    chunk = self.sock.recv(1 << 20)
                    if not chunk:
                        raise ConnectionResetError("the emulator closed the connection")
                    self.buffer += chunk
                raw, _, self.buffer = self.buffer.partition(b"\n")
                response = json.loads(raw)
                break
            except (ConnectionRefusedError, socket.timeout) as error:
                self.close()
                if isinstance(error, socket.timeout):
                    raise EmulatorError(f"no answer within {timeout} s") from error
                raise EmulatorError(
                    f"nothing listens on {self.host}:{self.port}: start n2Den with "
                    f"EDEN_NX2_DEBUG_SERVER={self.host}:{self.port} and boot a game") from error
            except OSError as error:
                # A stale connection from an earlier boot: reconnect once.
                self.close()
                if attempt == 1:
                    raise EmulatorError(str(error)) from error
        if "error" in response:
            raise EmulatorError(response["error"])
        return response.get("result")


def default_timeout(name, params):
    """The emulator's own timeout_s default, so the adapter waits at least as long."""
    if name == "capture_frame":
        return 120
    if name == "sample":
        return int(params.get("seconds", 10)) + 90
    if name == "guest_threads":
        return int(params.get("seconds", 10)) + 30
    if name == "jit_hot":
        return int(params.get("seconds", 20)) + 60
    if name == "sched_trace":
        return int(params.get("seconds", 1)) + 60
    if name in ("input", "input_sequence"):
        steps = params.get("steps", []) if name == "input_sequence" else [params]
        return sum(int(step.get("frames", 0)) for step in steps) // 20 + 10
    return 5


def text(value):
    return {"type": "text", "text": value if isinstance(value, str) else json.dumps(value, indent=1)}


class Adapter:
    def __init__(self, emulator, catalog=None):
        self.catalog = catalog or {}
        self.emulator = emulator
        self.captures = set()

    def run(self, name, arguments):
        """Runs one tool; returns its result and the PNG it produced, if any."""
        if name not in TOOL_NAMES:
            raise EmulatorError(f"unknown tool {name}")
        if name == "capture_image":
            return self.capture_image(arguments)
        params = dict(arguments)
        index = params.pop("index", False)
        timeout = max(int(params.get("timeout_s", default_timeout(name, params))), 1) + 30
        result = self.emulator.call(name, params, timeout)
        if name == "shaders":
            for entry in result.get("shaders", []):
                if entry["key"].lower() in self.catalog:
                    entry["catalogue"] = self.catalog[entry["key"].lower()]
        if name == "screenshot":
            png = result.pop("png_base64", None)
            return result, base64.b64decode(png) if png else None
        if name == "capture_frame":
            self.captures.add(os.path.realpath(result["directory"]))
            if index:
                result["index"] = self.build_index(result["directory"])
        return result, None

    def call_tool(self, name, arguments):
        result, png = self.run(name, arguments)
        content = [text(result)]
        if png is not None:
            content.insert(0, {"type": "image", "data": base64.b64encode(png).decode(),
                               "mimeType": "image/png"})
        return content

    def build_index(self, directory):
        script = os.path.join(HERE, "frame_dump_index.py")
        done = subprocess.run([sys.executable, "-I", script, directory], capture_output=True,
                              text=True, timeout=600)
        if done.returncode != 0:
            return f"frame_dump_index.py failed: {done.stderr.strip()[-2000:]}"
        return os.path.join(directory, "index.html")

    def capture_image(self, arguments):
        directory = os.path.realpath(arguments["directory"])
        if directory not in self.captures:
            raise EmulatorError("not a capture made in this session; run capture_frame first")
        path = os.path.realpath(os.path.join(directory, arguments["name"] + ".png"))
        if os.path.dirname(path) != directory or not os.path.isfile(path):
            raise EmulatorError(f"no image {arguments['name']}.png in {directory}")
        with open(path, "rb") as file:
            return {"path": path}, file.read()

    def handle(self, message):
        """Returns the response to one JSON-RPC message, or None for a notification."""
        method = message.get("method")
        if "id" not in message:
            return None
        reply = {"jsonrpc": "2.0", "id": message["id"]}
        params = message.get("params") or {}
        if method == "initialize":
            wanted = params.get("protocolVersion")
            reply["result"] = {
                "protocolVersion": wanted if wanted in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "n2den-debug", "version": "1"},
            }
        elif method == "ping":
            reply["result"] = {}
        elif method == "tools/list":
            reply["result"] = {"tools": TOOLS}
        elif method == "tools/call":
            try:
                content = self.call_tool(params.get("name"), params.get("arguments") or {})
                reply["result"] = {"content": content, "isError": False}
            except (EmulatorError, KeyError, ValueError, TypeError) as error:
                message_text = f"missing argument {error}" if isinstance(error, KeyError) else str(error)
                reply["result"] = {"content": [text(message_text)], "isError": True}
        else:
            reply["error"] = {"code": -32601, "message": f"method not found: {method}"}
        return reply


def serve(adapter, stdin, stdout):
    for line in stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except ValueError:
            response = {"jsonrpc": "2.0", "id": None,
                        "error": {"code": -32700, "message": "parse error"}}
        else:
            response = adapter.handle(message)
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()


def call_once(adapter, request_text, out, stdout):
    """--call: one request, its JSON result on stdout, its PNG (if any) saved to `out`."""
    try:
        request = json.loads(request_text)
        result, png = adapter.run(request["method"], request.get("params") or {})
    except (EmulatorError, KeyError, ValueError, TypeError, AttributeError) as error:
        message = f"missing {error} in the request" if isinstance(error, KeyError) else str(error)
        stdout.write(json.dumps({"error": message}) + "\n")
        return 1
    if png is not None:
        if out:
            with open(out, "wb") as file:
                file.write(png)
            result["png"] = os.path.abspath(out).replace("\\", "/")
        else:
            result["png"] = "not saved: pass --out <path>"
    stdout.write(json.dumps(result, indent=1) + "\n")
    return 0


def load_catalog(path):
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    programs = data.get("programs", {})
    if isinstance(programs, dict):
        return {key.lower(): value for key, value in programs.items()}
    # Hash-only entries belong to a different lowering/build and cannot identify native code.
    return {entry["key"].lower(): entry for entry in programs if entry.get("key")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--server", default=os.environ.get("N2DEN_DEBUG_SERVER", os.environ.get("EDEN_NX2_DEBUG_SERVER", DEFAULT_SERVER)),
                        help="host:port of EDEN_NX2_DEBUG_SERVER (default: that variable, else "
                             f"{DEFAULT_SERVER})")
    parser.add_argument("--call", metavar="JSON",
                        help='run one request and exit, e.g. \'{"method":"screenshot","params":{}}\'')
    parser.add_argument("--out", metavar="PNG", help="with --call: where to save the PNG")
    parser.add_argument("--catalog", default=os.environ.get("N2DEN_DEBUG_CATALOG"))
    args = parser.parse_args()
    adapter = Adapter(Emulator(args.server), load_catalog(args.catalog))
    if args.call is not None:
        sys.exit(call_once(adapter, args.call, args.out, sys.stdout))
    serve(adapter, sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
