# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise the actual AV1 initialization predicate with observed guest formats.

Set CXX to a GCC/Clang compiler. N2DEN_VIDEO_SOURCE can select a pre-fix source
snapshot for a red/green check without changing the working tree.
"""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class VideoDecoderFormats(unittest.TestCase):
    def test_av1_rgba_nv12_and_legacy_command_200(self):
        source = Path(os.environ.get("N2DEN_VIDEO_SOURCE", str(ROOT / "src/core/hle/service/am/service/native_video_decoder.cpp")))
        match = re.search(r"const bool is_av1 = (.*?);", source.read_text(), re.S)
        self.assertIsNotNone(match)
        assert match is not None
        compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
        if compiler is None:
            self.fail("Set CXX to a GCC/Clang C++ compiler")
        assert compiler is not None
        program = r'''
#include <cassert>
enum class NativeVideoKind { Decoder, Encoder };
struct Request { unsigned codec, output_format, input_format, output_mode; };
bool Supports(NativeVideoKind kind, bool is_200, const Request* request) {
    return ''' + match[1] + r''';
}
int main() {
    constexpr auto decoder = NativeVideoKind::Decoder;
    Request request{3, 3, 3, 1}; // OoT's observed command 201: AV1 to NV12.
    assert(Supports(decoder, false, &request));
    request.output_format = 2; // Existing FE RGBA output.
    assert(Supports(decoder, false, &request));
    request.output_format = 0;
    assert(!Supports(decoder, false, &request));
    request = {1, 3, 3, 1}; // Existing command-200 AV1 pairing.
    assert(Supports(decoder, true, &request));
    assert(!Supports(decoder, false, &request));
    request = {3, 3, 1, 1};
    assert(!Supports(decoder, false, &request));
    request = {3, 3, 3, 0};
    assert(!Supports(decoder, false, &request));
    request = {3, 3, 3, 1};
    assert(!Supports(NativeVideoKind::Encoder, false, &request));
}
'''
        env = os.environ.copy()
        env["PATH"] = str(Path(compiler).resolve().parent) + os.pathsep + env.get("PATH", "")
        with tempfile.TemporaryDirectory(prefix="n2den-video-") as directory:
            cpp = Path(directory) / "formats.cpp"
            exe = Path(directory) / ("formats.exe" if os.name == "nt" else "formats")
            cpp.write_text(program)
            result = subprocess.run([compiler, "-std=c++20", "-O0", str(cpp), "-o", str(exe)],
                                    capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
