# SPDX-License-Identifier: GPL-3.0-or-later
"""Compile and exercise the arbiter's actual 64-bit read block in isolation.

The fake memory models the existing range validator's overflow behavior and
unmapped reads. This exercises the production guard without a running scheduler.
Run with CXX pointing to a GCC/Clang C++ compiler (not clang-cl).
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class Wait64MemoryValidation(unittest.TestCase):
    def test_invalid_ranges_never_read_and_valid_values_keep_all_bits(self):
        source = (ROOT / "src/core/hle/kernel/k_address_arbiter.cpp").read_text()
        begin = source.index("            auto& memory = GetCurrentMemory(system.Kernel());")
        end = source.index("        } else {", begin)
        block = source[begin:end].replace("GetCurrentMemory(system.Kernel())", "fake_memory")
        compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
        if compiler is None:
            self.fail("Set CXX to a GCC/Clang C++ compiler")
        assert compiler is not None
        program = r'''
#include <cassert>
#include <cstdint>
using u64 = std::uint64_t;
using s64 = std::int64_t;
struct Memory {
    int reads{};
    u64 bits{0x1122334455667788ULL};
    bool IsValidVirtualAddress(u64 address) const { return address == 0x1000; }
    bool IsValidVirtualAddressRange(u64 address, u64 size) const {
        // The existing validator incorrectly accepts a wrapped end address.
        return address + size < address || address == 0x1000;
    }
    u64 Read64(u64 address) { ++reads; return address == 0x1000 ? bits : 0; }
};
bool Read(Memory& fake_memory, u64 addr, s64& user_value) {
    bool succeeded{};
''' + block + r'''
    return succeeded;
}
int main() {
    Memory memory;
    s64 value{};
    assert(!Read(memory, 0xfffffffffffffff8ULL, value));
    assert(memory.reads == 0);
    assert(!Read(memory, 0x2000, value));
    assert(memory.reads == 0);
    assert(Read(memory, 0x1000, value));
    assert(static_cast<u64>(value) == memory.bits);
    memory.bits = 0xffffffffffffffffULL;
    assert(Read(memory, 0x1000, value));
    assert(value == -1);
    assert(memory.reads == 2);
}
'''
        env = os.environ.copy()
        env["PATH"] = str(Path(compiler).resolve().parent) + os.pathsep + env.get("PATH", "")
        with tempfile.TemporaryDirectory(prefix="n2den-wait64-") as directory:
            cpp = Path(directory) / "guard.cpp"
            exe = Path(directory) / ("guard.exe" if os.name == "nt" else "guard")
            cpp.write_text(program)
            compiled = subprocess.run([compiler, "-std=c++20", "-O0", str(cpp), "-o", str(exe)],
                                      capture_output=True, text=True, env=env)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
