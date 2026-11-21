# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression checks for the real SVC generator's address-arbiter ABI."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("svc_generator", ROOT / "tools/svc_generator.py")
assert spec is not None and spec.loader is not None
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class WaitForAddressAbi(unittest.TestCase):
    def declaration(self, bitness):
        declaration = next(decl for imm, decl in generator.SVCS if imm == 0x34)
        return generator.parse_declaration(declaration, bitness)

    def test_aarch64_preserves_full_comparison_value(self):
        parsed = self.declaration(generator.BIT_64)
        arguments = parsed[2]
        self.assertEqual(arguments[2].type_name, "int64_t")
        registers = generator.get_registers(parsed, generator.BIT_64)
        wrapper = generator.emit_wrapper(parsed[1], generator.SUFFIX_NAMES[generator.BIT_64],
                                         registers, arguments, generator.REG_SIZES[generator.BIT_64])
        self.assertIn("int64_t value = Convert<int64_t>(GetArg64(args, 2));", wrapper)

    def test_aarch32_retains_existing_register_layout(self):
        parsed = self.declaration(generator.BIT_32)
        arguments = parsed[2]
        self.assertEqual(arguments[2].type_name, "int32_t")
        registers = generator.get_registers(parsed, generator.BIT_32)
        wrapper = generator.emit_wrapper(parsed[1], generator.SUFFIX_NAMES[generator.BIT_32],
                                         registers, arguments, generator.REG_SIZES[generator.BIT_32])
        self.assertIn("int32_t value = Convert<int32_t>(GetArg32(args, 2));", wrapper)
        self.assertIn("timeout_ns_gather[0] = GetArg32(args, 3);", wrapper)
        self.assertIn("timeout_ns_gather[1] = GetArg32(args, 4);", wrapper)
        self.assertIn("int64_t timeout_ns = Convert<int64_t>(timeout_ns_gather);", wrapper)


if __name__ == "__main__":
    unittest.main()
