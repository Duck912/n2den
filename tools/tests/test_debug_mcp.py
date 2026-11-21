"""Behavioral tests for the deliberately small n2Den debug MCP adapter."""
import importlib.util
import json
import os
import tempfile
from pathlib import Path
import unittest

PATH = Path(os.environ.get("N2DEN_DEBUG_ADAPTER", Path(__file__).parents[1] / "n2den" / "debug_mcp.py"))
spec = importlib.util.spec_from_file_location("n2den_debug_mcp", PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DebugAdapterTests(unittest.TestCase):
    def test_only_diagnostics_and_the_authorized_a_button_are_exposed(self):
        self.assertEqual({tool["name"] for tool in module.TOOLS}, {
            "status", "screenshot", "shaders", "draws", "dump_shader",
            "reject_shaders", "reject_draws", "press_a",
            "compute_state", "watch_compute", "rebuild_compute", "read_gpu", "log_filter",
            "set_frontend", "frontends", "dump_ir", "dumps", "gpu_activity", "skip_draws", "video_state", "video_inject",
        })

    def test_hash_only_catalogue_entries_are_not_mistaken_for_native_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.json"
            catalog.write_text(json.dumps({"programs": [
                {"key": "11b41a4ac8650c71", "object": "Link hair"},
                {"hash": "7ce70774d14b16cf", "object": "Link hat"},
            ]}))
            self.assertEqual(module.load_catalog(catalog), {
                "11b41a4ac8650c71": {"key": "11b41a4ac8650c71", "object": "Link hair"},
            })


if __name__ == "__main__":
    unittest.main()
