import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arivan.memory import get_profile


class ArivanCliTest(unittest.TestCase):
    def run_arivan(self, *args, env=None):
        clean = os.environ.copy()
        clean.pop("ARIVAN_MODEL", None)
        clean.pop("COLI_MODEL", None)
        if env:
            clean.update(env)
        return subprocess.run(
            [sys.executable, "-m", "arivan.cli", *args],
            cwd=ROOT,
            env=clean,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
            timeout=10,
        )

    def test_version_uses_arivan_identity(self):
        result = self.run_arivan("--version")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "arivan 0.1.0.dev0")

    def test_default_welcome_matches_product_copy(self):
        result = self.run_arivan()
        self.assertEqual(result.returncode, 2)
        self.assertIn("அறிவன் — Think. Code. Solve.", result.stdout)
        self.assertIn("Ready to explore your codebase.", result.stdout)
        self.assertIn("ARIVAN_MODEL", result.stdout)

    def test_profile_json_is_machine_readable(self):
        result = self.run_arivan("profile", "8gb", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        profile = json.loads(result.stdout)
        self.assertEqual(profile["name"], "8gb")
        self.assertEqual(profile["engine_budget_gib"], 5.25)
        self.assertEqual(profile["kv_slots"], 1)
        self.assertEqual(profile["vision_tokens"], 128)
        self.assertEqual(profile["mtp_depth"], 1)

    def test_glm53_profile_maps_only_supported_bootstrap_controls(self):
        env = get_profile("8gb").engine_environment("glm53")
        self.assertEqual(env["RAM_GB"], "5.25")
        self.assertEqual(env["GLM53_EXPERT_GB"], "0.500")
        self.assertEqual(env["GLM53_MAX_IMAGE_TOKENS"], "128")
        self.assertEqual(env["GLM53_PREFILL_CHUNK"], "8")
        self.assertNotIn("TRUNK_RESIDENT_LAYERS", env)

    def test_glm_profile_enables_existing_trunk_streaming_switch(self):
        env = get_profile("8gb").engine_environment("glm")
        self.assertEqual(env["TRUNK_RESIDENT_LAYERS"], "0")
        self.assertEqual(env["DRAFT"], "1")

    def test_native_and_python_8gb_contracts_do_not_drift(self):
        source = (ROOT / "c" / "arivan_memory.c").read_text(encoding="utf-8")
        self.assertIn('{"8gb", 8 * GIB, 5376 * MIB, 384 * MIB, 512 * MIB,', source)


if __name__ == "__main__":
    unittest.main()
