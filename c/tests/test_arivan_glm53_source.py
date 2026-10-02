from pathlib import Path
import unittest


SOURCE = (Path(__file__).resolve().parents[1] / "glm53.c").read_text(
    encoding="utf-8"
)


class ArivanGlm53SourceTest(unittest.TestCase):
    def test_profile_uses_native_expert_planner(self):
        self.assertIn("arivan_memory_plan_experts(", SOURCE)
        self.assertIn("m->memory_profile.expert_cache_bytes", SOURCE)

    def test_zero_cache_uses_owned_bounded_staging(self):
        self.assertIn("if (cache->cap == 0)", SOURCE)
        self.assertIn("m->estage[i % m->estage_cap]", SOURCE)
        self.assertIn("expert_read(m, index, eid, slot, 1)", SOURCE)
        self.assertIn("if (!metal_slot && !force_owned)", SOURCE)

    def test_runtime_phase_transitions_are_wired(self):
        self.assertIn("ARIVAN_PHASE_VISION", SOURCE)
        self.assertIn("ARIVAN_PHASE_PREFILL", SOURCE)
        self.assertIn("ARIVAN_PHASE_DECODE", SOURCE)
        self.assertIn('glm53_memory_report(m, "session-open")', SOURCE)


if __name__ == "__main__":
    unittest.main()
