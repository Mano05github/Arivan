from pathlib import Path
import unittest


SOURCE = (Path(__file__).resolve().parents[1] / "glm53.c").read_text(
    encoding="utf-8"
)
MAKEFILE = (Path(__file__).resolve().parents[1] / "Makefile").read_text(
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

    def test_profile_starts_before_model_allocations(self):
        declaration = SOURCE.index("static void model_load_range(")
        load = SOURCE.index("static void model_load_range(", declaration + 1)
        body = SOURCE[load:SOURCE.index("/* ---------- vision ----------", load)]
        self.assertLess(body.index("glm53_memory_init(m)"), body.index("load_cfg(&m->c"))
        self.assertIn("ARIVAN_MEM_RESIDENT_WEIGHTS", SOURCE)
        self.assertIn("ARIVAN_MEM_WORKSPACE", SOURCE)
        self.assertIn("arivan_memory_reclassify", SOURCE)
        self.assertIn('glm53_memory_report(m, "load-complete")', SOURCE)

    def test_matrix_ownership_is_explicit(self):
        self.assertIn("uint64_t bytes;", SOURCE)
        self.assertIn("mat.bytes = output_bytes", SOURCE)

    def test_every_glm53_link_path_includes_native_memory_module(self):
        self.assertIn("glm53: glm53$(EXE)", MAKEFILE)
        self.assertIn("$(SEGMENT_BUILD_DIR)/arivan_memory.o", MAKEFILE)


if __name__ == "__main__":
    unittest.main()
