from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PrimaryPresetParityTest(unittest.TestCase):
    def test_seven_primary_presets_are_explicit_and_safe(self) -> None:
        paths = sorted((ROOT / "presets").glob("*.json"))
        self.assertEqual(len(paths), 7)
        seen = set()
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn(payload["preset_id"], seen)
            seen.add(payload["preset_id"])
            settings = payload["settings"]
            self.assertEqual(settings["weight_scope"], "critic_only")
            self.assertEqual(settings["weighting_mode"], "batch_norm")
            self.assertEqual(settings["her_ratio"], 0.8)
            self.assertEqual(settings["updates_per_step"], 1)
            self.assertEqual(settings["local_search_max"], 512)
            self.assertEqual(settings["M_B"], 16)
            self.assertEqual(settings["M_Q_min"], 4)
            self.assertEqual(settings["N_local_min"], 4)
            self.assertEqual(settings["N_gap_min"], 128)
            self.assertEqual(settings["action_bins_per_dim"], 5)
            self.assertEqual(settings["lambda_support"], 1.0)
            self.assertTrue((ROOT / payload["source"] / payload["entrypoint"]).is_file())

    def test_task_budgets_and_evaluation_grids(self) -> None:
        for path in sorted((ROOT / "presets").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["task_label"] == "FPP":
                self.assertEqual(payload["paper_steps"], 1_000_000)
                expected = 25_000 if payload["backbone"] == "SAC" else 10_000
                self.assertEqual(payload["settings"]["eval_freq"], expected)
            else:
                self.assertEqual(payload["paper_steps"], 2_000_000)
                self.assertEqual(payload["settings"]["eval_freq"], 10_000)


if __name__ == "__main__":
    unittest.main()
