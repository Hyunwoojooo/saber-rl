"""Regression coverage for the serialized production-config boundary."""
import copy
import dataclasses
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream/code/src"))

from diagnose_retrieval import restore_config
from metacog_lite_v0.config import LiteConfig, SeedConfig


def saved_config():
    config = LiteConfig(task_name="drawer_open", assignment_experiment=True,
                        assignment_rule="support", local_search_max=512,
                        weight_scope="critic_only", lambda_support=1.0)
    saved = json.loads(json.dumps(config.as_dict()))
    saved.update(thread_hint={"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                              "OPENBLAS_NUM_THREADS": "1"},
                 terminal_convention={"timeout_bootstrap": True})
    return saved


class CheckpointConfigTests(unittest.TestCase):
    def test_trainer_metadata_restores_without_mutating_saved_record(self):
        saved = saved_config()
        before = copy.deepcopy(saved)
        cfg = restore_config(saved)
        self.assertEqual(saved, before)
        self.assertIsInstance(cfg.seeds, SeedConfig)
        restored = json.loads(json.dumps(dataclasses.asdict(cfg)))
        for key in restored:
            self.assertEqual(restored[key], saved[key], key)
        self.assertFalse(hasattr(cfg, "thread_hint"))

    def test_unknown_field_is_not_silently_filtered(self):
        saved = saved_config()
        saved["new_learning_parameter"] = 4
        with self.assertRaisesRegex(ValueError, "Unrecognized.*new_learning_parameter"):
            restore_config(saved)

    def test_missing_learning_field_does_not_use_a_default(self):
        saved = saved_config()
        del saved["lambda_support"]
        with self.assertRaisesRegex(ValueError, "Missing.*lambda_support"):
            restore_config(saved)

    def test_assignment_flags_remain_learning_config(self):
        saved = saved_config()
        saved["assignment_log_enabled"] = False
        saved["lambda_support"] = 2.0
        cfg = restore_config(saved)
        self.assertFalse(cfg.assignment_log_enabled)
        self.assertEqual(cfg.lambda_support, 2.0)

    def test_only_known_metadata_can_be_absent(self):
        saved = saved_config()
        for key in ("thread_hint", "terminal_convention", "resolved_env_id"):
            del saved[key]
        self.assertEqual(restore_config(saved).local_search_max, 512)

    def test_invalid_thread_metadata_is_rejected(self):
        for hint in (None, "one", {"OMP_NUM_THREADS": 1}):
            with self.subTest(hint=hint):
                saved = saved_config()
                saved["thread_hint"] = hint
                with self.assertRaisesRegex(ValueError, "thread_hint"):
                    restore_config(saved)

    def test_seed_schema_cannot_silently_change(self):
        for change in ("missing", "extra", "wrong_type"):
            with self.subTest(change=change):
                saved = saved_config()
                if change == "missing":
                    del saved["seeds"]["assignment_seed"]
                elif change == "extra":
                    saved["seeds"]["unknown_seed"] = 1
                else:
                    saved["seeds"] = 1
                with self.assertRaisesRegex(ValueError, "seed configuration"):
                    restore_config(saved)


if __name__ == "__main__":
    unittest.main()
