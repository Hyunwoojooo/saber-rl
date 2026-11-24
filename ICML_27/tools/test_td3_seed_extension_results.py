"""Regression checks for incomplete-cohort publication, without training."""
import copy
import unittest

from td3_seed_extension_results import (
    ARMS, DRAWERCLOSE_HER_SABER, FIELDS, completed_keys, make_readme, status_snapshot, summarize,
)


def queue_fixture(complete=("her",)):
    return {"updated_utc": "2026-11-05T00:00:00+00:00", "protocol": {
        "task": "DrawerOpen-v3", "backbone": "TD3", "order": list(ARMS),
        "seeds": list(range(6, 11)), "steps_per_run": 2000000, "total_new_runs": 20,
        "bootstrap_resamples": 10000, "bootstrap_seed": 2027},
        "arms": {a: {"status": "completed" if a in complete else "queued", "seeds": {
            str(s): {"status": "completed" if a in complete else "queued",
                     "step": 2000000 if a in complete else 0,
                     "exit_code": 0, "nonfinite_loss": []} for s in range(6, 11)}} for a in ARMS}}


def result_fixture(arms=("her",)):
    return [{"arm": a, "seed": s, **{m: 0.0 for m in FIELDS}, "hit90_step": None}
            for a in arms for s in range(1, 11)]


class PublicationTests(unittest.TestCase):
    def test_her_only_has_three_aggregates_and_no_paired_ci(self):
        report = summarize(result_fixture(), queue_fixture())
        self.assertEqual(report["new_completed_runs"], 5)
        self.assertEqual(report["historical_reference_runs"], 5)
        self.assertFalse(report["full_queue_completed"])
        for cohort in report["cohorts"].values():
            self.assertEqual(set(cohort["summary"]), {"her"})
            self.assertEqual(cohort["paired"], {})

    def test_unfinished_results_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize(result_fixture(("her", "saber")), queue_fixture())

    def test_missing_zero_run_is_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            summarize(result_fixture()[:-1], queue_fixture())

    def test_duplicate_seed_is_rejected(self):
        rows = result_fixture()
        rows[-1] = dict(rows[-2])
        with self.assertRaises(ValueError):
            summarize(rows, queue_fixture())

    def test_partial_saber_has_no_aggregate_or_contrast(self):
        queue = queue_fixture()
        queue["arms"]["saber"]["status"] = "running"
        queue["arms"]["saber"]["seeds"]["6"] = copy.deepcopy(queue["arms"]["her"]["seeds"]["6"])
        rows = result_fixture() + [r for r in result_fixture(("saber",)) if r["seed"] <= 6]
        report = summarize(rows, queue)
        self.assertEqual(report["partial_arms"], ["saber"])
        self.assertEqual(report["new_completed_runs"], 6)
        for cohort in report["cohorts"].values():
            self.assertEqual(set(cohort["summary"]), {"her"})
            self.assertEqual(cohort["paired"], {})

    def test_complete_pair_enables_all_three_contrasts(self):
        report = summarize(result_fixture(("her", "saber")), queue_fixture(("her", "saber")))
        for name, cohort in report["cohorts"].items():
            self.assertEqual(set(cohort["paired"]), {"saber_minus_her"})
            value = cohort["paired"]["saber_minus_her"]["auc"]
            self.assertEqual(value["n"], 10 if name == "pooled_1_10" else 5)
            self.assertEqual(value["ci95"], [0.0, 0.0])

    def test_invalid_completion_is_rejected(self):
        for key, value in (("exit_code", 1), ("step", 1900000), ("nonfinite_loss", ["critic"])):
            queue = queue_fixture()
            queue["arms"]["her"]["seeds"]["6"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                completed_keys(queue)

    def test_wrong_backbone_is_rejected(self):
        queue = queue_fixture()
        queue["protocol"]["backbone"] = "DDPG"
        with self.assertRaises(ValueError):
            completed_keys(queue)

    def test_drawerclose_pair_excludes_other_arm_results_and_sources(self):
        study = DRAWERCLOSE_HER_SABER
        queue = queue_fixture(("her", "saber", "hgr"))
        queue["protocol"]["task"] = study.task
        queue["arms"]["hgr"]["seeds"]["6"]["result"] = {"auc": .9}
        snapshot = status_snapshot(queue, study)
        self.assertEqual(snapshot["arms"]["hgr"], {"status": "completed"})
        self.assertEqual(snapshot["arms"]["per"], {"status": "queued"})
        self.assertIn("result", queue["arms"]["hgr"]["seeds"]["6"])
        self.assertEqual(status_snapshot(snapshot, study), snapshot)
        rows = result_fixture(("her", "saber"))
        report = summarize(rows, snapshot, study)
        self.assertEqual(report["complete_arms"], ["her", "saber"])
        self.assertEqual(report["new_completed_runs"], 10)
        self.assertFalse(report["full_queue_completed"])
        self.assertIn("drawerclose_td3_her_saber_results.py verify", make_readme(report, snapshot, study))
        with self.assertRaises(ValueError):
            summarize(rows + result_fixture(("hgr",)), snapshot, study)

    def test_drawerclose_pair_requires_all_ten_valid_completions(self):
        study = DRAWERCLOSE_HER_SABER
        for arm_status, update in (
            ("running", {}),
            ("completed", {"status": "running", "step": 1500000}),
            ("completed", {"exit_code": 1}),
            ("completed", {"nonfinite_loss": ["loss/critic"]}),
        ):
            queue = queue_fixture(("her", "saber"))
            queue["protocol"]["task"] = study.task
            queue["arms"]["saber"]["status"] = arm_status
            queue["arms"]["saber"]["seeds"]["8"].update(update)
            with self.subTest(arm_status=arm_status, update=update), self.assertRaises(ValueError):
                completed_keys(status_snapshot(queue, study), study)


if __name__ == "__main__":
    unittest.main()
