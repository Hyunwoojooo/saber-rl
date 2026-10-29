"""CPU-only checks that publication cannot promote incomplete cohorts."""
import unittest

from ddpg_completed_extension_results import ARMS, completed_keys, summarize


class CompletedPublicationTests(unittest.TestCase):
    def fixture(self):
        queue = {"updated_utc": "2026-10-29T06:00:00+00:00",
                 "protocol": {"order": list(ARMS), "seeds": list(range(6, 11))}, "arms": {}}
        for arm in ARMS:
            seeds = {}
            for seed in range(6, 11):
                done = arm in ARMS[:4] or (arm == "replay_query" and seed < 10)
                seeds[str(seed)] = {"status": "completed" if done else "running" if arm == "replay_query" else "queued",
                                    "step": 2000000 if done else 1000000,
                                    "exit_code": 0 if done else None, "nonfinite_loss": []}
            queue["arms"][arm] = {"seeds": seeds}
        keys = completed_keys(queue)
        keys |= {(arm, seed) for arm in ARMS[:5] for seed in range(1, 6)}
        rows = [{"arm": arm, "seed": seed, "auc": seed / 20, "final_quarter_auc": .5,
                 "final_success": .5, "hit90_step": None} for arm, seed in sorted(keys)]
        return queue, rows

    def test_partial_method_is_seed_level_only(self):
        q, rows = self.fixture()
        report = summarize(rows, q)
        self.assertEqual(report["new_completed_runs"], 24)
        self.assertEqual(report["historical_reference_runs"], 25)
        self.assertEqual(report["partial_arms"], ["replay_query"])
        self.assertFalse(report["full_queue_completed"])
        for cohort in report["cohorts"].values():
            self.assertEqual(set(cohort["summary"]), set(ARMS[:4]))
            self.assertNotIn("saber_minus_replay_query", cohort["paired"])

    def test_incomplete_run_cannot_be_published(self):
        q, rows = self.fixture()
        rows.append({**rows[0], "arm": "replay_query", "seed": 10})
        with self.assertRaises(ValueError):
            summarize(rows, q)

    def test_completed_failure_to_learn_is_retained(self):
        q, rows = self.fixture()
        for r in rows:
            if r["arm"] == "per" and r["seed"] in (7, 8):
                r.update(auc=0., final_quarter_auc=0., final_success=0.)
        report = summarize(rows, q)
        self.assertEqual(report["cohorts"]["pooled_1_10"]["paired"]["saber_minus_per"]["auc"]["n"], 10)
        self.assertEqual(report["new_completed_runs"], 24)

    def test_no_missing_or_duplicate_completed_rows(self):
        q, rows = self.fixture()
        for invalid in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]]):
            with self.assertRaises(ValueError):
                summarize(invalid, q)

    def test_pairing_is_by_seed_not_input_order(self):
        q, rows = self.fixture()
        self.assertEqual(summarize(rows, q), summarize(rows[::-1], q))
        self.assertEqual(summarize(rows, q)["cohorts"]["pooled_1_10"]["paired"]["saber_minus_hgr"]["auc"]["ci95"], [0., 0.])

    def test_bad_completion_records_rejected(self):
        for key, value in (("exit_code", 1), ("step", 1990000), ("nonfinite_loss", ["loss/critic"])):
            q, _ = self.fixture()
            q["arms"]["per"]["seeds"]["7"][key] = value
            with self.assertRaises(ValueError):
                completed_keys(q)


if __name__ == "__main__":
    unittest.main()
