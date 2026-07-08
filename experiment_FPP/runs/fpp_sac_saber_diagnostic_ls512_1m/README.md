# fpp_sac_saber_diagnostic_ls512_1m

Artifact contents intentionally exclude checkpoint .pt files and raw large metrics.csv / metrics.jsonl from git.

Included per seed: summary.json, config.json, env_snapshot.json, run.log, metrics_selected_10k.csv, diagnostic_metrics.jsonl, and diagnostic_samples.jsonl.

This is an instrumented SAC+HER+SABER diagnostic run for FetchPickAndPlace. It is intended for sample-level diagnostic evidence, not as a replacement for the main SAC performance row unless final/AUC behavior is judged comparable.

Aggregate:

{
  "ESS_w_normalized_mean": 0.8900233859806448,
  "ESS_w_normalized_std": 0.0012716476396474477,
  "auc_success_mean": 0.5586363636363636,
  "auc_success_std": 0.02751269289954428,
  "best_env_success_mean": 1.0,
  "best_env_success_std": 0.0,
  "final_env_success_mean": 0.99,
  "final_env_success_std": 0.020000000000000018,
  "final_goal_success_mean": 0.99,
  "final_goal_success_std": 0.020000000000000018,
  "mean_w_P_mean": 1.29765625,
  "mean_w_P_std": 0.027885262964960543,
  "steps_per_sec_mean": 12.854749362685117,
  "steps_per_sec_std": 0.035189634367441984,
  "support_rate_mean": 0.29765625,
  "support_rate_std": 0.027885262964960543
}

Diagnostic aggregate is stored in diagnostic_analysis_summary.json.
