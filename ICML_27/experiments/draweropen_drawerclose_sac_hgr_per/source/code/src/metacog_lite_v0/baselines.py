"""Loss adapters for the retained FPP HGR and full-PER conventions."""
import numpy as np


def per_beta_at_step(config, step):
    progress = np.clip(float(step) / max(1, config.per_beta_steps), 0.0, 1.0)
    return float(config.per_beta_start + progress * (config.per_beta_end - config.per_beta_start))


def apply_baseline_weights(config, wb, batch, hgr):
    if hgr is not None:
        critic, actor = hgr.loss_weights(batch)
        # Retain HGR's max-normalized relabeled-branch IS weights (no mean renormalization).
        wb.omega_critic, wb.omega_actor = critic, actor
        wb.w_p_raw = critic.copy()
        return
    if config.method != "sac_her_per_full":
        return
    raw = np.asarray(batch.is_weight, dtype=np.float32)
    if raw.shape != batch.indices.shape or not np.isfinite(raw).all() or np.any(raw <= 0):
        raise FloatingPointError("Invalid PER critic weights")
    wb.w_p_raw = raw.copy()
    wb.omega_critic = raw / float(raw.mean())
    wb.omega_actor = np.ones_like(raw)
    wb.extra_metrics.update({"per/beta": float(batch.per_beta),
        "per/sampling_probability_mean": float(np.mean(batch.sampling_prob)),
        "per/is_weight_raw_mean": float(raw.mean()),
        "per/critic_weight_mean": float(wb.omega_critic.mean()),
        "per/critic_weight_max": float(wb.omega_critic.max()),
        "per/actor_weight_mean": 1.0})


PER_FIELDS = ("per/beta", "per/sampling_probability_mean", "per/is_weight_raw_mean",
              "per/critic_weight_mean", "per/critic_weight_max", "per/actor_weight_mean")
