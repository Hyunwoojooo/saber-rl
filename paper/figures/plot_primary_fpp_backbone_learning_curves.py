#!/usr/bin/env python3
"""Build the three-panel FPP cross-backbone primary-comparison figure."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEEDS = tuple(range(1, 6))

BACKBONES = (
    ("DDPG", "(a) DDPG", 1_000_000, 10_000,
     (0, 250_000, 500_000, 750_000, 1_000_000),
     ("0", "0.25", "0.50", "0.75", "1.0")),
    ("TD3", "(b) TD3", 1_000_000, 10_000,
     (0, 250_000, 500_000, 750_000, 1_000_000),
     ("0", "0.25", "0.50", "0.75", "1.0")),
    ("SAC", "(c) SAC", 1_000_000, 50_000,
     (0, 250_000, 500_000, 750_000, 1_000_000),
     ("0", "0.25", "0.50", "0.75", "1.0")),
)
METHODS = ("HER", "SABER", "HGR", "TD-error weighting", "PER")
DRAW_ORDER = ("HGR", "TD-error weighting", "PER", "HER", "SABER")

STYLES = {
    "HER": ("#333333", 1.45, (4.0, 2.2), 0.085),
    "SABER": ("#D55E00", 2.35, (), 0.16),
    "HGR": ("#0072B2", 1.15, (5.0, 2.0, 1.2, 2.0), 0.075),
    "TD-error weighting": ("#009E73", 1.15, (1.2, 2.0), 0.075),
    "PER": ("#CC79A7", 1.15, (6.0, 2.0), 0.075),
}

# These exact paths exclude FPP 500k legacy runs and TD3 SABER seeds 6--10.
SPECS = {
    ("DDPG", "HER"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/runs/B0/"
        "fetch_pick_and_place_b0_s{seed}_1000000_ls1024/"
        "metrics_selected_10k.csv", None),
    ("DDPG", "SABER"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/runs/b3_critic_only_runtime_opt_v4_ls512/"
        "fetch_pick_and_place_b3_critic_only_runtime_opt_v4_ls512_"
        "s{seed}_1000000_ls512/metrics_selected_10k.csv", None),
    ("DDPG", "HGR"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/paper_artifacts/fpp_ddpg_hgr_pair_priority_is_1m/"
        "fetch_pick_and_place_ddpg_hgr_tdpriority_s{seed}_1000000/"
        "metrics_selected_10k.csv", None),
    ("DDPG", "TD-error weighting"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/runs/per_style_loss_weighting_ls512/"
        "fetch_pick_and_place_per_style_loss_weighting_ls512_"
        "s{seed}_1000000_ls512/metrics_selected_10k.csv", None),
    ("DDPG", "PER"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/runs/ddpg_her_per_full_ls512/"
        "fetch_pick_and_place_ddpg_her_per_full_ls512_"
        "s{seed}_1000000_ls512/metrics_selected_10k.csv", None),
    ("TD3", "HER"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/paper_artifacts/fpp_td3_her_vs_saber_n10_1m/"
        "data/td3_her/{seed}/metrics_selected_10k.csv", None),
    ("TD3", "SABER"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/paper_artifacts/fpp_td3_her_vs_saber_n10_1m/"
        "data/td3_saber/{seed}/metrics_selected_10k.csv", None),
    ("TD3", "HGR"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/paper_artifacts/fpp_td3_hgr_pair_priority_is_1m/"
        "fetch_pick_and_place_td3_hgr_tdpriority_s{seed}_1000000/"
        "metrics_selected_10k.csv", None),
    ("TD3", "TD-error weighting"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/paper_artifacts/"
        "fpp_td3_her_per_style_loss_weighting_ls512_1m/data/"
        "td3_her_per_style_loss_weighting/seed{seed}/eval_metrics.csv", None),
    ("TD3", "PER"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/paper_artifacts/fpp_td3_sac_her_per_full_1m/"
        "data/td3_her_per_full/seed{seed}/eval_metrics.csv", None),
    ("SAC", "HER"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/"
        "sac_her_baseline_ls512/"
        "fetch_pick_and_place_sac_her_baseline_ls512_s{seed}_1000000_ls512/"
        "metrics_selected_10k.csv", None),
    ("SAC", "SABER"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/paper_artifacts/fpp_sac_controls_and_redq_1m/"
        "sac_saber_critic_only_v4_ls512/"
        "fetch_pick_and_place_sac_saber_critic_only_v4_ls512_"
        "s{seed}_1000000_ls512/metrics_selected_10k.csv", None),
    ("SAC", "HGR"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/paper_artifacts/fpp_sac_hgr_pair_priority_is_1m/"
        "fetch_pick_and_place_sac_hgr_tdpriority_s{seed}_1000000/"
        "metrics_selected_25k.csv", None),
    ("SAC", "TD-error weighting"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/paper_artifacts/"
        "fpp_sac_her_per_style_loss_weighting_ls512_1m/data/"
        "sac_her_per_style_loss_weighting/seed{seed}/eval_metrics.csv", None),
    ("SAC", "PER"): (
        "raw", "eval/env_success_mean",
        "experiment_FPP/paper_artifacts/fpp_td3_sac_her_per_full_1m/"
        "data/sac_her_per_full/seed{seed}/eval_metrics.csv", None),
}

EXPECTED_AUC = {
    ("DDPG", "HER"): 0.538788, ("DDPG", "SABER"): 0.609091,
    ("DDPG", "HGR"): 0.338182,
    ("DDPG", "TD-error weighting"): 0.568384,
    ("DDPG", "PER"): 0.460303,
    ("TD3", "HER"): 0.371616161616162,
    ("TD3", "SABER"): 0.442373737373737,
    ("TD3", "HGR"): 0.367323232323232,
    ("TD3", "TD-error weighting"): 0.351363636363636,
    ("TD3", "PER"): 0.368888888888889,
    ("SAC", "HER"): 0.471052631578947,
    ("SAC", "SABER"): 0.583421052631579,
    ("SAC", "HGR"): 0.483684210526316,
    ("SAC", "TD-error weighting"): 0.543684210526316,
    ("SAC", "PER"): 0.689736842105263,
}


def finite(value: str | None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def read_curve(path: Path, spec: tuple, seed: int) -> dict[int, float]:
    kind, metric, _, method_key = spec
    if not path.is_file():
        raise FileNotFoundError(path)
    curve = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"step", metric}
        if kind == "combined":
            required |= {"seed", "method_key"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            if kind == "combined" and (
                int(row["seed"]) != seed or row["method_key"] != method_key
            ):
                continue
            value = finite(row[metric])
            if value is None:
                continue
            step = int(float(row["step"]))
            if step in curve:
                raise ValueError(f"{path}: duplicate finite evaluation at {step}")
            curve[step] = value
    return curve


def auc(curve: dict[int, float], grid: tuple[int, ...]) -> float:
    integral = sum(
        0.5 * (curve[a] + curve[b]) * (b - a)
        for a, b in zip(grid, grid[1:])
    )
    return integral / (grid[-1] - grid[0])


def load() -> tuple[dict, list[dict[str, str]]]:
    aggregated = {}
    source_paths = set()
    for backbone, _, budget, grid_step, _, _ in BACKBONES:
        grid = tuple(range(grid_step, budget + 1, grid_step))
        for method in METHODS:
            spec = SPECS[(backbone, method)]
            curves = {}
            for seed in SEEDS:
                path = ROOT / spec[2].format(seed=seed)
                source_paths.add(path)
                observed = read_curve(path, spec, seed)
                curve = {step: observed[step] for step in grid if step in observed}
                if tuple(sorted(curve)) != grid:
                    missing = sorted(set(grid) - set(curve))
                    extra = sorted(set(curve) - set(grid))
                    raise ValueError(
                        f"{backbone}/{method}/seed {seed}: invalid grid; "
                        f"missing={missing[:5]}, extra={extra[:5]}"
                    )
                if any(not 0 <= value <= 1 for value in curve.values()):
                    raise ValueError(
                        f"{backbone}/{method}/seed {seed}: invalid success"
                    )
                curves[seed] = curve

            means = tuple(
                statistics.fmean(curves[seed][step] for seed in SEEDS)
                for step in grid
            )
            sds = tuple(
                statistics.stdev(curves[seed][step] for seed in SEEDS)
                for step in grid
            )
            seed_aucs = tuple(auc(curves[seed], grid) for seed in SEEDS)
            mean_auc = statistics.fmean(seed_aucs)
            if not math.isclose(
                mean_auc, EXPECTED_AUC[(backbone, method)], abs_tol=5e-7
            ):
                raise ValueError(
                    f"{backbone}/{method}: AUC {mean_auc:.9f} differs from "
                    f"frozen {EXPECTED_AUC[(backbone, method)]:.9f}"
                )
            aggregated[(backbone, method)] = {
                "grid": grid, "mean": means, "sd": sds,
                "mean_auc": mean_auc, "auc_sd": statistics.stdev(seed_aucs),
            }

    sources = [
        {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(source_paths)
    ]
    return aggregated, sources


def write_data(aggregated: dict) -> None:
    path = OUT / "primary_fpp_backbone_learning_curves_data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("backbone", "method", "step", "mean_success", "sample_sd",
             "n_seeds", "mean_normalized_auc")
        )
        for backbone, _, _, _, _, _ in BACKBONES:
            for method in METHODS:
                row = aggregated[(backbone, method)]
                for step, mean, sd in zip(row["grid"], row["mean"], row["sd"]):
                    writer.writerow(
                        (backbone, method, step, f"{mean:.9f}", f"{sd:.9f}",
                         len(SEEDS), f"{row['mean_auc']:.9f}")
                    )


def write_manifest(sources: list[dict[str, str]]) -> None:
    payload = {
        "figure": "primary_fpp_backbone_learning_curves.pdf",
        "task": "FetchPickAndPlace-v4",
        "seeds": list(SEEDS),
        "aggregation": "checkpoint-wise mean and sample standard deviation",
        "smoothing": False,
        "interpolation": False,
        "grids": {
            "DDPG": "10k--1M by 10k",
            "TD3": "10k--1M by 10k",
            "SAC": "50k--1M by 50k (exact shared grid)",
        },
        "sources": sources,
    }
    (OUT / "primary_fpp_backbone_learning_curves_sources.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def dash(pdf: canvas.Canvas, pattern: tuple[float, ...]) -> None:
    pdf.setDash(list(pattern)) if pattern else pdf.setDash()


def legend(pdf: canvas.Canvas, width: float, y: float) -> None:
    size, line_width, gap, text_gap = 7.1, 14.0, 11.0, 3.5
    widths = []
    for method in METHODS:
        font = "Helvetica-Bold" if method == "SABER" else "Helvetica"
        widths.append(line_width + text_gap + stringWidth(method, font, size))
    x = 0.5 * (width - sum(widths) - gap * (len(METHODS) - 1))
    for method, item_width in zip(METHODS, widths):
        color, stroke_width, pattern, _ = STYLES[method]
        pdf.saveState()
        pdf.setStrokeColor(HexColor(color))
        pdf.setLineWidth(stroke_width)
        pdf.setLineCap(1)
        dash(pdf, pattern)
        pdf.line(x, y + 2.1, x + line_width, y + 2.1)
        pdf.restoreState()
        font = "Helvetica-Bold" if method == "SABER" else "Helvetica"
        pdf.setFont(font, size)
        pdf.setFillColor(HexColor("#222222"))
        pdf.drawString(x + line_width + text_gap, y, method)
        x += item_width + gap


def panel(
    pdf: canvas.Canvas, backbone_row: tuple, index: int, left: float,
    width: float, bottom: float, top: float, aggregated: dict
) -> None:
    backbone, title, budget, _, x_ticks, x_labels = backbone_row
    height = top - bottom
    xmap = lambda step: left + width * step / budget
    ymap = lambda value: bottom + height * min(1.0, max(0.0, value))

    pdf.saveState()
    pdf.setStrokeColor(HexColor("#D8DADD"))
    pdf.setLineWidth(0.35)
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        pdf.line(left, ymap(tick), left + width, ymap(tick))
    for tick in x_ticks:
        pdf.line(xmap(tick), bottom, xmap(tick), top)
    pdf.restoreState()
    pdf.setStrokeColor(HexColor("#55585C"))
    pdf.setLineWidth(0.6)
    pdf.line(left, bottom, left, top)
    pdf.line(left, bottom, left + width, bottom)

    for method in DRAW_ORDER:
        row = aggregated[(backbone, method)]
        color, _, _, alpha = STYLES[method]
        upper = [
            (xmap(step), ymap(mean + sd))
            for step, mean, sd in zip(row["grid"], row["mean"], row["sd"])
        ]
        lower = [
            (xmap(step), ymap(mean - sd))
            for step, mean, sd in zip(row["grid"], row["mean"], row["sd"])
        ]
        path = pdf.beginPath()
        path.moveTo(*upper[0])
        for point in upper[1:]:
            path.lineTo(*point)
        for point in reversed(lower):
            path.lineTo(*point)
        path.close()
        pdf.saveState()
        pdf.setFillColor(HexColor(color))
        pdf.setFillAlpha(alpha)
        pdf.drawPath(path, fill=1, stroke=0)
        pdf.restoreState()

    for method in DRAW_ORDER:
        row = aggregated[(backbone, method)]
        color, stroke_width, pattern, _ = STYLES[method]
        points = [
            (xmap(step), ymap(mean))
            for step, mean in zip(row["grid"], row["mean"])
        ]
        path = pdf.beginPath()
        path.moveTo(*points[0])
        for point in points[1:]:
            path.lineTo(*point)
        pdf.saveState()
        pdf.setStrokeColor(HexColor(color))
        pdf.setLineWidth(stroke_width)
        pdf.setLineCap(1)
        pdf.setLineJoin(1)
        dash(pdf, pattern)
        pdf.drawPath(path, fill=0, stroke=1)
        pdf.restoreState()

    pdf.setFillColor(HexColor("#1F2328"))
    pdf.setFont("Helvetica-Bold", 8.2)
    pdf.drawCentredString(left + 0.5 * width, top + 7.0, title)
    pdf.setFont("Helvetica", 6.5)
    for tick, label in zip(x_ticks, x_labels):
        pdf.drawCentredString(xmap(tick), bottom - 10.0, label)
    if index == 0:
        for tick, label in zip(
            (0.0, 0.25, 0.5, 0.75, 1.0),
            ("0", "0.25", "0.50", "0.75", "1"),
        ):
            pdf.drawRightString(left - 4.0, ymap(tick) - 2.2, label)
    pdf.setFont("Helvetica", 6.7)
    pdf.drawCentredString(
        left + 0.5 * width, bottom - 22.0, "Environment steps (millions)"
    )


def write_pdf(aggregated: dict) -> None:
    output = OUT / "primary_fpp_backbone_learning_curves.pdf"
    page_width, page_height = 7.0 * inch, 2.55 * inch
    pdf = canvas.Canvas(
        str(output), pagesize=(page_width, page_height), pageCompression=1
    )
    pdf.setTitle("FetchPickAndPlace cross-backbone learning curves")
    pdf.setAuthor("Anonymous")
    left, right, panel_gap = 34.0, 7.0, 11.0
    panel_width = (page_width - left - right - 2 * panel_gap) / 3
    bottom, top = 31.0, 143.0
    legend(pdf, page_width, page_height - 12.0)
    for index, backbone_row in enumerate(BACKBONES):
        panel_left = left + index * (panel_width + panel_gap)
        panel(
            pdf, backbone_row, index, panel_left, panel_width,
            bottom, top, aggregated
        )
    pdf.saveState()
    pdf.translate(9.0, 0.5 * (bottom + top))
    pdf.rotate(90)
    pdf.setFillColor(HexColor("#1F2328"))
    pdf.setFont("Helvetica", 7.2)
    pdf.drawCentredString(0, 0, "Success rate")
    pdf.restoreState()
    pdf.showPage()
    pdf.save()


def main() -> None:
    aggregated, sources = load()
    write_data(aggregated)
    write_manifest(sources)
    write_pdf(aggregated)
    for backbone, _, _, _, _, _ in BACKBONES:
        values = ", ".join(
            f"{method}={aggregated[(backbone, method)]['mean_auc']:.3f}"
            for method in METHODS
        )
        print(f"{backbone}: {values}")


if __name__ == "__main__":
    main()
