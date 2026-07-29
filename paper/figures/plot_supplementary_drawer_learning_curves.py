#!/usr/bin/env python3
"""Build four-panel Drawer primary-comparison learning curves."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "paper/supplementary_artifacts/primary_seed_results/mean_learning_curves.csv"
SOURCE_MANIFEST = ROOT / "paper/supplementary_artifacts/primary_seed_results/source_manifest.json"
OUTPUT = OUT / "supplementary_drawer_primary_learning_curves.pdf"
FIGURE_MANIFEST = OUT / "supplementary_drawer_primary_learning_curves_sources.json"
SEEDS = (1, 2, 3, 4, 5)
METHODS = ("HER", "SABER", "HGR", "PER")
DRAW_ORDER = ("HGR", "PER", "HER", "SABER")
DISPLAY_NAMES = {"SABER": "SABER (Ours)"}
PANELS = (
    ("DrawerOpen", "DDPG", "(a) DrawerOpen - DDPG"),
    ("DrawerOpen", "TD3", "(b) DrawerOpen - TD3"),
    ("DrawerClose", "DDPG", "(c) DrawerClose - DDPG"),
    ("DrawerClose", "TD3", "(d) DrawerClose - TD3"),
)
STYLES = {
    "HER": ("#333333", 1.45, (4.0, 2.2), 0.10),
    "SABER": ("#C62828", 1.50, (), 0.20),
    "HGR": ("#0072B2", 1.15, (5.0, 2.0, 1.2, 2.0), 0.10),
    "PER": ("#8E4A6F", 1.15, (6.0, 2.0), 0.10),
}
FONT_REGULAR_PATH = Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf")
FONT_BOLD_PATH = Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf")
FONT_REGULAR = "SupplementFigureSans"
FONT_BOLD = "SupplementFigureSans-Bold"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load() -> dict[tuple[str, str, str], dict[str, tuple[float, ...] | tuple[int, ...]]]:
    if not DATA.is_file():
        raise FileNotFoundError(
            f"{DATA} is missing; run paper/analysis/build_supplementary_results.py"
        )
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    with DATA.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["task"] not in {"DrawerOpen", "DrawerClose"}:
                continue
            grouped.setdefault(
                (row["task"], row["backbone"], row["method"]), []
            ).append(row)

    output = {}
    expected_grid = tuple(range(10_000, 2_000_001, 10_000))
    for task, backbone, _ in PANELS:
        for method in METHODS:
            key = (task, backbone, method)
            rows = sorted(grouped.get(key, ()), key=lambda row: int(row["step"]))
            grid = tuple(int(row["step"]) for row in rows)
            if grid != expected_grid:
                raise ValueError(f"{key}: invalid exact grid")
            if any(int(row["n_seeds"]) != len(SEEDS) for row in rows):
                raise ValueError(f"{key}: unexpected seed count")
            output[key] = {
                "grid": grid,
                "mean": tuple(float(row["mean_success"]) for row in rows),
                "sd": tuple(float(row["sample_sd"]) for row in rows),
            }
    return output


def set_dash(pdf: canvas.Canvas, pattern: tuple[float, ...]) -> None:
    pdf.setDash(list(pattern)) if pattern else pdf.setDash()


def legend(pdf: canvas.Canvas, page_width: float, y: float) -> None:
    size, line_width, gap, text_gap = 9.0, 16.0, 15.0, 4.0
    item_widths = []
    for method in METHODS:
        font = FONT_BOLD if method == "SABER" else FONT_REGULAR
        label = DISPLAY_NAMES.get(method, method)
        item_widths.append(line_width + text_gap + pdfmetrics.stringWidth(label, font, size))
    x = 0.5 * (page_width - sum(item_widths) - gap * (len(METHODS) - 1))
    for method, item_width in zip(METHODS, item_widths):
        color, width, pattern, _ = STYLES[method]
        pdf.saveState()
        pdf.setStrokeColor(HexColor(color))
        pdf.setLineWidth(width)
        pdf.setLineCap(1)
        set_dash(pdf, pattern)
        pdf.line(x, y + 2.1, x + line_width, y + 2.1)
        pdf.restoreState()
        font = FONT_BOLD if method == "SABER" else FONT_REGULAR
        pdf.setFont(font, size)
        pdf.setFillColor(HexColor("#222222"))
        pdf.drawString(x + line_width + text_gap, y, DISPLAY_NAMES.get(method, method))
        x += item_width + gap


def panel(
    pdf: canvas.Canvas,
    panel_spec: tuple[str, str, str],
    index: int,
    left: float,
    bottom: float,
    width: float,
    height: float,
    values: dict,
) -> None:
    task, backbone, title = panel_spec
    top = bottom + height
    xmap = lambda step: left + width * step / 2_000_000
    ymap = lambda value: bottom + height * min(1.0, max(0.0, value))

    pdf.saveState()
    pdf.setStrokeColor(HexColor("#D0D0D0"))
    pdf.setLineWidth(0.45)
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        pdf.line(left, ymap(tick), left + width, ymap(tick))
    for tick in (0, 500_000, 1_000_000, 1_500_000, 2_000_000):
        pdf.line(xmap(tick), bottom, xmap(tick), top)
    pdf.restoreState()
    pdf.setStrokeColor(HexColor("#55585C"))
    pdf.setLineWidth(0.6)
    pdf.line(left, bottom, left, top)
    pdf.line(left, bottom, left + width, bottom)

    for method in DRAW_ORDER:
        row = values[(task, backbone, method)]
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
        row = values[(task, backbone, method)]
        color, stroke_width, pattern, _ = STYLES[method]
        points = [(xmap(step), ymap(mean)) for step, mean in zip(row["grid"], row["mean"])]
        path = pdf.beginPath()
        path.moveTo(*points[0])
        for point in points[1:]:
            path.lineTo(*point)
        pdf.saveState()
        pdf.setStrokeColor(HexColor(color))
        pdf.setLineWidth(stroke_width)
        pdf.setLineCap(1)
        pdf.setLineJoin(1)
        set_dash(pdf, pattern)
        pdf.drawPath(path, fill=0, stroke=1)
        pdf.restoreState()

    pdf.setFillColor(HexColor("#1F2328"))
    pdf.setFont(FONT_BOLD, 9.5)
    pdf.drawCentredString(left + width / 2, top + 8.0, title)
    pdf.setFont(FONT_REGULAR, 8.5)
    for tick, label in zip(
        (0, 500_000, 1_000_000, 1_500_000, 2_000_000),
        ("0", "0.5", "1.0", "1.5", "2.0"),
    ):
        pdf.drawCentredString(xmap(tick), bottom - 12.0, label)
    if index % 2 == 0:
        for tick, label in zip(
            (0.0, 0.25, 0.5, 0.75, 1.0),
            ("0", "0.25", "0.50", "0.75", "1"),
        ):
            pdf.drawRightString(left - 5.0, ymap(tick) - 3.0, label)
    if index >= 2:
        pdf.setFont(FONT_REGULAR, 9.0)
        pdf.drawCentredString(left + width / 2, bottom - 27.0, "Environment steps (millions)")


def write_pdf(values: dict) -> None:
    for font_path in (FONT_REGULAR_PATH, FONT_BOLD_PATH):
        if not font_path.is_file():
            raise FileNotFoundError(font_path)
    pdfmetrics.registerFont(TTFont(FONT_REGULAR, str(FONT_REGULAR_PATH)))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, str(FONT_BOLD_PATH)))

    page_width, page_height = 7.0 * inch, 5.15 * inch
    pdf = canvas.Canvas(
        str(OUTPUT),
        pagesize=(page_width, page_height),
        pageCompression=1,
        invariant=1,
        initialFontName=FONT_REGULAR,
        initialFontSize=9.0,
    )
    pdf.setTitle("Drawer primary-comparison learning curves")
    pdf.setAuthor("Anonymous")
    left_margin, right_margin, column_gap = 42.0, 10.0, 23.0
    top_margin, bottom_margin, row_gap = 40.0, 40.0, 39.0
    panel_width = (page_width - left_margin - right_margin - column_gap) / 2
    panel_height = (page_height - top_margin - bottom_margin - row_gap) / 2
    legend(pdf, page_width, page_height - 15.0)
    for index, panel_spec in enumerate(PANELS):
        column = index % 2
        row = index // 2
        left = left_margin + column * (panel_width + column_gap)
        bottom = bottom_margin + (1 - row) * (panel_height + row_gap)
        panel(pdf, panel_spec, index, left, bottom, panel_width, panel_height, values)
    pdf.saveState()
    pdf.translate(12.0, 0.5 * page_height)
    pdf.rotate(90)
    pdf.setFillColor(HexColor("#1F2328"))
    pdf.setFont(FONT_REGULAR, 9.0)
    pdf.drawCentredString(0, 0, "Success rate")
    pdf.restoreState()
    pdf.showPage()
    pdf.save()


def main() -> None:
    values = load()
    write_pdf(values)
    manifest = {
        "figure": str(OUTPUT.relative_to(ROOT)),
        "panels": [list(panel) for panel in PANELS],
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "aggregation": "checkpoint-wise mean and sample standard deviation",
        "evaluation_grid": "10k--2M by 10k",
        "smoothing": False,
        "interpolation": False,
        "source_files": [
            {"path": str(DATA.relative_to(ROOT)), "sha256": sha256(DATA)},
            {"path": str(SOURCE_MANIFEST.relative_to(ROOT)), "sha256": sha256(SOURCE_MANIFEST)},
        ],
    }
    FIGURE_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
