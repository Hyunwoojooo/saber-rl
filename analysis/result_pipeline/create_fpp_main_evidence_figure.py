#!/usr/bin/env python3
"""Create the FPP main SABER evidence figure as SVG and PNG.

The script uses only the Python standard library and ImageMagick for rendering.
It is intentionally explicit so values and labels are easy to inspect.
"""

from __future__ import annotations

import html
import os
import subprocess
from pathlib import Path


WIDTH = 3000
HEIGHT = 1700

OUT_DIR = Path("output/figures")
SVG_PATH = OUT_DIR / "fpp_saber_main_evidence.svg"
PNG_PATH = OUT_DIR / "fpp_saber_main_evidence.png"

FONT = "Arial"
TEXT = "#111827"
AXIS = "#1f2937"
GRID = "#d9dee7"
HER_FILL = "#d7d7d7"
SABER_FILL = "url(#saberHatch)"
SABER_BASE = "#0f5f6a"
ERR = "#111827"

BACKBONES = ["DDPG", "TD3", "SAC"]

FINAL = {
    "HER": ([0.940, 0.840, 0.790], [0.042, 0.188, 0.288]),
    "HER+SABER": ([0.960, 0.970, 0.990], [0.042, 0.045, 0.022]),
}

AUC = {
    "HER": ([0.539, 0.372, 0.471], [0.125, 0.032, 0.079]),
    "HER+SABER": ([0.609, 0.442, 0.583], [0.038, 0.035, 0.030]),
}

HIT90 = {
    "HER": [5, 3, 3],
    "HER+SABER": [5, 5, 5],
}


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def line(x1: float, y1: float, x2: float, y2: float, **attrs: object) -> str:
    base = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    base.update(attrs)
    return f"<line {fmt_attrs(base)} />"


def rect(x: float, y: float, w: float, h: float, **attrs: object) -> str:
    base = {"x": x, "y": y, "width": w, "height": h}
    base.update(attrs)
    return f"<rect {fmt_attrs(base)} />"


def text(x: float, y: float, value: str, **attrs: object) -> str:
    base = {"x": x, "y": y}
    base.update(attrs)
    return f"<text {fmt_attrs(base)}>{esc(value)}</text>"


def fmt_attrs(attrs: dict[str, object]) -> str:
    pairs = []
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, float):
            value = f"{value:.2f}"
        pairs.append(f'{key.replace("_", "-")}="{esc(value)}"')
    return " ".join(pairs)


def y_map(value: float, ymin: float, ymax: float, plot_top: float, plot_bottom: float) -> float:
    value = max(ymin, min(ymax, value))
    return plot_bottom - (value - ymin) / (ymax - ymin) * (plot_bottom - plot_top)


def panel(
    label: str,
    title: str,
    y_label: str,
    panel_x: float,
    panel_y: float,
    panel_w: float,
    plot_top: float,
    plot_bottom: float,
    y_min: float,
    y_max: float,
    ticks: list[float],
    data: dict[str, tuple[list[float], list[float]]] | None = None,
    hit_data: dict[str, list[int]] | None = None,
    annotations: list[str] | None = None,
) -> list[str]:
    out: list[str] = []
    plot_left = panel_x + 130
    plot_right = panel_x + panel_w - 35
    plot_w = plot_right - plot_left
    group_w = plot_w / 3
    bar_w = 70
    bar_gap = 16

    out.append(
        text(
            panel_x + 4,
            panel_y + 10,
            f"({label}) {title}",
            font_family=FONT,
            font_size=58,
            font_weight=700,
            fill=TEXT,
        )
    )

    out.append(
        text(
            panel_x + 28,
            (plot_top + plot_bottom) / 2,
            y_label,
            font_family=FONT,
            font_size=52,
            font_weight=600,
            fill=TEXT,
            text_anchor="middle",
            transform=f"rotate(-90 {panel_x + 28:.2f} {(plot_top + plot_bottom) / 2:.2f})",
        )
    )

    # Grid and y ticks.
    for tick in ticks:
        y = y_map(tick, y_min, y_max, plot_top, plot_bottom)
        out.append(line(plot_left, y, plot_right, y, stroke=GRID, stroke_width=2))
        if y_max == 5:
            tick_label = f"{int(tick)}"
        else:
            tick_label = f"{tick:.2f}"
            if tick == 0:
                tick_label = "0.00"
        out.append(
            text(
                plot_left - 18,
                y + 15,
                tick_label,
                font_family=FONT,
                font_size=42,
                fill=TEXT,
                text_anchor="end",
            )
        )

    out.append(line(plot_left, plot_bottom, plot_right, plot_bottom, stroke=AXIS, stroke_width=2.4))
    out.append(line(plot_left, plot_top, plot_left, plot_bottom, stroke=AXIS, stroke_width=2.4))

    for idx, name in enumerate(BACKBONES):
        cx = plot_left + group_w * (idx + 0.5)
        out.append(
            text(
                cx,
                plot_bottom + 60,
                name,
                font_family=FONT,
                font_size=48,
                font_weight=600,
                fill=TEXT,
                text_anchor="middle",
            )
        )

    series = ["HER", "HER+SABER"]
    for s_idx, series_name in enumerate(series):
        fill = HER_FILL if series_name == "HER" else SABER_FILL
        stroke = "#6b7280" if series_name == "HER" else "#063f49"
        for idx in range(3):
            cx = plot_left + group_w * (idx + 0.5)
            x = cx + (-bar_w - bar_gap / 2 if s_idx == 0 else bar_gap / 2)
            if data is not None:
                means, stds = data[series_name]
                mean = means[idx]
                std = stds[idx]
                y = y_map(mean, y_min, y_max, plot_top, plot_bottom)
                bar_h = plot_bottom - y
                out.append(rect(x, y, bar_w, bar_h, fill=fill, stroke=stroke, stroke_width=2.2))

                err_top = y_map(min(mean + std, y_max), y_min, y_max, plot_top, plot_bottom)
                err_bot = y_map(max(mean - std, y_min), y_min, y_max, plot_top, plot_bottom)
                err_x = x + bar_w / 2
                out.append(line(err_x, err_top, err_x, err_bot, stroke=ERR, stroke_width=3.0))
                out.append(line(err_x - 17, err_top, err_x + 17, err_top, stroke=ERR, stroke_width=3.0))
                out.append(line(err_x - 17, err_bot, err_x + 17, err_bot, stroke=ERR, stroke_width=3.0))
            else:
                assert hit_data is not None
                value = hit_data[series_name][idx]
                y = y_map(value, y_min, y_max, plot_top, plot_bottom)
                bar_h = plot_bottom - y
                out.append(rect(x, y, bar_w, bar_h, fill=fill, stroke=stroke, stroke_width=2.2))
                label_y = max(plot_top + 42, y - 16)
                out.append(
                    text(
                        x + bar_w / 2,
                        label_y,
                        f"{value}/5",
                        font_family=FONT,
                        font_size=42,
                        font_weight=700,
                        fill=TEXT,
                        text_anchor="middle",
                    )
                )

    if annotations:
        for idx, ann in enumerate(annotations):
            cx = plot_left + group_w * (idx + 0.5)
            out.append(
                text(
                    cx,
                    plot_top + 42,
                    ann,
                    font_family=FONT,
                    font_size=38,
                    fill="#374151",
                    text_anchor="middle",
                )
            )

    return out


def build_svg() -> str:
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">'
    )
    parts.append(
        """
<defs>
  <pattern id="saberHatch" patternUnits="userSpaceOnUse" width="16" height="16">
    <rect x="0" y="0" width="16" height="16" fill="#0f5f6a" />
    <path d="M-4,16 L16,-4 M0,20 L20,0" stroke="#ffffff" stroke-width="2.2" opacity="0.55" />
  </pattern>
</defs>
""".strip()
    )
    parts.append(rect(0, 0, WIDTH, HEIGHT, fill="#ffffff"))

    panel_y = 122
    plot_top = 285
    plot_bottom = 1348
    margin_x = 92
    gap = 86
    panel_w = (WIDTH - 2 * margin_x - 2 * gap) / 3

    parts.extend(
        panel(
            "a",
            "Final success",
            "Final success",
            margin_x,
            panel_y,
            panel_w,
            plot_top,
            plot_bottom,
            0.0,
            1.05,
            [0.0, 0.25, 0.50, 0.75, 1.00],
            data=FINAL,
        )
    )
    parts.extend(
        panel(
            "b",
            "Success AUC",
            "Success AUC",
            margin_x + panel_w + gap,
            panel_y,
            panel_w,
            plot_top,
            plot_bottom,
            0.0,
            0.75,
            [0.0, 0.25, 0.50, 0.75],
            data=AUC,
            annotations=["CI includes 0", "AUC Δ > 0", "AUC Δ > 0"],
        )
    )
    parts.extend(
        panel(
            "c",
            "Hit90 reached seeds",
            "Hit90 reached seeds",
            margin_x + 2 * (panel_w + gap),
            panel_y,
            panel_w,
            plot_top,
            plot_bottom,
            0,
            5,
            [0, 1, 2, 3, 4, 5],
            hit_data=HIT90,
        )
    )

    # Shared legend.
    legend_x = WIDTH - 570
    legend_y = 58
    parts.append(rect(legend_x, legend_y - 40, 54, 34, fill=HER_FILL, stroke="#6b7280", stroke_width=2))
    parts.append(
        text(legend_x + 70, legend_y - 11, "HER", font_family=FONT, font_size=46, fill=TEXT)
    )
    parts.append(rect(legend_x + 220, legend_y - 40, 54, 34, fill=SABER_FILL, stroke="#063f49", stroke_width=2))
    parts.append(
        text(legend_x + 290, legend_y - 11, "HER+SABER", font_family=FONT, font_size=46, fill=TEXT)
    )

    # Note strip.
    strip_y = 1590
    parts.append(line(90, strip_y, WIDTH - 90, strip_y, stroke="#b8c0cc", stroke_width=2))
    parts.append(
        text(
            WIDTH / 2,
            strip_y + 58,
            "Matched one-update-per-step HER backbones; error bars show seed standard deviation.",
            font_family=FONT,
            font_size=44,
            fill="#1f2937",
            text_anchor="middle",
        )
    )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text(build_svg(), encoding="utf-8")
    swift_renderer = Path("analysis/result_pipeline/render_fpp_main_evidence_figure.swift")
    if swift_renderer.exists():
        env = os.environ.copy()
        env.setdefault("CLANG_MODULE_CACHE_PATH", "/private/tmp/saber-swift-module-cache")
        subprocess.run(["swift", str(swift_renderer)], check=True, env=env, stdout=subprocess.DEVNULL)
    else:
        subprocess.run(["magick", str(SVG_PATH), str(PNG_PATH)], check=True)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
