#!/usr/bin/env python3
"""Build the AAAI-compliant SABER pipeline figure as SVG and vector PDF."""

from __future__ import annotations

import math
from html import escape
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent
SVG_PATH = HERE / "saber_pipeline_weighting_source.svg"
PDF_PATH = HERE / "saber_pipeline_weighting.pdf"

# Native size is one AAAI column. All visible text is at least 9 pt.
PAGE_W = 3.3 * 72.0
PAGE_H = 2.4667 * 72.0

FONT_REGULAR_PATH = Path(
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
)
FONT_BOLD_PATH = Path(
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
)
FONT_REGULAR = "FigureSans"
FONT_BOLD = "FigureSans-Bold"

WHITE = "#FFFFFF"
TEXT = "#1F2937"
MUTED = "#4B5563"
GRAY_STROKE = "#6B7280"
GRAY_FILL = "#F3F4F6"
BLUE = "#1F77B4"
BLUE_FILL = "#EAF4FB"
RED = "#C62828"
RED_FILL = "#FFF3F3"


def _hex_rgb(value: str) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _relative_luminance(value: str) -> float:
    channels = []
    for channel in _hex_rgb(value):
        channels.append(
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
        )
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast_ratio(first: str, second: str) -> float:
    high, low = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (high + 0.05) / (low + 0.05)


def _validate_style_contract() -> None:
    # AAAI-27 requires colored content to exceed 4.5:1 contrast.
    pairs = [
        (TEXT, WHITE),
        (MUTED, WHITE),
        (BLUE, WHITE),
        (RED, WHITE),
        (RED, RED_FILL),
    ]
    for foreground, background in pairs:
        ratio = _contrast_ratio(foreground, background)
        if ratio < 4.5:
            raise ValueError(
                f"Insufficient contrast {ratio:.3f}: {foreground} on {background}"
            )


def _star_points(cx: float, cy: float, radius: float) -> list[tuple[float, float]]:
    points = []
    for index in range(10):
        angle = -math.pi / 2 + index * math.pi / 5
        current_radius = radius if index % 2 == 0 else radius * 0.43
        points.append(
            (
                cx + current_radius * math.cos(angle),
                cy + current_radius * math.sin(angle),
            )
        )
    return points


class SvgBackend:
    def __init__(self) -> None:
        self.items: list[str] = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="3.3in" '
                f'height="2.4667in" viewBox="0 0 {PAGE_W:.2f} {PAGE_H:.2f}" '
                'role="img" aria-labelledby="title desc">'
            ),
            "<title id=\"title\">SABER pipeline and local support weighting</title>",
            (
                "<desc id=\"desc\">Uniform replay and HER remain unchanged. "
                "SABER queries whether the current-policy action bin is occupied "
                "by same-context replay actions, then mean-normalizes the resulting "
                "critic-loss weights.</desc>"
            ),
            f'<rect width="100%" height="100%" fill="{WHITE}"/>',
        ]

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str,
        stroke: str,
        stroke_width: float = 0.75,
        radius: float = 3.0,
    ) -> None:
        if stroke_width < 0.5:
            raise ValueError("AAAI figure strokes must be at least 0.5 pt")
        self.items.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" '
            f'height="{height:.2f}" rx="{radius:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:.2f}"/>'
        )

    def line(
        self,
        points: list[tuple[float, float]],
        *,
        color: str,
        width: float = 0.75,
    ) -> None:
        if width < 0.5:
            raise ValueError("AAAI figure strokes must be at least 0.5 pt")
        path = " ".join(
            ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
            for index, (x, y) in enumerate(points)
        )
        self.items.append(
            f'<path d="{path}" fill="none" stroke="{color}" '
            f'stroke-width="{width:.2f}" stroke-linecap="round" '
            'stroke-linejoin="round"/>'
        )

    def polygon(
        self,
        points: list[tuple[float, float]],
        *,
        fill: str,
        stroke: str | None = None,
        stroke_width: float = 0.75,
    ) -> None:
        if stroke is not None and stroke_width < 0.5:
            raise ValueError("AAAI figure strokes must be at least 0.5 pt")
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        stroke_attr = "none" if stroke is None else stroke
        self.items.append(
            f'<polygon points="{coords}" fill="{fill}" stroke="{stroke_attr}" '
            f'stroke-width="{stroke_width:.2f}"/>'
        )

    def circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        *,
        fill: str,
        stroke: str | None = None,
        stroke_width: float = 0.75,
    ) -> None:
        if stroke is not None and stroke_width < 0.5:
            raise ValueError("AAAI figure strokes must be at least 0.5 pt")
        stroke_attr = "none" if stroke is None else stroke
        self.items.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{radius:.2f}" '
            f'fill="{fill}" stroke="{stroke_attr}" '
            f'stroke-width="{stroke_width:.2f}"/>'
        )

    def text(
        self,
        x: float,
        baseline_y: float,
        value: str,
        *,
        size: float = 9.0,
        bold: bool = False,
        anchor: str = "middle",
        color: str = TEXT,
    ) -> None:
        if size < 9.0:
            raise ValueError("AAAI figure text must be at least 9 pt")
        weight = "700" if bold else "400"
        self.items.append(
            f'<text x="{x:.2f}" y="{baseline_y:.2f}" '
            f'font-family="Helvetica, Arial, sans-serif" font-size="{size:.2f}pt" '
            f'font-weight="{weight}" text-anchor="{anchor}" fill="{color}">'
            f"{escape(value)}</text>"
        )

    def save(self) -> None:
        self.items.append("</svg>")
        SVG_PATH.write_text("\n".join(self.items) + "\n", encoding="utf-8")


class PdfBackend:
    def __init__(self) -> None:
        pdfmetrics.registerFont(TTFont(FONT_REGULAR, str(FONT_REGULAR_PATH)))
        pdfmetrics.registerFont(TTFont(FONT_BOLD, str(FONT_BOLD_PATH)))
        self.canvas = canvas.Canvas(
            str(PDF_PATH),
            pagesize=(PAGE_W, PAGE_H),
            pageCompression=1,
            invariant=1,
            initialFontName=FONT_REGULAR,
            initialFontSize=9.0,
        )
        self.canvas.setFillColorRGB(*_hex_rgb(WHITE))
        self.canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    @staticmethod
    def _pdf_y(y: float) -> float:
        return PAGE_H - y

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str,
        stroke: str,
        stroke_width: float = 0.75,
        radius: float = 3.0,
    ) -> None:
        if stroke_width < 0.5:
            raise ValueError("AAAI figure strokes must be at least 0.5 pt")
        self.canvas.setFillColorRGB(*_hex_rgb(fill))
        self.canvas.setStrokeColorRGB(*_hex_rgb(stroke))
        self.canvas.setLineWidth(stroke_width)
        self.canvas.roundRect(
            x,
            PAGE_H - y - height,
            width,
            height,
            radius,
            fill=1,
            stroke=1,
        )

    def line(
        self,
        points: list[tuple[float, float]],
        *,
        color: str,
        width: float = 0.75,
    ) -> None:
        if width < 0.5:
            raise ValueError("AAAI figure strokes must be at least 0.5 pt")
        self.canvas.setStrokeColorRGB(*_hex_rgb(color))
        self.canvas.setLineWidth(width)
        path = self.canvas.beginPath()
        first_x, first_y = points[0]
        path.moveTo(first_x, self._pdf_y(first_y))
        for x, y in points[1:]:
            path.lineTo(x, self._pdf_y(y))
        self.canvas.drawPath(path, fill=0, stroke=1)

    def polygon(
        self,
        points: list[tuple[float, float]],
        *,
        fill: str,
        stroke: str | None = None,
        stroke_width: float = 0.75,
    ) -> None:
        if stroke is not None and stroke_width < 0.5:
            raise ValueError("AAAI figure strokes must be at least 0.5 pt")
        self.canvas.setFillColorRGB(*_hex_rgb(fill))
        if stroke is not None:
            self.canvas.setStrokeColorRGB(*_hex_rgb(stroke))
        self.canvas.setLineWidth(stroke_width)
        path = self.canvas.beginPath()
        first_x, first_y = points[0]
        path.moveTo(first_x, self._pdf_y(first_y))
        for x, y in points[1:]:
            path.lineTo(x, self._pdf_y(y))
        path.close()
        self.canvas.drawPath(path, fill=1, stroke=1 if stroke else 0)

    def circle(
        self,
        cx: float,
        cy: float,
        radius: float,
        *,
        fill: str,
        stroke: str | None = None,
        stroke_width: float = 0.75,
    ) -> None:
        if stroke is not None and stroke_width < 0.5:
            raise ValueError("AAAI figure strokes must be at least 0.5 pt")
        self.canvas.setFillColorRGB(*_hex_rgb(fill))
        if stroke is not None:
            self.canvas.setStrokeColorRGB(*_hex_rgb(stroke))
        self.canvas.setLineWidth(stroke_width)
        self.canvas.circle(
            cx,
            self._pdf_y(cy),
            radius,
            fill=1,
            stroke=1 if stroke else 0,
        )

    def text(
        self,
        x: float,
        baseline_y: float,
        value: str,
        *,
        size: float = 9.0,
        bold: bool = False,
        anchor: str = "middle",
        color: str = TEXT,
    ) -> None:
        if size < 9.0:
            raise ValueError("AAAI figure text must be at least 9 pt")
        font = FONT_BOLD if bold else FONT_REGULAR
        self.canvas.setFont(font, size)
        self.canvas.setFillColorRGB(*_hex_rgb(color))
        if anchor == "middle":
            self.canvas.drawCentredString(x, self._pdf_y(baseline_y), value)
        elif anchor == "end":
            self.canvas.drawRightString(x, self._pdf_y(baseline_y), value)
        else:
            self.canvas.drawString(x, self._pdf_y(baseline_y), value)

    def save(self) -> None:
        self.canvas.showPage()
        self.canvas.save()


def draw_arrow(
    backend: SvgBackend | PdfBackend,
    points: list[tuple[float, float]],
    *,
    color: str = TEXT,
    width: float = 0.9,
    head_size: float = 3.2,
) -> None:
    backend.line(points, color=color, width=width)
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    angle = math.atan2(y2 - y1, x2 - x1)
    left = (
        x2 - head_size * math.cos(angle - math.pi / 6),
        y2 - head_size * math.sin(angle - math.pi / 6),
    )
    right = (
        x2 - head_size * math.cos(angle + math.pi / 6),
        y2 - head_size * math.sin(angle + math.pi / 6),
    )
    backend.polygon([(x2, y2), left, right], fill=color)


def draw_multiline_box(
    backend: SvgBackend | PdfBackend,
    x: float,
    y: float,
    width: float,
    height: float,
    lines: list[str],
    *,
    fill: str,
    stroke: str,
    bold: bool = True,
) -> None:
    backend.rect(x, y, width, height, fill=fill, stroke=stroke, stroke_width=0.8)
    line_height = 9.2
    total_height = line_height * len(lines)
    first_baseline = y + (height - total_height) / 2 + 7.7
    for index, line in enumerate(lines):
        backend.text(
            x + width / 2,
            first_baseline + index * line_height,
            line,
            size=9.0,
            bold=bold,
        )


def draw_grid(
    backend: SvgBackend | PdfBackend,
    x: float,
    y: float,
    size: float,
    *,
    positive: bool,
) -> None:
    cells = 5
    cell = size / cells
    backend.rect(
        x,
        y,
        size,
        size,
        fill=WHITE,
        stroke=TEXT,
        stroke_width=0.75,
        radius=0.0,
    )
    for index in range(1, cells):
        offset = index * cell
        backend.line(
            [(x + offset, y), (x + offset, y + size)], color=TEXT, width=0.55
        )
        backend.line(
            [(x, y + offset), (x + size, y + offset)], color=TEXT, width=0.55
        )

    replay_cells = [(0, 0), (2, 0), (4, 1), (1, 2), (3, 3), (0, 4), (4, 4)]
    query_cell = (1, 2) if positive else (3, 2)
    for col, row in replay_cells:
        cx = x + (col + 0.5) * cell
        cy = y + (row + 0.5) * cell
        backend.circle(cx, cy, 2.45, fill=BLUE)

    qx = x + (query_cell[0] + 0.5) * cell
    qy = y + (query_cell[1] + 0.5) * cell
    query_radius = 3.0
    backend.polygon(
        _star_points(qx, qy, query_radius),
        fill=RED,
    )


def draw_figure(backend: SvgBackend | PdfBackend) -> None:
    # Fixed pipeline: SABER changes the coefficient entering the critic loss.
    top_y = 4.0
    box_h = 34.0
    widths = [36.0, 36.0, 38.0, 43.0, 36.0]
    gap = 10.0
    x_positions = [4.0]
    for width in widths[:-1]:
        x_positions.append(x_positions[-1] + width + gap)

    boxes = [
        (["Uniform", "replay"], BLUE_FILL, BLUE),
        (["HER", "relabel"], GRAY_FILL, GRAY_STROKE),
        (["Bellman", "target"], GRAY_FILL, GRAY_STROKE),
        (["Weighted", "critic loss"], RED_FILL, RED),
        (["Critic", "update"], GRAY_FILL, GRAY_STROKE),
    ]
    for index, (lines, fill, stroke) in enumerate(boxes):
        draw_multiline_box(
            backend,
            x_positions[index],
            top_y,
            widths[index],
            box_h,
            lines,
            fill=fill,
            stroke=stroke,
        )
        if index < len(boxes) - 1:
            draw_arrow(
                backend,
                [
                    (x_positions[index] + widths[index] + 1.5, top_y + box_h / 2),
                    (x_positions[index + 1] - 1.5, top_y + box_h / 2),
                ],
                color=TEXT,
                width=0.8,
                head_size=2.8,
            )

    backend.text(
        PAGE_W / 2,
        54.0,
        "Local support query (SABER)",
        size=10.0,
        bold=True,
        color=RED,
    )
    backend.text(
        PAGE_W / 2,
        65.5,
        "Same state-goal-horizon context",
        size=9.0,
        color=MUTED,
    )

    # Legend uses both color and shape, so the query remains readable in grayscale.
    backend.circle(69.0, 74.0, 2.6, fill=BLUE)
    backend.text(74.0, 77.0, "replay action", size=9.0, anchor="start")
    backend.polygon(
        _star_points(136.0, 74.0, 3.6),
        fill=RED,
    )
    backend.text(142.0, 77.0, "policy query", size=9.0, anchor="start")

    left_center = 66.0
    right_center = 169.0
    backend.text(left_center, 89.5, "Support-positive", size=9.0, bold=True)
    backend.text(right_center, 89.5, "Support-negative", size=9.0, bold=True)

    grid_size = 42.0
    grid_y = 94.5
    draw_grid(backend, left_center - grid_size / 2, grid_y, grid_size, positive=True)
    draw_grid(backend, right_center - grid_size / 2, grid_y, grid_size, positive=False)

    backend.text(
        left_center,
        146.5,
        "\u03c8\u1d62=1, raw weight=2",
        size=9.0,
        color=TEXT,
    )
    backend.text(
        right_center,
        146.5,
        "\u03c8\u1d62=0, raw weight=1",
        size=9.0,
        color=TEXT,
    )

    norm_x = 19.0
    norm_y = 152.0
    norm_w = 199.0
    norm_h = 24.0
    backend.rect(
        norm_x,
        norm_y,
        norm_w,
        norm_h,
        fill=RED_FILL,
        stroke=RED,
        stroke_width=0.85,
        radius=3.0,
    )
    backend.text(
        PAGE_W / 2,
        162.0,
        "All samples retained",
        size=9.0,
        bold=True,
    )
    backend.text(
        PAGE_W / 2,
        172.5,
        "Mean-one normalized critic weight: \u03a9\u1d62",
        size=9.0,
        color=RED,
    )

    # The red return path makes the precise intervention point explicit.
    draw_arrow(
        backend,
        [
            (norm_x + norm_w, norm_y + norm_h / 2),
            (229.5, norm_y + norm_h / 2),
            (229.5, 44.5),
            (x_positions[3] + widths[3] / 2, 44.5),
            (x_positions[3] + widths[3] / 2, top_y + box_h),
        ],
        color=RED,
        width=1.05,
        head_size=3.3,
    )


def main() -> None:
    _validate_style_contract()

    svg = SvgBackend()
    draw_figure(svg)
    svg.save()

    pdf = PdfBackend()
    draw_figure(pdf)
    pdf.save()


if __name__ == "__main__":
    main()
