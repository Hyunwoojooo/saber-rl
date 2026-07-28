#!/usr/bin/env python3
"""Compose the three task renderings into one single-column PDF figure."""

from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


OUT = Path(__file__).resolve().parent
FONT_REGULAR_PATH = Path(
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
)
FONT_BOLD_PATH = Path(
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
)
FONT_REGULAR = "FigureSans"
FONT_BOLD = "FigureSans-Bold"
TASKS = (
    ("(a)", "FetchPickAndPlace", "fetch_pick_and_place_v4_render.png"),
    ("(b)", "DrawerOpen", "drawer_open_v3_render.png"),
    ("(c)", "DrawerClose", "drawer_close_v3_render.png"),
)


def main() -> None:
    for font_path in (FONT_REGULAR_PATH, FONT_BOLD_PATH):
        if not font_path.is_file():
            raise FileNotFoundError(font_path)
    pdfmetrics.registerFont(TTFont(FONT_REGULAR, str(FONT_REGULAR_PATH)))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, str(FONT_BOLD_PATH)))

    page_width = 3.3 * inch
    gap = 1.5
    panel_width = (page_width - gap * (len(TASKS) - 1)) / len(TASKS)
    image_height = panel_width * 3.0 / 4.0
    image_y = 13.0
    page_height = image_y + image_height
    output = OUT / "experimental_task_panels.pdf"

    pdf = canvas.Canvas(
        str(output),
        pagesize=(page_width, page_height),
        pageCompression=1,
        invariant=1,
        initialFontName=FONT_REGULAR,
        initialFontSize=9.0,
    )
    pdf.setTitle("Experimental task renderings")
    pdf.setAuthor("Anonymous")

    for index, (panel_label, task_name, filename) in enumerate(TASKS):
        source = OUT / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        x = index * (panel_width + gap)
        pdf.drawImage(
            ImageReader(str(source)),
            x,
            image_y,
            width=panel_width,
            height=image_height,
            preserveAspectRatio=True,
            anchor="c",
        )

        pdf.saveState()
        pdf.setFillColor(HexColor("#FFFFFF"))
        pdf.setFillAlpha(0.86)
        pdf.rect(
            x + 2.0,
            image_y + image_height - 12.0,
            17.0,
            11.0,
            fill=1,
            stroke=0,
        )
        pdf.restoreState()
        pdf.setFillColor(HexColor("#111111"))
        pdf.setFont(FONT_BOLD, 9.0)
        pdf.drawString(x + 4.0, image_y + image_height - 10.0, panel_label)
        pdf.setFont(FONT_REGULAR, 9.0)
        if pdfmetrics.stringWidth(task_name, FONT_REGULAR, 9.0) > panel_width:
            raise ValueError(f"Task label does not fit its panel: {task_name}")
        pdf.drawCentredString(x + 0.5 * panel_width, 2.0, task_name)

    pdf.showPage()
    pdf.save()


if __name__ == "__main__":
    main()
