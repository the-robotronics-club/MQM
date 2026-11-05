"""Consistent category colors and person IDs for photos and live video."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

CATEGORY_COLORS = {
    "Queue": "#3b82f6",
    "Sitting": "#ff9f1c",
    "Other standing": "#22c55e",
    "Person": "#94a3b8",
}
CATEGORY_CODES = {"Queue": "Q", "Sitting": "S", "Other standing": "O", "Person": "P"}
LEGEND = "Blue Q: queue · Orange S: sitting · Green O: other standing. Predictions are estimates."


def annotate_people(image: Image.Image, people: list[dict]) -> Image.Image:
    annotated = image.copy()
    drawing = ImageDraw.Draw(annotated)
    font = ImageFont.load_default(size=max(12, image.width // 110))
    line_width = max(2, image.width // 700)
    for index, person in enumerate(people, 1):
        category = person.get("category", "Person")
        color = CATEGORY_COLORS[category]
        x1, y1, x2, y2 = person["bbox"]
        drawing.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label = f"{index} {CATEGORY_CODES[category]}"
        bounds = drawing.textbbox((x1 + 2, y1 + 2), label, font=font)
        drawing.rectangle(bounds, fill=color)
        drawing.text((x1 + 2, y1 + 2), label, fill="black", font=font)
    return annotated
