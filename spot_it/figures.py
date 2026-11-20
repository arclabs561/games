"""The README's figures, drawn as SVG with no third-party packages.

    uv run python -m spot_it.figures      # rewrites spot_it/figures/*.svg

Each of the 57 symbols of the plane of order 7 is drawn as a generic glyph: one of eight
shapes in one of eight colors, with the color's number, 1 to 8, written inside so that no
symbol is told from another by color alone. The colors are the Okabe-Ito set with brown in
place of yellow (too faint on white); every two stay at least 0.076 apart in Oklab under a
simulation of each of the three kinds of color blindness (Machado et al., 2009). `pair` draws two cards of the plane and their one shared
symbol. `missing` removes two cards at random (seed 0), counts the symbols of the 55 left,
and draws the two cards that the counts give back. A test compares both with the
committed files.
"""

import math
import random
from html import escape
from pathlib import Path

from spot_it.missing import missing_by_counting
from spot_it.plane import projective_plane, symbol_counts

HERE = Path(__file__).parent
OUT = HERE / "figures"
SANS = "Helvetica, Arial, sans-serif"
INK, MUTED, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#d9d8d4", "#2a78d6", "#d95926"


def num(x: float) -> str:
    """A coordinate with at most two decimals and no trailing zeros, so the bytes are stable."""
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


class Svg:
    """A list of SVG elements on a white background, with a title and a description."""

    def __init__(self, width: int, height: int, title: str, desc: str):
        self.width, self.height, self.title, self.desc = width, height, title, desc
        self.items: list[str] = []

    def add(self, tag: str, content: str = "", **attrs) -> None:
        text = "".join(
            f' {k.rstrip("_").replace("_", "-")}="{num(v) if isinstance(v, float | int) else escape(v)}"'
            for k, v in attrs.items()
        )
        self.items.append(
            f"<{tag}{text}>{escape(content)}</{tag}>" if content else f"<{tag}{text}/>"
        )

    def text(self, x: float, y: float, s: str, size: float = 13, **attrs) -> None:
        attrs.setdefault("fill", INK)
        self.add("text", s, x=x, y=y, font_size=size, **attrs)

    def render(self) -> str:
        return "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}"'
                f' viewBox="0 0 {self.width} {self.height}" role="img" font-family="{SANS}">',
                f"<title>{escape(self.title)}</title>",
                f"<desc>{escape(self.desc)}</desc>",
                f'<rect width="{self.width}" height="{self.height}" fill="#ffffff"/>',
                *self.items,
                "</svg>",
                "",
            ]
        )


# Glyphs

COLORS = ["#000000", "#e69f00", "#56b4e9", "#009e73", "#8c510a", "#0072b2", "#d55e00", "#cc79a7"]
# the color of the number written on a glyph of each color: dark on the light fills
LABEL_INK = ["#ffffff", INK, INK, "#ffffff", "#ffffff", "#ffffff", INK, INK]
SHAPES = ["circle", "square", "triangle", "diamond", "star", "hexagon", "plus", "ring"]


def polygon(x: float, y: float, r: float, corners: int, turn: float = 90, inner: float = 1) -> str:
    points = []
    for i in range(corners * (2 if inner != 1 else 1)):
        a = math.radians(turn + i * 360 / len(range(corners * (2 if inner != 1 else 1))))
        rr = r if i % 2 == 0 or inner == 1 else r * inner
        points.append(f"{num(x + rr * math.cos(a))},{num(y - rr * math.sin(a))}")
    return " ".join(points)


def glyph(svg: Svg, symbol: int, x: float, y: float, r: float) -> None:
    """Symbol s is shape s mod 8 in color s div 8, with the color's number inside."""
    shape, color = SHAPES[symbol % 8], COLORS[symbol // 8]
    label = LABEL_INK[symbol // 8] if shape != "ring" else INK
    if shape == "circle":
        svg.add("circle", cx=x, cy=y, r=r * 0.85, fill=color)
    elif shape == "ring":
        svg.add("circle", cx=x, cy=y, r=r * 0.7, fill="none", stroke=color, stroke_width=r * 0.35)
    elif shape == "square":
        svg.add("rect", x=x - r * 0.75, y=y - r * 0.75, width=r * 1.5, height=r * 1.5, fill=color)
    elif shape == "plus":
        w = r * 0.55
        svg.add("rect", x=x - r, y=y - w / 2, width=2 * r, height=w, fill=color)
        svg.add("rect", x=x - w / 2, y=y - r, width=w, height=2 * r, fill=color)
    else:
        corners, turn, inner = {
            "triangle": (3, 90, 1),
            "diamond": (4, 90, 1),
            "star": (5, 90, 0.45),
            "hexagon": (6, 0, 1),
        }[shape]
        svg.add("polygon", points=polygon(x, y, r, corners, turn, inner), fill=color)
    dy = {"triangle": r * 0.05}.get(shape, 0)
    svg.add(
        "text",
        str(symbol // 8 + 1),
        x=x,
        y=y + dy + r * 0.32,
        font_size=r * (0.7 if shape in ("triangle", "star") else 0.9),
        font_weight="bold",
        fill=label,
        text_anchor="middle",
    )


def card(
    svg: Svg, symbols, x: float, y: float, radius: float, mark=frozenset(), seed: int = 0
) -> None:
    """A round card: one symbol in the middle and seven around it, turned by a seeded angle."""
    rng = random.Random(seed)
    svg.add("circle", cx=x, cy=y, r=radius, fill="#ffffff", stroke=INK, stroke_width=2)
    order = sorted(symbols)
    rng.shuffle(order)
    turn = rng.uniform(0, 360)
    places = [(x, y)] + [
        (
            x + 0.62 * radius * math.cos(math.radians(turn + i * 360 / 7)),
            y - 0.62 * radius * math.sin(math.radians(turn + i * 360 / 7)),
        )
        for i in range(7)
    ]
    for s, (px, py) in zip(order, places, strict=True):
        if s in mark:
            svg.add(
                "circle",
                cx=px,
                cy=py,
                r=radius * 0.24,
                fill="none",
                stroke=INK,
                stroke_width=1.5,
                stroke_dasharray="3 2",
            )
        glyph(svg, s, px, py, radius * 0.15)


# Two cards


def pair() -> str:
    deck = projective_plane(7)
    a, b = deck[10], deck[40]
    (shared,) = a & b
    svg = Svg(
        520,
        270,
        "Two cards of the plane of order 7 and their one shared symbol",
        "Two round cards with eight generic symbols each: shapes in colors, each with its "
        "color's number inside. Exactly one symbol "
        "is on both cards, and it is circled with a dashed line on each.",
    )
    card(svg, a, 130, 125, 110, frozenset({shared}), seed=1)
    card(svg, b, 390, 125, 110, frozenset({shared}), seed=2)
    svg.text(
        260,
        258,
        "Cards 10 and 40 of the plane of order 7; the dashed circles mark the one shared symbol.",
        12,
        fill=MUTED,
        text_anchor="middle",
    )
    return svg.render()


# The two missing cards


def missing(seed: int = 0) -> str:
    deck = projective_plane(7)
    removed = random.Random(seed).sample(range(len(deck)), 2)
    left = [c for i, c in enumerate(deck) if i not in removed]
    counts = symbol_counts(left, 57)
    found = missing_by_counting(left, 7)
    assert sorted(found, key=sorted) == sorted((deck[i] for i in removed), key=sorted)
    group = {s: k for k, c in enumerate(found) for s in c}
    tint = {0: "#dcebfb", 1: "#fde3d8"}
    cell, cols, gx, gy = 46, 8, 20, 54
    svg = Svg(
        700,
        gy + cell * 8 + 30,
        "Symbol counts in a 55-card deck, and the two cards they give back",
        "A grid of the 57 symbols, each with the number of the 55 cards that show it: 42 are "
        "on 8 cards, 14 on 7 and one on 6. The symbols on 7 cards are shaded by which of the "
        "two missing cards they belong to, and the symbol on 6 cards, on both, is outlined. "
        "On the right are the two missing cards built from them.",
    )
    svg.text(gx, 24, "How many of the 55 cards show each symbol", 14, font_weight="bold")
    svg.text(
        gx,
        42,
        "shaded: on 7 cards, colored by missing card; outlined: on 6 cards, on both",
        12,
        fill=MUTED,
    )
    for s in range(57):
        x, y = gx + (s % cols) * cell, gy + (s // cols) * cell
        if counts[s] == 7:
            svg.add(
                "rect", x=x + 1, y=y + 1, width=cell - 2, height=cell - 2, rx=4, fill=tint[group[s]]
            )
        if counts[s] == 6:
            svg.add(
                "rect",
                x=x + 1,
                y=y + 1,
                width=cell - 2,
                height=cell - 2,
                rx=4,
                fill="#ffffff",
                stroke=INK,
                stroke_width=2,
            )
        glyph(svg, s, x + cell / 2, y + 17, 10)
        svg.text(
            x + cell / 2,
            y + 41,
            str(counts[s]),
            11,
            fill=INK if counts[s] < 8 else MUTED,
            text_anchor="middle",
        )
    cx = gx + cols * cell + 150
    svg.text(cx, 24, "The two missing cards", 14, font_weight="bold", text_anchor="middle")
    for k, c in enumerate(found):
        y = gy + 95 + k * 200
        svg.add("circle", cx=cx, cy=y, r=96, fill=tint[k])
        card(svg, c, cx, y, 88, frozenset(s for s in c if counts[s] == 6), seed=3 + k)
    return svg.render()


FIGURES = {"pair": pair, "missing": missing}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, draw in FIGURES.items():
        (OUT / f"{name}.svg").write_text(draw(), encoding="utf-8")


if __name__ == "__main__":
    main()
