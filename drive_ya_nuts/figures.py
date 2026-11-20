"""The README's figures, drawn as SVG with no third-party packages.

    uv run python -m drive_ya_nuts.figures      # rewrites drive_ya_nuts/figures/*.svg

`puzzle` draws the original seven nuts in their one solution, found here by a direct search
over every center, rotation and order of the outer nuts. `distribution` draws the exact
counts in `exact_counts.csv`. A test compares both with the committed files.
"""

import csv
import math
from html import escape
from itertools import permutations
from pathlib import Path

from drive_ya_nuts.solver import nuts

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


# The puzzle

type Order = tuple[int, ...]


def edge(order: Order, turn: int, side: int) -> int:
    """The number on side `side` of a nut turned by `turn` sixths, sides counted counterclockwise."""
    return order[(side - turn) % 6]


def solutions() -> list[tuple[tuple[Order, int], ...]]:
    """Every solution with the center unturned, as (order, turn) for the center and sides 0 to 5.

    Outer nut d sits beyond side d of the center, so its side d + 3 touches the center's side d,
    and its side d + 2 touches side d + 5 of outer nut d + 1.
    """
    orders = sorted(n.order for n in nuts)
    found = []
    for center in orders:
        rest = [o for o in orders if o != center]
        for outer in permutations(rest):
            placed = []
            for d, order in enumerate(outer):
                want = edge(center, 0, d)
                turn = next(t for t in range(6) if edge(order, t, d + 3) == want)
                placed.append((order, turn))
            if all(
                edge(*placed[d], d + 2) == edge(*placed[(d + 1) % 6], (d + 5) % 6) for d in range(6)
            ):
                found.append(((center, 0), *placed))
    return found


def puzzle() -> str:
    (solution,) = solutions()
    r = 62  # circumradius of a nut
    cx, cy = 230, 200
    svg = Svg(
        460,
        410,
        "The seven Drive Ya Nuts nuts in their one solution",
        "Seven hexagons, one in the center and six around it, each with the numbers 1 to 6 on "
        "its sides. Every pair of touching sides shows the same number, and each matching pair "
        "is drawn in the same color.",
    )
    palette = ["#2a78d6", "#d95926", "#1baf7a", "#c98500", "#d55181", "#6250d6"]
    for k, (order, turn) in enumerate(solution):
        if k == 0:
            x, y = cx, cy
        else:
            a = math.radians(30 + 60 * (k - 1))
            x, y = cx + math.sqrt(3) * r * math.cos(a), cy - math.sqrt(3) * r * math.sin(a)
        corners = " ".join(
            f"{num(x + r * math.cos(math.radians(60 * i)))},{num(y - r * math.sin(math.radians(60 * i)))}"
            for i in range(6)
        )
        svg.add("polygon", points=corners, fill="#f4f3ef", stroke=INK, stroke_width=2)
        for side in range(6):
            value = edge(order, turn, side)
            a = math.radians(30 + 60 * side)
            tx, ty = x + 0.66 * r * math.cos(a), y - 0.66 * r * math.sin(a)
            svg.text(
                tx,
                ty + 6,
                str(value),
                17,
                fill=palette[value - 1],
                font_weight="bold",
                text_anchor="middle",
            )
        svg.add("circle", cx=x, cy=y, r=9, fill="#ffffff", stroke=MUTED)
    svg.text(
        10,
        398,
        "Nuts read counterclockwise, as in solver.py. Touching sides match.",
        12,
        fill=MUTED,
    )
    return svg.render()


# How many solutions a random set has

SUPERSCRIPT = str.maketrans(
    "0123456789", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"
)


def distribution() -> str:
    with (HERE / "exact_counts.csv").open() as f:
        rows = [(int(r["solutions"]), int(r["sets"])) for r in csv.DictReader(f)]
    total = sum(n for _, n in rows)
    counts = dict(rows)
    left, top, width, height = 70, 44, 560, 260
    lo, hi = 2, 11  # log10 of the axis ends
    bw = width / 21

    def sy(n: float) -> float:
        return top + height - (math.log10(n) - lo) / (hi - lo) * height

    svg = Svg(
        left + width + 20,
        top + height + 60,
        "How many solutions a set of seven distinct nuts has",
        "Dots on a log scale for the 59,487,568,920 sets of seven distinct nuts by their "
        "number of solutions, from exact_counts.csv: 93.195% have none, 5.346% have one, 1.459% "
        "have two or more, no set has 17 to 19, and no set has more than 20.",
    )
    for p in range(lo, hi + 1):
        y = sy(10**p)
        svg.add("line", x1=left, y1=y, x2=left + width, y2=y, stroke=GRID)
        svg.text(
            left - 8, y + 4, "10" + str(p).translate(SUPERSCRIPT), 11, fill=MUTED, text_anchor="end"
        )
    # dots, not bars: on a log axis a bar's length means nothing; a count of 0 cannot be drawn,
    # so it is written in as "0"
    for k in range(21):
        x = left + k * bw
        if counts.get(k):
            svg.add(
                "circle",
                cx=x + bw / 2,
                cy=sy(counts[k]),
                r=5,
                fill="#2a78d6" if k != 1 else "#d95926",
            )
        else:
            svg.text(x + bw / 2, top + height - 6, "0", 11, fill=MUTED, text_anchor="middle")
        if k % 2 == 0 or k == 1:
            svg.text(x + bw / 2, top + height + 16, str(k), 11, fill=MUTED, text_anchor="middle")
    for k in (0, 1, 2):
        share = f"{100 * counts[k] / total:.3f}%"
        svg.text(left + (k + 1) * bw - 4, sy(counts[k]) + 4, share, 11)
    more = sum(n for k, n in counts.items() if k >= 2)
    svg.text(
        left + 6 * bw,
        sy(counts[1]) - 14,
        f"one solution: {100 * counts[1] / total:.3f}%; two or more, all together: "
        f"{100 * more / total:.3f}%",
        12,
    )
    svg.text(left + width / 2, top + height + 36, "solutions", 12, fill=MUTED, text_anchor="middle")
    svg.text(10, 16, "sets of seven nuts (log scale)", 12, fill=MUTED)
    svg.text(
        left + width,
        16,
        "orange: one solution, like the original",
        12,
        fill=MUTED,
        text_anchor="end",
    )
    return svg.render()


FIGURES = {"puzzle": puzzle, "distribution": distribution}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, draw in FIGURES.items():
        (OUT / f"{name}.svg").write_text(draw(), encoding="utf-8")


if __name__ == "__main__":
    main()
