"""Draw the README's figures as SVG, with no plotting library, so the bytes are stable.

    uv run python -m yugioh.figures    # writes yugioh/figures/*.svg

- starters.svg: the chance of at least one starter against the number of starters.
- deck_size.svg: the fewest starters for 90% against the deck size.
- hands.svg: six hands dealt from the example deck with seed 1, and the needs each meets.
"""

from fractions import Fraction
from pathlib import Path
from random import Random

from yugioh.odds import (
    FIRST,
    ROLES,
    SECOND,
    Name,
    fewest,
    named,
    no_brick,
    p_at_least_one,
    physical,
    starter,
    starter_and_extender,
    summary,
)
from yugioh.tables import EXAMPLE, SEED

FIGURES = Path(__file__).parent / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NINETY = Fraction(9, 10)
FONT = 'font-family="Helvetica, Arial, sans-serif"'


def svg(width: int, height: int, body: list[str], title: str) -> str:
    """An SVG document with a white background and a title for screen readers."""
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" {FONT} role="img">'
    )
    return "\n".join(
        [head, f"<title>{title}</title>", f'<rect width="{width}" height="{height}" fill="#fff"/>']
        + body
        + ["</svg>", ""]
    )


def text(x: float, y: float, s: str, size: int = 13, fill: str = INK, anchor: str = "start",
         weight: str = "normal") -> str:  # fmt: skip
    """A text element."""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>'
    )


class Axes:
    """A plot area mapping data coordinates to pixels."""

    def __init__(self, left, top, width, height, xlim, ylim):
        self.left, self.top, self.width, self.height = left, top, width, height
        self.xlim, self.ylim = xlim, ylim

    def x(self, v: float) -> float:
        """Pixel column of data value v."""
        a, b = self.xlim
        return self.left + (v - a) / (b - a) * self.width

    def y(self, v: float) -> float:
        """Pixel row of data value v."""
        a, b = self.ylim
        return self.top + self.height - (v - a) / (b - a) * self.height

    def frame(self, xticks, yticks, xlabel, ylabel, yfmt=str) -> list[str]:
        """Grid lines, tick labels and axis titles."""
        out = []
        for v in yticks:
            y = self.y(v)
            out.append(
                f'<line x1="{self.left}" x2="{self.left + self.width}" y1="{y:.1f}" '
                f'y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>'
            )
            out.append(text(self.left - 8, y + 4, yfmt(v), 12, MUTED, "end"))
        base = self.top + self.height
        out.append(
            f'<line x1="{self.left}" x2="{self.left + self.width}" y1="{base}" y2="{base}" '
            f'stroke="{MUTED}" stroke-width="1"/>'
        )
        for v in xticks:
            out.append(text(self.x(v), base + 18, str(v), 12, MUTED, "middle"))
        out.append(text(self.left + self.width / 2, base + 40, xlabel, 13, INK, "middle"))
        out.append(text(self.left, self.top - 14, ylabel, 13, INK, "start"))
        return out

    def line(self, points, color, dash="", step=False) -> list[str]:
        """A 2 px line through the points, as steps if asked, with a round marker on each point."""
        coords = [(self.x(a), self.y(b)) for a, b in points]
        if step:
            path = [coords[0]]
            for (_, y0), (x1, y1) in zip(coords, coords[1:], strict=False):
                path += [(x1, y0), (x1, y1)] if y1 != y0 else [(x1, y1)]
            coords_line = path
        else:
            coords_line = coords
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords_line)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        out = [f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="2"{extra}/>']
        for (x, y), (a, b) in zip(coords, points, strict=True):
            out.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}" stroke="#fff" '
                f'stroke-width="1.5"><title>{a}: {b:.2f}</title></circle>'
            )
        return out


def legend(x: float, y: float, items: list[tuple[str, str, str]]) -> list[str]:
    """A row of line swatches with labels."""
    out = []
    for label, color, dash in items:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(
            f'<line x1="{x}" x2="{x + 22}" y1="{y - 4}" y2="{y - 4}" stroke="{color}" '
            f'stroke-width="2"{extra}/>'
        )
        out.append(text(x + 28, y, label, 12, INK))
        x += 36 + 7 * len(label)
    return out


def starters_figure() -> str:
    """P(at least one starter) against the number of starters."""
    ax = Axes(64, 70, 640, 300, (0, 20), (0, 100))
    series = [
        ("40 cards, going first", 40, FIRST, BLUE, ""),
        ("40 cards, going second", 40, SECOND, ORANGE, ""),
        ("60 cards, going first", 60, FIRST, AQUA, "6 4"),
    ]
    body = ax.frame(range(0, 21, 2), range(0, 101, 20), "Starters in the deck",
                    "Chance of at least one starter in hand (%)", lambda v: f"{v}")  # fmt: skip
    for level in (90,):
        y = ax.y(level)
        body.append(
            f'<line x1="{ax.left}" x2="{ax.left + ax.width}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="{MUTED}" stroke-width="1" stroke-dasharray="2 3"/>'
        )
    for _label, size, n, color, dash in series:
        pts = [(k, 100 * float(p_at_least_one(size, k, n))) for k in range(0, 21)]
        body += ax.line(pts, color, dash)
    k = fewest(NINETY, 40, FIRST)
    assert k is not None
    px, py = ax.x(k), ax.y(100 * float(p_at_least_one(40, k, FIRST)))
    tx, ty = ax.x(15), ax.y(62)
    body.append(
        f'<line x1="{px:.1f}" y1="{py + 5:.1f}" x2="{tx + 4:.1f}" y2="{ty - 14:.1f}" '
        f'stroke="{MUTED}" stroke-width="1"/>'
    )
    body.append(text(tx, ty, f"{k} starters give 90.00% going first", 12, INK))
    body.append(text(tx, ty + 16, "(dotted line: 90%)", 12, MUTED))
    body += legend(ax.left, 36, [(s[0], s[3], s[4]) for s in series])
    return svg(760, 430, body, "Chance of at least one starter against the number of starters")


def deck_size_figure() -> str:
    """Fewest starters for 90% against deck size."""
    ax = Axes(64, 70, 640, 300, (40, 60), (10, 24))
    body = ax.frame(range(40, 61, 2), range(10, 25, 2), "Cards in the main deck",
                    "Fewest starters for at least one in hand 90% of the time")  # fmt: skip
    series = [("going first (5 cards)", FIRST, BLUE), ("going second (6 cards)", SECOND, ORANGE)]
    for _label, n, color in series:
        pts = [(size, fewest(NINETY, size, n)) for size in range(40, 61)]
        body += ax.line(pts, color, step=True)
    body += legend(ax.left, 36, [(label, color, "") for label, _n, color in series])
    return svg(760, 430, body, "Fewest starters for a 90% chance against deck size")


FILL = {"starter": "#d6e6f9", "extender": "#fbe0d4", "brick": "#52514e", "other": "#ffffff"}
INITIAL = {"starter": "S", "extender": "E", "brick": "B", "other": "O"}


def dealt_hands(count: int = 6, seed: int = SEED) -> list[list[Name]]:
    """The first `count` hands of five from the example deck with the seeded generator."""
    rng = Random(seed)
    cards = physical(named(EXAMPLE))
    out = []
    for _ in range(count):
        hand = rng.sample(cards, FIRST)
        out.append(sorted(hand, key=lambda c: (ROLES.index(c.role), c.name)))
    return out


def card(x: float, y: float, c: Name) -> list[str]:
    """A blank card: its role's initial and the number of its name."""
    dark = c.role == "brick"
    ink = "#fff" if dark else INK
    number = c.name.split()[-1]
    return [
        f'<rect x="{x}" y="{y}" width="54" height="76" rx="6" fill="{FILL[c.role]}" '
        f'stroke="{MUTED}" stroke-width="1"><title>{c.name}</title></rect>',
        text(x + 27, y + 44, INITIAL[c.role], 26, ink, "middle", "bold"),
        text(x + 27, y + 66, f"{c.role} {number}", 9, ink, "middle"),
    ]


def hands_figure() -> str:
    """Six dealt hands with a mark for each need."""
    needs = [("starter", starter), ("starter + extender", starter_and_extender),
             ("no brick", no_brick)]  # fmt: skip
    body = [
        text(
            20,
            28,
            "Six hands of five from the example deck (12 S, 9 E, 3 B, 16 O), seed 1",
            14,
            INK,
            "start",
            "bold",
        ),  # fmt: skip
    ]
    for j, (label, _) in enumerate(needs):
        body.append(text(392 + 120 * j, 58, label, 12, MUTED, "middle"))
    for i, hand in enumerate(dealt_hands()):
        y = 70 + 88 * i
        for j, c in enumerate(hand):
            body += card(20 + 62 * j, y, c)
        counts, _ = summary(hand)
        for j, (_, need) in enumerate(needs):
            ok = need(counts)
            body.append(
                text(
                    392 + 120 * j,
                    y + 46,
                    "yes" if ok else "no",
                    15,
                    INK if ok else MUTED,
                    "middle",
                    "bold" if ok else "normal",
                )  # fmt: skip
            )
    return svg(760, 70 + 88 * 6 + 10, body, "Six dealt hands and the needs each one meets")


FIGS = {"starters.svg": starters_figure, "deck_size.svg": deck_size_figure,
        "hands.svg": hands_figure}  # fmt: skip


def main() -> None:
    """Write every figure."""
    FIGURES.mkdir(exist_ok=True)
    for name, draw in FIGS.items():
        (FIGURES / name).write_text(draw())


if __name__ == "__main__":
    main()
