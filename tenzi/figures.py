"""The README's figures, drawn as SVG with no third-party packages.

    uv run python -m tenzi.figures      # rewrites tenzi/figures/*.svg

`rolls` draws the first two turns of a seeded game with the fixed-face rule. `turns` draws
the exact distribution of the number of turns for the three rules, from the Markov chain in
`tenzi.py`. A test compares both with the committed files.
"""

import random
from collections import Counter
from html import escape
from pathlib import Path

from tenzi.tenzi import DICE, FACES, STRATEGIES, expected_turns, turn_distribution

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


# Two turns of a game

PIPS = {
    1: [(0, 0)],
    2: [(-1, 1), (1, -1)],
    3: [(-1, 1), (0, 0), (1, -1)],
    4: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    5: [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)],
    6: [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1)],
}


def die(svg: Svg, x: float, y: float, face: int, kept: bool, size: float = 44) -> None:
    fill, pip = ("#2a78d6", "#ffffff") if kept else ("#ffffff", INK)
    svg.add(
        "rect", x=x, y=y, width=size, height=size, rx=8, fill=fill, stroke=INK, stroke_width=1.5
    )
    for dx, dy in PIPS[face]:
        svg.add(
            "circle",
            cx=x + size / 2 + dx * size * 0.27,
            cy=y + size / 2 + dy * size * 0.27,
            r=size * 0.08,
            fill=pip,
        )


def play_turns(seed: int, turns: int) -> list[tuple[list[int], int]]:
    """Each turn's dice and the face kept: the most common face after the first roll, then fixed."""
    rng = random.Random(seed)
    dice = [rng.randint(1, FACES) for _ in range(DICE)]
    counts = Counter(dice)
    face = max(sorted(counts), key=lambda f: counts[f])
    played = [(list(dice), face)]
    for _ in range(turns - 1):
        dice = [d if d == face else rng.randint(1, FACES) for d in dice]
        played.append((list(dice), face))
    return played


def rolls(seed: int = 4) -> str:
    played = play_turns(seed, 2)
    svg = Svg(
        640,
        160,
        "The first two turns of a game of Tenzi",
        "Two rows of ten dice. After the first roll the most common face is kept (shaded blue); "
        "the second row shows the kept dice and the others rolled again.",
    )
    for row, (dice, face) in enumerate(played):
        y = 14 + row * 70
        kept = sum(d == face for d in dice)
        svg.text(10, y + 28, f"Turn {row + 1}", 13, font_weight="bold")
        for i, d in enumerate(dice):
            die(svg, 80 + i * 50, y, d, d == face)
        svg.text(586, y + 28, f"{kept} kept", 13, fill=MUTED)
    svg.text(
        10, 150, f"Seed {seed}. Shaded: dice showing the kept face, which stay put.", 12, fill=MUTED
    )
    return svg.render()


# How many turns a game takes


def turns(upto: int = 50) -> str:
    rules = [
        ("target first", "face named before the first roll", "#6250d6"),
        ("fixed face", "keep the most common face", "#2a78d6"),
        ("switching", "switch to a larger group (dashed)", "#d95926"),
    ]
    dists = {}
    for name, _, _ in rules:
        first, step = STRATEGIES[name]
        dists[name] = turn_distribution(upto, step, first)
    means = {name: expected_turns(STRATEGIES[name][1], STRATEGIES[name][0]) for name, _, _ in rules}
    left, top, width, height, ymax = 60, 70, 540, 240, 0.08

    def sx(t: float) -> float:
        return left + (t - 1) / (upto - 1) * width

    def sy(p: float) -> float:
        return top + height - p / ymax * height

    svg = Svg(
        left + width + 30,
        top + height + 50,
        "The chance that a game of Tenzi ends on each turn, for three rules",
        "Three exact distributions over turns 1 to 50. Keeping the most common face and "
        "switching to a larger group both peak at turn 12; naming the face before the first "
        "roll shifts the curve later. Means: 15.35, 15.29 and 16.56 turns.",
    )
    for p in (0.0, 0.02, 0.04, 0.06, 0.08):
        svg.add("line", x1=left, y1=sy(p), x2=left + width, y2=sy(p), stroke=GRID)
        svg.text(left - 8, sy(p) + 4, f"{p:.2f}", 11, fill=MUTED, text_anchor="end")
    for t in (1, 10, 20, 30, 40, 50):
        svg.text(sx(t), top + height + 16, str(t), 11, fill=MUTED, text_anchor="middle")
    svg.text(
        left + width / 2,
        top + height + 36,
        "turn on which the game ends (the first roll is turn 1)",
        12,
        fill=MUTED,
        text_anchor="middle",
    )
    svg.text(10, 20, "chance", 12, fill=MUTED)
    for k, (name, label, color) in enumerate(rules):
        dist = dists[name]
        line = " ".join(f"{num(sx(t))},{num(sy(float(dist[t])))}" for t in range(1, upto + 1))
        dash = {"stroke_dasharray": "6 4"} if name == "switching" else {}
        svg.add("polyline", points=line, fill="none", stroke=color, stroke_width=2, **dash)
        y = 30 + 17 * k
        svg.add("rect", x=left + 250, y=y - 9, width=12, height=12, fill=color)
        svg.text(left + 268, y + 2, f"{label}: mean {float(means[name]):.2f}", 12)
    return svg.render()


FIGURES = {"rolls": rolls, "turns": turns}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, draw in FIGURES.items():
        (OUT / f"{name}.svg").write_text(draw(), encoding="utf-8")


if __name__ == "__main__":
    main()
