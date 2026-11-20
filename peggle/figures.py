"""The README's figures, drawn as SVG with no third-party packages.

    uv run python -m peggle.figures      # rewrites peggle/figures/*.svg

`record` draws ten games, six of them won, and the posterior of the winner's chance p under
the uniform prior; the shaded area, p > 1/2, is 743/1024. `stopping` draws, for equal
players, the exact chance that someone is declared better, for caps of 10 to 200 games,
from `stopping.py`. A test compares both with the committed files.
"""

from fractions import Fraction
from html import escape
from math import comb
from pathlib import Path

from peggle import stopping
from peggle.peggle import better

HERE = Path(__file__).parent
OUT = HERE / "figures"
HALF = Fraction(1, 2)
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


# Ten games and what they say about p

RECORD = "WLWWLWLWWL"  # six wins in ten games, in an arbitrary order


def density(p: float, k: int, n: int) -> float:
    """The posterior density of p after k wins in n games, under the uniform prior."""
    return (n + 1) * comb(n, k) * p**k * (1 - p) ** (n - k)


def record() -> str:
    k, n = RECORD.count("W"), len(RECORD)
    chance = better(k, n)
    left, top, width, height = 60, 90, 520, 200
    ymax = 3.0

    def sx(p: float) -> float:
        return left + p * width

    def sy(d: float) -> float:
        return top + height - d / ymax * height

    svg = Svg(
        left + width + 30,
        top + height + 54,
        "Ten games, six won, and the chance that the winner is the better player",
        "A row of ten squares marks six wins and four losses. Below it, the posterior density "
        "of the winner's chance p of winning a game, under a uniform prior, peaks at 0.6; the "
        "area to the right of 1/2, shaded, is 743/1024, about 0.726.",
    )
    svg.text(10, 30, "Record", 13, font_weight="bold")
    for i, r in enumerate(RECORD):
        x = 80 + i * 40
        won = r == "W"
        svg.add(
            "rect",
            x=x,
            y=12,
            width=34,
            height=34,
            rx=4,
            fill="#2a78d6" if won else "#ffffff",
            stroke=INK,
            stroke_width=1.5,
        )
        svg.text(
            x + 17,
            35,
            r,
            15,
            fill="#ffffff" if won else INK,
            font_weight="bold",
            text_anchor="middle",
        )
    svg.text(490, 35, f"{k} wins in {n}", 13, fill=MUTED)
    xs = [i / 200 for i in range(201)]
    shaded = [x for x in xs if x >= 0.5]
    area = " ".join(f"{num(sx(x))},{num(sy(density(x, k, n)))}" for x in shaded)
    svg.add(
        "polygon",
        points=f"{num(sx(0.5))},{num(sy(0))} {area} {num(sx(1))},{num(sy(0))}",
        fill="#dcebfb",
    )
    for d in (0, 1, 2, 3):
        svg.add("line", x1=left, y1=sy(d), x2=left + width, y2=sy(d), stroke=GRID)
        svg.text(left - 8, sy(d) + 4, str(d), 11, fill=MUTED, text_anchor="end")
    for p in (0, 0.25, 0.5, 0.75, 1):
        svg.text(sx(p), top + height + 16, f"{p:g}", 11, fill=MUTED, text_anchor="middle")
    line = " ".join(f"{num(sx(x))},{num(sy(density(x, k, n)))}" for x in xs)
    svg.add("polyline", points=line, fill="none", stroke="#2a78d6", stroke_width=2)
    svg.add(
        "line",
        x1=sx(0.5),
        y1=top,
        x2=sx(0.5),
        y2=top + height,
        stroke=MUTED,
        stroke_dasharray="4 3",
    )
    svg.text(
        sx(0.72),
        sy(1.0),
        f"P(p > 1/2) = {chance.numerator}/{chance.denominator} ≈ {float(chance):.3f}",
        13,
        text_anchor="middle",
    )
    svg.text(
        left + width / 2,
        top + height + 36,
        "p, the winner's chance of winning a game",
        12,
        fill=MUTED,
        text_anchor="middle",
    )
    svg.text(10, top - 10, "posterior density, uniform prior", 12, fill=MUTED)
    return svg.render()


# Equal players and stopping rules

CAPS = range(10, 201, 10)


def stopping_rules() -> str:
    arms = [
        ("play until 95% sure", lambda cap: stopping.outcome(HALF, cap), "#d95926"),
        ("one look after all the games", lambda cap: stopping.fixed_length(HALF, cap), "#3d3d3d"),
        (
            "play until the Bayes factor reaches 20",
            lambda cap: stopping.outcome(HALF, cap, stopping.bayes_factor_rule),
            "#2a78d6",
        ),
    ]
    left, top, width, height, ymax = 60, 80, 520, 240, 0.7

    def sx(cap: float) -> float:
        return left + (cap - 10) / 190 * width

    def sy(v: float) -> float:
        return top + height - v / ymax * height

    svg = Svg(
        left + width + 30,
        top + height + 54,
        "Equal players: the chance that someone is declared better",
        "Exact chances for two players of equal skill, for caps of 10 to 200 games. Playing "
        "until one is 95% sure declares someone better more and more often, 0.528 at 100 "
        "games; one look after all the games stays near 0.09; playing until the Bayes factor "
        "reaches 20 stays below 0.05.",
    )
    for v in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        svg.add("line", x1=left, y1=sy(v), x2=left + width, y2=sy(v), stroke=GRID)
        svg.text(left - 8, sy(v) + 4, f"{v:.1f}", 11, fill=MUTED, text_anchor="end")
    svg.add(
        "line",
        x1=left,
        y1=sy(0.05),
        x2=left + width,
        y2=sy(0.05),
        stroke=MUTED,
        stroke_dasharray="4 3",
    )
    svg.text(left + width, sy(0.05) - 4, "0.05", 11, fill=MUTED, text_anchor="end")
    for cap in (10, 50, 100, 150, 200):
        svg.text(sx(cap), top + height + 16, str(cap), 11, fill=MUTED, text_anchor="middle")
    svg.text(
        left + width / 2,
        top + height + 36,
        "cap: the most games they will play",
        12,
        fill=MUTED,
        text_anchor="middle",
    )
    svg.text(10, 20, "chance that someone is declared better", 12, fill=MUTED)
    for a, (label, rule, color) in enumerate(arms):
        values = []
        for cap in CAPS:
            o = rule(cap)
            values.append(float(o.first + o.second))
        line = " ".join(f"{num(sx(c))},{num(sy(v))}" for c, v in zip(CAPS, values, strict=True))
        svg.add("polyline", points=line, fill="none", stroke=color, stroke_width=2)
        for c, v in zip(CAPS, values, strict=True):
            svg.add("circle", cx=sx(c), cy=sy(v), r=3, fill=color)
        y = 36 + 17 * a
        svg.add("rect", x=left + 10, y=y - 9, width=12, height=12, fill=color)
        svg.text(left + 28, y + 2, f"{label}: {values[9]:.3f} at 100 games", 12)
    return svg.render()


FIGURES = {"record": record, "stopping": stopping_rules}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, draw in FIGURES.items():
        (OUT / f"{name}.svg").write_text(draw(), encoding="utf-8")


if __name__ == "__main__":
    main()
