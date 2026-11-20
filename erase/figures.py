"""The README's figure: two deals, one that clears and one that cannot, drawn as SVG.

    python3 erase/figures.py      # rewrites erase/figures/deals.svg

Standard library only. The groups are the ones `erase.py` finds; a test compares the
figure with the committed file.
"""

from html import escape
from pathlib import Path

from erase import group, solve

HERE = Path(__file__).parent
OUT = HERE / "figures"
DEALS = [  # hand, center: the two examples of the README's usage
    ((4, 4, 6, 9, 5, 8, 5), (7, 5, 1, 8)),
    ((0, 4, 4, 4, 8, 8, 8), (0, 3, 7, 9)),
]
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


def card_shape(svg: Svg, x: float, y: float, value: int, center: bool) -> None:
    fill = "#f4f3ef" if center else "#ffffff"
    svg.add("rect", x=x, y=y, width=42, height=60, rx=6, fill=fill, stroke=INK, stroke_width=1.5)
    svg.text(x + 21, y + 39, str(value), 22, font_weight="bold", text_anchor="middle")


def deals() -> str:
    svg = Svg(
        660,
        420,
        "Two deals of Erase: one that clears and one that cannot",
        "Each deal shows seven hand cards and four center cards. The first hand, 4 4 5 5 6 8 9 "
        "with center 1 5 7 8, clears in groups that make 10, or in one group of all seven hand cards. "
        "The second hand, 0 4 4 4 8 8 8 with center 0 3 7 9, cannot be cleared.",
    )
    for d, (hand, center) in enumerate(DEALS):
        top = 14 + d * 206
        groups = solve(hand, center)
        title = "This deal clears" if groups is not None else "This deal cannot be cleared"
        svg.text(10, top + 14, title, 14, font_weight="bold")
        svg.text(10, top + 50, "hand", 12, fill=MUTED)
        for i, v in enumerate(sorted(hand)):
            card_shape(svg, 60 + i * 48, top + 24, v, False)
        svg.text(10, top + 120, "center", 12, fill=MUTED)
        for i, v in enumerate(sorted(center)):
            card_shape(svg, 60 + i * 48, top + 94, v, True)
        x = 420
        if groups is None:
            svg.text(x, top + 50, "erase.py finds no set of groups", 13)
            svg.text(x, top + 68, "that erases every hand card.", 13)
        else:
            svg.text(x, top + 50, "groups that make 10:", 13, fill=MUTED)
            for i, g in enumerate(groups):
                expr, _, note = g.partition("   [center ")
                text = expr.replace("*", "×").replace("-", "−")
                svg.text(
                    x,
                    top + 72 + 20 * i,
                    f"{text}   center {note.rstrip(']')}" if note else text,
                    14,
                    font_family="Menlo, Consolas, monospace",
                )
            one = group(tuple(sorted(hand)), center)
            assert one is not None
            expr, _, note = one.partition("   [center ")
            text = expr.replace("*", "\u00d7").replace("-", "\u2212")
            extra = f", with center {note.rstrip(']')}" if note else ", with no center card"
            svg.text(
                10,
                top + 178,
                f"or all seven in one group: {text}{extra}",
                13,
                font_family="Menlo, Consolas, monospace",
            )
    svg.text(
        10,
        412,
        "Shaded cards lie in the center and can be used again; hand cards are erased.",
        12,
        fill=MUTED,
    )
    return svg.render()


FIGURES = {"deals": deals}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, draw in FIGURES.items():
        (OUT / f"{name}.svg").write_text(draw(), encoding="utf-8")


if __name__ == "__main__":
    main()
