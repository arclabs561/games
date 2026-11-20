"""The README's picture of a deal, drawn as SVG with the standard library only.

    uv run python -m hanabi.figures      # rewrites hanabi/figures/deal.svg

`deal` draws the start of two-player deal 161 of plan H1: both hands, the five empty
piles, the hint tokens and lives, and the 40 cards of the deck in the order they will be
drawn. A test compares the drawing with the committed file.
"""

from html import escape
from pathlib import Path

from hanabi.rules import COLOR_NAMES, Rules, color_of, new_game, rank_of, seeded_deck

HERE = Path(__file__).parent
OUT = HERE / "figures"
SANS = "Helvetica, Arial, sans-serif"
INK, MUTED, LINE = "#0b0b0b", "#52514e", "#bdbbb5"
# fill and text color of each card color, in the order of COLOR_NAMES (red, yellow, green, blue, white)
FILL = ["#c8372d", "#e0a800", "#2c8c4a", "#2a64c4", "#f2f1ec"]
TEXT = ["#ffffff", INK, "#ffffff", "#ffffff", INK]
COLOR_WORDS = ["red", "yellow", "green", "blue", "white"]
SEED = 161


class Svg:
    """A list of SVG elements on a white background, with a title and a description."""

    def __init__(self, width: int, height: int, title: str, desc: str):
        self.width, self.height, self.title, self.desc = width, height, title, desc
        self.items: list[str] = []

    def add(self, tag: str, content: str = "", **attrs) -> None:
        text = "".join(
            f' {k.rstrip("_").replace("_", "-")}="{escape(str(v))}"' for k, v in attrs.items()
        )
        self.items.append(
            f"<{tag}{text}>{escape(content)}</{tag}>" if content else f"<{tag}{text}/>"
        )

    def text(self, x: int, y: int, s: str, size: int = 13, **attrs) -> None:
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


def card(svg: Svg, c: int, rules: Rules, x: int, y: int, w: int, h: int) -> None:
    """A card face: its rank large in the middle, its color's letter in the corner."""
    k = color_of(c, rules)
    svg.add("rect", x=x, y=y, width=w, height=h, rx=w // 8, fill=FILL[k], stroke=INK)
    svg.text(
        x + w // 2,
        y + h // 2 + h // 7,
        str(rank_of(c, rules)),
        size=h // 2,
        fill=TEXT[k],
        text_anchor="middle",
        font_weight="bold",
    )
    svg.text(x + w // 10 + 2, y + h // 5 + 2, COLOR_NAMES[k], size=max(9, h // 6), fill=TEXT[k])


def deal() -> str:
    rules = Rules(players=2)
    state = new_game(rules, seeded_deck(rules, SEED))
    svg = Svg(
        780,
        390,
        f"Two-player Hanabi deal {SEED}, before the first turn",
        "Two hands of five cards, five empty piles, 8 hint tokens, 3 lives, and the 40 cards "
        "of the deck in draw order.",
    )
    w, h = 54, 76
    for p, hand in enumerate(state.hands):
        y = 34 + p * 112
        svg.text(20, y - 10, f"Player {p + 1}, seen by player {2 - p}", size=13, fill=MUTED)
        for i, c in enumerate(hand):
            card(svg, c, rules, 20 + i * (w + 8), y, w, h)
    # the piles
    px = 400
    svg.text(px, 24, "Piles, built from 1 to 5", size=13, fill=MUTED)
    for k in range(rules.colors):
        x = px + k * (w + 10)
        svg.add(
            "rect",
            x=x,
            y=34,
            width=w,
            height=h,
            rx=w // 8,
            fill="#ffffff",
            stroke=FILL[k] if k < 4 else MUTED,
            stroke_width=2,
            stroke_dasharray="5 3",
        )
        svg.text(
            x + w // 2, 34 + h // 2 + 4, COLOR_WORDS[k], size=11, fill=MUTED, text_anchor="middle"
        )
    # tokens and lives
    svg.text(px, 146, f"Hint tokens: {state.hints}", size=13, fill=MUTED)
    for i in range(state.hints):
        svg.add("circle", cx=px + 8 + i * 22, cy=164, r=8, fill="#2a64c4")
    svg.text(px + 200, 146, f"Lives: {state.lives}", size=13, fill=MUTED)
    for i in range(state.lives):
        svg.add(
            "circle",
            cx=px + 208 + i * 22,
            cy=164,
            r=8,
            fill="#ffffff",
            stroke="#c8372d",
            stroke_width=3,
        )
    # the deck
    rest = state.deck[state.pos :]
    svg.text(
        20,
        268,
        f"The deck: {len(rest)} cards, drawn from the left; no player sees them",
        size=13,
        fill=MUTED,
    )
    dw, dh = 34, 46
    for i, c in enumerate(rest):
        card(svg, c, rules, 20 + (i % 20) * (dw + 4), 278 + (i // 20) * (dh + 8), dw, dh)
    return svg.render()


FIGURES = {"deal": deal}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, draw in FIGURES.items():
        (OUT / f"{name}.svg").write_text(draw())


if __name__ == "__main__":
    main()
