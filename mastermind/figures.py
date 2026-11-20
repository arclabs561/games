"""The README's figures, drawn as SVG with no third-party packages.

    uv run python -m mastermind.figures      # rewrites mastermind/figures/*.svg

`terminal` draws a Fallout 3 style screen and the session that `fallout3.py` plays on it,
the same session as the transcript in the README. `guesses` draws how many guesses two of
the strategies in `strategies.py` and the two optimal trees in `results/optimal.json` (written
by `optimal.py`, a ten-minute search) need, over all 1,296 codes. A test compares both
figures with the committed files.
"""

from collections import Counter
from html import escape
from pathlib import Path

from mastermind import strategies
from mastermind.fallout3 import best_next_guess, likeness

HERE = Path(__file__).parent
OUT = HERE / "figures"
MONO = "Menlo, Consolas, 'DejaVu Sans Mono', monospace"
# The screen of the README's transcript, and its password.
SCREEN = (
    "BRAIN CHAIR DRINK FLAME GHOST HEART KNIFE LIGHT MONEY "
    "NIGHT PLANT QUEEN RIVER SHEEP STONE TRAIN WATER YOUTH"
).split()
PASSWORD = "WATER"
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


# A terminal and one session on it


def session(screen=SCREEN, password=PASSWORD) -> list[tuple[str, int, int]]:
    """The helper's guesses on `screen` until one candidate is left: (guess, candidates, likeness)."""
    tried: dict[str, int] = {}
    played = []
    while True:
        guess, _, left = best_next_guess(screen, tried)
        if left == 1:
            assert guess == password
            return played
        played.append((guess, left, likeness(guess, password)))
        tried[guess] = likeness(guess, password)


def terminal() -> str:
    played = session()
    green, dark, dim = "#7ee2a8", "#0f2418", "#5fae80"
    svg = Svg(
        640,
        330,
        "A terminal screen of 18 five-letter words, and the helper's guesses",
        "A dark terminal panel lists 18 words in two columns. The log on the right shows the "
        "helper's guesses PLANT, QUEEN and RIVER with likeness 0, 1 and 2, after which one "
        "candidate, WATER, is left: the password, entered with the fourth and last attempt.",
    )
    svg.add("rect", x=10, y=10, width=620, height=310, rx=10, fill=dark)
    svg.text(30, 42, "PASSWORD REQUIRED", 15, fill=green, font_family=MONO)
    svg.text(30, 64, "4 ATTEMPTS", 15, fill=green, font_family=MONO)
    guessed = {g for g, _, _ in played}
    for i, word in enumerate(SCREEN):
        x, y = 30 + (i // 9) * 140, 100 + (i % 9) * 23
        color = dim if word in guessed else green
        svg.text(x, y, f"0x{0xF4A0 + 12 * i:04X}  {word}", 15, fill=color, font_family=MONO)
    x, y = 340, 100
    for guess, left, k in played:
        svg.text(x, y, f">{guess}", 15, fill=green, font_family=MONO)
        svg.text(x + 70, y, f"({left} candidates)", 13, fill=dim, font_family=MONO)
        svg.text(x, y + 23, f">Likeness={k}", 15, fill=green, font_family=MONO)
        y += 52
    svg.text(x, y, f">{PASSWORD}", 15, fill=green, font_family=MONO)
    svg.text(x + 70, y, "(1 candidate)", 13, fill=dim, font_family=MONO)
    svg.text(x, y + 23, ">Exact match", 15, fill=green, font_family=MONO, font_weight="bold")
    return svg.render()


# Guesses per code for two heuristics and the two optimal trees

OPTIMAL = HERE / "results" / "optimal.json"


def optimal_counts() -> list[tuple[str, Counter]]:
    """The two optimal trees of `optimal.py`, played against every code by `optimal.play`."""
    import json

    from mastermind.optimal import Game, play, tree_from_json

    record = json.loads(OPTIMAL.read_text())
    codes = Game.codes_of(record["colors"], record["holes"])
    out = []
    for row in record["trees"]:
        name = "best" if row["limit"] >= 12 else f"best within {row['limit']}"
        out.append((name, Counter(play(tree_from_json(row["tree"], codes), codes))))
    return out


def guesses() -> str:
    arms = [
        ("most parts", Counter(strategies.guess_counts(strategies.most_parts)), "#2a78d6"),
        ("minimax", Counter(strategies.guess_counts(strategies.minimax)), "#d95926"),
    ]
    arms += [
        (name, c, color)
        for (name, c), color in zip(optimal_counts(), ("#1baf7a", "#3d3d3d"), strict=True)
    ]
    left, top, width, height, top_value = 60, 70, 640, 240, 700
    group = width / 6
    bw = (group - 12) / len(arms)

    def sy(v: float) -> float:
        return top + height - v / top_value * height

    svg = Svg(
        left + width + 20,
        top + height + 56,
        "Codes solved in each number of guesses, for two heuristics and two optimal strategies",
        "Grouped bars over 1 to 6 guesses for the 1,296 codes. Most parts solves 635 codes in "
        "four guesses but needs six for 7 codes; minimax never needs a sixth guess. The optimal "
        "tree solves 663 in four and needs six for 2 codes; the optimal tree within five "
        "guesses solves 660 in four and none in six.",
    )
    for v in range(0, top_value + 1, 100):
        svg.add("line", x1=left, y1=sy(v), x2=left + width, y2=sy(v), stroke=GRID)
        svg.text(left - 8, sy(v) + 4, str(v), 11, fill=MUTED, text_anchor="end")
    for g in range(1, 7):
        x0 = left + (g - 1) * group + 6
        for a, (_, counts, color) in enumerate(arms):
            n = counts[g]
            x = x0 + a * bw
            if n:
                svg.add(
                    "rect", x=x + 1, y=sy(n), width=bw - 2, height=top + height - sy(n), fill=color
                )
            svg.text(x + bw / 2, sy(n) - 5, str(n), 10, text_anchor="middle")
        svg.text(x0 + 2 * bw, top + height + 16, str(g), 12, fill=MUTED, text_anchor="middle")
    svg.text(
        left + width / 2,
        top + height + 38,
        "guesses, including the last one",
        12,
        fill=MUTED,
        text_anchor="middle",
    )
    svg.text(10, 20, "codes", 12, fill=MUTED)
    for i, (name, counts, color) in enumerate(arms):
        x, y = left + 20 + (i % 2) * 300, 26 + (i // 2) * 20
        total = sum(k * v for k, v in counts.items())
        svg.add("rect", x=x, y=y, width=12, height=12, fill=color)
        svg.text(x + 18, y + 11, f"{name}: {total:,} guesses in all", 12, fill=MUTED)
    return svg.render()


FIGURES = {"terminal": terminal, "guesses": guesses}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, draw in FIGURES.items():
        (OUT / f"{name}.svg").write_text(draw(), encoding="utf-8")


if __name__ == "__main__":
    main()
