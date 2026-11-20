"""The README's figures, drawn as SVG from the committed results, with no third-party packages.

    uv run python -m codenames.figures          # rewrites codenames/figures/*.svg

Each figure is a function that returns the SVG text, so a test can compare it with the
committed file. The board is dealt with a fixed seed from `words.txt`; every other figure
reads the JSON files in `results/`.
"""

import json
import random
from html import escape
from pathlib import Path

from codenames.game import deal
from codenames.luce import ASSASSIN, NEUTRAL, OPPONENT, TEAM
from codenames.words import board_words

HERE = Path(__file__).parent
RESULTS = HERE / "results"
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


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


# The board

KIND_STYLE = {  # fill, text color, label
    TEAM: (BLUE, "#ffffff", "team"),
    OPPONENT: (ORANGE, "#ffffff", "opponent"),
    NEUTRAL: ("#ebe7dc", INK, "neutral"),
    ASSASSIN: ("#2b2b2b", "#ffffff", "assassin"),
}


def board(seed: int = 0) -> str:
    b = deal(board_words(), random.Random(seed))
    w, h, gap, left, top = 148, 58, 6, 10, 10
    svg = Svg(
        5 * (w + gap) + 2 * left - gap,
        5 * (h + gap) + top + 44,
        "A Codenames board dealt from this folder's word list",
        f"25 WordNet nouns from words.txt dealt by game.deal with seed {seed}: 9 team words, "
        "8 opponent words, 7 neutral words and the assassin, each cell labeled with its kind.",
    )
    for i, (word, kind) in enumerate(zip(b.words, b.kinds, strict=True)):
        fill, color, label = KIND_STYLE[kind]
        x, y = left + (i % 5) * (w + gap), top + (i // 5) * (h + gap)
        svg.add("rect", x=x, y=y, width=w, height=h, rx=4, fill=fill)
        svg.text(x + w / 2, y + 33, word, 16, fill=color, text_anchor="middle", font_weight="bold")
        svg.text(x + w - 6, y + h - 6, label, 9, fill=color, text_anchor="end")
    y = top + 5 * (h + gap) + 18
    x = left
    for kind, n in ((TEAM, 9), (OPPONENT, 8), (NEUTRAL, 7), (ASSASSIN, 1)):
        fill, _, label = KIND_STYLE[kind]
        svg.add("rect", x=x, y=y - 11, width=14, height=14, rx=2, fill=fill, stroke=MUTED)
        svg.text(x + 20, y, f"{label} ({n})", 13, fill=MUTED)
        x += 150
    return svg.render()


# The cross-play matrix

NAMES = {
    "potion": "potion",
    "wordllama": "wordllama",
    "granite": "granite",
    "bge": "bge",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3 0.6B",
    "microsoft/harrier-oss-v1-270m": "harrier 270m",
    "microsoft/harrier-oss-v1-0.6b": "harrier 0.6b",
    "google/embeddinggemma-2": "Gemma 2",
    "eat": "EAT counts",
    "ensemble-without-bge": "ensemble without bge",
    "ensemble-without-embeddinggemma-2": "ensemble without Gemma 2",
}
MODELS = list(NAMES)[:8]


def shade(t: float) -> str:
    """A single blue ramp from white (t = 0) to dark blue (t = 1)."""
    lo, hi = (247, 250, 255), (16, 64, 140)
    return "#" + "".join(f"{round(a + (b - a) * t):02x}" for a, b in zip(lo, hi, strict=True))


def crossplay() -> str:
    table = load("crossplay.json")["table"]
    givers = [*MODELS, "ensemble-without-bge", "ensemble-without-embeddinggemma-2"]
    guessers = [*MODELS, "eat"]
    cw, ch, left, top = 62, 26, 190, 92
    svg = Svg(
        left + cw * len(guessers) + 20,
        top + ch * len(givers) + 40,
        "Expected team words found per clue, giver by guesser",
        "A heat map of results/crossplay.json: rows are givers, columns are guessers, and each "
        "cell is the exact expected number of team words found per clue over 200 boards. The "
        "outlined diagonal, where giver and guesser share a model, is the darkest in each row.",
    )
    svg.text(left, 18, "Guesser", 13, fill=MUTED)
    svg.text(10, top - 10, "Giver", 13, fill=MUTED)
    for j, g in enumerate(guessers):
        x, y = left + j * cw + cw / 2, top - 10
        svg.text(
            x, y, NAMES[g], 12, text_anchor="start", transform=f"rotate(-40 {num(x)} {num(y)})"
        )
    for i, giver in enumerate(givers):
        y = top + i * ch
        svg.text(left - 8, y + ch / 2 + 4, NAMES[giver], 12, text_anchor="end")
        for j, g in enumerate(guessers):
            value = table[giver][g]["found"][0]
            t = max(0.0, min(1.0, (value - 0.7) / (2.2 - 0.7)))
            x = left + j * cw
            svg.add("rect", x=x, y=y, width=cw - 2, height=ch - 2, fill=shade(t))
            color = "#ffffff" if t > 0.55 else INK
            svg.text(
                x + (cw - 2) / 2,
                y + ch / 2 + 3,
                f"{value:.2f}",
                12,
                fill=color,
                text_anchor="middle",
            )
            if giver == g:
                svg.add(
                    "rect",
                    x=x,
                    y=y,
                    width=cw - 2,
                    height=ch - 2,
                    fill="none",
                    stroke=INK,
                    stroke_width=2,
                )
    y = top + ch * len(givers) + 22
    svg.text(
        10,
        y,
        "Darker cells find more words. Outlined: giver and guesser share a model.",
        12,
        fill=MUTED,
    )
    return svg.render()


# Population training against self-play (plan C3)


def population() -> str:
    c3 = load("c3.json")
    held = c3["values"]["held_out"]
    rows = [
        (t["giver"], t["held_out"], s["mean"], p["mean"])
        for t, s, p in zip(
            c3["tasks"], held["self"]["per_task"], held["population"]["per_task"], strict=True
        )
    ]
    left, top, width, rh = 250, 50, 420, 24
    x0, x1 = 0.0, 1.0

    def sx(v: float) -> float:
        return left + (v - x0) / (x1 - x0) * width

    svg = Svg(
        left + width + 30,
        top + rh * len(rows) + 60,
        "Held-out guesser's value: population training against self-play",
        "For each of the 14 tasks of plan C3 (a giver and a held-out model), the mean expected "
        "turn value with the held-out guesser for the self-play policy (hollow) and the "
        "population policy (filled), over five seeds and 200 test boards. The filled dot is to "
        "the right in every row.",
    )
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = sx(tick)
        svg.add("line", x1=x, y1=top - 8, x2=x, y2=top + rh * len(rows), stroke=GRID)
        svg.text(x, top + rh * len(rows) + 16, f"{tick:g}", 12, fill=MUTED, text_anchor="middle")
    svg.text(
        left + width / 2,
        top + rh * len(rows) + 36,
        "expected turn value with the held-out guesser (words per turn)",
        12,
        fill=MUTED,
        text_anchor="middle",
    )
    svg.add("circle", cx=left, cy=18, r=5, fill="#ffffff", stroke=MUTED, stroke_width=2)
    svg.text(left + 10, 22, "self-play", 12, fill=MUTED)
    svg.add("circle", cx=left + 100, cy=18, r=5, fill=BLUE)
    svg.text(left + 110, 22, "population of six other models", 12, fill=MUTED)
    for i, (giver, held_out, self_value, pop_value) in enumerate(rows):
        y = top + i * rh + rh / 2
        svg.text(
            left - 10,
            y + 4,
            f"{NAMES[giver]} giver, {NAMES[held_out]} held out",
            12,
            text_anchor="end",
        )
        svg.add(
            "line", x1=sx(self_value), y1=y, x2=sx(pop_value), y2=y, stroke=GRID, stroke_width=3
        )
        svg.add(
            "circle", cx=sx(self_value), cy=y, r=5, fill="#ffffff", stroke=MUTED, stroke_width=2
        )
        svg.add("circle", cx=sx(pop_value), cy=y, r=5, fill=BLUE)
    return svg.render()


# The primary result of every experiment


def findings() -> str:
    c1, c2, c3, c4 = load("c1.json"), load("c2.json"), load("c3.json"), load("c4.json")
    c6, c7, c8, c10 = load("c6.json"), load("c7.json"), load("c8.json"), load("c10.json")

    def mci(d: dict) -> tuple[float, float, float]:
        return d["mean"], d["ci"][0], d["ci"][1]

    guessing = [
        ("C1: linear map on potion, minus untrained potion", mci(c1["primary"])),
        ("C4: gpt-oss-120b, minus EmbeddingGemma 2", mci(c4["guess"]["h_guess"])),
        ("C8: LoRA with teacher's labels, minus without", mci(c8["primary"])),
        ("C2: LoRA on Qwen3 0.6B, minus the potion map", mci(c2["primary"])),
    ]
    giving = [
        ("C3: population training, minus self-play", mci(c3["primary"]["mean_over_tasks"])),
        ("C10: listener added to the population", mci(c10["primary"]["h_add_strangers"])),
        ("C6: language model choosing from 120 actions", mci(c6["primary"]["h_strangers"])),
        ("C7: GRPO-trained model, minus feature policy", mci(c7["difference"]["held_out"])),
        (
            "C10: language model as listener, minus first choice",
            mci(c10["primary"]["h_listen_strangers"]),
        ),
        ("C4: language-model clues, minus embedding giver", mci(c4["give"]["h_give"])),
    ]
    left, width, rh = 330, 360, 24
    svg = Svg(
        left + width + 30,
        2 * 60 + rh * (len(guessing) + len(giving)) + 60,
        "The primary comparison of each experiment, with its 95% interval",
        "Two dot plots. The top one gives differences in the share of held-out human turns where "
        "the guesser picks the person's word; only C1, C4 and C8 lie above zero. The bottom one "
        "gives differences in words per turn with guessers the giver never trained with; only "
        "C3, population training, lies above zero.",
    )

    def panel(top: float, rows, lo: float, hi: float, ticks, title: str, unit: str) -> float:
        def sx(v: float) -> float:
            return left + (v - lo) / (hi - lo) * width

        svg.text(10, top, title, 14, font_weight="bold")
        top += 16
        bottom = top + rh * len(rows)
        for tick in ticks:
            svg.add("line", x1=sx(tick), y1=top, x2=sx(tick), y2=bottom, stroke=GRID)
            svg.text(
                sx(tick),
                bottom + 16,
                (f"{tick:+g}" if tick else "0").replace("-", "\u2212"),
                12,
                fill=MUTED,
                text_anchor="middle",
            )
        svg.add("line", x1=sx(0), y1=top, x2=sx(0), y2=bottom, stroke=MUTED, stroke_width=1.5)
        svg.text(left + width / 2, bottom + 34, unit, 12, fill=MUTED, text_anchor="middle")
        for i, (label, (mean, a, b)) in enumerate(rows):
            y = top + i * rh + rh / 2
            color = BLUE if a > 0 else ORANGE if b < 0 else MUTED
            svg.text(left - 10, y + 4, label, 12, text_anchor="end")
            svg.add("line", x1=sx(a), y1=y, x2=sx(b), y2=y, stroke=color, stroke_width=2)
            svg.add("circle", cx=sx(mean), cy=y, r=5, fill=color)
            svg.text(sx(b) + 6, y + 4, f"{mean:+.3f}".replace("-", "\u2212"), 11, fill=MUTED)
        return bottom + 60

    top = panel(
        24,
        guessing,
        -0.06,
        0.16,
        (-0.05, 0.0, 0.05, 0.1, 0.15),
        "Predicting human guesses",
        "difference in the share of 551 test turns where the guess is the person's word",
    )
    top = panel(
        top,
        giving,
        -0.5,
        0.5,
        (-0.4, -0.2, 0.0, 0.2, 0.4),
        "Giving clues to guessers never trained with",
        "difference in expected words per turn",
    )
    svg.text(
        10, top - 6, "Blue: interval above 0. Orange: below 0. Gray: includes 0.", 12, fill=MUTED
    )
    return svg.render()


FIGURES = {"board": board, "crossplay": crossplay, "population": population, "findings": findings}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, draw in FIGURES.items():
        (OUT / f"{name}.svg").write_text(draw(), encoding="utf-8")


if __name__ == "__main__":
    main()
