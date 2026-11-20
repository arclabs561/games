"""Pictures of hands and piles, for egyptian_ratscrew.ipynb and the README.

    uv run --group egyptian_ratscrew python -m egyptian_ratscrew.figures   # writes the SVGs

Cards are plain rectangles with their rank; suits never matter in this game, so none are
drawn. The SVGs are written with no date and a fixed hash salt, so the same code writes
the same bytes; a test checks the committed files.
"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from egyptian_ratscrew.game import CHANCES, Card, slappable
from egyptian_ratscrew.termination import beggar_loop_tricks, moves, slap_free_cycle, start

HERE = Path(__file__).resolve().parent
ROYAL, PLAIN, MARK = "#f3e3c3", "#ffffff", "#D55E00"

type Row = tuple[str, Sequence[Card | None], Sequence[int]]  # label, cards (None: a gap), marks


def draw_rows(
    rows: Sequence[Row], title: str = "", headers: Sequence[tuple[int, str]] = ()
) -> Figure:
    """One row of cards per entry, each with a label on the left; None leaves a gap. Royals
    are shaded, and the cards at the marked positions get a colored edge. `headers` puts a
    heading above the given card positions."""
    width = max(len(cards) for _, cards, _ in rows)
    top = len(rows) + (0.6 if headers else 0)
    fig, ax = plt.subplots(figsize=(1.9 + 0.48 * width, 0.62 * top + (0.45 if title else 0.15)))
    for k, text in headers:
        ax.text(k * 0.8, len(rows) + 0.15, text, ha="left", va="center", fontsize=9, style="italic")
    for r, (label, cards, marked) in enumerate(rows):
        y = len(rows) - 1 - r
        ax.text(-0.25, y + 0.38, label, ha="right", va="center", fontsize=9)
        for k, card in enumerate(cards):
            if card is None:
                continue
            edge, lw = (MARK, 2.2) if k in marked else ("#55524b", 0.9)
            ax.add_patch(
                FancyBboxPatch(
                    (k * 0.8, y), 0.62, 0.78, boxstyle="round,pad=0,rounding_size=0.08",
                    fc=ROYAL if card in CHANCES else PLAIN, ec=edge, lw=lw,
                )
            )  # fmt: skip
            ax.text(k * 0.8 + 0.31, y + 0.39, str(card), ha="center", va="center", fontsize=11,
                    fontweight="bold", color="#222222")  # fmt: skip
    ax.set_xlim(-3.3, width * 0.8)
    ax.set_ylim(-0.1, top)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.2)
    return fig


def slap_marks(pile: Sequence[Card]) -> list[int]:
    """The positions of the matching cards if the pile can be slapped: a pair or a sandwich."""
    if not slappable(pile):
        return []
    n = len(pile)
    return [n - 2, n - 1] if pile[-1] == pile[-2] else [n - 3, n - 1]


def draw_piles(piles: Sequence[Sequence[Card]]) -> Figure:
    """Piles, bottom card on the left; the cards that make a pile slappable are marked."""
    rows = [
        (("pair" if p[-1] == p[-2] else "sandwich") if slappable(p) else "no slap", p, slap_marks(p))
        for p in piles
    ]  # fmt: skip
    return draw_rows(rows)


TRAP_DEAL: tuple[tuple[Card, ...], tuple[Card, ...]] = (("J", 2, 3), (2, "J", 3))


def draw_trap() -> Figure:
    """The six-card deal that loops forever with no slappable pile: the hands at the deal and
    after each trick, and the trick just picked up, until the hands repeat."""
    assert slap_free_cycle(start(TRAP_DEAL)) is not None
    rows: list[Row] = []
    state, pile, seen = start(TRAP_DEAL), [], []

    def row(label: str, trick: Sequence[Card]) -> Row:
        h0, h1 = state.hands
        cards = [*h0, *[None] * (5 - len(h0)), *h1, *[None] * (5 - len(h1)), *trick]
        return label, cards, []

    rows.append(row("deal", []))
    while state.hands not in seen:
        seen.append(state.hands)
        while True:  # lay cards until someone picks up the pile
            pile.append(state.hands[state.player][0])
            (_, state), *_ = moves(state)
            if not state.pile:
                break
        rows.append(row(f"player {state.player} takes", pile))
        pile = []
    assert len(beggar_loop_tricks(start(TRAP_DEAL))) == len(rows) - 2  # the first trick leads in
    return draw_rows(
        rows, headers=[(0, "player 0 holds"), (5, "player 1 holds"), (10, "the trick")]
    )


def save_svg(fig: Figure, path: Path) -> None:
    """Write an SVG whose bytes depend only on the drawing: no date, a fixed hash salt."""
    with mpl.rc_context({"svg.hashsalt": "egyptian_ratscrew", "svg.fonttype": "path"}):
        fig.savefig(path, format="svg", metadata={"Date": None}, facecolor="white")
    plt.close(fig)


FIGURES = {
    "slaps.svg": lambda: draw_piles([(5, "K", 7, 7), (9, 7, "K", 7), (9, 7, "K", 8)]),
    "trap.svg": draw_trap,
}


def main(folder: Path = HERE / "figures") -> None:
    folder.mkdir(exist_ok=True)
    for name, make in FIGURES.items():
        save_svg(make(), folder / name)
        print("wrote", folder / name)


if __name__ == "__main__":
    main()
