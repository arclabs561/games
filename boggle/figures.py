"""Pictures of Boggle boards, for boggle.ipynb and the README.

    uv run --group boggle python -m boggle.figures   # writes the README's SVGs

Each board is drawn as a grid of plain squares with its letters, and a word as a line
through the centers of its squares. The SVGs are written with no date and a fixed hash
salt, so the same code writes the same bytes; a test checks the committed files.
"""

import random
from collections.abc import Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from boggle.board import Board, find_path, parse_board, rand_board

HERE = Path(__file__).resolve().parent


def draw_board(
    board: Board, words: Sequence[str] = (), title: str = "", size: float = 0.55
) -> Figure:
    """The board as a grid of letter squares, with each word in `words` traced through the
    squares of one path that spells it (the first word in the darkest line)."""
    n = len(board)
    width = max(size * n + 0.6, 0.085 * len(title) + 0.3)  # wide enough for the title
    fig, ax = plt.subplots(
        figsize=(width, size * n + (0.8 if title else 0.3) + (0.3 if words else 0))
    )
    for i, row in enumerate(board):
        for j, face in enumerate(row):
            ax.add_patch(
                FancyBboxPatch(
                    (j + 0.06, n - 1 - i + 0.06), 0.88, 0.88, boxstyle="round,pad=0,rounding_size=0.12",
                    fc="#f4f1ea", ec="#55524b", lw=1.0, zorder=1,
                )
            )  # fmt: skip
            ax.text(j + 0.5, n - 0.5 - i, face.capitalize(), ha="center", va="center",
                    fontsize=13 if len(face) == 1 else 10, fontweight="bold", color="#222222", zorder=3)  # fmt: skip
    colors = ("#D55E00", "#0072B2", "#009E73")
    for word, color in zip(words, colors, strict=False):
        path = find_path(board, word)
        assert path is not None, word
        xs, ys = [j + 0.5 for _, j in path], [n - 0.5 - i for i, _ in path]
        ax.plot(xs, ys, color=color, lw=4, alpha=0.55, solid_capstyle="round", label=word, zorder=2)
        ax.plot(xs[:1], ys[:1], "o", color=color, ms=9, alpha=0.55, zorder=2)
    if words:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncols=len(words), frameon=False,
                  fontsize=9, handlelength=1.5)  # fmt: skip
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout(pad=0.2)
    return fig


def save_svg(fig: Figure, path: Path) -> None:
    """Write an SVG whose bytes depend only on the drawing: no date, a fixed hash salt."""
    with mpl.rc_context({"svg.hashsalt": "boggle", "svg.fonttype": "path"}):
        fig.savefig(path, format="svg", metadata={"Date": None}, facecolor="white")
    plt.close(fig)


FIGURES = {
    "random-board.svg": lambda: draw_board(rand_board(random.Random(0)), ["traveler"], "A shake of the cubes (seed 0)"),
    "best-board.svg": lambda: draw_board(
        parse_board("perslatgsineters"), ["replastering"], "The best board for ENABLE2K: 3,625 points"
    ),
    "best-5x5.svg": lambda: draw_board(
        parse_board("ligdrmanesietildsracsepes"), ["interspecies"],
        "Best 5 by 5 board found: 10,397 points (enable1.txt)"
    ),
}  # fmt: skip


def main(folder: Path = HERE / "figures") -> None:
    folder.mkdir(exist_ok=True)
    for name, make in FIGURES.items():
        with plt.style.context("default"):  # unaffected by styles other modules set
            save_svg(make(), folder / name)
        print("wrote", folder / name)


if __name__ == "__main__":
    main()
