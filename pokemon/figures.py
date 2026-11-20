"""Pictures of the type chart, for pokemon.ipynb and the README.

    uv run --group pokemon python -m pokemon.figures   # writes the README's SVGs

The SVGs are drawn with fixed settings (no date, a fixed hash salt) so that the same code
writes the same bytes; a test checks the committed files.
"""

from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from pokemon.scoring import CHART, SUPER, TYPES, incoming, outgoing

HERE = Path(__file__).resolve().parent
LEVELS = (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(1), Fraction(2), Fraction(4))
COLORS = ("#3b3b3b", "#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#b2182b")
SYMBOL = {
    Fraction(0): "0",
    Fraction(1, 4): "¼",
    Fraction(1, 2): "½",
    Fraction(2): "2",
    Fraction(4): "4",
}
SHORT = {t: t[:3].capitalize() for t in TYPES} | {"fighting": "Fgt", "fairy": "Fai"}


def _grid(ax: Axes, rows: Sequence[Sequence[Fraction]], row_labels, col_labels) -> None:
    """A table of multipliers, one colored cell each, neutral cells left blank."""
    for i, row in enumerate(rows):  # rectangles, not imshow: an SVG then embeds no PNG,
        for j, m in enumerate(row):  # whose compressed bytes vary between zlib builds
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc=COLORS[LEVELS.index(m)], lw=0))
    ax.set_xlim(-0.5, len(col_labels) - 0.5)
    ax.set_ylim(len(row_labels) - 0.5, -0.5)
    ax.set_aspect("equal")
    for i, row in enumerate(rows):
        for j, m in enumerate(row):
            if m != 1:
                ax.text(j, i, SYMBOL[m], ha="center", va="center", fontsize=7,
                        color="white" if m in (0, Fraction(1, 4), 4) else "black")  # fmt: skip
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=7)
    ax.set_xticks([x - 0.5 for x in range(1, len(col_labels))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(row_labels))], minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="both", length=0)
    for side in ax.spines.values():
        side.set_visible(False)


def draw_chart(highlight: Sequence[str] = ()) -> Figure:
    """The 18 by 18 chart: attacking types down the side, defending types across the top.
    Rows in `highlight` are outlined, and their super-effective cells are what a cover needs."""
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    _grid(ax, [[CHART[a][d] for d in TYPES] for a in TYPES], [SHORT[t] for t in TYPES],
          [SHORT[t] for t in TYPES])  # fmt: skip
    ax.set_xlabel("defending type", fontsize=8)
    ax.set_ylabel("attacking type", fontsize=8)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    for t in highlight:
        i = TYPES.index(t)
        ax.add_patch(Rectangle((-0.5, i - 0.5), 18, 1, fill=False, lw=1.4, ec="black"))
    fig.tight_layout()
    return fig


def draw_cover(attackers: Sequence[str]) -> Figure:
    """For each defending type, which of `attackers` hits it super-effectively: the 18
    columns must each have a mark for the set to cover the chart."""
    fig, ax = plt.subplots(figsize=(5.6, 0.26 * len(attackers) + 0.75))
    rows = [[SUPER if CHART[a][d] == SUPER else Fraction(1) for d in TYPES] for a in attackers]
    _grid(ax, rows, [SHORT[t] for t in attackers], [SHORT[t] for t in TYPES])
    fig.tight_layout()
    return fig


def draw_typing(t: Sequence[str]) -> Figure:
    """A typing's two rows: the best it deals to each type (outgoing) and what each type
    deals it (incoming). Attack sums the first row, defense subtracts the sum of the second."""
    fig, ax = plt.subplots(figsize=(5.6, 1.45))
    out, inc = outgoing(t), incoming(t)
    _grid(ax, [[out[d] for d in TYPES], [inc[a] for a in TYPES]],
          [f"deals (sum {float(sum(out.values())):g})", f"takes (sum {float(sum(inc.values())):g})"],
          [SHORT[x] for x in TYPES])  # fmt: skip
    ax.set_title("/".join(x.capitalize() for x in t), fontsize=9)
    fig.tight_layout()
    return fig


def save_svg(fig: Figure, path: Path) -> None:
    """Write an SVG whose bytes depend only on the drawing: no date, a fixed hash salt."""
    with mpl.rc_context({"svg.hashsalt": "pokemon", "svg.fonttype": "path"}):
        fig.savefig(path, format="svg", metadata={"Date": None}, facecolor="white")
    plt.close(fig)


FIGURES = {
    "type-chart.svg": lambda: draw_chart(),
    "steel-fairy.svg": lambda: draw_typing(("fairy", "steel")),
}


def main(folder: Path = HERE / "figures") -> None:
    folder.mkdir(exist_ok=True)
    for name, make in FIGURES.items():
        with plt.style.context("default"):  # unaffected by styles other modules set
            save_svg(make(), folder / name)
        print("wrote", folder / name)


if __name__ == "__main__":
    main()
