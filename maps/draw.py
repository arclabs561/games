"""Pictures: shelves of book spines, and plots shared by the notebooks and the README figures.

Figures are drawn and saved inside style(), which starts from matplotlib's defaults, so
that settings made by other modules do not reach them and these settings do not leak out.
The notebooks apply RC themselves.
"""

from contextlib import contextmanager
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.typing import RcKeyType

RC: dict[RcKeyType, Any] = {
    "svg.hashsalt": "maps",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


@contextmanager
def style():
    """matplotlib's default style with RC on top, for the duration of a block."""
    with plt.style.context("default"), mpl.rc_context(RC):
        yield


def spine_sizes(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Widths and heights of n books, fixed by the seed so that every shelf of a figure matches."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.7, 1.3, n), rng.uniform(0.72, 1.0, n)


def draw_shelf(ax, hexes: list[str], order: list[int], label: str = "", seed: int = 0) -> None:
    """Books standing in the given order: spine i has the color hexes[i]."""
    widths, heights = spine_sizes(len(hexes), seed)
    x = 0.0
    for i in order:
        ax.add_patch(
            Rectangle((x, 0), widths[i], heights[i], facecolor=hexes[i], edgecolor="#555", lw=0.3)
        )
        x += widths[i]
    ax.plot([0, x], [0, 0], color="#333", lw=1.5)
    ax.set_xlim(-0.3, x + 0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
    if label:
        ax.text(-0.5, 0.5, label, ha="right", va="center", fontsize=9)


def shelves(hexes: list[str], orders: dict[str, list[int]], width: float = 8.0) -> Figure:
    """One shelf per ordering, stacked, labeled on the left."""
    fig, axes = plt.subplots(len(orders), 1, figsize=(width, 0.55 * len(orders) + 0.2))
    axes = np.atleast_1d(axes)
    for ax, (label, order) in zip(axes, orders.items(), strict=True):
        draw_shelf(ax, hexes, order, label)
    fig.subplots_adjust(left=1.5 / width, right=0.99, top=0.98, bottom=0.02, hspace=0.25)
    return fig


def save_svg(fig: Figure, path) -> None:
    """Write an SVG with no date, so that the same figure gives the same bytes."""
    fig.savefig(path, format="svg", metadata={"Date": None}, facecolor="white")
    plt.close(fig)
