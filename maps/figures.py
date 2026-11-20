"""The README's figures, drawn from the committed results.

    uv run python -m maps.figures   # writes maps/figures/*.svg

Every figure reads results/maps.json and results/maps.npz only, so it redraws in a
second; a test checks that the committed SVGs are what this writes.
"""

from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from maps import data, draw, study

plt = draw.plt
OUT = Path(__file__).parent / "figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
INK, MUTED = "#0b0b0b", "#52514e"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cfe0f5", "#2a78d6", "#0b2f5e"])


def hand_shelves() -> Figure:
    """The 12-book shelf sorted by hue two ways, along the Hilbert curve, and shortest."""
    res = study.stored()["hand"]
    hexes = study.hand_hexes()
    labels = {
        "HSV hue": "hue (HSV)",
        "Oklab hue": "hue (Oklab)",
        "Hilbert curve": "Hilbert curve",
        "shortest": "shortest",
    }
    orders = {
        f"{text}  {res[key]['jump']:.2f}": [study.HAND.index(c) for c in res[key]["order"]]
        for key, text in labels.items()
    }
    return draw.shelves(hexes, orders, width=6.5)


def css_shelves() -> Figure:
    """All 139 CSS colors in each method's order, with total jump."""
    res = study.stored()["css"]
    _, hexes = data.css_shelf()
    orders = {f"{name}  {r['jump']:.1f}": r["order"] for name, r in res.items()}
    return draw.shelves(hexes, orders, width=8.0)


def jl_plot() -> Figure:
    """Worst distortion of 400 word vectors against target dimension, with two bounds."""
    res = study.stored()["jl"]["words"]
    ks = np.array(res["k"])
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for color, kind, label in zip(
        SERIES, ("gaussian", "binary", "sparse"), ("Gaussian", "±1", "sparse ±√3, 0"), strict=True
    ):
        ax.plot(ks, res["median_worst"][kind], "o-", color=color, lw=2, ms=4, label=label)
    for key, style, label in (
        ("union_chi2", "--", "exact union bound"),
        ("achlioptas_beta1", ":", "Achlioptas (β = 1)"),
    ):
        ax.plot(ks, res["bounds"][key], style, color=MUTED, lw=1.5, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ticks = [0.2, 0.3, 0.5, 1, 2, 3, 5]
    ax.set_yticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("dimensions k")
    ax.set_ylabel("worst distortion ε (median of 20 maps)")
    ax.grid(True, which="major", color="#e6e6e3", lw=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def scatter_maps(name: str = "pokemon") -> Figure:
    """PCA, SNE and t-SNE maps side by side, colored by stat total (Pokémon)."""
    maps = study.stored_maps()
    _, stats, _ = data.pokemon()
    total = stats.sum(axis=1)
    labels = ["PCA", "SNE", "t-SNE"]
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.9))
    for ax, label in zip(axes, labels, strict=True):
        y = maps[f"{name}/{label}"]
        ax.scatter(y[:, 0], y[:, 1], c=total, cmap=BLUES, s=5, linewidths=0)
        ax.set_title(label, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#cccccc")
    fig.tight_layout()
    return fig


def tradeoff() -> Figure:
    """Trustworthiness against continuity at k = 10 for each map of the Pokémon stats."""
    res = study.stored()
    sne_res = res["sne"]["pokemon"]
    som_res = res["som"]["pokemon"]
    points = {label: (r["T"][1], r["C"][1]) for label, r in sne_res.items()}
    points["SOM"] = (som_res["T"][1], som_res["C"][1])
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    offsets = {"SNE": (-6, 5, "right"), "symmetric SNE": (6, -10, "left")}
    for label, (t, c) in points.items():
        dx, dy, ha = offsets.get(label, (6, 3, "left"))
        ax.plot(t, c, "o", color=SERIES[0], ms=6)
        ax.annotate(
            label, (t, c), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=8, color=INK
        )
    ax.set_xlabel("trustworthiness T(10)")
    ax.set_ylabel("continuity C(10)")
    ax.grid(True, color="#e6e6e3", lw=0.6)
    lo = min(min(p) for p in points.values()) - 0.01
    ax.set_xlim(lo, 1.0)
    ax.set_ylim(lo, 1.0)
    fig.tight_layout()
    return fig


def color_som() -> Figure:
    """Kohonen's color map: 20 x 20 units, each painted with its weight."""
    w = study.stored_maps()["colors/weights"].astype(float).clip(0, 1)
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.imshow(w.reshape(*study.COLOR_SHAPE, 3), interpolation="nearest")
    ax.axis("off")
    fig.tight_layout()
    return fig


FIGURES = {
    "shelf": hand_shelves,
    "css-shelves": css_shelves,
    "jl": jl_plot,
    "sne-pokemon": scatter_maps,
    "tradeoff": tradeoff,
    "som-colors": color_som,
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, make in FIGURES.items():
        draw.save_svg(make(), OUT / f"{name}.svg")
        print(f"wrote maps/figures/{name}.svg")


if __name__ == "__main__":
    main()
