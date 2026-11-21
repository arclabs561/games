"""The README's figures, drawn from the committed results.

    uv run python -m maps.figures   # writes maps/figures/*.svg

Every figure reads results/maps.json and results/maps.npz only, so it redraws in a
second; a test checks that the committed SVGs are what this writes.
"""

from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from maps import data, draw, ranktsne, study

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
    rows, cols = study.COLOR_SHAPE
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    # Squares, not an embedded PNG, whose bytes depend on the zlib build.
    for i, rgb in enumerate(w):
        r, c = divmod(i, cols)
        ax.add_patch(Rectangle((c, rows - 1 - r), 1, 1, facecolor=rgb, edgecolor="none"))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


SET_TITLES = {"pokemon": "Pokémon stats", "colors": "CSS colors", "words": "Codenames words"}


def ranktsne_scores() -> Figure:
    """T(10) against C(10) for each seed and each affinity, with the PCA start marked."""
    res = study.stored()["ranktsne"]
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.9))
    for ax, name in zip(axes, ranktsne.SETS, strict=True):
        for color, kind in zip(SERIES, ranktsne.AFFINITIES, strict=False):
            rand = res[name]["random"][kind]
            t = [row[1] for row in rand["T"]]
            c = [row[1] for row in rand["C"]]
            label = "Gaussian (perplexity 30)" if kind == "Gaussian" else "kNN (k = 15)"
            ax.plot(
                t, c, "o", mfc="none", mec=color, ms=4, mew=0.9, label=f"{label}, 20 random starts"
            )
            pca = res[name]["pca"][kind]
            ax.plot(
                pca["T"][1],
                pca["C"][1],
                "D",
                color=color,
                ms=6,
                mec=INK,
                mew=0.6,
                label=f"{label}, PCA start",
            )
        ax.set_title(SET_TITLES[name], color=INK)
        ax.set_xlabel("trustworthiness T(10)")
        ax.grid(True, color="#e6e6e3", lw=0.6)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("continuity C(10)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    return fig


def ranktsne_maps() -> Figure:
    """The PCA-start maps of each set under the two affinities."""
    maps = ranktsne.stored_maps()
    layout = study.stored()["ranktsne"]
    _, stats, _ = data.pokemon()
    shade = {
        "pokemon": BLUES((stats.sum(axis=1) - stats.sum(axis=1).min()) / np.ptp(stats.sum(axis=1))),
        "colors": data.css_shelf()[1],
        "words": MUTED,
    }
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.4))
    for col, name in enumerate(ranktsne.SETS):
        for row, kind in enumerate(ranktsne.AFFINITIES):
            ax = axes[row, col]
            y = maps[f"{name}/{kind}"]
            if name == "colors":  # a gray rim, so that white and near-white colors show
                ax.scatter(
                    y[:, 0], y[:, 1], c=shade[name], s=10, edgecolors="#aaaaaa", linewidths=0.3
                )
            else:
                ax.scatter(y[:, 0], y[:, 1], c=shade[name], s=4, linewidths=0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="datalim")
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#cccccc")
            if row == 0:
                ax.set_title(SET_TITLES[name], color=INK)
            else:
                dcor = layout[name]["layout"]["pca"]["dcor"]
                ax.set_xlabel(f"distance correlation {dcor:.3f}", fontsize=8, color=MUTED)
            if col == 0:
                ax.set_ylabel("Gaussian" if kind == "Gaussian" else "kNN, k = 15", color=INK)
    fig.tight_layout()
    return fig


def guesses_plot() -> Figure:
    """Share of clue-board pairs whose pick changes, and worst distortion, against k."""
    res = study.stored()["guesses"]
    ks = np.array(res["k"])
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.5, 5.2), sharex=True)
    names = ("Gaussian", "±1", "sparse ±√3, 0")
    for color, kind, label in zip(SERIES, ("gaussian", "binary", "sparse"), names, strict=True):
        rows = res["kinds"][kind]
        mean = [r["change"]["mean"] for r in rows]
        top.fill_between(
            ks,
            [r["change"]["min"] for r in rows],
            [r["change"]["max"] for r in rows],
            color=color,
            alpha=0.15,
            lw=0,
        )
        top.plot(ks, mean, "o-", color=color, lw=1.8, ms=3.5, label=label)
        bottom.plot(
            ks, [r["median_worst"] for r in rows], "o-", color=color, lw=1.8, ms=3.5, label=label
        )
    pca_k = ks[: len(res["pca"])]
    top.plot(
        pca_k[:-1],
        res["pca"][:-1],
        "s--",
        color=MUTED,
        lw=1.3,
        ms=3.5,
        label="PCA (exact; none change at 256)",
    )
    top.axhline(0.05, color=MUTED, lw=0.8, ls=":")
    top.set_yscale("log")
    yt = [0.05, 0.1, 0.2, 0.5, 1]
    top.set_yticks(yt, [f"{v:.0%}" for v in yt])
    top.set_ylabel("picks changed")
    top.legend(frameon=False, fontsize=7.5, loc="upper right")
    bottom.axhline(0.2, color=MUTED, lw=0.8, ls=":")
    bottom.set_yscale("log")
    bt = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
    bottom.set_yticks(bt, [str(v) for v in bt])
    bottom.set_ylabel("worst distortion ε")
    bottom.set_xscale("log", base=2)
    bottom.set_xticks(ks, [f"{k:,}" for k in ks], fontsize=7)
    bottom.set_xlabel("dimensions k (the vectors have 256)")
    for ax in (top, bottom):
        ax.minorticks_off()
        ax.grid(True, which="major", color="#e6e6e3", lw=0.6)
    fig.tight_layout()
    return fig


def storage_plot() -> Figure:
    """Picks changed against stored bits per word: random projections and quantizers."""
    res = study.stored()
    proj, quant = res["guesses"], res["quantize"]
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    k = np.array(proj["k"])
    ax.plot(
        32 * k,
        [r["change"]["mean"] for r in proj["kinds"]["gaussian"]],
        "o-",
        color=SERIES[0],
        lw=1.8,
        ms=3.5,
        label="Gaussian projection to k dimensions, 32 bits each",
    )
    for i in (0, 4, 8, 12):
        ax.annotate(
            f"k = {k[i]:,}",
            (32 * k[i], proj["kinds"]["gaussian"][i]["change"]["mean"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color=SERIES[0],
        )
    bits = np.array(quant["bits"])
    styles = {
        "RaBitQ": (SERIES[1], "s-"),
        "rotated rounding": (SERIES[2], "^-"),
        "rounding": (MUTED, "D--"),
    }
    labels = {
        "RaBitQ": "RaBitQ (qntz)",
        "rotated rounding": "rounding after a random rotation",
        "rounding": "rounding each dimension",
    }
    for method, (color, style) in styles.items():
        rows = quant["methods"][method]
        y = [r["rate"] if "rate" in r else r["change"]["mean"] for r in rows]
        ax.plot(
            256 * bits,
            y,
            style,
            color=color,
            lw=1.5,
            ms=3.5,
            label=labels[method] + ", b bits per dimension",
        )
    for i, b in enumerate(bits):
        if b in (1, 2, 4, 8):
            y = quant["methods"]["RaBitQ"][i]["change"]["mean"]
            ax.annotate(
                f"b = {b}",
                (256 * b, y),
                xytext=(-6, -11) if b < 8 else (6, -3),
                textcoords="offset points",
                fontsize=7,
                color=SERIES[1],
                ha="right" if b < 8 else "left",
            )
    ax.axvline(256 * 16, color=MUTED, lw=0.8, ls=":")
    ax.text(256 * 16 * 1.08, 0.004, "the committed\nfloat16 vectors", fontsize=7, color=MUTED)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    yt = [0.003, 0.01, 0.03, 0.1, 0.3, 1]
    ax.set_yticks(yt, [f"{v:.1%}" if v < 0.01 else f"{v:.0%}" for v in yt])
    xt = [2**e for e in range(8, 22, 2)]
    ax.set_xticks(xt, [f"{v:,}" for v in xt], fontsize=7)
    ax.minorticks_off()
    ax.set_xlabel("stored bits per word")
    ax.set_ylabel("picks changed")
    ax.grid(True, which="major", color="#e6e6e3", lw=0.6)
    ax.set_ylim(0.003, 1.3)
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.tight_layout()
    return fig


FIGURES = {
    "shelf": hand_shelves,
    "css-shelves": css_shelves,
    "jl": jl_plot,
    "sne-pokemon": scatter_maps,
    "tradeoff": tradeoff,
    "som-colors": color_som,
    "ranktsne": ranktsne_scores,
    "ranktsne-maps": ranktsne_maps,
    "guesses": guesses_plot,
    "storage": storage_plot,
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, make in FIGURES.items():
        with draw.style():
            draw.save_svg(make(), OUT / f"{name}.svg")
        print(f"wrote maps/figures/{name}.svg")


if __name__ == "__main__":
    main()
