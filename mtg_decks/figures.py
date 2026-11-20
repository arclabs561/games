"""Figures for the README, from the committed aggregates and results.

uv run --group mtg_decks python -m mtg_decks.figures
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from maps import sne
from mtg_decks import cards as cardfacts
from mtg_decks import study

HERE = Path(__file__).parent
SET = "BLB"

# a muted color for each Magic color, gray for colorless; gold cards get a dark edge
SWATCH = {
    "W": "#c9b458",
    "U": "#3b75af",
    "B": "#4d4d4d",
    "R": "#c0392b",
    "G": "#3a8a4f",
    "": "#b0b0b0",
}
NAMES = {"W": "white", "U": "blue", "B": "black", "R": "red", "G": "green"}


def results() -> dict:
    return json.loads((study.RESULTS / f"{SET}.json").read_text())


def card_map() -> Figure:
    """t-SNE of the PPMI embedding (cosine distance), each card colored by its color."""
    data = study.load(study.DATA / SET)
    x = study.ppmi_embedding(data.s, data.held, data.n_decks)
    sim = study.embed.cosine(x)
    y = sne.embed(np.clip(2 - 2 * sim, 0, None), perplexity=20.0, seed=0).y
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    for c, (a, b) in zip(data.cards, y, strict=True):
        if len(c.colors) <= 1:
            ax.scatter(a, b, s=16, color=SWATCH[c.colors], lw=0)
    for c, (a, b) in zip(data.cards, y, strict=True):
        if len(c.colors) == 2:
            ax.scatter(a, b, s=30, color=SWATCH[c.colors[0]], edgecolor=SWATCH[c.colors[1]], lw=2)
        elif len(c.colors) > 2:
            ax.scatter(a, b, s=30, color="#e0e0e0", edgecolor="black", lw=1)
    # label each pair beside its most-played gold card
    for g in cardfacts.PAIRS:
        gold = [i for i, c in enumerate(data.cards) if c.colors == g]
        if gold:
            i = max(gold, key=lambda i: (data.cards[i].decks, -i))
            ax.annotate(
                g, y[i], xytext=(6, 6), textcoords="offset points", fontsize=9, weight="bold"
            )
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=SWATCH[k], label=NAMES.get(k, "colorless"))
        for k in "WUBRG"
    ] + [plt.Line2D([], [], marker="o", ls="", color=SWATCH[""], label="colorless")]
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=False, ncol=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    fig.tight_layout()
    return fig


def neighbor_bars() -> Figure:
    """What a card's ten nearest neighbors share with it, by method."""
    q = results()["question1"]
    tests = [
        ("pair", "same color pair"),
        ("color", "share its color"),
        ("tribe_partner", "partner color,\nall neighbors"),
        ("tribe_partner_offcolor", "partner color,\noff-color neighbors"),
    ]
    arms = [
        ("PPMI + eigenvectors", {k: (q["ppmi"][k],) * 3 for k, _ in tests}, "#3b75af"),
        (
            "skip-gram, 5 seeds",
            {k: (q["sgns"][k]["mean"], q["sgns"][k]["min"], q["sgns"][k]["max"]) for k, _ in tests},
            "#3a8a4f",
        ),
        (
            "shuffled decks, 5 seeds",
            {k: (q["null"][k]["mean"], q["null"][k]["min"], q["null"][k]["max"]) for k, _ in tests},
            "#999999",
        ),
        ("random neighbors", {k: (q["chance"][k],) * 3 for k, _ in tests}, "#dddddd"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    width = 0.2
    for a, (label, vals, color) in enumerate(arms):
        xs = np.arange(len(tests)) + (a - 1.5) * width
        mean = np.array([vals[k][0] for k, _ in tests])
        lo = mean - np.array([vals[k][1] for k, _ in tests])
        hi = np.array([vals[k][2] for k, _ in tests]) - mean
        ax.bar(xs, mean, width, color=color, label=label, edgecolor="#555", lw=0.4)
        if label.endswith("seeds"):
            ax.errorbar(xs, mean, yerr=[lo, hi], fmt="none", ecolor="black", lw=0.8, capsize=2)
    ax.set_xticks(np.arange(len(tests)), [t for _, t in tests])
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of the 10 nearest neighbors\n(partner color: share of cards right)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    fig.tight_layout()
    return fig


def swap_plan(pair: str = "WB", n: int = 8) -> Figure:
    """The largest moves of the transport plan from the average losing deck to the winning one."""
    data = study.load(study.DATA / SET)
    x = study.ppmi_embedding(data.s, data.held, data.n_decks)
    size = json.loads((study.DATA / SET / "source.json").read_text())[
        "mean_vocabulary_copies_per_deck"
    ]
    lost, won = data.targets[(pair, "lost")], data.targets[(pair, "won")]
    plan, _ = study.transport.emd(lost, won, study.transport.ground_cost(x))
    sw = study.transport.swaps(plan)[:n]
    cuts = sorted({i for i, _, _ in sw}, key=lambda i: -sum(m for a, _, m in sw if a == i))
    adds = sorted({j for _, j, _ in sw}, key=lambda j: -sum(m for _, b, m in sw if b == j))
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * max(len(cuts), len(adds)) + 0.8))
    yc = {i: -k for k, i in enumerate(cuts)}
    ya = {j: -k for k, j in enumerate(adds)}
    top = max(m for *_, m in sw)
    for i, j, m in sw:
        ax.plot(
            [0, 1],
            [yc[i], ya[j]],
            color="#3b75af",
            lw=0.5 + 6 * m / top,
            alpha=0.7,
            solid_capstyle="butt",
        )
        ax.text(
            0.12,
            yc[i] + 0.12 * (ya[j] - yc[i]) + 0.1,
            f"{100 * m * size:.1f}",
            fontsize=7,
            ha="center",
            color="#333",
        )
    for i in cuts:
        ax.text(-0.02, yc[i], data.names[i], ha="right", va="center", fontsize=8)
    for j in adds:
        ax.text(1.02, ya[j], data.names[j], ha="left", va="center", fontsize=8)
    ax.text(-0.02, 0.8, "cut (losing decks hold more)", ha="right", fontsize=8, weight="bold")
    ax.text(1.02, 0.8, "add (winning decks hold more)", ha="left", fontsize=8, weight="bold")
    ax.text(0.5, 0.8, "copies per 100 decks", ha="center", fontsize=8, color="#333")
    ax.set_xlim(-0.9, 1.9)
    ax.set_ylim(-max(len(cuts), len(adds)) + 0.4, 1.2)
    ax.axis("off")
    fig.tight_layout()
    return fig


def tradeoff() -> Figure:
    """MMR's trade-off: relevance kept against redundancy, mean over the ten pairs."""
    rows = results()["question3"]["mean"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    xs = [r["redundancy"] for r in rows]
    ys = [r["relevance_kept"] for r in rows]
    ax.plot(xs, ys, "-o", color="#3b75af", ms=4)
    for r in rows:
        if r["lambda"] in (1.0, 0.7, 0.5, 0.3, 0.0):
            ax.annotate(
                f"λ = {r['lambda']:g}",
                (r["redundancy"], r["relevance_kept"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
            )
    ax.set_xlabel("redundancy: mean cosine similarity within the list of 5")
    ax.set_ylabel("relevance kept\n(share of the top-5 list's)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


FIGURES = {
    "card_map.svg": card_map,
    "neighbors.svg": neighbor_bars,
    "swaps.svg": swap_plan,
    "tradeoff.svg": tradeoff,
}


def save_svg(fig: Figure, path: Path) -> None:
    """Write an SVG whose bytes depend only on the drawing: no date, a fixed hash salt."""
    with mpl.rc_context({"svg.hashsalt": "mtg_decks", "svg.fonttype": "path"}):
        fig.savefig(path, format="svg", metadata={"Date": None}, facecolor="white")
    plt.close(fig)


def main(folder: Path = HERE / "figures") -> None:
    folder.mkdir(exist_ok=True)
    for name, make in FIGURES.items():
        with plt.style.context("default"):  # unaffected by styles other modules set
            save_svg(make(), folder / name)
        print("wrote", folder / name)


if __name__ == "__main__":
    main()
