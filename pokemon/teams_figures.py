"""Pictures for the team study, drawn from ``results/teams.json``.

    uv run --group pokemon python -m pokemon.teams_figures   # writes figures/teams-*.svg

- ``teams-cards.svg``: the heaviest team of the full-battle equilibrium as six cards, colored
  by type, with each set's ability and moves.
- ``teams-graph.svg``: the teams of the full-battle equilibrium, with an arrow from each team
  to every team it beats on average.
- ``teams-ranks.svg``: each Pokémon's rank by its ranking score at every level, next to its
  rank by use on the ladder.

The SVGs use ``figures.save_svg``, so the same data gives the same bytes.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from pokemon.figures import save_svg

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "teams.json"

# A plain color for each type; any distinct colors would do.
TYPE_COLOR = {
    "normal": "#a8a77a", "fire": "#e8743b", "water": "#5b8fd6", "electric": "#e3c13b",
    "grass": "#6fb04e", "ice": "#8fd1cf", "fighting": "#b0413e", "poison": "#9450a0",
    "ground": "#c9a45a", "flying": "#9c8fe0", "psychic": "#e0608a", "bug": "#9aaa2a",
    "rock": "#a8923c", "ghost": "#6b5a92", "dragon": "#6a4ee0", "dark": "#6b5a4e",
    "steel": "#9fa0b8", "fairy": "#d98fbf",
}  # fmt: skip
LEVEL_NAMES = {"1": "types", "2": "moves", "3": "abilities", "4": "stats", "5": "simulator"}


def title(identifier: str) -> str:
    """A Pokémon's name as Showdown writes it ('ting-lu' -> 'Ting-Lu'); for a move or an
    ability, the words capitalized ('close-combat' -> 'Close Combat')."""
    from pokemon import teams_model

    names = {p.identifier: p.showdown for p in teams_model.all_pokemon()}
    return names.get(identifier) or " ".join(w.capitalize() for w in identifier.split("-"))


def load() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def text_color(hex_color: str) -> str:
    r, g, b = (int(hex_color[k : k + 2], 16) for k in (1, 3, 5))
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 150 else "white"


def draw_cards(data: dict, team: list[str], caption: str) -> Figure:
    """Six cards: the name on a band of its type colors, then ability and four moves."""
    from pokemon import teams_model

    sets = {s["pokemon"]: s for s in data["sets"]}
    moves = teams_model.all_moves()
    typing = {p.identifier: p.typing for p in teams_model.all_pokemon()}
    fig, ax = plt.subplots(figsize=(8, 4.1))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2.05)
    ax.axis("off")
    for k, name in enumerate(team):
        x, y = k % 3, 1 - k // 3
        ax.add_patch(FancyBboxPatch((x + 0.04, y + 0.04), 0.92, 0.92, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc="white", ec="#555555", lw=0.8))  # fmt: skip
        types = typing[name]
        w = 0.92 / len(types)
        for j, t in enumerate(types):
            ax.add_patch(Rectangle((x + 0.04 + j * w, y + 0.74), w, 0.22, fc=TYPE_COLOR[t], lw=0))
            ax.text(x + 0.04 + (j + 0.5) * w, y + 0.765, t, ha="center", va="bottom", fontsize=6.5,
                    color=text_color(TYPE_COLOR[t]))  # fmt: skip
        ax.text(x + 0.5, y + 0.885, title(name), ha="center", va="center", fontsize=10,
                weight="bold", color=text_color(TYPE_COLOR[types[0]]))  # fmt: skip
        ax.text(x + 0.5, y + 0.66, title(sets[name]["ability"]), ha="center", va="center",
                fontsize=7.5, style="italic", color="#333333")  # fmt: skip
        for j, m in enumerate(sets[name]["moves"]):
            mt = moves[m].type
            yy = y + 0.5 - j * 0.12
            ax.add_patch(Rectangle((x + 0.12, yy - 0.045), 0.76, 0.09, fc=TYPE_COLOR[mt], lw=0))
            ax.text(x + 0.5, yy, f"{title(m)}  ({moves[m].power})", ha="center", va="center",
                    fontsize=7, color=text_color(TYPE_COLOR[mt]))  # fmt: skip
    ax.text(1.5, 2.03, caption, ha="center", va="top", fontsize=8.5)
    fig.tight_layout(pad=0.2)
    return fig


def draw_graph(data: dict, shown: int = 7) -> Figure:
    """The heaviest teams of the full-battle equilibrium on a circle; an arrow from T to U
    when T beats U on average, as wide as the margin."""
    b = data["common"]["battles"]
    teams = data["common"]["teams"]
    support = sorted(b["equilibrium"], key=lambda i: (-Fraction(b["equilibrium"][i]), int(i)))[
        :shown
    ]
    k = len(support)
    pos = {i: (math.cos(math.pi / 2 - 2 * math.pi * n / k), math.sin(math.pi / 2 - 2 * math.pi * n / k))
           for n, i in enumerate(support)}  # fmt: skip
    m = data["common"]["battles_matrix"]
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.set_xlim(-1.3, 3.4)
    ax.set_ylim(-1.15, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    for i in support:
        for j in support:
            v = Fraction(m[int(i)][int(j)])
            if v > 0:
                ax.add_patch(FancyArrowPatch(pos[i], pos[j], arrowstyle="-|>", mutation_scale=9,
                                             lw=0.3 + 4 * float(v), color="#777777",
                                             shrinkA=18, shrinkB=18, alpha=0.8,
                                             connectionstyle="arc3,rad=0.08"))  # fmt: skip
    for n, i in enumerate(support):
        x, y = pos[i]
        w = float(Fraction(b["equilibrium"][i]))
        ax.add_patch(
            plt.Circle((x, y), 0.07 + 0.25 * math.sqrt(w), fc="#4a7ab8", ec="white", zorder=3)
        )
        ax.text(x, y, f"T{n + 1}", ha="center", va="center", color="white", fontsize=8, zorder=4)
        members = [title(p) for p in teams[int(i)]]
        ax.text(1.4, 1.15 - n * 0.3, f"T{n + 1}  {w:.0%}  " + ", ".join(members[:3]) + ",\n"
                + "            " + ", ".join(members[3:]), fontsize=6.5, va="center")  # fmt: skip
    ax.text(1.4, 1.15 - k * 0.3, f"the {k} heaviest of {len(b['equilibrium'])} teams in the equilibrium\n"
            "circle area: weight; arrow T to U: T wins more often, wider for a larger margin",
            fontsize=6.5, va="center", color="#555555")  # fmt: skip
    fig.tight_layout(pad=0.2)
    return fig


def draw_ranks(data: dict, highlight: int = 8) -> Figure:
    """Rank by ranking score at each level, and rank by use; the top of level 5 in color."""
    pool = data["pool"]
    levels = sorted(data["levels"])
    ranks = {
        lv: {p: r + 1 for r, p in enumerate(data["levels"][lv]["unit_ranking"])} for lv in levels
    }
    use = {
        p: r + 1 for r, p in enumerate(sorted(pool, key=lambda p: -data["usage"][pool.index(p)]))
    }
    cols = [*levels, "use"]
    fig, ax = plt.subplots(figsize=(8, 5.4))
    top = data["levels"][levels[-1]]["unit_ranking"][:highlight]
    cmap = plt.get_cmap("tab10")
    for p in pool:
        ys = [ranks[lv][p] for lv in levels] + [use[p]]
        if p in top:
            c = cmap(top.index(p))
            ax.plot(range(len(cols)), ys, color=c, lw=2, marker="o", ms=4, zorder=3)
            ax.text(len(cols) - 0.9, use[p], title(p), color=c, fontsize=7, va="center")
            ax.text(-0.1, ys[0], title(p), color=c, fontsize=7, va="center", ha="right")
        else:
            ax.plot(range(len(cols)), ys, color="#cccccc", lw=0.7, zorder=1)
    ax.set_xticks(range(len(cols)), [f"{c}: {LEVEL_NAMES[c]}" if c in LEVEL_NAMES else "ladder use" for c in cols],
                  fontsize=8)  # fmt: skip
    ax.set_ylim(len(pool) + 0.5, 0.5)
    ax.set_xlim(-1.6, len(cols) + 0.4)
    ax.set_ylabel("rank (1 = highest ranking score)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def figures(data: dict) -> dict:
    out = {"teams-ranks.svg": draw_ranks(data)}
    if "battles" in data["common"]:
        b = data["common"]["battles"]
        top = max(b["equilibrium"], key=lambda i: Fraction(b["equilibrium"][i]))
        w = float(Fraction(b["equilibrium"][top]))
        out["teams-cards.svg"] = draw_cards(
            data,
            data["common"]["teams"][int(top)],
            f"The heaviest team ({w:.0%}) of the equilibrium of full simulated battles: "
            "fixed sets, no items",
        )
        out["teams-graph.svg"] = draw_graph(data)
    return out


def main(folder: Path = HERE / "figures") -> None:
    folder.mkdir(exist_ok=True)
    for name, fig in figures(load()).items():
        save_svg(fig, folder / name)
        print("wrote", folder / name)


if __name__ == "__main__":
    main()
