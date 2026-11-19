"""Plan P1: does the type chart's equilibrium predict which typings people play?

Reads the usage table that ``pokemon.usage build`` writes, solves the games of
``equilibrium.py`` over the typings each format uses, and compares equilibrium weight with
usage share by Spearman correlation (``experiments/plans/P1.md``).

Primary metric, fixed in the plan: in ``gen9ou`` at cutoff 1500, pooled over 2025-11 to
2026-10, Δρ = ρ(duel equilibrium weight, usage) − ρ(number of Pokémon available, usage) over
typings, with a 95% percentile bootstrap interval over typings (10,000 resamples, seed 0).

Everything else is secondary: the same for ``gen1ou``, for other cutoffs and per month; the
attack game's defender mix; the stats baseline; support mass. Two comparisons are outside the
plan and are labeled exploratory: the log-ratio game over typings, and the games over single
types compared with usage by single type.

    uv run --group pokemon python -m pokemon.p1 --usage pokemon/results/usage-typings.json \\
        --out pokemon/results/p1.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from pokemon import equilibrium as eq
from pokemon.pokemon_types import chart

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESAMPLES = 10_000
SEED = 0
PRIMARY = {"format": "gen9ou", "cutoff": 1500, "game": "duel", "level": "typing"}

Typing = tuple[str, ...]
Matrix = list[list[Fraction]]


# Correlations and the bootstrap.


def spearman_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman's ρ of each row of x with the same row of y; ties get average ranks.

    A row in which either vector is constant has no correlation and gives NaN.
    """
    rx = rankdata(np.atleast_2d(x), axis=1)
    ry = rankdata(np.atleast_2d(y), axis=1)
    cx = rx - rx.mean(axis=1, keepdims=True)
    cy = ry - ry.mean(axis=1, keepdims=True)
    den = np.sqrt((cx * cx).sum(axis=1) * (cy * cy).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (cx * cy).sum(axis=1) / np.where(den > 0, den, 1), np.nan)


def spearman(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    """Spearman's ρ of two vectors."""
    return float(spearman_rows(np.asarray(x, float), np.asarray(y, float))[0])


def delta_rows(idx: np.ndarray, model, base, usage) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each row of indices: ρ(model, usage), ρ(base, usage), and their difference."""
    m, b, u = (np.asarray(v, float)[idx] for v in (model, base, usage))
    rm, rb = spearman_rows(m, u), spearman_rows(b, u)
    return rm, rb, rm - rb


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    """The 2.5th and 97.5th percentiles of the defined values."""
    v = values[~np.isnan(values)]
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap(model, base, usage, resamples: int = RESAMPLES, seed: int = SEED) -> dict:
    """Δρ = ρ(model, usage) − ρ(base, usage), with percentile intervals over typings.

    Each resample draws n typings with replacement, keeping each typing's three values
    together. Resamples in which a correlation is undefined (a constant vector) are dropped
    and counted.
    """
    n = len(usage)
    idx = np.random.default_rng(seed).integers(0, n, size=(resamples, n))
    rm, rb, d = delta_rows(idx, model, base, usage)
    point = delta_rows(np.arange(n)[None, :], model, base, usage)
    return {
        "n": n,
        "rho_model": float(point[0][0]),
        "rho_base": float(point[1][0]),
        "delta": float(point[2][0]),
        "rho_model_interval": percentile_interval(rm),
        "rho_base_interval": percentile_interval(rb),
        "delta_interval": percentile_interval(d),
        "undefined_resamples": int(np.isnan(d).sum()),
        "resamples": resamples,
        "seed": seed,
    }


# The games over the typings a format uses.


def games(players: Sequence[Typing], c: dict) -> dict[str, tuple[Matrix, str, bool]]:
    """Name -> (matrix, which player's mix models the user, whether it is in the plan).

    In the attack game the rows are the chart's attacking types and the columns are the
    format's typings, so the defender's (column) mix is the one compared with usage.
    """
    return {
        "duel": (eq.duel_matrix(players, c), "row", True),
        "attack": (eq.attack_matrix(players, c), "column", True),
        "log-ratio k=1": (
            eq.symmetric_matrix(lambda a, b: eq.log_ratio(a, b, -1), players, c),
            "row",
            False,
        ),
        "log-ratio k=2": (
            eq.symmetric_matrix(lambda a, b: eq.log_ratio(a, b, -2), players, c),
            "row",
            False,
        ),
    }


def solve(g: Matrix, side: str) -> dict:
    """The side's equilibrium weights, exactly, with ranges if they are not unique.

    If the optimal strategy is unique, its weights are used. If not, each weight's least and
    greatest optimal value are found exactly (``weight_ranges``) and the midpoint is used, as
    the plan says; the two ends give a sensitivity check.
    """
    x, y, v = eq.exact_equilibrium(g)
    h, w, p = (g, v, x) if side == "row" else (eq.transpose_negated(g), -v, y)
    if eq.has_unique_certificate(g, x, y, v):
        unique = True
        ranges = [(q, q) for q in p]
    else:
        ranges = eq.weight_ranges(h, w)
        unique = all(lo == hi for lo, hi in ranges)
    mid = [(lo + hi) / 2 for lo, hi in ranges]
    return {
        "value": v,
        "unique": unique,
        "weights": mid,
        "low": [lo for lo, _ in ranges],
        "high": [hi for _, hi in ranges],
        "matrix": h,
        "mix": p,
    }


def support_mass(weights: Sequence[Fraction], usage: Sequence[float], count: Sequence[int]):
    """Share of usage on the support, and share of available Pokémon whose typing is in it."""
    on = [w > 0 for w in weights]
    use = sum(u for u, s in zip(usage, on, strict=True) if s) / sum(usage)
    avail = sum(k for k, s in zip(count, on, strict=True) if s) / sum(count)
    return float(use), float(avail)


def dropped(names: Sequence[str], solved: dict, usage: Sequence[float], k: int = 10) -> list[dict]:
    """The k most used typings with zero weight, and what beats each in the duel.

    For a symmetric game h, typing j earns -Σᵢ xᵢ h[i][j] against the mix x; ``beaten_by``
    lists the three support typings i with the largest xᵢ h[i][j].
    """
    h, x = solved["matrix"], solved["mix"]
    zero = [j for j, w in enumerate(solved["weights"]) if w == 0]
    zero.sort(key=lambda j: -usage[j])
    out = []
    for j in zero[:k]:
        against = eq.payoffs_against_columns(h, x)[j]
        parts = sorted(
            ((x[i] * h[i][j], i) for i in range(len(x)) if x[i] > 0), key=lambda t: -t[0]
        )
        out.append(
            {
                "typing": names[j],
                "usage": usage[j],
                "payoff_against_mix": float(-against),
                "beaten_by": [
                    {"typing": names[i], "weight": float(x[i]), "contribution": float(c)}
                    for c, i in parts[:3]
                    if c > 0
                ],
            }
        )
    return out


def rows_for(table: dict, kind: str, fmt: str, cutoff: int, month: str) -> list[dict]:
    return [
        r
        for r in table[kind]
        if r["format"] == fmt and r["cutoff"] == cutoff and r["month"] == month
    ]


def ordered(c: dict, rows: list[dict], level: str) -> tuple[list[Typing], list[dict]]:
    """The format's typings (or types) in the chart's order, with their rows."""
    key = "typing" if level == "typing" else "type"
    by = {r[key]: r for r in rows}
    universe = eq.typings(c) if level == "typing" else eq.singles(c)
    names = [t for t in universe if "/".join(t) in by]
    assert len(names) == len(by), sorted(set(by) - {"/".join(t) for t in names})
    return names, [by["/".join(t)] for t in names]


def analyze(
    table: dict,
    fmt: str,
    cutoff: int,
    level: str,
    resamples: int = RESAMPLES,
    only: Callable[[str], bool] = lambda _: True,
) -> list[dict]:
    """Every game against usage for one format, cutoff and level (typing or type)."""
    g_number = next(r["generation"] for r in table["typings"] if r["format"] == fmt)
    c = chart(g_number)
    kind = "typings" if level == "typing" else "types"
    players, rows = ordered(c, rows_for(table, kind, fmt, cutoff, "pooled"), level)
    names = ["/".join(t) for t in players]
    usage = [r["share"] for r in rows]
    count = [r["pokemon"] for r in rows]
    stats = [r["max_total"] for r in rows]
    months = sorted({r["month"] for r in table[kind] if r["format"] == fmt} - {"pooled"})
    out = []
    for game, (g, side, planned) in games(players, c).items():
        if not only(game):
            continue
        s = solve(g, side)
        w = [float(q) for q in s["weights"]]
        result = {
            "format": fmt,
            "cutoff": cutoff,
            "level": level,
            "game": game,
            "in_plan": planned and level == "typing",
            "primary": {"format": fmt, "cutoff": cutoff, "game": game, "level": level} == PRIMARY,
            "unique": s["unique"],
            "value": str(s["value"]),
            "support": [
                {"typing": n, "weight": str(q), "float": float(q)}
                for n, q in zip(names, s["weights"], strict=True)
                if q > 0
            ],
            "vs_availability": bootstrap(w, count, usage, resamples),
            "vs_stats": bootstrap(w, stats, usage, resamples),
            "support_mass": dict(
                zip(("usage", "available"), support_mass(s["weights"], usage, count), strict=True)
            ),
        }
        if not s["unique"]:
            result["delta_at_range_ends"] = {
                end: spearman([float(q) for q in s[end]], usage) - spearman(count, usage)
                for end in ("low", "high")
            }
        per_month = []
        for m in months:
            mrows = {r[level]: r for r in rows_for(table, kind, fmt, cutoff, m)}
            mu = [mrows[n]["share"] if n in mrows else 0.0 for n in names]
            per_month.append(
                {"month": m, "delta": spearman(w, mu) - spearman(count, mu), "rho": spearman(w, mu)}
            )
        result["per_month"] = per_month
        if game == "duel" and level == "typing":
            result["most_used_dropped"] = dropped(names, s, usage)
        out.append(result)
    return out


def run(table: dict, resamples: int = RESAMPLES) -> dict:
    """Every comparison the plan asks for, and the exploratory ones."""
    results = []
    cutoffs = sorted({(r["format"], r["cutoff"]) for r in table["typings"]})
    for fmt, cutoff in cutoffs:
        for level in ("typing", "type"):
            results += analyze(table, fmt, cutoff, level, resamples)
    primary = next(r for r in results if r["primary"])
    lo, hi = primary["vs_availability"]["delta_interval"]
    return {
        "plan": "experiments/plans/P1.md",
        "primary": {
            "delta": primary["vs_availability"]["delta"],
            "interval": [lo, hi],
            "predicts_beyond_availability": lo > 0,
        },
        "files": [
            {k: f[k] for k in ("format", "month", "cutoff", "sha256")} for f in table["files"]
        ],
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--usage", type=Path, default=RESULTS / "usage-typings.json")
    parser.add_argument("--out", type=Path, default=RESULTS / "p1.json")
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    args = parser.parse_args()
    table = json.loads(args.usage.read_text(encoding="utf-8"))
    out = run(table, args.resamples)
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    p = out["primary"]
    print(
        f"primary: Δρ = {p['delta']:.3f}, 95% interval [{p['interval'][0]:.3f}, {p['interval'][1]:.3f}]"
    )
    for r in out["results"]:
        a, s = r["vs_availability"], r["vs_stats"]
        print(
            f"{r['format']} {r['cutoff']:>4} {r['level']:6} {r['game']:14} n={a['n']:3} "
            f"unique={r['unique']!s:5} ρ(eq)={a['rho_model']:+.2f} ρ(avail)={a['rho_base']:+.2f} "
            f"Δ={a['delta']:+.2f} [{a['delta_interval'][0]:+.2f}, {a['delta_interval'][1]:+.2f}] "
            f"ρ(stats)={s['rho_base']:+.2f} mass {r['support_mass']['usage']:.2f}"
            f"/{r['support_mass']['available']:.2f}"
        )


if __name__ == "__main__":
    main()
