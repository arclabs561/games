"""Every number in the README, section by section.

    uv run python -m yahtzee.study optimal     # value, checks, distribution, per box; 6 minutes
    uv run python -m yahtzee.study rules       # what each rule is worth; 6 minutes
    uv run python -m yahtzee.study openings    # writes openings.csv; 1 minute
    uv run python -m yahtzee.study greedy      # the greedy strategy, exactly; 5 minutes
    uv run python -m yahtzee.study simulate    # seeded games of both strategies; 3 minutes

Times are for one core. `optimal` also writes distribution.csv.
"""

import csv
import sys
import time
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from yahtzee import policy
from yahtzee.rules import CATEGORIES, OFFICIAL, Rules
from yahtzee.simulate import deal, interval, play
from yahtzee.solver import reachable_count, solve, turn

HERE = Path(__file__).parent
# Kelly and Liese's exact value, as Verhoeff's trivia page gives it.
NUMERATOR = 4837423778178559905031968925883862839950072036299117171073109676394093468257472452669937843237871315151422338846646267957265652473530478355
DENOMINATOR = 19000886425012870870451396938693991151248393833957109655563098372206519682663244561350767261332742248200665160873051839214724243263586304
GREEDY = lambda s: np.zeros(len(s))  # noqa: E731


def show(roll) -> str:
    """A roll or keep as digits, or a dash for nothing."""
    return "".join(map(str, roll)) or "-"


def expected(g: policy.Graph) -> float:
    """The expected final score from a strategy graph's statistics."""
    s = g.stats
    return float(
        s["points"].sum() + s["bonus"][0] + g.tables.rules.upper_bonus * s["upper_bonus"][0]
    )


def report(dist: np.ndarray) -> None:
    """Print a score distribution's summary and quantiles."""
    s = policy.summary(dist)
    mode, least = int(s["mode"]), int(s["min"])
    cdf = np.cumsum(dist)
    print(f"  mean {s['mean']:.10f}  sd {s['sd']:.6f}  median {s['median']}  mode {s['mode']}")
    print(f"  P(mode) {dist[mode]:.5f}  min {least}  max {s['max']}  P(min) {dist[least]:.3e}")
    print(f"  P(score >= 248) {1 - cdf[247]:.5f}  P(score <= 248) {cdf[248]:.5f}")
    for q in (0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.99):
        print(f"  quantile {q:.2f}: {int(np.searchsorted(cdf, q))}")
    for x in (x for x in (200, 300, 400, 500, 1000) if x <= len(cdf)):
        print(f"  P(score >= {x}) {1 - cdf[x - 1]:.5f}")


def optimal() -> None:
    start = time.time()
    t, values = solve(OFFICIAL)
    value = values[t.index(0, 0, 0)]
    getcontext().prec = 30
    exact = Decimal(NUMERATOR) / Decimal(DENOMINATOR)
    print(f"optimal expected score {value:.10f} ({time.time() - start:.0f} s)")
    print(f"  Verhoeff 254.5896, Glenn 254.59; Kelly and Liese's fraction is {exact:.10f}")
    print(f"  difference from the fraction {value - float(exact):.6f}")
    print(f"  reachable states {reachable_count(OFFICIAL):,d} (Verhoeff: 536,448)")
    tr = turn(t, values, np.array([t.index(0, 0, 0)]), choices=True)
    k0, k1 = tr.keep_values[0][0], tr.keep_values[1][0]
    k, b = t.keep_index, t.bit
    print("Verhoeff's examples, expected final score:")
    print(
        f"  roll 11666: keep 666 {k0[k[(6, 6, 6)]]:.2f} (265.12); full house now {25 + values[t.index(b['full_house'], 0, 0)]:.2f} (253.91)"
    )
    for keep, pub in (
        ((3, 4), 245.17),
        ((1, 1), 245.14),
        ((4,), 244.96),
        ((3,), 244.74),
        ((), 244.55),
        ((6,), 244.52),
    ):
        print(f"  second roll 11346, keep {show(keep)}: {k1[k[keep]]:.2f} ({pub})")
    print(
        f"  final roll 66661: sixes {24 + values[t.index(b['sixes'], 24, 0)]:.2f} (268.23), four of a kind {25 + values[t.index(b['four_kind'], 0, 0)]:.2f} (260.54)"
    )
    v0, v2 = tr.levels[0][0], tr.levels[-1][0]
    worst, last = v0.argmin(), v2.argmin()
    print(f"  Yahtzee in the first turn {50 + values[t.index(b['yahtzee'], 0, 1)]:.2f} (320.84)")
    print(
        f"  worst first roll {show(t.rolls[worst])}, keep {show(t.keeps[tr.keeps[0][0, worst]])}: {v0[worst]:.2f} (11236, keep 6: 249.83)"
    )
    print(
        f"  worst final roll {show(t.rolls[last])} in {CATEGORIES[tr.box[0, last]]}: {v2[last]:.2f} (23446, chance: 238.96)"
    )
    g = policy.graph(t, values)
    print(
        f"strategy graph: {sum(len(layer.states) for layer in g.layers):,d} states reached; mean {expected(g):.10f}"
    )
    s = g.stats
    print(f"  {'box':15s} {'mean':>6s} {'sd':>6s} {'% zero':>7s}")
    for c, p, q, z in zip(CATEGORIES, s["points"], s["squares"], s["zeros"], strict=True):
        print(f"  {c:15s} {p:6.2f} {np.sqrt(q - p * p):6.2f} {100 * z:7.2f}")
    ub = s["upper_bonus"][0]
    print(f"  upper bonus: mean {35 * ub:.2f}, earned in {100 * ub:.2f}% of games")
    print(f"  Yahtzee bonus: mean {s['bonus'][0]:.2f}; bonuses per game {s['bonuses'][0]:.4f}")
    print(f"  Yahtzees rolled per game {s['yahtzees'][0]:.4f}; Jokers {s['jokers'][0]:.4f}")
    dist = policy.distribution(g)
    print("final score:")
    report(dist)
    with open(HERE / "distribution.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["score", "probability"])
        w.writerows([x, repr(float(p))] for x, p in enumerate(dist) if p > 0)
    for count in ("yahtzees", "bonuses"):
        d = policy.distribution(g, count)
        print(f"{count} per game: " + ", ".join(f"P({i}) {p:.5f}" for i, p in enumerate(d[:5])))
    print(f"({time.time() - start:.0f} s)")


VARIANTS: list[tuple[str, dict[str, Any]]] = [
    ("official rules", {}),
    ("no Joker", {"joker": "none"}),
    ("no Yahtzee bonus", {"yahtzee_bonus": 0}),
    ("neither", {"yahtzee_bonus": 0, "joker": "none"}),
    ("no upper bonus", {"upper_bonus": 0}),
    ("no upper bonus, no Joker", {"upper_bonus": 0, "joker": "none"}),
    ("no upper bonus, no Yahtzee bonus", {"upper_bonus": 0, "yahtzee_bonus": 0}),
    ("no bonuses, no Joker", {"upper_bonus": 0, "yahtzee_bonus": 0, "joker": "none"}),
    ("forced Joker", {"joker": "forced"}),
]


def rules() -> None:
    for name, kw in VARIANTS:
        start = time.time()
        t, values = solve(Rules(**kw))
        print(
            f"{name:34s} {values[t.index(0, 0, 0)]:.10f} ({time.time() - start:.0f} s)", flush=True
        )
        if name == "neither":  # Verhoeff: 245.87, standard deviation 39.82
            g = policy.graph(t, values)
            report(policy.distribution(g))


class Opening(NamedTuple):
    """The best and second-best keep after one first roll of the game."""

    roll: str
    ways: int  # ordered rolls, of 7,776, that show these dice
    keep: str
    value: float
    next_keep: str
    next_value: float


def openings() -> None:
    t, values = solve(OFFICIAL)
    tr = turn(t, values, np.array([t.index(0, 0, 0)]), choices=True)
    k0 = tr.keep_values[0][0]
    rows = []
    for i, roll in enumerate(t.rolls):
        options = sorted({int(j) for j in t.subkeeps[i]}, key=lambda j: (-k0[j], -len(t.keeps[j])))
        best, second = options[0], options[1]
        assert best == tr.keeps[0][0, i]
        ways = round(t.first[i] * 6**5)
        rows.append(
            Opening(
                show(roll),
                ways,
                show(t.keeps[best]),
                float(k0[best]),
                show(t.keeps[second]),
                float(k0[second]),
            )
        )
    with open(HERE / "openings.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["roll", "ways", "keep", "expected", "next_keep", "next_expected", "difference"])
        for r in rows:
            w.writerow(
                [
                    r.roll,
                    r.ways,
                    r.keep,
                    f"{r.value:.4f}",
                    r.next_keep,
                    f"{r.next_value:.4f}",
                    f"{r.value - r.next_value:.4f}",
                ]
            )
    print(f"start of game {float(t.first @ tr.levels[0][0]):.10f}")
    whole = [r for r in rows if r.roll == r.keep]
    print(f"keep all five: {sum(r.ways for r in whole)} of 7776 ways: {[r.roll for r in whole]}")
    print(f"reroll all five: {sum(r.ways for r in rows if r.keep == '-')} ways")
    common, single, others = 0, 0, []
    for r in rows:
        top = max(r.roll.count(v) for v in set(r.roll))
        if top == 1:
            single += r.ways
        elif any(r.keep == v * top for v in set(r.roll) if r.roll.count(v) == top):
            common += r.ways
        elif r.roll != r.keep:
            others.append(r)
    print(f"no two dice alike: {single} ways; keep exactly a most common face: {common} ways")
    print(f"a pair or better, but keep something else ({sum(r.ways for r in others)} ways):")
    print("  " + ", ".join(f"{r.roll} keeps {r.keep}" for r in others))
    by_gap = sorted(rows, key=lambda r: r.value - r.next_value)
    print(
        "closest:",
        [(r.roll, r.keep, r.next_keep, round(r.value - r.next_value, 4)) for r in by_gap[:8]],
    )
    print(
        "largest:",
        [(r.roll, r.keep, r.next_keep, round(r.value - r.next_value, 4)) for r in by_gap[-5:]],
    )
    for roll in (
        "12345",
        "12346",
        "11222",
        "11666",
        "55666",
        "11236",
        "12456",
        "13456",
        "11345",
        "11346",
    ):
        r = next(x for x in rows if x.roll == roll)
        print(f"  {roll}: keep {r.keep} {r.value:.4f}, else {r.next_keep} {r.next_value:.4f}")


def greedy() -> None:
    orders = {
        "score card order": CATEGORIES,
        "reversed": CATEGORIES[::-1],
        "Glenn's rule-based fallback order": (
            "aces",
            "chance",
            "twos",
            "yahtzee",
            "four_kind",
            "large_straight",
            "small_straight",
            "threes",
            "full_house",
            "fours",
            "three_kind",
            "fives",
            "sixes",
        ),
    }
    for name, order in orders.items():
        t, greedy_values = solve(Rules(categories=order), anchor=GREEDY)
        g = policy.graph(t, greedy_values)
        print(
            f"greedy, ties to the first box in {name}: {expected(g):.6f}; upper bonus in {100 * g.stats['upper_bonus'][0]:.2f}% of games",
            flush=True,
        )
        if order == CATEGORIES:
            report(policy.distribution(g))


def simulate(games: int = 20_000, seed: int = 1) -> None:
    t, values = solve(OFFICIAL)
    _, greedy_values = solve(OFFICIAL, anchor=GREEDY)
    dice = deal(t, games, seed)
    best, simple = play(t, values, dice), play(t, greedy_values, dice)
    for name, x in (("optimal", best), ("greedy", simple), ("difference", best - simple)):
        m, h = interval(x)
        print(f"{name:10s} {m:8.3f} +- {h:.3f}  (sd {x.std(ddof=1):.2f})")
    print(
        f"{games:,d} games, seed {seed}; optimal scores more in {np.mean(best > simple):.4f}, less in {np.mean(best < simple):.4f}"
    )


def main(argv: list[str]) -> None:
    sections = {
        "optimal": optimal,
        "rules": rules,
        "openings": openings,
        "greedy": greedy,
        "simulate": simulate,
    }
    for name in argv or sections:
        print(f"== {name}")
        sections[name]()


if __name__ == "__main__":
    main(sys.argv[1:])
