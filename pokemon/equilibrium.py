"""The type chart as a zero-sum game, solved exactly.

Three games, each a payoff matrix g: the row player wins g[i][j] and the column player loses it.

- **Attack game.** The row player picks an attacking type, the column player a defending type
  (or a pair of types), and the payoff is the damage multiplier.
- **Duel game** (the model of Schmitz 2021). Two Pokémon, identical except for type, each
  attack with a move of their own type. The payoff is the share of HP the winner has left
  when the loser faints. If the row type deals multiplier a and takes b, it scores 1 - b/a
  when a > b, -(1 - a/b) when a < b, and 0 when a = b. A Pokémon with two types attacks with
  the better of its two types.
- **Log-ratio game.** Single types again, with payoff log2(a/b). The multipliers are powers
  of two, so the payoffs are integers once an immunity is given a finite log2 (``immune``).

The duel and log-ratio games are symmetric (g = -gᵀ), so their value is 0.

Floating-point LP (scipy's HiGHS) finds an equilibrium; ``exact_equilibrium`` then solves the
indifference equations on its support with Fractions and checks every equilibrium inequality
exactly. ``has_unique_certificate`` proves, also exactly, that no other equilibrium exists
when the solution is strict; otherwise ``weight_ranges`` finds by LP the least and greatest
weight each strategy takes over all optimal strategies.

    uv run --group pokemon python -m pokemon.equilibrium
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from fractions import Fraction
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from pokemon.pokemon_types import gen6

Matrix = list[list[Fraction]]
Strategy = list[Fraction]

TYPES: tuple[str, ...] = tuple(gen6)  # the chart's order: normal, fire, water, ...
# A Pokémon's typing: one type, or two in chart order. 18 + 153 = 171.
TYPINGS: tuple[tuple[str, ...], ...] = tuple((t,) for t in TYPES) + tuple(combinations(TYPES, 2))
GEN7_CSV = Path(__file__).with_name("pokemon_gen7.csv")


def existing_typings(path: Path = GEN7_CSV) -> tuple[tuple[str, ...], ...]:
    """The typings that some Pokémon or form in ``pokemon_gen7.csv`` has, in TYPINGS order."""
    with path.open(encoding="utf-8") as f:
        found = {
            tuple(sorted((t.lower() for t in (r["type1"], r["type2"]) if t), key=TYPES.index))
            for r in csv.DictReader(f)
        }
    return tuple(t for t in TYPINGS if t in found)


def multiplier(attack: str, typing: Sequence[str]) -> Fraction:
    """Damage multiplier of an attack against a typing: the product over its types."""
    m = Fraction(1)
    for t in typing:
        m *= Fraction(gen6[attack].get(t, 1.0))
    return m


def best_multiplier(attacker: Sequence[str], defender: Sequence[str]) -> Fraction:
    """The larger multiplier among the attacker's own types (it picks its better move)."""
    return max(multiplier(t, defender) for t in attacker)


# The games.


def attack_matrix(defenders: Sequence[Sequence[str]] = tuple((t,) for t in TYPES)) -> Matrix:
    """Rows are attacking types, columns are defending typings, entries are multipliers."""
    return [[multiplier(a, d) for d in defenders] for a in TYPES]


def duel_score(a: Fraction, b: Fraction) -> Fraction:
    """Share of HP the winner keeps, signed: dealing a per hit and taking b per hit."""
    if a == b:
        return Fraction(0)
    if a > b:
        return 1 - b / a
    return -(1 - a / b)


def log_ratio(a: Fraction, b: Fraction, immune: int) -> Fraction:
    """log2(a/b), with log2(0) taken to be ``immune`` (a negative integer)."""

    def lg(m: Fraction) -> int:
        if m == 0:
            return immune
        return m.numerator.bit_length() - m.denominator.bit_length()  # exact for powers of 2

    return Fraction(lg(a) - lg(b))


def symmetric_matrix(
    score: Callable[[Fraction, Fraction], Fraction],
    typings: Sequence[Sequence[str]] = tuple((t,) for t in TYPES),
) -> Matrix:
    """g[i][j] = score(damage i deals j, damage j deals i), each using its best own type."""
    return [[score(best_multiplier(p, q), best_multiplier(q, p)) for q in typings] for p in typings]


def duel_matrix(typings: Sequence[Sequence[str]] = tuple((t,) for t in TYPES)) -> Matrix:
    """The duel game of Schmitz (2021), over single types or over typings."""
    return symmetric_matrix(duel_score, typings)


def log_ratio_matrix(immune: int = -2) -> Matrix:
    """The log-ratio game over single types."""
    return symmetric_matrix(lambda a, b: log_ratio(a, b, immune))


# Solving.


def lp_strategy(g: Matrix) -> tuple[np.ndarray, float]:
    """A maximin strategy for the row player and the value, by floating-point LP."""
    arr = np.array(g, dtype=float)
    m, n = arr.shape
    # Variables x_1..x_m, v. Maximize v subject to (xᵀg)_j >= v, sum x = 1, x >= 0.
    c = np.zeros(m + 1)
    c[-1] = -1
    a_ub = np.hstack([-arr.T, np.ones((n, 1))])
    a_eq = np.hstack([np.ones((1, m)), np.zeros((1, 1))])
    bounds = [(0, None)] * m + [(None, None)]
    # The dual simplex method returns a vertex, which the support equations then pin down.
    res = linprog(
        c, A_ub=a_ub, b_ub=np.zeros(n), A_eq=a_eq, b_eq=[1], bounds=bounds, method="highs-ds"
    )
    assert res.status == 0, res.message
    return res.x[:m], float(res.x[-1])


def transpose_negated(g: Matrix) -> Matrix:
    """The game from the column player's side."""
    return [[-g[i][j] for i in range(len(g))] for j in range(len(g[0]))]


def solve_linear(rows: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction] | None:
    """One exact solution of a linear system (free variables set to 0), or None if inconsistent."""
    n = len(rows[0])
    aug = [list(r) + [b] for r, b in zip(rows, rhs, strict=True)]
    pivots = []
    r = 0
    for c in range(n):
        p = next((i for i in range(r, len(aug)) if aug[i][c] != 0), None)
        if p is None:
            continue
        aug[r], aug[p] = aug[p], aug[r]
        aug[r] = [x / aug[r][c] for x in aug[r]]
        for i in range(len(aug)):
            if i != r and aug[i][c] != 0:
                f = aug[i][c]
                aug[i] = [x - f * y for x, y in zip(aug[i], aug[r], strict=True)]
        pivots.append(c)
        r += 1
    if any(all(x == 0 for x in row[:-1]) and row[-1] != 0 for row in aug):
        return None
    sol = [Fraction(0)] * n
    for i, c in enumerate(pivots):
        sol[c] = aug[i][-1]
    return sol


def payoffs_against_columns(g: Matrix, x: Strategy) -> list[Fraction]:
    """(xᵀg)_j: what row strategy x earns against each pure column."""
    return [sum((x[i] * g[i][j] for i in range(len(g))), Fraction(0)) for j in range(len(g[0]))]


def payoffs_of_rows(g: Matrix, y: Strategy) -> list[Fraction]:
    """(gy)_i: what each pure row earns against column strategy y."""
    return [sum((g[i][j] * y[j] for j in range(len(y))), Fraction(0)) for i in range(len(g))]


def is_equilibrium(g: Matrix, x: Strategy, y: Strategy, v: Fraction) -> bool:
    """Exact check: x and y are probability vectors, x guarantees v, and y holds rows to v."""

    def is_distribution(p: Strategy) -> bool:
        return all(q >= 0 for q in p) and sum(p) == 1

    return (
        is_distribution(x)
        and is_distribution(y)
        and min(payoffs_against_columns(g, x)) >= v
        and max(payoffs_of_rows(g, y)) <= v
    )


def exact_from_support(
    g: Matrix, rows: list[int], cols: list[int]
) -> tuple[Strategy, Fraction] | None:
    """Row strategy on ``rows`` that makes every column in ``cols`` pay the same v."""
    eqs = [[g[i][j] for i in rows] + [Fraction(-1)] for j in cols]
    eqs.append([Fraction(1)] * len(rows) + [Fraction(0)])
    sol = solve_linear(eqs, [Fraction(0)] * len(cols) + [Fraction(1)])
    if sol is None:
        return None
    x = [Fraction(0)] * len(g)
    for i, p in zip(rows, sol[:-1], strict=True):
        x[i] = p
    return x, sol[-1]


def exact_equilibrium(g: Matrix, tol: float = 1e-7) -> tuple[Strategy, Strategy, Fraction]:
    """An equilibrium (x, y, v) in exact rationals, checked with ``is_equilibrium``.

    For each player the LP gives a vertex x. Its support, and the opponent's pure replies
    that x holds exactly to the value, give linear equations whose only solution is x.
    """

    def exact_side(h: Matrix) -> tuple[Strategy, Fraction] | None:
        xf, vf = lp_strategy(h)
        rows = [i for i, p in enumerate(xf) if p > tol]
        arr = np.array(h, dtype=float)
        tight = [j for j, u in enumerate(xf @ arr) if abs(u - vf) < tol]
        return exact_from_support(h, rows, tight)

    g_col = transpose_negated(g)
    rx = exact_side(g)
    ry = exact_side(g_col)
    assert rx is not None and ry is not None, "support equations are inconsistent"
    (x, v), (y, w) = rx, ry
    assert v == -w, (v, w)
    assert is_equilibrium(g, x, y, v), "the exact candidate is not an equilibrium"
    return x, y, v


def weight_ranges(g: Matrix, v: Fraction) -> list[tuple[float, float]]:
    """Least and greatest weight each row gets over all of the row player's optimal strategies.

    The optimal strategies are the polytope {x >= 0, sum x = 1, xᵀg >= v}; two LPs per row.
    """
    arr = np.array(g, dtype=float)
    m, n = arr.shape
    a_ub, b_ub, a_eq = -arr.T, -float(v) * np.ones(n), np.ones((1, m))
    ranges = []
    for i in range(m):
        c = np.zeros(m)
        c[i] = 1
        lo = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[1])
        hi = linprog(-c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[1])
        assert lo.status == 0 and hi.status == 0
        ranges.append((max(lo.fun, 0.0), -hi.fun))
    return ranges


def rank(rows: list[list[Fraction]]) -> int:
    """Rank of a matrix, by exact elimination."""
    rows = [list(r) for r in rows]
    r = 0
    for c in range(len(rows[0]) if rows else 0):
        p = next((i for i in range(r, len(rows)) if rows[i][c] != 0), None)
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        for i in range(r + 1, len(rows)):
            f = rows[i][c] / rows[r][c]
            rows[i] = [a - f * b for a, b in zip(rows[i], rows[r], strict=True)]
        r += 1
    return r


def has_unique_certificate(g: Matrix, x: Strategy, y: Strategy, v: Fraction) -> bool:
    """An exact proof that (x, y) is the only equilibrium, when one exists.

    Any optimal x' earns v against y, so it uses only rows i with (gy)_i = v; and it earns
    exactly v against every column y uses. If those rows are exactly x's support, and the
    equations "x' pays v on y's support, and sums to 1" have one solution, then x' = x.
    The same argument, with the players swapped, covers y.
    """
    s_rows = [i for i, p in enumerate(x) if p > 0]
    t_cols = [j for j, q in enumerate(y) if q > 0]
    if [i for i, u in enumerate(payoffs_of_rows(g, y)) if u == v] != s_rows:
        return False
    if [j for j, u in enumerate(payoffs_against_columns(g, x)) if u == v] != t_cols:
        return False
    one, zero = Fraction(1), Fraction(0)
    rows_eqs = [[g[i][j] for i in s_rows] + [-one] for j in t_cols]
    rows_eqs.append([one] * len(s_rows) + [zero])
    cols_eqs = [[g[i][j] for j in t_cols] + [-one] for i in s_rows]
    cols_eqs.append([one] * len(t_cols) + [zero])
    return rank(rows_eqs) == len(s_rows) + 1 and rank(cols_eqs) == len(t_cols) + 1


def is_unique(g: Matrix, v: Fraction, tol: float = 1e-9) -> bool:
    """Whether each player has exactly one optimal strategy.

    Tries the exact certificate first; if it does not apply, compares the least and greatest
    weight of every strategy by LP. In a symmetric game (g = -gᵀ) both players have the same
    optimal strategies, so one side is enough.
    """
    x, y, w = exact_equilibrium(g)
    assert w == v
    if has_unique_certificate(g, x, y, v):
        return True
    sides = [(g, v)] if g == transpose_negated(g) else [(g, v), (transpose_negated(g), -v)]
    return all(hi - lo <= tol for h, w in sides for lo, hi in weight_ranges(h, w))


def never_played(names: Sequence, g: Matrix, v: Fraction, tol: float = 1e-9) -> list:
    """Row strategies with weight 0 in every optimal strategy."""
    return [n for n, (_, hi) in zip(names, weight_ranges(g, v), strict=True) if hi <= tol]


# Rock-paper-scissors triangles.


def super_effective(a: str, b: str) -> bool:
    """Whether type a's attacks do double damage to type b."""
    return gen6[a].get(b, 1.0) == 2.0


def triangles() -> list[tuple[str, str, str]]:
    """Every 3-cycle a -> b -> c -> a of super-effective attacks, once each.

    A cycle is written starting from its type earliest in the chart's order.
    """
    found = []
    for a, b, c in permutations(TYPES, 3):
        if TYPES.index(a) < min(TYPES.index(b), TYPES.index(c)) and (
            super_effective(a, b) and super_effective(b, c) and super_effective(c, a)
        ):
            found.append((a, b, c))
    return found


def is_strict(cycle: tuple[str, str, str]) -> bool:
    """Whether each type also resists (or is immune to) the type it hits."""
    a, b, c = cycle
    return all(gen6[y].get(x, 1.0) < 1 for x, y in ((a, b), (b, c), (c, a)))


def support(names: Sequence, p: Strategy) -> dict:
    """The strategies with positive weight, heaviest first."""
    return dict(
        sorted(((n, q) for n, q in zip(names, p, strict=True) if q > 0), key=lambda t: -t[1])
    )


def label(typing: Sequence[str]) -> str:
    """'water/flying' for a pair, 'water' for one type."""
    return "/".join(typing)


def report(title: str, g: Matrix, row_names: Sequence, col_names: Sequence) -> None:
    """Print one game's exact equilibrium."""
    x, y, v = exact_equilibrium(g)
    print(f"== {title} ==")
    print(f"value {v} = {float(v):.4f}; unique equilibrium: {is_unique(g, v)}")
    for name, side, p in (("row", row_names, x), ("column", col_names, y)):
        s = support(side, p)
        print(f"{name} support ({len(s)}):")
        for k, q in s.items():
            print(f"  {k:<18} {float(q):.4f}  ({q})")
    print()


def main() -> None:
    singles = list(TYPES)
    typings = [label(t) for t in TYPINGS]
    report("Attack game, single defending types", attack_matrix(), singles, singles)
    report("Attack game, defending typings", attack_matrix(TYPINGS), singles, typings)
    report("Duel game, single types", duel_matrix(), singles, singles)
    report("Duel game, typings", duel_matrix(TYPINGS), typings, typings)
    existing = existing_typings()
    names = [label(t) for t in existing]
    report(
        f"Duel game, the {len(existing)} typings in pokemon_gen7.csv",
        duel_matrix(existing),
        names,
        names,
    )
    print("== Log-ratio game: types never played, by the log2 given to an immunity ==")
    for immune in (-1, -2, -3, -4, -6, -10):
        g = log_ratio_matrix(immune)
        _, _, v = exact_equilibrium(g)
        never = never_played(singles, g, v)
        print(f"  {immune:>3}: unique {is_unique(g, v)!s:<5} never ({len(never)}): {never}")
    print()
    cycles = triangles()
    strict = [c for c in cycles if is_strict(c)]
    print(f"== {len(cycles)} super-effective triangles, {len(strict)} strict ==")
    for c in cycles:
        print(f"  {' -> '.join(c)}{'   (strict)' if is_strict(c) else ''}")


if __name__ == "__main__":
    main()
