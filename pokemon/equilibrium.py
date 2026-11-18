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
when the solution is strict. Otherwise ``optimal_weight_bound`` finds the least and greatest
weight a strategy takes over all optimal strategies: LP suggests a primal and a dual vertex,
both are rebuilt in Fractions, and they prove the bound when their values meet.

Every game can be played on the chart of Generation 1, of Generations 2 to 5, or of
Generation 6 on (``CHARTS``), and over the typings that exist in a generation.

    uv run --group pokemon python -m pokemon.equilibrium
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
from itertools import combinations, permutations

import numpy as np
from scipy.optimize import linprog

from pokemon import pokedex
from pokemon.pokemon_types import chart, gen6

Matrix = list[list[Fraction]]
Strategy = list[Fraction]
Chart = dict[str, dict[str, float]]
Typing = tuple[str, ...]

TYPES: tuple[str, ...] = tuple(gen6)  # the chart's order: normal, fire, water, ...


def singles(c: Chart = gen6) -> tuple[Typing, ...]:
    """Each type of a chart as a typing of its own."""
    return tuple((t,) for t in c)


def typings(c: Chart = gen6) -> tuple[Typing, ...]:
    """One type, or two in chart order: 18 + 153 = 171 with 18 types."""
    return singles(c) + tuple(combinations(c, 2))


TYPINGS: tuple[Typing, ...] = typings()

# The three charts: Generation 1, Generations 2 to 5 (Steel and Dark), 6 on (Fairy).
CHARTS: dict[str, Chart] = {"1": chart(1), "2-5": chart(2), "6-9": chart(6)}


def generation_6_split() -> dict[str, Chart]:
    """Generation 6 made two changes; each alone, applied to the Generation 2 to 5 chart.

    'steel change': Steel no longer resists Ghost and Dark, and there is no Fairy.
    'fairy': Fairy is added, and Steel still resists Ghost and Dark.
    """
    six = chart(6)
    steel = {a: {d: m for d, m in row.items() if d != "fairy"} for a, row in six.items()}
    del steel["fairy"]
    fairy = {a: dict(row) for a, row in six.items()}
    fairy["ghost"]["steel"] = fairy["dark"]["steel"] = 0.5
    return {"steel change": steel, "fairy": fairy}


def existing_typings(generation: int = 7) -> tuple[Typing, ...]:
    """The typings that some form has as of a generation, in ``typings`` order."""
    found = pokedex.existing_typings(generation)
    return tuple(t for t in typings(chart(generation)) if t in found)


def multiplier(attack: str, typing: Sequence[str], c: Chart = gen6) -> Fraction:
    """Damage multiplier of an attack against a typing: the product over its types."""
    m = Fraction(1)
    for t in typing:
        m *= Fraction(c[attack].get(t, 1.0))
    return m


def best_multiplier(attacker: Sequence[str], defender: Sequence[str], c: Chart = gen6) -> Fraction:
    """The larger multiplier among the attacker's own types (it picks its better move)."""
    return max(multiplier(t, defender, c) for t in attacker)


# The games.


def attack_matrix(defenders: Sequence[Sequence[str]] | None = None, c: Chart = gen6) -> Matrix:
    """Rows are attacking types, columns are defending typings, entries are multipliers."""
    if defenders is None:
        defenders = singles(c)
    return [[multiplier(a, d, c) for d in defenders] for a in c]


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
    players: Sequence[Sequence[str]] | None = None,
    c: Chart = gen6,
) -> Matrix:
    """g[i][j] = score(damage i deals j, damage j deals i), each using its best own type."""
    if players is None:
        players = singles(c)
    return [
        [score(best_multiplier(p, q, c), best_multiplier(q, p, c)) for q in players]
        for p in players
    ]


def duel_matrix(players: Sequence[Sequence[str]] | None = None, c: Chart = gen6) -> Matrix:
    """The duel game of Schmitz (2021), over single types or over typings."""
    return symmetric_matrix(duel_score, players, c)


def log_ratio_matrix(immune: int = -2, c: Chart = gen6) -> Matrix:
    """The log-ratio game over single types."""
    return symmetric_matrix(lambda a, b: log_ratio(a, b, immune), None, c)


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


def optimal_weight_bound(
    g: Matrix, v: Fraction, c: Sequence[Fraction]
) -> tuple[Fraction, Strategy]:
    """The greatest c·x over the row player's optimal strategies, and an x reaching it.

    The optimal strategies are P = {x >= 0, sum x = 1, (xᵀg)_j >= v for every column j}.
    Floating-point LP finds a vertex x of P and a vertex (y, λ) of the dual problem: minimize
    λ - v·sum(y) subject to y >= 0 and λ >= c_i + (gy)_i for every row i. Both are rebuilt in
    Fractions from the equations that hold at them. Then for every x' in P,

        c·x' <= sum_i x'_i (λ - (gy)_i) = λ - x'ᵀgy <= λ - v·sum(y),

    so if the exact primal x reaches λ - v·sum(y), that number is the maximum. Both
    vectors are checked exactly; floating point only suggests which equations to solve.
    """
    arr, cf = np.array(g, dtype=float), np.array([float(q) for q in c])
    m, n = arr.shape
    tol = 1e-7

    primal = linprog(
        -cf,
        A_ub=-arr.T,
        b_ub=-float(v) * np.ones(n),
        A_eq=np.ones((1, m)),
        b_eq=[1],
        method="highs-ds",
    )
    assert primal.status == 0, primal.message
    rows = [i for i in range(m) if primal.x[i] > tol]
    tight = [j for j, u in enumerate(primal.x @ arr) if abs(u - float(v)) < tol]
    eqs = [[g[i][j] for i in rows] for j in tight] + [[Fraction(1)] * len(rows)]
    sol = solve_linear(eqs, [v] * len(tight) + [Fraction(1)])
    assert sol is not None
    x = [Fraction(0)] * m
    for i, p in zip(rows, sol, strict=True):
        x[i] = p
    assert all(p >= 0 for p in x) and sum(x) == 1
    assert all(u >= v for u in payoffs_against_columns(g, x)), "primal is infeasible"
    best = sum((ci * xi for ci, xi in zip(c, x, strict=True)), Fraction(0))

    # Dual variables y_1..y_n, λ: minimize λ - v·sum(y) with (gy)_i - λ <= -c_i.
    dual = linprog(
        np.concatenate([-float(v) * np.ones(n), [1.0]]),
        A_ub=np.hstack([arr, -np.ones((m, 1))]),
        b_ub=-cf,
        bounds=[(0, None)] * n + [(None, None)],
        method="highs-ds",
    )
    assert dual.status == 0, dual.message
    cols = [j for j in range(n) if dual.x[j] > tol]
    lam_f = dual.x[-1]
    tight_rows = [i for i in range(m) if abs(cf[i] + arr[i] @ dual.x[:n] - lam_f) < tol]
    # Unknowns y_cols, λ: tight rows hold with equality, and the bound equals c·x.
    eqs = [[g[i][j] for j in cols] + [Fraction(-1)] for i in tight_rows]
    eqs.append([-v] * len(cols) + [Fraction(1)])
    sol = solve_linear(eqs, [-c[i] for i in tight_rows] + [best])
    assert sol is not None
    y = [Fraction(0)] * n
    for j, q in zip(cols, sol[:-1], strict=True):
        y[j] = q
    lam = sol[-1]
    assert all(q >= 0 for q in y), "dual is infeasible"
    assert all(ci + u <= lam for ci, u in zip(c, payoffs_of_rows(g, y), strict=True))
    assert lam - v * sum(y) == best, "the bounds do not meet"
    return best, x


def weight_ranges(g: Matrix, v: Fraction) -> list[tuple[Fraction, Fraction]]:
    """Least and greatest weight of each row over the row player's optimal strategies, exactly.

    Two calls of ``optimal_weight_bound`` per row: maximize x_i, and maximize -x_i.
    """
    m = len(g)
    ranges = []
    for i in range(m):
        e = [Fraction(int(k == i)) for k in range(m)]
        hi = optimal_weight_bound(g, v, e)[0]
        lo = -optimal_weight_bound(g, v, [-q for q in e])[0]
        ranges.append((lo, hi))
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


def spread(g: Matrix, v: Fraction) -> tuple[str, int, Fraction, Fraction] | None:
    """A strategy whose weight differs between two optimal strategies, or None if none does.

    Returns (side, index, least weight, greatest weight), side 'row' or 'column'. Each
    weight is proved exactly by ``optimal_weight_bound``. In a symmetric game (g = -gᵀ) both
    players have the same optimal strategies, so one side is enough.
    """
    sides = [("row", g, v)]
    if g != transpose_negated(g):
        sides.append(("column", transpose_negated(g), -v))
    for side, h, w in sides:
        m = len(h)
        for i in range(m):
            e = [Fraction(int(k == i)) for k in range(m)]
            hi = optimal_weight_bound(h, w, e)[0]
            lo = -optimal_weight_bound(h, w, [-q for q in e])[0]
            if lo != hi:
                return side, i, lo, hi
    return None


def optimal_vertices(g: Matrix, v: Fraction) -> list[Strategy]:
    """Optimal row strategies that maximize and minimize each weight in turn, without repeats."""
    m = len(g)
    found: list[Strategy] = []
    for i in range(m):
        for sign in (1, -1):
            e = [Fraction(sign * int(k == i)) for k in range(m)]
            x = optimal_weight_bound(g, v, e)[1]
            if x not in found:
                found.append(x)
    return found


def optimal_set_dimension(g: Matrix, v: Fraction, points: Sequence[Strategy]) -> int:
    """The dimension of the row player's optimal set in a symmetric game, proved exactly.

    ``points`` are optimal strategies. Their average y is optimal, and so is y for the
    column player (g = -gᵀ). Any optimal x then uses only rows i with (gy)_i = v and earns
    exactly v on every column y uses. Those equations, with sum x = 1, cut out an affine
    space containing the optimal set; its dimension bounds the set's from above. The points
    span an affine space of their own inside the set, which bounds it from below. The two
    must meet, or this fails.
    """
    assert g == transpose_negated(g)
    k = len(points)
    y = [sum((p[i] for p in points), Fraction(0)) / k for i in range(len(g))]
    rows = [i for i, u in enumerate(payoffs_of_rows(g, y)) if u == v]
    cols = [j for j, q in enumerate(y) if q > 0]
    eqs = [[g[i][j] for i in rows] for j in cols] + [[Fraction(1)] * len(rows)]
    upper = len(rows) - rank(eqs)
    diffs = [[p[i] - points[0][i] for i in range(len(g))] for p in points[1:]]
    lower = rank(diffs) if diffs else 0
    assert lower == upper, (lower, upper)
    return upper


def is_unique(g: Matrix, v: Fraction) -> bool:
    """Whether each player has exactly one optimal strategy, decided exactly.

    Tries the certificate first; if it does not apply, looks for a strategy whose least and
    greatest optimal weights differ (``spread``).
    """
    x, y, w = exact_equilibrium(g)
    assert w == v
    return has_unique_certificate(g, x, y, v) or spread(g, v) is None


def never_played(names: Sequence, g: Matrix, v: Fraction) -> list:
    """Row strategies with weight 0 in every optimal strategy, decided exactly."""
    return [n for n, (_, hi) in zip(names, weight_ranges(g, v), strict=True) if hi == 0]


# Rock-paper-scissors triangles.


def super_effective(a: str, b: str, c: Chart = gen6) -> bool:
    """Whether type a's attacks do double damage to type b."""
    return c[a].get(b, 1.0) == 2.0


def triangles(c: Chart = gen6) -> list[tuple[str, str, str]]:
    """Every 3-cycle a -> b -> c -> a of super-effective attacks, once each.

    A cycle is written starting from its type earliest in the chart's order.
    """
    order = list(c)
    found = []
    for a, b, d in permutations(order, 3):
        if order.index(a) < min(order.index(b), order.index(d)) and (
            super_effective(a, b, c) and super_effective(b, d, c) and super_effective(d, a, c)
        ):
            found.append((a, b, d))
    return found


def is_strict(cycle: tuple[str, str, str], c: Chart = gen6) -> bool:
    """Whether each type also resists (or is immune to) the type it hits."""
    a, b, d = cycle
    return all(c[y].get(x, 1.0) < 1 for x, y in ((a, b), (b, d), (d, a)))


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


def leaders(names: Sequence, p: Strategy, k: int = 3) -> str:
    """'water/ground 0.108, ...': the k heaviest strategies, and how many there are."""
    s = list(support(names, p).items())
    top = ", ".join(f"{n if isinstance(n, str) else label(n)} {float(q):.3f}" for n, q in s[:k])
    return f"{len(s)} [{top}]"


def uniqueness(g: Matrix, x: Strategy, y: Strategy, v: Fraction, names: Sequence) -> str:
    """'unique (certificate)', or one strategy whose optimal weight is not fixed."""
    if has_unique_certificate(g, x, y, v):
        return "unique (certificate)"
    found = spread(g, v)
    if found is None:
        return "unique (exact weight ranges)"
    side, i, lo, hi = found
    name = names[0][i] if side == "row" else names[1][i]
    return f"not unique: {side} {name if isinstance(name, str) else label(name)} from {lo} to {hi}"


def chart_games(c: Chart) -> dict[str, tuple[Matrix, Sequence, Sequence]]:
    """The games on one chart: (matrix, row names, column names)."""
    t, ty = list(c), [label(p) for p in typings(c)]
    return {
        "attack": (attack_matrix(None, c), t, t),
        "attack vs typings": (attack_matrix(typings(c), c), t, ty),
        "duel": (duel_matrix(None, c), t, t),
        "duel over typings": (duel_matrix(typings(c), c), ty, ty),
        "log-ratio k=1": (log_ratio_matrix(-1, c), t, t),
        "log-ratio k=2": (log_ratio_matrix(-2, c), t, t),
    }


def main() -> None:
    singles = list(TYPES)
    names = [label(t) for t in TYPINGS]
    report("Attack game, single defending types", attack_matrix(), singles, singles)
    report("Attack game, defending typings", attack_matrix(TYPINGS), singles, names)
    report("Duel game, single types", duel_matrix(), singles, singles)
    report("Duel game, typings", duel_matrix(TYPINGS), names, names)

    print("== Log-ratio game: an immunity counts as -k; exact ==")
    for immune in (-1, -2, -3, -4, -6, -10):
        g = log_ratio_matrix(immune)
        vertices = optimal_vertices(g, Fraction(0))
        dim = optimal_set_dimension(g, Fraction(0), vertices)
        never = never_played(singles, g, Fraction(0))
        print(
            f"  k={-immune:>2}: optimal set of dimension {dim}; never played ({len(never)}): {never}"
        )
        if dim == 1:
            ghost = TYPES.index("ghost")
            a, b = sorted(vertices, key=lambda x: x[ghost])
            t = b[ghost]
            assert all(b[i] == (1 - t) * a[i] + t * (i == ghost) for i in range(len(a)))
            print(
                f"        a segment: from a strategy without Ghost to (1 - t) it + t Ghost, t = {t}"
            )
    print()

    print("== Each chart: Generation 1, Generations 2 to 5, Generation 6 on ==")
    for name, c in CHARTS.items():
        print(f"-- Generation {name}: {len(c)} types --")
        for game, (g, rows, cols) in chart_games(c).items():
            x, y, v = exact_equilibrium(g)
            print(f"  {game}: value {v}; {uniqueness(g, x, y, v, (rows, cols))}")
            print(f"    row {leaders(rows, x)}; column {leaders(cols, y)}")
    print()

    print("== Generation 6's two changes, each alone ==")
    for name, c in generation_6_split().items():
        t = list(c)
        for game, g in (
            ("duel", duel_matrix(None, c)),
            ("log-ratio k=1", log_ratio_matrix(-1, c)),
            ("log-ratio k=2", log_ratio_matrix(-2, c)),
        ):
            x, y, v = exact_equilibrium(g)
            print(f"  {name}, {game}: {uniqueness(g, x, y, v, (t, t))}; {leaders(t, x, 4)}")
    print()

    print("== Typings that exist, by generation, with that generation's chart ==")
    for gen in range(1, 10):
        c, existing = chart(gen), existing_typings(gen)
        damage = {t: sum((multiplier(a, t, c) for a in c), Fraction(0)) for t in existing}
        least = min(damage.values())
        best = [label(t) for t in existing if damage[t] == least]
        g = duel_matrix(existing, c)
        x, y, v = exact_equilibrium(g)
        print(
            f"  {gen}: {len(pokedex.forms_as_of(gen))} forms, {len(existing)} typings; "
            f"least damage {least} ({', '.join(best)}) of {len(c)} neutral"
        )
        print(f"     duel: {leaders(existing, x)}; {uniqueness(g, x, y, v, (existing, existing))}")
    print()

    cycles = triangles()
    strict = [c for c in cycles if is_strict(c)]
    print(f"== {len(cycles)} super-effective triangles, {len(strict)} strict ==")
    for c in cycles:
        print(f"  {' -> '.join(c)}{'   (strict)' if is_strict(c) else ''}")


if __name__ == "__main__":
    main()
