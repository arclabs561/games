"""How much of the type chart is a ranking: a Hodge decomposition of the log-ratio game.

The **flow** between types i and j is A[i][j] = log2(E(i, j) / E(j, i)), where E(i, j) is the
multiplier when i attacks j: how many doublings i's attacks gain on j's. It is skew-symmetric,
an edge flow on the complete graph of types. An immunity needs a finite log2(0); call it
``immune`` (-k), as in the log-ratio game of ``equilibrium.py``. The flow is then integral.

Jiang, Lim, Yao and Ye (2011) split any edge flow into orthogonal parts:

- a **gradient** part s_i - s_j, the least-squares fit of a score s per type (a ranking);
- a **curl** part, cycles around triangles (locally cyclic);
- a **harmonic** part, cycles around longer loops only (globally cyclic).

On a complete graph with every triangle filled in, the harmonic part is zero, and the
least-squares score is the row mean, s_i = (1/n) sum_j A[i][j]. Everything here is exact in
Fractions, and both facts are checked rather than assumed: ``cone`` writes the residual as
a sum of triangle cycles, and ``tests/test_pokemon_hodge.py`` compares s with numpy's
least squares.

    uv run --group pokemon python -m pokemon.hodge
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from pokemon.equilibrium import CHARTS, Chart, Matrix, log_ratio_matrix
from pokemon.pokemon_types import gen6

IMMUNE = (-1, -2, -3, -4, -6, -10)

Triangle = tuple[int, int, int]


def flow(immune: int = -2, c: Chart = gen6) -> Matrix:
    """A[i][j] = log2 E(i, j) - log2 E(j, i), with log2(0) taken to be ``immune``."""
    return log_ratio_matrix(immune, c)


def score(a: Matrix) -> list[Fraction]:
    """The least-squares ranking on the complete graph: s_i = (1/n) sum_j A[i][j].

    It solves the normal equations n s_i - sum_j s_j = sum_j A[i][j], and sum s = 0.
    """
    n = len(a)
    return [sum(row, Fraction(0)) / n for row in a]


def gradient(s: list[Fraction]) -> Matrix:
    """The flow of a score: G[i][j] = s_i - s_j."""
    return [[si - sj for sj in s] for si in s]


def minus(a: Matrix, b: Matrix) -> Matrix:
    """Entrywise a - b."""
    return [[x - y for x, y in zip(ra, rb, strict=True)] for ra, rb in zip(a, b, strict=True)]


def inner(a: Matrix, b: Matrix) -> Fraction:
    """The inner product of two edge flows: the sum over pairs i < j of a[i][j]·b[i][j]."""
    n = len(a)
    return sum((a[i][j] * b[i][j] for i in range(n) for j in range(i + 1, n)), Fraction(0))


def divergence(a: Matrix) -> list[Fraction]:
    """Net outflow at each type: sum_j a[i][j]."""
    return [sum(row, Fraction(0)) for row in a]


def curl(a: Matrix, t: Triangle) -> Fraction:
    """Circulation around a triangle i -> j -> k -> i."""
    i, j, k = t
    return a[i][j] + a[j][k] + a[k][i]


def boundary_flow(weights: dict[Triangle, Fraction], n: int) -> Matrix:
    """curl*: the edge flow that sends weight w around each triangle i -> j -> k -> i."""
    out = [[Fraction(0)] * n for _ in range(n)]
    for (i, j, k), w in weights.items():
        for p, q in ((i, j), (j, k), (k, i)):
            out[p][q] += w
            out[q][p] -= w
    return out


def cone(r: Matrix) -> dict[Triangle, Fraction]:
    """Triangle weights whose boundary flow is r, if r is divergence-free.

    Put weight r[i][j] on each triangle 0 -> i -> j -> 0, for 0 < i < j. Edge (i, j) gets
    r[i][j]. Edge (0, i) gets sum over j of r[i][j] for j not 0 or i, which is -r[i][0] when
    r's divergence at i is 0, that is, r[0][i]. So r is all curl, and no harmonic part is left.
    """
    n = len(r)
    return {(0, i, j): r[i][j] for i in range(1, n) for j in range(i + 1, n) if r[i][j]}


def decompose(a: Matrix) -> tuple[list[Fraction], Matrix, Matrix]:
    """(s, gradient, cyclic) with a = gradient + cyclic, checked exactly.

    The checks: the parts are orthogonal, the cyclic part has no divergence, and it is the
    boundary flow of ``cone`` triangle weights (so its harmonic part is zero).
    """
    s = score(a)
    g = gradient(s)
    r = minus(a, g)
    assert inner(g, r) == 0
    assert all(d == 0 for d in divergence(r))
    assert boundary_flow(cone(r), len(a)) == r
    return s, g, r


def gradient_share(a: Matrix) -> Fraction:
    """The share of the squared norm of a that its gradient part carries."""
    _, g, _ = decompose(a)
    return inner(g, g) / inner(a, a)


def share_in_k(c: Chart = gen6) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    """The gradient share as a ratio of quadratics in k, where an immunity counts as -k.

    A = A0 + k·B, with A0 the flow at k = 0 and B the immunity pattern. The gradient map is
    linear, so the share is (p0 + p1 k + p2 k²) / (q0 + q1 k + q2 k²); returns (p, q).
    """
    a0 = flow(0, c)
    b = minus(flow(-1, c), a0)
    g0, gb = gradient(score(a0)), gradient(score(b))
    p = (inner(g0, g0), 2 * inner(g0, gb), inner(gb, gb))
    q = (inner(a0, a0), 2 * inner(a0, b), inner(b, b))
    return p, q


def ranking(names: Sequence[str], s: list[Fraction]) -> list[tuple[str, Fraction]]:
    """Types by score, highest first; ties in chart order."""
    return sorted(zip(names, s, strict=True), key=lambda t: -t[1])


def transitive_game_winner(s: list[Fraction]) -> list[int]:
    """In the game whose payoff is the gradient part alone, the pure strategies that are optimal.

    The row payoff s_i - s_j is largest for the largest s_i whatever j is, so the top-scored
    types dominate.
    """
    top = max(s)
    return [i for i, x in enumerate(s) if x == top]


def main() -> None:
    for name, c in CHARTS.items():
        types = list(c)
        print(f"== Generation {name} chart ({len(types)} types) ==")
        for immune in IMMUNE:
            a = flow(immune, c)
            s, _, _ = decompose(a)
            share = gradient_share(a)
            top = ", ".join(f"{t} {float(x):+.3f}" for t, x in ranking(types, s)[:3])
            low = ", ".join(f"{t} {float(x):+.3f}" for t, x in ranking(types, s)[-2:])
            print(
                f"  k={-immune:>2}: ranking part {share} = {float(share):.4f}; top {top}; "
                f"bottom {low}"
            )
        p, q = share_in_k(c)
        print(f"  as k grows the share tends to {p[2] / q[2]} = {float(p[2] / q[2]):.4f}")
        print()


if __name__ == "__main__":
    main()
