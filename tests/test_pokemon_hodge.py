"""The ranking and cyclic parts of the log-ratio chart."""

from fractions import Fraction
from itertools import combinations

import numpy as np
import pytest

from pokemon.equilibrium import CHARTS, TYPES, is_equilibrium
from pokemon.hodge import (
    boundary_flow,
    cone,
    curl,
    decompose,
    divergence,
    flow,
    gradient,
    gradient_share,
    inner,
    ranking,
    score,
    share_in_k,
    transitive_game_winner,
)

F = Fraction


def matrix(rows):
    return [[F(x) for x in r] for r in rows]


RPS = matrix([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])


def test_small_flows_by_hand():
    # Rock-paper-scissors is one cycle: no ranking at all.
    assert score(RPS) == [0, 0, 0] and gradient_share(RPS) == 0
    assert curl(RPS, (0, 1, 2)) == 3
    # A flow that is a ranking already: scores 1, 0, -1.
    ranked = gradient([F(1), F(0), F(-1)])
    assert score(ranked) == [1, 0, -1] and gradient_share(ranked) == 1
    # Their sum splits back into the two.
    s, g, r = decompose(
        [[a + b for a, b in zip(x, y, strict=True)] for x, y in zip(RPS, ranked, strict=True)]
    )
    assert g == ranked and r == RPS


def test_score_is_least_squares():
    """Against numpy's least squares over the 153 pairs, for each chart and two values of k."""
    for c in CHARTS.values():
        for immune in (-1, -2):
            a = flow(immune, c)
            n = len(a)
            pairs = list(combinations(range(n), 2))
            d = np.zeros((len(pairs), n))
            for row, (i, j) in enumerate(pairs):
                d[row, i], d[row, j] = 1, -1
            b = np.array([float(a[i][j]) for i, j in pairs])
            fit = np.linalg.lstsq(d, b, rcond=None)[0]
            fit -= fit.mean()
            assert np.allclose(fit, [float(x) for x in score(a)], atol=1e-12)


def test_no_harmonic_part():
    a = flow(-2)
    _, _, r = decompose(a)
    assert all(x == 0 for x in divergence(r))
    weights = cone(r)
    assert boundary_flow(weights, 18) == r
    assert all(t[0] == 0 for t in weights)  # triangles through Normal alone suffice
    # The cyclic part keeps every triangle's circulation; the ranking part has none.
    g = gradient(score(a))
    for t in combinations(range(18), 3):
        assert curl(g, t) == 0 and curl(r, t) == curl(a, t)


@pytest.mark.parametrize(
    "chart, immune, share",
    [
        ("6-9", -1, F(14, 297)),
        ("6-9", -2, F(89, 1701)),
        ("6-9", -3, F(118, 2025)),
        ("6-9", -4, F(157, 2457)),
        ("6-9", -6, F(53, 729)),
        ("6-9", -10, F(601, 7317)),
        ("2-5", -1, F(128, 2465)),
        ("2-5", -2, F(40, 697)),
        ("1", -1, F(22, 515)),
        ("1", -2, F(86, 1755)),
    ],
)
def test_gradient_share(chart, immune, share):
    assert gradient_share(flow(immune, CHARTS[chart])) == share


def test_share_as_k_grows():
    for name, limit in (("1", F(1, 15)), ("2-5", F(8, 85)), ("6-9", F(5, 54))):
        c = CHARTS[name]
        p, q = share_in_k(c)
        assert p[2] / q[2] == limit
        for k in (1, 2, 3, 10):
            value = (p[0] + p[1] * k + p[2] * k * k) / (q[0] + q[1] * k + q[2] * k * k)
            assert value == gradient_share(flow(-k, c))
    # The share rises with k on the current chart, from 14/297 toward 5/54, below 10%.
    shares = [gradient_share(flow(-k)) for k in (1, 2, 3, 4, 6, 10)]
    assert shares == sorted(shares) and shares[-1] < F(5, 54) < F(1, 10)


def test_steel_ranks_first():
    for immune in (-1, -2, -3, -4, -6, -10):
        for name in ("2-5", "6-9"):
            c = CHARTS[name]
            s = score(flow(immune, c))
            assert ranking(list(c), s)[0][0] == "steel"
            assert [list(c)[i] for i in transitive_game_winner(s)] == ["steel"]
    # Steel's score is (6 + k)/18: it hits 3 types super-effectively and 4 weakly (net -1), and
    # takes 3 super-effective, 10 resisted and 1 immune hit (net -7 - k).
    for k in (1, 2, 10):
        assert score(flow(-k))[TYPES.index("steel")] == F(6 + k, 18)
    assert ranking(TYPES, score(flow(-1)))[-1] == ("grass", F(-5, 18))


def test_generation_1_ranks_psychic_first():
    c = CHARTS["1"]
    for immune in (-2, -3, -10):
        assert ranking(list(c), score(flow(immune, c)))[0][0] == "psychic"
    # With k = 1, Psychic and Dragon tie.
    s = score(flow(-1, c))
    assert [list(c)[i] for i in transitive_game_winner(s)] == ["psychic", "dragon"]


def test_the_parts_as_games():
    """The ranking part alone is won by pure Steel; the cyclic part alone by the uniform mix."""
    s, g, r = decompose(flow(-2))
    n = len(s)
    steel = [F(int(t == "steel")) for t in TYPES]
    assert is_equilibrium(g, steel, steel, F(0))
    uniform = [F(1, n)] * n
    assert is_equilibrium(r, uniform, uniform, F(0))
    assert inner(g, r) == 0
