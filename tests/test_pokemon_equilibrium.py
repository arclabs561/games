"""Equilibria of the type-chart games, and the super-effective triangles."""

from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from pokemon.equilibrium import (
    TYPES,
    TYPINGS,
    attack_matrix,
    best_multiplier,
    duel_matrix,
    duel_score,
    exact_equilibrium,
    existing_typings,
    has_unique_certificate,
    is_equilibrium,
    is_strict,
    is_unique,
    log_ratio,
    log_ratio_matrix,
    lp_strategy,
    multiplier,
    never_played,
    payoffs_of_rows,
    solve_linear,
    super_effective,
    support,
    transpose_negated,
    triangles,
    weight_ranges,
)

RPS = [
    [Fraction(0), Fraction(-1), Fraction(1)],
    [Fraction(1), Fraction(0), Fraction(-1)],
    [Fraction(-1), Fraction(1), Fraction(0)],
]
THIRD = [Fraction(1, 3)] * 3


def test_multipliers():
    assert multiplier("water", ["fire"]) == 2
    assert multiplier("ice", ["ground", "flying"]) == 4
    assert multiplier("fire", ["water", "rock"]) == Fraction(1, 4)
    assert multiplier("electric", ["water", "ground"]) == 0
    # Water/ground hits dragon/fire with ground (2 x 1), not water (1/2 x 2).
    assert best_multiplier(["water", "ground"], ["dragon", "fire"]) == 2
    assert best_multiplier(["water", "ground"], ["fire", "electric"]) == 4


def test_typings():
    assert len(TYPINGS) == 18 + 18 * 17 // 2 == 171


def test_duel_score_matches_schmitz_examples():
    # Water against fire: water deals 2, takes 1/2, and wins with 3/4 of its HP.
    assert duel_score(Fraction(2), Fraction(1, 2)) == Fraction(3, 4)
    assert duel_score(Fraction(1, 2), Fraction(2)) == Fraction(-3, 4)
    # Ground against electric: ground takes nothing, so it keeps all its HP.
    assert duel_score(Fraction(2), Fraction(0)) == 1
    # Double damage against normal damage leaves half.
    assert duel_score(Fraction(2), Fraction(1)) == Fraction(1, 2)
    assert duel_score(Fraction(0), Fraction(0)) == duel_score(Fraction(1, 2), Fraction(1, 2)) == 0


def test_log_ratio():
    assert log_ratio(Fraction(2), Fraction(1, 2), immune=-2) == 2
    assert log_ratio(Fraction(1), Fraction(0), immune=-2) == 2
    assert log_ratio(Fraction(1, 4), Fraction(4), immune=-3) == -4
    assert all(e.denominator == 1 for row in log_ratio_matrix(-3) for e in row)


@pytest.mark.parametrize(
    "g", [duel_matrix(), log_ratio_matrix(-1), log_ratio_matrix(-4)], ids=["duel", "log1", "log4"]
)
def test_symmetric_games_are_skew(g):
    assert g == transpose_negated(g)


def test_solve_linear():
    # x + y = 3, x - y = 1.
    assert solve_linear(
        [[Fraction(1), Fraction(1)], [Fraction(1), Fraction(-1)]], [Fraction(3), Fraction(1)]
    ) == [2, 1]
    assert (
        solve_linear(
            [[Fraction(1), Fraction(1)], [Fraction(2), Fraction(2)]], [Fraction(1), Fraction(3)]
        )
        is None
    )


def test_is_equilibrium_on_rock_paper_scissors():
    assert is_equilibrium(RPS, THIRD, THIRD, Fraction(0))
    assert not is_equilibrium(RPS, [Fraction(1), Fraction(0), Fraction(0)], THIRD, Fraction(0))
    assert not is_equilibrium(RPS, THIRD, THIRD, Fraction(1, 100))


def test_small_games_by_hand():
    assert exact_equilibrium(RPS) == (THIRD, THIRD, 0)
    # Rows (3, -1) and (-1, 1). The row mix p solves 3p - (1-p) = -p + (1-p), so p = 1/3,
    # and the column mix is the same by symmetry of g.
    g = [[Fraction(3), Fraction(-1)], [Fraction(-1), Fraction(1)]]
    x, y, v = exact_equilibrium(g)
    assert x == y == [Fraction(1, 3), Fraction(2, 3)] and v == Fraction(1, 3)
    # g dominated row gets no weight: row 2 is row 1 minus one. The column player then
    # takes column 1, where row 1 pays 0.
    x, y, v = exact_equilibrium([[Fraction(0), Fraction(1)], [Fraction(-1), Fraction(0)]])
    assert x == y == [1, 0] and v == 0


def agrees_with_lp(g, x, v):
    xf, vf = lp_strategy(g)
    return np.allclose(xf, [float(p) for p in x], atol=1e-9) and abs(vf - float(v)) < 1e-9


def test_attack_game():
    g = attack_matrix()
    x, y, v = exact_equilibrium(g)
    assert is_equilibrium(g, x, y, v)
    assert v == Fraction(4158, 3979)
    assert is_unique(g, v)
    assert agrees_with_lp(g, x, v)
    assert list(support(TYPES, x))[:2] == ["ground", "ice"]
    assert list(support(TYPES, y))[:3] == ["normal", "fairy", "ghost"]
    assert len(support(TYPES, x)) == len(support(TYPES, y)) == 11


def test_attack_game_against_typings():
    g = attack_matrix(TYPINGS)
    x, y, v = exact_equilibrium(g)
    assert is_equilibrium(g, x, y, v)
    assert (
        v == Fraction(13207, 13501) < 1
    )  # a typed defender holds the attacker under neutral damage
    assert len(support(TYPES, x)) == 12
    assert next(iter(support(TYPINGS, y))) == ("flying", "steel")


# Schmitz (2021), "The Objectively Best Pokemon Type, According to Game Theory", to 3 places.
SCHMITZ_2021 = {
    "steel": 0.236,
    "dragon": 0.175,
    "water": 0.161,
    "ground": 0.127,
    "electric": 0.083,
    "grass": 0.070,
    "flying": 0.068,
    "ghost": 0.040,
    "fairy": 0.038,
    "dark": 0.002,
    "fighting": 0.001,
}


def test_duel_game_reproduces_schmitz():
    g = duel_matrix()
    x, y, v = exact_equilibrium(g)
    assert is_equilibrium(g, x, y, v) and v == 0 and x == y
    assert is_unique(g, v)
    assert agrees_with_lp(g, x, v)
    weights = support(TYPES, x)
    assert weights["steel"] == Fraction(627, 2653)
    assert {t: round(float(q), 3) for t, q in weights.items()} == SCHMITZ_2021
    assert sorted(never_played(TYPES, g, v)) == sorted(
        ["normal", "fire", "poison", "psychic", "bug", "rock", "ice"]
    )
    # Strict: every type left out loses against the equilibrium.
    pay = payoffs_of_rows(g, y)
    assert all(pay[i] < 0 for i in range(18) if x[i] == 0)


def test_duel_game_over_typings():
    g = duel_matrix(TYPINGS)
    x, y, v = exact_equilibrium(g)
    assert is_equilibrium(g, x, y, v) and v == 0
    weights = support(TYPINGS, x)
    assert len(weights) == 23
    assert all(len(t) == 2 for t in weights)  # no single type is played
    assert list(weights)[:3] == [("electric", "flying"), ("grass", "rock"), ("water", "ground")]
    # Schmitz (2023) ranks water/flying and poison/dark first; here both lose to the mix.
    pay = dict(zip(TYPINGS, payoffs_of_rows(g, y), strict=True))
    assert pay["water", "flying"] < 0 and pay["poison", "dark"] < 0
    assert is_unique(g, v)


def test_duel_game_over_existing_typings():
    typings = existing_typings()
    assert len(typings) == 146
    g = duel_matrix(typings)
    x, y, v = exact_equilibrium(g)
    assert is_equilibrium(g, x, y, v) and has_unique_certificate(g, x, y, v)
    weights = support(typings, x)
    assert len(weights) == 27
    assert next(iter(weights)) == ("water", "ground")
    assert ("water", "flying") not in weights


def test_log_ratio_game_depends_on_immunities():
    g = log_ratio_matrix(-1)  # an immunity counted as a resistance
    _, _, v = exact_equilibrium(g)
    assert is_unique(g, v)
    assert sorted(never_played(TYPES, g, v)) == sorted(
        ["normal", "ice", "poison", "flying", "psychic", "bug", "rock"]
    )
    g = log_ratio_matrix(-2)  # an immunity counted as a double resistance
    x, y, v = exact_equilibrium(g)
    assert is_equilibrium(g, x, y, v)
    assert not is_unique(g, v)


def brute_triangles():
    """Every set of three types that some ordering makes a super-effective cycle."""
    g = nx.DiGraph([(a, b) for a in TYPES for b in TYPES if super_effective(a, b)])
    return {frozenset(c) for c in nx.simple_cycles(g, length_bound=3) if len(c) == 3}


def test_triangles():
    found = triangles()
    assert len(found) == 16
    assert {frozenset(c) for c in found} == brute_triangles()
    assert ("fire", "grass", "water") in found
    strict = [c for c in found if is_strict(c)]
    assert strict == [
        ("fire", "grass", "water"),
        ("fire", "steel", "rock"),
        ("grass", "ground", "poison"),
        ("fighting", "rock", "flying"),
        ("fighting", "dark", "psychic"),
    ]


def test_no_two_types_hit_each_other_super_effectively():
    mutual = {
        (a, b) for a in TYPES for b in TYPES if super_effective(a, b) and super_effective(b, a)
    }
    assert mutual == {("dragon", "dragon"), ("ghost", "ghost")}


@pytest.mark.parametrize(
    "game",
    [
        lambda: attack_matrix(),
        lambda: attack_matrix(TYPINGS),
        lambda: duel_matrix(),
        lambda: duel_matrix(TYPINGS),
        lambda: log_ratio_matrix(-1),
    ],
    ids=["attack", "attack-typings", "duel", "duel-typings", "log1"],
)
def test_uniqueness_has_an_exact_proof(game):
    g = game()
    assert has_unique_certificate(g, *exact_equilibrium(g))


def test_certificate_agrees_with_lp_ranges():
    # The second method: every type's least and greatest optimal weight coincide.
    g = duel_matrix()
    x, _, v = exact_equilibrium(g)
    for (lo, hi), p in zip(weight_ranges(g, v), x, strict=True):
        assert abs(lo - float(p)) < 1e-9 and abs(hi - float(p)) < 1e-9


def test_no_certificate_when_not_unique():
    g = log_ratio_matrix(-2)
    assert not has_unique_certificate(g, *exact_equilibrium(g))
    ghost = TYPES.index("ghost")
    lo, hi = weight_ranges(g, Fraction(0))[ghost]
    assert lo < 1e-9 < 0.06 < hi
