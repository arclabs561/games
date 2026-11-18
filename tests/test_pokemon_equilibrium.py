"""Equilibria of the type-chart games, and the super-effective triangles."""

from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from pokemon.equilibrium import (
    CHARTS,
    TYPES,
    TYPINGS,
    attack_matrix,
    best_multiplier,
    duel_matrix,
    duel_score,
    exact_equilibrium,
    existing_typings,
    generation_6_split,
    has_unique_certificate,
    is_equilibrium,
    is_strict,
    is_unique,
    log_ratio,
    log_ratio_matrix,
    lp_strategy,
    multiplier,
    never_played,
    optimal_set_dimension,
    optimal_vertices,
    optimal_weight_bound,
    payoffs_of_rows,
    solve_linear,
    spread,
    super_effective,
    support,
    transpose_negated,
    triangles,
    typings,
    weight_ranges,
)
from pokemon.pokemon_types import chart

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
    # The second method: every type's least and greatest optimal weight coincide, exactly.
    g = duel_matrix()
    x, _, v = exact_equilibrium(g)
    assert weight_ranges(g, v) == [(p, p) for p in x]


def test_weight_bounds_by_hand():
    # Every mix is optimal in the zero game; in matching pennies only the even mix is.
    zero = [[Fraction(0)] * 2 for _ in range(2)]
    assert weight_ranges(zero, Fraction(0)) == [(0, 1), (0, 1)]
    pennies = [[Fraction(1), Fraction(-1)], [Fraction(-1), Fraction(1)]]
    half = Fraction(1, 2)
    assert weight_ranges(pennies, Fraction(0)) == [(half, half), (half, half)]
    value, x = optimal_weight_bound(RPS, Fraction(0), [Fraction(1), Fraction(2), Fraction(0)])
    assert value == 1 and x == THIRD


def test_no_certificate_when_not_unique():
    g = log_ratio_matrix(-2)
    assert not has_unique_certificate(g, *exact_equilibrium(g))
    ghost = TYPES.index("ghost")
    assert weight_ranges(g, Fraction(0))[ghost] == (0, Fraction(5, 79))


# The log-ratio game, exactly: the types never played, and how far Ghost's weight can go.
LOG_RATIO = {
    -1: (["normal", "ice", "poison", "flying", "psychic", "bug", "rock"], None),
    -2: (
        ["normal", "ice", "fighting", "poison", "psychic", "bug", "rock", "dark"],
        Fraction(5, 79),
    ),
    -3: (["normal", "fighting", "poison", "psychic", "bug", "dark"], Fraction(154, 1915)),
    -4: (["normal", "fighting", "poison", "psychic", "bug", "dark"], Fraction(142, 2431)),
    -6: (["normal", "fighting", "poison", "psychic", "bug", "dark"], Fraction(118, 3583)),
    -10: (
        ["normal", "grass", "ice", "fighting", "poison", "psychic", "bug", "dark"],
        Fraction(9, 557),
    ),
}


@pytest.mark.parametrize("immune", sorted(LOG_RATIO, reverse=True))
def test_log_ratio_optimal_set(immune):
    never, ghost_max = LOG_RATIO[immune]
    g = log_ratio_matrix(immune)
    v = Fraction(0)
    assert never_played(TYPES, g, v) == never
    vertices = optimal_vertices(g, v)
    if ghost_max is None:
        assert len(vertices) == 1 and optimal_set_dimension(g, v, vertices) == 0
        return
    # A segment: from a mix a without Ghost to (1 - t) a + t Ghost, with t = ghost_max.
    assert optimal_set_dimension(g, v, vertices) == 1
    ghost = TYPES.index("ghost")
    a, b = sorted(vertices, key=lambda x: x[ghost])
    assert a[ghost] == 0 and b[ghost] == ghost_max
    t = ghost_max
    assert b == [(1 - t) * a[i] + t * (i == ghost) for i in range(18)]
    assert payoffs_of_rows(g, a)[ghost] == 0  # Ghost ties the mix without it
    assert not is_unique(g, v)


# Each chart. Generation 1 has 15 types, Generations 2 to 5 have 17, and 6 on have 18.


def test_generation_1_attack_game():
    c = CHARTS["1"]
    g = attack_matrix(None, c)
    x, y, v = exact_equilibrium(g)
    assert v == 1
    # Nothing resists or is immune to Dragon in Generation 1, so Dragon alone guarantees 1.
    assert min(g[list(c).index("dragon")]) == 1
    assert {t: q for t, q in support(list(c), y).items()} == {
        "ghost": Fraction(1, 2),
        "normal": Fraction(1, 4),
        "psychic": Fraction(1, 4),
    }
    found = spread(g, v)
    assert found is not None
    side, i, lo, hi = found
    assert (side, list(c)[i], lo, hi) == ("row", "fire", 0, Fraction(32, 97))


def test_generation_1_duel():
    c = CHARTS["1"]
    g = duel_matrix(None, c)
    x, y, v = exact_equilibrium(g)
    assert has_unique_certificate(g, x, y, v)
    w = support(list(c), x)
    assert list(w)[:3] == ["dragon", "psychic", "rock"]
    assert w["dragon"] == Fraction(127, 477) and w["psychic"] == Fraction(94, 477)
    assert sorted(set(c) - set(w)) == ["electric", "fire", "ghost", "grass", "normal", "poison"]


def test_generations_2_to_5():
    c = CHARTS["2-5"]
    t = list(c)
    g = attack_matrix(None, c)
    x, y, v = exact_equilibrium(g)
    assert v == Fraction(52363, 49845) and has_unique_certificate(g, x, y, v)
    assert list(support(t, y))[:2] == ["dark", "ghost"]
    g = duel_matrix(None, c)
    x, y, v = exact_equilibrium(g)
    assert has_unique_certificate(g, x, y, v)
    w = support(t, x)
    assert w["dragon"] == Fraction(62, 271) and w["steel"] == Fraction(48, 271)
    assert sorted(set(t) - set(w)) == sorted(
        ["normal", "fire", "grass", "poison", "psychic", "bug", "rock", "dark"]
    )
    g = duel_matrix(typings(c), c)
    assert has_unique_certificate(g, *exact_equilibrium(g))


@pytest.mark.parametrize("immune", [-1, -2, -3, -4, -6, -10])
def test_log_ratio_unique_before_generation_6(immune):
    for name in ("1", "2-5"):
        g = log_ratio_matrix(immune, CHARTS[name])
        assert has_unique_certificate(g, *exact_equilibrium(g))


def test_generation_6_changes_one_at_a_time():
    split = generation_6_split()
    steel, fairy = split["steel change"], split["fairy"]
    assert len(steel) == 17 and len(fairy) == 18
    assert (
        multiplier("ghost", ["steel"], steel) == 1 and multiplier("ghost", ["steel"], fairy) == 0.5
    )
    # The Steel change alone puts Dark first in the duel; Fairy alone keeps Steel first.
    g = duel_matrix(None, steel)
    x, _, _ = exact_equilibrium(g)
    assert next(iter(support(list(steel), x))) == "dark"
    g = duel_matrix(None, fairy)
    x, _, _ = exact_equilibrium(g)
    assert next(iter(support(list(fairy), x))) == "steel"
    # The log-ratio optimal set first stops being a point with Fairy (at k = 1).
    g = log_ratio_matrix(-1, fairy)
    assert spread(g, Fraction(0)) is not None
    for immune in (-1, -2):
        g = log_ratio_matrix(immune, steel)
        assert has_unique_certificate(g, *exact_equilibrium(g))


# The typings that exist in each generation, each with its own chart.

LEAST_DAMAGE = {
    1: (14, ["psychic", "electric/flying"]),
    **{g: (12, ["flying/steel"]) for g in (2, 3, 4, 5)},
    **{g: (Fraction(53, 4), ["steel/fairy"]) for g in (6, 7, 8, 9)},
}


@pytest.mark.parametrize("gen", sorted(LEAST_DAMAGE))
def test_least_damage_by_generation(gen):
    c = chart(gen)
    damage = {t: sum(multiplier(a, t, c) for a in c) for t in existing_typings(gen)}
    least = min(damage.values())
    assert (least, ["/".join(t) for t in damage if damage[t] == least]) == LEAST_DAMAGE[gen]


DUEL_LEADERS = {
    1: (13, ("ice", "psychic")),
    4: (15, ("water", "ground")),
    5: (19, ("water", "ground")),
    6: (21, ("water", "ground")),
    7: (27, ("water", "ground")),
    8: (29, ("water", "ground")),
    9: (23, ("electric", "flying")),
}


@pytest.mark.parametrize("gen", sorted(DUEL_LEADERS))
def test_duel_over_existing_typings(gen):
    existing = existing_typings(gen)
    g = duel_matrix(existing, chart(gen))
    x, y, v = exact_equilibrium(g)
    assert has_unique_certificate(g, x, y, v)
    w = support(existing, x)
    assert (len(w), next(iter(w))) == DUEL_LEADERS[gen]


def test_water_dragon_never_loses_in_generations_2_and_3():
    for gen in (2, 3):
        existing = existing_typings(gen)
        g = duel_matrix(existing, chart(gen))
        row = g[existing.index(("water", "dragon"))]
        assert min(row) == 0  # a pure optimal strategy, since the value is 0
        assert spread(g, Fraction(0)) is not None  # and not the only one
    # Dragon/Steel arrives in Generation 4 and beats it.
    existing = existing_typings(4)
    g = duel_matrix(existing, chart(4))
    i, j = existing.index(("water", "dragon")), existing.index(("dragon", "steel"))
    assert g[i][j] < 0
