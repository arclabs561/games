"""Coverage and attack/defense scores (pokemon/scoring.py, which pokemon.ipynb imports)."""

from fractions import Fraction
from itertools import combinations

import pytest
from scipy.stats import percentileofscore

from pokemon import figures
from pokemon.pokemon_types import gen6
from pokemon.scoring import (
    CHART,
    SUPER,
    TYPES,
    best_coverage_scores,
    by_typing,
    conflict,
    coverage_score,
    covers,
    hits,
    incoming,
    multiplier,
    outgoing,
    percentile,
    pokemon_scores,
    pokemon_table,
    random_teams,
    raw_scores,
    rescaled,
    starter,
    typing_scores,
)


def test_chart_is_the_generation_6_chart_with_neutral_entries_written_out():
    assert len(TYPES) == 18
    for a in TYPES:
        for d in TYPES:
            assert CHART[a][d] == gen6[a].get(d, 1.0)
    assert sum(CHART[a][d] != 1 for a in TYPES for d in TYPES) == 120


def test_multiplier_of_a_pair_is_the_product():
    assert multiplier("ice", ("dragon", "flying")) == 4
    assert multiplier("electric", ("ground", "water")) == 0
    assert multiplier("fire", ("fire", "water")) == Fraction(1, 4)


def test_hits_unions_over_attackers():
    assert hits(["bug"], SUPER) == {"grass", "psychic", "dark"}
    assert hits(["normal", "fighting"], Fraction(0)) == {"ghost"}


def brute_covers(k):
    """Covers by the definition, without `hits`."""
    return [
        c
        for c in combinations(sorted(TYPES), k)
        if all(any(gen6[a].get(d, 1.0) == 2.0 for a in c) for d in TYPES)
    ]


def test_no_six_types_cover_and_ten_sets_of_seven_do():
    assert covers(6) == brute_covers(6) == []
    seven = covers(7)
    assert seven == brute_covers(7)
    assert len(seven) == 10
    assert all({"fighting", "ground"} <= set(s) for s in seven)


def test_coverage_score_by_hand_and_the_best_sets():
    # Ground: super-effective on 5 types, resisted by Grass and Bug, no effect on Flying.
    assert coverage_score(["ground"]) == 10 - 2 - 2
    top, best = best_coverage_scores(7)
    assert top == 18 and len(best) == 16
    assert [len(s) for s in best].count(6) == 4
    assert all(coverage_score(s) == 18 for s in best)


def test_outgoing_incoming_and_conflict():
    assert outgoing(("ghost", "ice"))["normal"] == 1
    assert min(outgoing(("ground", "ice")).values()) == 1
    assert incoming(("fairy", "ice"))["steel"] == 4
    assert sum(incoming(("fairy", "steel")).values()) == Fraction(53, 4)
    assert conflict(("fighting", "ice")) == Fraction(29, 2)
    assert conflict(("water",)) == 0
    assert raw_scores(("normal",)) == (
        sum(CHART["normal"].values()),
        -sum(incoming(("normal",)).values()),
    )


def test_rescaled_and_percentile_match_their_definitions():
    assert rescaled([Fraction(1), Fraction(3), Fraction(2)]) == [0, 1, Fraction(1, 2)]
    values = [1, 2, 2, 4, 7, 7, 7]
    for x in (0, 1, 2, 3, 7, 8):
        assert percentile(values, x) == pytest.approx(percentileofscore(values, x))


@pytest.fixture(scope="module")
def gen7():
    table = pokemon_table(7)
    groups = by_typing(table)
    scores = typing_scores(list(groups))
    return table, groups, scores, pokemon_scores(table, scores)


def test_the_notebook_numbers(gen7):
    table, groups, scores, poke = gen7
    assert (len(table), len(groups)) == (955, 146)
    best_defense = max(scores, key=lambda t: scores[t][1])
    assert best_defense == ("fairy", "steel") and scores[best_defense][1] == 1
    assert round(poke["blissey"][0], 1) == 83.2
    assert [round(p, 1) for p in poke["toxapex"]] == [66.9, 81.2]
    combined = {name: s + c for name, (s, c) in poke.items()}
    assert round(percentile(list(combined.values()), combined["toxapex"]), 1) == 90.2
    starters = [r for r in table if starter(r["ndex"], final=True) and not r["mega"]]
    assert max(starters, key=lambda r: combined[r["name"]])["name"] == "empoleon"


def test_starter_blocks():
    assert [n for n in range(1, 10) if starter(n)] == list(range(1, 10))
    assert [n for n in range(720, 735) if starter(n, final=True)] == [724, 727, 730]
    assert not starter(10) and not starter(151)


def test_random_teams_are_seeded_means(gen7):
    _, groups, _, _ = gen7
    teams = random_teams(list(groups), 5, seed=1)
    assert teams == random_teams(list(groups), 5, seed=1)
    lone = random_teams([("water",)], 1, seed=0, size=1)
    assert lone == [raw_scores(("water",))]


def test_committed_svgs_are_what_the_script_writes(tmp_path):
    figures.main(tmp_path)
    for name in figures.FIGURES:
        assert (tmp_path / name).read_bytes() == (figures.HERE / "figures" / name).read_bytes(), (
            name
        )
