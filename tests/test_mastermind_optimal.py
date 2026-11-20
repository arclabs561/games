"""The exhaustive search for the best Mastermind strategy (mastermind/optimal.py)."""

import pytest

from mastermind.optimal import INF, Game, brute_force_cost, play

SMALL = [(2, 2), (3, 2), (2, 3), (4, 2), (2, 4), (3, 3)]


@pytest.mark.parametrize(("colors", "holes"), SMALL)
def test_search_matches_brute_force(colors, holes):
    """Same least cost as the plain recursion, with and without each depth limit."""
    game = Game(colors, holes)
    for limit in (12, holes + 1, holes, holes - 1):
        assert game.cost(game.all_codes(), limit) == brute_force_cost(colors, holes, limit)


@pytest.mark.parametrize(("colors", "holes"), [*SMALL, (4, 3), (5, 3)])
def test_tree_attains_the_cost(colors, holes):
    """The tree read back from the search, played with `score`, costs what the search says."""
    game = Game(colors, holes)
    for limit in (12, holes + 1):
        cost = game.cost(game.all_codes(), limit)
        if cost >= INF:
            continue
        counts = play(game.strategy_tree(limit), game.codes)
        assert sum(counts) == cost and max(counts) <= limit


def test_bounds_and_symmetries():
    game = Game(6, 4)
    assert len(game.maps) == 24 * 720 and len(game.reply_list) == 14
    assert game.lower[12][1] == 1 and game.lower[12][14] == 1 + 2 * 13
    assert game.lower[12][15] == 1 + 2 * 13 + 3 and game.lower[1][2] == INF
    least, stabilizer = game.canonical(game.all_codes())
    assert list(least) == list(range(game.n)) and len(stabilizer) == len(game.maps)
    # The five classes of first guess: 1111, 1112, 1122, 1123 and 1234.
    assert len(set(stabilizer.min(0).tolist())) == 5


def test_committed_optimal_trees():
    """The trees the full search wrote (results/optimal.json), played against every code."""
    import json
    from collections import Counter
    from pathlib import Path

    from mastermind.optimal import Game, summary, tree_from_json, tree_to_json

    path = Path(__file__).parent.parent / "mastermind" / "results" / "optimal.json"
    record = json.loads(path.read_text())
    codes = Game.codes_of(6, 4)
    assert (record["colors"], record["holes"], record["least_worst_case"]) == (6, 4, 5)
    totals = []
    for row in record["trees"]:
        tree = tree_from_json(row["tree"], codes)
        assert tree_to_json(tree, codes) == row["tree"]
        played = summary(tree, codes, row["limit"])
        assert {k: row[k] for k in played} == played
        assert len(Counter(play(tree, codes))) == row["worst"]
        totals.append((row["limit"], row["total"], row["worst"], row["first_guess"]))
    assert totals == [(12, 5625, 6, "1123"), (5, 5626, 5, "1123")]


def test_tree_json_round_trip_small_game():
    from mastermind.optimal import summary, tree_from_json, tree_to_json

    game = Game(3, 3)
    tree = game.strategy_tree()
    assert tree_from_json(tree_to_json(tree, game.codes), game.codes) == tree
    assert summary(tree, game.codes, 12)["total"] == game.cost(game.all_codes())


# W. Goddard, "Mastermind revisited", J. Combin. Math. Combin. Comput. 51 (2004), the table of
# r(n, k): the least average number of guesses, to three decimals, keyed (colors, pegs).
GODDARD_AVERAGE = {
    (2, 2): 2.000,
    (2, 3): 2.250,
    (2, 4): 2.750,
    (2, 5): 3.031,
    (2, 6): 3.500,
    (3, 2): 2.333,
    (3, 3): 2.704,
    (3, 4): 3.037,
    (3, 5): 3.358,
    (4, 2): 2.813,
    (4, 3): 3.219,
    (4, 4): 3.535,
    (5, 2): 3.240,
    (5, 3): 3.608,
    (6, 2): 3.667,
    (6, 3): 3.954,
    (7, 2): 4.041,
}
# The least worst case for the same games. Goddard's table agrees except in its two-peg column
# at odd numbers of colors, where it is one higher than his own Theorem 1, f(2, k) = k // 2 + 2;
# the search agrees with the theorem.
LEAST_WORST_CASE = {
    (2, 2): 3,
    (2, 3): 3,
    (2, 4): 4,
    (2, 5): 4,
    (2, 6): 5,
    (3, 2): 3,
    (3, 3): 4,
    (3, 4): 4,
    (3, 5): 4,
    (4, 2): 4,
    (4, 3): 4,
    (4, 4): 4,
    (5, 2): 4,
    (5, 3): 5,
    (6, 2): 5,
    (6, 3): 5,
    (7, 2): 5,
}


@pytest.mark.parametrize(("colors", "holes"), sorted(GODDARD_AVERAGE))
def test_published_small_games(colors, holes):
    """The least average matches Goddard's table, and the least worst case his Theorem 1."""
    game = Game(colors, holes)
    codes = game.all_codes()
    assert abs(game.cost(codes, 12) / game.n - GODDARD_AVERAGE[colors, holes]) <= 0.0005 + 1e-9
    least = LEAST_WORST_CASE[colors, holes]
    assert game.cost(codes, least) < INF and game.cost(codes, least - 1) >= INF
    if holes == 2:
        assert least == colors // 2 + 2
