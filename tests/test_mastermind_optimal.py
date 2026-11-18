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
