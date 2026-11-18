"""The Mastermind strategies reproduce the published distributions."""

from collections import Counter

from mastermind import strategies as mm

# B. Kooi, "Yet another Mastermind strategy", ICGA Journal 28(1), 2005, Table 7:
# how many of the 1,296 codes each strategy solves in 1, 2, 3, ... guesses.
KOOI_2005 = {
    mm.first_consistent: [1, 4, 25, 108, 305, 602, 196, 49, 6],
    mm.minimax: [1, 6, 62, 533, 694],
    mm.expected_size: [1, 10, 54, 645, 583, 3],
    mm.entropy: [1, 4, 71, 612, 596, 12],
    mm.most_parts: [1, 12, 72, 635, 569, 7],
}


def test_reply_is_symmetric_on_a_sample():
    for g in range(0, len(mm.CODES), 97):
        for s in range(0, len(mm.CODES), 89):
            assert mm.REPLIES[g][s] == mm.REPLIES[s][g]


def test_strategies_match_kooi_table_7():
    for strategy, published in KOOI_2005.items():
        dist = Counter(mm.guess_counts(strategy))
        assert [dist[k] for k in range(1, len(published) + 1)] == published, strategy.__name__
        assert sum(dist.values()) == len(mm.CODES)


def test_random_consistent_is_seeded():
    assert mm.random_consistent_counts(1) == mm.random_consistent_counts(1)


def test_pareto_front():
    points = {"a": (4.4, 6), "b": (4.5, 5), "c": (4.6, 6), "d": (4.4, 6)}
    assert mm.pareto_front(points) == {"a", "b", "d"}
