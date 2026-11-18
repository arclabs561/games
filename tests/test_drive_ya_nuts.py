"""Drive-Ya-Nuts: the solver's counts against an independent brute force."""

import random
from collections.abc import Sequence
from itertools import permutations

from drive_ya_nuts.solver import Nut, count_solutions, hoeffding_trials, nuts


def brute_force(pool: Sequence[Sequence[int]]) -> int:
    """Count solutions with explicit hexagon geometry, sharing no code with the solver.

    Directions 0..5 go counterclockwise. The center nut is turned so that its first value
    faces direction 0; that fixes the rotation of the whole board. The outer nut in direction
    i must show the center's value on its side facing the center (direction i + 3), which
    fixes its rotation. Neighboring outer nuts i and i + 1 touch across direction i + 2.
    """
    count = 0
    for c, center in enumerate(pool):
        outer = [nut for j, nut in enumerate(pool) if j != c]
        for placed in permutations(outer):
            sides = []  # sides[i][d]: value of outer nut i on its side facing direction d
            for i, nut in enumerate(placed):
                r = (i + 3 - nut.index(center[i])) % 6
                sides.append([nut[(d - r) % 6] for d in range(6)])
            if all(sides[i][(i + 2) % 6] == sides[(i + 1) % 6][(i + 5) % 6] for i in range(6)):
                count += 1
    return count


def test_original_puzzle_has_one_solution():
    assert count_solutions(nuts) == 1
    assert brute_force([n.order for n in nuts]) == 1


def test_random_piece_sets_agree_with_brute_force():
    classes = [(1, *rest) for rest in permutations(range(2, 7))]  # one rotation per nut
    rng = random.Random(0)
    totals = []
    for _ in range(200):  # about 7% of random sets are solvable
        pool = rng.sample(classes, 7)
        expected = brute_force(pool)
        assert count_solutions({Nut(order) for order in pool}) == expected, pool
        totals.append(expected)
    assert max(totals) >= 2  # the sample includes sets with one and with several solutions


def test_hoeffding_trials():
    assert hoeffding_trials(epsilon=0.01, alpha=0.1) == 14979  # as in the Boggle notebook
    assert hoeffding_trials(epsilon=0.01, alpha=0.1, m=10) > 14979
