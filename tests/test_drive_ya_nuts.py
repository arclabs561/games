"""Drive-Ya-Nuts: the solver's counts against an independent brute force."""

import random
from collections import Counter
from collections.abc import Sequence
from itertools import combinations, permutations
from math import comb, factorial

import numpy as np

from drive_ya_nuts.classes import (
    action,
    burnside_classes,
    class_distribution,
    count_rings,
    cycle_type,
    direct_classes,
    fixed_count,
    fixed_sets,
    read_classes,
    renamings,
    stabilizer,
    stabilizer_orders,
)
from drive_ya_nuts.exact import (
    Order,
    all_orders,
    candidates,
    check_sets,
    edge_sequences,
    exact_distribution,
    ranks_without_c0,
    read_csv,
    rings_around_c0,
    run_lengths,
    sorted_ranks,
)
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


# The exact distribution over all sets (drive_ya_nuts/exact.py), checked on nuts with fewer
# sides, where every set can be solved directly.


def ring_count(pool: Sequence[Sequence[int]], n: int) -> int:
    """Count solutions for nuts with n sides by trying every center and order of outer nuts.

    The center reads c_0..c_{n-1}, and outer nut i faces it with c_i. Outer nuts i and i + 1
    touch where nut i shows the number before c_i and nut i + 1 the number after c_{i+1}
    (reading counterclockwise), which is the hexagon rule of `brute_force` above.
    """
    count = 0
    for c, center in enumerate(pool):
        outer = [nut for j, nut in enumerate(pool) if j != c]
        for placed in permutations(outer):
            before = [nut[nut.index(center[i]) - 1] for i, nut in enumerate(placed)]
            after = [nut[(nut.index(center[i]) + 1) % n] for i, nut in enumerate(placed)]
            count += all(before[i] == after[(i + 1) % n] for i in range(n))
    return count


def all_rings(n: int) -> Counter[frozenset[Order]]:
    """Solutions of every set of n + 1 nuts with n sides, found center by center, without renaming."""
    nuts = all_orders(n)
    found: Counter[frozenset[Order]] = Counter()

    def extend(center, placed):
        i = len(placed)
        if i == n:
            closes = placed[-1][placed[-1].index(center[-1]) - 1]
            if closes == placed[0][(placed[0].index(center[0]) + 1) % n]:
                found[frozenset([center, *placed])] += 1
            return
        for nut in nuts:
            if nut == center or nut in placed:
                continue
            j = nut.index(center[i])
            if i and nut[(j + 1) % n] != placed[-1][placed[-1].index(center[i - 1]) - 1]:
                continue
            extend(center, [*placed, nut])

    for center in nuts:
        extend(center, [])
    return found


def distribution(found: Counter[frozenset[Order]], n: int) -> dict[int, int]:
    hist = Counter(found.values())
    hist[0] = comb(factorial(n - 1), n + 1) - len(found)
    return dict(sorted(hist.items()))


def test_ring_rule_is_the_hexagon_rule():
    rng = random.Random(1)
    for _ in range(40):
        pool = rng.sample(all_orders(6), 7)
        assert ring_count(pool, 6) == brute_force(pool), pool


def test_exact_four_sides():
    # Six nuts, and each of the six sets of five has two solutions.
    sets = list(combinations(all_orders(4), 5))
    assert Counter(ring_count(s, 4) for s in sets) == {2: 6}
    assert exact_distribution(4) == {0: 0, 2: 6}


def test_exact_five_sides():
    found = all_rings(5)
    assert exact_distribution(5) == distribution(found, 5)
    assert exact_distribution(5) == {
        0: 120064, 1: 9264, 2: 2904, 3: 1200, 4: 420, 5: 480, 6: 120, 7: 24, 10: 120
    }  # fmt: skip
    rng = random.Random(2)
    for s in rng.sample(list(combinations(all_orders(5), 6)), 300):
        assert ring_count(s, 5) == found[frozenset(s)], s


def test_edge_sequences_and_candidates():
    assert sum(1 for _ in edge_sequences(6)) == 1144
    for e in edge_sequences(5):
        for i in range(5):
            for nut in candidates(5, i, e[i - 1], e[i]):
                j = nut.index(i + 1)
                assert (nut[(j + 1) % 5], nut[j - 1]) == (e[i - 1], e[i])


def test_ranks_number_the_sets_with_c0():
    sets = np.array([(0, *s) for s in combinations(range(1, 24), 5)], dtype=np.uint8)
    np.random.default_rng(0).permuted(sets, axis=1, out=sets)  # rows need not be sorted
    assert sorted(ranks_without_c0(sets).tolist()) == list(range(comb(23, 5)))


def test_run_lengths():
    rng = random.Random(3)
    for _ in range(200):
        values = sorted(rng.randrange(rng.randint(1, 20)) for _ in range(rng.randint(0, 50)))
        expected = Counter(Counter(values).values())
        assert run_lengths(np.array(values, np.uint32), block=rng.randint(1, 8)) == expected


def test_recorded_six_sided_counts():
    counts = read_csv()
    assert sum(counts.values()) == comb(120, 7)
    assert counts[1] == 3_180_020_160
    # Every solution is a ring around some center: 120 centers, each in the same number of rings.
    rings = len(rings_around_c0(6))
    assert rings == 43_500_008
    assert sum(k * v for k, v in counts.items()) == 120 * rings


def test_check_sets_five_sides():
    ranks = sorted_ranks(5)
    assert check_sets(ranks, 5, 200, 0, lambda pool: ring_count(pool, 5)) == 0
    fewer = np.delete(ranks, np.s_[::3])  # drop every third ring
    wrong = check_sets(fewer, 5, 200, 0, lambda pool: ring_count(pool, 5))
    assert wrong > 0


# Classes of sets up to renaming the numbers (drive_ya_nuts/classes.py): Burnside's lemma
# against a direct listing of every class for four and five sides.


def direct_orbits(n: int) -> list[list[frozenset[Order]]]:
    """Every class of sets of n + 1 nuts, found by renaming each unseen set in all n! ways."""
    nuts = all_orders(n)

    def renamed(g, nut):
        o = [g[v - 1] for v in nut]
        i = o.index(1)
        return tuple(o[i:] + o[:i])

    seen: set[frozenset[Order]] = set()
    orbits = []
    for s in combinations(nuts, n + 1):
        if frozenset(s) in seen:
            continue
        orbit = {frozenset(renamed(g, nut) for nut in s) for g in permutations(range(1, n + 1))}
        seen |= orbit
        orbits.append(sorted(orbit, key=sorted))
    return orbits


def test_classes_by_burnside_match_direct_listing():
    for n in (4, 5):
        found = all_rings(n)
        orbits = direct_orbits(n)
        assert sum(len(o) for o in orbits) == comb(factorial(n - 1), n + 1)
        assert burnside_classes(n) == len(orbits)
        # Solutions are the same throughout a class.
        assert all(len({found[s] for s in orbit}) == 1 for orbit in orbits)
        by_solutions = Counter(found[orbit[0]] for orbit in orbits)
        assert class_distribution(n, exact_distribution(n)) == {0: 0, **by_solutions}
        assert direct_classes(n, 1) == {k: v for k, v in sorted(by_solutions.items()) if k}
        by_symmetry = Counter(factorial(n) // len(orbit) for orbit in orbits)
        assert stabilizer_orders(n) == dict(sorted(by_symmetry.items()))
    assert len(direct_orbits(5)) == 1176


def test_count_rings_agrees_with_brute_force():
    rng = random.Random(4)
    for _ in range(40):
        pool = rng.sample(all_orders(6), 7)
        assert count_rings(pool, 6) == brute_force(pool), pool


def test_six_sided_classes():
    assert burnside_classes(6) == 82_628_379
    # Only renamings of cycle types (6), (3, 3) and (2, 2, 2) fix any set of seven nuts.
    fixing = {cycle_type(g): fixed_count(action(g, 6), 7) for g in renamings(6)}
    assert {t: f for t, f in fixing.items() if f} == {
        (6,): 44, (3, 3): 4788, (2, 2, 2): 311_144, (1,) * 6: comb(120, 7)
    }  # fmt: skip
    original = [nut.canonical_order() for nut in nuts]
    assert stabilizer(original, 6) == [(1, 2, 3, 4, 5, 6)]


def test_recorded_classes_are_consistent_with_the_set_counts():
    classes, sets = read_classes(), read_csv()
    assert sum(classes.values()) == burnside_classes(6)
    assert classes[1] == 4_417_343
    # 720 times the classes with k solutions is the number of pairs (renaming, set with k
    # solutions it fixes). Leaving out the identity leaves the fixed sets of the others.
    others = {k: 720 * classes[k] - sets[k] for k in sets}
    assert sum(others.values()) == 120 * 44 + 40 * 4788 + 15 * 311_144
    # Solve the sets fixed by one 6-cycle and one (3, 3) renaming; the rest must come from
    # the fifteen renamings of type (2, 2, 2).
    rest = Counter(others)
    orders = all_orders(6)
    for kind, many in ((6,), 120), ((3, 3), 40):
        g = next(g for g in renamings(6) if cycle_type(g) == kind)
        for s in fixed_sets(action(g, 6), 7):
            rest[count_rings([orders[j] for j in s], 6)] -= many
    assert all(v >= 0 and v % 15 == 0 for v in rest.values()), rest
