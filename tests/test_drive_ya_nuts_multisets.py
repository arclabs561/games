"""Classes of solvable multisets of nuts (drive_ya_nuts/multisets.py): two methods and brute force."""

import json
import random
from itertools import combinations_with_replacement

import numpy as np
import pytest

from drive_ya_nuts import multisets as ms
from drive_ya_nuts.classes import action, count_rings, renamings
from drive_ya_nuts.exact import all_orders, exact_distribution


def test_four_sides_by_brute_force():
    assert ms.brute_force(4) == 1
    keys, _ = ms.direct(4)
    assert len(keys) == 1
    assert ms.burnside(4, sum(v for k, v in exact_distribution(4).items() if k))["classes"] == 1


def test_five_sides_two_methods_agree():
    keys, _ = ms.direct(5)
    distinct = sum(v for k, v in exact_distribution(5).items() if k)
    assert ms.burnside(5, distinct)["classes"] == len(keys) == 276  # brute force: 276 (60 s)


def test_five_sides_random_multisets_against_the_keys():
    # A multiset is solvable exactly when its class key is among the listed keys.
    n = 5
    keys, _ = ms.direct(n)
    listed = set(keys.tolist())
    orders = all_orders(n)
    p = ms.to_c0(n)
    rng = random.Random(0)
    solvable = 0
    for _ in range(3000):
        m = sorted(rng.choices(range(len(orders)), k=n + 1))
        key = int(ms.least_keys(np.array([m], np.uint8), p)[0])
        is_solvable = count_rings([orders[j] for j in m], n) > 0
        assert (key in listed) == is_solvable, m
        solvable += is_solvable
    assert solvable > 30


def test_least_key_is_a_class_invariant():
    n = 5
    orders = all_orders(n)
    p = ms.to_c0(n)
    gs = [action(g, n) for g in renamings(n)]
    rng = random.Random(1)
    for _ in range(50):
        m = sorted(rng.choices(range(len(orders)), k=n + 1))
        keys = {int(ms.least_keys(np.array([sorted(g[j] for j in m)], np.uint8), p)[0]) for g in gs}
        assert len(keys) == 1


def test_fixed_multisets_by_hand():
    # A permutation of 4 items with cycles (0)(1 2)(3): multisets of size 2 fixed by it are
    # {0,0}, {3,3}, {0,3} and {1,2}.
    found = sorted(sorted(m) for m in ms.fixed_multisets([0, 2, 1, 3], 2))
    assert found == [[0, 0], [0, 3], [1, 2], [3, 3]]
    assert len(list(combinations_with_replacement(range(4), 2))) == 10


def test_recorded_six_sided_counts():
    record = json.loads(ms.RESULTS.read_text())
    assert record["sides"] == 6
    assert record["direct"]["classes"] == record["burnside"]["classes"]
    by_d = record["direct"]["classes_by_distinct_nuts"]
    assert by_d["7"] == 5_623_238  # the classes of sets of distinct nuts (classes.py)
    assert sum(by_d.values()) == record["direct"]["classes"]
    per_ring = record["direct"]["classes_by_rings_around_c0"]
    assert sum(int(k) * v for k, v in per_ring.items()) == 1144 * 6**6
    b = record["burnside"]
    assert b["solvable_distinct"] == 4_048_011_480
    total = b["solvable_distinct"] + b["solvable_repeated"]
    total += sum(k * f for k, f in b["fixed_by_type"].values())
    assert total == 720 * b["classes"]


@pytest.mark.parametrize("count", sorted(ms.EXAMPLES))
def test_comment_examples_are_solvable(count):
    orders = all_orders(6)
    index = {o: j for j, o in enumerate(orders)}
    from drive_ya_nuts.exact import canonical

    pool = [orders[index[canonical([int(c) for c in reversed(nut)])]] for nut in ms.EXAMPLES[count]]
    assert count_rings(pool, 6) > 0
