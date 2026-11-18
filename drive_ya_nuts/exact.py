r"""Exact distribution of the number of solutions over all sets of seven distinct nuts.

There are 120 distinct nuts and C(120, 7) = 59,487,568,920 sets of seven. Solving each set
is out of reach, so this module counts arrangements instead of sets.

A **ring** is a center nut together with an outer nut in each of the six directions, every
touching pair of edges matching. A set's solution count is the number of rings made of its
seven nuts, with the center turned to a fixed position (as in `solver.count_solutions`).

1. **Relabeling.** Renaming the numbers 1..6 by a permutation maps nuts to nuts, rings to rings
   and sets to sets, and keeps every set's solution count. Any nut can be renamed to
   C0 = (1, 2, 3, 4, 5, 6), so every ring is a renamed ring around C0.
2. **Rings around C0.** Call e_i the number on the edge shared by outer nuts i and i + 1. The
   outer nut in direction i reads (i + 1, e_{i-1}, x, y, z, e_i) counterclockwise, with x, y, z
   the other three numbers in any of 3! orders. There are 1,144 edge sequences, so at most
   1,144 * 6^6 rings around C0; the 43,500,008 with seven distinct nuts are kept.
3. **Sets that contain C0.** A ring with center C and C0 among its outer nuts is a ring around
   C0 renamed by a permutation that takes C0 to C. For each C, rename the rings around C0 that
   contain the nut that permutation sends to C0. Together with the rings around C0 themselves,
   this lists every ring whose set contains C0, each once.
4. **Count.** Sorting the sets of those rings gives, for each set S that contains C0, its
   solution count n(S). Each nut is in the same number of sets with n(S) = k, so the number
   of all sets with n(S) = k is 120 / 7 times the number that contain C0.

The construction is the same for nuts with n sides (n + 1 nuts to a set, (n - 1)! distinct
nuts), which gives small cases to check against brute force.
"""

import argparse
import csv
import random
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from itertools import permutations, product
from math import comb, factorial
from pathlib import Path

import numpy as np

from drive_ya_nuts.solver import Nut, count_solutions

CSV_PATH = Path(__file__).with_name("exact_counts.csv")

Order = tuple[int, ...]  # a nut's numbers read counterclockwise


def canonical(order: Sequence[int]) -> Order:
    """The rotation of a nut's order that starts with 1."""
    i = list(order).index(1)
    return tuple(order[i:]) + tuple(order[:i])


def all_orders(n: int) -> list[Order]:
    """Every distinct nut with n sides, in sorted order; the first is (1, 2, ..., n)."""
    return [(1, *rest) for rest in permutations(range(2, n + 1))]


def edge_sequences(n: int) -> Iterator[Order]:
    """Edge numbers e_0..e_{n-1} for rings around C0: e_i borders outer nuts i and i + 1.

    Outer nut i faces the center with i + 1, and e_{i-1} and e_i are its neighbors of i + 1,
    so e_i differs from i + 1, from i + 2 and from e_{i-1}.
    """
    for e in product(range(1, n + 1), repeat=n):
        if all(e[i] not in (i + 1, (i + 1) % n + 1, e[i - 1]) for i in range(n)):
            yield e


def candidates(n: int, i: int, before: int, after: int) -> list[Order]:
    """Outer nuts in direction i: they face the center with i + 1, then read `before` ... `after`.

    `before` is e_{i-1}, shared with outer nut i - 1; `after` is e_i, shared with outer nut i + 1.
    """
    c = i + 1
    rest = [v for v in range(1, n + 1) if v not in (c, before, after)]
    return [canonical((c, before, *mid, after)) for mid in permutations(rest)]


def rings_around_c0(n: int) -> np.ndarray:
    """All rings around C0 with distinct nuts, as an array of outer-nut indices, one row each."""
    index = {order: j for j, order in enumerate(all_orders(n))}
    k = factorial(n - 3)
    choice = np.array(list(product(range(k), repeat=n)), dtype=np.intp).reshape(-1, n)
    blocks = []
    for e in edge_sequences(n):
        table = np.array(
            [[index[o] for o in candidates(n, i, e[i - 1], e[i])] for i in range(n)],
            dtype=np.intp,
        )
        rows = table[np.arange(n), choice]
        s = np.sort(rows, axis=1)
        keep = (s[:, 0] != 0) & (np.diff(s, axis=1) != 0).all(axis=1)  # C0 has index 0
        blocks.append(rows[keep].astype(np.uint8))
    return np.concatenate(blocks) if blocks else np.zeros((0, n), np.uint8)


def renamings(n: int) -> np.ndarray:
    """Row C maps each nut index to its index after renaming C0 to nut C (number v to C[v-1])."""
    orders = all_orders(n)
    index = {order: j for j, order in enumerate(orders)}
    return np.array(
        [[index[canonical([c[v - 1] for v in o])] for o in orders] for c in orders],
        dtype=np.uint8,
    )


def ranks_without_c0(sets: np.ndarray) -> np.ndarray:
    """Rank of each row's nuts other than C0 among the (n + 1)-element sets with C0.

    Each row holds n + 1 distinct nut indices including 0 (C0). The other n are ranked among
    the n-element subsets of 1..(n - 1)! - 1 by the combinatorial number system.
    """
    s = np.sort(sets, axis=1)[:, 1:].astype(np.int64) - 1  # drop the 0 that every row has
    ranks = np.zeros(len(s), dtype=np.int64)
    for j in range(s.shape[1]):
        table = np.array([comb(x, j + 1) for x in range(int(s.max(initial=0)) + 1)], np.int64)
        ranks += table[s[:, j]]
    return ranks


def rings_in_sets_with_c0(n: int, rings: np.ndarray) -> Iterator[np.ndarray]:
    """Every ring whose set contains C0, as rows of n + 1 nut indices, center first."""
    for start in range(0, len(rings), 1 << 22):
        chunk = rings[start : start + (1 << 22)]
        yield np.column_stack([np.zeros(len(chunk), np.uint8), chunk])
    rename = renamings(n)
    for c in range(1, len(rename)):
        r = int(np.flatnonzero(rename[c] == 0)[0])  # the nut that becomes C0
        picked = rename[c][rings[(rings == r).any(axis=1)]]
        yield np.column_stack([np.full(len(picked), c, np.uint8), picked])


def run_lengths(values: np.ndarray, block: int = 1 << 24) -> Counter[int]:
    """How many runs of equal values a sorted array has of each length, in bounded memory."""
    hist: Counter[int] = Counter()
    run = 0  # length so far of the run that reaches the end of the previous block
    for start in range(0, len(values), block):
        x = values[start : start + block]
        if run and x[0] != values[start - 1]:
            hist[run] += 1
            run = 0
        starts = np.flatnonzero(x[1:] != x[:-1]) + 1
        if len(starts) == 0:
            run += len(x)
            continue
        hist[run + int(starts[0])] += 1
        for k, v in enumerate(np.bincount(np.diff(starts)).tolist()):
            if v:
                hist[k] += v
        run = len(x) - int(starts[-1])
    if run:
        hist[run] += 1
    return hist


def sorted_ranks(n: int) -> np.ndarray:
    """Sorted set ranks of every ring whose set contains C0: a set with k solutions appears k times."""
    assert comb(factorial(n - 1) - 1, n) < 2**32  # ranks fit in uint32
    rings = rings_around_c0(n)
    # Each ring has n + 1 nuts, so by symmetry n + 1 times as many rings have C0 somewhere.
    ranks = np.empty((n + 1) * len(rings), dtype=np.uint32)
    filled = 0
    for sets in rings_in_sets_with_c0(n, rings):
        ranks[filled : filled + len(sets)] = ranks_without_c0(sets)
        filled += len(sets)
    assert filled == len(ranks), (filled, len(ranks))
    ranks.sort()
    return ranks


def exact_distribution(n: int = 6, ranks: np.ndarray | None = None) -> dict[int, int]:
    """Number of sets of n + 1 distinct nuts with each number of solutions."""
    nuts, size = factorial(n - 1), n + 1
    if ranks is None:
        ranks = sorted_ranks(n)
    runs = run_lengths(ranks)  # runs[k]: sets with C0 and exactly k solutions
    counts = {}
    for k, v in sorted(runs.items()):
        # Every nut is in the same number of these sets, and each set has `size` nuts.
        assert v * nuts % size == 0, (k, v)
        counts[k] = v * nuts // size
    counts[0] = comb(nuts, size) - sum(counts.values())
    return dict(sorted(counts.items()))


def unrank(rank: int, n: int) -> list[Order]:
    """The set with C0 whose `ranks_without_c0` is `rank`, as nut orders."""
    picked = []
    for j in range(n, 0, -1):
        x = j - 1
        while comb(x + 1, j) <= rank:
            x += 1
        picked.append(x + 1)
        rank -= comb(x, j)
    orders = all_orders(n)
    return [orders[0]] + [orders[i] for i in picked]


def solve_hexagons(pool: list[Order]) -> int:
    """Solutions of a set of six-sided nuts, by the search in `solver.py`."""
    return count_solutions({Nut(order) for order in pool})


def check_sets(
    ranks: np.ndarray, n: int, trials: int, seed: int, solve: Callable[[list[Order]], int]
) -> int:
    """Solve random sets with C0 one at a time; return how many disagree with `ranks`.

    Even trials draw a set uniformly; odd trials draw a solvable set, weighted by its solutions.
    """
    rng = random.Random(seed)
    wrong = 0
    for t in range(trials):
        if t % 2:
            rank = int(ranks[rng.randrange(len(ranks))])
        else:
            rank = rng.randrange(comb(factorial(n - 1) - 1, n))
        key = np.uint32(rank)
        expected = int(np.searchsorted(ranks, key, "right") - np.searchsorted(ranks, key))
        wrong += solve(unrank(rank, n)) != expected
    return wrong


def read_csv(path: Path = CSV_PATH) -> dict[int, int]:
    with path.open() as f:
        return {int(row["solutions"]): int(row["sets"]) for row in csv.DictReader(f)}


def write_csv(counts: dict[int, int], path: Path = CSV_PATH) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["solutions", "sets"])
        w.writerows(sorted(counts.items()))


def report(counts: dict[int, int]) -> None:
    total = sum(counts.values())
    solvable = total - counts[0]
    print(f"{total:,} sets, {sum(k * v for k, v in counts.items()):,} solutions in all")
    print(f"solvable: {solvable:,} ({solvable / total:.4%})")
    print(f"exactly one solution: {counts[1]:,} ({counts[1] / total:.4%})")
    print(f"exactly one, among solvable: {counts[1] / solvable:.4%}")
    print()
    print(f"{'solutions':>9}  {'sets':>14}  {'share':>9}")
    for k, v in counts.items():
        print(f"{k:>9}  {v:>14,}  {v / total:>9.4%}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sides", type=int, default=6, help="sides per nut (default 6)")
    parser.add_argument("--write", action="store_true", help=f"write {CSV_PATH.name}")
    parser.add_argument(
        "--check",
        type=int,
        default=0,
        metavar="N",
        help="also solve N random sets with C0 one at a time and compare (six sides only)",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed for --check")
    args = parser.parse_args(argv)
    if args.sides != 6 and (args.write or args.check):
        parser.error("--write and --check are for six sides")
    t0 = time.perf_counter()
    ranks = sorted_ranks(args.sides)
    counts = exact_distribution(args.sides, ranks)
    if args.check:
        wrong = check_sets(ranks, args.sides, args.check, args.seed, solve_hexagons)
        print(f"solved {args.check} sets with seed {args.seed} one at a time: {wrong} disagree")
    del ranks
    print(f"computed in {time.perf_counter() - t0:.0f} s")
    report(counts)
    if args.write:
        write_csv(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
