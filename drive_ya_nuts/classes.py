r"""Essentially different sets of nuts: sets of seven distinct nuts up to relabeling.

Renaming the numbers 1..6 by a permutation g maps every nut to a nut and every set of nuts
to a set with the same number of solutions. Two sets are **equivalent** if a renaming takes
one to the other; a **class** is a set of equivalent sets. Nuts are already taken up to
rotation (`exact.all_orders`), so a class is a set of nuts up to rotating each nut and
renaming the numbers.

1. **Burnside's lemma.** The number of classes is the average, over the 720 renamings g, of
   the number of sets that g fixes. A set is fixed by g exactly when it is a union of cycles
   of g acting on the 120 nuts, so that number is the coefficient of x^7 in the product of
   (1 + x^L) over the cycle lengths L.
2. **Classes by solutions.** Applied to the sets with k solutions alone, the same average
   counts the classes with k solutions. For g = identity the count is the exact table
   (`exact_counts.csv`); for every other g the fixed sets are listed and solved one by one.
   Conjugate renamings fix the same number of sets with k solutions, so one renaming per
   cycle type is enough.

For nuts with n sides the same steps apply with n + 1 nuts to a set and n! renamings, and
for four and five sides the tests compare them with a direct listing of every class.
"""

import argparse
import csv
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from itertools import permutations
from math import comb, factorial
from pathlib import Path

import numpy as np

from drive_ya_nuts.exact import Order, all_orders, canonical, read_csv, sorted_ranks

CSV_PATH = Path(__file__).with_name("class_counts.csv")

Renaming = tuple[int, ...]  # number v becomes g[v - 1]


def renamings(n: int) -> list[Renaming]:
    """All n! renamings of the numbers 1..n."""
    return list(permutations(range(1, n + 1)))


def rename(g: Renaming, order: Sequence[int]) -> Order:
    """The nut that `order` becomes when every number v is renamed g[v - 1]."""
    return canonical([g[v - 1] for v in order])


def action(g: Renaming, n: int) -> list[int]:
    """The permutation of nut indices (in `all_orders(n)`) that the renaming g induces."""
    orders = all_orders(n)
    index = {o: j for j, o in enumerate(orders)}
    return [index[rename(g, o)] for o in orders]


def cycles(p: Sequence[int]) -> list[list[int]]:
    """The cycles of a permutation of 0..len(p) - 1."""
    seen: set[int] = set()
    out = []
    for i in range(len(p)):
        cycle = []
        while i not in seen:
            seen.add(i)
            cycle.append(i)
            i = p[i]
        if cycle:
            out.append(cycle)
    return out


def cycle_type(g: Renaming) -> tuple[int, ...]:
    """Cycle lengths of the renaming g on the numbers, largest first."""
    return tuple(sorted((len(c) for c in cycles([v - 1 for v in g])), reverse=True))


def fixed_count(p: Sequence[int], size: int) -> int:
    """Sets of `size` nuts fixed by the nut permutation p: coefficient of x^size in prod (1 + x^L)."""
    coef = [1] + [0] * size
    for length in (len(c) for c in cycles(p)):
        for s in range(size, length - 1, -1):
            coef[s] += coef[s - length]
    return coef[size]


def fixed_sets(p: Sequence[int], size: int) -> Iterator[tuple[int, ...]]:
    """Every set of `size` nuts fixed by p, as sorted nut indices: unions of cycles of p."""
    cs = cycles(p)

    def extend(start: int, chosen: list[int]) -> Iterator[tuple[int, ...]]:
        if len(chosen) == size:
            yield tuple(sorted(chosen))
            return
        for j in range(start, len(cs)):
            if len(chosen) + len(cs[j]) <= size:
                yield from extend(j + 1, chosen + cs[j])

    yield from extend(0, [])


def burnside_classes(n: int = 6) -> int:
    """Number of classes of sets of n + 1 distinct nuts with n sides, by Burnside's lemma."""
    total = sum(fixed_count(action(g, n), n + 1) for g in renamings(n))
    assert total % factorial(n) == 0
    return total // factorial(n)


def stabilizer_orders(n: int = 6) -> dict[int, int]:
    """Number of classes whose sets are fixed by exactly s renamings, for each s.

    A class whose sets are fixed by s renamings has n! / s sets. The sets fixed by some
    renaming other than the identity are listed; every other set lies in a class of n! sets.
    """
    size = n + 1
    weights = np.array([128**i for i in range(size - 1, -1, -1)], np.int64)
    keys = [
        np.array(fixed, np.int64) @ weights
        for g in renamings(n)[1:]  # the first renaming is the identity
        if (fixed := list(fixed_sets(action(g, n), size)))
    ]
    _, times = np.unique(np.concatenate(keys), return_counts=True) if keys else ([], [])
    symmetric = Counter((np.asarray(times) + 1).tolist())  # + 1 for the identity
    free = comb(factorial(n - 1), size) - sum(symmetric.values())
    if free:
        symmetric[1] = free
    out = {}
    for s, sets in sorted(symmetric.items()):
        assert sets * s % factorial(n) == 0, (s, sets)
        out[s] = sets * s // factorial(n)
    return out


def count_rings(pool: Sequence[Order], n: int) -> int:
    """Solutions of a set of n + 1 nuts with n sides: rings, with the center's turn fixed.

    The center reads c_0..c_{n-1}, and outer nut i faces it with c_i. Outer nuts i and i + 1
    touch where nut i shows the number before c_i and nut i + 1 the number after c_{i+1},
    reading counterclockwise (the rule of `solver.recur`).
    """
    pos = [{v: j for j, v in enumerate(nut)} for nut in pool]

    def before(k: int, v: int) -> int:
        return pool[k][pos[k][v] - 1]

    def after(k: int, v: int) -> int:
        return pool[k][(pos[k][v] + 1) % n]

    def extend(center: Order, outer: list[int], i: int, used: int, first: int, prev: int) -> int:
        """Rings that complete outer nuts 0..i - 1; `prev` is what outer nut i must match."""
        if i == n:
            return int(prev == after(first, center[0]))
        found = 0
        for k in outer:
            if not used >> k & 1 and (i == 0 or after(k, center[i]) == prev):
                nxt = k if i == 0 else first
                found += extend(center, outer, i + 1, used | 1 << k, nxt, before(k, center[i]))
        return found

    count = 0
    for c, center in enumerate(pool):
        count += extend(center, [k for k in range(len(pool)) if k != c], 0, 0, -1, -1)
    return count


def class_distribution(n: int, set_counts: dict[int, int]) -> dict[int, int]:
    """Number of classes with each number of solutions, by Burnside's lemma weighted by k.

    `set_counts[k]` is the number of sets with k solutions (the renaming g = identity).
    """
    orders = all_orders(n)
    by_type: dict[tuple[int, ...], list[Renaming]] = {}
    for g in renamings(n):
        by_type.setdefault(cycle_type(g), []).append(g)
    total = Counter(set_counts)
    for kind, gs in by_type.items():
        if kind == (1,) * n:
            continue
        fixed = Counter(
            count_rings([orders[j] for j in s], n) for s in fixed_sets(action(gs[0], n), n + 1)
        )
        for k, v in fixed.items():
            total[k] += len(gs) * v
    out = {}
    for k, v in sorted(total.items()):
        assert v % factorial(n) == 0, (k, v)
        out[k] = v // factorial(n)
    return out


def long_runs(
    values: np.ndarray, least: int, block: int = 1 << 24
) -> tuple[np.ndarray, np.ndarray]:
    """Values of a sorted array that occur at least `least` times, and how often, in bounded memory."""
    keep_v, keep_n = [], []
    start = 0
    while start < len(values):
        stop = min(start + block, len(values))
        while stop < len(values) and values[stop] == values[stop - 1]:
            stop += 1  # never split a run between blocks
        x = values[start:stop]
        edges = np.concatenate([[0], np.flatnonzero(x[1:] != x[:-1]) + 1, [len(x)]])
        lengths = np.diff(edges)
        big = lengths >= least
        keep_v.append(x[edges[:-1][big]])
        keep_n.append(lengths[big])
        start = stop
    if not keep_v:
        return np.zeros(0, values.dtype), np.zeros(0, np.int64)
    return np.concatenate(keep_v), np.concatenate(keep_n)


def unrank_many(ranks: np.ndarray, n: int) -> np.ndarray:
    """Vectorized `exact.unrank`: the sets with C0 of the given ranks, as rows of nut indices."""
    rest = ranks.astype(np.int64).copy()
    cols = [np.zeros(len(ranks), np.int64)]  # C0
    top = factorial(n - 1)
    for j in range(n, 0, -1):
        table = np.array([comb(x, j) for x in range(top)], np.int64)
        x = np.searchsorted(table, rest, side="right") - 1  # largest x with C(x, j) <= rest
        cols.append(x + 1)
        rest -= table[x]
    return np.column_stack(cols)


def canonical_keys(sets: np.ndarray, n: int) -> np.ndarray:
    """One number per class: the least packed sorted set over all renamings of each row."""
    best = np.full(len(sets), np.iinfo(np.int64).max, np.int64)
    weights = np.array([128**i for i in range(sets.shape[1] - 1, -1, -1)], np.int64)
    for g in renamings(n):
        renamed = np.sort(np.asarray(action(g, n), np.int64)[sets], axis=1)
        np.minimum(best, renamed @ weights, out=best)
    return best


def direct_classes(n: int, least: int, ranks: np.ndarray | None = None) -> dict[int, int]:
    """Classes with each number k >= `least` of solutions, by listing every set with C0.

    Every class has a member that contains C0, so it is enough to take the canonical key
    of each set with C0 and k solutions (from `exact.sorted_ranks`) and count distinct keys.
    """
    if ranks is None:
        ranks = sorted_ranks(n)
    values, solutions = long_runs(ranks, least)
    keys = canonical_keys(unrank_many(values, n), n)
    out = {}
    for k in sorted(set(solutions.tolist())):
        out[k] = len(np.unique(keys[solutions == k]))
    return out


def stabilizer(pool: Sequence[Order], n: int) -> list[Renaming]:
    """The renamings that map the set `pool` to itself."""
    target = {canonical(o) for o in pool}
    return [g for g in renamings(n) if {rename(g, o) for o in pool} == target]


def read_classes(path: Path = CSV_PATH) -> dict[int, int]:
    with path.open() as f:
        return {int(row["solutions"]): int(row["classes"]) for row in csv.DictReader(f)}


def write_classes(counts: dict[int, int], path: Path = CSV_PATH) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["solutions", "classes"])
        w.writerows(sorted(counts.items()))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help=f"write {CSV_PATH.name}")
    parser.add_argument(
        "--direct",
        type=int,
        default=0,
        metavar="K",
        help="also count the classes with at least K solutions directly (2 GB, minutes)",
    )
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    n = 6
    classes = burnside_classes(n)
    print(f"{comb(120, 7):,} sets of seven distinct nuts fall into {classes:,} classes")
    for kind, gs in sorted(Counter(cycle_type(g) for g in renamings(n)).items()):
        f = fixed_count(action(next(g for g in renamings(n) if cycle_type(g) == kind), n), 7)
        print(f"  renamings of cycle type {kind}: {gs:>3}, each fixes {f:,} sets")
    print("classes by the number of renamings that fix their sets:", stabilizer_orders(n))
    counts = class_distribution(n, read_csv())
    assert sum(counts.values()) == classes
    print(f"computed in {time.perf_counter() - t0:.0f} s")
    print(f"{'solutions':>9}  {'classes':>11}  {'share':>9}")
    for k, v in counts.items():
        print(f"{k:>9}  {v:>11,}  {v / classes:>9.4%}")
    if args.direct:
        t0 = time.perf_counter()
        direct = direct_classes(n, args.direct)
        same = direct == {k: v for k, v in counts.items() if k >= args.direct}
        print(f"classes with {args.direct} or more solutions, listed directly in", end=" ")
        print(f"{time.perf_counter() - t0:.0f} s: {'agree' if same else 'DISAGREE'}")
        if not same:
            print(direct)
            return 1
    if args.write:
        write_classes(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
