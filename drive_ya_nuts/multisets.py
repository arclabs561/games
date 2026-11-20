r"""Classes of solvable sets when a nut may appear more than once.

A comment on the Mathematics Stack Exchange question about Drive Ya Nuts (question 3306917,
2019) reports 6,901,100 "distinct cog sets" with a solution. This module defines that count
exactly and computes it in two ways that share only the listing of rings around C0.

- A **multiset** is seven nuts, where a nut may appear more than once.
- It is **solvable** if some ring is made of exactly those seven nuts.
- Two multisets are **equivalent** if a renaming of the numbers 1..6 turns one into the
  other; a **class** is a set of equivalent multisets. The count asked for is the number of
  classes of solvable multisets.

1. **Direct listing** (the comment's method). Every class has a member with a ring around
   C0 = (1, 2, ..., 6), so it is enough to take each of the 1,144 * 6^6 rings around C0,
   repeated nuts included. A renaming keeps C0 in a multiset only if it turns one of the
   multiset's nuts into C0, and for each nut there are six such renamings, one for each
   side that becomes the 1. So the least of the 42 renamed multisets (sorted and packed into
   one integer) is the same for every member of a class that contains C0. Distinct least
   keys are classes.
2. **Burnside's lemma.** The number of classes is the average over the 720 renamings of the
   number of solvable multisets each one fixes. For the identity that is every solvable
   multiset: the 4,048,011,480 sets of distinct nuts (`exact_counts.csv`) plus those with a
   repeated nut, counted from every ring whose multiset contains C0 and weighted by the
   number of distinct nuts (a multiset with d distinct nuts contains d nuts, and by symmetry
   each nut is in the same number of them). For any other renaming the fixed multisets are
   unions of whole cycles, with one multiplicity per cycle, and each is solved by search.

Both work for nuts with n sides (n + 1 nuts to a multiset), which the tests compare with a
brute force over every multiset for four and five sides.

    uv run --group drive_ya_nuts python -m drive_ya_nuts.multisets          # about 10 minutes, 3 GB
    uv run --group drive_ya_nuts python -m drive_ya_nuts.multisets --sides 5
"""

import argparse
import json
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from itertools import combinations_with_replacement, product
from math import factorial
from pathlib import Path

import numpy as np

from drive_ya_nuts.classes import action, count_rings, cycle_type, cycles, renamings
from drive_ya_nuts.exact import (
    Order,
    all_orders,
    candidates,
    canonical,
    edge_sequences,
    read_csv,
)
from drive_ya_nuts.exact import renamings as center_renamings

RESULTS = Path(__file__).with_name("multiset_counts.json")
REPORTED = 6_901_100  # the Stack Exchange comment


def all_rings_around_c0(n: int) -> np.ndarray:
    """Every ring around C0, repeated nuts included, as rows of n outer-nut indices."""
    index = {order: j for j, order in enumerate(all_orders(n))}
    k = factorial(n - 3)
    choice = np.array(list(product(range(k), repeat=n)), dtype=np.intp).reshape(-1, n)
    blocks = []
    for e in edge_sequences(n):
        table = np.array(
            [[index[o] for o in candidates(n, i, e[i - 1], e[i])] for i in range(n)],
            dtype=np.intp,
        )
        blocks.append(table[np.arange(n), choice].astype(np.uint8))
    return np.concatenate(blocks)


def to_c0(n: int) -> np.ndarray:
    """P[c, r] permutes nut indices by the renaming that turns nut c into C0, side r first."""
    orders = all_orders(n)
    out = np.empty((len(orders), n, len(orders)), np.uint8)
    for c, order in enumerate(orders):
        for r in range(n):
            g = [0] * n
            for i in range(n):
                g[order[(r + i) % n] - 1] = i + 1  # the number on side r + i becomes i + 1
            out[c, r] = action(tuple(g), n)
    return out


def pack(rows: np.ndarray) -> np.ndarray:
    """Each row of nut indices, sorted and packed into one integer (7 bits per nut)."""
    s = np.sort(rows, axis=1).astype(np.int64)
    key = np.zeros(len(s), np.int64)
    for j in range(s.shape[1]):
        key = key << 7 | s[:, j]
    return key


def unpack(key: int, size: int) -> list[int]:
    return [(key >> (7 * (size - 1 - j))) & 127 for j in range(size)]


def least_keys(multisets: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The class key of each multiset: the least of its renamings that contain C0."""
    best = np.full(len(multisets), np.iinfo(np.int64).max, np.int64)
    for j in range(multisets.shape[1]):
        nut = multisets[:, j : j + 1].astype(np.intp)
        for r in range(p.shape[1]):
            renamed = p[nut, r, multisets.astype(np.intp)]
            np.minimum(best, pack(renamed), out=best)
    return best


def direct(n: int, chunk: int = 1 << 21) -> tuple[np.ndarray, np.ndarray]:
    """Class keys of the solvable multisets, and how many rings around C0 fall in each."""
    rings = all_rings_around_c0(n)
    p = to_c0(n)
    keys = np.empty(len(rings), np.int64)
    for start in range(0, len(rings), chunk):
        part = rings[start : start + chunk]
        rows = np.column_stack([np.zeros(len(part), np.uint8), part])
        keys[start : start + chunk] = least_keys(rows, p)
    del rings
    return np.unique(keys, return_counts=True)


def distinct_nuts(keys: np.ndarray, size: int) -> np.ndarray:
    """How many different nuts each packed multiset holds."""
    count = np.ones(len(keys), np.int64)
    prev = (keys >> (7 * (size - 1))) & 127
    for j in range(1, size):
        cur = (keys >> (7 * (size - 1 - j))) & 127
        count += cur != prev
        prev = cur
    return count


def repeated_with_c0(n: int) -> Counter[int]:
    """Multisets with C0 and a repeated nut that some ring uses, by number of distinct nuts."""
    rings = all_rings_around_c0(n)
    repeated = (rings == 0).any(axis=1)
    for i in range(n):
        for j in range(i + 1, n):
            repeated |= rings[:, i] == rings[:, j]
    rings = rings[repeated]
    del repeated
    keys = [pack(np.column_stack([np.zeros(len(rings), np.uint8), rings]))]
    rename = center_renamings(n)
    for c in range(1, len(rename)):
        r = int(np.flatnonzero(rename[c] == 0)[0])  # the nut that this renaming makes C0
        picked = rename[c][rings[(rings == r).any(axis=1)]]
        keys.append(pack(np.column_stack([np.full(len(picked), c, np.uint8), picked])))
    joined = np.concatenate(keys)
    del keys
    unique = np.unique(joined)
    del joined
    return Counter(distinct_nuts(unique, n + 1).tolist())


def fixed_multisets(p: Sequence[int], size: int) -> Iterator[list[int]]:
    """Every multiset of `size` nuts fixed by the nut permutation p: a multiplicity per cycle."""
    cs = cycles(p)

    def extend(j: int, left: int, chosen: list[int]) -> Iterator[list[int]]:
        if left == 0:
            yield chosen
            return
        if j == len(cs):
            return
        for m in range(left // len(cs[j]) + 1):
            yield from extend(j + 1, left - m * len(cs[j]), chosen + cs[j] * m)

    yield from extend(0, size, [])


def solvable(pool: Sequence[Order], n: int) -> bool:
    return count_rings(pool, n) > 0


def fixed_solvable(n: int) -> dict[tuple[int, ...], tuple[int, int]]:
    """For each cycle type other than the identity: (renamings, solvable multisets each fixes)."""
    orders = all_orders(n)
    by_type: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    for g in renamings(n):
        by_type.setdefault(cycle_type(g), []).append(g)
    out = {}
    for kind, gs in sorted(by_type.items()):
        if kind == (1,) * n:
            continue
        fixed = fixed_multisets(action(gs[0], n), n + 1)
        out[kind] = (len(gs), sum(solvable([orders[j] for j in m], n) for m in fixed))
    return out


def burnside(n: int, distinct_solvable: int) -> dict:
    """Classes of solvable multisets by Burnside's lemma, with the terms that make it up."""
    nuts = factorial(n - 1)
    by_d = repeated_with_c0(n)
    repeated = 0
    for d, count in by_d.items():
        assert count * nuts % d == 0, (d, count)
        repeated += count * nuts // d
    others = fixed_solvable(n)
    total = distinct_solvable + repeated + sum(k * f for k, f in others.values())
    assert total % factorial(n) == 0, total
    return {
        "solvable_distinct": distinct_solvable,
        "solvable_repeated": repeated,
        "fixed_by_type": {" ".join(map(str, k)): list(v) for k, v in others.items()},
        "classes": total // factorial(n),
    }


def brute_force(n: int) -> int:
    """Classes of solvable multisets of n + 1 nuts by listing every multiset (small n only)."""
    orders = all_orders(n)
    gs = [action(g, n) for g in renamings(n)]
    seen: set[tuple[int, ...]] = set()
    classes = 0
    for m in combinations_with_replacement(range(len(orders)), n + 1):
        if m in seen or not solvable([orders[j] for j in m], n):
            continue
        classes += 1
        seen.update(tuple(sorted(g[j] for j in m)) for g in gs)
    return classes


def clockwise_key(nuts: Sequence[str], n: int = 6) -> int:
    """The class key of a multiset given as nuts read clockwise, as on Stack Exchange.

    This module reads nuts counterclockwise, so each nut is reversed. Mirroring every nut
    maps solutions to solutions, so the counts agree.
    """
    index = {o: j for j, o in enumerate(all_orders(n))}
    rows = [index[canonical([int(ch) for ch in reversed(nut)])] for nut in nuts]
    return int(least_keys(np.array([rows], np.uint8), to_c0(n))[0])


# Examples from the comment, nuts read clockwise: four classes have one ring around C0,
# and two have 120; one example of each kind is given there.
EXAMPLES = {
    1: ["123456", "124653", "132645", "135426", "153462", "156243", "164235"],
    120: ["123456", "123456", "123645", "126453", "146253", "156243", "156423"],
}


def direct_record(n: int) -> dict:
    """The direct listing's counts, as written to the results file."""
    t0 = time.perf_counter()
    keys, rings = direct(n)
    by_d = Counter(distinct_nuts(keys, n + 1).tolist())
    per_class = Counter(rings.tolist())
    record = {
        "classes": len(keys),
        "classes_by_distinct_nuts": {str(k): v for k, v in sorted(by_d.items())},
        "classes_by_rings_around_c0": {str(k): v for k, v in sorted(per_class.items())},
    }
    if n == 6:
        found = {}
        for count, nuts in EXAMPLES.items():
            key = clockwise_key(nuts)
            i = int(np.searchsorted(keys, key))
            found[str(count)] = int(rings[i]) if i < len(keys) and keys[i] == key else 0
        record["examples"] = found
    print(f"direct listing ({time.perf_counter() - t0:.0f} s):", record["classes"], flush=True)
    return record


def burnside_record(n: int) -> dict:
    """Burnside's lemma, with the count of solvable sets of distinct nuts from `exact`."""
    t0 = time.perf_counter()
    if n == 6:
        distinct = sum(v for k, v in read_csv().items() if k > 0)
    else:
        from drive_ya_nuts.exact import exact_distribution

        distinct = sum(v for k, v in exact_distribution(n).items() if k > 0)
    record = burnside(n, distinct)
    print(f"Burnside's lemma ({time.perf_counter() - t0:.0f} s):", record, flush=True)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sides", type=int, default=6)
    parser.add_argument("--write", action="store_true", help=f"write {RESULTS.name}")
    parser.add_argument(
        "--part",
        choices=["both", "direct", "burnside"],
        default="both",
        help="run one method only; with --write, the other is kept from the file",
    )
    args = parser.parse_args(argv)
    n = args.sides
    record: dict = {"sides": n}
    if args.part != "both" and args.write and RESULTS.exists():
        record = json.loads(RESULTS.read_text())
    if args.part in ("both", "direct"):
        record["direct"] = direct_record(n)
    if args.part in ("both", "burnside"):
        record["burnside"] = burnside_record(n)
    if args.write:
        RESULTS.write_text(json.dumps(record, indent=1) + "\n")
    if "direct" not in record or "burnside" not in record:
        return 0
    agree = record["burnside"]["classes"] == record["direct"]["classes"]
    print("the two methods", "agree" if agree else "DISAGREE", end="; ")
    if n == 6:
        print(f"the comment reports {REPORTED:,}:", end=" ")
        print("agree" if record["direct"]["classes"] == REPORTED else "differ", end="")
    print()
    return 0 if agree else 1


if __name__ == "__main__":
    raise SystemExit(main())
