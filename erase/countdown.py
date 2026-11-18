"""The Countdown numbers round: how many of its problems can be hit exactly?

Six numbers are drawn from 1 to 10 (two of each) and 25, 50, 75 and 100. The players combine
them with +, -, x and / to reach a target from 101 to 999. Each number is used at most once,
and every intermediate result must be a positive whole number. Following J.-M. Alliot, "The
(Final) Countdown" (arXiv:1502.05450, 2015), a problem is one selection of six numbers (as a
multiset, so 13,243 of them, each counted once) and one target, 13,243 x 899 = 11,905,457
problems in all. His Table 2 reports 10,858,746 (91.21%) solvable exactly, and 743,896 more at
distance 1; 1,226 selections reach every target, and {1, 1, 2, 2, 3, 3} reaches none. His
distances count only reachable values from 101 to 999; counting values of any size, more
problems have a value at distance 1 (744,561 rather than 743,896).

`reachable` builds, for every subset of the six numbers, the set of values an expression using
exactly that subset can take, from the values of its two parts. `reachable_by_moves` is the
obvious search that repeatedly replaces two numbers by one; the tests compare the two.

    python3 erase/countdown.py     # about 3 minutes
"""

from bisect import bisect_left
from collections import Counter
from itertools import combinations

NUMBERS = sorted([*range(1, 11)] * 2 + [25, 50, 75, 100])
TARGETS = range(101, 1000)


def selections() -> list[tuple[int, ...]]:
    """The different multisets of six numbers that can be drawn."""
    return sorted(set(combinations(NUMBERS, 6)))


def combine(a: int, b: int) -> set[int]:
    """Every positive whole number one operation makes from a and b."""
    out = {a + b, a * b}
    hi, lo = max(a, b), min(a, b)
    if hi > lo:
        out.add(hi - lo)
    if hi % lo == 0:
        out.add(hi // lo)
    return out


def reachable(numbers: tuple[int, ...]) -> set[int]:
    """Every value of an expression that uses some of the numbers, each at most once."""
    n = len(numbers)
    values: list[set[int]] = [set() for _ in range(1 << n)]
    for i, x in enumerate(numbers):
        values[1 << i].add(x)
    for mask in range(1, 1 << n):
        sub = (mask - 1) & mask
        while sub:  # each split of mask into two nonempty parts, once
            other = mask ^ sub
            if sub < other:
                for a in values[sub]:
                    for b in values[other]:
                        values[mask] |= combine(a, b)
            sub = (sub - 1) & mask
    return set().union(*values)


def reachable_by_moves(numbers: tuple[int, ...]) -> set[int]:
    """The same set, by searching every sequence of moves that replace two numbers by one."""
    seen: set[tuple[int, ...]] = set()
    found: set[int] = set()

    def search(state: tuple[int, ...]) -> None:
        if state in seen:
            return
        seen.add(state)
        found.update(state)
        for i, j in combinations(range(len(state)), 2):
            rest = state[:i] + state[i + 1 : j] + state[j + 1 :]
            for v in combine(state[i], state[j]):
                search(tuple(sorted(rest + (v,))))

    search(tuple(sorted(numbers)))
    return found


def distance(values: list[int], target: int) -> int:
    """How far the nearest of the sorted reachable values is from the target."""
    i = bisect_left(values, target)
    return min(abs(v - target) for v in values[max(i - 1, 0) : i + 1])


def main() -> None:
    sels = selections()
    # Distance to the nearest value of any size, and, as Alliot's Table 2 counts it, to the
    # nearest value from 101 to 999.
    near_any: Counter[int] = Counter()
    near_range: Counter[int] = Counter()
    hits_all, hits_none = [], []
    for s in sels:
        values = sorted(reachable(s))
        in_range = [v for v in values if v in TARGETS]
        for t in TARGETS:
            near_any[distance(values, t)] += 1
            near_range[distance(in_range, t) if in_range else len(TARGETS)] += 1
        hits_all += [s] if len(in_range) == len(TARGETS) else []
        hits_none += [s] if not in_range else []
    total = len(sels) * len(TARGETS)
    print(f"{len(sels):,} selections x {len(TARGETS)} targets = {total:,} problems")
    print("distance  nearest value 101-999      nearest value of any size")
    cum_range = cum_any = 0
    for d in range(5):
        cum_range += near_range[d]
        cum_any += near_any[d]
        print(
            f"{d:>8}  {near_range[d]:>10,} {near_range[d] / total:6.2%} ({cum_range / total:6.2%})"
            f"   {near_any[d]:>10,} {near_any[d] / total:6.2%} ({cum_any / total:6.2%})"
        )
    print(f"{len(hits_all):,} selections reach every target; none reached by {hits_none}")


if __name__ == "__main__":
    main()
