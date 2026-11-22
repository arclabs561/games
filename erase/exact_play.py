"""Exact play on all four-card centers at once. Used by report.py exact_play."""

from collections import Counter
from functools import cache
from itertools import combinations_with_replacement
from math import comb, prod

VALUES = range(11)
RULES = (
    "pairs",
    "addition",
    "plus_minus",
    "at_most_3",
    "at_most_4",
    "running",
    "unrestricted",
)
CENTERS = list(combinations_with_replacement(VALUES, 4))
ALL = (1 << len(CENTERS)) - 1
WITH = {v: sum(1 << i for i, c in enumerate(CENTERS) if v in c) for v in VALUES}
CENTER_COUNTS = [tuple(Counter(center).items()) for center in CENTERS]
NONZERO_CENTERS = ALL ^ WITH[0]
CHOOSE = tuple(tuple(comb(n, k) for k in range(5)) for n in range(5))


def pattern_weights(hand):
    """Weight of each value pattern when physically distinct cards are dealt."""
    counts = Counter(hand)
    hand_ways = prod(CHOOSE[4][n] for n in counts.values())
    choices = [CHOOSE[4 - counts[v]] for v in VALUES]
    return [hand_ways * prod(choices[v][k] for v, k in center) for center in CENTER_COUNTS]


def indices(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def weighted(mask, weights):
    return sum(weights[i] for i in indices(mask))


def splits(cards):
    # Distinguish the first occurrence. Swapping the two parts cannot change a partition.
    for mask in range(1, (1 << len(cards)) - 1, 2):
        yield (
            tuple(v for i, v in enumerate(cards) if mask >> i & 1),
            tuple(v for i, v in enumerate(cards) if not (mask >> i & 1)),
        )


@cache
def values(cards):
    if len(cards) == 1:
        return {cards[0]}
    out = set()
    for left, right in splits(cards):
        for a in values(left):
            for b in values(right):
                out.update((a + b, a - b, b - a, a * b))
    return out


@cache
def running(cards, target):
    if len(cards) == 1:
        return cards[0] == target
    for i, v in enumerate(cards):
        if i and cards[i - 1] == v:
            continue
        rest = cards[:i] + cards[i + 1 :]
        if running(rest, target - v) or running(rest, target + v):
            return True
        if v == 0 and target == 0:
            return True
        if v and target % v == 0 and running(rest, target // v):
            return True
    return False


@cache
def whole(cards, target):
    """Try each split at the last operation. A running total is a quick witness."""
    if len(cards) <= 4:
        return target in values(cards)
    if running(cards, target):
        return True
    for a, b in set(splits(cards)):
        if len(a) > len(b):
            a, b = b, a
        for x in values(a):
            if whole(b, target - x) or whole(b, x - target) or whole(b, target + x):
                return True
            if (x == 0 and target == 0) or (x and target % x == 0 and whole(b, target // x)):
                return True
    return False


@cache
def group(cards):
    n = len(cards)
    if n < 2:
        return (False,) * len(RULES)
    total = sum(cards)
    sums = 1  # bit k means a subset sums to k
    for v in cards:
        sums |= sums << v
    plus_minus = (total + 10) % 2 == 0 and (sums >> ((total + 10) // 2)) & 1
    small = n <= 4 and 10 in values(cards)
    return (
        n == 2 and total == 10,
        total == 10,
        plus_minus,
        n <= 3 and small,
        small,
        running(cards, 10),
        whole(cards, 10),
    )


@cache
def clearing(cards):
    # Bit i is set when center pattern i allows this hand to be erased.
    bits = [ALL if ok else 0 for ok in group(cards)]
    for c in VALUES:
        if cards.count(c) == 4:
            continue
        for r, ok in enumerate(group(tuple(sorted((*cards, c))))):
            if ok:
                bits[r] |= WITH[c]
    for left, right in set(splits(cards)):
        x, y = clearing(left), clearing(right)
        for r in range(len(RULES)):
            bits[r] |= x[r] & y[r]
    return bits


def greedy(hand):
    counts = Counter(hand)
    for a in range(5):
        take = min(counts[a], counts[10 - a])
        counts[a] -= take
        counts[10 - a] -= take
    counts[5] %= 2
    # Each surviving value disappears exactly in centers containing its complement.
    cases = [((), ALL)]
    for v, k in sorted(counts.items()):
        if k:
            mask = WITH[10 - v]
            cases = [(rest, bits & mask) for rest, bits in cases] + [
                (rest + (v,) * k, bits & ~mask) for rest, bits in cases
            ]
    return sum((bits if not rest else bits & clearing(rest)[-1]) for rest, bits in cases)


def survey():
    """Stuck deals for each rule. Greedy failure is conditional on unrestricted solvability."""
    rules = (*RULES, "greedy_stranded")
    result = {deck: {rule: 0 for rule in rules} for deck in ("zero", "no_zero")}
    for row in result.values():
        row.update(total=0, patterns=0, hands=0)
    for hand in combinations_with_replacement(VALUES, 7):
        if max(Counter(hand).values()) > 4:
            continue
        weights = pattern_weights(hand)
        cleared = clearing(hand)
        failed = [ALL ^ bits for bits in cleared]
        failed.append(cleared[-1] & ~greedy(hand))
        for deck, valid in (("zero", ALL), ("no_zero", NONZERO_CENTERS if 0 not in hand else 0)):
            if not valid:
                continue
            row = result[deck]
            row["hands"] += 1
            row["total"] += weighted(valid, weights)
            row["patterns"] += sum(weights[i] > 0 for i in indices(valid))
            for rule, mask in zip(rules, failed, strict=True):
                row[rule] += weighted(mask & valid, weights)
    for deck, n in (("zero", 44), ("no_zero", 40)):
        row = result[deck]
        if row["total"] != comb(n, 7) * comb(n - 7, 4):
            raise ValueError(f"deal weights do not sum to deck total: {deck}")
        row["brackets_only"] = row["running"] - row["unrestricted"]
        row["solvable"] = row["total"] - row["unrestricted"]
    return result
