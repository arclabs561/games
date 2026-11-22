"""Exhaustive hand-size, division and congruence tables for report.py."""

from collections import Counter
from fractions import Fraction
from functools import cache
from itertools import combinations, combinations_with_replacement
from math import comb

from exact_play import ALL, CENTERS, VALUES, WITH, indices, pattern_weights, splits, weighted, whole


@cache
def clears(hand):
    if len(hand) >= 2 and whole(hand, 10):
        return ALL
    bits = 0
    for c in VALUES:
        if hand.count(c) < 4 and whole(tuple(sorted((*hand, c))), 10):
            bits |= WITH[c]
    for a, b in set(splits(hand)):
        if bits == ALL:
            break
        bits |= clears(a) & clears(b)
    return bits


@cache
def has_move(hand):
    bits = clears(hand)
    for i in range(len(hand)):
        if bits == ALL:
            break
        if len(hand) > 1:
            bits |= has_move(hand[:i] + hand[i + 1 :])
    return bits


@cache
def div_values(cards, fractions):
    if len(cards) == 1:
        return frozenset(cards)
    out = set()
    for a, b in set(splits(cards)):
        for x in div_values(a, fractions):
            for y in div_values(b, fractions):
                out.update((x + y, x - y, y - x, x * y))
                if y and (fractions or x % y == 0):
                    out.add(Fraction(x, y) if fractions else x // y)
                if x and (fractions or y % x == 0):
                    out.add(Fraction(y, x) if fractions else y // x)
    return frozenset(out)


@cache
def div_whole(cards, t, fractions):
    if len(cards) <= 4:
        return t in div_values(cards, fractions)
    for a, b in set(splits(cards)):
        if len(a) > len(b):
            a, b = b, a
        for x in div_values(a, fractions):
            if x and (fractions or t % x == 0):
                if div_whole(b, Fraction(t, x) if fractions else t // x, fractions):
                    return True
            if x == 0 and t == 0:
                return True  # x times any expression on b
            if x and div_whole(b, t * x, fractions):  # b / x = t
                return True
            if x and t and (fractions or x % t == 0):
                if div_whole(b, Fraction(x, t) if fractions else x // t, fractions):
                    return True  # x / b = t, with nonzero b
            if any(div_whole(b, v, fractions) for v in (t - x, x - t, t + x)):
                return True
    return False


@cache
def some_part(cards, fractions):
    return any(
        div_whole(part, 10, fractions)
        for k in range(1, len(cards) + 1)
        for part in set(combinations(cards, k))
    )


@cache
def div_group(cards, fractions):
    # With a real zero, reaching +10 is equivalent to reaching it on some
    # nonzero submultiset. Dropping zero branches only introduces a possible sign;
    # reversing differences and moving signs preserves positive reachability.
    if 0 in cards:
        return some_part(tuple(v for v in cards if v), fractions)
    for v, n in Counter(cards).items():
        if n >= 2:
            rest = list(cards)
            rest.remove(v)
            rest.remove(v)
            if some_part(tuple(rest), fractions):
                return True  # v-v absorbs the unused cards
    return div_whole(cards, 10, fractions)


@cache
def div_clears(hand, center, fractions):
    for c in (None, *center):
        cards = tuple(sorted((*hand, c))) if c is not None else hand
        if len(cards) > 1 and div_group(cards, fractions):
            return True
    # For a real zero in the hand any first legal move implies a one-group
    # solution, by padding its nonzero witness with the zero and the leftovers.
    if 0 in hand:
        return False
    return any(
        div_clears(a, center, fractions) and div_clears(b, center, fractions)
        for a, b in set(splits(hand))
    )


def shape(h, c, m):
    loud = [v for v in h if v % m]
    if len(loud) > 1:
        return None
    x = next(iter(loud), 0)
    for v in c:
        signed = {0, x, v, x + v, x - v, x * v}
        if any((z - 10) % m == 0 or (-z - 10) % m == 0 for z in signed):
            return None
    return (m, len(loud))


def printed_shapes(hand, center):
    hs, cs = set(hand), set(center)

    def one(rest, extra):
        return sum(v not in rest for v in hand) == 1 and hs <= rest | extra

    return [
        ((4, 0), hs <= {0, 4, 8} and cs.isdisjoint({2, 6, 10})),
        ((3, 0), hs | cs <= {0, 3, 6, 9}),
        ((4, 1), one({0, 4, 8}, {1, 3, 5, 7, 9}) and cs <= {0, 4, 8}),
        ((7, 1), one({0, 7}, {1, 6, 8}) and cs <= {0, 1, 6, 7, 8}),
        ((7, 0), hs <= {0, 7} and cs.isdisjoint({3, 4, 10})),
        ((9, 0), hs <= {0, 9} and cs.isdisjoint({1, 8, 10})),
        ((6, 0), hs <= {0, 6} and cs.isdisjoint({2, 4, 8, 10})),
        ((9, 1), one({0, 9}, {2, 7}) and cs <= {0, 2, 7, 9}),
    ]


def survey():
    """Count hand-size outcomes, and classify all stuck seven-card patterns."""
    sizes = []
    division = {"patterns": 0, "integer": [0, 0], "fraction": [0, 0], "four_zeros": 0}
    shape_deals = Counter()
    missed = Counter()
    missed_centers = {}
    zero_hand = caught_zero = dead_patterns = 0
    for n in range(2, 9):
        stuck = partial = 0
        for hand in combinations_with_replacement(VALUES, n):
            if max(Counter(hand).values()) > 4:
                continue
            failed = ALL ^ clears(hand)
            if not failed:
                continue
            weights = pattern_weights(hand)
            stuck += weighted(failed, weights)
            partial += weighted(failed & has_move(hand), weights)
            if n != 7:
                continue
            for i in indices(failed):
                w = weights[i]
                if not w:
                    continue
                dead_patterns += 1
                center = tuple(sorted(set(CENTERS[i])))
                hits = [name for name, ok in printed_shapes(hand, center) if ok]
                if bool(hits) != any(shape(hand, center, m) for m in (3, 4, 6, 7, 9)):
                    raise ValueError(f"congruence shapes disagree on {hand}, {center}")
                for modulus, loud in hits:
                    shape_deals[f"{modulus}-{loud}"] += w
                zero_hand += w * (0 in hand)
                caught_zero += w * (0 in hand and bool(hits))
                if not hits:
                    missed[hand] += w
                    missed_centers.setdefault(hand, set()).update(center)
                if div_clears(hand, center, False):
                    continue
                no_zero = w if 0 not in hand and 0 not in center else 0
                division["integer"][0] += w
                division["integer"][1] += no_zero
                if not div_clears(hand, center, True):
                    division["fraction"][0] += w
                    division["fraction"][1] += no_zero
                    division["patterns"] += 1
                    division["four_zeros"] += hand.count(0) == 4
        sizes.append(
            {"size": n, "total": comb(44, n) * comb(44 - n, 4), "stuck": stuck, "partial": partial}
        )
    return {
        "sizes": sizes,
        "division": division,
        "shape_deals": dict(sorted(shape_deals.items())),
        "zero_hand": zero_hand,
        "caught_zero": caught_zero,
        "dead_patterns": dead_patterns,
        "missed": [
            {"hand": list(h), "center": sorted(missed_centers[h]), "deals": w}
            for h, w in sorted(missed.items(), key=lambda item: (-item[1], item[0]))
        ],
    }


def make_ten():
    digits = list(combinations_with_replacement(range(10), 4))
    made = [h for h in digits if 10 in div_values(h, True)]
    return {
        "hands": len(digits),
        "made": len(made),
        "fraction_only": [list(h) for h in made if 10 not in div_values(h, False)],
        "distinct_nonzero": sum(len(set(h)) == 4 and 0 not in h for h in made),
    }
