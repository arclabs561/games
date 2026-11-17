"""Erase: can a hand be cleared by making 10 with +, - and x?

A group is some hand cards plus at most one center card, at least two cards in all,
combined with +, - and x (each card used once) to make 10. The hand cards in a group
are erased; center cards stay and may be reused.

    python3 erase/erase.py 4 4 6 9 5 8 5 --center 7 5 1 8
    python3 erase/erase.py 0 4 4 4 8 8 8 --center 0 3 7 9
"""

import argparse
from functools import lru_cache
from itertools import combinations


@lru_cache(maxsize=None)
def table(ms):
    """value -> expression, for a small sorted tuple of cards (each used once)."""
    if len(ms) == 1:
        return {ms[0]: str(ms[0])}
    out = {}
    n = len(ms)
    for k in range(1, n // 2 + 1):
        for idx in combinations(range(n), k):
            a_cards = tuple(ms[i] for i in idx)
            b_cards = tuple(ms[i] for i in range(n) if i not in idx)
            for a, ea in table(a_cards).items():
                for b, eb in table(b_cards).items():
                    for v, e in (
                        (a + b, f"({ea}+{eb})"),
                        (a - b, f"({ea}-{eb})"),
                        (b - a, f"({eb}-{ea})"),
                        (a * b, f"({ea}*{eb})"),
                    ):
                        out.setdefault(v, e)
    return out


SMALL = 4


@lru_cache(maxsize=None)
def make(ms, t):
    """An expression using every card of the sorted tuple ms exactly once with value t, or None.

    Every expression splits at its root into two parts; take the smaller part A, list all
    of its values a, and ask recursively whether the larger part can make what is missing.
    """
    n = len(ms)
    if n <= SMALL:
        return table(ms).get(t)
    seen = set()
    for k in range(1, n // 2 + 1):
        for idx in combinations(range(n), k):
            a_cards = tuple(ms[i] for i in idx)
            if a_cards in seen:
                continue
            seen.add(a_cards)
            b_cards = tuple(ms[i] for i in range(n) if i not in idx)
            for a, ea in table(a_cards).items():
                for need, fmt in (
                    (t - a, "({ea}+{eb})"),
                    (a - t, "({ea}-{eb})"),
                    (t + a, "({eb}-{ea})"),
                ):
                    eb = make(b_cards, need)
                    if eb:
                        return fmt.format(ea=ea, eb=eb)
                if a != 0 and t % a == 0:
                    eb = make(b_cards, t // a)
                    if eb:
                        return f"({ea}*{eb})"
                if a == 0 and t == 0:  # zero times anything
                    return f"({ea}*({'+'.join(map(str, b_cards))}))"
    return None


def multisets(k, avail, lo=0):
    """Sorted tuples of k card values, with at most avail[v] copies of each value v."""
    if k == 0:
        yield ()
        return
    for v in range(lo, 11):
        if avail[v]:
            avail[v] -= 1
            for rest in multisets(k - 1, avail, v):
                yield (v,) + rest
            avail[v] += 1


def in_families(m):
    """The 3-family (all multiples of 3) or the 4-family (no 2, 6, 10; at most one odd card)."""
    three = all(v % 3 == 0 for v in m)
    four = not {2, 6, 10} & set(m) and sum(v % 2 for v in m) <= 1
    return three or four


def group(hand_cards, center):
    """An expression making 10 from these hand cards plus at most one center card, or None."""
    for extra in [()] + [(c,) for c in sorted(set(center))]:
        cards = tuple(sorted(hand_cards + extra))
        if len(cards) > 1:
            e = make(cards, 10)
            if e:
                return e + (f"   [center {extra[0]}]" if extra else "")
    return None


def solve(hand, center):
    """A list of groups that erases the whole hand, or None."""
    hand = tuple(sorted(hand))
    if not hand:
        return []
    first, rest = hand[0], hand[1:]
    for k in range(len(rest) + 1):
        for pick in combinations(range(len(rest)), k):
            g = group((first,) + tuple(rest[i] for i in pick), center)
            if g:
                tail = solve(tuple(rest[i] for i in range(len(rest)) if i not in pick), center)
                if tail is not None:
                    return [g] + tail
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("hand", type=int, nargs="+")
    ap.add_argument("--center", type=int, nargs="*", default=[])
    a = ap.parse_args()
    groups = solve(tuple(a.hand), tuple(a.center))
    if groups is None:
        print("cannot be cleared")
    else:
        for g in groups:
            print(g)


if __name__ == "__main__":
    main()
