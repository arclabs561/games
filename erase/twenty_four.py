#!/usr/bin/env python3
"""The 24 game: which hands of four cards from 1 to 13 make 24 with +, -, x and /?

Each card is used once, any bracketing is allowed, and arithmetic is exact (Fraction).
Three rules for division are counted:

    fraction   any division by a nonzero value (the usual rule)
    integer    every intermediate result is a whole number (division only when exact)
    nodiv      no division at all

    python3 erase/twenty_four.py            # counts for all 1,820 hands (about 10 s)
    python3 erase/twenty_four.py 3 3 8 8    # one expression per rule, or "none"
    python3 erase/twenty_four.py --check    # also check every hand by brute force (about 2 min)
"""

import sys
from fractions import Fraction
from functools import cache
from itertools import combinations, combinations_with_replacement, permutations, product
from math import comb, prod

TARGET = 24
CARDS = range(1, 14)
RULES = ("fraction", "integer", "nodiv")


def combine(a, b, ea, eb, rule):
    """(value, expression) pairs from one operation on a and b, in both orders."""
    out = [(a + b, f"({ea}+{eb})"), (a - b, f"({ea}-{eb})"), (b - a, f"({eb}-{ea})")]
    out.append((a * b, f"({ea}*{eb})"))
    if rule != "nodiv":
        for x, y, ex, ey in ((a, b, ea, eb), (b, a, eb, ea)):
            if y != 0 and (rule == "fraction" or x % y == 0):
                out.append((x / y, f"({ex}/{ey})"))
    return out


@cache
def table(cards, rule):
    """value -> one expression, for every value the sorted tuple of cards can make."""
    if len(cards) == 1:
        return {Fraction(cards[0]): str(cards[0])}
    out = {}
    n = len(cards)
    # Every expression splits at its root into two nonempty parts; take each split once.
    for k in range(1, n // 2 + 1):
        for idx in combinations(range(n), k):
            if 2 * k == n and 0 not in idx:  # an even split: the other half was already taken
                continue
            a_cards = tuple(cards[i] for i in idx)
            b_cards = tuple(cards[i] for i in range(n) if i not in idx)
            for a, ea in table(a_cards, rule).items():
                for b, eb in table(b_cards, rule).items():
                    for v, e in combine(a, b, ea, eb, rule):
                        out.setdefault(v, e)
    return out


def solve(hand, rule="fraction"):
    """An expression using each card of the hand once with value 24, or None."""
    return table(tuple(sorted(hand)), rule).get(Fraction(TARGET))


HANDS = list(combinations_with_replacement(CARDS, 4))


# Brute force, sharing no code with the search above: every order of the four cards, every
# choice of three operators, and each of the five bracketings of four leaves.
SHAPES = (
    "(({0}{4}{1}){5}{2}){6}{3}",
    "({0}{4}({1}{5}{2})){6}{3}",
    "({0}{4}{1}){5}({2}{6}{3})",
    "{0}{4}(({1}{5}{2}){6}{3})",
    "{0}{4}({1}{5}({2}{6}{3}))",
)
OPS = {
    "+": lambda x, y: x + y,
    "-": lambda x, y: x - y,
    "*": lambda x, y: x * y,
    "/": lambda x, y: x / y,
}


def evaluate(shape, cards, ops, rule):
    """The exact value of one bracketing, or None if a step breaks the rule."""
    a, b, c, d = (Fraction(v) for v in cards)
    x, y, z = ops

    def op(o, p, q):
        if p is None or q is None:
            return None
        if o == "/" and (q == 0 or rule == "nodiv" or (rule == "integer" and p % q != 0)):
            return None
        return OPS[o](p, q)

    match shape:
        case 0:
            return op(z, op(y, op(x, a, b), c), d)
        case 1:
            return op(z, op(x, a, op(y, b, c)), d)
        case 2:
            return op(y, op(x, a, b), op(z, c, d))
        case 3:
            return op(x, a, op(z, op(y, b, c), d))
        case _:
            return op(x, a, op(y, b, op(z, c, d)))


def brute(hand, rule="fraction"):
    """Every expression for the hand with value 24, written out, by exhaustive listing."""
    found = []
    for cards in set(permutations(hand)):
        for ops in product(OPS, repeat=3):
            if rule == "nodiv" and "/" in ops:
                continue
            for s, shape in enumerate(SHAPES):
                if evaluate(s, cards, ops, rule) == TARGET:
                    found.append(shape.format(*cards, *ops))
    return found


def solvable(rule):
    """The hands that make 24 under a rule."""
    return [h for h in HANDS if solve(h, rule)]


def deck_probability(hands):
    """Chance that four cards dealt from a 52-card deck form one of these hands."""
    return Fraction(sum(prod(comb(4, h.count(v)) for v in set(h)) for h in hands), comb(52, 4))


def main():
    args = sys.argv[1:]
    if args and args[0] != "--check":
        hand = tuple(int(a) for a in args)
        for rule in RULES:
            print(f"{rule:9}", solve(hand, rule) or "none")
        return
    counts = {rule: solvable(rule) for rule in RULES}
    for rule, hands in counts.items():
        p = deck_probability(hands)
        print(f"{rule:9} {len(hands):5} of {len(HANDS)}   from a deck {float(p):.4f} ({p})")
    need_fraction = sorted(set(counts["fraction"]) - set(counts["integer"]))
    print(f"{len(need_fraction)} hands need a fraction:")
    for h in need_fraction:
        print("  ", *h, "  ", solve(h))
    if "--check" in args:
        for rule in RULES:
            assert [h for h in HANDS if brute(h, rule)] == counts[rule], rule
        print("brute force agrees on every hand for all three rules")


if __name__ == "__main__":
    main()
