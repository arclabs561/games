"""Optimal play by plain recursion, for checking `solver` on small versions of the game.

Nothing here is shared with the solver except the scoring rules in `rules.py`. The dice are
ordered (a roll of three dice is one of faces^3 tuples), a keep is any subset of their
positions, and the state is the whole score card with every box's points, not the compressed
state of the solver. Arithmetic is exact, with fractions. The cost grows like faces^dice squared
times the number of score cards, so only games with a few dice, faces and boxes are feasible.
"""

from collections.abc import Callable
from fractions import Fraction
from functools import cache
from itertools import product

from yahtzee.rules import UPPER, Rules, bonus, box_score, legal

type Card = tuple[int | None, ...]  # points per box, None while open


def brute_force(rules: Rules, utility: Callable[[int], Fraction] | None = None) -> Fraction:
    """The expected final score under optimal play, from an empty score card.

    With `utility`, the expected utility of the final score under the play that maximizes it,
    for example the chance of scoring at least T with `lambda s: Fraction(s >= T)`.
    """
    cats = rules.categories
    faces = range(1, rules.faces + 1)

    def final(card: Card, extra: int) -> int:
        upper = sum(p for c, p in zip(cats, card, strict=True) if c in UPPER and p is not None)
        earned = rules.upper_bonus if upper >= rules.threshold else 0
        return sum(p for p in card if p is not None) + extra + earned

    @cache
    def game(card: Card, extra: int) -> Fraction:
        """Expected final value from the start of a turn; `extra` is the Yahtzee bonus so far."""
        if None not in card:
            s = final(card, extra)
            return Fraction(s) if utility is None else utility(s)
        rolls = list(product(faces, repeat=rules.dice))
        return sum((roll_value(card, extra, d, rules.rolls - 1) for d in rolls), Fraction(0)) / len(
            rolls
        )

    @cache
    def roll_value(card: Card, extra: int, dice: tuple[int, ...], left: int) -> Fraction:
        """Expected final score after a roll showing `dice`, with `left` rerolls to come."""
        best = score_value(card, extra, tuple(sorted(dice)))
        if left == 0:
            return best
        n = len(dice)
        for kept in product((False, True), repeat=n):
            free = [i for i in range(n) if not kept[i]]
            total = Fraction(0)
            for new in product(faces, repeat=len(free)):
                d = list(dice)
                for i, v in zip(free, new, strict=True):
                    d[i] = v
                total += roll_value(card, extra, tuple(d), left - 1)
            best = max(best, total / rules.faces ** len(free))
        return best

    @cache
    def score_value(card: Card, extra: int, roll: tuple[int, ...]) -> Fraction:
        """Expected final score after filling the best open box with this roll."""
        open_boxes = tuple(c for c, p in zip(cats, card, strict=True) if p is None)
        filled = frozenset(c for c, p in zip(cats, card, strict=True) if p is not None)
        y50 = "yahtzee" in cats and card[cats.index("yahtzee")] == 50
        best = None
        for c in legal(rules, roll, open_boxes):
            i = cats.index(c)
            new = card[:i] + (box_score(rules, roll, c, filled),) + card[i + 1 :]
            v = game(new, extra + bonus(rules, roll, y50))
            best = v if best is None else max(best, v)
        assert best is not None
        return best

    return game((None,) * len(cats), 0)
