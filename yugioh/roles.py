"""Cards with more than one role, hand traps going second, and the turn's one Normal Summon.

`odds.py` gives each card one role. Here a deck is a list of **groups**: cards that share a set
of roles, for example 3 cards that are each a starter and an extender, and whether each needs
the turn's one **Normal Summon**. A **hand trap** is a card played from the hand during the
opponent's turn, which is what going second is about; a **breaker** is a card that clears the
opponent's board on the player's own turn. Both are roles like the others.

A **need** is a list of **slots** that distinct cards of the hand must fill, each slot naming
the roles that may fill it, with a limit on how many of the chosen cards use the Normal Summon.
"A starter and an extender" is two slots, so one card that is both cannot meet it alone. "A
starter or a hand trap" is one slot that either role fills. "A starter and an extender with one
Normal Summon" is two slots and a limit of one.

`p_need` sums the multivariate hypergeometric law over the counts of each group, as `odds.py`
does over roles, and decides each hand by a search over groups (`fits`). `p_brute` lists every
hand of physical cards of a small deck and decides it by trying every way to give the slots to
distinct cards of the hand (`fits_cards`), which shares nothing with `fits`. `simulate` deals
seeded hands and decides them the same way as `p_brute`.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, permutations
from math import comb
from random import Random

from yugioh.odds import hands


@dataclass(frozen=True)
class Group:
    """`cards` cards that each have every role in `roles`; `summon` if each needs the Normal Summon."""

    roles: frozenset[str]
    cards: int
    summon: bool = False


def group(cards: int, *roles: str, summon: bool = False) -> Group:
    return Group(frozenset(roles), cards, summon)


@dataclass(frozen=True)
class Need:
    """Slots for distinct cards, each a set of roles that may fill it, and a Normal Summon limit."""

    slots: tuple[frozenset[str], ...]
    summons: int | None = None  # the most chosen cards that need the Normal Summon; None for any


def need(*slots: str, summons: int | None = None) -> Need:
    """A need from slots written as "starter" or "starter|hand trap"."""
    return Need(tuple(frozenset(s.split("|")) for s in slots), summons)


type Deck = tuple[Group, ...]


def size(deck: Deck) -> int:
    return sum(g.cards for g in deck)


def fits(deck: Deck, counts: Sequence[int], want: Need) -> bool:
    """Whether a hand with `counts` cards of each group meets the need: a search over groups."""
    left = list(counts)

    def fill(i: int, summons: int) -> bool:
        if i == len(want.slots):
            return True
        for j, g in enumerate(deck):
            if left[j] == 0 or not g.roles & want.slots[i]:
                continue
            used = summons + g.summon
            if want.summons is not None and used > want.summons:
                continue
            left[j] -= 1
            ok = fill(i + 1, used)
            left[j] += 1
            if ok:
                return True
        return False

    return fill(0, 0)


def p_need(deck: Deck, n: int, want: Need, total: int | None = None) -> Fraction:
    """The exact chance that n cards drawn meet the need.

    The deck has `total` cards (default: the groups alone); cards outside the groups have no role.
    """
    total = size(deck) if total is None else total
    counts = tuple(g.cards for g in deck) + (total - size(deck),)
    ways = sum(w for h, w in hands(counts, n) if fits(deck, h[:-1], want))
    return Fraction(ways, comb(total, n))


def p_going_second(deck: Deck, n: int, want: Need, total: int) -> Fraction:
    """The trap must be in the opening n−1 cards. The last card can fill its other roles."""
    if not 1 <= n <= total or total < size(deck):
        raise ValueError("draw and groups must fit in the deck")
    groups = (*deck, group(total - size(deck)))
    counts = tuple(g.cards for g in groups)
    good = 0
    for last, g in enumerate(groups):
        if not g.cards:
            continue
        rest = tuple(c - (j == last) for j, c in enumerate(counts))
        late = Group(g.roles - {"hand trap"}, 1, g.summon)
        for h, ways in hands(rest, n - 1):
            if fits((*groups, late), (*h, 1), want):
                good += g.cards * ways
    return Fraction(good, total * comb(total - 1, n - 1))


def physical(deck: Deck, total: int) -> list[Group | None]:
    """One entry per card, None for cards with no role."""
    return [g for g in deck for _ in range(g.cards)] + [None] * (total - size(deck))


def fits_cards(hand: Sequence[Group | None], want: Need) -> bool:
    """Whether some slots-to-cards assignment of distinct cards meets the need (the slow check)."""
    for chosen in permutations(range(len(hand)), len(want.slots)):
        cards = [hand[i] for i in chosen]
        if any(c is None or not c.roles & s for c, s in zip(cards, want.slots, strict=True)):
            continue
        summons = sum(c.summon for c in cards if c is not None)
        if want.summons is None or summons <= want.summons:
            return True
    return False


def p_brute(deck: Deck, n: int, want: Need, total: int) -> Fraction:
    """The same chance by listing every set of n physical cards."""
    cards = physical(deck, total)
    good = sum(fits_cards(h, want) for h in combinations(cards, n))
    return Fraction(good, comb(total, n))


def simulate(
    deck: Deck,
    n: int,
    want: Need,
    total: int,
    trials: int,
    seed: int,
    *,
    going_second: bool = False,
) -> int:
    """How many of `trials` seeded random hands meet the need."""
    rng = Random(seed)
    cards = physical(deck, total)
    hits = 0
    for _ in range(trials):
        hand = rng.sample(cards, n)
        last = hand[-1] if going_second else None
        if last is not None:
            hand[-1] = Group(last.roles - {"hand trap"}, 1, last.summon)
        hits += fits_cards(hand, want)
    return hits


def p_naive(deck: Deck, n: int, roles: Sequence[str], total: int) -> Fraction:
    """The chance of at least one card with each role, where one card may count for several.

    This is what a calculator that counts each two-role card in both roles computes.
    """
    counts = tuple(g.cards for g in deck) + (total - size(deck),)
    ways = 0
    for h, w in hands(counts, n):
        if all(any(k and r in g.roles for k, g in zip(h, deck, strict=False)) for r in roles):
            ways += w
    return Fraction(ways, comb(total, n))
