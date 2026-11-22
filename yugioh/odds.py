"""Opening-hand odds: exact multivariate hypergeometric probabilities, as fractions.

Vocabulary, used the same way in every module of this package:

- A **role** is what a card does in an opening hand. A **starter** begins the deck's main
  play by itself (a card that searches a starter counts as one). An **extender** adds to a
  play once a starter has resolved but cannot begin one. A **brick** is dead in the opening
  hand. **Other** is everything else: cards that are playable but neither start nor extend.
- A **deck** is its size and the number of cards in each role (`Deck`). A **named deck** is a
  list of card names, each with a role and 1 to 3 copies (`Name`), for questions in which two
  copies of one card differ from two different cards.
- A **hand** is the number of cards of each role drawn, a tuple (starters, extenders, bricks,
  other). The player who goes first has 5 cards on the first turn, the player who goes
  second 6 (the 5 of the opening hand and the normal draw).
- A **need** is a condition on the hand: `starter`, `starter_and_extender`, `no_brick` and
  others below. A **named need** also sees how many different names of each role were drawn.

`p_hand` sums the multivariate hypergeometric law over hands; `p_named` runs a dynamic
program over names; `p_brute` lists every hand of physical cards; `simulate` deals seeded
random hands. The tests check that the four agree.
"""

from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from itertools import combinations
from math import comb
from random import Random

ROLES = ("starter", "extender", "brick", "other")
FIRST, SECOND = 5, 6  # cards in hand on the first turn, going first and going second

Hand = tuple[int, ...]  # cards drawn of each role, in the order of ROLES
Need = Callable[[Hand], bool]
NamedNeed = Callable[[Hand, Hand], bool]  # (cards drawn per role, different names per role)


@dataclass(frozen=True)
class Deck:
    """A deck of `size` cards; the cards that are not starters, extenders or bricks are other."""

    size: int
    starters: int = 0
    extenders: int = 0
    bricks: int = 0

    def __post_init__(self):
        if self.other < 0 or min(self.starters, self.extenders, self.bricks) < 0:
            raise ValueError(f"roles do not fit in the deck: {self}")

    @property
    def other(self) -> int:
        """Cards that are neither starters, extenders nor bricks."""
        return self.size - self.starters - self.extenders - self.bricks

    def counts(self) -> Hand:
        """Cards of each role, in the order of ROLES."""
        return (self.starters, self.extenders, self.bricks, self.other)


@dataclass(frozen=True)
class Name:
    """One card name in a named deck: its role and its number of copies (1 to 3)."""

    name: str
    role: str
    copies: int

    def __post_init__(self):
        if self.role not in ROLES or not 1 <= self.copies <= 3:
            raise ValueError("a named card needs a known role and one to three copies")


# Needs. Each takes a hand (s, e, b, o).


def starter(h: Hand) -> bool:
    """At least one starter."""
    return h[0] >= 1


def starter_and_extender(h: Hand) -> bool:
    """At least one starter and at least one extender."""
    return h[0] >= 1 and h[1] >= 1


def two_engine(h: Hand) -> bool:
    """At least one starter and a second engine card: another starter or an extender."""
    return h[0] >= 1 and h[0] + h[1] >= 2


def no_brick(h: Hand) -> bool:
    """No brick."""
    return h[2] == 0


def starter_no_brick(h: Hand) -> bool:
    """At least one starter and no brick."""
    return starter(h) and no_brick(h)


NEEDS: dict[str, Need] = {
    "at least one starter": starter,
    "a starter and an extender": starter_and_extender,
    "a starter and another engine card": two_engine,
    "no brick": no_brick,
    "a starter and no brick": starter_no_brick,
}


def two_different_starters(counts: Hand, distinct: Hand) -> bool:
    """Starters of at least two different names (a named need)."""
    return distinct[0] >= 2


def by_role(need: Need) -> NamedNeed:
    """A need on roles, seen as a named need that ignores names."""
    return lambda counts, distinct: need(counts)


# Exact: the multivariate hypergeometric law.


def hands(counts: Hand, n: int) -> Iterator[tuple[Hand, int]]:
    """Every hand of n cards from a deck with these role counts, with its number of ways."""
    if len(counts) == 1:
        if n <= counts[0]:
            yield (n,), comb(counts[0], n)
        return
    for k in range(min(n, counts[0]) + 1):
        for rest, ways in hands(counts[1:], n - k):
            yield (k, *rest), comb(counts[0], k) * ways


def p_hand(deck: Deck, n: int, need: Need) -> Fraction:
    """Chance that n cards drawn from the deck meet the need: exact, by summing over hands."""
    ways = sum(w for h, w in hands(deck.counts(), n) if need(h))
    return Fraction(ways, comb(deck.size, n))


def p_at_least_one(size: int, k: int, n: int) -> Fraction:
    """Chance of at least one of k given cards in n drawn from size: 1 − C(size − k, n)/C(size, n)."""
    return 1 - Fraction(comb(size - k, n), comb(size, n))


def p_starter_and_extender(deck: Deck, n: int) -> Fraction:
    """The same as p_hand(deck, n, starter_and_extender), by inclusion and exclusion."""
    s, e = deck.starters, deck.extenders
    misses = comb(deck.size - s, n) + comb(deck.size - e, n) - comb(deck.size - s - e, n)
    return 1 - Fraction(misses, comb(deck.size, n))


def fewest(target: Fraction | float, size: int, n: int, need_cards=p_at_least_one) -> int | None:
    """Fewest starters (or other cards) k with need_cards(size, k, n) at least the target."""
    for k in range(size + 1):
        if need_cards(size, k, n) >= target:
            return k
    return None


# Named decks: the same chances when copies matter.


def named(deck: Deck, copies: int = 3) -> list[Name]:
    """A named deck with the roles of `deck`, each role split into names of `copies` copies."""
    if not 1 <= copies <= 3:
        raise ValueError("use one to three copies per name")
    names = []
    for role, count in zip(ROLES, deck.counts(), strict=True):
        i = 0
        while count > 0:
            c = min(copies, count)
            names.append(Name(f"{role} {i + 1}", role, c))
            count -= c
            i += 1
    return names


def merged_names(names: Sequence[Name]) -> list[Name]:
    """Combine rows of the same card, enforcing one role and the copy limit."""
    cards: dict[str, Name] = {}
    for card in names:
        previous = cards.get(card.name)
        if previous is not None:
            if previous.role != card.role:
                raise ValueError(f"{card.name} has conflicting roles")
            card = Name(card.name, card.role, previous.copies + card.copies)
        cards[card.name] = card
    return list(cards.values())


def p_named(names: Sequence[Name], n: int, need: NamedNeed) -> Fraction:
    """Chance that n cards drawn meet the named need: exact, by a dynamic program over names.

    The state after some names is (cards drawn, cards per role, different names per role),
    weighted by the number of ways to draw it; drawing j of a name's c copies has C(c, j) ways.
    """
    names = merged_names(names)
    states: Counter[tuple[int, Hand, Hand]] = Counter({(0, (0,) * 4, (0,) * 4): 1})
    for card in names:
        r = ROLES.index(card.role)
        nxt: Counter[tuple[int, Hand, Hand]] = Counter()
        for (drawn, counts, distinct), ways in states.items():
            for j in range(min(card.copies, n - drawn) + 1):
                c = tuple(x + j * (i == r) for i, x in enumerate(counts))
                d = tuple(x + (j > 0) * (i == r) for i, x in enumerate(distinct))
                nxt[drawn + j, c, d] += ways * comb(card.copies, j)
        states = nxt
    size = sum(card.copies for card in names)
    good = sum(w for (drawn, c, d), w in states.items() if drawn == n and need(c, d))
    return Fraction(good, comb(size, n))


def physical(names: Sequence[Name]) -> list[Name]:
    """One entry per physical card: a name with 3 copies appears three times."""
    return [card for card in merged_names(names) for _ in range(card.copies)]


def summary(hand: Sequence[Name]) -> tuple[Hand, Hand]:
    """Cards per role and different names per role in a hand of physical cards."""
    counts = tuple(sum(c.role == r for c in hand) for r in ROLES)
    distinct = tuple(len({c.name for c in hand if c.role == r}) for r in ROLES)
    return counts, distinct


def p_brute(names: Sequence[Name], n: int, need: NamedNeed) -> Fraction:
    """The same chance by listing every set of n physical cards (the slow check)."""
    cards = physical(names)
    good = total = 0
    for hand in combinations(cards, n):
        total += 1
        good += need(*summary(hand))
    return Fraction(good, total)


def simulate(names: Sequence[Name], n: int, need: NamedNeed, trials: int, seed: int) -> int:
    """Number of seeded random hands, out of `trials`, that meet the need."""
    rng = Random(seed)
    cards = physical(names)
    return sum(need(*summary(rng.sample(cards, n))) for _ in range(trials))


def wilson(hits: int, trials: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    p = hits / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * ((p * (1 - p) / trials + z * z / (4 * trials**2)) ** 0.5) / (1 + z * z / trials)
    return centre - half, centre + half


@cache
def at_least_one_table(size: int, n: int, top: int = 20) -> list[Fraction]:
    """P(at least one starter) for 0..top starters."""
    return [p_at_least_one(size, k, n) for k in range(top + 1)]


if __name__ == "__main__":
    deck = Deck(40, starters=12, extenders=9, bricks=3)
    for label, need in NEEDS.items():
        a, b = p_hand(deck, FIRST, need), p_hand(deck, SECOND, need)
        print(f"{label:36} {float(a):7.2%} {float(b):7.2%}")
