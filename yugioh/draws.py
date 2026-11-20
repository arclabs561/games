"""Draw and search cards in the opening hand, modeled simply.

Three generic kinds of card, each modeled as follows and nothing more:

- A **cantrip** is played at once and replaced by the top card of the deck ("draw 1"). Any
  number can be played.
- A **draw-two** is played and replaced by the top two cards ("draw 2"). Only one can be
  played per turn; a second copy stays in hand as a dead card, counted as other. Any cost
  it has (cards banished, life points) is ignored.
- A **looker** is played after all draws: it reveals the top `look` cards and takes one of
  them into the hand; the rest leave. Only one is played per turn. It takes a card that
  makes the hand meet the need, if one of the revealed cards does, and is not played when
  the hand already meets it.

A card that searches a starter from the deck is already counted as a starter (see
`yugioh.odds`), so search needs no model of its own. The engine cards are counted as other
in the final hand. Every card is played whenever it can be, with no cost to the rest of the
turn: these numbers are upper bounds on what the cards give in play.

`p_engine` is exact, by recursion over the order of the draws; `simulate_engine` shuffles
and deals seeded decks with separate code. A proved identity checks both: c cantrips in a
deck of N + c cards give the same hands as N cards without them.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from math import comb
from random import Random

from yugioh.odds import FIRST, Deck, Hand, Need, hands

# Kinds of card in an engine deck, in this order.
STARTER, EXTENDER, BRICK, OTHER, CANTRIP, DRAW2, LOOKER = range(7)


@dataclass(frozen=True)
class Engine:
    """Draw and search cards added to a deck; `look` is how many cards a looker reveals."""

    cantrips: int = 0
    draw2: int = 0
    lookers: int = 0
    look: int = 6

    @property
    def cards(self) -> int:
        """Engine cards in the deck."""
        return self.cantrips + self.draw2 + self.lookers


def kinds(deck: Deck, engine: Engine) -> tuple[int, ...]:
    """Cards of each kind; the engine cards come out of the deck's other cards."""
    s, e, b, o = deck.counts()
    if o < engine.cards:
        raise ValueError("the engine cards do not fit among the other cards")
    return (s, e, b, o - engine.cards, engine.cantrips, engine.draw2, engine.lookers)


def add(hand: Hand, role: int) -> Hand:
    """The hand with one more card of role 0 to 3."""
    return tuple(x + (i == role) for i, x in enumerate(hand))


def reveals(remaining: tuple[int, ...], k: int) -> Iterator[tuple[Hand, int]]:
    """Every set of k revealed cards, by role (engine cards count as other), with its ways."""
    s, e, b, o, *engine = remaining
    yield from hands((s, e, b, o + sum(engine)), min(k, sum(remaining)))


def p_engine(deck: Deck, engine: Engine, n: int, need: Need) -> Fraction:
    """Chance that the hand meets the need after the engine cards are played: exact."""
    look = engine.look

    @cache
    def finish(remaining: tuple[int, ...], hand: Hand, looker: bool) -> Fraction:
        """The chance once all draws are done; a looker in hand is played if it helps."""
        if need(hand):
            return Fraction(1)
        if not looker:
            return Fraction(0)
        total = comb(sum(remaining), min(look, sum(remaining)))
        good = sum(
            w
            for seen, w in reveals(remaining, look)
            if any(seen[r] and need(add(hand, r)) for r in range(4))
        )
        return Fraction(good, total)

    @cache
    def draw(remaining: tuple[int, ...], hand: Hand, owed: int, used2: bool, looker: bool):
        """The chance with `owed` cards still to draw from `remaining`."""
        if owed == 0:
            return finish(remaining, hand, looker)
        left = sum(remaining)
        p = Fraction(0)
        for kind, count in enumerate(remaining):
            if count == 0:
                continue
            rest = tuple(x - (i == kind) for i, x in enumerate(remaining))
            if kind == CANTRIP:
                q = draw(rest, hand, owed, used2, looker)  # played: replaced by one card
            elif kind == DRAW2 and not used2:
                q = draw(rest, hand, owed + 1, True, looker)  # played: replaced by two
            elif kind == LOOKER and not looker:
                q = draw(rest, hand, owed - 1, used2, True)
            else:
                role = kind if kind < 4 else OTHER  # a second draw-two or looker is dead
                q = draw(rest, add(hand, role), owed - 1, used2, looker)
            p += Fraction(count, left) * q
        return p

    return draw(kinds(deck, engine), (0, 0, 0, 0), n, False, False)


def deal(order: list[int], n: int, look: int) -> tuple[Hand, list[int]]:
    """Walk a shuffled deck: the hand after cantrips and one draw-two, and the cards left."""
    hand = [0, 0, 0, 0]
    top, owed, used2, looker = 0, n, False, False
    while owed > 0:
        kind = order[top]
        top += 1
        if kind == CANTRIP:
            continue
        if kind == DRAW2 and not used2:
            used2, owed = True, owed + 1
            continue
        owed -= 1
        if kind == LOOKER and not looker:
            looker = True
        else:
            hand[kind if kind < 4 else OTHER] += 1
    return tuple(hand), (order[top : top + look] if looker else [])


def simulate_engine(deck: Deck, engine: Engine, n: int, need: Need, trials: int, seed: int) -> int:
    """Seeded random decks, out of `trials`, whose hand meets the need after the engine."""
    rng = Random(seed)
    cards = [kind for kind, count in enumerate(kinds(deck, engine)) for _ in range(count)]
    hits = 0
    for _ in range(trials):
        rng.shuffle(cards)
        hand, revealed = deal(cards, n, engine.look)
        ok = need(hand) or any(need(add(hand, min(k, OTHER))) for k in revealed)
        hits += ok
    return hits


def starter_equivalent(deck: Deck, engine: Engine, n: int, need: Need) -> Fraction:
    """How many starters, added in place of other cards, give the chance the engine gives.

    Found by linear interpolation between the whole numbers of extra starters around it.
    """
    target = p_engine(deck, engine, n, need)
    base = [Deck(deck.size, deck.starters + k, deck.extenders, deck.bricks) for k in range(0, 8)]
    values = [p_engine(d, Engine(), n, need) for d in base if d.other >= 0]
    for k in range(len(values) - 1):
        lo, hi = values[k], values[k + 1]
        if lo <= target <= hi and hi > lo:
            return k + (target - lo) / (hi - lo)
    raise ValueError("the engine is worth more than seven starters")


if __name__ == "__main__":
    from yugioh.odds import starter

    d = Deck(40, starters=12)
    for eng in (Engine(), Engine(draw2=3), Engine(lookers=3)):
        print(eng, f"{float(p_engine(d, eng, FIRST, starter)):.4f}")
