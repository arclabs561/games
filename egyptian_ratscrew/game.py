"""Egyptian Ratscrew: dealing, slapping, challenges and whole games.

Suits never matter, so a card is just its rank: 2 to 10, or J, Q, K, A. A royal challenges
the next player to lay another royal within 1, 2, 3 or 4 cards; a pair or a sandwich on top
of the pile (the same rank twice in a row, or with one card between) can be slapped, and the
slapper takes the pile.

    uv run python -m egyptian_ratscrew.game
"""

import random
from collections import deque
from collections.abc import Sequence
from fractions import Fraction
from functools import cache
from math import comb
from typing import NamedTuple

type Card = int | str
CHANCES: dict[Card, int] = {"J": 1, "Q": 2, "K": 3, "A": 4}
ROYALS: tuple[str, ...] = ("J", "Q", "K", "A")
RANKS: tuple[Card, ...] = (*range(2, 11), *ROYALS)
assert tuple(CHANCES) == ROYALS
DECK: tuple[Card, ...] = tuple(rank for rank in RANKS for _ in range(4))

assert len(DECK) == 52


def slappable(pile: Sequence[Card]) -> bool:
    """A pair (the top two cards match) or a sandwich (the top card matches the third)."""
    return (len(pile) >= 2 and pile[-1] == pile[-2]) or (len(pile) >= 3 and pile[-1] == pile[-3])


assert not slappable([]) and not slappable(["Q"])
assert slappable(["Q", "Q"]) and slappable([2, "Q", 2])
assert not slappable(["Q", "K"]) and not slappable([2, "Q", 3])


def deal(players: int, rng: random.Random) -> list[deque[Card]]:
    """Shuffle a deck and deal it round the table; the left end of a hand is its top."""
    deck = list(DECK)
    rng.shuffle(deck)
    return [deque(deck[i::players]) for i in range(players)]


def answer_chance(chances: int, royals: int = 15, others: int = 36) -> Fraction:
    """The chance that a challenged player turns up a royal within `chances` cards, when the
    unseen cards are `royals` royals and `others` other cards in random order."""
    return 1 - Fraction(comb(others, chances), comb(royals + others, chances))


@cache
def challenger_keeps(chances: int, royals: tuple[int, int, int, int], others: int) -> Fraction:
    """The chance that the player who laid the last royal takes the pile, with no slaps.

    The challenged player owes `chances` cards; `royals` counts the unseen J, Q, K and A, and
    `others` the unseen other cards, all in random order. Each royal the challenged player
    lays makes them the challenger, so the roles swap.
    """
    if chances == 0:
        return Fraction(1)
    unseen = sum(royals) + others
    if unseen == 0:
        return Fraction(1)
    keep = Fraction(others, unseen) * challenger_keeps(chances - 1, royals, others - 1)
    for i, count in enumerate(royals):
        if count:
            fewer = royals[:i] + (count - 1,) + royals[i + 1 :]
            swapped = 1 - challenger_keeps(CHANCES[ROYALS[i]], fewer, others)
            keep += Fraction(count, unseen) * swapped
    return keep


def first_challenge(royal: str) -> Fraction:
    """`challenger_keeps` at the start of a game: only the challenging royal has been seen."""
    royals = tuple(4 - (r == royal) for r in ROYALS)
    return challenger_keeps(CHANCES[royal], royals, 36)


assert answer_chance(1) == Fraction(15, 51)
assert challenger_keeps(4, (0, 0, 0, 0), 10) == 1  # no royals left to answer with
assert first_challenge("A") < first_challenge("J")  # more chances to answer an ace


def simulate_challenge(royal: str, rng: random.Random) -> bool:
    """Play one challenge out with no slaps, from the 51 unseen cards in random order. True if
    the player who laid `royal` takes the pile. A check on `first_challenge`."""
    unseen = list(DECK)
    unseen.remove(royal)
    rng.shuffle(unseen)
    owed, challenger = CHANCES[royal], 0
    for card in unseen:
        if card in CHANCES:
            challenger, owed = 1 - challenger, CHANCES[card]
        else:
            owed -= 1
            if owed == 0:
                break
    return challenger == 0


class Game(NamedTuple):
    """How one game went: cards laid, the winner (None if unfinished), the number of times a
    pile was picked up, and the number of cards paid for misslaps."""

    cards: int
    winner: int | None
    tricks: int
    paid: int


def play_game(
    slap_shares: Sequence[float],
    rng: random.Random,
    max_cards: int = 100_000,
    hands: Sequence[Sequence[Card]] | None = None,
    slaps: bool = True,
    misslaps: Sequence[float] | None = None,
) -> Game:
    """Play one game.

    Every slappable pile is slapped, and player i wins the slap with chance slap_shares[i];
    players with no cards may still slap back in. The game ends when one player holds every
    card that is not on the pile and no challenge is open. The winner is None if that has not
    happened after max_cards cards. `hands` replaces the random deal; with slaps=False the
    game is Beggar-my-neighbour.

    With `misslaps`, after each card that cannot be slapped player i slaps by mistake with
    chance misslaps[i], and pays for it with the top card of their hand, face down under the
    pile, where it goes to whoever takes the pile. A player with no cards who misslaps is out
    of the game and slaps no more. A pile left on the table at the end counts as a trick, the
    last one, which the winner picks up.
    """
    players = len(slap_shares)
    hands = deal(players, rng) if hands is None else [deque(h) for h in hands]
    pile: list[Card] = []
    burned: list[Card] = []  # cards paid for misslaps, face down under the pile
    out: set[int] = set()  # players who misslapped with no cards to pay
    challenger: int | None = None  # who laid the royal now being answered
    owed = 0  # cards the current player still owes the challenger
    player = 0
    tricks = paid = 0

    def next_with_cards(i: int) -> int:
        """The next player after i who holds cards, or i itself if nobody else does."""
        for step in range(1, players):
            j = (i + step) % players
            if hands[j]:
                return j
        return i

    def collect(winner: int) -> None:
        nonlocal player, challenger, owed, tricks
        hands[winner].extend(burned)
        hands[winner].extend(pile)
        burned.clear()
        pile.clear()
        player, challenger, owed = winner, None, 0
        tricks += 1

    def slapper() -> int | None:
        """Who wins a slap: drawn by slap share among the players still in the game."""
        if not out:
            return rng.choices(range(players), weights=slap_shares)[0]
        weights = [0 if i in out else s for i, s in enumerate(slap_shares)]
        return rng.choices(range(players), weights=weights)[0] if any(weights) else None

    for laid in range(max_cards):
        if challenger is None and sum(map(bool, hands)) == 1:
            last = tricks + bool(pile or burned)
            return Game(laid, next(i for i in range(players) if hands[i]), last, paid)
        if not hands[player]:
            if challenger is not None:  # cannot answer: the challenger takes the pile
                collect(challenger)
            else:
                player = next_with_cards(player)
        card = hands[player].popleft()
        pile.append(card)
        if slaps and slappable(pile) and (winner := slapper()) is not None:
            collect(winner)
            continue
        if misslaps:
            for i, chance in enumerate(misslaps):
                if chance and i not in out and rng.random() < chance:
                    if hands[i]:
                        burned.append(hands[i].popleft())
                        paid += 1
                    else:
                        out.add(i)
                        if challenger == i:  # an open challenge lapses with its player
                            challenger, owed = None, 0
        if card in CHANCES:
            challenger, owed = player, CHANCES[card]
            player = next_with_cards(player)
            if player == challenger:  # nobody else can answer
                collect(challenger)
        elif challenger is not None:
            owed -= 1
            if owed == 0 or not hands[player]:
                collect(challenger)
        else:
            player = next_with_cards(player)
    return Game(max_cards, None, tricks, paid)


def play(
    slap_shares: Sequence[float],
    rng: random.Random,
    max_cards: int = 100_000,
    hands: Sequence[Sequence[Card]] | None = None,
    slaps: bool = True,
) -> tuple[int, int | None]:
    """Play one game with `play_game` and return (cards laid, winner)."""
    game = play_game(slap_shares, rng, max_cards, hands, slaps)
    return game.cards, game.winner


def main() -> None:
    print("Chance that a challenge is answered, and that the challenger keeps the pile,")
    print("at the start of a game with no slaps:")
    for royal in ROYALS:
        n = CHANCES[royal]
        print(
            f"  {royal} ({n} card{'s' * (n > 1)}): answered {float(answer_chance(n)):.3f}, "
            f"challenger keeps {float(first_challenge(royal)):.3f}"
        )


if __name__ == "__main__":
    main()
