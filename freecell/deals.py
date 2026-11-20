"""Cards, decks and the Microsoft deal numbers.

A **card** is an integer: 4 × (rank − 1) + suit, with ranks 1 (ace) to 13 (king) and suits
in the order clubs, diamonds, hearts, spades (0 to 3). This is the order of the Microsoft
deck, so the ace of clubs is 0 and the king of spades is 51. Diamonds and hearts are red.

A **layout** is a tuple of columns; each column is a tuple of cards from the bottom (dealt
first, deepest) to the top (the card that can move).

Microsoft FreeCell numbers its deals. Deal n seeds the C library's random number generator
with n, and the generator is

    state ← (214013 × state + 2531011) mod 2³¹,   rand = state ÷ 2¹⁶ (0 to 32767).

Starting from the deck in order, the k-th card dealt (k = 0, 1, ..., 51) is at index
rand mod (52 − k); the last card of the deck takes its place, and the deck shrinks by one.
Card k goes to column k mod 8, so the first eight cards are the bottoms of the columns.
The algorithm is the one Jim Horne disclosed, as written down by Rosetta Code ("Deal cards
for FreeCell") and in fc-solve's `gen_ms_boards__rand.h`.
"""

from collections.abc import Iterator

RANKS = "A23456789TJQK"
SUITS = "CDHS"

type Card = int
type Column = tuple[Card, ...]
type Layout = tuple[Column, ...]


def card(rank: int, suit: int) -> Card:
    """The card of this rank (1 to 13) and suit (0 to 3)."""
    return 4 * (rank - 1) + suit


def rank(c: Card) -> int:
    """Rank of a card, 1 (ace) to 13 (king)."""
    return c // 4 + 1


def suit(c: Card) -> int:
    """Suit of a card: 0 clubs, 1 diamonds, 2 hearts, 3 spades."""
    return c % 4


def is_red(c: Card) -> bool:
    """Diamonds and hearts are red."""
    return c % 4 in (1, 2)


def name(c: Card) -> str:
    """Two-letter name, such as 'AS' or 'TD'."""
    return RANKS[rank(c) - 1] + SUITS[suit(c)]


def parse_card(text: str) -> Card:
    """Inverse of `name`; accepts '10' for ten."""
    text = text.strip().upper().replace("10", "T")
    return card(RANKS.index(text[0]) + 1, SUITS.index(text[1]))


def ms_rand(seed: int) -> Iterator[int]:
    """The Microsoft C runtime's rand(), seeded with `seed`."""
    state = seed
    while True:
        state = (214013 * state + 2531011) % 2**31
        yield state >> 16


def ms_order(deal: int) -> list[Card]:
    """The 52 cards of Microsoft deal `deal` (1 to 2³¹ − 1) in the order they are dealt."""
    if not 1 <= deal < 2**31:
        raise ValueError("Microsoft deals are numbered 1 to 2**31 - 1")
    deck = list(range(52))
    rand = ms_rand(deal)
    out = []
    while deck:
        j = next(rand) % len(deck)
        out.append(deck[j])
        deck[j] = deck[-1]
        deck.pop()
    return out


def deal_columns(order: list[Card], columns: int = 8) -> Layout:
    """Deal cards in rows across `columns` columns; the first card dealt is a bottom card."""
    return tuple(tuple(order[i::columns]) for i in range(columns))


def ms_deal(deal: int) -> Layout:
    """The layout of Microsoft deal `deal`: eight columns, four of 7 cards and four of 6."""
    return deal_columns(ms_order(deal))


def show(layout: Layout) -> str:
    """Rows of the layout as text, the bottom cards first, the way deals are published."""
    height = max(map(len, layout))
    rows = []
    for i in range(height):
        rows.append(" ".join(name(col[i]) if i < len(col) else "  " for col in layout).rstrip())
    return "\n".join(rows)


def parse_layout(text: str) -> Layout:
    """Inverse of `show` for full rows: rows of card names, bottom cards first."""
    rows = [line.split() for line in text.strip().splitlines()]
    width = len(rows[0])
    return tuple(tuple(parse_card(row[i]) for row in rows if i < len(row)) for i in range(width))


def small_deck(ranks: int, suits: tuple[int, ...] = (0, 1, 2, 3)) -> list[Card]:
    """A smaller deck: ranks 1 to `ranks` of the given suits, in card order."""
    return sorted(card(r, s) for r in range(1, ranks + 1) for s in suits)


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:] or ["1", "11982"]:
        print(f"Deal {arg}\n{show(ms_deal(int(arg)))}\n")
