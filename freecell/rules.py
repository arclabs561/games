"""FreeCell rules as small functions, on a deck of any size.

The rules as modeled:

- The tableau has a fixed number of **columns** (eight in the standard game); the deal fills
  them. A card on top of a column may move.
- Each **free cell** holds one card. The standard game has four.
- Each suit has a **foundation**, built up from the ace. A card goes there when the card one
  rank below it of its suit is there (an ace when the foundation is empty). Cards never
  leave a foundation.
- A card may move onto the top card of a column if that card is one rank higher and of the
  other color, onto an empty column, into an empty free cell, or onto its foundation.
- A **move** moves one card. Moves of several cards at once ("supermoves") are shorthand for
  single-card moves through free cells and empty columns, so they reach no new positions and
  change no verdict.
- The game is **won** when every card is on its foundation.

A **position** here is a triple: the columns (a tuple of `bytes`, bottom card first), the
free cells (sorted `bytes`) and the foundations (a 4-tuple, the top rank of each suit, 0 if
none). Columns are interchangeable, and so are free cells, so `key` sorts both: two
positions with the same key are the same up to the order of columns and cells, and one is won
exactly when the other is.

A move is written by cards, not by places, so that it does not depend on the order of the
columns: `(card, "home")`, `(card, "cell")`, `(card, "empty")` or `(card, target)` where
`target` is the card it goes onto.

**Safe autoplay.** A card of rank r is **safe** when it can go to its foundation and every
card that could ever be placed on it (rank r − 1, the other color) is already on a
foundation or is an ace. Moving a safe card home never turns a won position into a lost one.
Take any winning line from the position, delete the moves of that card, and replace each
move of an ace onto it by a move of that ace home (always legal, since an ace not yet home
has an empty foundation); then delete the later moves of that ace. The rest stays legal:
nothing else is ever placed on the card or on the ace, and their old places only become free.
So a solver may play safe cards at once; `autoplay` does. Solver B and the brute force in
`brute.py` do not, and the tests check that all give the same verdicts. The rule plays aces
and twos at once, as Microsoft FreeCell does, and other cards later than it does.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field

from freecell.deals import Card, Layout, is_red, name, rank, suit

type Cols = tuple[bytes, ...]
type Position = tuple[Cols, bytes, tuple[int, int, int, int]]
type Move = tuple[Card, int | str]


@dataclass(frozen=True)
class Rules:
    """Free cells, columns, and the deck: ranks 1 to `ranks` of the suits in `suits`."""

    cells: int = 4
    columns: int = 8
    ranks: int = 13
    suits: tuple[int, ...] = (0, 1, 2, 3)
    # For each card, the cards of the deck that may be placed on it (rank one lower, other color).
    _hosted: dict[Card, tuple[Card, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        hosted = {}
        for c in self.deck():
            hosted[c] = tuple(d for d in self.deck() if can_stack(d, c))
        object.__setattr__(self, "_hosted", hosted)

    def deck(self) -> list[Card]:
        """Every card of the deck, in card order."""
        return sorted(4 * (r - 1) + s for r in range(1, self.ranks + 1) for s in self.suits)

    def size(self) -> int:
        """Number of cards in the deck."""
        return self.ranks * len(self.suits)


def can_stack(c: Card, onto: Card) -> bool:
    """Can card `c` sit on card `onto` in a column (one rank lower, other color)?"""
    return rank(onto) == rank(c) + 1 and is_red(onto) != is_red(c)


STANDARD = Rules()


def start(rules: Rules, layout: Layout) -> Position:
    """The position before any move, with empty free cells and foundations."""
    if len(layout) != rules.columns:
        raise ValueError(f"expected {rules.columns} columns")
    if sorted(c for col in layout for c in col) != rules.deck():
        raise ValueError("the layout does not hold the deck")
    return tuple(bytes(col) for col in layout), b"", (0, 0, 0, 0)


def key(pos: Position) -> bytes:
    """The position with columns and cells sorted; the foundations follow from the cards."""
    cols, cells, _ = pos
    return b"\xff".join(sorted(cols)) + b"\xfe" + cells


def unkey(rules: Rules, k: bytes) -> Position:
    """A position with this key. The foundations follow: a suit's cards still out are the
    ranks above its foundation."""
    cols_part, cells = k.split(b"\xfe")
    cols = tuple(cols_part.split(b"\xff"))
    out = [0, 0, 0, 0]
    for s in rules.suits:
        left = sum(1 for col in cols for c in col if c % 4 == s) + sum(
            1 for c in cells if c % 4 == s
        )
        out[s] = rules.ranks - left
    return cols, cells, (out[0], out[1], out[2], out[3])


def home_count(pos: Position) -> int:
    """Number of cards on the foundations."""
    return sum(pos[2])


def won(rules: Rules, pos: Position) -> bool:
    """Is every card on its foundation?"""
    return home_count(pos) == rules.size()


def goes_home(found: tuple[int, ...], c: Card) -> bool:
    """Can card `c` go to its foundation now?"""
    return found[suit(c)] == rank(c) - 1


def safe(rules: Rules, found: tuple[int, ...], c: Card) -> bool:
    """Can `c` go home now, with every card that could sit on it home or an ace?"""
    return goes_home(found, c) and all(
        rank(d) == 1 or found[suit(d)] >= rank(d) for d in rules._hosted[c]
    )


def _home(found: tuple[int, int, int, int], c: Card) -> tuple[int, int, int, int]:
    f = list(found)
    f[suit(c)] += 1
    return f[0], f[1], f[2], f[3]


def autoplay(rules: Rules, pos: Position) -> tuple[Position, list[Move]]:
    """Play safe cards home until none is left; return the position and the moves made."""
    cols, cells, found = pos
    made: list[Move] = []
    changed = True
    while changed:
        changed = False
        for i, col in enumerate(cols):
            if col and safe(rules, found, col[-1]):
                c = col[-1]
                cols = cols[:i] + (col[:-1],) + cols[i + 1 :]
                found = _home(found, c)
                made.append((c, "home"))
                changed = True
        for c in cells:
            if safe(rules, found, c):
                cells = cells.replace(bytes([c]), b"")
                found = _home(found, c)
                made.append((c, "home"))
                changed = True
    return (cols, cells, found), made


def moves(rules: Rules, pos: Position) -> Iterator[tuple[Move, Position]]:
    """Every legal single-card move and the position after it (no autoplay).

    Moves that give the same key are listed once: a card goes to only the first empty
    column, a column's only card does not move to an empty column, and a free cell card
    does not move to another free cell.
    """
    cols, cells, found = pos
    empty = next((i for i, col in enumerate(cols) if not col), None)
    tops = [(i, col[-1]) for i, col in enumerate(cols) if col]
    sources: list[tuple[int | None, Card]] = [(i, c) for i, c in tops]
    sources += [(None, c) for c in cells]
    for src, c in sources:
        if src is None:
            rest_cols, rest_cells = cols, cells.replace(bytes([c]), b"")
        else:
            rest_cols = cols[:src] + (cols[src][:-1],) + cols[src + 1 :]
            rest_cells = cells
        if goes_home(found, c):
            yield (c, "home"), (rest_cols, rest_cells, _home(found, c))
        for j, t in tops:
            if j != src and can_stack(c, t):
                new = rest_cols[:j] + (rest_cols[j] + bytes([c]),) + rest_cols[j + 1 :]
                yield (c, t), (new, rest_cells, found)
        if empty is not None and not (src is not None and len(cols[src]) == 1):
            new = rest_cols[:empty] + (bytes([c]),) + rest_cols[empty + 1 :]
            yield (c, "empty"), (new, rest_cells, found)
        if src is not None and len(cells) < rules.cells:
            yield (c, "cell"), (rest_cols, bytes(sorted(cells + bytes([c]))), found)


def apply(rules: Rules, pos: Position, move: Move) -> Position:
    """The position after `move`; raises ValueError if the move is not legal."""
    c, dest = move
    cols, cells, found = pos
    src = next((i for i, col in enumerate(cols) if col and col[-1] == c), None)
    if src is None and c not in cells:
        raise ValueError(f"{name(c)} cannot move")
    if src is None:
        cols2, cells2 = cols, cells.replace(bytes([c]), b"")
    else:
        cols2, cells2 = cols[:src] + (cols[src][:-1],) + cols[src + 1 :], cells
    if dest == "home":
        if not goes_home(found, c):
            raise ValueError(f"{name(c)} cannot go home")
        return cols2, cells2, _home(found, c)
    if dest == "cell":
        if src is None or len(cells) >= rules.cells:
            raise ValueError("no free cell")
        return cols2, bytes(sorted(cells2 + bytes([c]))), found
    if dest == "empty":
        j = next((i for i, col in enumerate(cols2) if not col and i != src), None)
        if j is None:
            raise ValueError("no empty column")
    else:
        assert isinstance(dest, int)
        j = next((i for i, col in enumerate(cols2) if col and col[-1] == dest), None)
        if j is None or not can_stack(c, dest):
            raise ValueError(f"{name(c)} cannot go onto {name(dest)}")
    new = cols2[:j] + (cols2[j] + bytes([c]),) + cols2[j + 1 :]
    return new, cells2, found


def replay(rules: Rules, layout: Layout, line: list[Move]) -> bool:
    """Do these moves, each legal in turn, win the deal? (No autoplay: every move is listed.)"""
    pos = start(rules, layout)
    try:
        for move in line:
            pos = apply(rules, pos, move)
    except ValueError:
        return False
    return won(rules, pos)


def move_name(move: Move) -> str:
    """A move as text, such as '5H>6S', 'QC>cell', 'AS>home', 'KD>empty'."""
    c, dest = move
    return f"{name(c)}>{dest if isinstance(dest, str) else name(dest)}"
