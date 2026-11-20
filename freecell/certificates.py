"""Short reasons that a deal is lost, and how many lost deals each one explains.

Three kinds, from the cheapest to check to the most general:

- **Blocked at the start.** No card can move at all. With a free cell this never happens
  (any top card fits in it), so it is a certificate for 0 free cells only.

- **Stuck cards** (0 free cells). Call a set M of cards **closed** if every card that ever
  moves lies in it. Build the smallest candidate from below: start with no cards, and add a
  card X once (1) every card dealt above X is in M, and (2) X has somewhere to go: X is an
  ace, or the card one rank below it in its suit is in M (it must have gone home first), or
  some card Y of the other color and one rank higher is in M or has every card dealt above
  it in M (so Y can be on top of a column), or some other column has all its cards in M (so
  it can be empty). Repeat until nothing changes. Every card that ever moves gets added, by
  induction on the time of its first move: just before it, the cards dealt above it, its
  destination's cards and the emptied column's cards all moved earlier. So a card left out
  never moves, never goes home, and the deal is lost. The check is a short fixpoint
  computation on the deal alone.

- **A fence** (any number of free cells): the set of all reachable positions, checked by
  `freecell.fence`. Its size is the length of the certificate. Within a fence, the cards
  that no reachable position moves (Solver A's and Solver B's `frozen`) say where the deal
  is stuck; when that list is empty, every card moves in some line and the deal still never
  finishes, and the fence is the only reason given here.

`static_stuck` is sound for 0 free cells only: with a free cell, any card can always reach
it in the over-approximation, so the fixpoint would add every card.
"""

from freecell.deals import Card, Layout, is_red, rank, suit


def blocked_at_start(layout: Layout, cells: int) -> bool:
    """Is there no legal move in the first position?"""
    if cells > 0 and any(layout):
        return False
    tops = [col[-1] for col in layout if col]
    if any(not col for col in layout) and any(len(col) > 1 for col in layout):
        return False
    for c in tops:
        if rank(c) == 1:
            return False
        if any(rank(t) == rank(c) + 1 and is_red(t) != is_red(c) for t in tops):
            return False
    return True


def static_stuck(layout: Layout) -> frozenset[Card]:
    """Cards that can never move with 0 free cells, by the fixpoint above."""
    above = {}
    column_of = {}
    for i, col in enumerate(layout):
        for j, c in enumerate(col):
            above[c] = col[j + 1 :]
            column_of[c] = i
    deck = [c for col in layout for c in col]
    by_card = {(rank(c), suit(c)): c for c in deck}
    moved: set[Card] = set()

    def exposable(y: Card) -> bool:
        return y in moved or all(a in moved for a in above[y])

    changed = True
    while changed:
        changed = False
        for x in deck:
            if x in moved or not all(a in moved for a in above[x]):
                continue
            below = by_card.get((rank(x) - 1, suit(x)))
            home = rank(x) == 1 or (below is not None and below in moved)
            hosts = [
                y
                for y in deck
                if rank(y) == rank(x) + 1 and is_red(y) != is_red(x) and exposable(y)
            ]
            empty = any(
                i != column_of[x] and all(c in moved for c in col) for i, col in enumerate(layout)
            )
            if home or hosts or empty:
                moved.add(x)
                changed = True
    return frozenset(deck) - moved


def explain(deal: int, cells: int) -> str:
    """The cheapest reason found that Microsoft deal `deal` is lost with `cells` free cells."""
    from freecell.deals import ms_deal, name
    from freecell.rules import Rules
    from freecell.solver import solve

    layout = ms_deal(deal)
    if blocked_at_start(layout, cells):
        return f"deal {deal}, {cells} cells: lost, no card can move at the start"
    if cells == 0 and (stuck := static_stuck(layout)):
        aces = " ".join(name(c) for c in sorted(stuck) if c < 4)
        some = " ".join(name(c) for c in sorted(stuck)[:8]) + (" ..." if len(stuck) > 8 else "")
        return (
            f"deal {deal}, 0 cells: lost, {len(stuck)} cards can never move ({some}); "
            f"aces among them: {aces}"
        )
    res = solve(Rules(cells=cells), layout, use_supermoves=False)
    if res.verdict == "won":
        return f"deal {deal}, {cells} cells: won in {len(res.line)} moves"
    frozen = " ".join(name(c) for c in sorted(res.frozen)) or "none"
    return (
        f"deal {deal}, {cells} cells: lost; {res.states:,} positions reachable, at most "
        f"{res.most_home} cards home; cards that never move: {frozen}"
    )


if __name__ == "__main__":
    import sys

    args = [int(a) for a in sys.argv[1:]] or [11982, 4]
    print(explain(args[0], args[1] if len(args) > 1 else 4))
