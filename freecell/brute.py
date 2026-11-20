"""Brute force: every position reachable by single-card moves, with nothing pruned.

A **literal position** keeps the columns in their places: moving a card to the third column
and moving it to the fifth are different positions, even when both columns are empty. Free
cells are a set (a card in the first cell and the same card in the third are one position),
as in Sells (2026). No card goes home unless a move puts it there.

`reachable` lists every literal position reachable from the deal, breadth first. It is slow
and simple, and shares no code with the solvers beyond the deal and the card helpers. The
tests compare its verdicts with Solvers A and B on every deal of several small decks.
"""

from collections import deque

from freecell.deals import Card, Layout, is_red, rank, suit


def _home(found: tuple[int, ...], c: Card) -> tuple[int, ...]:
    f = list(found)
    f[suit(c)] += 1
    return tuple(f)


def successors(state, cells: int):
    """Literal positions one move away. A state is (columns, sorted cells, foundations)."""
    cols, cell, found = state
    out = []
    sources = [("col", i, col[-1]) for i, col in enumerate(cols) if col]
    sources += [("cell", i, c) for i, c in enumerate(cell)]
    for kind, i, c in sources:
        if kind == "col":
            rest = cols[:i] + (cols[i][:-1],) + cols[i + 1 :]
            rest_cell = cell
        else:
            rest = cols
            rest_cell = cell[:i] + cell[i + 1 :]
        if found[suit(c)] == rank(c) - 1:
            out.append((rest, rest_cell, _home(found, c)))
        for j, col in enumerate(rest):
            if kind == "col" and j == i:
                continue
            if not col or (rank(col[-1]) == rank(c) + 1 and is_red(col[-1]) != is_red(c)):
                out.append((rest[:j] + (col + (c,),) + rest[j + 1 :], rest_cell, found))
        if kind == "col" and len(cell) < cells:
            out.append((rest, tuple(sorted(cell + (c,))), found))
    return out


def reachable(layout: Layout, cells: int, limit: int | None = None) -> tuple[int, bool, int]:
    """(literal positions reached, whether one is won, most cards home in any of them)."""
    start = (tuple(tuple(col) for col in layout), (), (0, 0, 0, 0))
    total = sum(map(len, layout))
    seen = {start}
    queue = deque([start])
    most = 0
    while queue:
        state = queue.popleft()
        home = sum(state[2])
        most = max(most, home)
        if home == total:
            return len(seen), True, most
        for nxt in successors(state, cells):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
        if limit is not None and len(seen) > limit:
            raise RuntimeError("limit reached")
    return len(seen), False, most


def brute_winnable(layout: Layout, cells: int) -> bool:
    """Is some literal position reachable from the deal won?"""
    return reachable(layout, cells)[1]
