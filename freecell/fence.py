"""Check a certificate that a deal is lost, without the solvers.

A **fence** is a set of positions such that the deal's first position is in it, every legal
move from a position in it leads to a position in it, and no position in it is won. Every
position reachable from the deal is then in the fence (by induction on the number of moves),
so none is won and the deal is lost. Sells (2026) checks such a set for deal 11982 in Lean.

Here a position is written as text: its columns from bottom to top as card names, sorted and
joined by '|', then '/' and the free cells, sorted. With `autoplay=True` every position is
taken after safe cards have been played home (the rule of `freecell.rules`, written again
here), which is how Solver A stores positions; with `autoplay=False` it is taken as is, as
Solver B stores them. The checker has its own move generator and shares no code with the
solvers; it imports only the deal and the card names.

    uv run python -m freecell.fence 11982     # writes both solvers' fences and checks them
"""

import sys

from freecell.deals import Layout, ms_deal, name, parse_card


def text(cols: list[list[int]], cells: list[int]) -> str:
    """The text of a position: sorted columns, then the sorted free cells."""
    col_texts = sorted("".join(name(c) for c in col) for col in cols)
    return "|".join(col_texts) + "/" + "".join(sorted(name(c) for c in cells))


def parse(s: str) -> tuple[list[list[int]], list[int]]:
    """Inverse of `text`."""
    cols_part, cells_part = s.split("/")
    cols = [[parse_card(t[i : i + 2]) for i in range(0, len(t), 2)] for t in cols_part.split("|")]
    cells = [parse_card(cells_part[i : i + 2]) for i in range(0, len(cells_part), 2)]
    return cols, cells


def _red(c: int) -> bool:
    return c % 4 in (1, 2)


def _rank(c: int) -> int:
    return c // 4 + 1


def check(layout: Layout, cells: int, fence: set[str], autoplay: bool) -> tuple[bool, str]:
    """Is `fence` a fence for the deal with `cells` free cells? Returns (ok, reason)."""
    deck = sorted(c for col in layout for c in col)
    suits = sorted({c % 4 for c in deck})
    ranks = max(_rank(c) for c in deck)
    ncols = len(layout)

    def found_of(cols, cell):
        out = {}
        for s in suits:
            out[s] = ranks - sum(
                1 for c in [*cell, *(x for col in cols for x in col)] if c % 4 == s
            )
        return out

    def settle(cols, cell):
        """Play safe cards home while there are any (only when `autoplay`)."""
        if not autoplay:
            return cols, cell
        changed = True
        while changed:
            changed = False
            found = found_of(cols, cell)
            for c in [col[-1] for col in cols if col] + list(cell):
                if found[c % 4] != _rank(c) - 1:
                    continue
                # cards that could sit on c: one rank lower, the other color
                hosts = [d for d in deck if _rank(d) == _rank(c) - 1 and _red(d) != _red(c)]
                if all(_rank(d) == 1 or found[d % 4] >= _rank(d) for d in hosts):
                    cols = [col[:-1] if col and col[-1] == c else col for col in cols]
                    cell = [x for x in cell if x != c]
                    changed = True
                    break
        return cols, cell

    def successors(cols, cell):
        found = found_of(cols, cell)
        out = []
        for i, col in enumerate(cols):
            if not col:
                continue
            c = col[-1]
            rest = cols[:i] + [col[:-1]] + cols[i + 1 :]
            out.extend(_places(c, rest, cell, found, from_col=i))
            if len(cell) < cells:
                out.append((rest, cell + [c]))
        for k, c in enumerate(cell):
            out.extend(_places(c, cols, cell[:k] + cell[k + 1 :], found, from_col=None))
        return out

    def _places(c, cols, cell, found, from_col):
        out = []
        if found[c % 4] == _rank(c) - 1:
            out.append((cols, cell))  # the card goes home
        for j, col in enumerate(cols):
            if j == from_col:
                continue
            if not col or (_rank(col[-1]) == _rank(c) + 1 and _red(col[-1]) != _red(c)):
                out.append((cols[:j] + [col + [c]] + cols[j + 1 :], cell))
        return out

    normal = {text(*parse(s)) for s in fence}
    if len(normal) != len(fence):
        return False, "positions not in sorted form"
    first = settle([list(col) for col in layout], [])
    if text(*first) not in normal:
        return False, "the deal is not in the fence"
    for s in normal:
        cols, cell = parse(s)
        if len(cols) != ncols:
            return False, f"wrong number of columns: {s}"
        if not cell and not any(cols):
            return False, f"won position in the fence: {s}"
        for nxt in successors(cols, cell):
            t = text(*settle(*nxt))
            if t not in normal:
                return False, f"a move leaves the fence: {s} -> {t}"
    return True, f"fence of {len(normal)} positions holds"


def fence_a(deal: int, cells: int) -> set[str]:
    """Solver A's fence for a lost Microsoft deal, as text."""
    from freecell.rules import Rules
    from freecell.solver import solve

    res = solve(Rules(cells=cells), ms_deal(deal), use_supermoves=False, keep_fence=True)
    if res.verdict != "lost" or res.fence is None:
        raise ValueError(f"deal {deal} is not lost with {cells} cells")
    out = set()
    for k in res.fence:
        cols_part, cell = k.split(b"\xfe")
        out.add(text([list(col) for col in cols_part.split(b"\xff")], list(cell)))
    return out


def fence_b(deal: int, cells: int) -> set[str]:
    """Solver B's fence for a lost Microsoft deal, as text, from its arrays of what lies
    beneath each card."""
    from freecell import native

    (ans,) = native.run([native.ms_problem(deal, cells)], fence=True)
    if ans.verdict != "lost" or ans.fence is None:
        raise ValueError(f"deal {deal} is not lost with {cells} cells")
    return {_columns_from_array(row, 8) for row in ans.fence}


def _columns_from_array(under: tuple[int, ...], ncols: int) -> str:
    base, cell_code, home, absent = 52, 53, 54, 55
    above = {u: c for c, u in enumerate(under) if u < 52}
    cols = []
    for c, u in enumerate(under):
        if u == base:
            col = [c]
            while col[-1] in above:
                col.append(above[col[-1]])
            cols.append(col)
    cols += [[] for _ in range(ncols - len(cols))]
    cells = [c for c, u in enumerate(under) if u == cell_code]
    assert all(u in (base, cell_code, home, absent) or u < 52 for u in under)
    return text(cols, cells)


def main() -> None:
    deal = int(sys.argv[1]) if len(sys.argv) > 1 else 11982
    cells = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    for label, fence, auto in (
        ("Solver A (after autoplay)", fence_a(deal, cells), True),
        ("Solver B (no autoplay)", fence_b(deal, cells), False),
    ):
        ok, reason = check(ms_deal(deal), cells, fence, autoplay=auto)
        print(f"deal {deal}, {cells} cells, {label}: {'OK' if ok else 'FAILED'}, {reason}")


if __name__ == "__main__":
    main()
