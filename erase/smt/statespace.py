"""A third engine: search states of "combine two values" moves, as 24-game solvers do.

Confirms that all seven cards of {1, 6, 6, 7, 7, 7, 7} cannot make 10 together, the one
exception in the big-hands theorem. Standard library only; a few seconds.
"""

from functools import cache
from itertools import combinations


@cache
def finals(state):
    """Every value reachable by repeatedly replacing two values with a+b, a-b, b-a or a*b."""
    if len(state) == 1:
        return frozenset(state)
    out = set()
    for i, j in combinations(range(len(state)), 2):
        a, b = state[i], state[j]
        rest = state[:i] + state[i + 1 : j] + state[j + 1 :]
        for v in {a + b, a - b, b - a, a * b}:
            out |= finals(tuple(sorted(rest + (v,))))
    return frozenset(out)


if __name__ == "__main__":
    ok = 10 in finals((1, 6, 6, 7, 7, 7, 7))
    print("all seven cards of 1,6,6,7,7,7,7 make 10:", ok)
    if ok:
        raise ValueError("the exceptional hand makes ten using every card")
