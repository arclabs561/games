"""Check "cannot make 10" claims with z3 (pip install z3-solver); about two minutes.

A straight-line program of n-1 steps, each combining two unused values with +, - or x.
Bit-vectors of 26 bits are exact here: with six cards of value at most 10, no value
exceeds 11^6 < 2^25 in absolute value (Lemma 13 of the paper).
"""

import time

import z3

W = 26

# The ten six-card multisets from the paper's appendix that have no part making 10.
SIX = [
    (1, 1, 7, 7, 7, 7),
    (1, 6, 7, 7, 7, 7),
    (1, 7, 7, 7, 7, 8),
    (2, 2, 9, 9, 9, 9),
    (2, 7, 9, 9, 9, 9),
    (6, 6, 6, 7, 7, 7),
    (6, 6, 7, 7, 7, 7),
    (6, 7, 7, 7, 7, 8),
    (7, 7, 7, 7, 8, 8),
    (7, 7, 9, 9, 9, 9),
]


def select(idx, vals):
    e = vals[-1]
    for i in range(len(vals) - 2, -1, -1):
        e = z3.If(idx == i, vals[i], e)
    return e


def can_make(cards, target=10):
    """sat if some intermediate value of some program equals target, i.e. some
    sub-collection of the cards makes it; unsat if none does."""
    n = len(cards)
    s = z3.Solver()
    pool = [z3.BitVecVal(c, W) for c in cards]
    uses = {i: [] for i in range(2 * n - 1)}
    produced = []
    for st in range(n - 1):
        size = n + st
        a, b, op = z3.Int(f"a{st}"), z3.Int(f"b{st}"), z3.Int(f"op{st}")
        s.add(0 <= a, a < b, b < size, 0 <= op, op <= 3)
        x, y = select(a, pool[:size]), select(b, pool[:size])
        v = z3.BitVec(f"v{st}", W)
        s.add(v == z3.If(op == 0, x + y, z3.If(op == 1, x - y, z3.If(op == 2, y - x, x * y))))
        pool.append(v)
        produced.append(v)
        for i in range(size):
            uses[i] += [z3.If(a == i, 1, 0), z3.If(b == i, 1, 0)]
    for i in range(2 * n - 2):
        s.add(z3.Sum(uses[i]) <= 1)
    s.add(z3.Or([p == target for p in produced]))
    return s.check()


if __name__ == "__main__":
    t = time.time()
    for m in SIX:
        r = can_make(m)
        print(m, "some part makes 10:", r, f"{time.time() - t:.0f}s", flush=True)
        assert r == z3.unsat
