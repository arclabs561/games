#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "z3-solver>=5.4",
# ]
# ///
"""Check the induction steps of the hand proofs with z3: uv run erase/smt/z3_lemmas.py.

Each step is stated for all integers and z3 is asked for a counterexample; "proved" means
none exists. A congruence a = b (mod m) is written as a - b = m*k with an explicit
witness k, so every goal is free of quantifiers. Lemma numbers follow the paper.
"""

from z3 import And, If, Implies, Int, Ints, Not, Or, Solver, unsat


def prove(name, claim):
    s = Solver()
    s.add(Not(claim))
    r = s.check()
    print(f"{'proved' if r == unsat else 'FAILED'}  {name}")
    assert r == unsat


def sign(e):
    return Or(e == -1, e == 0, e == 1)


def absz(t):
    return If(t >= 0, t, -t)


v1, v2, x, y, m, k1, k2, e1, e2, a, b = Ints("v1 v2 x y m k1 k2 e1 e2 a b")

# Lemma 5 (one loud card), node step: v1 = e1*x (mod m), v2 = 0 (mod m).
hyp = And(sign(e1), v1 == e1 * x + m * k1, v2 == m * k2)
prove("one loud: + keeps e1*x", Implies(hyp, v1 + v2 == e1 * x + m * (k1 + k2)))
prove("one loud: - keeps e1*x", Implies(hyp, v1 - v2 == e1 * x + m * (k1 - k2)))
prove("one loud: reversed - flips the sign", Implies(hyp, v2 - v1 == (-e1) * x + m * (k2 - k1)))
prove("one loud: x by a silent value erases it", Implies(hyp, v1 * v2 == 0 * x + m * (v1 * k2)))


# Lemma 6 (two loud cards), split step: v1 = e1*x, v2 = e2*y (mod m).
# Results land in S = {0, +-x, +-y, +-x+-y, +-xy}.
def in_s(t, kk):
    forms = [0, x, -x, y, -y, x + y, x - y, -x + y, -x - y, x * y, -(x * y)]
    return Or([t == f + m * kk for f in forms])


hyp2 = And(sign(e1), sign(e2), v1 == e1 * x + m * k1, v2 == e2 * y + m * k2)
prove("two loud: sum in S", Implies(hyp2, in_s(v1 + v2, k1 + k2)))
prove("two loud: difference in S", Implies(hyp2, in_s(v1 - v2, k1 - k2)))
prove(
    "two loud: product in S",
    Implies(hyp2, in_s(v1 * v2, e1 * x * k2 + e2 * y * k1 + m * k1 * k2)),
)
s_, kk = Int("s_"), Int("kk")
prove("two loud: S = -S", Implies(in_s(s_, kk), in_s(-s_, -kk)))

# Lemma 11 (negatives flip), node step: the three identities the step uses.
prove("flip: -(a-b) = b-a", -(a - b) == b - a)
prove("flip: a<0 => -(a+b) = (-a) - b", Implies(a < 0, -(a + b) == (-a) - b))
prove(
    "flip: a*b<0 => exactly one factor negative",
    Implies(a * b < 0, Or(And(a < 0, b > 0), And(a > 0, b < 0))),
)

# Lemma 10 (wastebasket), node step: combining s*u and t*w, with signs s and t,
# gives +-(u+-w) or +-(u*w).
s1, t1, u, w = Ints("s1 t1 u w")
hyp3 = And(Or(s1 == 1, s1 == -1), Or(t1 == 1, t1 == -1))


def pm(val, cands):
    return Or([Or(val == c, val == -c) for c in cands])


prove("wastebasket: sum", Implies(hyp3, pm(s1 * u + t1 * w, [u + w, u - w])))
prove("wastebasket: difference", Implies(hyp3, pm(s1 * u - t1 * w, [u + w, u - w])))
prove("wastebasket: product", Implies(hyp3, pm((s1 * u) * (t1 * w), [u * w])))

# Lemma 13 (size).
prove("size: |a+b| <= |a|+|b|", absz(a + b) <= absz(a) + absz(b))
prove("size: |a-b| <= |a|+|b|", absz(a - b) <= absz(a) + absz(b))
prove("size: |ab| = |a||b|", absz(a * b) == absz(a) * absz(b))


# Theorem 8 (five moduli), the modulus-8 step, by enumeration over every loud hand
# card x and center card c in 0..10: if the mod-8 test passes, so does the mod-4 test.
def reach(xx, cc, mm):
    forms = [0, xx, -xx, cc, -cc, xx + cc, xx - cc, -xx + cc, -xx - cc, xx * cc, -xx * cc]
    return {t % mm for t in forms}


bad = [
    (xx, cc)
    for xx in range(11)
    for cc in range(11)
    if 10 % 8 not in reach(xx if xx % 8 else 0, cc, 8)
    and 10 % 4 in reach(xx if xx % 4 else 0, cc, 4)
]
print(("proved" if not bad else "FAILED") + "  five moduli: mod-8 test implies mod-4 test")
assert not bad
