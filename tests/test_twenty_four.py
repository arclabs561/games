import ast
import operator
from fractions import Fraction

from twenty_four import HANDS, brute, deck_probability, solvable, solve

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: Fraction}


def _value(expr):
    """Exact value and sorted cards of an expression."""
    leaves = []

    def ev(n):
        if isinstance(n, ast.BinOp):
            return OPS[type(n.op)](ev(n.left), ev(n.right))
        assert isinstance(n, ast.Constant) and 1 <= n.value <= 13
        leaves.append(n.value)
        return Fraction(n.value)

    return ev(ast.parse(expr, mode="eval").body), sorted(leaves)


def _steps_are_integers(expr):
    def ev(n):
        if isinstance(n, ast.BinOp):
            v = OPS[type(n.op)](ev(n.left), ev(n.right))
            assert v.denominator == 1, expr
            return v
        return Fraction(n.value)

    ev(ast.parse(expr, mode="eval").body)


def test_hand_count():
    assert len(HANDS) == 1820  # C(13 + 4 - 1, 4)


def test_expressions_are_correct():
    for hand in [(1, 2, 3, 4), (3, 3, 8, 8), (1, 5, 5, 5), (6, 6, 6, 6), (13, 13, 12, 1)]:
        for rule in ("fraction", "integer", "nodiv"):
            e = solve(hand, rule)
            if e:
                assert _value(e) == (24, sorted(hand))
                if rule != "fraction":
                    _steps_are_integers(e)
                if rule == "nodiv":
                    assert "/" not in e


def test_famous_hands():
    assert solve((3, 3, 8, 8)) and not solve((3, 3, 8, 8), "integer")
    assert solve((1, 5, 5, 5)) and not solve((1, 5, 5, 5), "integer")
    assert solve((1, 2, 7, 7), "integer") and not solve((1, 2, 7, 7), "nodiv")  # (7*7-1)/2
    assert not solve((1, 1, 1, 1)) and not solve((13, 13, 13, 13))


def test_published_counts():
    # 1,362 of 1,820 (4nums.com, gottfriedville.net); 404 of 495 for 1 to 9 and 566 of 715
    # for 1 to 10 (solve24game.com); 16 hands need a fraction (puzzletip.com); 0.8046 from
    # a 52-card deck (4nums.com).
    good = solvable("fraction")
    assert len(good) == 1362
    assert sum(max(h) <= 9 for h in good) == 404
    assert sum(max(h) <= 10 for h in good) == 566
    assert round(float(deck_probability(good)), 4) == 0.8046
    integer = solvable("integer")
    assert set(integer) <= set(good) and len(good) - len(integer) == 16
    nodiv = solvable("nodiv")
    assert set(nodiv) <= set(integer) and len(nodiv) == 1163


def test_brute_force_agrees_on_a_sample():
    for hand in HANDS[::15]:
        for rule in ("fraction", "integer", "nodiv"):
            found = brute(hand, rule)
            assert bool(found) == bool(solve(hand, rule)), (hand, rule)
            for e in found[:3]:
                assert _value(e) == (24, sorted(hand))


def test_brute_force_on_the_hard_hands():
    assert brute((3, 3, 8, 8)) == ["8/(3-(8/3))"]
    assert brute((3, 3, 8, 8), "integer") == []
