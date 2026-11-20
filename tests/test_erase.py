import ast
import operator
from itertools import combinations, combinations_with_replacement, islice

from verify_certs import check

from erase import in_families, make, multisets, solve


def _evaluate(expr):
    """Value and card list of an expression; unlike verify_certs.value, allows 0."""
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}
    leaves = []

    def ev(n):
        if isinstance(n, ast.BinOp):
            return ops[type(n.op)](ev(n.left), ev(n.right))
        assert isinstance(n, ast.Constant) and 0 <= n.value <= 10
        leaves.append(n.value)
        return n.value

    return ev(ast.parse(expr, mode="eval").body), leaves


def _assert_clears(hand, center):
    erased = []
    for g in solve(hand, center):
        expr, _, note = g.partition("   [center ")
        v, leaves = _evaluate(expr)
        assert v == 10, g
        if note:
            c = int(note.rstrip("]"))
            assert c in center
            leaves.remove(c)
        erased += leaves
    assert sorted(erased) == sorted(hand)


def test_dead_deal():
    assert solve((0, 4, 4, 4, 8, 8, 8), (0, 3, 7, 9)) is None


def test_rules_sheet_hand_clears():
    _assert_clears((4, 4, 6, 9, 5, 8, 5), (7, 5, 1, 8))


def test_greedy_trap_deal_clears():
    # Pairing 3+7 and 4+6 first leaves 3, 8, 8, 8, which cannot be cleared.
    _assert_clears((3, 3, 4, 7, 8, 8, 8), (1, 2, 6, 6))


def test_three_family_is_dead():
    assert solve((3, 3, 6, 6, 9, 9, 9), (0, 3, 6, 9)) is None


def test_exception_needs_a_subset():
    m = (1, 6, 6, 7, 7, 7, 7)
    assert not in_families(m)
    assert make(m, 10) is None
    assert make((1, 6, 6, 7, 7, 7), 10) is not None


def test_zero_hand_is_one_group():
    # With a 0 in hand, the whole hand is one group whenever any part makes 10.
    assert make((0, 2, 3, 3, 3, 3, 9), 10) is not None


def test_search_certificates_pass_the_checker():
    # The first 300 seven-card cases of Theorem 15, certified by the search and
    # checked by the independent checker. make_certs.py does all 265,148.
    cases = (m for m in multisets(7, [0] + [4] * 10) if not in_families(m))
    for m in islice(cases, 300):
        check(m, {"expr": make(m, 10), "uses": "all"})


def _alternating_sum(cards):
    s = sorted(cards, reverse=True)
    return sum(s[0::2]) - sum(s[1::2])


def test_wild_card_value_is_in_range():
    # Proposition 22: 10 minus the alternating sum is always a legal wild value.
    for k in range(3, 8):
        for cards in combinations_with_replacement(range(11), k):
            assert 0 <= _alternating_sum(cards) <= max(cards)


def test_wild_card_rescues_the_dead_deal():
    hand, center = (0, 4, 4, 4, 8, 8, 8), (0, 3, 7, 9)
    wild = 10 - _alternating_sum(hand)  # a wild card in the center
    assert make(tuple(sorted(hand + (wild,))), 10) is not None
    _assert_clears(hand, center[:3] + (wild,))
    rest = hand[1:]  # a wild card in place of the hand's 0
    wild = 10 - _alternating_sum(rest)
    assert make(tuple(sorted(rest + (wild,))), 10) is not None


# Section 5 puzzles: each printed answer, as (hand cards, center card or None, expression).
PUZZLE_SETS = {
    (2, 3, 4): "2*3+4",
    (1, 2, 4): "(1+4)*2",
    (3, 6, 8): "3*6-8",
    (2, 4, 9): "(9-4)*2",
    (2, 2, 2, 2): "2*2*2+2",
    (1, 1, 1, 4): "(1+1)*(1+4)",
    (3, 3, 8, 9): "3*3+9-8",
    (4, 7, 9, 9): "4*7-9-9",
}
HOMEMADE = {(2, 5, 8, 9): "2*5*(9-8)", (3, 5, 5, 7, 8, 9, 9): "(3+7)+(5-5)*8*9*9"}
PUZZLE_DEALS = [
    (
        (1, 2, 3, 4, 6, 7, 8),
        (2, 4, 6, 8),
        [((7, 1, 3), None, "7*1+3"), ((2, 8), None, "2+8"), ((4, 6), None, "4+6")],
    ),
    (
        (1, 1, 1, 3, 3, 6, 6),
        (3, 5, 6, 8),
        [((1, 1), 5, "(1+1)*5"), ((1, 3), 3, "3*3+1"), ((3, 6, 6), 5, "6+6+3-5")],
    ),
    (
        (1, 1, 3, 3, 3, 3, 5),
        (4, 6, 8, 8),
        [((1, 1), 8, "1+1+8"), ((3, 3), 4, "3+3+4"), ((3, 3, 5), 4, "3*4+3-5")],
    ),
]


def _parts(cards, sizes):
    """Sorted sub-multisets of the given sizes."""
    return {tuple(sorted(p)) for k in sizes for p in combinations(cards, k)}


def test_puzzle_set_answers():
    for cards, expr in PUZZLE_SETS.items():
        v, leaves = _evaluate(expr)
        assert v == 10 and sorted(leaves) == sorted(cards), expr
        # no smaller part makes 10, so every answer needs two or more steps
        assert all(make(p, 10) is None for p in _parts(cards, range(2, len(cards)))), cards


def test_puzzle_deal_answers():
    for hand, center, groups in PUZZLE_DEALS:
        used = []
        for cards, ctr, expr in groups:
            v, leaves = _evaluate(expr)
            assert v == 10, expr
            assert sorted(leaves) == sorted(cards + ((ctr,) if ctr is not None else ())), expr
            assert ctr is None or ctr in center
            used += cards
        assert sorted(used) == sorted(hand)
        _assert_clears(hand, center)


def test_puzzle_deal_claims():
    # puzzles 4 and 5: no hand card makes 10 with just one other card
    for hand, center, _ in PUZZLE_DEALS[1:]:
        assert all(make(p, 10) is None for p in _parts(hand, [2]))
        assert all(make(tuple(sorted((a, c))), 10) is None for a in hand for c in center)
    # puzzles 3 and 5: an odd number of odd hand cards and no odd center card (Proposition 20)
    for hand, center, _ in (PUZZLE_DEALS[0], PUZZLE_DEALS[2]):
        assert sum(v % 2 for v in hand) % 2 == 1 and all(c % 2 == 0 for c in center)


def test_cards_one_to_ten_stuck_deals():
    # Proposition 18: with no 0s in the deck, only the two families block a deal.
    from math import comb

    from count import ways

    deck = [0] + [4] * 10
    dead = 0
    for hand in multisets(7, deck[:]):
        if not in_families(hand):
            continue
        left = [deck[v] - hand.count(v) for v in range(11)]
        for center in multisets(4, left[:]):
            if all(in_families(hand + (c,)) for c in center):
                assert solve(hand, center) is None
                dead += ways(hand, deck) * ways(center, left)
    assert dead == comb(12, 7) * comb(5, 4) + comb(8, 7) * comb(21, 4) == 51840
    assert solve((4, 4, 4, 4, 8, 8, 8), (3, 7, 8, 9)) is None


def test_homemade_puzzles_and_pigeonhole():
    for cards, expr in HOMEMADE.items():
        v, leaves = _evaluate(expr)
        assert v == 10 and sorted(leaves) == sorted(cards), expr
    # every seven-card hand holds two equal cards or two neighbors
    for hand in multisets(7, [4] * 11):
        s = set(hand)
        assert len(s) < 7 or any(v + 1 in s for v in s)


def test_addition_piles_agree_with_the_simulator():
    """count.clears_by_addition (minimal center sets) against simulate's group packing."""
    import random

    from count import clears_by_addition, piles
    from simulate import groups, most_erased

    rng = random.Random(7)
    deck = [v for v in range(11) for _ in range(4)]
    for _ in range(400):
        rng.shuffle(deck)
        hand, center = sorted(deck[:7]), sorted(deck[7:11])
        packed = most_erased(7, groups(hand, center, False)) == 7
        assert clears_by_addition(hand, center) == packed, (hand, center)
    assert piles((10,)) == (1,)  # a lone 10 needs a center 0
    assert not clears_by_addition((5, 5, 5), (1, 2, 3, 4))
    assert clears_by_addition((5, 5, 5), (5,))


def _pairs_clear(hand, center):
    """Brute force for the printed rules: erase a pair from the hand, or one card whose
    complement is in the center, until nothing is left."""
    if not hand:
        return True
    first, rest = hand[0], hand[1:]
    if 10 - first in center and _pairs_clear(rest, center):
        return True
    return any(
        rest[i] == 10 - first and _pairs_clear(rest[:i] + rest[i + 1 :], center)
        for i in range(len(rest))
    )


def test_pairs_needs_agree_with_brute_force():
    import random

    from count import pairs_needs

    rng = random.Random(8)
    deck = [v for v in range(11) for _ in range(4)]
    for _ in range(2000):
        rng.shuffle(deck)
        hand, center = tuple(sorted(deck[:7])), set(deck[7:11])
        assert (pairs_needs(hand) <= center) == _pairs_clear(hand, center), (hand, center)


def test_statespace_engine_confirms_the_exception():
    """The third engine of Theorem 15 (erase/smt/statespace.py), run here so CI covers it:
    all seven cards of 1 6 6 7 7 7 7 make every whole number from 0 to 20 except 10."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "erase" / "smt" / "statespace.py"
    spec = importlib.util.spec_from_file_location("erase_statespace", path)
    assert spec and spec.loader
    statespace = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(statespace)
    values = statespace.finals((1, 6, 6, 7, 7, 7, 7))
    assert [v for v in range(21) if v not in values] == [10]


def test_original_version_hands():
    """With no center cards the exception of Theorem 15 is partly stuck: one 7 is left."""
    assert solve((1, 6, 6, 7, 7, 7, 7), ()) is None
    assert solve((1, 6, 6, 7, 7, 7), ()) is not None
    assert solve((4, 4, 6, 9, 5, 8, 5), ()) is not None


def test_simulate_helpers():
    from simulate import groups, most_erased

    # puzzle 3: with + and -, three odd hand cards and no odd center card leave one card
    assert most_erased(7, groups([1, 2, 3, 4, 6, 7, 8], [2, 4, 6, 8], True)) == 6
    # addition only: three pairs, and the 5 pairs with the center 5
    assert most_erased(7, groups([1, 2, 3, 5, 7, 8, 9], [0, 0, 0, 5], False)) == 7
    # a 10 can only be erased with 0s under addition
    assert most_erased(1, groups([10], [1, 2, 3, 4], False)) == 0
    assert most_erased(1, groups([10], [0, 2, 3, 4], False)) == 1
