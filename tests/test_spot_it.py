"""Spot It: finite fields, projective planes, the missing cards, extensions and chances."""

import random
from fractions import Fraction
from itertools import combinations

from spot_it import chance, extend, field, missing, plane


def test_field_axioms():
    for q in (2, 3, 4, 5, 7, 8, 9):
        f = field.gf(q)
        assert field.axiom_failures(f.add, f.mul) == []
    for n in (4, 6, 8, 9):
        assert field.axiom_failures(*field.integers_mod(n)) == [
            "every nonzero element has an inverse"
        ]
    assert field.prime_power(6) is None and field.prime_power(8) == (2, 3)
    # In GF(4) = {0, 1, x, x + 1}, written 0, 1, 2, 3: x · x = x + 1 and x · (x + 1) = 1.
    f4 = field.gf(4)
    assert f4.mul[2][2] == 3 and f4.mul[2][3] == 1


def test_planes():
    for q in (2, 3, 4, 5, 7, 8, 9):
        assert plane.is_projective_plane(plane.projective_plane(q), q)
        assert plane.is_projective_plane(plane.field_slope_deck(q), q)
    for p in (2, 3, 5, 7):
        assert plane.is_projective_plane(plane.slope_deck(p), p)
    for n in (4, 6, 8, 9):
        assert not plane.has_spot_it_property(plane.slope_deck(n))
    # Fano plane: 7 cards of 3 symbols, every pair of cards sharing one symbol.
    fano = plane.projective_plane(2)
    assert len(fano) == 7 and plane.shared_counts(fano) == {1: 21}


def test_missing_cards_every_pair_small_orders():
    for n in (2, 3, 4):
        result = missing.check_order(n)
        assert (
            result["counting finds them"] == result["search finds exactly them"] == result["pairs"]
        )


def test_missing_cards_order_seven_sample():
    deck = plane.projective_plane(7)
    rng = random.Random(0)
    for _ in range(20):
        i, j = sorted(rng.sample(range(57), 2))
        rest = [c for k, c in enumerate(deck) if k not in (i, j)]
        removed = sorted([deck[i], deck[j]], key=sorted)
        assert missing.missing_by_counting(rest, 7) == removed
        assert missing.cards_meeting_once(rest, 57, 8) == removed
        assert sorted(extend.complete(rest, 7)[55:], key=sorted) == removed


def test_identities_and_forced_completion():
    for n in (3, 7):
        for fam in extend.random_subfamilies(n, trials=3, seed=1):
            assert extend.identities_hold(fam, n)
            d = n * n + n + 1 - len(fam)
            if d <= n + 2:  # n is odd here, so d = n + 2 also completes
                assert plane.is_projective_plane(extend.complete(fam, n), n)


def test_maximal_families_order_three():
    s = extend.summarize_maximal(3)
    assert s["maximal"] == 1440 and s["planes"] == 720 and s["other sizes"] == {5: 720}
    # Each of the others is the complete-graph pattern: ten symbols on two cards, three unused.
    assert dict(s["other profiles"]) == {((0, 3), (2, 10)): 720}
    assert extend.summarize_maximal(2)["other sizes"] == {}


def test_complete_graph_family():
    for n in range(2, 8):
        fam = extend.complete_graph_family(n)
        maximal = not missing.cards_meeting_once(fam, n * n + n + 1, n + 1, limit=1)
        assert maximal == (n % 2 == 1)


def test_order_two_decks_on_any_symbols():
    assert extend.five_card_decks_of_order_two() == {"7 symbols": 6300, "sunflower": 315}


def test_uniqueness_counts():
    for n, aut, completions in ((2, 168, 1), (3, 5616, 8)):
        assert extend.automorphisms(plane.projective_plane(n)) == aut
        assert extend.expected_completions(n, aut) == completions
        assert extend.count_completions(extend.pencil_and_line(n), n) == completions


def test_random_search_respects_the_bound():
    sizes = extend.search_maximal(4, range(6, 19), trials=3, seed=0)
    assert sizes and max(sizes) <= 21 - 6


def test_symbol_counts_and_shared_symbol():
    deck = chance.deck_55(3, 30)
    assert chance.count_profile(deck) == {6: 1, 7: 14, 8: 42}
    assert chance.shared_symbol(deck) == chance.shared_symbol_by_pairs(deck)
    assert set(chance.shared_symbol(chance.PLANE).values()) == {Fraction(1, 57)}
    assert sum(chance.shared_symbol(deck).values()) == 1
    assert chance.fixed_symbol_count() == {
        6: Fraction(1, 57),
        7: Fraction(14, 57),
        8: Fraction(14, 19),
    }


def test_three_cards():
    assert chance.three_share(chance.PLANE) == Fraction(6, 55)
    assert chance.three_share(chance.deck_55()) == Fraction(6, 55)
    # By brute force on the 55-card deck: triples of cards with a common symbol.
    deck = chance.deck_55()
    common = sum(1 for a, b, c in combinations(deck, 3) if a & b & c)
    assert Fraction(common, 26235) == Fraction(6, 55)


def test_triplet_free_sets():
    counts = chance.triplet_free_counts(chance.PLANE)
    assert counts[:6] == [chance.triplet_free_formula(k) for k in range(6)]
    assert counts[8] == 7**5 - 7**2 and counts[9] == 0  # conics; no nine
    assert counts[7] == 8 * counts[8]
    # For odd q the q + 1 card sets are the conics; for q = 4 there are 168 hyperovals.
    assert chance.triplet_free_sets(plane.projective_plane(5), 6) == chance.conics(5)
    pg5 = chance.triplet_free_counts(plane.projective_plane(5), kmax=7)
    assert pg5[6] == 5**5 - 5**2 and pg5[5] == 6 * pg5[6] and pg5[7] == 0
    pg4 = chance.triplet_free_counts(plane.projective_plane(4), kmax=7)
    assert pg4[6] == 168 and pg4[7] == 0
