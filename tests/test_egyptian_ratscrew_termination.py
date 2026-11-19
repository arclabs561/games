"""Egyptian Ratscrew as a Markov chain: the move function, trapped states, and exact tail rates."""

import random
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from egyptian_ratscrew.game import deal, play
from egyptian_ratscrew.termination import (
    LOOPS,
    _transitions,
    beggar_loop_tricks,
    can_end,
    chain,
    deals,
    decay_rate,
    decks_with_traps,
    ended,
    fast_beggar_loop,
    moves,
    one_player_wins_slaps,
    play_from,
    royal_slap,
    slap_free_cycle,
    start,
    survival,
    tail_rate,
    template_decks,
    uniform_bound,
)

SMALL = (2, 2, 3, 3, "J", "Q")
TRAP = (2, 2, 3, 3, "J", "J")


def test_moves_replay_play_game():
    """Drawing slap winners with the same random numbers, `moves` plays the same games as
    `game.play`, for two, three and four players with uneven shares."""
    for players in (2, 3, 4):
        for seed in range(40):
            shares = [random.Random(seed).random() + 0.1 for _ in range(players)]
            hands = deal(players, random.Random(seed))
            expected = play(shares, random.Random(seed + 1), hands=hands)
            assert play_from(start(hands), shares, random.Random(seed + 1)) == expected


def test_a_slap_offers_every_player_the_pile():
    [(_, first)] = moves(start([(5, 2), (5, 3)]))
    after = moves(first)  # 5 on 5 is a pair
    assert [w for w, _ in after] == [0, 1]
    assert after[0][1].hands == ((2, 5, 5), (3,)) and after[1][1].hands == ((2,), (3, 5, 5))
    assert [w for w, _ in moves(first, slaps=False)] == [None]


def test_the_six_card_trap():
    """J 2 3 against 2 J 3 never offers a slap and never ends: after two cards it enters a
    loop of six cards, the tricks 2 J 3 and 3 J 2."""
    state = start([("J", 2, 3), (2, "J", 3)])
    loop = slap_free_cycle(state)
    assert loop is not None and len(loop) == 6
    tricks = beggar_loop_tricks(state)
    assert sorted(tricks) == [(2, "J", 3), (3, "J", 2)]
    assert play_from(state, [0.5, 0.5], random.Random(0), max_cards=1_000) == (1_000, None)


def test_trapped_states_of_the_six_card_deck():
    c = chain(TRAP)
    good = can_end(c)
    trapped = [k for k, g in enumerate(good) if not g]
    assert len(c.states) == 804 and len(trapped) == 16
    assert sum(w for k, w in c.deals.items() if not good[k]) == 16
    assert all(slap_free_cycle(c.states[k]) is not None for k in trapped)


def test_deal_weights_count_orders():
    assert sum(w for _, w in deals(SMALL)) == 720
    assert sum(w for _, w in deals(TRAP, 3)) == 720


def test_one_player_winning_every_slap_ends_every_game_that_can_end():
    c = chain(SMALL)
    assert all(can_end(c))
    assert all(any(one_player_wins_slaps(s, p) for p in (0, 1)) for s in c.states)


def test_uniform_bound_holds_from_every_state():
    """Theorem 1's bound P(T > m L) <= (1 - delta)^m, checked from every state for m <= 5."""
    c = chain(SMALL)
    shares = [0.5, 0.5]
    steps, delta = uniform_bound(c, shares)
    assert (steps, delta) == (12, 0.25)
    src, dst, prob = _transitions(c, shares)
    alive = ~np.array(c.ended)
    u = alive.astype(float)
    for t in range(1, 5 * steps + 1):
        u = np.bincount(src, weights=prob * u[dst], minlength=len(u)) * alive
        assert u.max() <= (1 - delta) ** (t // steps) + 1e-12


def test_decay_rate_is_the_largest_eigenvalue():
    """Power iteration against a dense eigenvalue computation on the six-card deck, for two
    share vectors."""
    c = chain(SMALL)
    alive = ~np.array(c.ended)
    for shares in ([0.5, 0.5], [0.8, 0.2]):
        src, dst, prob = _transitions(c, shares)
        q = np.zeros((len(c.states), len(c.states)))
        keep = alive[src] & alive[dst]
        np.add.at(q, (src[keep], dst[keep]), prob[keep])
        assert decay_rate(c, shares) == pytest.approx(max(abs(np.linalg.eigvals(q))), abs=1e-9)


def test_mean_length_two_ways():
    """The sum of the survival chances equals the mean length from solving (I - Q) m = 1."""
    c = chain(SMALL)
    shares = [0.5, 0.5]
    src, dst, prob = _transitions(c, shares)
    alive = ~np.array(c.ended)
    n = len(c.states)
    q = np.zeros((n, n))
    keep = alive[src] & alive[dst]
    np.add.at(q, (src[keep], dst[keep]), prob[keep])
    m = np.linalg.solve(np.eye(n) - q, alive.astype(float))
    weights = np.zeros(n)
    for k, w in c.deals.items():
        weights[k] = w / 720
    assert survival(c, shares, 500).sum() == pytest.approx(weights @ m, abs=1e-9)


def test_survival_by_listing_every_deal_and_slap():
    """P(the game lasts more than t cards) for t <= 6, by enumerating every order of the deck
    and every run of slap winners, in exact fractions."""
    shares = [Fraction(1, 2)] * 2

    def last_more_than(state, t, winners):
        for _ in range(t):
            if ended(state):
                return False
            options = moves(state)
            state = options[0][1] if len(options) == 1 else options[next(winners)][1]
        return not ended(state)

    c = chain(SMALL)
    exact = survival(c, [0.5, 0.5], 6)
    for t in range(7):
        total = Fraction(0)
        for s, weight in deals(SMALL):
            for winners in product((0, 1), repeat=t):
                if last_more_than(s, t, iter(winners)):
                    total += weight * shares[0] ** t
        assert float(total / 720) == pytest.approx(exact[t], abs=1e-12)


def test_known_loops_offer_royal_slaps():
    """Casella et al.'s 62-trick cycle: 440 cards, and 7 of its tricks hold a royal pair or
    sandwich, so a slap is offered whatever ranks the other cards have."""
    tricks = beggar_loop_tricks(start(LOOPS[0]))
    assert (len(tricks), sum(map(len, tricks)), sum(map(royal_slap, tricks))) == (62, 440, 7)
    for hands in LOOPS:
        assert sum(map(len, hands)) == 52
        assert any(map(royal_slap, beggar_loop_tricks(start(hands))))


def test_royal_slap():
    assert royal_slap(("-", "Q", "-", "Q")) and royal_slap(("A", "A"))
    assert not royal_slap(("-", "-")) and not royal_slap(("K", "-", "-", "K"))


def test_tail_rate():
    rate, lo, hi = tail_rate([10, 20, 30, 40], 10)  # three endings over 60 cards
    assert rate == pytest.approx(0.05) and lo < rate < hi


def test_no_trap_with_four_cards():
    assert decks_with_traps(4, 2) == [] and decks_with_traps(4, 3) == []


def test_fast_loop_finder_agrees_with_the_slow_one():
    """On every 20,000th of the first 600,000 template decks: same ending or looping, same loop length, and the same
    answer about royal slaps."""
    from itertools import islice

    for deck in islice(template_decks(), 0, 600_000, 20_000):
        fast = fast_beggar_loop(deck, "J-")
        try:
            slow = beggar_loop_tricks(start([deck, "J-"]))
        except ValueError:
            slow = None
        assert (fast is None) == (slow is None)
        if fast is not None and slow is not None:
            assert sum(map(len, fast)) == sum(map(len, slow))
            assert any(map(royal_slap, fast)) == any(map(royal_slap, slow))
