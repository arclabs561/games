"""Hanabi, plan H1: the second solver, the free-order test, and the merged records."""

import random
from collections import Counter
from functools import cache

import pytest

from hanabi import dfs_solver as sb
from hanabi.h1 import summarize
from hanabi.rules import Rules, full_deck, make_card, seeded_deck
from hanabi.solver import brute_winnable, root_certificates, winnable
from hanabi.upper_bound import classify, deals, multiset_orders

TINY = [
    Rules(players=2, colors=2, counts=(2, 1), hand_size=1, max_hints=2, lives=1),
    Rules(players=2, colors=2, counts=(2, 1), hand_size=2, max_hints=1, lives=2),
    Rules(players=3, colors=2, counts=(2, 1), hand_size=1, max_hints=1, lives=2),
    Rules(
        players=1,
        colors=1,
        counts=(3, 2, 2, 1),
        hand_size=2,
        max_hints=0,
        lives=1,
        discard_when_full=True,
    ),
]


@pytest.mark.parametrize("rules", TINY)
def test_second_solver_agrees_with_brute_force_on_every_deal(rules):
    for deck in multiset_orders(full_deck(rules)):
        brute = brute_winnable(rules, deck)
        for cuts in ("none", "time", "all"):
            ok, actions = sb.solve(rules, deck, cuts=cuts)
            assert ok == brute, (deck, cuts)
            if ok:
                assert sb.replay_actions(rules, deck, actions) == rules.max_score
        if not brute:
            continue
        assert sb.free_order_test(rules, deck), deck  # a won deal is won in the relaxation
        assert not sb.certificates(rules, deck)


def test_second_solver_reproduces_a_small_deck_count():
    rules = Rules(players=2, colors=1, hand_size=3)
    won = sum(w for deck, w in deals(rules) if sb.solve(rules, deck)[0])
    assert won == 72006


def best_schedule(have: Counter, deck: list[int], top: list[int], ranks: int, last: int) -> int:
    """The most plays in the schedule relaxation, by trying every choice."""

    @cache
    def go(have_t, k, top_t, left):
        if k == len(deck) and left == 0:
            return 0
        held = Counter(dict(have_t))
        best = 0
        options = [None] + [
            c for c in range(len(top_t)) if top_t[c] < ranks and held[c * ranks + top_t[c]]
        ]
        for c in options:
            h, t = held.copy(), list(top_t)
            got = 0
            if c is not None:
                h[c * ranks + t[c]] -= 1
                t[c] += 1
                got = 1
            if k < len(deck):
                h[deck[k]] += 1
                nxt = (tuple(sorted((+h).items())), k + 1, tuple(t), left)
            else:
                nxt = (tuple(sorted((+h).items())), k, tuple(t), left - 1)
            best = max(best, got + go(*nxt))
        return best

    return go(tuple(sorted(have.items())), 0, tuple(top), last)


def test_greedy_schedule_is_optimal():
    rules = Rules(players=2, colors=2, counts=(2, 1, 1), hand_size=2)
    rng = random.Random(1)
    for _ in range(200):
        deck = full_deck(rules)
        rng.shuffle(deck)
        g = sb.Game(rules, deck)
        pile, hands, pos = g.start()
        have = Counter(c for h in hands for c in h)
        want = best_schedule(have, deck[pos:], list(pile), rules.ranks, rules.players)
        assert sb.schedule_plays(g, pile, hands, pos, rules.players) == want, deck


def deck_of(names: str, rules: Rules) -> list[int]:
    return [make_card("RYGBW".index(n[0]), int(n[1]), rules) for n in names.split()]


def test_free_order_catches_the_turn_order_losses_of_the_readme():
    one = Rules(players=2, colors=1, hand_size=2)
    two = Rules(colors=2, counts=(3, 2, 1), hand_size=4)
    for rules, names in [
        (one, "R1 R1 R4 R5 R1 R3 R3 R2 R2 R4"),
        (two, "R1 R1 R1 R2 R3 Y2 Y2 Y3 R2 Y1 Y1 Y1"),
    ]:
        deck = deck_of(names, rules)
        assert not brute_winnable(rules, deck)
        assert not root_certificates(rules, deck)
        assert "free-order" in sb.certificates(rules, deck)


def test_free_order_never_drops_a_win_on_a_small_deck():
    rules = Rules(players=2, colors=1, hand_size=2)
    lost = Counter()
    for deck, w in deals(rules):
        won = winnable(rules, deck)[0]
        free = sb.free_order_test(rules, deck)
        assert free or not won
        lost[won, free] += w
    assert lost == {(True, True): 72138, (False, False): 3462}


def test_a_full_two_player_deal_lost_only_to_turn_order():
    rules = Rules(players=2)
    deck = seeded_deck(rules, 12365)
    assert root_certificates(rules, deck) == []
    assert sb.free_order_test(rules, deck) is False
    assert sb.last_card_test(rules, deck)  # the last card is B4, the other B4 comes too late


@pytest.mark.parametrize("rules", TINY[:3])
def test_last_card_test_never_drops_a_win(rules):
    for deck in multiset_orders(full_deck(rules)):
        assert not (sb.last_card_test(rules, deck) and brute_winnable(rules, deck)), deck


def test_last_card_on_a_small_deck():
    rules = Rules(players=2, colors=1, hand_size=2)
    tally = Counter()
    for deck, w in deals(rules):
        won = winnable(rules, deck)[0]
        last = sb.last_card_test(rules, deck)
        assert not (won and last)
        tally[won, bool(root_certificates(rules, deck)), last] += w
    assert tally[False, False, True] == 240 and tally[False, False, False] == 312


def test_classify_gives_both_verdicts():
    rec = classify(Rules(players=3), 0, solver="both")
    assert rec["verdict"] == rec["verdict_b"] == "won"
    rec = classify(Rules(players=2), 161, solver="both", b_budgets=(1000,))
    assert rec["verdict"] == "lost" and rec["free_order"] == "lost" and rec["best"] == 24


def test_summary_stops_on_a_disagreement():
    rec = {"players": 2, "certificates": [], "seconds": 0, "seconds_b": 0, "budget": 1}
    recs = {(2, s): {**rec, "seed": s, "verdict": "won", "verdict_b": "won"} for s in range(9)}
    out = summarize(recs, range(9))[2]
    assert out["won"] == 9 and out["agree"] == 9 and out["share"]["low"] == 1
    recs[2, 3] = {**recs[2, 3], "verdict_b": "lost"}
    with pytest.raises(AssertionError, match="disagree"):
        summarize(recs, range(9))


def test_constraint_model_on_a_tiny_deck():
    pytest.importorskip("ortools")
    from hanabi.solver_cp import replay_cp, winnable_cp

    rules = TINY[1]
    for deck in list(multiset_orders(full_deck(rules)))[::9]:
        v, line = winnable_cp(rules, deck, 10)
        assert v == ("won" if brute_winnable(rules, deck) else "lost"), deck
        if line:
            assert replay_cp(rules, deck, line) == rules.max_score
