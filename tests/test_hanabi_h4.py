"""Hanabi, plan H4: JaxMARL against rules.py, the IPPO loop, and the statistics."""

import json
import random

import numpy as np
import pytest

from hanabi import h4
from hanabi.jax_bridge import (
    H4_RULES,
    max_turns,
    noop,
    replay,
    to_jaxmarl_action,
    to_rules_action,
)
from hanabi.rules import Rules, hint, legal_actions, new_game, play, seeded_deck, step


def random_states(rules: Rules, n: int, seed: int = 0):
    """States met in random play, for checks that need many positions."""
    rng = random.Random(seed)
    for g in range(n):
        state = new_game(rules, seeded_deck(rules, seed + g))
        while not state.over:
            yield state
            step(state, rng.choice(legal_actions(state)))


@pytest.mark.parametrize("players", [2, 3, 4])
def test_action_numbers_round_trip(players):
    rules = Rules(players=players, colors=2)
    for state in random_states(rules, 30):
        for action in legal_actions(state):
            a = to_jaxmarl_action(action, state.current, rules)
            assert 0 <= a < noop(rules)
            assert to_rules_action(a, state.current, rules) == action


def test_turn_bound_holds_in_random_play():
    longest = max(len(list(random_states(H4_RULES, 1, seed))) for seed in range(300))
    assert longest <= max_turns(H4_RULES)


def test_replay_matches_a_game_played_in_rules_py():
    rng = random.Random(3)
    for seed in range(20):
        deck = seeded_deck(H4_RULES, seed)
        state, numbers = new_game(H4_RULES, deck), []
        while not state.over:
            action = rng.choice(legal_actions(state))
            numbers.append(to_jaxmarl_action(action, state.current, H4_RULES))
            step(state, action)
        again, events = replay(numbers + [0, 5, 10], H4_RULES, deck)  # extra moves are ignored
        assert again.score == state.score and len(events) == len(numbers)
    with pytest.raises(ValueError):
        replay([noop(H4_RULES)], H4_RULES, seeded_deck(H4_RULES, 0))


def test_hint_then_play_is_counted_for_the_hinter():
    rules = H4_RULES
    deck = next(
        d
        for d in (seeded_deck(rules, s) for s in range(400_000, 400_100))
        if any(c % rules.ranks == 0 for c in d[5:10])
    )
    one = next(i for i, c in enumerate(deck[5:10]) if c % rules.ranks == 0)
    state, numbers = new_game(rules, deck), []
    for action in (hint(1, "rank", 1), play(one)):
        numbers.append(to_jaxmarl_action(action, state.current, rules))
        step(state, action)
    counts = h4.replay_counts(numbers, deck)
    assert counts[0, h4.F["rank_hints"]] == 1 and counts[0, h4.F["rank_hint_then_play"]] == 1
    assert counts[1, h4.F["plays"]] == 1 and counts[1, h4.F["misplays"]] == 0
    assert counts[1, h4.F[f"play_slot{one}"]] == 1
    with pytest.raises(AssertionError):
        h4.replay_counts(numbers, deck, score=1)  # the game has not ended


def test_iqm_and_bootstrap_on_known_matrices():
    assert h4.iqm([0, 1, 2, 3, 4, 5, 6, 100]) == 3.5
    rng = np.random.default_rng(1)
    deals = 50
    # arm a: every pair scores 6 on every deal; arm b: self-play 9, cross-play 3
    a = np.full((4, 4, deals), 6.0)
    b = np.full((4, 4, deals), 3.0)
    for i in range(4):
        b[i, i] = 9.0
    stats = h4.arm_stats({"a": a, "b": b + rng.normal(0, 1e-9, b.shape)})
    assert stats["a"]["cross"] == {"value": 6.0, "interval": [6.0, 6.0]}
    assert stats["b"]["self_minus_cross"]["value"] == pytest.approx(6.0)
    diff = stats["cross b minus a"]
    assert diff["value"] == pytest.approx(-3.0)
    assert diff["interval"][0] == pytest.approx(-3.0) and diff["interval"][1] == pytest.approx(-3.0)


def test_bootstrap_interval_covers_a_seed_effect():
    # seeds differ by a constant; deals add noise. The interval must be wider than 0.
    rng = np.random.default_rng(2)
    n, deals = 8, 200
    seed_effect = rng.normal(0, 1, n)
    x = seed_effect[:, None, None] + seed_effect[None, :, None] + rng.normal(0, 1, (n, n, deals))
    boot = h4.two_level({"x": x}, reps=2000)
    lo, hi = np.nanpercentile(boot["x"]["cross"], [2.5, 97.5])
    point = h4.point(x)["cross"]
    assert lo < point < hi and hi - lo > 0.5


def test_committed_ceilings_match_a_fresh_solve():
    table = json.loads(h4.CEILINGS.read_text())
    assert table["seeds"] == [400_000, 401_999] and len(table["ceiling"]) == 2000
    assert table["winnable"] == table["ceiling"].count(10)
    for k in list(range(0, 2000, 97)) + [31, 84]:
        assert h4.ceiling(400_000 + k)["ceiling"] == table["ceiling"][k]
