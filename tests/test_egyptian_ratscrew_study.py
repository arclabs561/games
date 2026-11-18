"""Egyptian Ratscrew study: seeded batches, win-chance models, and the recorded large runs."""

import json
import random
from collections import Counter
from math import exp

import pytest

from egyptian_ratscrew.game import play_game
from egyptian_ratscrew.study import (
    RUNS,
    chunk_summary,
    entropy,
    fit,
    games,
    length_summary,
    power,
    proportional,
    random_shares,
    rotated,
    softmax,
    spearman,
    two_player_against_field,
    vector_wins,
    wilson,
)


def test_wilson_interval():
    lo, hi = wilson(514, 1000)
    assert lo < 0.514 < hi and abs((hi - lo) / 2 - 0.031) < 0.001
    assert wilson(0, 10)[0] == pytest.approx(0, abs=1e-12)


def test_rotated_games_report_players_not_seats():
    """Game g of a rotated batch is an ordinary game with the shares moved round by g seats."""
    shares = [0.6, 0.3, 0.1]
    batch = games(shares, 6, 7, rotate=True)
    rng = random.Random(7)
    for g, game in enumerate(batch):
        seat = g % 3
        plain = play_game(rotated(shares, seat), rng)
        assert game.cards == plain.cards
        assert plain.winner is not None and game.winner == (plain.winner - seat) % 3


def test_models_agree_in_two_player_games():
    """Every model is a share's strength over the total; in two-player games the softmax and
    field models are both the logistic curve in 2s - 1."""
    for s in (0.5, 0.6, 0.9):
        assert softmax(5)([s, 1 - s])[0] == pytest.approx(1 / (1 + exp(-5 * (2 * s - 1))))
        assert two_player_against_field(5)([s, 1 - s]) == pytest.approx(softmax(5)([s, 1 - s]))
    shares = [0.5, 0.3, 0.2]
    for model in (proportional, power(2.5), softmax(8), two_player_against_field(5)):
        assert sum(model(shares)) == pytest.approx(1)
        assert model([1 / 3] * 3) == pytest.approx([1 / 3] * 3)
    assert power(1)(shares) == pytest.approx(shares)


def test_fit_recovers_a_known_parameter():
    data = []
    for shares in ([0.6, 0.4], [0.5, 0.3, 0.2], [0.4, 0.3, 0.2, 0.1]):
        data.append((shares, [round(100_000 * p) for p in softmax(7)(shares)]))
    assert fit(softmax, data, 0.1, 30) == pytest.approx(7, abs=0.01)
    data = [(s, [round(100_000 * p) for p in power(2.5)(s)]) for s, _ in data]
    assert fit(power, data, 0.1, 10) == pytest.approx(2.5, abs=0.01)


def test_summaries():
    assert length_summary(Counter({10: 3, 20: 1}))["median"] == 10
    assert length_summary(Counter({10: 3, 20: 1}))["mean"] == 12.5
    assert entropy([0.25] * 4) == pytest.approx(entropy([0.5, 0.5]) * 2)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1)
    rng = random.Random(1)
    for players in (3, 4):
        shares = random_shares(players, rng)
        assert len(shares) == players and sum(shares) == pytest.approx(1)


@pytest.fixture(scope="module")
def runs():
    return json.loads(RUNS.read_text())


def replay(run, chunk=0):
    return chunk_summary((run["shares"], run["seed"], chunk, run["options"]))["wins"]


def test_recorded_runs_replay(runs):
    """The first chunk of several recorded runs, played again, gives the recorded wins."""
    assert replay(runs["first_player_2"]) == runs["first_player_2"]["chunk_wins"][0]
    assert replay(runs["beggar_my_neighbour"]) == runs["beggar_my_neighbour"]["chunk_wins"][0]
    for run in (runs["two_player_curve"][-1], runs["misslaps"][-1]):
        assert replay(run) == run["chunk_wins"][0]


def test_recorded_vector_replays(runs):
    """The recorded share vector with the most unequal shares, played again in full."""
    record = max(runs["vectors_3"], key=lambda r: max(r["shares"]))
    assert vector_wins((record["shares"], record["seed"])) == record


def test_recorded_runs_are_consistent(runs):
    for name in ("first_player_2", "first_player_4", "beggar_my_neighbour"):
        run = runs[name]
        assert sum(run["wins"]) + run["unfinished"] == run["games"]
        assert [sum(c[i] for c in run["chunk_wins"]) for i in range(len(run["wins"]))] == run[
            "wins"
        ]
        assert sum(run["tricks_counts"].values()) == sum(run["wins"])
    for name in ("vectors_3", "vectors_4"):
        assert all(sum(r["wins"]) == 4_000 for r in runs[name])
