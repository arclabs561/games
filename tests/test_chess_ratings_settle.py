"""Chess: games until a new rating settles, checked by brute force, iteration and simulation."""

import json
from itertools import product
from pathlib import Path

import numpy as np

from chess_ratings import settle
from chess_ratings.glicko import NEW, Glicko, lichess_game
from chess_ratings.settle import (
    color_advantage,
    deviation_path,
    exact_distribution,
    games_to_establish,
    games_to_establish_formula,
    summary,
)
from peggle.peggle import wins_needed

RESULTS = Path(__file__).parent.parent / "chess_ratings" / "results"


def test_closed_form_agrees_with_iteration():
    for rd in (45, 60, 80, 110, 150, 200, 350, 500):
        assert games_to_establish_formula(rd) == games_to_establish(rd, advantage=0.0)


def test_deviation_path_does_not_depend_on_results():
    path = deviation_path(12)
    for record in [(1,) * 12, (0,) * 12, (1, 0) * 6, (0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1)]:
        me = NEW
        for k, s in enumerate(record):
            opp = Glicko(me.rating, settle.OPPONENT_RD, 0.06)
            if k % 2 == 0:
                me, _ = lichess_game(me, opp, float(s))
            else:
                _, me = lichess_game(opp, me, 1.0 - s)
            assert abs(me.deviation - path[k]) < 1e-9


def test_dynamic_program_against_every_record():
    for n in range(1, 9):
        ratings, probs = exact_distribution(1100, n)
        brute: dict[float, float] = {}
        for record in product((1, 0), repeat=n):
            me, p = NEW, 1.0
            for k, s in enumerate(record):
                q = float(settle.win_chance(1100, me.rating))
                p *= q if s else 1 - q
                opp = Glicko(me.rating, settle.OPPONENT_RD, 0.06)
                if k % 2 == 0:
                    me, _ = lichess_game(me, opp, float(s))
                else:
                    _, me = lichess_game(opp, me, 1.0 - s)
            key = round(me.rating, 9)
            brute[key] = brute.get(key, 0.0) + p
        got: dict[float, float] = {}
        for r, p in zip(ratings, probs, strict=True):
            got[round(float(r), 9)] = got.get(round(float(r), 9), 0.0) + float(p)
        assert brute.keys() == got.keys()
        assert all(abs(brute[k] - got[k]) < 1e-12 for k in brute)


def test_vectorized_update_matches_lichess_game():
    rng = np.random.default_rng(0)
    r = rng.uniform(800, 2200, 50)
    phi = rng.uniform(45, 400, 50) / settle.SCALE
    opp = r + rng.normal(0, 150, 50)
    score = rng.choice([0.0, 0.5, 1.0], 50)
    for game in (0, 1):
        r_new, phi_new = settle.update(r, phi, opp, 70.0, score, game)
        for i in range(50):
            me = Glicko(r[i], phi[i] * settle.SCALE, 0.06)
            them = Glicko(opp[i], 70.0, 0.06)
            if color_advantage(game) > 0:
                want, _ = lichess_game(me, them, score[i])
            else:
                _, want = lichess_game(them, me, 1 - score[i])
            assert abs(want.rating - r_new[i]) < 1e-6
            assert abs(want.deviation - phi_new[i] * settle.SCALE) < 1e-6


def test_simulation_agrees_with_exact_distribution():
    exact = summary(*exact_distribution(1900, 12), 1900)
    paths = settle.simulate(1900, 12, 40_000, seed=11)
    share = float((np.abs(paths[12] - 1900) <= 100).mean())
    lo, hi = settle.wilson(int(share * 40_000), 40_000)
    assert lo <= exact["within_100"] <= hi
    assert abs(paths[12].mean() - exact["mean"]) < 4 * exact["sd"] / 200


def test_fewest_wins_matches_peggle():
    assert all(settle.fewest_wins(n) == wins_needed(n) for n in range(4, 300))


def test_noise_floor_matches_simulation():
    paths = settle.simulate(1500, 400, 4000, seed=12)
    assert abs(paths[200:].std(axis=1).mean() - settle.noise_floor()) < 1.5


def test_committed_results_are_what_the_script_writes():
    committed = json.loads((RESULTS / "settle.json").read_text())
    assert committed == json.loads(settle.to_text(settle.compute()))
