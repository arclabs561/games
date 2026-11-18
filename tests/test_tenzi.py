"""Tenzi: exact turn distributions, checked against seeded simulation."""

import random
import sys
from fractions import Fraction
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).parent.parent / "tenzi"))

import tenzi  # noqa: E402


def test_first_roll():
    first = tenzi.first_roll()
    assert sum(first.values()) == 1
    assert min(first) == 2  # ten dice on six faces: some face shows at least twice
    assert first[10] == Fraction(6, 6**10)


def test_distribution_sums_to_the_mean():
    for step in (tenzi.reroll, tenzi.reroll_switching):
        dist = tenzi.turn_distribution(200, step)
        assert 1 - sum(dist.values()) < Fraction(1, 10**12)
        assert abs(sum(t * p for t, p in dist.items()) - tenzi.expected_turns(step)) < 1e-9


def test_switching_helps_a_little():
    fixed, switching = tenzi.expected_turns(), tenzi.expected_turns(tenzi.reroll_switching)
    assert 0 < fixed - switching < Fraction(1, 10)


def test_simulation_agrees_with_exact_mean():
    for play, step in (
        (tenzi.play_fixed, tenzi.reroll),
        (tenzi.play_switching, tenzi.reroll_switching),
    ):
        games = [play(random.Random(i)) for i in range(4000)]
        half = 4 * stdev(games) / len(games) ** 0.5
        assert abs(mean(games) - float(tenzi.expected_turns(step))) < half


def test_committed_csv_is_current(tmp_path, monkeypatch):
    monkeypatch.setattr(tenzi, "CSV", tmp_path / "turns.csv")
    tenzi.write_csv()
    committed = (Path(tenzi.__file__).parent / "turns.csv").read_text()
    assert (tmp_path / "turns.csv").read_text() == committed


def test_examples_tenzi_game_is_the_fixed_face_strategy():
    sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
    from dice_monte_carlo import tenzi_game

    games = [tenzi_game(rng=random.Random(i)) for i in range(4000)]
    half = 4 * stdev(games) / len(games) ** 0.5
    assert abs(mean(games) - float(tenzi.expected_turns())) < half
