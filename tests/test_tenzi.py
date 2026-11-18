"""Tenzi: exact turn distributions, checked against seeded simulation."""

import random
from fractions import Fraction
from pathlib import Path
from statistics import mean, stdev

from tenzi import tenzi


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


def test_monte_carlo_game_is_the_fixed_face_strategy():
    from tenzi.monte_carlo import tenzi_game

    games = [tenzi_game(rng=random.Random(i)) for i in range(4000)]
    half = 4 * stdev(games) / len(games) ** 0.5
    assert abs(mean(games) - float(tenzi.expected_turns())) < half


def test_target_first_is_the_longest_of_ten_geometric_waits():
    first, step = tenzi.STRATEGIES["target first"]
    mean_t, var = tenzi.target_first_moments()
    assert tenzi.expected_turns(step, first) == mean_t
    dist_mean, dist_var = tenzi.moments(tenzi.turn_distribution(200, step, first))
    assert abs(dist_mean - mean_t) < 1e-12 and abs(dist_var - var) < 1e-9
    # The MathFest slides report 15.565 and 6.834: the same game without its first roll.
    assert round(float(mean_t) - 1, 3) == 15.565 and round(float(var) ** 0.5, 3) == 6.834


def test_race_identities_exactly():
    first, step = tenzi.STRATEGIES["target first"]
    states = tenzi.state_distributions(60, step, first)
    done = [s.get(tenzi.DICE, 0) for s in states]
    win, tie, one_left = tenzi.race(1, states)
    assert win == done and tie == 0 and one_left == 0
    win, tie, _ = tenzi.race(2, states)
    assert tie == sum(p * p for p in done)  # both players finish on the same turn
    alive = 1 - sum(done)
    assert sum(win) == 1 - alive**2


def test_race_agrees_with_simulation():
    for name in ("target first", "switching"):
        first, step = tenzi.STRATEGIES[name]
        states = [
            {m: float(p) for m, p in s.items()} for s in tenzi.state_distributions(100, step, first)
        ]
        win, tie, one_left = tenzi.race(3, states)
        sims = [tenzi.play_race(random.Random(i), 3, name) for i in range(3000)]
        exact = (sum(t * w for t, w in enumerate(win, 1)), tie, one_left)
        samples = (
            [t for t, _, _ in sims],
            [float(k > 1) for _, k, _ in sims],
            [c for *_, c in sims],
        )
        for value, xs in zip(exact, samples, strict=True):
            assert abs(mean(xs) - value) < 4 * stdev(xs) / len(xs) ** 0.5


def test_optimal_matches_value_iteration_on_small_games():
    from tenzi import optimal

    for dice, faces in [(3, 2), (4, 2), (3, 3), (4, 3)]:
        brute = optimal.brute_force(dice, faces)
        for any_dice in (False, True):
            assert abs(optimal.solve(dice, faces, any_dice)[1] - brute) < 1e-9


def test_switching_is_optimal():
    from tenzi import optimal

    for any_dice in (False, True):
        policy, best, values = optimal.solve(any_dice=any_dice)
        assert all(keep == (roll[0],) for roll, keep in policy.items())
        assert best == tenzi.expected_turns(tenzi.reroll_switching)
        assert values[(1,)] == values[()]  # one kept die is as good as a rerolled one
