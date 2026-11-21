"""Tenzi: exact turn distributions, checked against seeded simulation."""

import json
import random
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path
from statistics import mean, stdev

from tenzi import maximum, tenzi


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
    for seed, play, step in (
        (11, tenzi.play_fixed, tenzi.reroll),
        (12, tenzi.play_switching, tenzi.reroll_switching),
    ):
        rng = random.Random(seed)  # one generator per check, so the checks are independent
        games = [play(rng) for _ in range(4000)]
        half = 4 * stdev(games) / len(games) ** 0.5
        assert abs(mean(games) - float(tenzi.expected_turns(step))) < half


def test_committed_csv_is_current(tmp_path, monkeypatch):
    monkeypatch.setattr(tenzi, "CSV", tmp_path / "turns.csv")
    tenzi.write_csv()
    committed = (Path(tenzi.__file__).parent / "turns.csv").read_text()
    assert (tmp_path / "turns.csv").read_text() == committed


def test_monte_carlo_game_is_the_fixed_face_strategy():
    from tenzi.monte_carlo import tenzi_game

    rng = random.Random(13)
    games = [tenzi_game(rng=rng) for _ in range(4000)]
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
    for seed, name in ((14, "target first"), (15, "switching")):
        first, step = tenzi.STRATEGIES[name]
        states = [
            {m: float(p) for m, p in s.items()} for s in tenzi.state_distributions(100, step, first)
        ]
        win, tie, one_left = tenzi.race(3, states)
        rng = random.Random(seed)
        sims = [tenzi.play_race(rng, 3, name) for _ in range(3000)]
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


def test_by_turn_matches_brute_force_on_small_games():
    from tenzi import optimal

    for dice, faces in [(3, 2), (4, 2), (3, 3), (4, 3)]:
        brute = optimal.brute_force_by_turn(dice, faces, 8)
        for any_dice in (False, True):
            chances, worse = optimal.by_turn(dice, faces, any_dice)
            assert not worse
            assert all(abs(float(c) - b) < 1e-12 for c, b in zip(chances, brute, strict=False))


def test_switching_is_best_for_every_deadline():
    from tenzi import optimal

    switching = tenzi.turn_distribution(200, step=tenzi.reroll_switching)
    for any_dice in (False, True):
        chances, worse = optimal.by_turn(any_dice=any_dice)
        assert not worse and len(chances) == 163
        assert 1 - chances[-1] < Fraction(1, 10**12) <= 1 - chances[-2]
        cumulative = Fraction(0)
        for t, best in enumerate(chances, 1):  # the best chances are switching's own
            cumulative += switching[t]
            assert best == cumulative


def test_maximum_preserves_decimal_precision():
    import subprocess
    import sys

    # Collection may already have imported maximum in the test process.
    code = """
from decimal import getcontext
getcontext().prec = 17
from tenzi import maximum
assert getcontext().prec == 17
assert len(maximum.rate().as_tuple().digits) == 60
assert getcontext().prec == 17
"""
    subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], check=True
    )


@maximum.sixty_digits()
def test_harmonic_approximation_and_its_fourier_terms():
    from decimal import Decimal

    mean_t, _ = tenzi.target_first_moments()
    assert maximum.exact_mean(10) == mean_t
    for n in (1, 2, 3, 4):
        assert abs(maximum.tail_sum_mean(n) - maximum.exact_mean(n)) < Fraction(1, 10**25)
    gap = maximum.decimal(mean_t) - maximum.harmonic_approximation(10)
    assert f"{gap:.3e}" == "4.177e-10"
    # The identity is exact: the Fourier sum, cut at k = 200, accounts for all of the gap except
    # its tail, whose terms fall like k^-(n + 1).
    for n, tail in ((5, 10**-17), (10, 10**-30), (20, 10**-45), (50, 10**-55)):
        exact_gap = maximum.decimal(maximum.exact_mean(n)) - maximum.harmonic_approximation(n)
        assert abs(exact_gap - maximum.fourier_sum(n)) < Decimal(tail)
    # At ten dice the k = 1 term is nearly all of the gap, and it is at most its stated size.
    term, size = maximum.fourier_term(10, 6, 1), maximum.term_size(10)
    assert abs(term) <= size and abs(term - gap) < gap / 1000
    assert f"{maximum.wobble_amplitude():.2e}" == "1.45e-23"
    assert maximum.dice_for_wobble() == 856
    assert maximum.term_size(10**4) / maximum.wobble_amplitude() < Decimal("1.1")


def test_small_games_switching_is_best_for_every_deadline():
    from tenzi import optimal

    for dice in range(2, 7):
        for faces in range(2, 7):
            for any_dice in (False, True):
                assert not optimal.by_turn(dice, faces, any_dice)[1]


# ---------- results.json, which the README and the note read ----------

COMMITTED = json.loads((Path(tenzi.__file__).parent / "results.json").read_text())


def test_results_file_is_current():
    from tenzi import report

    quick = json.loads(report.dump(report.results(quick=True)))
    assert set(COMMITTED) == set(quick) | report.SLOW
    for key, value in quick.items():
        assert COMMITTED[key] == value, key
    assert COMMITTED["small_games_switching_best"] is True


def test_simulated_results_are_current():
    # Common random numbers would make the simulations agree with each other by construction.
    assert len(set(tenzi.SEEDS.values())) == len(tenzi.SEEDS)
    sims = COMMITTED["simulated"]
    assert sims["seeds"] == tenzi.SEEDS
    for name, (first, step) in tenzi.STRATEGIES.items():
        center, half = (
            float(x) for x in tenzi.interval(tenzi.simulate(name, sims["games"])).split(" ± ")
        )
        assert [center, half] == sims[name], name
        assert abs(center - float(tenzi.expected_turns(step, first))) < half


def test_quoted_values_agree_with_the_scripts():
    r, h = COMMITTED, COMMITTED["harmonic"]
    # the slides report the named-face game without its first roll, and these state vectors
    assert f"{r['target_first_shifted']['done_by_4']:.4f}" == "0.0014"
    assert f"{r['target_first_shifted']['on_20']:.4f}" == "0.0401"
    mean_t, var = tenzi.target_first_moments()
    assert r["strategies"]["target first"]["mean"] == float(mean_t)
    assert f"{r['strategies']['target first']['sd']:.3f}" == f"{float(var) ** 0.5:.3f}" == "6.834"
    assert h["gap_10_sci"] == [4.18, 10] and f"{h['gap_10']:.2e}" == "4.18e-10"
    assert h["exact_10"] == f"{maximum.decimal(maximum.exact_mean(10)):.12f}"
    # the race table quotes these players; the note's chain figure, these counts
    assert set(r["race"]["switching"]) == {"2", "3", "4", "6", "10", "20", "30"}
    assert r["turns_left_fixed"]["9"] == 6.0


def floats(value):
    if isinstance(value, float):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from floats(v)
    elif isinstance(value, list):
        for v in value:
            yield from floats(v)


def test_no_quoted_float_sits_on_a_rounding_tie():
    """Python rounds a float's binary value; Typst rounds its shortest decimal form, half away
    from zero (2.675 gives 2.67 in Python and 2.68 in Typst). They agree unless the shortest
    form ends in a 5 just past the digits printed, so no stored value may."""
    for x in floats(COMMITTED):
        for digits in range(5):
            typst = Decimal(repr(x)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
            assert f"{x:.{digits}f}" == f"{typst:.{digits}f}", (x, digits)


def test_note_reads_its_numbers_from_results():
    note = (Path(tenzi.__file__).parent / "tenzi.typ").read_text()
    assert '#let R = json("results.json")' in note
    for name in ("rules", "results", "checks", "open"):
        assert f'shared(readme, "{name}"' in note
    # numbers once typed by hand, now read from results.json
    for stale in (
        "15.2944 turns",
        "[15.29], [15.11]",
        "0.0159, 0.2505",
        "16.564 thin 848",
        "34.46 k",
    ):
        assert stale not in note, stale
