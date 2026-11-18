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


def test_harmonic_approximation_and_its_fourier_terms():
    from decimal import Decimal

    from tenzi import maximum

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


def note_text() -> str:
    return " ".join((Path(tenzi.__file__).parent / "tenzi.typ").read_text().split())


def test_note_quotes_the_deadline_chances():
    from tenzi import optimal

    chances, _ = optimal.by_turn()
    note = note_text()
    quoted = ", ".join(f"{float(chances[t - 1]):.4f}" for t in (5, 10, 14, 20))
    assert f"{quoted} and {float(chances[29]):.4f}" in note
    assert f"up to turn {len(chances)}" in note


def sci(x, digits: int) -> str:
    """A number as the note writes it, such as $4.2 times 10^(-10)$."""
    mantissa, exponent = f"{float(x):.{digits}e}".split("e")
    return f"{mantissa} times 10^({int(exponent)})"


def test_note_quotes_the_harmonic_formula():
    from tenzi import maximum

    note = note_text()
    approx = f"{maximum.harmonic_approximation(10):.12f}"
    exact = f"{maximum.decimal(maximum.exact_mean(10)):.12f}"
    spaced = [f"{x[:6]} thin {x[6:9]} thin {x[9:12]} thin {x[12:]}" for x in (approx, exact)]
    assert f"{spaced[0]}, $" in note and f"${spaced[1]}$" in note
    gap = maximum.decimal(maximum.exact_mean(10)) - maximum.harmonic_approximation(10)
    assert f"differ by ${sci(gap, 2)}$" in note
    assert f"gives {maximum.harmonic_approximation(1):.3f} against the exact 6" in note
    gap20 = maximum.decimal(maximum.exact_mean(20)) - maximum.harmonic_approximation(20)
    assert f"off by ${sci(gap20, 1)}$" in note
    assert f"come to ${sci(maximum.term_size(10), 2)}$" in note
    assert f"come to ${sci(maximum.term_size(10, k=2), 1)}$" in note
    assert f"approx {sci(maximum.wobble_amplitude(), 2)}$" in note
    assert f"from {maximum.dice_for_wobble()} dice on" in note
    for n in (1, 2, 3, 5, 10, 20, 100):
        gap = maximum.decimal(maximum.exact_mean(n)) - maximum.harmonic_approximation(n)
        row = (
            f"[{n}], [{float(maximum.exact_mean(n)):.6f}], [${sci(gap, 1)}$],"
            f" [${sci(maximum.fourier_term(n, 6, 1), 1)}$],"
        )
        assert row in note, row


def test_readme_quotes_the_new_results():
    from tenzi import maximum, optimal

    readme = (Path(tenzi.__file__).parent / "README.md").read_text()
    gap = maximum.decimal(maximum.exact_mean(10)) - maximum.harmonic_approximation(10)
    assert f"within {float(gap):.2e}".replace("e-10", "·10⁻¹⁰") in readme
    assert f"give {float(maximum.fourier_term(10, 6, 1)):.3e}".replace("e-10", "·10⁻¹⁰") in readme
    assert f"amplitude {float(maximum.wobble_amplitude()):.2e}".replace("e-23", "·10⁻²³") in readme
    assert f"from {maximum.dice_for_wobble()} dice on" in readme
    assert f"up to turn {len(optimal.by_turn()[0])}" in readme
