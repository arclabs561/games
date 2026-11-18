"""Peggle: playing until someone is 95% sure, by exact dynamic programming and by simulation."""

from fractions import Fraction
from itertools import product
from math import sqrt
from pathlib import Path

from peggle import stopping
from peggle.peggle import better

HALF = Fraction(1, 2)


def brute_force(p: Fraction, cap: int) -> stopping.Outcome:
    """Enumerate all 2^cap sequences of cap games and stop each one at its first verdict."""
    totals = {1: Fraction(0), -1: Fraction(0), 0: Fraction(0)}
    games = Fraction(0)
    for wins in product((True, False), repeat=cap):
        k = 0
        for n, won in enumerate(wins, 1):
            k += won
            v = stopping.verdict(k, n)
            if v or n == cap:
                break
        # The 2^(cap - n) sequences that share this prefix all stop here; count each prefix once.
        if all(wins[n:]):
            chance = stopping.path_chance(k, n, p)
            totals[v] += chance
            games += n * chance
    return stopping.Outcome(totals[1], totals[-1], totals[0], games)


def test_dynamic_program_matches_brute_force():
    for cap in range(1, 13):
        for p in (HALF, Fraction(3, 5), Fraction(1, 7)):
            assert stopping.outcome(p, cap) == brute_force(p, cap)


def test_chances_add_up_and_equal_players_are_symmetric():
    for cap in (1, 7, 40, 100):
        o = stopping.outcome(HALF, cap)
        assert o.first + o.second + o.neither == 1
        assert o.first == o.second
        q = stopping.outcome(Fraction(3, 5), cap)
        assert q.first + q.second + q.neither == 1


def test_thresholds_match_verdicts():
    bounds = stopping.thresholds(60)
    for n in range(1, 61):
        lo, hi = bounds[n]
        for k in range(n + 1):
            assert stopping.verdict(k, n) == (1 if k >= hi else -1 if k <= lo else 0)


def test_optional_stopping_declares_equal_players_unequal():
    for cap in (100, 1000):
        stopped = stopping.outcome(HALF, cap)
        one_look = stopping.fixed_length(HALF, cap)
        assert one_look.first + one_look.second < Fraction(1, 10)
        assert stopped.first + stopped.second > 5 * (one_look.first + one_look.second)
    assert float(stopping.outcome(HALF, 1000).first * 2) > 0.72


def test_the_uniform_prior_reading_is_calibrated():
    for cap in (10, 30):
        assert stopping.bayes(cap) == stopping.bayes(cap, by_integral=True)
    for cap in (100, 1000):
        b = stopping.bayes(cap)
        assert b.wrong_given_declared < 1 - stopping.SURE
        # Each verdict is wrong with chance at most 5%, so the average over verdicts is too.
        assert all(better(s.k, s.n) >= stopping.SURE for s in stopping.stops(cap) if s.verdict > 0)


def test_laplace_rule_gives_the_prior_path_chance():
    for n in range(9):
        for wins in product((True, False), repeat=n):
            assert stopping.laplace_path_chance(list(wins)) == stopping.prior_path_chance(
                sum(wins), n
            )


def test_simulation_agrees_with_the_exact_chances():
    cap, trials = 60, 40_000
    for seed, p in enumerate((0.5, 0.6, None)):
        sim = stopping.simulate(p, cap, trials, seed)
        if p is None:
            b = stopping.bayes(cap)
            pairs = [(sim["first"] + sim["second"], b.declared), (sim["wrong"], b.wrong)]
        else:
            o = stopping.outcome(Fraction(str(p)), cap)
            pairs = [(sim["first"], o.first), (sim["second"], o.second)]
        for got, exact in pairs:
            q = float(exact)
            assert abs(got - q) <= 4 * sqrt(q * (1 - q) / trials)


def test_readme_numbers_are_current():
    readme = (Path(stopping.__file__).parent / "README.md").read_text()
    o = stopping.outcome(HALF, 100)
    assert f"{float(o.first + o.second):.3f}" in readme
    assert f"{float(stopping.bayes(1000).wrong_given_declared):.1%}" in readme


# Armitage, McPherson and Rowe (1969), Table 1, columns 2 alpha = 0.01 and 0.05 (five decimals).
ARMITAGE_N = (10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 150)
ARMITAGE = {
    "0.01": (0.00781, 0.01538, 0.01840, 0.02266, 0.02746, 0.03087, 0.03406, 0.03687, 0.03931,
             0.04319, 0.04679, 0.05090, 0.05345, 0.05586, 0.06085, 0.06462, 0.06619),
    "0.05": (0.05469, 0.08191, 0.10662, 0.12140, 0.13355, 0.14328, 0.15351, 0.16399, 0.17117,
             0.18583, 0.19973, 0.20889, 0.21941, 0.22731, 0.24187, 0.25503, 0.26108),
}  # fmt: skip


def test_repeated_binomial_tests_match_armitage_mcpherson_rowe():
    off = []
    for two_alpha, published in ARMITAGE.items():
        rule = stopping.binomial_test(Fraction(two_alpha) / 2)
        for n, value in zip(ARMITAGE_N, published, strict=True):
            o = stopping.outcome(HALF, n, rule)
            error = abs(float(o.first + o.second) - value)
            assert error < 2e-5
            if error > 6e-6:
                off.append((two_alpha, n))
    # The other 32 entries agree to the five printed decimals.
    assert off == [("0.01", 30), ("0.05", 140)]


def test_stopped_posterior_averages_to_the_prior():
    for cap in (1, 10, 100):
        assert sum(
            s.paths * stopping.prior_path_chance(s.k, s.n) * better(s.k, s.n)
            for s in stopping.stops(cap)
        ) == Fraction(1, 2)


def test_note_numbers_are_current():
    note = (Path(stopping.__file__).parent / "peggle.typ").read_text()
    for p, cap in ((HALF, 100), (HALF, 1000), (Fraction(55, 100), 1000), (Fraction(7, 10), 1000)):
        o = stopping.outcome(p, cap)
        first = o.first + o.second if p == HALF else o.first
        assert f"[{float(first):.3f}]" in note
        assert f"[{float(o.games):.1f}]" in note
    for cap in (100, 1000):
        assert f"{float(stopping.bayes(cap).wrong_given_declared):.1%}" in note
