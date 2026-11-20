"""Peggle: playing until someone is 95% sure, by exact dynamic programming and by simulation."""

from fractions import Fraction
from itertools import product
from math import sqrt
from pathlib import Path

from peggle import stopping
from peggle.peggle import better

HALF = Fraction(1, 2)


def brute_force(p: Fraction, cap: int, rule: stopping.Rule = stopping.verdict) -> stopping.Outcome:
    """Enumerate all 2^cap sequences of cap games and stop each one at its first verdict."""
    totals = {1: Fraction(0), -1: Fraction(0), 0: Fraction(0)}
    games = Fraction(0)
    for wins in product((True, False), repeat=cap):
        k = 0
        for n, won in enumerate(wins, 1):
            k += won
            v = rule(k, n)
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


def test_other_rules_match_brute_force():
    for cap in (8, 10, 12):
        for p in (HALF, Fraction(3, 5)):
            for rule in (stopping.bayes_factor_rule, stopping.lead_rule(3)):
                assert stopping.outcome(p, cap, rule) == brute_force(p, cap, rule)


def test_bayes_factor_is_a_martingale_for_equal_players():
    for n in range(40):
        for k in range(n + 1):
            after = (stopping.bayes_factor(k + 1, n + 1) + stopping.bayes_factor(k, n + 1)) / 2
            assert after == stopping.bayes_factor(k, n)
            # It is the prior average of the likelihood ratio (2p)^k (2 - 2p)^(n - k).
            assert stopping.bayes_factor(k, n) == 2**n * stopping.prior_path_chance(k, n)


def test_bayes_factor_rule_and_ville():
    for cap in (1, 5, 9, 30, 200):
        # Optional stopping: the factor where play stops still averages exactly 1.
        assert (
            sum(
                s.paths * Fraction(1, 2**s.n) * stopping.bayes_factor(s.k, s.n)
                for s in stopping.stops(cap, stopping.bayes_factor_rule)
            )
            == 1
        )
    chances = []
    for cap in (100, 1000):
        o = stopping.outcome(HALF, cap, stopping.bayes_factor_rule)
        chances.append(o.first + o.second)
    # The chance only grows with the cap; Ville's inequality keeps it below 1/20 for every cap.
    assert [f"{float(c):.4f}" for c in chances] == ["0.0276", "0.0357"]
    assert chances[0] < chances[1] < Fraction(1, stopping.EVIDENCE)
    assert stopping.bayes_factor_bounds(1000) == stopping.thresholds(
        1000, stopping.bayes_factor_rule
    )
    far = stopping.verdict_chance_float(stopping.bayes_factor_bounds(1000))
    assert abs(far - float(chances[1])) < 1e-12
    far = stopping.verdict_chance_float(stopping.bayes_factor_bounds(10_000))
    assert f"{far:.4f}" == "0.0387" and far < 1 / stopping.EVIDENCE
    assert [stopping.bounds_hi(n) for n in (20, 100, 1000)] == [17, 66, 556]


def test_wald_test_is_a_lead_of_fifteen():
    assert stopping.wald_lead() == 15
    assert stopping.lead_exact(Fraction(55, 100), 14)[0] > Fraction(5, 100)
    worse, games = stopping.lead_exact(Fraction(55, 100), 15)
    assert worse == Fraction(9**15, 9**15 + 11**15)
    assert f"{float(worse):.4f}" == "0.0470" and f"{float(games):.1f}" == "135.9"
    assert stopping.lead_exact(HALF, 15) == (HALF, 225)
    for p in (HALF, Fraction(55, 100), Fraction(7, 10)):
        o = stopping.outcome(p, 1000, stopping.lead_rule(15))
        worse, games = stopping.lead_exact(p, 15)
        assert abs(o.second - worse) <= o.neither
        assert o.games <= games and (p != HALF or o.neither > Fraction(1, 1000))
        if p != HALF:
            # Undecided after 1,000 games: 4.3e-5 at p = 0.55, 1.2e-39 at p = 0.7.
            assert o.neither < Fraction(1, 10**4) and games - o.games < Fraction(1, 100)


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


def test_repeated_binomial_tests_match_armitage_mcpherson_rowe():
    off = []
    for two_alpha, n, published, computed in stopping.armitage_comparison():
        error = abs(computed - published)
        assert error < 2e-5
        if f"{computed:.5f}" != f"{published:.5f}":
            off.append((two_alpha, n))
    # The other 31 entries agree to the five printed decimals. At n = 150 the exact 0.2610858
    # rounds to 0.26109, one unit in the last place above the published 0.26108.
    assert off == [("0.01", 30), ("0.05", 140), ("0.05", 150)]


def test_stopped_posterior_averages_to_the_prior():
    for cap in (1, 10, 100):
        assert sum(
            s.paths * stopping.prior_path_chance(s.k, s.n) * better(s.k, s.n)
            for s in stopping.stops(cap)
        ) == Fraction(1, 2)


def test_new_rules():
    bf = {cap: stopping.outcome(HALF, cap, stopping.bayes_factor_rule) for cap in (100, 1000)}
    assert [f"{float(o.first + o.second):.4f}" for o in bf.values()] == ["0.0276", "0.0357"]
    assert stopping.bayes_factor(8, 8) == Fraction(256, 9) and stopping.bounds_hi(8) == 8
    p55 = Fraction(55, 100)
    o = stopping.outcome(p55, 1000, stopping.bayes_factor_rule)
    assert f"{float(o.second):.1%}" == "0.4%" and f"{float(o.first):.0%}" == "51%"
    worse, _ = stopping.lead_exact(p55, 15)
    assert f"{float(stopping.lead_exact(p55, 14)[0]):.1%}" == "5.7%" and worse < Fraction(1, 20)


def test_safe_threshold_matches_a_search_over_every_threshold():
    cap = 12
    posteriors = sorted({better(k, n) for n in range(1, cap + 1) for k in range(n + 1)})
    posteriors = [b for b in posteriors if b >= stopping.SURE]
    for alpha in (Fraction(1, 20), Fraction(1, 50)):
        chances = [brute_force(HALF, cap, stopping.posterior_rule(b)) for b in posteriors]
        safe = [o.first + o.second <= alpha for o in chances]
        i = safe.index(True)
        assert all(safe[i:])  # a stricter threshold is never less safe
        assert stopping.safe_threshold(cap, alpha) == (posteriors[i - 1], posteriors[i])


def test_stricter_thresholds():
    alpha = Fraction(5, 100)
    w, t = stopping.safe_threshold(100)
    assert w < stopping.STRICT[100] <= t
    lower = {100: Fraction("0.997037"), 1000: Fraction("0.9987002")}
    for cap, shown in stopping.STRICT.items():
        # The posterior just below the shown threshold, the largest one that is not safe.
        below = max(
            better(k - 1, n)
            for n in range(1, cap + 1)
            if 2 * (k := stopping.fewest_wins(n, shown)) - 2 > n
        )
        assert lower[cap] <= below < shown
        chance = stopping.equal_verdict_chance(cap, stopping.posterior_rule(shown))
        assert chance <= alpha < stopping.equal_verdict_chance(cap, stopping.posterior_rule(below))
        assert f"{float(chance):.4f}" == "0.0497"
    # the note types the two lower thresholds by hand; the rest it reads from results.json
    note = " ".join((Path(stopping.__file__).parent / "peggle.typ").read_text().split())
    assert "a threshold of 0.997037 or 0.9987002, a little lower" in note
