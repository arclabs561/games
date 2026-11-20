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


def test_new_rules_in_the_note_and_readme():
    folder = Path(stopping.__file__).parent
    note = " ".join((folder / "peggle.typ").read_text().split())
    readme = (folder / "README.md").read_text()
    bf = {cap: stopping.outcome(HALF, cap, stopping.bayes_factor_rule) for cap in (100, 1000)}
    for cap, o in bf.items():
        assert f"{float(o.first + o.second):.4f} for $N = {cap}$" in note
        assert f"{float(o.first + o.second):.4f} (N = {cap:,})" in readme
    far = f"{stopping.verdict_chance_float(stopping.bayes_factor_bounds(10_000)):.4f}"
    assert f"to {far} at $N = 10000$" in note and f"{far} at N = 10,000" in readme
    assert (
        f"It takes {stopping.bounds_hi(100)} wins in 100 games, or {stopping.bounds_hi(1000)}"
        in note
    )
    assert f"$B_10 = 1024 slash 2310 approx {float(stopping.bayes_factor(6, 10)):.2f}$" in note
    assert f"$B_1000 approx {float(stopping.bayes_factor(527, 1000)):.2f}$" in note
    assert stopping.bayes_factor(8, 8) == Fraction(256, 9) and stopping.bounds_hi(8) == 8
    p55 = Fraction(55, 100)
    o = stopping.outcome(p55, 1000, stopping.bayes_factor_rule)
    assert f"{float(o.second):.1%}" == "0.4%" and f"{float(o.first):.0%}" == "51%"
    assert f"after {float(o.games):.0f} games on average" in note
    worse, games = stopping.lead_exact(p55, 15)
    assert f"= {float(worse):.4f}$, after {float(games):.1f} games" in note
    assert f"= {float(worse):.4f}, after {float(games):.1f} games" in readme
    assert f"lead of 14 would err {float(stopping.lead_exact(p55, 14)[0]):.1%}" in note
    # Every row of the table of rules.
    rows = [
        ("posterior 95%", 100, stopping.verdict),
        ("posterior 95%", 1000, stopping.verdict),
        ("Bayes factor 20", 100, stopping.bayes_factor_rule),
        ("Bayes factor 20", 1000, stopping.bayes_factor_rule),
        ("lead of 15", 1000, stopping.lead_rule(15)),
    ]
    for name, cap, rule in rows:
        equal, edge = stopping.outcome(HALF, cap, rule), stopping.outcome(p55, cap, rule)
        cells = [
            name,
            f"{cap:,}",
            f"{float(equal.first + equal.second):.3f}",
            f"{float(equal.games):.1f}",
            f"{float(edge.first):.3f}",
            f"{float(edge.second):.3f}",
            f"{float(edge.games):.1f}",
        ]
        assert "[" + "], [".join(cells) + "]" in note, cells
    cells = ["lead of 15", "none", "1", "225", f"{1 - float(worse):.3f}", f"{float(worse):.3f}"]
    assert "[" + "], [".join(cells) + f"], [{float(games):.1f}]" in note


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


def test_stricter_thresholds_in_the_note_and_readme():
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
    note = " ".join((Path(stopping.__file__).parent / "peggle.typ").read_text().split())
    readme = (Path(stopping.__file__).parent / "README.md").read_text()
    p55 = Fraction(55, 100)
    cells = []
    for cap, shown in stopping.STRICT.items():
        rule = stopping.posterior_rule(shown)
        equal, edge = stopping.outcome(HALF, cap, rule), stopping.outcome(p55, cap, rule)
        row = [
            f"posterior {shown.numerator / shown.denominator}",
            f"{cap:,}",
            f"{float(equal.first + equal.second):.3f}",
            f"{float(equal.games):.1f}",
            f"{float(edge.first):.3f}",
            f"{float(edge.second):.3f}",
            f"{float(edge.games):.1f}",
        ]
        assert "[" + "], [".join(row) + "]" in note, row
        cells.append((f"{float(edge.first):.3f}", f"{float(edge.games):.1f}"))
        assert f"{stopping.fewest_wins(cap, shown)} " in note
    (a, ga), (b, gb) = cells
    assert f"chance {a} and {b}, against 0.061 and 0.512" in note
    assert f"after {ga} and {gb} games on average" in note
    assert f"chance {a} (N = 100) or {b} (N = 1,000), after {ga} or {gb} games" in readme
