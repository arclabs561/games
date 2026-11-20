"""Peggle variations: other priors, match formats, ratings, and chances that change."""

from fractions import Fraction
from itertools import product
from math import comb
from pathlib import Path

from peggle import stopping, variations
from peggle.peggle import beta_integral, better

HALF = Fraction(1, 2)
P55 = Fraction(55, 100)
FOLDER = Path(variations.__file__).parent


def note() -> str:
    return " ".join((FOLDER / "peggle.typ").read_text().split())


def test_beta_prior_coin_flips_equal_the_integral():
    for a in (1, 2, 3, 12):
        for n in range(1, 9):
            for k in range(n + 1):
                assert variations.better_beta(k, n, a) == variations.better_beta_by_integral(
                    k, n, a
                )
    assert all(
        variations.better_beta(k, n, 1) == better(k, n) for n in range(1, 30) for k in range(n + 1)
    )
    # Beta(12, 12) is a record of 11 wins and 11 losses on the uniform prior.
    assert variations.better_beta(6, 10, 12) == better(17, 32)


def test_beta_path_chance_is_the_prior_average():
    assert all(
        variations.beta_path_chance(k, n, 1) == stopping.prior_path_chance(k, n)
        for n in range(8)
        for k in range(n + 1)
    )
    # Summed over all sequences of n games it is 1.
    for a in (2, 12):
        for n in (1, 5, 9):
            assert sum(comb(n, k) * variations.beta_path_chance(k, n, a) for k in range(n + 1)) == 1


def test_jeffreys_exact_form_matches_quadrature():
    for n in range(1, 15):
        for k in range(n + 1):
            exact = variations.better_jeffreys(k, n)
            assert abs(exact - variations.better_jeffreys_by_quadrature(k, n)) < 1e-12
            assert abs(exact - variations.jeffreys_float(k, n)) < 1e-12
            assert variations.jeffreys_part(k, n) == -variations.jeffreys_part(n - k, n)
    assert variations.jeffreys_part(6, 10) == Fraction(512, 693)


def test_fewest_wins_under_each_prior():
    for n in range(1, 60):
        assert variations.fewest_wins_beta(n, 1) == next(
            (k for k in range(n + 1) if better(k, n) >= stopping.SURE), n + 1
        )
        assert variations.fewest_wins_jeffreys(n) == next(
            (k for k in range(n + 1) if variations.better_jeffreys(k, n) >= 0.95), n + 1
        )
    rows = [
        (variations.fewest_wins_beta(n, 1), variations.fewest_wins_jeffreys(n))
        for n in (10, 20, 50, 100, 200, 1000)
    ]
    assert rows == [(8, 8), (14, 14), (31, 31), (59, 59), (112, 112), (527, 527)]
    skeptic = [variations.fewest_wins_beta(n, 12) for n in (10, 20, 50, 100, 200, 1000)]
    assert skeptic == [10, 16, 33, 60, 113, 527]


def test_prior_rules_match_brute_force():
    # Each of the 2^cap sequences of cap games counts once, stopped at its first verdict.
    for rule in (variations.JEFFREYS_RULE, variations.beta_rule(2)):
        for cap in (4, 8, 10):
            exact = stopping.equal_verdict_chance(cap, rule)
            count = Fraction(0)
            for wins in product((1, 0), repeat=cap):
                k = 0
                for n, won in enumerate(wins, 1):
                    k += won
                    if rule(k, n):
                        count += Fraction(1, 2**cap)
                        break
            assert count == exact


def test_calibration_shares_are_below_five_percent():
    rows = variations.calibration(200, (10, 100, 200))
    assert all(wrong < Fraction(5, 100) for _, _, _, wrong in rows)
    total = sum(declared for _, _, declared, _ in rows)
    assert total == stopping.bayes(200).declared


def test_lead_chain_agrees_with_the_gamblers_ruin():
    for c in (1, 2, 3, 7, 15):
        for p in (HALF, P55, Fraction(3, 5), Fraction(10, 11), Fraction(1, 3)):
            h, m = variations.lead_chain(p, c)
            worse, games = stopping.lead_exact(p, c)
            assert m == games
            assert (1 - h if p > HALF else h) == worse or p == HALF and h == HALF


def test_lead_rule_with_a_cap_approaches_the_closed_form():
    worse, games = stopping.lead_exact(P55, 15)
    for cap in (200, 1000):
        o = stopping.outcome(P55, cap, stopping.lead_rule(15))
        # Play beyond the cap can only add to each side.
        assert o.second <= worse <= o.second + o.neither
        assert o.games <= games


def test_best_of_matches_the_dynamic_program():
    for m in range(1, 8):
        n = 2 * m - 1
        for p in (HALF, P55, Fraction(2, 3)):
            o = stopping.outcome(p, n, variations.first_to(m))
            assert o.first == variations.best_of(p, n)
            assert o.games == variations.best_of_games(p, n)
            assert o.neither == 0
        h = Fraction(1, 10**9)
        slope = (variations.best_of(HALF + h, n) - variations.best_of(HALF - h, n)) / (2 * h)
        assert abs(slope - variations.slope_at_half_best_of(n)) < Fraction(1, 10**12)


def test_match_format_numbers():
    assert f"{float(variations.best_of(P55, 7)):.4f}" == "0.6083"
    assert 1 - stopping.lead_exact(P55, 2)[0] == Fraction(121, 202)
    n = variations.fewest_best_of(P55)
    assert n == 269
    assert variations.best_of(P55, 267) < Fraction(95, 100) <= variations.best_of(P55, 269)


def test_elo():
    assert abs(variations.elo_chance(400) - 10 / 11) < 1e-15
    assert abs(variations.elo_gap(0.55) - 34.86) < 0.01
    assert [variations.lead_for_gap(g) for g in (variations.elo_gap(0.55), 100, 200, 400)] == [
        15,
        6,
        3,
        2,
    ]
    assert stopping.lead_exact(Fraction(10, 11), 2) == (Fraction(1, 101), Fraction(242, 101))
    # The float lead agrees with the exact one where p is a fraction.
    worse, games = variations.lead_float(0.55, 15)
    exact = stopping.lead_exact(P55, 15)
    assert abs(worse - float(exact[0])) < 1e-12 and abs(games - float(exact[1])) < 1e-9
    assert variations.fewest_best_of_float(0.55) == variations.fewest_best_of(P55)


def brute_changing(cap: int, rule: stopping.Rule, chance) -> stopping.Outcome:
    """Every sequence of cap games, each stopped at its first verdict."""
    totals = {1: Fraction(0), -1: Fraction(0), 0: Fraction(0)}
    games = Fraction(0)
    seen = set()
    for wins in product((1, 0), repeat=cap):
        k, last, weight = 0, 0, Fraction(1)
        for n, won in enumerate(wins, 1):
            c = chance(n, last)
            weight *= c if won else 1 - c
            k += won
            last = 1 if won else -1
            v = rule(k, n)
            if v or n == cap:
                break
        if wins[:n] in seen:
            continue
        seen.add(wins[:n])
        totals[v] += weight
        games += n * weight
    return stopping.Outcome(totals[1], totals[-1], totals[0], games)


def test_changing_chances_match_brute_force():
    edge = Fraction(3, 5)
    for chance in (
        variations.alternate(edge),
        variations.winner_starts(edge),
        variations.loser_starts(edge),
        variations.step_change(Fraction(2, 5), Fraction(3, 5), 4),
    ):
        for cap in (1, 5, 10):
            for rule in (stopping.verdict, variations.at_end(stopping.verdict, cap)):
                exact = variations.changing_outcome(cap, rule, chance, 5)
                assert exact == brute_changing(cap, rule, chance)
    for cap in (10, 100):
        assert variations.changing_outcome(
            cap, stopping.verdict, variations.constant(P55), 20
        ) == stopping.outcome(P55, cap)


def test_hoeffding_one_look_is_at_most_the_iid_chance():
    """Hoeffding (1956): independent games with average chance 1/2 fall in a central interval
    at least as often as fair coin flips, so one look declares someone better no more often."""
    for cap in (10, 50, 100):
        iid = stopping.fixed_length(HALF, cap)
        for chance in (
            variations.alternate(Fraction(3, 5)),
            variations.alternate(Fraction(7, 10)),
            variations.step_change(Fraction(2, 5), Fraction(3, 5), cap // 2),
        ):
            o = variations.changing_outcome(
                cap, variations.at_end(stopping.verdict, cap), chance, 10
            )
            assert o.first + o.second <= iid.first + iid.second


def test_committed_csvs_are_current(tmp_path, monkeypatch):
    monkeypatch.setattr(variations, "HERE", tmp_path)
    variations.write_csvs()
    for name in ("priors.csv", "walk.csv", "formats.csv"):
        assert (tmp_path / name).read_text() == (FOLDER / name).read_text(), name


def test_variation_numbers_in_the_note():
    text = note()
    r = variations.jeffreys_part(6, 10)
    assert f"$1 slash 2 + {r.numerator} slash ({r.denominator} pi)$" in text
    assert f"{variations.better_jeffreys(6, 10):.3f}" in text
    assert f"{float(variations.better_beta(6, 10, 12)):.3f}" in text
    for cap in (100, 1000):
        for rule in (variations.JEFFREYS_RULE, variations.beta_rule(12)):
            e = stopping.outcome(HALF, cap, rule)
            assert f"[{float(e.first + e.second):.3f}]" in text
        declared, wrong = variations.prior_check(cap, stopping.verdict, 12)
        assert f"{float(wrong / declared):.1%}" in text
    worse2 = 1 - stopping.lead_exact(P55, 2)[0]
    assert f"{float(worse2):.3f}" in text
    assert f"{float(variations.best_of(P55, 7)):.3f}" in text
    assert "best of 269" in text or "best-of-269" in text


def cells(*values: str) -> str:
    return "[" + "], [".join(values) + "]"


def test_tables_in_the_note():
    text = note()
    # Which prior? The fewest wins under each prior.
    for name, fewest in (
        ("uniform", lambda n: variations.fewest_wins_beta(n, 1)),
        ("Jeffreys", variations.fewest_wins_jeffreys),
        ("skeptical", lambda n: variations.fewest_wins_beta(n, 12)),
    ):
        row = cells(*(str(fewest(n)) for n in (10, 20, 50, 100, 200, 1000)))
        assert row in text, (name, row)
    above = beta_integral(11, 22, Fraction(3, 5)) / beta_integral(11, 22)
    assert f"Beta(12, 12) puts chance {float(above):.2f} above 0.6" in text
    # Calibration by when the verdict came.
    for (_, _, declared, wrong), label in zip(
        variations.calibration(1000, (10, 100, 1000)),
        ("1 to 10", "11 to 100", "101 to 1,000"),
        strict=True,
    ):
        assert cells(label, f"{float(declared):.3f}", f"{float(wrong):.1%}") in text
    # Priors under stopping, every row.
    for name, rule in (
        ("Jeffreys", variations.JEFFREYS_RULE),
        ("text(fill: purple)[skeptical", variations.beta_rule(12)),
    ):
        for cap in (100, 1000):
            e, g = stopping.outcome(HALF, cap, rule), stopping.outcome(P55, cap, rule)
            row = cells(
                f"{cap:,}",
                f"{float(e.first + e.second):.3f}",
                f"{float(e.games):.1f}",
                f"{float(g.first):.3f}",
                f"{float(g.second):.3f}",
                f"{float(g.games):.1f}",
            )
            assert row in text, (name, row)
    # Match formats.
    for n, label in ((1, "one game"), (3, "best of 3"), (5, "best of 5"), (7, "best of 7")):
        slope = variations.slope_at_half_best_of(n)
        row = cells(
            label,
            f"{float(variations.best_of(P55, n)):.3f}",
            f"{float(variations.best_of_games(P55, n)):.2f}",
            str(slope),
        )
        assert row in text, row
    worse, games = stopping.lead_exact(P55, 15)
    assert cells(f"{1 - float(worse):.3f}", f"{float(games):.2f}", "15") in text
    nb = variations.fewest_best_of(P55)
    row = cells(
        f"best of {nb}",
        f"{float(variations.best_of(P55, nb)):.3f}",
        f"{float(variations.best_of_games(P55, nb)):.2f}",
        f"{float(variations.slope_at_half_best_of(nb)):.1f}",
    )
    assert row in text, row
    # Ratings.
    for gap in (variations.elo_gap(0.55), 100, 200, 400):
        p = variations.elo_chance(gap)
        c = variations.lead_for_gap(gap)
        worse, games = variations.lead_float(p, c)
        row = cells(
            f"{p:.3f}",
            str(c),
            f"{worse:.3f}",
            f"{games:.1f}",
            str(variations.fewest_best_of_float(p)),
        )
        assert row in text, row
    p = variations.elo_chance(400 * 400 / 561)
    worse, games = variations.lead_float(p, 2)
    assert f"$p approx {p:.3f}$" in text
    assert f"erring {worse:.1%} of the time in {games:.2f} games" in text


def test_shooter_and_drift_numbers_in_the_note():
    text = note()
    for edge in (Fraction(3, 5), Fraction(7, 10)):
        for name, chance in (
            ("alternate", variations.alternate(edge)),
            ("winner", variations.winner_starts(edge)),
            ("loser", variations.loser_starts(edge)),
        ):
            look = variations.changing_outcome(
                100, variations.at_end(stopping.verdict, 100), chance, 10
            )
            stops = [
                variations.changing_outcome(cap, stopping.verdict, chance, 10)
                for cap in (100, 1000)
            ]
            row = cells(
                name,
                f"{float(edge)}",
                f"{float(look.first + look.second):.4f}",
                *(f"{float(o.first + o.second):.4f}" for o in stops),
            )
            assert row in text, row
    for before, after in ((Fraction(45, 100), P55), (Fraction(2, 5), Fraction(3, 5))):
        chance = variations.step_change(before, after, 500)
        o = variations.changing_outcome(1000, stopping.verdict, chance, 20)
        assert f"{float(o.first):.3f}" in text and f"{float(o.second):.3f}" in text
        assert f"{float(o.games):.1f} games" in text
    look = variations.changing_outcome(
        1000,
        variations.at_end(stopping.verdict, 1000),
        variations.step_change(Fraction(45, 100), P55, 500),
        20,
    )
    assert f"with chance {float(look.first + look.second):.3f}, below the 0.094" in text
