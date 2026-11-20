"""Write every number the README, its fragments in docs/ and the note quote to results.json.

    uv run python -m peggle.report     # about a minute; writes peggle/results.json

Exact chances are computed with fractions and stored as floats, rounded only where they are
printed; fractions the note prints as fractions are stored as [numerator, denominator]. Keys
name the rule, the chance p as "p50" for 1/2 or "p55" for 0.55 (a key cannot contain a dot,
because placeholders use dots as separators), and the cap N. `quick=True` leaves out the
simulation, for the tests.
"""

import json
from fractions import Fraction
from math import ceil, log, log10, sqrt
from pathlib import Path

from peggle import peggle, stopping, variations
from peggle.stopping import outcome

FOLDER = Path(__file__).resolve().parent
RESULTS = FOLDER / "results.json"
HALF = Fraction(1, 2)
P55 = Fraction(55, 100)
PS = {"p50": HALF, "p55": P55, "p60": Fraction(6, 10), "p70": Fraction(7, 10)}
CAPS = (100, 1000)
GAMES = (10, 20, 50, 100, 200, 1000)  # the columns of the tables of wins needed
SIM_TRIALS, SIM_CAP, SIM_SEED = 200_000, 100, 1
SLOW = {"simulation"}  # left out when quick


def frac(x: Fraction) -> list[int]:
    return [x.numerator, x.denominator]


def rule_outcomes(rule, ps=PS) -> dict:
    """For each p and cap: first, second and no one declared better, and the mean games."""
    out = {}
    for name, p in ps.items():
        for cap in CAPS:
            o = outcome(p, cap, rule)
            out.setdefault(name, {})[str(cap)] = [
                float(o.first),
                float(o.second),
                float(o.neither),
                float(o.games),
            ]
    return out


def coins() -> dict:
    bounds = {}
    for name, delta in (("d10", Fraction(1, 10)), ("d05", Fraction(1, 20))):
        chernoff = ceil((delta + 2) / delta**2 * log(2 / 0.05))
        hoeffding = ceil(log(2 / 0.05) / (2 * delta**2))
        bounds[name] = [chernoff, hoeffding]
    errors = {
        name: [peggle.largest_error(slope, lowest)[0] for lowest in (1, 30)]
        for name, slope in (("posterior", peggle.posterior_slope), ("fitted", peggle.fitted_slope))
    }
    return {
        "six_of_ten": frac(peggle.better(6, 10)),
        "six_of_ten_value": float(peggle.better(6, 10)),
        "bounds": bounds,
        "wins_needed": {
            str(n): [peggle.wins_needed(n), float(peggle.better(peggle.wins_needed(n), n))]
            for n in GAMES
        },
        "logistic_error": errors,  # largest |exact - logistic| for 1 <= n <= 300 and 30 <= n <= 300
    }


def priors() -> dict:
    a = variations.SKEPTIC
    m = a - 1  # the prior's own chance above 0.6, from its density q^m (1 - q)^m
    above = peggle.beta_integral(m, 2 * m, Fraction(3, 5)) / peggle.beta_integral(m, 2 * m)
    rules = {"jeffreys": variations.JEFFREYS_RULE, "skeptical": variations.beta_rule(a)}
    shortest = {
        "uniform": next(n for n in range(1, 20) if variations.fewest_wins_beta(n, 1) <= n),
        "jeffreys": next(n for n in range(1, 20) if variations.fewest_wins_jeffreys(n) <= n),
        "skeptical": next(n for n in range(1, 40) if variations.fewest_wins_beta(n, a) <= n),
    }
    wrong = {}
    for name, rule in (("uniform", stopping.verdict), ("skeptical", rules["skeptical"])):
        for cap in CAPS:
            declared, w = variations.prior_check(cap, rule, a)
            wrong.setdefault(name, {})[str(cap)] = float(w / declared)
    return {
        "six_of_ten": {
            "uniform": float(peggle.better(6, 10)),
            "jeffreys": variations.better_jeffreys(6, 10),
            "skeptical": float(variations.better_beta(6, 10, a)),
        },
        "jeffreys_part": frac(variations.jeffreys_part(6, 10)),  # the chance is 1/2 + this / pi
        "skeptic_above": float(above),
        "fewest": {
            "uniform": [variations.fewest_wins_beta(n, 1) for n in GAMES],
            "jeffreys": [variations.fewest_wins_jeffreys(n) for n in GAMES],
            "skeptical": [variations.fewest_wins_beta(n, a) for n in GAMES],
        },
        "shortest": shortest,
        # straight wins: the shortest verdict under the Jeffreys prior, in a puzzle's answer
        "jeffreys_straight": variations.better_jeffreys(shortest["jeffreys"], shortest["jeffreys"]),
        "outcomes": {
            name: rule_outcomes(rule, {"p50": HALF, "p55": P55}) for name, rule in rules.items()
        },
        "wrong_if_skeptical": wrong,  # p drawn from Beta(12, 12): share of verdicts that are wrong
    }


def strict() -> dict:
    out = {}
    for cap, shown in stopping.STRICT.items():
        rule = stopping.posterior_rule(shown)
        equal, edge = outcome(HALF, cap, rule), outcome(P55, cap, rule)
        out[str(cap)] = {
            "threshold": str(shown.numerator / shown.denominator),
            "equal": [float(equal.first + equal.second), float(equal.games)],
            "p55": [float(edge.first), float(edge.second), float(edge.games)],
            "wins_needed": stopping.fewest_wins(cap, shown),
        }
    return out


def formats() -> dict:
    out = {}
    for n in (1, 3, 5, 7):
        slope = variations.slope_at_half_best_of(n)
        out[f"best of {n}"] = [
            float(variations.best_of(P55, n)),
            float(variations.best_of_games(P55, n)),
            frac(slope),
        ]
    for c in (2, 15):
        worse, games = stopping.lead_exact(P55, c)
        out[f"win by {c}"] = [float(1 - worse), float(games), [c, 1]]
    nb = variations.fewest_best_of(P55)
    out["fewest_best_of"] = nb
    out["best of fewest"] = [
        float(variations.best_of(P55, nb)),
        float(variations.best_of_games(P55, nb)),
        float(variations.slope_at_half_best_of(nb)),
    ]
    out["win by 2 fraction"] = frac(1 - stopping.lead_exact(P55, 2)[0])
    return out


def elo() -> dict:
    rows = []
    for gap in (variations.elo_gap(0.55), 100, 200, 400):
        p = variations.elo_chance(gap)
        c = variations.lead_for_gap(gap)
        worse, games = variations.lead_float(p, c)
        rows.append([gap, p, c, worse, games, variations.fewest_best_of_float(p)])
    worse, games = stopping.lead_exact(Fraction(10, 11), 2)
    p561 = variations.elo_chance(400 * 400 / 561)
    w561, g561 = variations.lead_float(p561, 2)
    return {
        "rows": rows,
        "lead_constant": 400 * log10(19),
        "gap_55": variations.elo_gap(0.55),
        "gap_400": {"worse": frac(worse), "games": frac(games), "games_value": float(games)},
        "scale_561": [p561, w561, g561],
    }


def changing() -> dict:
    shooter = {}
    iid = stopping.fixed_length(HALF, 100)
    shooter["none"] = [
        float(iid.first + iid.second),
        *(float(outcome(HALF, cap).first * 2) for cap in CAPS),
    ]
    for edge, e in ((Fraction(3, 5), "e60"), (Fraction(7, 10), "e70")):
        for name, chance in (
            ("alternate", variations.alternate(edge)),
            ("winner", variations.winner_starts(edge)),
            ("loser", variations.loser_starts(edge)),
        ):
            look = variations.changing_outcome(
                100, variations.at_end(stopping.verdict, 100), chance, 10
            )
            stops = [variations.changing_outcome(cap, stopping.verdict, chance, 10) for cap in CAPS]
            shooter[f"{name} {e}"] = [
                float(look.first + look.second),
                *(float(o.first + o.second) for o in stops),
            ]
    drift = {}
    for before, after, name in (
        (Fraction(45, 100), P55, "45to55"),
        (Fraction(2, 5), Fraction(3, 5), "40to60"),
    ):
        chance = variations.step_change(before, after, 500)
        look = variations.changing_outcome(
            1000, variations.at_end(stopping.verdict, 1000), chance, 20
        )
        o = variations.changing_outcome(1000, stopping.verdict, chance, 20)
        drift[name] = {
            "one_look": float(look.first + look.second),
            "stop": [float(o.first), float(o.second), float(o.games)],
        }
    return {"shooter": shooter, "drift": drift}


def simulation() -> dict:
    """The seeded simulation that `python -m peggle.stopping` prints, and how many of the eight
    exact values fall inside its 95% intervals."""
    inside = 0
    for p in (HALF, P55, Fraction(7, 10), None):
        sim = stopping.simulate(None if p is None else float(p), SIM_CAP, SIM_TRIALS, SIM_SEED)
        if p is None:
            b = stopping.bayes(SIM_CAP)
            pairs = [(sim["first"] + sim["second"], b.declared), (sim["wrong"], b.wrong)]
        else:
            o = outcome(p, SIM_CAP)
            pairs = [(sim["first"], o.first), (sim["second"], o.second)]
        for got, exact in pairs:
            inside += abs(got - float(exact)) <= 1.96 * sqrt(got * (1 - got) / SIM_TRIALS)
    return {"trials": SIM_TRIALS, "cap": SIM_CAP, "seed": SIM_SEED, "inside": inside, "of": 8}


def results(quick: bool = False) -> dict:
    bf = stopping.bayes_factor_rule
    lead = stopping.wald_lead()
    worse, games = stopping.lead_exact(P55, lead)
    armitage = stopping.armitage_comparison()
    agree = sum(f"{c:.5f}" == f"{p:.5f}" for _, _, p, c in armitage)
    rules = {"posterior": stopping.verdict, "bayes factor": bf}
    out = {
        "coins": coins(),
        "equal_verdict": {  # equal players: the chance that someone is declared better
            name: {
                str(cap): float(o.first + o.second)
                for cap in CAPS
                for o in [outcome(HALF, cap, rule)]
            }
            for name, rule in rules.items()
        },
        "one_look": {
            str(cap): float(
                sum(getattr(stopping.fixed_length(HALF, cap), s) for s in ("first", "second"))
            )
            for cap in CAPS
        },
        "outcomes": {
            "posterior": rule_outcomes(stopping.verdict),
            "bayes factor": rule_outcomes(bf),
            "lead": rule_outcomes(stopping.lead_rule(lead)),
        },
        "uniform_prior": {  # p drawn from the uniform prior: verdict, wrong, wrong given a verdict
            str(cap): [
                float(b.declared),
                float(b.wrong),
                float(b.wrong_given_declared),
            ]
            for cap in CAPS
            for b in [stopping.bayes(cap)]
        },
        "calibration": [
            [float(declared), float(wrong)]
            for _, _, declared, wrong in variations.calibration(1000, (10, 100, 1000))
        ],
        "priors": priors(),
        "bayes_factor": {
            "six_of_ten": float(stopping.bayes_factor(6, 10)),
            "six_of_ten_fraction": [2**10, 11 * 210],  # 2^n and (n + 1) C(n, k), not reduced
            "at_527": float(stopping.bayes_factor(527, 1000)),
            "first_verdict": frac(stopping.bayes_factor(8, 8)),
            "first_verdict_value": float(stopping.bayes_factor(8, 8)),
            "wins_needed": {str(n): stopping.bounds_hi(n) for n in (20, 100, 1000)},
            "far": {"10000": stopping.verdict_chance_float(stopping.bayes_factor_bounds(10_000))},
        },
        "strict": strict(),
        "wald": {
            "lead": lead,
            "worse": float(worse),
            "games": float(games),
            "lead_14_worse": float(stopping.lead_exact(P55, lead - 1)[0]),
            "equal_games": int(stopping.lead_exact(HALF, lead)[1]),
        },
        "formats": formats(),
        "elo": elo(),
        "changing": changing(),
        "armitage": {
            "entries": len(armitage),
            "agree": agree,
            "off": len(armitage) - agree,
            "largest_difference": max(abs(c - p) for _, _, p, c in armitage),
        },
    }
    if not quick:
        out["simulation"] = simulation()
    return out


def dump(data: dict) -> str:
    """One line per result, so that a change shows up as a one-line diff."""
    lines = [f" {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}" for k, v in data.items()]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def main() -> None:
    RESULTS.write_text(dump(results()))
    print(f"wrote {RESULTS.relative_to(FOLDER.parent)}")


if __name__ == "__main__":
    main()
