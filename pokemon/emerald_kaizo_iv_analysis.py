#!/usr/bin/env python3
"""Exact and simulated catch-count analysis for Emerald Kaizo IV targets.

The model is deliberately conditional on a grass encounter producing the
target species. It does not model encounter-table species odds or the chance
that a throw fails to catch the Pokemon.

/// Assumptions
- Each of the six Gen III IVs is independent and uniform on 0..31.
- Emerald Kaizo does not document a grass-specific IV modifier, so grass and
  fishing encounters use the same IV distribution in this model.
- A "catch" means one successfully obtained Pokemon candidate.

The report is written next to this script by default.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from pathlib import Path
from typing import TypedDict

N_IVS = 6
IV_MAX = 31
IV_VALUES = IV_MAX + 1
TOTAL_VECTORS = IV_VALUES**N_IVS
NATURE_COUNT = 25
DEFAULT_SIMULATIONS = 1_000_000
DEFAULT_SEED = 20260817


class AllIVRow(TypedDict):
    criterion: str
    per_iv_tail: float
    p: float
    expected: float
    n90: int
    n95: int
    n99: int


class SumRow(TypedDict):
    percentile: float
    threshold: int
    p: float
    expected: float
    n90: int
    n95: int
    n99: int


class PracticalRow(TypedDict):
    criterion: str
    p: float
    expected: float
    n95: int


def all_iv_sum_distribution() -> Counter[int]:
    """Return the exact count of IV vectors for each six-IV sum."""
    distribution: Counter[int] = Counter({0: 1})
    for _ in range(N_IVS):
        next_distribution: Counter[int] = Counter()
        for partial_sum, count in distribution.items():
            for iv in range(IV_VALUES):
                next_distribution[partial_sum + iv] += count
        distribution = next_distribution
    return distribution


def percentile_threshold(distribution: Counter[int], percentile: float, total: int) -> int:
    """Return the smallest discrete value whose CDF reaches percentile."""
    target = percentile * total
    cumulative = 0
    for value in sorted(distribution):
        cumulative += distribution[value]
        if cumulative >= target:
            return value
    raise ValueError("percentile must be between 0 and 1")


def probability_all_at_least(minimum_iv: int) -> float:
    """Probability that every IV is at least minimum_iv."""
    if not 0 <= minimum_iv <= IV_MAX:
        raise ValueError("minimum IV must be between 0 and 31")
    return ((IV_MAX - minimum_iv + 1) / IV_VALUES) ** N_IVS


def probability_sum_at_least(distribution: Counter[int], minimum_sum: int, total: int) -> float:
    """Probability that the six-IV sum is at least minimum_sum."""
    return sum(count for value, count in distribution.items() if value >= minimum_sum) / total


def probability_nature_set(number_of_acceptable_natures: int) -> float:
    """Probability of landing in a selected set of Gen III natures."""
    if not 0 <= number_of_acceptable_natures <= NATURE_COUNT:
        raise ValueError("acceptable nature count must be between 0 and 25")
    return number_of_acceptable_natures / NATURE_COUNT


def probability_relevant_ivs_at_least(minimum_iv: int, relevant_stats: int) -> float:
    """Probability that selected IVs clear a threshold."""
    if not 0 <= relevant_stats <= N_IVS:
        raise ValueError("relevant stat count must be between 0 and 6")
    return ((IV_MAX - minimum_iv + 1) / IV_VALUES) ** relevant_stats


def practical_rows() -> list[PracticalRow]:
    """Return illustrative role-aware filters, not universal team advice."""
    rows: list[PracticalRow] = []
    criteria = (
        ("one chosen nature", probability_nature_set(1)),
        ("either of 2 role-appropriate natures", probability_nature_set(2)),
        ("2 role-relevant IVs >= 16", probability_relevant_ivs_at_least(16, 2)),
        (
            "2 relevant IVs >= 16 + either of 2 natures",
            probability_relevant_ivs_at_least(16, 2) * probability_nature_set(2),
        ),
        (
            "previous row + 1 preferred ability of 2",
            probability_relevant_ivs_at_least(16, 2) * probability_nature_set(2) * 0.5,
        ),
        ("all 6 IVs >= 16", probability_all_at_least(16)),
    )
    for criterion, p in criteria:
        rows.append(
            {
                "criterion": criterion,
                "p": p,
                "expected": expected_catches(p),
                "n95": catches_for_confidence(p, 0.95),
            }
        )
    return rows


def expected_catches(success_probability: float) -> float:
    """Expected number of independent candidates before the first success."""
    return 1.0 / success_probability if success_probability else math.inf


def catches_for_confidence(success_probability: float, confidence: float) -> int:
    """Smallest n with P(at least one success by n) >= confidence."""
    if not 0 < success_probability <= 1:
        raise ValueError("success probability must be in (0, 1]")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if success_probability == 1:
        return 1
    return math.ceil(math.log1p(-confidence) / math.log1p(-success_probability))


def success_by_n(success_probability: float, n: int) -> float:
    """Probability of at least one success in n independent candidates."""
    return 1.0 - (1.0 - success_probability) ** n


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction used by the regularized beta function."""
    max_iterations = 10_000
    epsilon = 3.0e-14
    tiny = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = max(abs(d), tiny) if d >= 0 else min(d, -tiny)
    d = 1.0 / d
    h = d
    for iteration in range(1, max_iterations + 1):
        m = float(iteration)
        m2 = 2.0 * m
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        d = max(abs(d), tiny) if d >= 0 else min(d, -tiny)
        c = 1.0 + numerator / c
        c = max(abs(c), tiny) if c >= 0 else min(c, -tiny)
        d = 1.0 / d
        h *= d * c
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        d = max(abs(d), tiny) if d >= 0 else min(d, -tiny)
        c = 1.0 + numerator / c
        c = max(abs(c), tiny) if c >= 0 else min(c, -tiny)
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= epsilon:
            return h
    raise ArithmeticError("beta continued fraction did not converge")


def regularized_beta(x: float, a: float, b: float) -> float:
    """Evaluate the regularized incomplete beta function."""
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must be between 0 and 1")
    if a <= 0.0 or b <= 0.0:
        raise ValueError("beta parameters must be positive")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    log_beta_term = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    beta_term = math.exp(log_beta_term)
    if x < (a + 1.0) / (a + b + 2.0):
        return beta_term * _beta_continued_fraction(a, b, x) / a
    return 1.0 - beta_term * _beta_continued_fraction(b, a, 1.0 - x) / b


def beta_quantile(probability: float, a: float, b: float) -> float:
    """Invert the regularized beta CDF by bisection."""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be between 0 and 1")
    lower = 0.0
    upper = 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if regularized_beta(midpoint, a, b) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def clopper_pearson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Return the two-sided exact binomial confidence interval."""
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("successes must be between 0 and trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else beta_quantile(alpha / 2.0, successes, trials - successes + 1)
    upper = (
        1.0
        if successes == trials
        else beta_quantile(1.0 - alpha / 2.0, successes + 1, trials - successes)
    )
    return lower, upper


def hoeffding_epsilon(trials: int, delta: float, two_sided: bool = True) -> float:
    """Additive Hoeffding error radius for a Bernoulli estimate."""
    multiplier = 2.0 / delta if two_sided else 1.0 / delta
    return math.sqrt(math.log(multiplier) / (2.0 * trials))


def simulate_all_at_least_count(minimum_iv: int, trials: int, seed: int) -> int:
    """Count all-IV successes in a seeded independent-IV simulation."""
    rng = random.Random(seed)
    successes = 0
    for _ in range(trials):
        ivs = [rng.randrange(IV_VALUES) for _ in range(N_IVS)]
        if all(iv >= minimum_iv for iv in ivs):
            successes += 1
    return successes


def simulate_all_at_least(minimum_iv: int, trials: int, seed: int) -> float:
    """Estimate the all-IV threshold probability with a seeded simulation."""
    return simulate_all_at_least_count(minimum_iv, trials, seed) / trials


def format_probability(probability: float) -> str:
    if probability == 0:
        return "0"
    return f"{probability:.8g} ({probability:.4%})"


def format_count(count: int) -> str:
    return f"{count:,}"


def all_iv_rows() -> list[AllIVRow]:
    rows: list[AllIVRow] = []
    for minimum_iv in (8, 16, 24, 28, 30, 31):
        p = probability_all_at_least(minimum_iv)
        rows.append(
            {
                "criterion": f"all 6 IVs >= {minimum_iv}",
                "per_iv_tail": (IV_MAX - minimum_iv + 1) / IV_VALUES,
                "p": p,
                "expected": expected_catches(p),
                "n90": catches_for_confidence(p, 0.90),
                "n95": catches_for_confidence(p, 0.95),
                "n99": catches_for_confidence(p, 0.99),
            }
        )
    return rows


def sum_rows(distribution: Counter[int]) -> list[SumRow]:
    rows: list[SumRow] = []
    for percentile in (0.50, 0.75, 0.90, 0.95, 0.99):
        threshold = percentile_threshold(distribution, percentile, TOTAL_VECTORS)
        p = probability_sum_at_least(distribution, threshold, TOTAL_VECTORS)
        rows.append(
            {
                "percentile": percentile,
                "threshold": threshold,
                "p": p,
                "expected": expected_catches(p),
                "n90": catches_for_confidence(p, 0.90),
                "n95": catches_for_confidence(p, 0.95),
                "n99": catches_for_confidence(p, 0.99),
            }
        )
    return rows


def render_report(trials: int, seed: int) -> str:
    distribution = all_iv_sum_distribution()
    all_rows = all_iv_rows()
    sum_rows_data = sum_rows(distribution)
    practical_rows_data = practical_rows()

    lines = [
        "# Emerald Kaizo IV Catch Analysis",
        "",
        "Generated by `emerald_kaizo_iv_analysis.py`.",
        "",
        "## Model",
        "",
        f"- Six IVs, each independently uniform on 0..31: `{TOTAL_VECTORS:,}` equally likely vectors.",
        "- Grass encounter-table odds are not included. This is conditional on obtaining the target species.",
        "- A catch means one successfully obtained Pokemon candidate; failed throws are not modeled.",
        "- No grass-specific IV modifier is documented for Emerald Kaizo, so this uses the Gen III marginal wild-IV model.",
        "- Probabilities and catch counts are exact under the independent uniform marginal model. Simulation is only a sanity check.",
        "",
        "For success probability `p`, the formulas are:",
        "",
        "- Expected catches: `E[N] = 1 / p`.",
        "- Confidence after `n` catches: `P(success by n) = 1 - (1 - p)^n`.",
        "- Catches for confidence `1 - alpha`: `ceil(log(alpha) / log(1 - p))`.",
        "",
        "## Practical Filter Examples",
        "",
        "A useful acceptance rule should reflect the Pokemon's role. The following examples assume a two-ability species, one preferred ability, two relevant stats such as Attack and Speed, and no encounter-slot or catch-failure cost.",
        "These are exact under the independent marginal model, but they are examples rather than universal recommendations.",
        "",
        "| Acceptance rule | Probability | Expected candidates | 95% confidence |",
        "|---|---:|---:|---:|",
    ]
    for row in practical_rows_data:
        lines.append(
            f"| {row['criterion']} | {format_probability(row['p'])} | "
            f"{row['expected']:,.2f} | {format_count(row['n95'])} |"
        )

    lines.extend(
        [
            "",
            "The all-six-IV rule is therefore not automatically sensible: it can be easier than a role-aware rule that also requires nature and ability, while still selecting a Pokemon with the wrong battle profile.",
            "",
            "## Nature, Ability, And Stat Relevance",
            "",
            "- There are 25 natures. One exact nature has marginal probability `1/25 = 4%`; two acceptable natures have probability `2/25 = 8%`.",
            "- A non-neutral nature raises one non-HP stat by 10% and lowers another by 10%. With EV gains disabled, this can matter more than several IV points in the relevant stat.",
            "- For a species with two normal abilities, one desired ability has marginal probability `1/2`. Kaizo changes many ability assignments, so this must be checked per species.",
            "- In Emerald, a lead Pokemon with Synchronize gives a 50% chance of forcing its nature on a wild encounter. For one desired nature, the marginal chance becomes `0.5 + 0.5/25 = 52%`, before accounting for RNG-frame effects.",
            "- Hidden Power is a special case: in Gen III its type and power depend on IV bit patterns, so total IV sum is not a valid proxy when Hidden Power is part of the target role.",
            "- Base stats, level, move category, ability, nature, and battle thresholds should replace generic IV percentiles when the question is whether the Pokemon can survive or secure a KO.",
            "",
        ]
    )
    lines.extend(
        [
            "## Which Bound Is Appropriate",
            "",
            "Under the independent-candidate model, the number of candidates until the first success is geometric, so its tail is known exactly: `P(N > n) = (1 - p)^n`. A concentration inequality is not needed for that model-based catch-count result.",
            f"For the `{trials:,}`-trial Monte Carlo check, a two-sided 95% Hoeffding radius is `+/- {hoeffding_epsilon(trials, 0.05):.6f}` and a one-sided 95% lower radius is `{hoeffding_epsilon(trials, 0.05, two_sided=False):.6f}`.",
            "Hoeffding is valid but additive and conservative. It becomes unhelpful for rare targets: its radius can be much larger than a probability such as 1/4096.",
            "For an empirical probability, Clopper-Pearson is the better finite-sample binomial interval. If `k` successes are observed in `m` simulations, use the one-sided lower bound `BetaInverse(delta; k, m-k+1)` and then substitute that lower bound into the geometric formula. For rare events, exact enumeration or a targeted RNG analysis is better than brute-force Monte Carlo; a zero-hit simulation cannot establish that the true probability is zero.",
            "If the exact model probability is only a lower bound `p_lower`, substitute it into the geometric formula. That produces a conservative catch count for a chosen operational confidence.",
            "",
            "## Grass, Fishing, And RNG",
            "",
            "The base Emerald source creates wild land and fishing Pokemon with random IVs, but the RNG is a deterministic 32-bit linear-congruential generator. Wild Pokemon are usually generated by the Gen 3 Method 2 timing pattern.",
            "Entering a grass tile consumes a different number of RNG calls from moving within grass. NPC movement, animations, the lead Pokemon, and Synchronize can also change the frame or spread. Therefore repeated soft resets at the same timing can reproduce the same IVs; successive ordinary encounters are not mathematically independent, although the independent model is a practical marginal approximation when the exact frame sequence is not being modeled.",
            "Fishing changes the species table and has its own input timing. It does not improve the IV distribution. If a target species occupies fraction `w` of a grass table, replace `p` by `w * p` when counting all grass encounters rather than target-species catches. The expected count is multiplied by `1 / w`.",
            "",
            "## Kaizo-Specific Scope",
            "",
            "The ROMHacking.net listing identifies the published Emerald Kaizo release as v2.1 and documents changed encounter tables, trainer data, map obstacles, and battle restrictions. The creator's thread says EV gains are disabled and warns that EV grinding is unnecessary, but neither source documents a grass-specific IV distribution change.",
            "The local Downloads ROM has an unresolved build identity and does not match the public 2023 1.1 patch output. This report therefore uses the intended marginal Gen 3 IV model rather than claiming binary-level verification for that file. A different Kaizo fork, an unofficial add-on, or an RNG-manipulation workflow needs a separate model.",
            "",
            "## All Six IVs Above a Threshold",
            "",
            "This is the strictest interpretation of 'IVs above a percentile': every stat must clear the threshold.",
            "`Per-IV tail` is the chance that one IV clears the threshold before requiring all six to do so.",
            "",
            "| Criterion | Per-IV tail | Success probability | Expected | 90% | 95% | 99% |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in all_rows:
        lines.append(
            f"| {row['criterion']} | {row['per_iv_tail']:.3%} | {format_probability(row['p'])} | "
            f"{row['expected']:,.2f} | {format_count(row['n90'])} | {format_count(row['n95'])} | "
            f"{format_count(row['n99'])} |"
        )

    lines.extend(
        [
            "",
            "## Total IV Sum Percentiles",
            "",
            "This is a less brittle criterion: accept a Pokemon when the sum of its six IVs is at least the discrete percentile threshold.",
            "Because the sum is discrete, the actual acceptance probability can be slightly above the nominal tail.",
            "",
            "| Sum percentile | Minimum sum | Success probability | Expected | 90% | 95% | 99% |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sum_rows_data:
        lines.append(
            f"| {row['percentile']:.0%} | {row['threshold']} | {format_probability(row['p'])} | "
            f"{row['expected']:,.2f} | {format_count(row['n90'])} | {format_count(row['n95'])} | "
            f"{format_count(row['n99'])} |"
        )

    lines.extend(
        [
            "",
            "## Simulation Check",
            "",
            f"Seed: `{seed}`. Candidates per estimate: `{trials:,}`.",
            "The interval is the two-sided 95% Clopper-Pearson interval for the independent simulation, not for the deterministic in-game RNG sequence.",
            "",
            "| Criterion | Model p | Simulated p | 95% exact interval | Absolute error |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in all_rows:
        simulated_count = simulate_all_at_least_count(
            int(str(row["criterion"]).rsplit(" ", 1)[-1]), trials, seed
        )
        simulated = simulated_count / trials
        lower, upper = clopper_pearson_interval(simulated_count, trials)
        lines.append(
            f"| {row['criterion']} | {row['p']:.8g} | {simulated:.8g} | "
            f"[{lower:.8g}, {upper:.8g}] | {abs(simulated - float(row['p'])):.8g} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Fishing does not improve IV quality; it changes the species encounter table.",
            "- The `all 6 IVs` table is useful for aggressive IV fishing, but it rejects many perfectly usable Pokemon.",
            "- For a first playthrough, a good nature, ability, typing, and role usually dominate small IV differences.",
            "- In a Nuzlocke, these numbers apply only if the ruleset allows repeated encounters. Standard first-encounter rules do not.",
            "- This analysis does not model species-slot odds, map encounter rates, catch failures, Synchronize, or frame-specific RNG manipulation.",
            "",
            "## Sources",
            "",
            "- [Bulbapedia: Individual values](https://bulbapedia.bulbagarden.net/wiki/Individual_values)",
            "- [ROMHacking.net: Pokemon Kaizo Emerald v2.1](https://www.romhacking.net/hacks/4291/)",
            "- [Creator thread: Pokemon Emerald Kaizo](https://www.pokecommunity.com/threads/pokemon-emerald-kaizo.395830/)",
            "- [pret/pokeemerald: wild encounter generation](https://raw.githubusercontent.com/pret/pokeemerald/master/src/wild_encounter.c)",
            "- [pret/pokeemerald: base RNG implementation](https://raw.githubusercontent.com/pret/pokeemerald/master/src/random.c)",
            "- [TASVideos: Gen 3 RNG mechanics](https://tasvideos.org/GameResources/GBA/PokemonGen3/RNG)",
            "- [Smogon: wild Pokemon RNG methods](https://www.smogon.com/ingame/rng/emerald_rng_part3)",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_SIMULATIONS,
        help="number of candidates in each Monte Carlo check",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("emerald_kaizo_iv_analysis.md"),
    )
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be positive")
    args.output.write_text(render_report(args.trials, args.seed), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
