"""Write every number the README, its fragments in docs/ and the note quote to results.json.

    uv run python -m tenzi.report     # about half a minute; writes tenzi/results.json

Exact values are stored as floats (or, for the 60-digit decimals of `maximum.py`, as strings
of the digits the note prints) and rounded only where they are printed. Simulated values are
stored as [mean, half-width of the 95% interval], rounded to the three decimals they are
quoted with. `quick=True` leaves out the simulations and the small games, for the tests.
"""

import json
from fractions import Fraction
from pathlib import Path

from tenzi import maximum, optimal, tenzi

FOLDER = Path(__file__).resolve().parent
RESULTS = FOLDER / "results.json"
GAMES = 100_000  # simulated games per strategy
RACES, RACE_PLAYERS = 20_000, 4  # simulated races, and players in each
RACE_TURNS = 200  # the race sums stop here
PLAYERS = (2, 3, 4, 6, 10, 20, 30)
DEADLINES = (5, 10, 14, 20, 30, 50)
HARMONIC_DICE = (1, 2, 3, 5, 10, 20, 50, 100)
SLOW = {"simulated", "race_simulated", "small_games_switching_best"}  # left out when quick


def interval(samples) -> list[float]:
    """[mean, half-width of the 95% interval], as tenzi.interval prints them."""
    center, half = tenzi.interval(samples).split(" ± ")
    return [float(center), float(half)]


def strategy(name: str) -> dict:
    first, step = tenzi.STRATEGIES[name]
    d = tenzi.turn_distribution(step=step, first=first)
    cum, median = Fraction(0), None
    for t in sorted(d):
        cum += d[t]
        if median is None and cum >= Fraction(1, 2):
            median = t
    mean_t, var = tenzi.moments(d)
    assert abs(mean_t - tenzi.expected_turns(step, first)) < Fraction(1, 10**9)
    return {
        "mean": float(tenzi.expected_turns(step, first)),
        "sd": float(var) ** 0.5,
        "median": median,
        "mode": max(d, key=lambda t: d[t]),
        "over_30": float(1 - sum(p for t, p in d.items() if t <= 30)),
    }


def race_table(name: str) -> dict:
    first, step = tenzi.STRATEGIES[name]
    states = [
        {m: float(p) for m, p in s.items()}
        for s in tenzi.state_distributions(RACE_TURNS, step, first)
    ]
    rows = {}
    for n in PLAYERS:
        win, tie, one_left = tenzi.race(n, states)
        rows[str(n)] = [sum(t * w for t, w in enumerate(win, 1)), tie, one_left]
    return rows


def sci(x) -> list:
    """A small positive number as [m, e] with x = m times 10^(-e) and three significant
    digits in m, for the README, which writes it as m·10<sup>−e</sup>."""
    mantissa, exponent = f"{float(x):.2e}".split("e")
    assert int(exponent) < 0
    return [float(mantissa), -int(exponent)]


def harmonic() -> dict:
    gap = {n: maximum.decimal(maximum.exact_mean(n)) - maximum.harmonic_approximation(n)
           for n in HARMONIC_DICE}  # fmt: skip
    lam = maximum.rate()
    y = 2 * maximum.PI / lam
    return {
        "rate": float(lam),
        "y": float(y),
        "largest_factor": float(10 / (100 + y * y).sqrt()),  # j / sqrt(j^2 + y^2) at j = 10
        "approx_10": f"{maximum.harmonic_approximation(10):.12f}",
        "exact_10": f"{maximum.decimal(maximum.exact_mean(10)):.12f}",
        "gap_10": float(gap[10]),
        "gap_10_sci": sci(gap[10]),
        "one_die": float(maximum.harmonic_approximation(1)),
        "term_10": float(maximum.fourier_term(10, 6, 1)),
        "size_k1": float(maximum.term_size(10)),
        "size_k2": float(maximum.term_size(10, k=2)),
        "wobble": float(maximum.wobble_amplitude()),
        "dice_for_wobble": maximum.dice_for_wobble(),
        "gap_over_wobble": float(gap[10] / maximum.wobble_amplitude()),
        "table": {
            str(n): [
                float(maximum.exact_mean(n)),
                float(gap[n]),
                float(maximum.fourier_term(n, 6, 1)),
            ]
            for n in HARMONIC_DICE
        },  # fmt: skip
    }


def results(quick: bool = False) -> dict:
    names = ("target first", "fixed face", "switching")
    target = tenzi.turn_distribution(step=tenzi.reroll, first=tenzi.target_first())
    chances, worse = optimal.by_turn()
    any_chances, any_worse = optimal.by_turn(any_dice=True)
    margins = {}
    for any_dice in (False, True):
        policy, best, values = optimal.solve(any_dice=any_dice)
        assert all(keep == (roll[0],) for roll, keep in policy.items())  # switching
        margins["any dice" if any_dice else "one face"] = float(
            min(
                min(
                    v for k, v in values.items() if k != keep and k in optimal.keeps(roll, any_dice)
                )
                - values[keep]
                for roll, keep in policy.items()
            )
        )
    left = tenzi.turns_left()
    out = {
        "first_roll": {str(m): float(p) for m, p in tenzi.first_roll().items()},
        "turns_left_fixed": {str(m): float(e) for m, e in left.items()},
        "strategies": {name: strategy(name) for name in names},
        "target_first_shifted": {  # the slides' state vectors: P(T <= 4) and P(T = 20)
            "done_by_4": float(sum(target[t] for t in range(1, 5))),
            "on_20": float(target[20]),
        },
        "race_turns": RACE_TURNS,
        "race": {name: race_table(name) for name in ("target first", "switching")},
        "optimal": {
            "mean": float(best),
            "kinds_of_roll": len([r for r in optimal.partitions(10, 6) if r != (10,)]),
            "margin": margins,
            "after_keeping": {str(k): float(values[(k,) if k else ()]) for k in range(10)},
        },
        "deadline": {
            "last": len(chances),
            "switching_best": not worse and not any_worse and chances == any_chances,
            "chances": {str(t): float(chances[t - 1]) for t in DEADLINES},
        },
        "harmonic": harmonic(),
    }
    if not quick:
        out["simulated"] = {
            "games": GAMES,
            "seeds": tenzi.SEEDS,
            **{name: interval(tenzi.simulate(name, GAMES)) for name in names},
        }
        race_sims = {"races": RACES, "players": RACE_PLAYERS}
        for name in ("target first", "switching"):
            rng = tenzi.random.Random(tenzi.SEEDS[f"race, {name}"])
            sims = [tenzi.play_race(rng, RACE_PLAYERS, name) for _ in range(RACES)]
            race_sims[name] = {
                "turn": interval([t for t, _, _ in sims]),
                "tie": interval([float(k > 1) for _, k, _ in sims]),
                "one_left": interval([c for *_, c in sims]),
            }
        out["race_simulated"] = race_sims
        out["small_games_switching_best"] = not any(
            optimal.by_turn(dice, faces, any_dice)[1]
            for dice in range(2, 7)
            for faces in range(2, 7)
            for any_dice in (False, True)
        )
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
