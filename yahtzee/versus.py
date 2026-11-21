"""Head to head: optimal play for the mean against the cheat sheet, one game each.

The rules score the higher total as the winner. Two players who keep separate score cards and
do not react to each other's play have independent final scores A and B, so

    P(A > B) = sum over a of P(A = a) P(B < a),

and the margin A - B has the distribution of the convolution of A with -B. Both distributions
are exact: optimal play's from `distribution.csv`, the cheat sheet's from following the card
forward and backward (`policy.graph` and `policy.distribution`). The check is independent
seeded games of each strategy, paired at random.

    uv run python -m yahtzee.versus     # writes results/versus.json; about 10 minutes
"""

import json
import sys
from pathlib import Path

import numpy as np

from yahtzee import human, policy
from yahtzee.learn import from_json
from yahtzee.rules import OFFICIAL
from yahtzee.simulate import deal
from yahtzee.simulate import play as play_values
from yahtzee.solver import Tables, solve
from yahtzee.target import optimal_distribution

HERE = Path(__file__).parent
RESULTS = HERE / "results"
GAMES, SEEDS = 20000, (3, 4)  # seeded games of optimal play and of the sheet; fresh seeds


def sheet_card() -> human.Strategy:
    """The cheat sheet, as `cards.py` committed it."""
    return from_json(json.loads((RESULTS / "cards.json").read_text())["cheat"]["card"])


def sheet_distribution(t: Tables) -> np.ndarray:
    """The exact distribution of the cheat sheet's final score."""
    card = sheet_card()
    _, values = human.card_values(card, t)
    return policy.distribution(policy.graph(t, values, rule=card))


def duel(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """P(A > B), P(A = B) and P(A < B) for independent scores with distributions a and b."""
    n = max(len(a), len(b))
    a, b = np.pad(a, (0, n - len(a))), np.pad(b, (0, n - len(b)))
    below = np.concatenate([[0.0], np.cumsum(b)[:-1]])  # P(B < s)
    win, tie = float(a @ below), float(a @ b)
    return {"win": win, "tie": tie, "loss": 1.0 - win - tie}


def margin(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int]:
    """The distribution of A - B, and the margin of its first entry."""
    return np.convolve(a, b[::-1]), -(len(b) - 1)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """The Wilson 95% interval for k successes in n trials."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def summary(d: np.ndarray, first: int) -> dict:
    """Mean, median, quartiles and a few tail chances of a margin distribution."""
    m = np.arange(len(d)) + first
    cdf = np.cumsum(d)
    q = {f"q{int(p * 100)}": int(m[np.searchsorted(cdf, p)]) for p in (0.25, 0.5, 0.75)}
    return {
        "mean": float(m @ d),
        **q,
        "by_50_or_more": float(d[m >= 50].sum()),
        "behind_by_50_or_more": float(d[m <= -50].sum()),
        "within_10": float(d[np.abs(m) <= 10].sum()),
    }


def simulate(t: Tables) -> dict:
    """Independent seeded games: optimal play (seed 3) against the sheet (seed 4)."""
    _, values = solve(OFFICIAL)
    a = play_values(t, values, deal(t, GAMES, SEEDS[0]))
    b = human.play(sheet_card(), t, GAMES, SEEDS[1])
    wins, ties = int((a > b).sum()), int((a == b).sum())
    return {
        "games": GAMES,
        "seeds": list(SEEDS),
        "win": wins / GAMES,
        "win_interval": wilson(wins, GAMES),
        "tie": ties / GAMES,
        "tie_interval": wilson(ties, GAMES),
        "mean_margin": float((a - b).mean()),
    }


def results() -> dict:
    t = Tables(OFFICIAL)
    best, sheet = optimal_distribution(), sheet_distribution(t)
    d, first = margin(best, sheet)
    s = np.arange(len(sheet))
    return {
        "sheet_mean": float(s @ sheet),
        "optimal_against_sheet": duel(best, sheet),
        "margin": summary(d, first),
        "optimal_against_optimal": duel(best, best),
        "sheet_against_sheet": duel(sheet, sheet),
        "simulated": simulate(t),
        "sheet_distribution": sheet.tolist(),
    }


def main(argv: list[str]) -> None:
    data = results()
    (RESULTS / "versus.json").write_text(json.dumps(data, indent=1) + "\n")
    print(json.dumps({k: v for k, v in data.items() if k != "sheet_distribution"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
