"""How many games until a new Lichess rating settles?

A new player starts at 1500 with deviation (RD) 500, and Lichess shows a question mark until the
RD falls below 110. Model: every opponent is rated exactly what the player is rated now (Lichess
pairs players by rating), has deviation `opponent_rd`, and has true strength equal to that rating;
colors alternate, starting with White; games are played back to back, so the deviation never
grows between them. The player's true strength is `true`, and a game is won with the Elo chance
1 / (1 + 10^((opponent - true) / 400)). Draws are left out except in one simulated variation.

Three results follow.
- The deviation path does not depend on the results, because the expected score depends only on
  the rating gap, which is zero (or White's advantage). Without the color advantage it has a
  closed form, 1/phi_n^2 = 1/phi_0^2 + n g(phi_opp)^2 / 4 on the Glicko-2 scale.
- The distribution of the rating after n games is exact for n up to 20, by a dynamic program
  over the 2^n records, checked by brute force over every record for small n.
- Longer horizons are simulated, with a seed per setting, and checked against the exact
  distribution at 20 games.

    uv run python -m chess_ratings.settle          # about 40 seconds; writes results/settle.json
"""

import json
from math import ceil, erf, lgamma, log, sqrt
from pathlib import Path

import numpy as np

from chess_ratings.glicko import (
    MIN_DEVIATION,
    MIN_RATING,
    NEW,
    PROVISIONAL,
    REGULATOR,
    SCALE,
    START_DEVIATION,
    WHITE_ADVANTAGE,
    expected,
    g,
)

RESULTS = Path(__file__).resolve().parent / "results"
OPPONENT_RD = 60.0
STRENGTHS = (800, 1100, 1500, 1900, 2300)
BOOST = REGULATOR["blitz"]


def win_chance(true: float, opponent: np.ndarray | float) -> np.ndarray | float:
    """The Elo chance that a player of strength `true` beats one of strength `opponent`."""
    return 1 / (1 + 10 ** ((np.asarray(opponent) - true) / 400))


def color_advantage(game: int, advantage: float = WHITE_ADVANTAGE) -> float:
    """White's edge in points, from the player's side: White in games 0, 2, 4, ..."""
    return advantage if game % 2 == 0 else -advantage


def deviation_path(
    games: int, opponent_rd: float = OPPONENT_RD, advantage: float = WHITE_ADVANTAGE
) -> list[float]:
    """The deviation after each of the first `games` games, opponents at the player's rating."""
    phi, out = START_DEVIATION / SCALE, []
    phi_j = opponent_rd / SCALE
    for k in range(games):
        e = expected(color_advantage(k, advantage) / SCALE, 0.0, phi_j)
        v = 1 / (g(phi_j) ** 2 * e * (1 - e))
        phi = max(1 / sqrt(1 / phi**2 + 1 / v), MIN_DEVIATION / SCALE)
        out.append(phi * SCALE)
    return out


def games_to_establish_formula(opponent_rd: float, start: float = START_DEVIATION) -> int:
    """Games until the RD is below 110, without the color advantage: the closed form."""
    need = (SCALE / PROVISIONAL) ** 2 - (SCALE / start) ** 2
    per_game = g(opponent_rd / SCALE) ** 2 / 4
    n = ceil(need / per_game)
    return n + 1 if n * per_game == need else n  # "below" is strict


def games_to_establish(opponent_rd: float, advantage: float = WHITE_ADVANTAGE) -> int:
    """Games until the RD is below 110, by iterating Lichess's update."""
    path = deviation_path(400, opponent_rd, advantage)
    return next(k + 1 for k, d in enumerate(path) if d < PROVISIONAL)


def steps(games: int, opponent_rd: float = OPPONENT_RD) -> list[tuple[float, float]]:
    """(gain if won, loss if lost) in points for each game, with the blitz regulator."""
    phi, out = START_DEVIATION / SCALE, []
    phi_j = opponent_rd / SCALE
    for k in range(games):
        e = expected(color_advantage(k) / SCALE, 0.0, phi_j)
        v = 1 / (g(phi_j) ** 2 * e * (1 - e))
        phi_new = 1 / sqrt(1 / phi**2 + 1 / v)
        win = phi_new**2 * g(phi_j) * (1 - e) * SCALE * BOOST
        loss = phi_new**2 * g(phi_j) * e * SCALE
        out.append((win, loss))
        phi = max(phi_new, MIN_DEVIATION / SCALE)
    return out


def exact_distribution(
    true: float, games: int, opponent_rd: float = OPPONENT_RD, start: float = NEW.rating
) -> tuple[np.ndarray, np.ndarray]:
    """Every rating reachable after `games` games, with its probability (no draws)."""
    ratings, probs = np.array([start]), np.array([1.0])
    for win, loss in steps(games, opponent_rd):
        p = win_chance(true, ratings)
        ratings = np.concatenate([ratings + win, np.maximum(ratings - loss, MIN_RATING)])
        probs = np.concatenate([probs * p, probs * (1 - p)])
    return ratings, probs


def summary(ratings: np.ndarray, probs: np.ndarray, true: float) -> dict[str, float]:
    """Mean, standard deviation and the chances of being within 50 and 100 points of `true`."""
    mean = float(probs @ ratings)
    return {
        "mean": mean,
        "sd": float(sqrt(probs @ (ratings - mean) ** 2)),
        "within_50": float(probs[np.abs(ratings - true) <= 50].sum()),
        "within_100": float(probs[np.abs(ratings - true) <= 100].sum()),
    }


def update(
    r: np.ndarray,
    phi: np.ndarray,
    opponent: np.ndarray,
    opponent_rd: float,
    score: np.ndarray,
    game: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Lichess's update for many players at once (no time gaps, so volatility drops out)."""
    phi_j = opponent_rd / SCALE
    gj = g(phi_j)
    e = 1 / (1 + np.exp(-gj * (r + color_advantage(game) - opponent) / SCALE))
    v = 1 / (gj**2 * e * (1 - e))
    phi_new = 1 / np.sqrt(1 / phi**2 + 1 / v)
    delta = phi_new**2 * gj * (score - e) * SCALE
    delta = np.where(delta > 0, delta * BOOST, delta)
    r_new = np.maximum(r + np.clip(delta, -700, 700), MIN_RATING)
    return r_new, np.maximum(phi_new, MIN_DEVIATION / SCALE)


def simulate(
    true: float,
    games: int,
    players: int,
    seed: int,
    opponent_rd: float = OPPONENT_RD,
    spread: float = 0.0,
    draw: float = 0.0,
) -> np.ndarray:
    """Ratings of `players` simulated players after each game: an array (games + 1, players).

    Opponents are rated the player's current rating plus a normal gap of sd `spread`; a game is
    drawn with chance `draw`, and otherwise won with the Elo chance.
    """
    rng = np.random.default_rng(seed)
    r = np.full(players, NEW.rating)
    phi = np.full(players, START_DEVIATION / SCALE)
    out = [r]
    for k in range(games):
        opponent = r + rng.normal(0, spread, players) if spread else r
        p = win_chance(true, opponent)
        u = rng.random(players)
        score = np.where(u < draw, 0.5, np.where(u < draw + (1 - draw) * p, 1.0, 0.0))
        r, phi = update(r, phi, opponent, opponent_rd, score, k)
        out.append(r)
    return np.array(out)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """The Wilson 95% interval for k successes in n trials."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def noise_floor(opponent_rd: float = OPPONENT_RD) -> float:
    """Standard deviation of a settled rating around the true strength, linearized.

    At RD 45 every decisive game moves the rating by about K/2 points. The mean pull back to the
    true strength is K c (true - r) per game with c = ln 10 / 1600, the Elo slope at an even
    game, and a result has variance 1/4. The AR(1) recursion x' = (1 - Kc) x + K e then has
    variance K^2/4 / (1 - (1 - Kc)^2).
    """
    phi, phi_j = MIN_DEVIATION / SCALE, opponent_rd / SCALE
    v = 1 / (g(phi_j) ** 2 / 4)
    k = (1 / (1 / phi**2 + 1 / v)) * g(phi_j) * SCALE
    c = np.log(10) / 1600
    return float(sqrt(k * k / 4 / (1 - (1 - k * c) ** 2)))


def binomial_pmf(n: int, p: float) -> np.ndarray:
    """P(j successes in n trials) for j = 0..n, in floating point."""
    j = np.arange(n + 1)
    lg = np.array([lgamma(x + 1) for x in range(n + 1)])
    return np.exp(lg[n] - lg[j] - lg[n - j] + j * log(p) + (n - j) * log(1 - p))


def fewest_wins(n: int) -> int:
    """Peggle's `wins_needed(n)` at 95%, in floating point: the least k with
    P(at most k heads in n + 1 fair flips) >= 0.95. The tests check it against the exact one."""
    cdf = np.cumsum(binomial_pmf(n + 1, 0.5))
    return int(np.argmax(cdf >= 0.95 - 1e-12))


def games_to_decide(gap: float, chance: float) -> int:
    """Decisive games between two players `gap` points apart until a uniform-prior verdict of
    95% for the stronger one comes with at least the given chance (Peggle's one-look rule)."""
    p = float(win_chance(gap, 0.0))
    n = 4  # the fewest games in which a clean sweep reaches 95%
    while True:
        if binomial_pmf(n, p)[fewest_wins(n) :].sum() >= chance:
            return n
        n += 1


def normal_two_sided(z: float) -> float:
    """P(|Z| > z) for a standard normal Z."""
    return 1 - erf(z / sqrt(2))


def compute() -> dict:
    """Every number that results/settle.json holds."""
    out: dict = {"opponent_rd": OPPONENT_RD}
    rds = (45, 60, 80, 110, 200, 350, 500)
    out["games_to_establish"] = {
        str(rd): {
            "lichess": games_to_establish(rd),
            "no_color_advantage": games_to_establish(rd, advantage=0.0),
            "closed_form": games_to_establish_formula(rd),
        }
        for rd in rds
    }
    out["deviation_path"] = [round(d, 2) for d in deviation_path(40)]
    out["steps"] = [[round(w, 2), round(lo, 2)] for w, lo in steps(12)]

    exact = {}
    for true in STRENGTHS:
        exact[str(true)] = {
            str(n): summary(*exact_distribution(true, n), true) for n in (5, 10, 15, 20)
        }
    out["exact"] = exact

    horizon, players = 300, 20_000
    simulated = {}
    for i, true in enumerate(STRENGTHS):
        seed = 100 + i
        paths = simulate(true, horizon, players, seed)
        err = np.abs(paths - true)
        within = (err <= 100).mean(axis=1)
        first = np.argmax(err <= 100, axis=0)
        settled = next(n for n in range(horizon + 1) if within[n] >= 0.9)
        k20 = int((err[20] <= 100).sum())
        simulated[str(true)] = {
            "seed": seed,
            "players": players,
            "within_100": {str(n): float(within[n]) for n in (10, 20, 30, 50, 100, 200, 300)},
            "within_100_at_20_interval": wilson(k20, players),
            "median_first_within_100": int(np.median(first)),
            "games_until_90_percent_within_100": settled,
            "sd_after_300": float(paths[-1].std()),
            "mean_after_300": float(paths[-1].mean()),
            "within_100_curve": [float(x) for x in within[:101]],
        }
    out["simulated"] = simulated

    # Draws and pairing spread change little: one setting each, its own seed.
    variants = {}
    for name, kwargs, seed in (
        ("draws_10_percent", {"draw": 0.10}, 201),
        ("pairing_spread_100", {"spread": 100.0}, 202),
    ):
        paths = simulate(1100, horizon, players, seed, **kwargs)
        within = (np.abs(paths - 1100) <= 100).mean(axis=1)
        variants[name] = {
            "seed": seed,
            "true": 1100,
            "games_until_90_percent_within_100": next(
                n for n in range(horizon + 1) if within[n] >= 0.9
            ),
            "sd_after_300": float(paths[-1].std()),
        }
    out["variants"] = variants

    floor = noise_floor()
    out["noise_floor_linear"] = floor
    out["equal_players_rating_gap_95"] = 1.959964 * sqrt(2) * floor
    out["games_to_decide"] = {
        str(gap): {
            "win_chance": float(win_chance(gap, 0.0)),
            "median": games_to_decide(gap, 0.5),
            "ninety": games_to_decide(gap, 0.9),
        }
        for gap in (25, 50, 100, 200, 400)
    }
    return out


def rounded(x, digits: int = 6):
    """Floats rounded to `digits` decimals, so the file does not depend on the last bits."""
    if isinstance(x, float):
        return round(x, digits)
    if isinstance(x, dict):
        return {k: rounded(v, digits) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [rounded(v, digits) for v in x]
    return x


def to_text(out: dict) -> str:
    return json.dumps(rounded(out), indent=1, sort_keys=True) + "\n"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    text = to_text(compute())
    (RESULTS / "settle.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
