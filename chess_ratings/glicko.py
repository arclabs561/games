"""Glicko-2 as Lichess runs it, for one rated game at a time.

Glickman's algorithm ("Example of the Glicko-2 system", 2022, glicko.net) with the constants
and steps of Lichess's servers, read from lila (modules/rating, modules/round) and scalachess
(rating/src/main/scala/glicko) on 2026-11-20:

- a new player starts at rating 1500, deviation 500 and volatility 0.09; tau is 0.75;
- each game is its own rating period, with no deviation increase inside it
  (`skipDeviationIncrease = true`);
- before a game the deviation grows with the days since the player's last game,
  phi^2 + 0.21436 * days * sigma^2, "chosen so a typical player's RD goes from 60 -> 110 in 1 year";
- in standard chess White gets 11.782457 rating points of advantage, half added to White's
  rating and half taken from Black's when the expected score is computed;
- blitz gains are multiplied by 1.005 (the rating regulator); no game moves a rating by more than
  700 points; ratings stay at or above 400, deviations within [45, 500], volatility at most 0.1;
- a rating is provisional while its deviation is 110 or more.
"""

from dataclasses import dataclass
from math import exp, log, pi, sqrt

SCALE = 173.7178  # Glicko-2 units per rating point, inverted
DEFAULT_RATING = 1500.0
TAU = 0.75
TOLERANCE = 1e-6

START_DEVIATION = 500.0
START_VOLATILITY = 0.09
MIN_DEVIATION = 45.0
MAX_DEVIATION = 500.0
MAX_VOLATILITY = 0.1
MIN_RATING = 400.0
MAX_DELTA = 700.0
PROVISIONAL = 110.0
PERIODS_PER_DAY = 0.21436
WHITE_ADVANTAGE = 11.782457
REGULATOR = {"blitz": 1.005, "bullet": 1.010, "rapid": 1.015, "classical": 1.010}


@dataclass(frozen=True)
class Glicko:
    """A rating, its deviation (RD) and its volatility, on the rating scale."""

    rating: float = DEFAULT_RATING
    deviation: float = START_DEVIATION
    volatility: float = START_VOLATILITY

    @property
    def provisional(self) -> bool:
        """Lichess shows a question mark while the deviation is at least 110."""
        return self.deviation >= PROVISIONAL


NEW = Glicko()


def g(phi: float) -> float:
    """Glickman's g, which discounts a result against an uncertain opponent."""
    return 1 / sqrt(1 + 3 * phi**2 / pi**2)


def expected(mu: float, mu_opponent: float, phi_opponent: float) -> float:
    """Expected score on the Glicko-2 scale."""
    return 1 / (1 + exp(-g(phi_opponent) * (mu - mu_opponent)))


def new_volatility(sigma: float, phi: float, v: float, delta: float, tau: float = TAU) -> float:
    """Step 5 of Glickman's algorithm: the Illinois method on f(x) = 0."""
    a = log(sigma**2)

    def f(x: float) -> float:
        ex = exp(x)
        return ex * (delta**2 - phi**2 - v - ex) / (2 * (phi**2 + v + ex) ** 2) - (x - a) / tau**2

    lo = a
    if delta**2 > phi**2 + v:
        hi = log(delta**2 - phi**2 - v)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        hi = a - k * tau
    f_lo, f_hi = f(lo), f(hi)
    while abs(hi - lo) > TOLERANCE:
        c = lo + (lo - hi) * f_lo / (f_hi - f_lo)
        f_c = f(c)
        if f_c * f_hi <= 0:
            lo, f_lo = hi, f_hi
        else:
            f_lo /= 2
        hi, f_hi = c, f_c
    return exp(lo / 2)


def rate(
    player: Glicko,
    results: list[tuple[Glicko, float]],
    elapsed: float = 0.0,
    advantages: list[float] | None = None,
) -> Glicko:
    """One Glicko-2 rating period: results are (opponent, score) pairs; no caps or regulator.

    `elapsed` is the number of rating periods added to the deviation in step 6 (Glickman uses 1;
    Lichess uses 0). `advantages` gives, per result, the rating points added to the player's
    rating minus those added to the opponent's when the expected score is computed.
    """
    mu = (player.rating - DEFAULT_RATING) / SCALE
    phi = player.deviation / SCALE
    advantages = advantages or [0.0] * len(results)
    terms = []
    for (opponent, score), advantage in zip(results, advantages, strict=True):
        mu_j = (opponent.rating - DEFAULT_RATING) / SCALE
        phi_j = opponent.deviation / SCALE
        e = expected(mu + advantage / SCALE, mu_j, phi_j)
        terms.append((g(phi_j), e, score))
    v = 1 / sum(gj**2 * e * (1 - e) for gj, e, _ in terms)
    gain = sum(gj * (s - e) for gj, e, s in terms)
    sigma = new_volatility(player.volatility, phi, v, v * gain)
    phi_star = sqrt(phi**2 + elapsed * sigma**2)
    phi_new = 1 / sqrt(1 / phi_star**2 + 1 / v)
    mu_new = mu + phi_new**2 * gain
    return Glicko(mu_new * SCALE + DEFAULT_RATING, phi_new * SCALE, sigma)


def cap(x: Glicko) -> Glicko:
    """Lichess's bounds on a stored rating (`Glicko.cap` in lila)."""
    return Glicko(
        max(x.rating, MIN_RATING),
        min(max(x.deviation, MIN_DEVIATION), MAX_DEVIATION),
        min(x.volatility, MAX_VOLATILITY),
    )


def live(x: Glicko, days: float) -> Glicko:
    """The deviation after `days` without a game, as Lichess reads it before the next game."""
    deviation = sqrt(x.deviation**2 + PERIODS_PER_DAY * days * x.volatility**2)
    return Glicko(x.rating, min(max(deviation, MIN_DEVIATION), MAX_DEVIATION), x.volatility)


def lichess_game(
    white: Glicko,
    black: Glicko,
    white_score: float,
    days: tuple[float, float] = (0.0, 0.0),
    speed: str = "blitz",
    advantage: float = WHITE_ADVANTAGE,
) -> tuple[Glicko, Glicko]:
    """Both players' ratings after one rated standard game, as Lichess computes them."""
    before = (cap(live(cap(white), days[0])), cap(live(cap(black), days[1])))
    after = (
        rate(before[0], [(before[1], white_score)], advantages=[advantage]),
        rate(before[1], [(before[0], 1 - white_score)], advantages=[-advantage]),
    )
    factor = REGULATOR.get(speed, 1.0)
    out = []
    for b, a in zip(before, after, strict=True):
        rating = a.rating
        if rating > b.rating:
            rating += (rating - b.rating) * (factor - 1)
        rating = min(max(rating, b.rating - MAX_DELTA), b.rating + MAX_DELTA)
        out.append(cap(Glicko(rating, a.deviation, a.volatility)))
    return out[0], out[1]


def shown(x: Glicko) -> int:
    """The integer rating Lichess shows and writes in PGN headers (truncated, as Scala's toInt)."""
    return int(x.rating)
