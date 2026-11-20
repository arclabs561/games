"""Chess: the Glicko-2 port against Glickman's worked example and scalachess's test vectors."""

from math import isclose, sqrt

import pytest

from chess_ratings import glicko
from chess_ratings.glicko import NEW, Glicko, lichess_game, live, rate

# scalachess test-kit/src/test/scala/rating/glicko (MIT license), commit 176d2f7:
# white and black before, the score for White, and White's and Black's rating, RD, volatility.
VECTORS = [
    (
        (1500, 350, 0.06),
        (1500, 350, 0.06),
        1.0,
        (1662.21, 290.2305, 0.0599993),
        (1337.79, 290.2305, 0.0599993),
    ),
    (
        (1500, 350, 0.06),
        (1500, 350, 0.06),
        0.5,
        (1500.00, 290.2305, 0.0599977),
        (1500.00, 290.2305, 0.0599977),
    ),
    (
        (1400, 79, 0.06),
        (1550, 110, 0.065),
        1.0,
        (1422.63, 77.4956, 0.06),
        (1506.32, 105.8706, 0.065),
    ),
    (
        (1400, 79, 0.06),
        (1550, 110, 0.065),
        0.0,
        (1389.99, 77.4956, 0.06),
        (1568.90, 105.8706, 0.065),
    ),
    (
        (1400, 79, 0.06),
        (1550, 110, 0.065),
        0.5,
        (1406.31, 77.4956, 0.06),
        (1537.61, 105.8706, 0.065),
    ),
    (
        (1200, 60, 0.053),
        (1850, 200, 0.062),
        1.0,
        (1216.73, 59.9006, 0.053013),
        (1635.99, 196.9873, 0.062028),
    ),
    (
        (1200, 60, 0.053),
        (1850, 200, 0.062),
        0.0,
        (1199.29, 59.9006, 0.052999),
        (1855.42, 196.9873, 0.061999),
    ),
    (
        (1200, 60, 0.053),
        (1850, 200, 0.062),
        0.5,
        (1208.01, 59.9006, 0.053002),
        (1745.71, 196.9873, 0.062006),
    ),
]


def close(x: Glicko, want: tuple[float, float, float]) -> bool:
    # scalachess's tolerances; it checks volatilities given to two digits to within 1e-5
    vol_tol = (
        0.00001 if len(str(want[2])) <= 5 else 0.0000001 if len(str(want[2])) >= 9 else 0.000001
    )
    return (
        abs(x.rating - want[0]) <= 0.005
        and abs(x.deviation - want[1]) <= 0.00005
        and abs(x.volatility - want[2]) <= vol_tol
    )


@pytest.mark.parametrize(("w", "b", "score", "want_w", "want_b"), VECTORS)
def test_scalachess_vectors(w, b, score, want_w, want_b):
    white, black = Glicko(*w), Glicko(*b)
    assert close(rate(white, [(black, score)]), want_w)
    assert close(rate(black, [(white, 1 - score)]), want_b)


def test_scalachess_color_advantage_vector():
    p = Glicko(1500, 350, 0.06)
    w = rate(p, [(p, 1.0)], advantages=[glicko.WHITE_ADVANTAGE])
    b = rate(p, [(p, 0.0)], advantages=[-glicko.WHITE_ADVANTAGE])
    assert close(w, (1658.56, 290.2538, 0.0599992))
    assert close(b, (1341.44, 290.2538, 0.0599992))


def test_scalachess_new_players():
    # RatingCalculatorTest: two new players (1500, 500, 0.09), White wins: 1741 and 1258, RD 396.
    w = rate(NEW, [(NEW, 1.0)])
    b = rate(NEW, [(NEW, 0.0)])
    assert abs(w.rating - 1741) <= 1 and abs(b.rating - 1258) <= 1
    assert abs(w.deviation - 396) <= 1
    assert isclose(w.volatility, 0.0899983, abs_tol=1e-7)


def test_glickman_worked_example():
    # Glickman, "Example of the Glicko-2 system": 1464.06, 151.52, 0.05999.
    p = Glicko(1500, 200, 0.06)
    opponents = [(Glicko(1400, 30), 1.0), (Glicko(1550, 100), 0.0), (Glicko(1700, 300), 0.0)]
    x = rate(p, opponents, elapsed=1)
    assert round(x.rating, 2) == 1464.05 or round(x.rating, 2) == 1464.06
    assert round(x.deviation, 2) == 151.52
    assert round(x.volatility, 5) == 0.05999


def test_lichess_comment_on_the_daily_increase():
    # "chosen so a typical player's RD goes from 60 -> 110 in 1 year", for volatility 0.06.
    after = live(Glicko(1500, 60, 0.06), 365)
    assert abs(after.deviation - 110) < 0.2
    assert after.deviation == pytest.approx(sqrt(60**2 + 0.21436 * 365 * (0.06 * 173.7178) ** 2))


def test_lichess_game_caps_and_regulator():
    w, b = lichess_game(NEW, NEW, 1.0)
    plain_w = rate(NEW, [(NEW, 1.0)], advantages=[glicko.WHITE_ADVANTAGE])
    assert w.rating - 1500 == pytest.approx((plain_w.rating - 1500) * 1.005)
    assert b.rating < 1500 and w.provisional and b.provisional
    low, _ = lichess_game(Glicko(403, 45, 0.06), Glicko(400, 45, 0.06), 0.0)
    assert low.rating == glicko.MIN_RATING
    settled, _ = lichess_game(Glicko(1500, 45, 0.06), Glicko(1500, 45, 0.06), 0.5)
    assert settled.deviation == glicko.MIN_DEVIATION and not settled.provisional
    assert glicko.shown(Glicko(1499.99)) == 1499
