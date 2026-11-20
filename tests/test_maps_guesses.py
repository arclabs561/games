"""Maps: how often a random projection changes the embedding guesser's pick."""

from itertools import combinations
from math import comb

import numpy as np
import pytest

from maps import data, guesses, jl, study


def random_points(n: int, d: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, d))


@pytest.mark.parametrize(
    "n, board, k, seed", [(9, 4, 2, 0), (10, 3, 3, 1), (8, 7, 2, 2), (11, 5, 4, 3)]
)
def test_exact_change_rate_matches_every_board(n, board, k, seed):
    x = random_points(n, 6, seed)
    y = x @ random_points(6, k, seed + 10)
    assert np.isclose(guesses.change_rate(x, y, board), guesses.change_rate_brute(x, y, board))


def test_no_map_changes_nothing():
    # With the same ranking, m_w is w's own rank and the chances add up to 1:
    # sum over r of C(n - 2 - r, b - 1) = C(n - 1, b), the hockey-stick identity.
    assert np.isclose(guesses.same_pick_weights(400, 25).sum(), 1)
    x = random_points(12, 5, 4)
    assert guesses.change_rate(x, x, 4) == 0
    assert guesses.change_rate(x, 3 * x, 4) == 0


def test_clue_ranks():
    cos = np.array([[1.0, 0.2, 0.5], [0.2, 1.0, 0.9], [0.5, 0.9, 1.0]])
    assert guesses.clue_ranks(cos).tolist() == [[-1, 1, 0], [1, -1, 0], [1, 0, -1]]


def margins_by_boards(x: np.ndarray, board: int) -> list[float]:
    cos = guesses.cosines(x)
    out = []
    for c in range(len(x)):
        others = [w for w in range(len(x)) if w != c]
        for b in combinations(others, board):
            top = sorted(cos[c, list(b)], reverse=True)
            out.append(top[0] - top[1])
    return out


@pytest.mark.parametrize("t", [0.05, 0.2, 0.6])
def test_margin_cdf_matches_every_board(t):
    x = random_points(9, 4, 5)
    m = margins_by_boards(x, 4)
    assert np.isclose(guesses.margin_cdf(x, t, 4), np.mean(np.array(m) <= t))


def test_pca_keeps_every_pick_at_full_rank():
    x = random_points(15, 6, 6)
    assert guesses.change_rate(x, guesses.pca_projection(x, 6), 5) == 0


def test_wilson():
    lo, hi = guesses.wilson(5, 20)
    assert np.isclose(lo, 0.1119, atol=1e-4) and np.isclose(hi, 0.4687, atol=1e-4)
    assert 3e39 < comb(399, 25) < 4e39  # why the rate is counted, not enumerated


# The committed results --------------------------------------------------------------------


def test_sampled_boards_agree_with_the_count():
    check = study.stored()["guesses"]["sampled_check"]
    assert check["wilson"][0] <= check["exact"] <= check["wilson"][1]


def test_some_maps_are_current():
    # Rerun the Gaussian maps of seeds 0 and 1 at k = 4 and 64, and PCA, against the file.
    res = study.stored()["guesses"]
    _, x = data.codenames_words()
    r = guesses.clue_ranks(guesses.cosines(x))
    for k in (4, 64):
        row = res["kinds"]["gaussian"][res["k"].index(k)]
        for s in (0, 1):
            y = x @ jl.projection(x.shape[1], k, "gaussian", np.random.default_rng(s))
            rate = 1 - guesses.same_pick(r, guesses.clue_ranks(guesses.cosines(y))).mean()
            assert study.same(round(float(rate), 4), row["rates"][s])
    pca = [guesses.change_rate(x, guesses.pca_projection(x, k)) for k in (4, 16)]
    assert study.same(study.rounded(pca), res["pca"][:3:2])


def test_distortion_matches_the_jl_section():
    # The maps are those of the distortion study, so the medians agree where both have k.
    res, jl_res = study.stored()["guesses"], study.stored()["jl"]["words"]
    for kind in jl.KINDS:
        for k, median in zip(jl_res["k"], jl_res["median_worst"][kind], strict=True):
            assert study.same(res["kinds"][kind][res["k"].index(k)]["median_worst"], median)
