"""Maps: t-SNE from neighbor ranks, PCA starts, and the layout measures."""

from itertools import product

import numpy as np
import pytest
from scipy.stats import t as student_t

from maps import colors, ranks, ranktsne, sne, study


def random_points(n: int, d: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, d))


def knn_p_by_loops(dist: np.ndarray, k: int) -> np.ndarray:
    """The kNN affinities from their definition: weight 1 if either is among the other's k
    nearest (ties by index), then normalized."""
    n = len(dist)
    near = [
        sorted((j for j in range(n) if j != i), key=lambda j: (dist[i, j], j))[:k] for i in range(n)
    ]
    v = np.zeros((n, n))
    for i, j in product(range(n), range(n)):
        if j in near[i] or i in near[j]:
            v[i, j] = 1
    return v / v.sum()


def test_knn_p_by_hand():
    # Points 0, 1, 3, 7 on a line with k = 1: nearest of 0 is 1, of 1 is 0, of 3 is 1
    # (2 against 4), of 7 is 3. The pairs {0,1}, {1,3}, {3,7} each get weight 1 in both
    # directions: six entries of 1/6.
    dist = colors.distances(np.array([[0.0], [1.0], [3.0], [7.0]]))
    p = sne.knn_p(ranks.rank_matrix(dist), 1)
    expected = np.zeros((4, 4))
    for i, j in [(0, 1), (1, 2), (2, 3)]:
        expected[i, j] = expected[j, i] = 1 / 6
    assert np.array_equal(p, expected)


@pytest.mark.parametrize("seed, k", [(0, 3), (1, 5), (2, 8)])
def test_knn_p_matches_its_definition(seed, k):
    dist = colors.distances(random_points(30, 4, seed))
    assert np.allclose(sne.knn_p(ranks.rank_matrix(dist), k), knn_p_by_loops(dist, k), atol=0)


def test_knn_p_sees_only_ranks():
    # Any increasing change of the distances leaves every ranking, and so P, unchanged.
    dist = colors.distances(random_points(40, 3, 3))
    p = sne.knn_p(ranks.rank_matrix(dist), 6)
    assert np.array_equal(p, sne.knn_p(ranks.rank_matrix(np.exp(dist) + dist**3), 6))
    assert np.isclose(p.sum(), 1) and np.array_equal(p, p.T) and (np.diag(p) == 0).all()


def test_pca_start():
    x = random_points(80, 5, 4) * np.arange(1, 6)
    y = sne.pca_start(x)
    assert np.isclose(y[:, 0].std(), 1e-2)
    pc = study.pca(x)
    assert np.allclose(y, pc * (1e-2 / pc[:, 0].std()))


def test_embed_takes_a_joint_target_and_a_start():
    x = random_points(30, 4, 5)
    d2 = sne.sq_distances(x)
    p = sne.joint_p(sne.conditional_p(d2, 8.0)[0])
    a = sne.embed(d2, "tsne", perplexity=8.0, seed=3, iterations=120).y
    b = sne.embed(np.zeros((30, 0)), "tsne", p=p, seed=3, iterations=120).y
    assert np.array_equal(a, b)
    start = sne.pca_start(x)
    c = sne.embed(d2, "tsne", p=p, seed=0, iterations=120, init=start).y
    d = sne.embed(d2, "tsne", p=p, seed=9, iterations=120, init=start).y
    assert np.array_equal(c, d)  # a given start leaves nothing to the seed
    with pytest.raises(ValueError):
        sne.embed(d2, "sne", p=p)


def dcor_by_sums(a: np.ndarray, b: np.ndarray) -> float:
    """Distance correlation from the V-statistic sums of Székely, Rizzo and Bakirov (2007)."""
    da, db = colors.distances(a), colors.distances(b)
    n = len(a)

    def dcov2(u, v):
        s1 = (u * v).sum() / n**2
        s2 = u.sum() / n**2 * v.sum() / n**2
        s3 = sum(u[i, j] * v[i, m] for i in range(n) for j in range(n) for m in range(n)) / n**3
        return s1 + s2 - 2 * s3

    return float(np.sqrt(dcov2(da, db) / np.sqrt(dcov2(da, da) * dcov2(db, db))))


def test_distance_correlation():
    a, b = random_points(9, 2, 6), random_points(9, 2, 7)
    assert np.isclose(ranktsne.distance_correlation(a, b), dcor_by_sums(a, b))
    turn = np.array([[0.6, -0.8], [0.8, 0.6]])
    assert np.isclose(ranktsne.distance_correlation(a, 3 * a @ turn + 5), 1)


def test_neighbor_overlap():
    a = random_points(30, 2, 8)
    assert ranktsne.neighbor_overlap(a, a, 5) == 1
    b = random_points(30, 2, 9)
    ra, rb = ranks.rank_matrix(colors.distances(a)), ranks.rank_matrix(colors.distances(b))
    by_loops = np.mean(
        [
            len(
                {*np.flatnonzero((ra[i] >= 1) & (ra[i] <= 5))}
                & {*np.flatnonzero((rb[i] >= 1) & (rb[i] <= 5))}
            )
            / 5
            for i in range(30)
        ]
    )
    assert np.isclose(ranktsne.neighbor_overlap(a, b, 5), by_loops)


def test_t_interval():
    v = np.random.default_rng(10).normal(size=20).tolist()
    lo, hi = student_t.interval(0.95, 19, loc=np.mean(v), scale=np.std(v, ddof=1) / np.sqrt(20))
    assert np.allclose(ranktsne.t_interval(v), [lo, hi])


# The committed results --------------------------------------------------------------------


def test_seed_zero_is_the_folders_t_sne_map():
    # The Gaussian run from seed 0 is the t-SNE map of the sne section, so its scores agree.
    res = study.stored()
    for name in ("pokemon", "words"):
        rand = res["ranktsne"][name]["random"]["Gaussian"]
        assert rand["T"][0] == res["sne"][name]["t-SNE"]["T"]
        assert rand["C"][0] == res["sne"][name]["t-SNE"]["C"]


def test_colors_runs_are_current():
    # The CSS colors are quick: rerun the PCA starts, seeds 0 and 1, and the k sweep.
    res = study.stored()["ranktsne"]["colors"]
    for kind in ranktsne.AFFINITIES:
        pca = ranktsne.run("colors", kind, "PCA")
        assert study.same(study.rounded(ranktsne.scores("colors", pca)), res["pca"][kind])
        assert np.array_equal(pca.astype(np.float32), ranktsne.stored_maps()[f"colors/{kind}"])
        for s in (0, 1):
            got = study.rounded(ranktsne.scores("colors", ranktsne.run("colors", kind, s)))
            assert study.same(got["T"], res["random"][kind]["T"][s])
            assert study.same(got["C"], res["random"][kind]["C"][s])
    for k in ranktsne.K_SWEEP:
        got = ranktsne.scores("colors", ranktsne.run("colors", "kNN", "PCA", k))
        assert study.same(study.rounded(got), res["k_sweep"][str(k)])


def test_summaries_match_the_stored_seeds():
    for name in ranktsne.SETS:
        res = study.stored()["ranktsne"][name]
        for kind in ranktsne.AFFINITIES:
            t10 = [row[1] for row in res["random"][kind]["T"]]
            assert study.same(study.rounded(ranktsne.summary(t10)), res["summary"]["T10"][kind])
        assert res["boundary_ties"] == ranktsne.boundary_ties(name)
