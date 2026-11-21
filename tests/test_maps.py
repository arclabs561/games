"""Maps: colors, exact orderings, rank scores, SNE and t-SNE, random projections, SOMs."""

import hashlib
import math
from itertools import permutations, product
from math import log

import numpy as np
import pytest
from scipy.spatial.distance import squareform
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.manifold._t_sne import _joint_probabilities, _kl_divergence

from maps import colors, data, draw, figures, jl, paths, portable, ranks, shelf, sne, som, study


def random_points(n: int, d: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, d))


# Portable functions -----------------------------------------------------------------------


def ulps(ours: np.ndarray, theirs: np.ndarray) -> float:
    """Largest difference in units in the last place of theirs."""
    return float((np.abs(ours - theirs) / np.spacing(np.abs(theirs))).max())


def by_math(f, *args: np.ndarray) -> np.ndarray:
    """f from Python's math module (the C library, without numpy's per-CPU kernels), elementwise."""
    return np.array([f(*v) for v in zip(*args, strict=True)])


def test_portable_functions_match_the_c_library():
    # Bounds are the largest differences seen, plus 1 ulp for the library's own error.
    rng = np.random.default_rng(0)
    x = rng.uniform(-700, 700, 20_000)
    assert ulps(portable.exp(x), by_math(math.exp, x)) <= 2
    pos = by_math(math.exp, rng.uniform(-700, 700, 20_000))
    assert ulps(portable.log(pos), by_math(math.log, pos)) <= 2
    assert ulps(portable.log2(pos), by_math(math.log2, pos)) <= 2
    near_one = rng.uniform(0.5, 2, 20_000)
    assert ulps(portable.log(near_one), by_math(math.log, near_one)) <= 3
    unit = rng.uniform(0.05, 1, 20_000)
    assert ulps(portable.power(unit, 2.4), by_math(lambda u: math.pow(u, 2.4), unit)) <= 4
    assert ulps(portable.cbrt(x), by_math(math.cbrt, x)) <= 4
    y = rng.normal(size=20_000)
    assert ulps(portable.atan2(y, x), by_math(math.atan2, y, x)) <= 5


def test_portable_special_values():
    ex = portable.exp(np.array([-np.inf, -746.0, 0.0, 710.0, np.inf, np.nan]))
    assert ex[:5].tolist() == [0.0, 0.0, 1.0, np.inf, np.inf] and np.isnan(ex[5])
    lg = portable.log(np.array([0.0, 1.0, np.inf, 5e-324, -1.0]))
    assert lg[:3].tolist() == [-np.inf, 0.0, np.inf] and np.isclose(lg[3], log(5e-324))
    assert np.isnan(lg[4])
    assert portable.cbrt(np.array([0.0, 8.0, -27.0])).tolist() == [0.0, 2.0, -3.0]
    assert portable.power(np.array([0.0, 1.0, 4.0]), 2.5).tolist() == [0.0, 1.0, 32.0]
    quadrants = portable.atan2(np.array([0.0, 1.0, 0.0, -1.0]), np.array([1.0, 0.0, -1.0, 0.0]))
    assert np.allclose(quadrants, [0, np.pi / 2, np.pi, -np.pi / 2], rtol=0, atol=1e-15)


@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_jacobi_matches_lapack(n):
    a = random_points(n, n, n)
    a = a + a.T
    values, vectors = portable.symmetric_eigen(a)
    theirs = np.linalg.eigh(a)[0][::-1]
    assert np.allclose(values, theirs, atol=1e-12)
    assert np.allclose(vectors.T @ vectors, np.eye(n), atol=1e-12)
    assert np.allclose(a @ vectors, vectors * values, atol=1e-12)


def test_principal_axes_match_svd():
    x = random_points(300, 6, 9) * np.arange(1, 7)
    sd, axes = portable.principal_axes(x, 3)
    _, s, vt = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
    vt = vt[:3] * np.sign(vt[np.arange(3), np.abs(vt[:3]).argmax(axis=1)])[:, None]
    assert np.allclose(axes, vt, atol=1e-12) and np.allclose(sd, s[:3] / np.sqrt(300))


def bits(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=float).tobytes()).hexdigest()[:16]


def test_same_bits_on_every_machine():
    # The hashes were recorded on one machine. A CPU whose numpy or OpenBLAS kernels round
    # differently must still give the same bits, since only IEEE basic operations are used.
    x = np.linspace(-40, 40, 10_001)
    funcs = [
        portable.exp(x),
        portable.log(np.abs(x) + 1e-3),
        portable.log2(np.abs(x) + 1e-3),
        portable.cbrt(x),
        portable.power(np.abs(x), 2.4),
        portable.atan2(x, x[::-1] + 0.5),
    ]
    assert bits(np.concatenate(funcs)) == "f84b02c5aeb2e67a"
    pts = random_points(40, 5, 0)
    d2 = sne.sq_distances(pts)
    tsne = sne.embed(d2, "tsne", perplexity=8, seed=0, iterations=300).y
    assert bits(tsne) == "a7a88ae45c015feb"
    assert bits(sne.embed(d2, "sne", perplexity=8, seed=0, iterations=300).y) == "db577f94e5dc5019"
    m = som.train(pts, (4, 4), steps=2000, seed=0, init="pca")
    assert bits(m.weights) == "c8d061e9465b6842"
    assert bits(study.pca(pts)) == "d5fed80c856b9203"


# Colors -----------------------------------------------------------------------------------


def test_oklab_matches_ottossons_table():
    # Ottosson's table of test values, rounded to three decimals as published.
    xyz = [(0.950, 1.000, 1.089), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    lab = [
        (1.000, 0.000, 0.000),
        (0.450, 1.236, -0.019),
        (0.922, -0.671, 0.263),
        (0.153, -1.415, -0.449),
    ]
    assert np.allclose(colors.xyz_to_oklab(np.array(xyz)), lab, atol=5e-4)


def test_oklab_of_srgb():
    white, black, gray = colors.oklab(["#ffffff", "#000000", "#808080"])
    assert np.allclose(white, [1, 0, 0], atol=1e-6) and np.allclose(black, 0, atol=1e-12)
    assert abs(gray[1]) < 1e-6 and abs(gray[2]) < 1e-6  # grays have no chroma


def test_css_colors():
    css = colors.css_colors()
    assert len(css) == 139 and len(set(css.values())) == 139
    assert "gray" in css and "grey" not in css  # the first name alphabetically is kept


def test_hsv_hue_of_gray_is_zero():
    # Why a hue sort fails: gray reports the hue of red.
    (gray_hue, _, _), (red_hue, _, _) = colors.hsv(["#808080", "#ff0000"])
    assert gray_hue == red_hue == 0


# Orderings --------------------------------------------------------------------------------


def test_points_on_a_line():
    # Four books at 0, 1, 3 and 7 on a line, given out of order: the shortest shelf
    # stands them in order, with total jump 7.
    x = np.array([[3.0], [0.0], [7.0], [1.0]])
    dist = colors.distances(x)
    length, order = paths.held_karp_path(dist)
    assert (length, paths.canonical(order)) == (7.0, [1, 3, 0, 2])
    assert paths.brute_force_path(dist) == (7.0, [1, 3, 0, 2])


@pytest.mark.parametrize("seed", range(6))
def test_exact_solvers_agree(seed):
    n = 5 + seed % 4
    dist = colors.distances(random_points(n, 3, seed))
    brute = paths.brute_force_path(dist)[0]
    held_karp, order = paths.held_karp_path(dist)
    program, order2 = paths.exact_path(dist)
    assert np.isclose(brute, held_karp) and np.isclose(brute, program)
    assert sorted(order) == sorted(order2) == list(range(n))
    assert np.isclose(paths.total_jump(order, dist), held_karp)
    greedy = paths.nn_two_opt(dist)
    assert paths.total_jump(greedy, dist) >= brute - 1e-12


def test_two_opt_never_lengthens():
    dist = colors.distances(random_points(12, 3, 7))
    for start in range(12):
        nn = paths.nearest_neighbor(dist, start)
        assert paths.total_jump(paths.two_opt(nn, dist), dist) <= paths.total_jump(nn, dist)


@pytest.mark.parametrize("d, bits", [(2, 3), (3, 2), (3, 3)])
def test_hilbert_curve_visits_every_cell_by_unit_steps(d, bits):
    cells = list(product(range(2**bits), repeat=d))
    index = {c: paths.hilbert_index(c, bits) for c in cells}
    assert sorted(index.values()) == list(range(len(cells)))
    walk = sorted(cells, key=index.__getitem__)
    assert all(
        sum(abs(a - b) for a, b in zip(p, q, strict=True)) == 1
        for p, q in zip(walk, walk[1:], strict=False)
    )


def test_hand_shelf_hue_sorts():
    hexes = study.hand_hexes()
    by_hue = [study.HAND[i] for i in shelf.hsv_hue(hexes)]
    # Grays have hue 0 and land at the red end; navy sits between orange and crimson.
    assert by_hue[:4] == ["white", "silver", "dimgray", "black"]
    assert by_hue[-3:] == ["navy", "crimson", "lightpink"]


def test_held_karp_on_nine_hand_books_matches_brute_force():
    dist = colors.distances(colors.oklab(study.hand_hexes()[:9]))
    assert np.isclose(paths.brute_force_path(dist)[0], paths.held_karp_path(dist)[0])


# Rank scores ------------------------------------------------------------------------------


def test_trustworthiness_matches_sklearn():
    x, y = random_points(40, 5, 0), random_points(40, 2, 1)
    for k in (1, 5, 12):
        assert np.isclose(
            ranks.trustworthiness(x, y, k), sklearn_trustworthiness(x, y, n_neighbors=k)
        )


def test_continuity_is_trustworthiness_reversed():
    x, y = random_points(30, 4, 2), random_points(30, 2, 3)
    assert np.isclose(ranks.continuity(x, y, 6), ranks.trustworthiness(y, x, 6))


def test_a_perfect_map_scores_one():
    x = random_points(25, 3, 4)
    assert ranks.trustworthiness(x, x, 5) == ranks.continuity(x, x, 5) == 1.0
    precision, recall = ranks.precision_recall(colors.distances(x), colors.distances(x), 5, [5])
    assert precision == recall == [1.0]


def test_tie_orders_bracket_every_compatible_order():
    # A shelf of 8 books: each book has neighbors at equal distance on both sides. Over
    # every order of the tied pairs (the penalty sums row by row, so rows are independent),
    # the best and worst cases are the extremes.
    x = random_points(8, 3, 5)
    data_dist = colors.distances(x)
    shelf_dist = ranks.line_distances([3, 0, 6, 1, 7, 2, 5, 4])
    k = 3
    r_data = ranks.rank_matrix(data_dist)
    lows, highs = [], []
    for i in range(8):
        others = [j for j in range(8) if j != i]
        groups = {}
        for j in others:
            groups.setdefault(shelf_dist[i, j], []).append(j)
        penalties = []
        for choice in product(*[list(permutations(g)) for _, g in sorted(groups.items())]):
            order = [j for g in choice for j in g]
            rank = {j: r + 1 for r, j in enumerate(order)}
            penalties.append(
                sum(r_data[i, j] - k for j in order if rank[j] <= k and r_data[i, j] > k)
            )
        lows.append(min(penalties))
        highs.append(max(penalties))
    scale = 2 / (8 * k * (2 * 8 - 3 * k - 1))
    tc = ranks.trust_continuity(data_dist, shelf_dist, [k])
    assert np.isclose(tc["trust_best"][0], 1 - scale * sum(lows))
    assert np.isclose(tc["trust_worst"][0], 1 - scale * sum(highs))


def test_line_distances():
    assert ranks.line_distances([2, 0, 1]).tolist() == [[0, 1, 1], [1, 0, 2], [1, 2, 0]]


# SNE and t-SNE ----------------------------------------------------------------------------


def test_perplexity_of_two_equidistant_neighbors():
    # Point 0 has two neighbors at the same distance and one far away: at perplexity 2 it
    # gives each near neighbor probability 1/2, up to the far point's share.
    x = np.array([[0.0], [1.0], [-1.0], [10.0]])
    p, _ = sne.conditional_p(sne.sq_distances(x), 2.0)
    assert np.allclose(p[0], [0, 0.5, 0.5, 0], atol=1e-6)


def test_rows_hit_the_perplexity():
    p, _ = sne.conditional_p(sne.sq_distances(random_points(50, 4, 6)), 12.0)
    assert np.allclose(p.sum(axis=1), 1)
    assert max(abs(2 ** sne.entropy_bits(row) - 12) for row in p) < 1e-6


def test_joint_p_matches_sklearn():
    d2 = sne.sq_distances(random_points(60, 5, 1))
    ours = sne.joint_p(sne.conditional_p(d2, 10.0)[0])
    theirs = squareform(_joint_probabilities(d2, 10.0, 0))
    assert np.abs(ours - theirs).max() < 1e-6  # sklearn stops its search at entropy within 1e-5


def test_tsne_cost_and_gradient_match_sklearn():
    x, y = random_points(40, 5, 2), random_points(40, 2, 3)
    p = sne.joint_p(sne.conditional_p(sne.sq_distances(x), 8.0)[0])
    cost, grad = sne.cost_grad("tsne", y, p)
    their_cost, their_grad = _kl_divergence(y.ravel(), squareform(p, checks=False), 1, 40, 2)
    assert np.isclose(cost, their_cost) and np.allclose(grad.ravel(), their_grad, atol=1e-14)


@pytest.mark.parametrize("kind", ["sne", "ssne", "tsne"])
def test_gradients_match_finite_differences(kind):
    x, y = random_points(25, 4, 4), random_points(25, 2, 5)
    p_cond = sne.conditional_p(sne.sq_distances(x), 6.0)[0]
    p = p_cond if kind == "sne" else sne.joint_p(p_cond)
    _, grad = sne.cost_grad(kind, y, p)
    h = 1e-6
    for i, c in product(range(0, 25, 6), range(2)):
        up, down = y.copy(), y.copy()
        up[i, c] += h
        down[i, c] -= h
        numeric = (sne.cost_grad(kind, up, p)[0] - sne.cost_grad(kind, down, p)[0]) / (2 * h)
        assert abs(numeric - grad[i, c]) < 1e-6


def test_tsne_separates_clusters():
    rng = np.random.default_rng(0)
    centers = 10 * np.eye(5)
    x = np.concatenate([c + rng.normal(size=(15, 5)) for c in centers])
    run = sne.embed(sne.sq_distances(x), "tsne", perplexity=10, seed=0, iterations=400)
    assert run.costs[-1] < run.costs[0]
    assert ranks.trustworthiness(x, run.y, 5) > 0.9


# Random projections -----------------------------------------------------------------------


def test_bounds():
    # 4 ln 400 / (1/8 - 1/24) = 48 ln 400 = 287.6 dimensions for eps = 1/2.
    assert np.isclose(jl.k_needed(400, 0.5), 48 * log(400))
    assert np.isclose(jl.k_needed(400, 0.5, beta=1), 72 * log(400))
    assert np.isclose(jl.eps_bound(400, int(jl.k_needed(400, 0.3)) + 1), 0.3, atol=1e-3)
    # Dasgupta and Gupta's eps keeps each pair's exact failure chance below 2 / n^2, and the
    # exact union bound at their success chance 1 / n needs a smaller eps.
    for k in (300, 1000):
        eps = jl.eps_bound(400, k)
        assert jl.gaussian_pair_tail(k, eps) <= 2 / 400**2
        assert jl.eps_union(400, k, 1 - 1 / 400) < eps
        assert jl.eps_union(400, k, 1 / 400) < jl.eps_bound(400, k, beta=1)


def test_ratio_variance_by_hand():
    # For a single coordinate, a +-1 map keeps every length (variance 0), and the sparse
    # map's z^2 is 3 with chance 1/3 and 0 otherwise (variance 3 - 1 = 2 per column).
    e1 = np.array([1.0, 0, 0, 0])
    assert jl.ratio_variance(e1, 8, 1.0) == 0.0
    assert jl.ratio_variance(e1, 8, 3.0) == 2 / 8
    rng = np.random.default_rng(0)
    pair = np.array([[0.0, 0, 0, 0], e1])
    assert np.allclose(jl.ratios(pair, jl.projection(4, 8, "binary", rng)), 1)


@pytest.mark.parametrize("kind", jl.KINDS)
def test_projection_entries(kind):
    r = jl.projection(300, 200, kind, np.random.default_rng(1)) * np.sqrt(200)
    assert abs(r.mean()) < 0.02 and abs((r**2).mean() - 1) < 0.02
    if kind == "sparse":
        assert abs((r == 0).mean() - 2 / 3) < 0.01


# Self-organizing maps ---------------------------------------------------------------------


def test_grid_and_best_unit():
    g = som.grid_coords((2, 3))
    assert g.tolist() == [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
    m = som.SOM(g, g.copy())
    assert som.quantization_error(m, g) == 0 and som.topographic_error(m, g + 0.1) == 0
    assert som.best_units(m, np.array([[0.9, 2.2]])).tolist() == [5]


def test_one_dimensional_map_orders_a_line():
    x = np.linspace(0, 1, 30)[:, None]
    m = som.train(x[::-1], (10,), steps=3000, seed=0, sigma=(3, 0.5), init="pca")
    order = som.line_order(m, x)
    assert order in (list(range(30)), list(range(29, -1, -1)))


# Data and results -------------------------------------------------------------------------


def test_data():
    names, stats, groups = data.pokemon()
    assert sum(len(g) for g in groups) == 721 and len(names) == 705
    assert ["Charmander", "Cyndaquil"] in groups
    words, vectors = data.codenames_words()
    assert words == (data.ROOT / "codenames" / "words.txt").read_text().split()
    assert vectors.shape == (400, 256)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-3)  # float16 storage


@pytest.mark.parametrize("name", study.QUICK)
def test_quick_results_are_current(name):
    assert study.same(study.section(name), study.stored()[name])


@pytest.mark.parametrize("name", list(figures.FIGURES))
def test_figures_are_current(name, tmp_path):
    with draw.style():
        draw.save_svg(figures.FIGURES[name](), tmp_path / "figure.svg")
    assert (tmp_path / "figure.svg").read_bytes() == (figures.OUT / f"{name}.svg").read_bytes()


def test_grays_have_oklab_hue_zero():
    lch = colors.oklch(["#000000", "#696969", "#ffffff", "#ff0000"])
    assert lch[:3, 2].tolist() == [0.0, 0.0, 0.0] and lch[3, 2] > 20
