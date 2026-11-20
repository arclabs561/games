"""Tests for the deck methods on small cases that can be checked by hand or by brute force."""

import itertools
import math

import numpy as np
import pytest

from mtg_decks import decks, embed, mmr, transport

# ---- decks ------------------------------------------------------------------------


def test_split_is_stable_and_near_half_quarter_quarter():
    ids = [f"d{i}" for i in range(4000)]
    s = [decks.split_of(d) for d in ids]
    assert s == [decks.split_of(d) for d in ids]
    shares = {k: s.count(k) / len(s) for k in decks.SPLITS}
    assert abs(shares["fit"] - 0.5) < 0.03 and abs(shares["dev"] - 0.25) < 0.03


def test_cooccurrence_by_hand():
    # three decks over cards a, b, c:  {a:2, b:1}, {a:1, c:1}, {a:1, b:1, c:3}
    x = np.array([[2, 1, 0], [1, 0, 1], [1, 1, 3]])
    s, m, held = decks.cooccurrence(x)
    assert held.tolist() == [3, 2, 2]
    assert s[0, 1] == 2 and s[0, 2] == 2 and s[1, 2] == 1  # decks holding both
    assert m[0, 1] == 2 * 1 + 1 * 1 and m[0, 2] == 1 + 3 and m[1, 2] == 3  # sums of products
    assert np.all(np.diag(s) == 0)


def test_pairs_file_round_trip(tmp_path):
    x = np.array([[2, 1, 0], [1, 0, 1], [1, 1, 3]])
    s, m, _ = decks.cooccurrence(x)
    names = ["Cat", "Bat", "Ant"]
    decks.write_pairs(tmp_path / "p.csv", names, s, m)
    lines = (tmp_path / "p.csv").read_text().splitlines()
    assert lines[0] == "NAME_1,NAME_2,COUNT_SET,COUNT_MULTISET"
    assert lines[1] == "Ant,Bat,1,3"  # names sorted within and across rows
    s2, m2 = decks.read_pairs(tmp_path / "p.csv", names)
    assert np.array_equal(s, s2) and np.array_equal(m, m2)


def test_long_format_builds_decks(tmp_path):
    p = tmp_path / "long.csv"
    p.write_text("deck,card,count\nx,Ash Blossom,3\nx,Maxx C,2\ny,Ash Blossom,1\n")
    d = decks.from_long(decks.read_long(p), {"x": {"group": "Snake-Eye", "games": 3, "wins": 2}})
    assert d.cards == ["Ash Blossom", "Maxx C"]
    assert d.counts.tolist() == [[3, 2], [1, 0]]
    assert d.group.tolist() == ["Snake-Eye", ""] and d.wins.tolist() == [2, 0]


# ---- embeddings -------------------------------------------------------------------


def test_ppmi_by_hand():
    # 4 decks; a and b together in 2, a in 3, b in 2: PMI = log(2 * 4 / (3 * 2)) = log(4/3)
    s = np.array([[0, 2], [2, 0]])
    p = embed.ppmi(s, np.array([3, 2]), 4)
    assert p[0, 1] == pytest.approx(math.log(4 / 3)) and p[0, 0] == 0


def test_ppmi_clips_negative_and_missing():
    s = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    p = embed.ppmi(s, np.array([10, 10, 5]), 20)  # log(1 * 20 / 100) < 0
    assert np.all(p == 0)


def test_eigen_embedding_matches_lapack_and_fixes_signs():
    rng = np.random.default_rng(0)
    a = rng.random((6, 6))
    m = a @ a.T  # positive semidefinite, so the eigen form equals the SVD form
    e = embed.eigen_embedding(m, 6)
    u, sv, _ = np.linalg.svd(m)
    assert np.allclose(np.abs(e), np.abs(u * np.sqrt(sv)), atol=1e-10)
    assert np.allclose(e @ e.T, m, atol=1e-10)
    assert all(e[np.argmax(np.abs(e[:, k])), k] > 0 for k in range(6))


def test_planted_archetypes_are_recovered_and_null_is_not():
    """Two archetypes of 6 cards; each deck draws 4 cards of one archetype and 1 random."""
    rng = np.random.default_rng(1)
    n_decks, rows = 3000, []
    for _ in range(n_decks):
        arch = rng.integers(2)
        own = rng.choice(6, size=4, replace=False) + 6 * arch
        other = rng.choice([c for c in range(12) if c not in own])
        row = np.zeros(12, dtype=int)
        row[own] = 1
        row[other] = 1
        rows.append(row)
    x = np.array(rows)
    labels = ["A"] * 6 + ["B"] * 6
    s, _, held = decks.cooccurrence(x)
    e = embed.eigen_embedding(embed.ppmi(s, held, n_decks), 4)
    assert embed.agreement(embed.neighbors(e, 3), labels) == 1.0
    null = embed.null_counts(held, (x > 0).sum(axis=1), seed=0)
    assert np.array_equal(null.sum(axis=0), held)
    assert np.array_equal(null.sum(axis=1), (x > 0).sum(axis=1))
    s0, _, h0 = decks.cooccurrence(null)
    e0 = embed.eigen_embedding(embed.ppmi(s0, h0, n_decks), 4)
    assert embed.agreement(embed.neighbors(e0, 3), labels) < 0.8
    w, c = embed.sgns(s, 4, seed=0, steps=600, batch=256)
    assert embed.agreement(embed.neighbors(w + c, 3), labels) == 1.0


def test_sgns_optimum_is_shifted_pmi():
    """Levy and Goldberg (2014): at the optimum w_i . c_j = PMI(i, j) - log k.

    With full-rank vectors, negatives drawn by the pair marginal (power 1), and every
    pair frequent, the trained dot products should track the shifted PMI closely.
    """
    rng = np.random.default_rng(2)
    a = rng.integers(20, 200, size=(6, 6))
    s = np.triu(a, 1) + np.triu(a, 1).T
    w, c = embed.sgns(s, 6, seed=0, negatives=2, steps=6000, batch=512, lr=0.1, power=1.0)
    target = embed.pair_pmi(s) - math.log(2)
    iu = np.triu_indices(6, 1)
    dots = 0.5 * ((w @ c.T) + (w @ c.T).T)
    assert np.corrcoef(dots[iu], target[iu])[0, 1] > 0.95


def test_chance_agreement_by_hand():
    # labels A, A, B: card 1 has 1 of 2 others with its label, card 2 has 0 of 2
    assert embed.chance_agreement(["A", "A", "B"]) == pytest.approx((0.5 + 0.5 + 0) / 3)
    assert embed.chance_agreement(["A", None, "A"]) == pytest.approx(0.5)


def test_neighbors_break_ties_by_index():
    x = np.array([[1.0, 0], [1.0, 0], [1.0, 0], [0, 1.0]])
    assert embed.neighbors(x, 2)[0].tolist() == [1, 2]


# ---- transport --------------------------------------------------------------------


def random_metric(n, rng, dim=3):
    return transport.ground_cost(rng.normal(size=(n, dim)))


def test_ground_cost_is_a_metric():
    d = random_metric(9, np.random.default_rng(3))
    assert np.allclose(d, d.T) and np.all(np.diag(d) == 0)
    for i, j, k in itertools.permutations(range(9), 3):
        assert d[i, k] <= d[i, j] + d[j, k] + 1e-12


@pytest.mark.parametrize("seed", range(5))
def test_emd_agrees_with_linear_program_and_brute_force(seed):
    rng = np.random.default_rng(seed)
    cost = random_metric(8, rng)
    src = rng.integers(0, 8, size=5).tolist()
    dst = rng.integers(0, 8, size=5).tolist()
    mu = transport.measure(np.bincount(src, minlength=8))
    nu = transport.measure(np.bincount(dst, minlength=8))
    plan, c = transport.emd(mu, nu, cost)
    assert np.allclose(plan.sum(axis=1), mu) and np.allclose(plan.sum(axis=0), nu)
    assert c == pytest.approx(transport.lp_cost(mu, nu, cost), abs=1e-9)
    assert c == pytest.approx(transport.brute_cost(src, dst, cost), abs=1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_kept_mass_lemma(seed):
    """With a metric cost the optimum equals transport of the surplus onto the deficit."""
    rng = np.random.default_rng(seed + 10)
    cost = random_metric(12, rng)
    mu, nu = rng.random(12), rng.random(12)
    mu, nu = mu / mu.sum(), nu / nu.sum()
    _, full = transport.emd(mu, nu, cost)
    kept = np.minimum(mu, nu)
    surplus, deficit = mu - kept, nu - kept
    _, moved = transport.emd(surplus / surplus.sum(), deficit / deficit.sum(), cost)
    assert full == pytest.approx(moved * surplus.sum(), abs=1e-9)


def test_total_variation_is_transport_with_unit_cost():
    rng = np.random.default_rng(4)
    mu, nu = rng.random(7), rng.random(7)
    mu, nu = mu / mu.sum(), nu / nu.sum()
    unit = 1 - np.eye(7)
    assert transport.total_variation(mu, nu) == pytest.approx(transport.emd(mu, nu, unit)[1])


def test_baseline_swaps_by_hand():
    # deck: a a b c ; target: a b d d  -> cut a (1/4) and c (1/4), add d (2/4)
    mu = transport.measure(np.array([2, 1, 1, 0]))
    nu = transport.measure(np.array([1, 1, 0, 2]))
    sw = transport.baseline_swaps(mu, nu)
    assert [(i, j) for i, j, _ in sw] == [(0, 3), (2, 3)]
    assert sum(m for *_, m in sw) == pytest.approx(0.5)


def test_transport_pairs_by_cost_and_baseline_by_rank():
    # cut a (0.3) and b (0.2); add c (0.3) and d (0.2). a is near d, b near c.
    cost = np.array([[0, 2, 2, 0.1], [2, 0, 0.1, 2], [2, 0.1, 0, 2], [0.1, 2, 2, 0]], dtype=float)
    mu = np.array([0.3, 0.2, 0.25, 0.25])
    nu = np.array([0.0, 0.0, 0.55, 0.45])
    plan, _ = transport.emd(mu, nu, cost)
    ot_pairs = {(i, j) for i, j, _ in transport.swaps(plan)}
    assert (0, 3) in ot_pairs and (1, 2) in ot_pairs
    base = transport.baseline_swaps(mu, nu)
    assert base[0][:2] == (0, 2)  # largest cut paired with largest add, whatever the cost
    mv = np.array([2, 5, 5, 2])
    assert transport.pairing_gap(transport.swaps(plan), mv) < transport.pairing_gap(base, mv)


# ---- MMR --------------------------------------------------------------------------


def test_mmr_by_hand():
    r = np.array([1.0, 0.9, 0.1])
    sim = np.array([[1, 0.95, 0], [0.95, 1, 0], [0, 0, 1]])
    assert mmr.mmr(r, sim, 1.0, 2) == [0, 1]  # top-k
    assert mmr.mmr(r, sim, 0.5, 2) == [0, 2]  # 0.5*0.89 - 0.5*0.95 < 0.5*0 - 0


@pytest.mark.parametrize("seed", range(6))
def test_mmr_matches_rankops(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(15, 4))
    sim = embed.cosine(x)
    r = rng.random(15)
    for lam in (1.0, 0.7, 0.5, 0.3, 0.0):
        assert mmr.mmr(r, sim, lam, 5) == mmr.mmr_rankops(r, sim, lam, 5)


def test_best_subset_is_at_least_greedy():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(10, 3))
    sim, r = embed.cosine(x), rng.random(10)
    for lam in (0.8, 0.5, 0.2):
        best = mmr.best_subset(r, sim, lam, 4)
        greedy = mmr.mmr(r, sim, lam, 4)
        assert mmr.objective(r, sim, lam, best) >= mmr.objective(r, sim, lam, greedy) - 1e-12


def test_list_measures_by_hand():
    r = np.array([3.0, 2.0, 1.0])
    assert mmr.relevance_kept(r, [0, 2]) == pytest.approx(4 / 5)
    sim = np.array([[1, 0.5, 0.2], [0.5, 1, 0.1], [0.2, 0.1, 1]])
    assert mmr.redundancy(sim, [0, 1, 2]) == pytest.approx((0.5 + 0.2 + 0.1) / 3)
    assert mmr.distinct(["x", "y", "x"], [0, 2]) == 1
