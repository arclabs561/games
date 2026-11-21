"""t-SNE from neighbor ranks alone, against t-SNE with Gaussian affinities.

t-SNE can be read as matching each point's ranking of its neighbors in the data with its
ranking on the map. The **Gaussian affinities** of van der Maaten and Hinton do not use
ranks alone: a Gaussian of chosen perplexity weights the near neighbors by distance. The
**kNN affinities** of Böhm, Berens and Kobak (JMLR 2022, section 3.1) keep only the ranks.
A pair has weight 1 if either point is among the other's k nearest, and 0 otherwise. Any
change of distance that keeps every point's order of neighbors leaves them unchanged.

Both run through the same optimizer (sne.embed) from two kinds of start: 20 random starts
(seeds 0 to 19, the same start for both affinities) and the scaled principal components,
which make the run deterministic.

    uv run python -m maps.study ranktsne   # this section (about 20 minutes on one core)
"""

from functools import cache
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from maps import colors, data, ranks, sne

SETS = ("pokemon", "colors", "words")
SEEDS = list(range(20))
PERPLEXITY = 30.0
K_NN = 15  # Böhm et al. use k = 15 against perplexity 30
K_SWEEP = [5, 10, 15, 30, 45]
KS = [5, 10, 20]
AFFINITIES = ("Gaussian", "kNN")
MAPS = Path(__file__).parent / "results" / "ranktsne.npz"


@cache
def points(name: str) -> np.ndarray:
    """The Pokémon stat lines, the 139 CSS colors in Oklab, or the 400 word vectors."""
    if name == "pokemon":
        return data.pokemon()[1]
    if name == "colors":
        return colors.oklab(data.css_shelf()[1])
    if name == "words":
        return data.codenames_words()[1]
    raise ValueError(name)


@cache
def data_ranks(name: str) -> np.ndarray:
    """Each point's ranking of the others in the data (ties ordered by index)."""
    return ranks.rank_matrix(colors.distances(points(name)))


def boundary_ties(name: str, k: int = K_NN) -> int:
    """Points whose k-th and (k+1)-th nearest neighbors are at the same distance, so that
    the kNN affinities depend on how ties are ordered."""
    dist = colors.distances(points(name))
    r = data_ranks(name)
    kth = np.array([dist[i, r[i] == k][0] for i in range(len(r))])
    nxt = np.array([dist[i, r[i] == k + 1][0] for i in range(len(r))])
    return int((kth == nxt).sum())


@cache
def affinity(name: str, kind: str, k: int = K_NN) -> np.ndarray:
    """The joint P: Gaussian of perplexity 30, or kNN with k neighbors."""
    if kind == "Gaussian":
        return sne.joint_p(sne.conditional_p(sne.sq_distances(points(name)), PERPLEXITY)[0])
    if kind == "kNN":
        return sne.knn_p(data_ranks(name), k)
    raise ValueError(kind)


@cache
def run(name: str, kind: str, start: int | str, k: int = K_NN) -> np.ndarray:
    """A t-SNE map from a random start (the seed) or from the principal components ("PCA")."""
    init = sne.pca_start(points(name)) if start == "PCA" else None
    seed = 0 if start == "PCA" else int(start)
    p = affinity(name, kind, k)
    return sne.embed(np.zeros((len(p), 0)), "tsne", p=p, seed=seed, init=init).y


def scores(name: str, y: np.ndarray) -> dict[str, list[float]]:
    """T(k) and C(k) at k = 5, 10 and 20."""
    tc = ranks.trust_continuity(colors.distances(points(name)), colors.distances(y), KS)
    return {"T": tc["trust"], "C": tc["cont"]}


def distance_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Székely, Rizzo and Bakirov's distance correlation of two maps of the same points.

    It is 1 when one map is a rotation, reflection or scaling of the other, and it is the
    measure Böhm et al. use to compare layouts.
    """

    def centered(y: np.ndarray) -> np.ndarray:
        d = colors.distances(y)
        return d - d.mean(axis=0) - d.mean(axis=1)[:, None] + d.mean()

    ca, cb = centered(a), centered(b)
    cov, va, vb = (ca * cb).mean(), (ca * ca).mean(), (cb * cb).mean()
    return float(np.sqrt(cov / np.sqrt(va * vb)))


def neighbor_overlap(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    """Mean share of each point's k nearest on map a that are among its k nearest on map b."""
    ra, rb = ranks.rank_matrix(colors.distances(a)), ranks.rank_matrix(colors.distances(b))
    near_a, near_b = (ra >= 1) & (ra <= k), (rb >= 1) & (rb <= k)
    return float((near_a & near_b).sum(axis=1).mean() / k)


def t_interval(values: list[float], level: float = 0.95) -> list[float]:
    """Student t interval for the mean of the values."""
    v = np.asarray(values, float)
    half = student_t.ppf((1 + level) / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return [float(v.mean() - half), float(v.mean() + half)]


def summary(values: list[float]) -> dict[str, float | list[float]]:
    """Mean, 95% t interval for the mean, and range over seeds."""
    return {
        "mean": float(np.mean(values)),
        "ci": t_interval(values),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def set_section(name: str) -> dict:
    """Every comparison for one point set."""
    per_seed = {kind: [scores(name, run(name, kind, s)) for s in SEEDS] for kind in AFFINITIES}
    pca = {kind: scores(name, run(name, kind, "PCA")) for kind in AFFINITIES}
    out: dict = {"n": len(points(name)), "boundary_ties": boundary_ties(name)}
    out["random"] = {
        kind: {m: [r[m] for r in rows] for m in ("T", "C")} for kind, rows in per_seed.items()
    }
    out["pca"] = pca
    out["summary"] = {}
    for m in ("T", "C"):
        for j, k in enumerate(KS):
            key = f"{m}{k}"
            g = [r[m][j] for r in per_seed["Gaussian"]]
            u = [r[m][j] for r in per_seed["kNN"]]
            out["summary"][key] = {
                "Gaussian": summary(g),
                "kNN": summary(u),
                "difference": summary([b - a for a, b in zip(g, u, strict=True)]),
                "pca": {kind: pca[kind][m][j] for kind in AFFINITIES},
                "random_below_pca": {
                    kind: float(np.mean([v < pca[kind][m][j] for v in vals]))
                    for kind, vals in (("Gaussian", g), ("kNN", u))
                },
            }
    pairs = list(zip(SEEDS, SEEDS[1:], strict=False))
    out["layout"] = {
        "pca": {
            "dcor": distance_correlation(run(name, "Gaussian", "PCA"), run(name, "kNN", "PCA")),
            "overlap10": neighbor_overlap(run(name, "Gaussian", "PCA"), run(name, "kNN", "PCA")),
        },
        "same_start": {
            "dcor": summary(
                [distance_correlation(run(name, "Gaussian", s), run(name, "kNN", s)) for s in SEEDS]
            ),
            "overlap10": summary(
                [neighbor_overlap(run(name, "Gaussian", s), run(name, "kNN", s)) for s in SEEDS]
            ),
        },
        "next_seed": {
            kind: {
                "dcor": summary(
                    [distance_correlation(run(name, kind, a), run(name, kind, b)) for a, b in pairs]
                ),
                "overlap10": summary(
                    [neighbor_overlap(run(name, kind, a), run(name, kind, b)) for a, b in pairs]
                ),
            }
            for kind in AFFINITIES
        },
    }
    out["k_sweep"] = {str(k): scores(name, run(name, "kNN", "PCA", k)) for k in K_SWEEP}
    return out


def section() -> dict:
    """The ranktsne section of results/maps.json; also writes the PCA-start maps."""
    out = {name: set_section(name) for name in SETS}
    np.savez_compressed(MAPS, **map_arrays())  # ty: ignore[invalid-argument-type]
    return out


def map_arrays() -> dict[str, np.ndarray]:
    """The PCA-start maps of each set under both affinities, as float32."""
    return {
        f"{name}/{kind}": run(name, kind, "PCA").astype(np.float32)
        for name in SETS
        for kind in AFFINITIES
    }


def stored_maps() -> dict[str, np.ndarray]:
    """The committed PCA-start maps."""
    with np.load(MAPS) as z:
        return {key: z[key] for key in z.files}
