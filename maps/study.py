"""Every number the README quotes, computed once and written to results/maps.json.

    uv run python -m maps.study          # all sections (about four minutes on one core)
    uv run python -m maps.study jl som   # some of them

The notebooks call the same cached functions, so they show the same runs. A test reruns
the quick sections and compares them with the committed file; each notebook's last cell
compares the slow sections it computes.
"""

import json
import sys
from functools import cache
from pathlib import Path

import numpy as np

from maps import colors, data, jl, portable, ranks, shelf, sne, som

RESULTS = Path(__file__).parent / "results" / "maps.json"
MAPS = Path(__file__).parent / "results" / "maps.npz"  # the maps that the figures draw
QUICK = ("hand", "jl", "som")
SLOW = ("css", "benchmark", "sne")

# A shelf with the colors that trouble a hue sort: grays, browns and pastels.
HAND = [
    "black",
    "dimgray",
    "silver",
    "white",
    "saddlebrown",
    "sienna",
    "tan",
    "peachpuff",
    "orange",
    "lightpink",
    "crimson",
    "navy",
]


def rounded(x, digits: int = 4):
    """Floats rounded for the JSON file, inside any nesting of dicts and lists."""
    if isinstance(x, dict):
        return {k: rounded(v, digits) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [rounded(v, digits) for v in x]
    if isinstance(x, float | np.floating):
        return None if np.isnan(x) else round(float(x), digits)
    return x


def same(a, b, tol: float = 2e-4) -> bool:
    """Equal structure, with floats equal within tol (for the last digit kept in the file)."""
    if isinstance(a, dict):
        return (
            isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k], tol) for k in a)
        )
    if isinstance(a, list):
        return (
            isinstance(b, list)
            and len(a) == len(b)
            and all(same(u, v, tol) for u, v in zip(a, b, strict=True))
        )
    if isinstance(a, float) and isinstance(b, float | int):
        return abs(a - b) <= tol
    return a == b


# Shelves ----------------------------------------------------------------------------------


def hand_hexes() -> list[str]:
    css = colors.css_colors()
    return [css[name] for name in HAND]


def hand_section() -> dict:
    """The 12-book shelf: each hue sort against the shortest ordering (Held-Karp)."""
    hexes = hand_hexes()
    out = {}
    for name in ("HSV hue", "Oklab hue", "Hilbert curve", "greedy + 2-opt", "shortest"):
        order = shelf.METHODS[name](hexes)
        out[name] = {"order": [HAND[i] for i in order], **shelf.score(hexes, order, (2, 5))}
    return out


@cache
def css_orders() -> dict[str, list[int]]:
    _, hexes = data.css_shelf()
    return {name: method(hexes) for name, method in shelf.METHODS.items()}


def css_section() -> dict:
    """All 139 CSS colors: every method's ordering scored."""
    _, hexes = data.css_shelf()
    return {
        name: {**shelf.score(hexes, order), "order": order} for name, order in css_orders().items()
    }


BENCH_SHELVES, BENCH_BOOKS = 100, 10


@cache
def benchmark_scores() -> dict[str, list[dict[str, float]]]:
    """Every method on 100 random shelves of 10 CSS colors (seeds 0 to 99)."""
    out: dict[str, list[dict[str, float]]] = {name: [] for name in shelf.METHODS}
    for seed in range(BENCH_SHELVES):
        _, hexes = data.random_shelf(BENCH_BOOKS, seed)
        for name, method in shelf.METHODS.items():
            out[name].append(shelf.score(hexes, method(hexes), (2, 4)))
    return out


def benchmark_section() -> dict:
    """Mean excess jump over the shortest, share of shelves solved, mean T(2) and C(2)."""
    scores = benchmark_scores()
    best = np.array([s["jump"] for s in scores["shortest"]])
    out = {}
    for name, rows in scores.items():
        excess = np.array([s["jump"] for s in rows]) / best - 1
        out[name] = {
            "mean_excess": float(excess.mean()),
            "max_excess": float(excess.max()),
            "solved": float((excess < 1e-9).mean()),
            "T2": float(np.mean([s["T2"] for s in rows])),
            "C2": float(np.mean([s["C2"] for s in rows])),
        }
    return out


# Random projections ----------------------------------------------------------------------

JL_SEEDS = 20
JL_KS = {"words": [4, 8, 16, 32, 64, 128, 256, 512, 1024], "pokemon": [1, 2, 3, 6, 12, 24, 48, 96]}


def point_sets() -> dict[str, np.ndarray]:
    _, words = data.codenames_words()
    _, stats, _ = data.pokemon()
    return {"words": words, "pokemon": stats}


@cache
def jl_worst() -> dict[str, dict[str, list[list[float]]]]:
    """Worst distortion for each set, kind, k and seed 0 to 19."""
    out: dict[str, dict[str, list[list[float]]]] = {}
    for name, x in point_sets().items():
        out[name] = {}
        for kind in jl.KINDS:
            out[name][kind] = [
                [
                    jl.worst(
                        jl.ratios(x, jl.projection(x.shape[1], k, kind, np.random.default_rng(s)))
                    )
                    for s in range(JL_SEEDS)
                ]
                for k in JL_KS[name]
            ]
    return out


VAR_K, VAR_DRAWS = 16, 20_000


@cache
def variance_check() -> dict[str, dict[str, dict[str, float]]]:
    """One pair from each set (the first two words; Shuckle and Blissey), 20,000 maps of each
    kind to k = 16 dimensions: the ratio's sample mean and variance, with the variance's
    standard error, against the exact variance."""
    names, stats, _ = data.pokemon()
    words = point_sets()["words"]
    pairs = {
        "words": words[0] - words[1],
        "pokemon": stats[names.index("Shuckle")] - stats[names.index("Blissey")],
    }
    out: dict[str, dict[str, dict[str, float]]] = {}
    for j, (name, a) in enumerate(pairs.items()):
        out[name] = {}
        for i, kind in enumerate(jl.KINDS):
            rng = np.random.default_rng(100 + 10 * j + i)
            sims = (
                np.array(
                    [
                        ((a @ jl.projection(len(a), VAR_K, kind, rng)) ** 2).sum()
                        for _ in range(VAR_DRAWS)
                    ]
                )
                / (a**2).sum()
            )
            var = sims.var(ddof=1)
            m4 = ((sims - sims.mean()) ** 4).mean()
            out[name][kind] = {
                "mean": float(sims.mean()),
                "variance": float(var),
                "variance_se": float(np.sqrt((m4 - var**2) / VAR_DRAWS)),
                "exact_variance": jl.ratio_variance(a, VAR_K, jl.FOURTH_MOMENT[kind]),
            }
    return out


def jl_section() -> dict:
    worst = jl_worst()
    out: dict = {"variance": variance_check()}
    for name, x in point_sets().items():
        n, d = x.shape
        out[name] = {"n": n, "d": d, "k": JL_KS[name], "median_worst": {}, "bounds": {}}
        for kind in jl.KINDS:
            out[name]["median_worst"][kind] = [float(np.median(w)) for w in worst[name][kind]]
        out[name]["bounds"] = {
            "dasgupta_gupta": [jl.eps_bound(n, k) for k in JL_KS[name]],
            "achlioptas_beta1": [jl.eps_bound(n, k, beta=1) for k in JL_KS[name]],
            "union_chi2": [jl.eps_union(n, k, 1 / n) for k in JL_KS[name]],
            "k_for_eps_half": jl.k_needed(n, 0.5),
            "k_for_eps_half_beta1": jl.k_needed(n, 0.5, beta=1),
        }
    return out


# SNE and t-SNE ---------------------------------------------------------------------------

PERPLEXITY = 30.0
KS = [5, 10, 20]


@cache
def embeddings(name: str) -> dict[str, np.ndarray]:
    """PCA, SNE, symmetric SNE and t-SNE maps of a point set (seed 0, perplexity 30)."""
    x = point_sets()[name]
    d2 = sne.sq_distances(x)
    out = {"PCA": pca(x)}
    for kind, label in (("sne", "SNE"), ("ssne", "symmetric SNE"), ("tsne", "t-SNE")):
        out[label] = sne.embed(d2, kind, perplexity=PERPLEXITY, seed=0).y
    return out


def pca(x: np.ndarray, dims: int = 2) -> np.ndarray:
    """Coordinates on the leading principal components, each signed so its largest loading
    is positive (Jacobi's method and fixed-order sums, so the same bits on any machine)."""
    _, axes = portable.principal_axes(x, dims)
    return portable.dot_rows(x - x.mean(axis=0), axes)


def map_scores(x: np.ndarray, y: np.ndarray, p: np.ndarray) -> dict[str, list[float] | float]:
    """t-SNE's cost of a map (KL with the Student-t q), and T and C at k = 5, 10, 20."""
    w = 1.0 / (1.0 + sne.sq_distances(y))
    np.fill_diagonal(w, 0.0)
    tc = ranks.trust_continuity(colors.distances(x), colors.distances(y), KS)
    return {"kl_t": sne.kl(p, w / w.sum()), "T": tc["trust"], "C": tc["cont"]}


def sne_section() -> dict:
    out = {}
    for name, x in point_sets().items():
        p = sne.joint_p(sne.conditional_p(sne.sq_distances(x), PERPLEXITY)[0])
        out[name] = {label: map_scores(x, y, p) for label, y in embeddings(name).items()}
    return out


# Self-organizing maps --------------------------------------------------------------------

COLOR_SHAPE, COLOR_STEPS = (20, 20), 20_000
POKE_SHAPE, POKE_STEPS = (16, 16), 40_000


COLOR_SNAPSHOTS = (0, 100, 1000, 5000, COLOR_STEPS)


@cache
def color_map() -> tuple[som.SOM, np.ndarray]:
    """Kohonen's color demonstration: a 20 x 20 map trained on 1,000 random sRGB colors."""
    x = np.random.default_rng(0).random((1000, 3))
    return som.train(x, COLOR_SHAPE, COLOR_STEPS, seed=1, snapshots=COLOR_SNAPSHOTS), x


@cache
def pokemon_map() -> tuple[som.SOM, np.ndarray]:
    """A 16 x 16 map of the Pokémon stat lines (raw stats, as for the SNE maps)."""
    _, x, _ = data.pokemon()
    return som.train(x, POKE_SHAPE, POKE_STEPS, seed=0, init="pca"), x


def som_section() -> dict:
    cm, cx = color_map()
    pm, px = pokemon_map()
    tc = ranks.trust_continuity(colors.distances(px), colors.distances(som.positions(pm, px)), KS)
    return {
        "colors": {
            "quantization_error": som.quantization_error(cm, cx),
            "topographic_error": som.topographic_error(cm, cx),
        },
        "pokemon": {
            "quantization_error": som.quantization_error(pm, px),
            "topographic_error": som.topographic_error(pm, px),
            "units_used": len(set(som.best_units(pm, px).tolist())),
            "T": tc["trust"],
            "C": tc["cont"],
            "T_best": tc["trust_best"],
            "T_worst": tc["trust_worst"],
            "C_best": tc["cont_best"],
            "C_worst": tc["cont_worst"],
        },
    }


SECTIONS = {
    "hand": hand_section,
    "css": css_section,
    "benchmark": benchmark_section,
    "jl": jl_section,
    "sne": sne_section,
    "som": som_section,
}


def stored() -> dict:
    """The committed results."""
    return json.loads(RESULTS.read_text())


def section(name: str) -> dict:
    """One section computed now, rounded as in the file."""
    return rounded(SECTIONS[name]())


def map_arrays() -> dict[str, np.ndarray]:
    """The t-SNE, SNE and PCA maps and the color map's weights, as float32."""
    out = {
        f"{name}/{label}": y.astype(np.float32)
        for name in ("pokemon", "words")
        for label, y in embeddings(name).items()
    }
    out["colors/weights"] = color_map()[0].weights.astype(np.float32)
    return out


def stored_maps() -> dict[str, np.ndarray]:
    """The committed maps."""
    with np.load(MAPS) as z:
        return {k: z[k] for k in z.files}


def main(names: list[str]) -> None:
    current = stored() if RESULTS.exists() else {}
    for name in names or list(SECTIONS):
        current[name] = section(name)
        print(f"computed {name}")
    if not names or "sne" in names:
        np.savez_compressed(MAPS, **map_arrays())  # ty: ignore[invalid-argument-type]
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(
        json.dumps({k: current[k] for k in SECTIONS if k in current}, indent=1) + "\n"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
