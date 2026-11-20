"""Ordering books by color: seven ways to put colors in a line, and how to score them.

uv run python -m maps.shelf   # every method on the 139 CSS colors, with timings
"""

import time
from collections.abc import Callable

import numpy as np

from maps import colors, paths, ranks, sne, som

Method = Callable[[list[str]], list[int]]


def hsv_hue(hexes: list[str]) -> list[int]:
    """Sort by hue as a color picker reports it (HSV), then by value; grays have hue 0."""
    h = colors.hsv(hexes)
    return paths.sort_by([(round(hue, 12), -v) for hue, _, v in h])


def oklch_hue(hexes: list[str]) -> list[int]:
    """Sort by Oklab hue angle, then by lightness."""
    lch = colors.oklch(hexes)
    return paths.sort_by([(round(hue, 9), lightness) for lightness, _, hue in lch])


def hilbert(hexes: list[str]) -> list[int]:
    """Sort by position along the Hilbert curve through the 256^3 sRGB cube."""
    rgb = np.rint(colors.hex_to_rgb(hexes) * 255).astype(int)
    return paths.sort_by([(paths.hilbert_index(tuple(int(c) for c in p), 8),) for p in rgb])


def greedy(hexes: list[str]) -> list[int]:
    """Nearest neighbor from every start, then 2-opt, in Oklab."""
    return paths.nn_two_opt(colors.distances(colors.oklab(hexes)))


def som_line(hexes: list[str], seed: int = 0) -> list[int]:
    """A one-dimensional self-organizing map with one unit per book, trained in Oklab."""
    lab = colors.oklab(hexes)
    n = len(hexes)
    m = som.train(lab, (n,), steps=200 * n, seed=seed, sigma=(n / 4, 0.5), init="pca")
    return som.line_order(m, lab)


def tsne_line(hexes: list[str], seed: int = 0, learning_rate: float = 10.0) -> list[int]:
    """One-dimensional t-SNE in Oklab, perplexity min(10, (n - 1) / 3); sort by coordinate.

    The paper's learning rate of 100 throws the points of a ten-book shelf hundreds of
    units apart, where the order stops improving; 10 works for 10 to 139 books.
    """
    return tsne_lines(hexes, [seed], learning_rate)[0]


def tsne_lines(hexes: list[str], seeds: list[int], learning_rate: float = 10.0) -> list[list[int]]:
    """tsne_line for each seed, with the data's probabilities computed once."""
    lab = colors.oklab(hexes)
    d2 = sne.sq_distances(lab)
    perplexity = min(10.0, (len(hexes) - 1) / 3)
    p_cond, _ = sne.conditional_p(d2, perplexity)
    orders = []
    for seed in seeds:
        run = sne.embed(
            d2,
            "tsne",
            dims=1,
            perplexity=perplexity,
            seed=seed,
            learning_rate=learning_rate,
            p_cond=p_cond,
        )
        orders.append(paths.sort_by([(float(v),) for v in run.y[:, 0]]))
    return orders


def exact(hexes: list[str]) -> list[int]:
    """The shortest ordering: Held-Karp up to 13 books, the integer program above that."""
    dist = colors.distances(colors.oklab(hexes))
    solve = paths.held_karp_path if len(hexes) <= 13 else paths.exact_path
    return paths.canonical(solve(dist)[1])


METHODS: dict[str, Method] = {
    "HSV hue": hsv_hue,
    "Oklab hue": oklch_hue,
    "Hilbert curve": hilbert,
    "1D SOM": som_line,
    "1D t-SNE": tsne_line,
    "greedy + 2-opt": greedy,
    "shortest": exact,
}


def score(hexes: list[str], order: list[int], ks: tuple[int, ...] = (2, 5)) -> dict[str, float]:
    """Total jump in Oklab, the largest single jump, and T and C at each k."""
    dist = colors.distances(colors.oklab(hexes))
    out = {
        "jump": paths.total_jump(order, dist),
        "max_jump": float(max(dist[a, b] for a, b in zip(order, order[1:], strict=False))),
    }
    tc = ranks.trust_continuity(dist, ranks.line_distances(order), list(ks))
    for k, t, c in zip(ks, tc["trust"], tc["cont"], strict=True):
        out[f"T{k}"], out[f"C{k}"] = t, c
    return out


def main() -> None:
    from maps.data import css_shelf

    _, hexes = css_shelf()
    for name, method in METHODS.items():
        start = time.perf_counter()
        order = method(hexes)
        took = time.perf_counter() - start
        s = score(hexes, order)
        cells = "  ".join(f"{k} {v:.3f}" for k, v in s.items())
        print(f"{name:15} {cells}  ({took:.1f} s)")


if __name__ == "__main__":
    main()
