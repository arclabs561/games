"""The point sets: book colors, Pokémon base stats and Codenames word vectors.

uv run python -m maps.data   # rebuild data/codenames-wordllama.npz (needs the codenames group)
"""

from pathlib import Path

import numpy as np

from maps import colors

HERE = Path(__file__).parent
ROOT = HERE.parent
VECTORS = HERE / "data" / "codenames-wordllama.npz"
STATS = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]


def css_shelf() -> tuple[list[str], list[str]]:
    """All 139 distinct CSS named colors: names and hex values, in alphabetical order."""
    css = colors.css_colors()
    return list(css), list(css.values())


def random_shelf(n: int, seed: int) -> tuple[list[str], list[str]]:
    """n distinct CSS colors drawn at random, in the order drawn."""
    names, hexes = css_shelf()
    pick = np.random.default_rng(seed).choice(len(names), n, replace=False)
    return [names[i] for i in pick], [hexes[i] for i in pick]


def pokemon() -> tuple[list[str], np.ndarray, list[list[str]]]:
    """Distinct base-stat lines of the 721 species of Generations 1 to 6.

    Pokemon.csv lists forms too; a species is its first row per Pokédex number. Species
    with the same six stats (Charmander and Cyndaquil, say) are one point here, named by
    the first. Returns names, the stats as floats, and the species behind each point.
    """
    import csv

    with open(ROOT / "pokemon" / "Pokemon.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    seen: set[str] = set()
    groups: dict[tuple[int, ...], list[str]] = {}
    for row in rows:
        if row["#"] in seen:
            continue
        seen.add(row["#"])
        groups.setdefault(tuple(int(row[s]) for s in STATS), []).append(row["Name"])
    lines = list(groups)
    return [groups[s][0] for s in lines], np.array(lines, float), [groups[s] for s in lines]


def pokemon_types() -> dict[str, str]:
    """Primary type of each species, by name."""
    import csv

    with open(ROOT / "pokemon" / "Pokemon.csv", newline="") as f:
        return {row["Name"]: row["Type 1"] for row in csv.DictReader(f)}


def codenames_words() -> tuple[list[str], np.ndarray]:
    """The 400 Codenames board words of codenames/words.txt and their wordllama vectors.

    Unit vectors in 256 dimensions, stored as float16 and returned as float64.
    """
    z = np.load(VECTORS)
    return list(z["words"]), z["vectors"].astype(np.float64)


def build_vectors() -> None:
    """Compute the word vectors with the codenames package and save them."""
    from codenames import embed

    words = (ROOT / "codenames" / "words.txt").read_text().split()
    vec = embed.vectors("wordllama", words).astype(np.float16)
    VECTORS.parent.mkdir(exist_ok=True)
    np.savez_compressed(VECTORS, words=np.array(words), vectors=vec)
    print(f"wrote {VECTORS.relative_to(ROOT)}: {vec.shape[0]} words x {vec.shape[1]} dims")


if __name__ == "__main__":
    build_vectors()
