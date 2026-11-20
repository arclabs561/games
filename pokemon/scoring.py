"""Coverage, attack and defense scores of typings, Pokémon and teams; used by pokemon.ipynb.

The vocabulary, in the order the notebook introduces it:

- A **type** is one of the 18 of the Generation 6 chart, and a **typing** is the one or two
  types a Pokémon has, as a tuple in alphabetical order: ``("fairy", "steel")``.
- The **multiplier** of an attack on a typing is the product of the chart's entries for each
  of its types.
- A set of attacking types **covers** the chart if, for every type, one of them is
  super-effective against it.
- A typing's **outgoing** multipliers are the best it can deal to each single type with a move
  of one of its own types; its **incoming** multipliers are what a move of each type deals it.
- Its **attack** and **defense** are the sum of the outgoing multipliers and minus the sum of
  the incoming ones, each rescaled to run from 0 to 1 over the typings that some Pokémon has.

Every number here is exact: multipliers and scores are Fractions.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from fractions import Fraction
from functools import cache
from itertools import chain, combinations
from math import prod

from pokemon import pokedex
from pokemon.pokemon_types import gen6

type Typing = tuple[str, ...]

TYPES: tuple[str, ...] = tuple(gen6)  # the chart's order: normal, fire, water, ...
SUPER, RESISTED, IMMUNE = Fraction(2), Fraction(1, 2), Fraction(0)


def typing(types: Iterable[str]) -> Typing:
    """A typing as the notebook writes it: its types in alphabetical order."""
    return tuple(sorted(types))


CHART: dict[str, dict[str, Fraction]] = {
    a: {d: Fraction(gen6[a].get(d, 1.0)) for d in TYPES} for a in TYPES
}  # every entry written out, 1 included, as a Fraction


def multiplier(attack: str, defender: Sequence[str]) -> Fraction:
    """The multiplier of a move of type `attack` on a Pokémon with the types `defender`."""
    return prod((CHART[attack][d] for d in defender), start=Fraction(1))


@cache
def _hit_by(attack: str, effect: Fraction) -> frozenset[str]:
    return frozenset(d for d in TYPES if CHART[attack][d] == effect)


def hits(attackers: Iterable[str], effect: Fraction) -> set[str]:
    """The single types on which a move of at least one of `attackers` has multiplier `effect`."""
    return set().union(*(_hit_by(a, effect) for a in attackers))


def covers(k: int) -> list[Typing]:
    """Every set of k attacking types that is super-effective against all 18 types."""
    return [c for c in combinations(sorted(TYPES), k) if hits(c, SUPER) == set(TYPES)]


def coverage_score(attackers: Iterable[str]) -> int:
    """An arbitrary score for a set of attacking types: two points for each type it hits
    super-effectively, minus one for each it hits weakly and two for each it cannot touch."""
    a = list(attackers)
    return 2 * len(hits(a, SUPER)) - len(hits(a, RESISTED)) - 2 * len(hits(a, IMMUNE))


def best_coverage_scores(largest: int = 7) -> tuple[int, list[Typing]]:
    """The highest `coverage_score` over sets of 1 to `largest` attacking types, and the
    sets that reach it, smallest first."""
    sets = chain.from_iterable(combinations(sorted(TYPES), k) for k in range(1, largest + 1))
    scored = [(coverage_score(s), s) for s in sets]
    top = max(s for s, _ in scored)
    return top, sorted((s for score, s in scored if score == top), key=lambda s: (len(s), s))


def outgoing(t: Sequence[str]) -> dict[str, Fraction]:
    """The best multiplier a move of one of the typing's own types deals to each single type."""
    return {d: max(multiplier(a, [d]) for a in t) for d in TYPES}


def incoming(t: Sequence[str]) -> dict[str, Fraction]:
    """The multiplier a move of each type deals to the typing."""
    return {a: multiplier(a, t) for a in TYPES}


def conflict(t: Sequence[str]) -> Fraction:
    """How much the typing's two types disagree on attack: summed over the defending types,
    the best multiplier one of them deals minus the worst. Zero for a single type."""
    return sum(
        (max(multiplier(a, [d]) for a in t) - min(multiplier(a, [d]) for a in t) for d in TYPES),
        start=Fraction(0),
    )


def raw_scores(t: Sequence[str]) -> tuple[Fraction, Fraction]:
    """(sum of outgoing multipliers, minus the sum of incoming ones), before rescaling."""
    return sum(outgoing(t).values(), start=Fraction(0)), -sum(
        incoming(t).values(), start=Fraction(0)
    )


def rescaled(values: Sequence[Fraction]) -> list[Fraction]:
    """The values moved and stretched to run from 0 (the least) to 1 (the greatest)."""
    lo, hi = min(values), max(values)
    return [(v - lo) / (hi - lo) for v in values]


def typing_scores(typings: Sequence[Typing]) -> dict[Typing, tuple[Fraction, Fraction]]:
    """Each typing's (attack, defense), each rescaled to [0, 1] over the typings given."""
    attack, defense = zip(*map(raw_scores, typings), strict=True)
    return dict(zip(typings, zip(rescaled(attack), rescaled(defense), strict=True), strict=True))


def percentile(values: Sequence[float | Fraction], x: float | Fraction) -> float:
    """The percentile rank of x among the values, in percent: the mean rank of the values equal
    to x (or where x would go), over the number of values. This is
    ``scipy.stats.percentileofscore`` with its default kind, "rank"."""
    below = sum(v < x for v in values)
    at_most = sum(v <= x for v in values)
    return (below + at_most + (at_most > below)) * 50 / len(values)


def pokemon_table(generation: int = 7) -> list[dict]:
    """One row per form as of a generation: identifier, species, National Pokédex number,
    the generation it was introduced in, typing (alphabetical), base-stat total, and whether it is legendary or mythical, or a Mega."""
    return [
        {
            "name": f["identifier"],
            "species": f["species"],
            "ndex": int(f["ndex"]),
            "generation": int(f["generation"]),
            "typing": typing(pokedex.typing(f, generation) or ()),
            "total": pokedex.total(f, generation),
            "legendary": f["legendary"] == "1" or f["mythical"] == "1",
            "mega": f["mega"] == "1",
        }
        for f in pokedex.forms_as_of(generation)
    ]


def by_typing(table: Iterable[dict]) -> dict[Typing, list[str]]:
    """The names of the forms with each typing, typings in order of first appearance."""
    groups: dict[Typing, list[str]] = {}
    for row in table:
        groups.setdefault(row["typing"], []).append(row["name"])
    return groups


def pokemon_scores(
    table: Sequence[dict], scores: dict[Typing, tuple[Fraction, Fraction]]
) -> dict[str, tuple[float, float]]:
    """Each form's (stat percentile, cover percentile): where its base-stat total ranks among
    all forms, and where its typing's attack plus defense ranks among the typings."""
    totals = [row["total"] for row in table]
    sums = [a + d for a, d in scores.values()]
    return {
        row["name"]: (
            percentile(totals, row["total"]),
            percentile(sums, sum(scores[row["typing"]])),
        )
        for row in table
    }


def random_teams(
    typings: Sequence[Typing], teams: int, seed: int, size: int = 6
) -> list[tuple[Fraction, Fraction]]:
    """The mean raw (attack, defense) of `teams` teams of `size` different typings, drawn
    uniformly from `typings` with ``random.Random(seed)``."""
    rng = random.Random(seed)
    out = []
    for _ in range(teams):
        team = rng.sample(list(typings), size)
        attack, defense = zip(*map(raw_scores, team), strict=True)
        out.append((sum(attack) / size, sum(defense) / size))
    return out


STARTER_BLOCKS = (1, 152, 252, 387, 495, 650, 722)  # National Pokédex numbers of the first
# starter of Generations 1 to 7; each generation's three lines take nine numbers from there.


def starter(ndex: int, final: bool = False) -> bool:
    """Whether a National Pokédex number belongs to one of the starters of Generations 1 to
    7 (the grass, fire and water lines); with `final`, only to their last evolutions."""
    for first in STARTER_BLOCKS:
        if first <= ndex < first + 9:
            return not final or (ndex - first) % 3 == 2
    return False
