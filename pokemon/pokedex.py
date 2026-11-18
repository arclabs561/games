"""The Pokémon and forms of each generation, read from ``pokeapi_forms.csv``.

A **form** is one row: a species (``charizard``) or one of its forms (``charizard-mega-x``,
``rotom-heat``). It exists **as of** generation g if it was introduced in g or earlier; Megas
count from Generation 6 on, though later games dropped them. Its **typing** as of g is the
typing it had in g's games: Clefairy is Normal through Generation 5 and Fairy from 6.

``sources.py`` builds the files from PokeAPI; this module only reads them.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from functools import cache
from pathlib import Path

from pokemon.pokemon_types import gen6

HERE = Path(__file__).resolve().parent
FORMS_CSV = HERE / "pokeapi_forms.csv"
STATS_PAST_CSV = HERE / "pokeapi_stats_past.csv"
STATS = ("hp", "attack", "defense", "sp_attack", "sp_defense", "speed")
NEWEST = 9
ORDER = {t: i for i, t in enumerate(gen6)}  # the chart's order: normal, fire, water, ...

Form = dict[str, str]
Typing = tuple[str, ...]


@cache
def forms() -> tuple[Form, ...]:
    """Every row of ``pokeapi_forms.csv``, in PokeAPI's order."""
    with FORMS_CSV.open(encoding="utf-8") as f:
        return tuple(csv.DictReader(f))


@cache
def stats_past() -> tuple[Form, ...]:
    """Every row of ``pokeapi_stats_past.csv``."""
    with STATS_PAST_CSV.open(encoding="utf-8") as f:
        return tuple(csv.DictReader(f))


def canonical(types: Iterable[str]) -> Typing:
    """A typing as a tuple in the chart's order, so Fire/Water and Water/Fire are one."""
    return tuple(sorted(types, key=ORDER.__getitem__))


def typing(form: Form, generation: int = NEWEST) -> Typing | None:
    """The form's typing as of a generation (in the chart's order), or None before it existed."""
    label = form[f"types_{generation}"]
    return canonical(label.split("/")) if label else None


def forms_as_of(generation: int = NEWEST) -> list[Form]:
    """The forms introduced in that generation or earlier."""
    return [f for f in forms() if int(f["generation"]) <= generation]


def existing_typings(generation: int = NEWEST) -> set[Typing]:
    """The typings some form has as of a generation."""
    return {t for f in forms_as_of(generation) if (t := typing(f, generation))}


def base_stats(form: Form, generation: int = NEWEST) -> dict[str, int]:
    """The six base stats as of a generation (2 or later; Generation 1 had one Special stat).

    A row of ``pokeapi_stats_past.csv`` holds through its generation; the earliest such wins.
    """
    if generation < 2:
        raise ValueError("Generation 1 had Special in place of Sp. Atk and Sp. Def")
    stats = {k: int(form[k]) for k in STATS}
    rows = [
        r
        for r in stats_past()
        if r["identifier"] == form["identifier"] and int(r["through_generation"]) >= generation
    ]
    for r in sorted(rows, key=lambda r: -int(r["through_generation"])):
        stats[r["stat"]] = int(r["base_stat"])
    return stats


def total(form: Form, generation: int = NEWEST) -> int:
    """The sum of the six base stats as of a generation."""
    return sum(base_stats(form, generation).values())


def label(t: Sequence[str]) -> str:
    """'water/flying' for a pair, 'water' for one type."""
    return "/".join(t)
