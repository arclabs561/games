"""The forms and typings of each generation, read from pokeapi_forms.csv."""

import pytest

from pokemon.equilibrium import TYPINGS, existing_typings
from pokemon.pokedex import base_stats, canonical, forms, forms_as_of, total, typing


def form(identifier):
    return next(f for f in forms() if f["identifier"] == identifier)


def labels(typings):
    return sorted("/".join(t) for t in typings)


def test_typings_change_by_generation():
    assert typing(form("clefairy"), 5) == ("normal",)
    assert typing(form("clefairy"), 6) == ("fairy",)
    assert typing(form("magnemite"), 1) == ("electric",)
    assert typing(form("magnemite"), 2) == ("electric", "steel")
    assert typing(form("rotom-heat"), 4) == ("electric", "ghost")
    assert typing(form("rotom-heat"), 5) == ("fire", "electric")
    assert typing(form("decidueye"), 6) is None
    assert canonical(["steel", "fire"]) == ("fire", "steel")


def test_counts_by_generation():
    forms_count = [len(forms_as_of(g)) for g in range(1, 10)]
    typings_count = [len(existing_typings(g)) for g in range(1, 10)]
    assert forms_count == [151, 251, 392, 508, 673, 811, 955, 1106, 1303]
    assert typings_count == [33, 55, 75, 88, 110, 133, 146, 155, 162]


# The 25 of the 171 typings that no form had as of Generation 7. The other 146 are exactly the
# typings of the Kaggle file pokemon_gen7.csv that this data replaced (compared when it was
# replaced), so the results that used that file carry over.
MISSING_AS_OF_7 = [
    "bug/dark", "bug/dragon", "dark/fairy", "electric/dark", "electric/fighting",
    "electric/poison", "fighting/fairy", "fighting/ground", "fire/fairy", "fire/grass",
    "fire/ice", "ground/fairy", "ice/bug", "ice/poison", "normal/bug", "normal/ghost",
    "normal/ice", "normal/poison", "normal/rock", "normal/steel", "poison/fairy",
    "poison/psychic", "poison/steel", "psychic/bug", "rock/ghost",
]  # fmt: skip


def test_generation_7_typings_match_the_old_file():
    assert labels(set(TYPINGS) - set(existing_typings(7))) == MISSING_AS_OF_7


def test_typings_since_generation_7():
    new = set(existing_typings(9)) - set(existing_typings(7))
    assert labels(new) == [
        "bug/dark", "dark/fairy", "electric/dark", "electric/fighting", "electric/poison",
        "fighting/fairy", "fighting/ground", "fire/grass", "fire/ice", "ice/bug",
        "normal/ghost", "normal/poison", "poison/fairy", "poison/psychic", "poison/steel",
        "psychic/bug",
    ]  # fmt: skip
    assert labels(set(TYPINGS) - set(existing_typings(9))) == [
        "bug/dragon", "fire/fairy", "ground/fairy", "ice/poison", "normal/bug",
        "normal/ice", "normal/rock", "normal/steel", "rock/ghost",
    ]  # fmt: skip
    # No typing has gone out of use: retyped forms kept or gained company.
    assert set(existing_typings(7)) <= set(existing_typings(9))


def test_base_stats_by_generation():
    aegislash = form("aegislash-shield")
    assert base_stats(aegislash)["defense"] == 140
    assert base_stats(aegislash, 7)["defense"] == 150
    assert total(form("pikachu"), 5) == total(form("pikachu")) - 20  # +10 Def, +10 Sp. Def in 6
    with pytest.raises(ValueError):
        base_stats(aegislash, 1)
