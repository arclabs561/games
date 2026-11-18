"""The type charts by generation, and the code that builds the data files from PokeAPI.

The download itself is not tested here (it needs the network); ``python -m pokemon.sources
--check`` rebuilds the files from the pinned commits and fails if they differ.
"""

import csv

import pytest

from pokemon.pokemon_types import chart, gen6
from pokemon.sources import (
    FORM_COLUMNS,
    FORMS_CSV,
    STATS_PAST_CSV,
    build_forms,
    full,
    is_left_out,
    parse_showdown,
    pokeapi_chart,
    same_chart,
    showdown_charts,
)


def test_chart_sizes():
    assert [len(chart(g)) for g in range(1, 10)] == [15, 17, 17, 17, 17, 18, 18, 18, 18]
    assert "steel" not in chart(1) and "dark" not in chart(1) and "fairy" not in chart(5)
    with pytest.raises(ValueError):
        chart(10)


def test_chart_from_generation_6_on_is_gen6():
    for g in (6, 7, 8, 9):
        assert same_chart(chart(g), gen6)


def test_generation_1_differences():
    one = full(chart(1))
    assert one["bug"]["poison"] == 2 and one["poison"]["bug"] == 2
    assert one["ghost"]["psychic"] == 0  # a programming error in the games, kept by every source
    assert one["ice"]["fire"] == 1
    # Everything else matches the Generation 6 chart on the 15 types.
    six = full(gen6)
    changed = {(a, d) for a in one for d in one if one[a][d] != six[a][d]}
    assert changed == {("bug", "poison"), ("poison", "bug"), ("ghost", "psychic"), ("ice", "fire")}


def test_generations_2_to_5_differences():
    two, six = full(chart(2)), full(gen6)
    assert two["ghost"]["steel"] == two["dark"]["steel"] == 0.5
    changed = {(a, d) for a in two for d in two if two[a][d] != six[a][d]}
    assert changed == {("ghost", "steel"), ("dark", "steel")}


SHOWDOWN_SNIPPET = """export const TypeChart = {
\tbug: {
\t\tdamageTaken: {
\t\t\tsandstorm: 3,
\t\t\tBug: 0,
\t\t\tFire: 1,
\t\t\tGhost: 3,
\t\t},
\t},
\tdark: {
\t\tinherit: true,
\t\tisNonstandard: 'Future',
\t},
\tfire: {
\t\tinherit: true,
\t\tdamageTaken: {
\t\t\tBug: 2,
\t\t\tFire: 2,
\t\t},
\t},
};
"""


def test_parse_showdown():
    # damageTaken is the defender's view: 0 neutral, 1 weak, 2 resists, 3 immune.
    assert parse_showdown(SHOWDOWN_SNIPPET) == {
        "bug": {"bug": 1.0, "fire": 2.0, "ghost": 0.0},
        "dark": None,
        "fire": {"bug": 0.5, "fire": 0.5},
    }


def test_showdown_inheritance():
    newest = SHOWDOWN_SNIPPET  # dark is marked as not yet existing, so it is left out
    older = "\tbug: {\n\t\tdamageTaken: {\n\t\t\tBug: 0,\n\t\t\tFire: 0,\n\t\t},\n\t},\n"
    files = {"data/typechart.ts": newest.encode(), "data/mods/gen4/typechart.ts": older.encode()}
    charts = showdown_charts(files)
    # Rows are attackers: fire attacking bug is 2 from Generation 5 on, 1 in 4 and before.
    assert charts[9]["fire"]["bug"] == charts[5]["fire"]["bug"] == 2.0
    assert charts[4]["fire"]["bug"] == charts[1]["fire"]["bug"] == 1.0
    assert charts[9]["bug"]["fire"] == 0.5


def table(header, *rows):
    return [dict(zip(header.split(","), r.split(","), strict=True)) for r in rows]


def tiny_pokeapi():
    """Three Pokémon: one retyped in Generation 2, a Gigantamax row, and a form with no default."""
    return {
        "types": table("id,identifier,generation_id", "1,normal,1", "9,steel,2", "13,electric,1"),
        "stats": table(
            "id,identifier",
            "1,hp",
            "2,attack",
            "3,defense",
            "4,special-attack",
            "5,special-defense",
            "6,speed",
            "7,accuracy",
        ),
        "version_groups": table("id,generation_id", "1,1", "3,2", "25,9"),
        "pokemon_species": table(
            "id,identifier,is_legendary,is_mythical", "81,magnemite,0,0", "1007,koraidon,1,0"
        ),
        "pokemon": table(
            "id,identifier,species_id",
            "81,magnemite,81",
            "10500,magnemite-gmax,81",
            "10264,koraidon-limited-build,1007",
        ),
        "pokemon_forms": table(
            "pokemon_id,introduced_in_version_group_id,is_default,is_mega",
            "81,1,1,0",
            "10500,25,1,0",
            "10264,25,0,0",
        ),
        "pokemon_types": table(
            "pokemon_id,type_id,slot", "81,13,1", "81,9,2", "10500,13,1", "10264,1,1"
        ),
        "pokemon_types_past": table("pokemon_id,generation_id,type_id,slot", "81,1,13,1"),
        "pokemon_stats": table(
            "pokemon_id,stat_id,base_stat",
            *(f"{p},{s},{10 * s}" for p in (81, 10500, 10264) for s in range(1, 8)),
        ),
        "pokemon_stats_past": table(
            "pokemon_id,generation_id,stat_id,base_stat", "81,5,3,55", "10500,8,3,1"
        ),
    }


def test_build_forms():
    forms, past = build_forms(tiny_pokeapi())
    assert [f["identifier"] for f in forms] == ["magnemite", "koraidon-limited-build"]
    magnemite, koraidon = forms
    assert magnemite["types_1"] == "electric" and magnemite["types_2"] == "electric/steel"
    assert magnemite["types_9"] == "electric/steel" and magnemite["generation"] == "1"
    assert koraidon["generation"] == "9" and koraidon["types_8"] == ""
    assert koraidon["legendary"] == "1" and koraidon["types_9"] == "normal"
    assert magnemite["defense"] == "30" and "accuracy" not in magnemite
    assert past == [
        {"identifier": "magnemite", "through_generation": "5", "stat": "defense", "base_stat": "55"}
    ]


def test_left_out():
    assert is_left_out("charizard-gmax") and is_left_out("raticate-totem-alola")
    assert is_left_out("pikachu-starter") and not is_left_out("pikachu-alola-cap")


def test_pokeapi_chart():
    t = {
        "types": table("id,identifier,generation_id", "8,ghost,1", "9,steel,2", "14,psychic,1"),
        "type_efficacy": table(
            "damage_type_id,target_type_id,damage_factor",
            "8,8,200",
            "8,9,100",
            "8,14,200",
            "9,9,50",
            "14,9,50",
        ),
        "type_efficacy_past": table(
            "damage_type_id,target_type_id,damage_factor,generation_id", "8,14,0,1", "8,9,50,5"
        ),
    }
    assert pokeapi_chart(t, 1) == {"ghost": {"ghost": 2.0, "psychic": 0.0}, "psychic": {}}
    assert pokeapi_chart(t, 3)["ghost"] == {"ghost": 2.0, "steel": 0.5, "psychic": 2.0}
    assert pokeapi_chart(t, 6)["ghost"]["steel"] == 1.0


def read(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_committed_files():
    forms = read(FORMS_CSV)
    assert list(forms[0]) == list(FORM_COLUMNS)
    assert len(forms) == 1303 and len(read(STATS_PAST_CSV)) == 235
    assert len({f["identifier"] for f in forms}) == 1303
    assert len({f["species"] for f in forms}) == 1025
    assert not any(is_left_out(f["identifier"]) for f in forms)
    for f in forms:
        gen = int(f["generation"])
        assert all((f[f"types_{g}"] == "") == (g < gen) for g in range(1, 10)), f
        types = f["types_9"].split("/")
        assert 1 <= len(types) <= 2 and all(t in gen6 for t in types)
