"""Usage shares from Smogon's chaos files, and the statistics of plan P1.

The downloads are not tested here (smogon.com is not reachable from every machine);
``python -m pokemon.usage build`` checks every name in the files against Showdown, and
``python -m pokemon.usage check`` checks every Showdown species.
"""

import json
from fractions import Fraction
from itertools import product

import numpy as np
import pytest
from scipy.stats import spearmanr

from pokemon import equilibrium as eq
from pokemon.p1 import (
    PRIMARY,
    analyze,
    bootstrap,
    dropped,
    solve,
    spearman,
    spearman_rows,
    support_mass,
)
from pokemon.pokedex import forms, typing
from pokemon.usage import (
    ALIASES,
    RESULTS,
    chaos_url,
    check_names,
    form,
    generation_of,
    identifier,
    months_between,
    parse_pokedex,
    showdown_typings,
    slug,
    summarize,
    total,
    type_shares,
    typing_shares,
)

# Names.


def test_slug():
    assert slug("Mr. Mime") == "mr-mime"
    assert slug("Farfetch’d") == "farfetchd"
    assert slug("Flabébé") == "flabebe"
    assert slug("Type: Null") == "type-null"
    assert slug("Nidoran♀") == "nidoran-f"
    assert slug("Great Tusk") == "great-tusk"


@pytest.mark.parametrize(
    ("name", "ident"),
    [
        ("Great Tusk", "great-tusk"),
        ("Landorus-Therian", "landorus-therian"),
        ("Landorus", "landorus-incarnate"),  # a species with no row of its own: its default
        ("Urshifu-Rapid-Strike", "urshifu-rapid-strike"),
        ("Ogerpon-Wellspring", "ogerpon-wellspring-mask"),
        ("Tauros-Paldea-Aqua", "tauros-paldea-aqua-breed"),
        ("Charizard-Gmax", "charizard"),
        ("Mimikyu-Busted-Totem", "mimikyu-busted"),
        ("Farfetch’d", "farfetchd"),
        ("Nidoran-F", "nidoran-f"),
        ("Pikachu-World", "pikachu-world-cap"),
        ("Not A Pokemon", None),
    ],
)
def test_identifier(name, ident):
    assert identifier(name) == ident


def test_cosmetic_formes_become_their_base():
    assert identifier("Gastrodon-East") is None
    assert identifier("Gastrodon-East", {"Gastrodon-East": "Gastrodon"}) == "gastrodon"


def test_every_alias_is_a_row():
    for ident in ALIASES.values():
        assert form(ident)["identifier"] == ident


def test_generation_1_totals_use_special():
    assert total(form("alakazam"), 1) == 55 + 50 + 45 + 135 + 120
    assert total(form("mewtwo"), 1) == 590
    assert total(form("alakazam"), 9) == 55 + 50 + 45 + 135 + 95 + 120


# Showdown's species data.


POKEDEX = """export const Pokedex = {
\tclefairy: {
\t\tnum: 35,
\t\tname: "Clefairy",
\t\ttypes: ["Fairy"],
\t\tbaseStats: { hp: 70, atk: 45, def: 48, spa: 60, spd: 65, spe: 35 },
\t},
\tgastrodon: {
\t\tnum: 423,
\t\tname: "Gastrodon",
\t\ttypes: ["Water", "Ground"],
\t\tcosmeticFormes: ["Gastrodon-East"],
\t},
\tfarfetchd: {
\t\tnum: 83,
\t\tname: "Farfetch\\u2019d",
\t\ttypes: ["Normal", "Flying"],
\t},
\tburmysandy: {
\t\tisCosmeticForme: true,
\t\tname: "Burmy-Sandy",
\t\tbaseSpecies: "Burmy",
\t},
\tpikachucosplay: {
\t\tnum: -1,
\t\tname: "Pikachu-Fake",
\t\ttypes: ["Electric"],
\t},
};
"""
MOD = """export const Pokedex = {
\tclefairy: {
\t\tinherit: true,
\t\ttypes: ["Normal"],
\t},
};
"""


def test_parse_pokedex():
    dex = parse_pokedex(POKEDEX)
    assert dex["farfetchd"]["name"] == "Farfetch’d"
    assert dex["gastrodon"]["types"] == ("water", "ground")
    assert dex["gastrodon"]["cosmetic"] == ["Gastrodon-East"]
    assert dex["burmysandy"]["base"] == "Burmy" and dex["burmysandy"]["types"] is None
    assert parse_pokedex(MOD)["clefairy"]["types"] == ("normal",)


def test_showdown_typings_follow_the_mods():
    files = {"data/pokedex.ts": POKEDEX, "data/mods/gen5/pokedex.ts": MOD}
    nine, base_of = showdown_typings(files, 9)
    assert nine["Clefairy"] == ("fairy",)
    assert nine["Gastrodon-East"] == ("water", "ground")
    assert base_of["Burmy-Sandy"] == "Burmy" and "Pikachu-Fake" not in nine
    assert showdown_typings(files, 5)[0]["Clefairy"] == ("normal",)
    assert showdown_typings(files, 1)[0]["Clefairy"] == ("normal",)
    assert showdown_typings(files, 6)[0]["Clefairy"] == ("fairy",)


def test_check_names():
    sd = {"Clefairy": ("fairy",), "Farfetch’d": ("normal", "flying")}
    assert check_names(["Clefairy", "Farfetch’d"], 9, sd, {}) == []
    assert check_names(["Clefairy"], 5, sd, {}) == [
        "Clefairy -> clefairy: ('normal',) here, ('fairy',) in Showdown"
    ]
    assert check_names(["Missingno"], 9, sd, {}) == [
        "Missingno: not a species in Showdown's pokedex.ts"
    ]


# Shares.


CHAOS = {
    "info": {"metagame": "gen9ou", "cutoff": 1500, "number of battles": 10},
    "data": {
        "Great Tusk": {"Raw count": 9, "usage": 0.3},
        "Landorus-Therian": {"Raw count": 6, "usage": 0.12},
        "Landorus": {"Raw count": 3, "usage": 0.06},
    },
}


def test_typing_shares_of_three_pokemon():
    shares = typing_shares(CHAOS, 9)
    assert shares.keys() == {("fighting", "ground"), ("ground", "flying")}
    assert shares["fighting", "ground"] == pytest.approx(0.3 / 6)
    assert shares["ground", "flying"] == pytest.approx(0.18 / 6)  # two forms, one typing
    singles = type_shares(shares)
    assert singles["ground"] == pytest.approx(0.48 / 12)
    assert singles["flying"] == pytest.approx(0.18 / 12)
    assert sum(singles.values()) == pytest.approx(sum(shares.values()))


def test_unmapped_names_stop():
    bad = {"data": {**CHAOS["data"], "Fakemon": {"usage": 0.1}}}
    with pytest.raises(ValueError, match="Fakemon"):
        typing_shares(bad, 9)


def test_summarize_pools_months_with_equal_weight():
    one = {"data": {"Great Tusk": {"usage": 0.6}}}
    two = {"data": {"Great Tusk": {"usage": 0.0006}, "Landorus": {"usage": 1.2}}}
    typings, types = summarize([("2025-11", one), ("2025-12", two)], 9, identifier)
    pooled = {r["typing"]: r for r in typings if r["month"] == "pooled"}
    assert pooled["fighting/ground"]["share"] == pytest.approx((0.1 + 0.0001) / 2)
    assert pooled["ground/flying"]["share"] == pytest.approx(0.2 / 2)
    assert pooled["ground/flying"]["pokemon"] == 1
    assert pooled["ground/flying"]["max_total"] == 600
    november = {r["typing"]: r for r in typings if r["month"] == "2025-11"}
    assert november["ground/flying"]["share"] == 0.0  # absent that month
    ground = next(r for r in types if r["month"] == "pooled" and r["type"] == "ground")
    assert ground["pokemon"] == 2 and ground["max_total"] == 600


def test_months_and_urls():
    months = months_between("2025-11..2026-10")
    assert len(months) == 12 and months[1] == "2025-12" and months[2] == "2026-01"
    assert months_between("2026-10") == ["2026-10"]
    assert chaos_url("gen1ou", "2026-10", 1760) == (
        "https://www.smogon.com/stats/2026-10/chaos/gen1ou-1760.json"
    )
    assert generation_of("gen9ou") == 9 and generation_of("gen1ou") == 1


# Correlations and the bootstrap.


def test_spearman_matches_scipy():
    rng = np.random.default_rng(3)
    for _ in range(50):
        x = rng.integers(0, 4, size=12).astype(float)  # many ties
        y = rng.normal(size=12)
        assert spearman(x, y) == pytest.approx(spearmanr(x, y).statistic)
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_bootstrap_against_exact_enumeration():
    """With 4 typings all 4^4 = 256 resamples are equally likely; compare the two laws."""
    model, base, usage = [0.5, 0.0, 0.3, 0.2], [3, 1, 2, 5], [0.4, 0.1, 0.2, 0.3]
    idx = np.array(list(product(range(4), repeat=4)))
    rm = spearman_rows(np.asarray(model)[idx], np.asarray(usage)[idx])
    rb = spearman_rows(np.asarray(base)[idx], np.asarray(usage)[idx])
    exact = (rm - rb)[~np.isnan(rm - rb)]
    b = bootstrap(model, base, usage, resamples=20_000, seed=1)
    rng = np.random.default_rng(1)
    draws = rng.integers(0, 4, size=(20_000, 4))
    sm = spearman_rows(np.asarray(model)[draws], np.asarray(usage)[draws])
    sb = spearman_rows(np.asarray(base)[draws], np.asarray(usage)[draws])
    sampled = (sm - sb)[~np.isnan(sm - sb)]
    assert b["undefined_resamples"] == 20_000 - len(sampled)
    # Same distribution: the largest gap between the two CDFs is small (KS at n = 20,000).
    grid = np.unique(exact)
    gap = max(abs((sampled <= t).mean() - (exact <= t).mean()) for t in grid)
    assert gap < 0.015
    assert sampled.mean() == pytest.approx(exact.mean(), abs=0.01)
    lo, hi = b["delta_interval"]
    assert np.percentile(exact, 2.5) - 0.05 <= lo <= hi <= np.percentile(exact, 97.5) + 0.05
    assert b["delta"] == pytest.approx(spearman(model, usage) - spearman(base, usage))


# The games.


def test_solve_rock_paper_scissors_with_a_dominated_strategy():
    one, z = Fraction(1), Fraction(0)
    # Rock, paper, scissors, and a fourth strategy that loses to everything.
    g = [
        [z, -one, one, one],
        [one, z, -one, one],
        [-one, one, z, one],
        [-one, -one, -one, z],
    ]
    s = solve(g, "row")
    assert s["unique"] and s["weights"] == [Fraction(1, 3)] * 3 + [z]
    out = dropped(["r", "p", "s", "d"], s, [0.1, 0.2, 0.3, 0.4], k=1)
    assert out[0]["typing"] == "d" and out[0]["payoff_against_mix"] == -1
    assert {b["typing"] for b in out[0]["beaten_by"]} == {"r", "p", "s"}
    assert support_mass(s["weights"], [0.1, 0.2, 0.3, 0.4], [1, 1, 1, 7]) == (
        pytest.approx(0.6),
        pytest.approx(0.3),
    )


def test_solve_reports_ranges_when_not_unique():
    """Two copies of rock: their weights can be split any way that sums to 1/3."""
    one, z = Fraction(1), Fraction(0)
    g = [
        [z, z, -one, one],
        [z, z, -one, one],
        [one, one, z, -one],
        [-one, -one, one, z],
    ]
    s = solve(g, "row")
    assert not s["unique"]
    assert s["low"][:2] == [z, z] and s["high"][:2] == [Fraction(1, 3)] * 2
    assert s["weights"] == [Fraction(1, 6), Fraction(1, 6), Fraction(1, 3), Fraction(1, 3)]


def test_attack_game_compares_the_defenders_mix():
    s = solve(eq.attack_matrix(), "column")
    assert s["unique"] and len(s["weights"]) == 18
    assert s["weights"][0] == max(s["weights"])  # Normal leads, as the README says


# The committed results, if present, reproduce.


USAGE = RESULTS / "usage-typings.json"
P1 = RESULTS / "p1.json"


@pytest.mark.skipif(not USAGE.exists(), reason="no usage table yet")
def test_usage_table_is_consistent():
    table = json.loads(USAGE.read_text(encoding="utf-8"))
    primary = [f for f in table["files"] if (f["format"], f["cutoff"]) == ("gen9ou", 1500)]
    assert sorted(f["month"] for f in primary) == months_between("2025-11..2026-10")
    assert all(len(f["sha256"]) == 64 for f in table["files"])
    known = {f["identifier"] for f in forms()}
    for fmt, names in table["names"].items():
        g = generation_of(fmt)
        for name, ident in names.items():
            assert ident in known and typing(form(ident), g) is not None
            assert identifier(name) in (ident, None)  # None: a cosmetic forme, via Showdown
    for f in table["files"]:
        month = [
            r
            for r in table["typings"]
            if (r["format"], r["cutoff"], r["month"]) == (f["format"], f["cutoff"], f["month"])
        ]
        assert sum(r["share"] for r in month) == pytest.approx(f["usage_sum"] / 6)


@pytest.mark.skipif(not (USAGE.exists() and P1.exists()), reason="no results yet")
def test_primary_result_reproduces():
    table = json.loads(USAGE.read_text(encoding="utf-8"))
    committed = json.loads(P1.read_text(encoding="utf-8"))
    assert PRIMARY == {"format": "gen9ou", "cutoff": 1500, "game": "duel", "level": "typing"}
    (r,) = analyze(table, "gen9ou", 1500, "typing", only=lambda g: g == "duel")
    assert r["primary"]
    assert r["vs_availability"]["delta"] == committed["primary"]["delta"]
    assert list(r["vs_availability"]["delta_interval"]) == committed["primary"]["interval"]
