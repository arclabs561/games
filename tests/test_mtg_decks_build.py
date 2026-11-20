"""The 17Lands reader and the build, end to end, on a small synthetic game file."""

import csv
import gzip
import json

import numpy as np
import pytest

from mtg_decks import build, cards, study
from mtg_decks import decks as dk
from mtg_decks import seventeen_lands as sl

# a toy set: three cards in each color, one gold card per pair, and the basics
COLORS = "WUBRG"
TOY = {f"{c}{k}": c for c in COLORS for k in range(3)} | {f"Gold {p}": p for p in cards.PAIRS}


def toy_files(tmp_path, n_drafts=600, seed=0):
    """A game file and a card list shaped like 17Lands' (the columns the reader uses)."""
    rng = np.random.default_rng(seed)
    names = sorted(TOY) + sorted(sl.BASICS)
    game = tmp_path / "games.csv.gz"
    with gzip.open(game, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            sl.META + ["rank"] + [f"deck_{n}" for n in names] + [f"sideboard_{n}" for n in names]
        )
        for d in range(n_drafts):
            pair = cards.PAIRS[d % 10]
            deck = {n: 0 for n in names}
            for n, col in TOY.items():
                if col in pair and len(col) == 1:
                    deck[n] = int(rng.integers(1, 3))
            deck[f"Gold {pair}"] = 1
            deck["Plains"] = 8
            skill = rng.choice([0.48, 0.56, 0.62])
            for build_index in range(1 + (d % 7 == 0)):
                for _ in range(int(rng.integers(1, 5))):
                    won = rng.random() < skill
                    row = [f"draft{d}", build_index, pair[::-1], str(won), f"{skill:.2f}", "gold"]
                    w.writerow(row + [deck[n] for n in names] + [0] * len(names))
    card_list = tmp_path / "cards.csv"
    with open(card_list, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "expansion",
                "name",
                "rarity",
                "color_identity",
                "mana_value",
                "types",
                "is_booster",
            ]
        )
        for i, n in enumerate(sorted(TOY)):
            w.writerow(
                [
                    i,
                    "TOY",
                    n,
                    "common",
                    TOY[n],
                    1 + i % 5,
                    "Creature - Bat Cleric" if i % 2 else "Instant",
                    "True",
                ]
            )
        w.writerow([99, "OLD", "W0", "rare", "U", 9, "Sorcery", "True"])  # another printing
    return game, card_list


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("toy")
    game, card_list = toy_files(tmp)
    out = tmp / "out"
    build.build(game, card_list, "TOY", out, min_decks=5, meta={"set": "TOY"})
    return out


def test_reader_folds_games_into_builds(tmp_path):
    game, _ = toy_files(tmp_path, n_drafts=20)
    d = sl.read_games(game, chunksize=7)  # chunks split builds; they must be joined again
    assert len(d) == 20 + 3  # drafts 0, 7 and 14 have two builds
    assert set(d.group.tolist()) <= set(cards.PAIRS)  # colors put in WUBRG order
    assert np.all(d.wins <= d.games) and d.games.sum() > 20


def test_card_list_prefers_the_sets_printing(tmp_path):
    _, card_list = toy_files(tmp_path, n_drafts=1)
    facts = sl.read_cards(card_list, "TOY")
    assert facts["W0"]["expansion"] == "TOY" and facts["W0"]["color_identity"] == "W"


def test_build_writes_aggregates(built):
    vocab = cards.read_cards(built / "cards.csv")
    assert {c.name for c in vocab} == set(TOY)  # basics are left out
    gold = {c.name: c for c in vocab}["Gold WB"]
    assert gold.pair == "WB" and gold.pair_share == 1.0
    meta = json.loads((built / "source.json").read_text())
    assert sum(v["drafts"] for v in meta["splits"].values()) == 600
    data = study.load(built)
    assert data.n_decks == meta["fit_decks_for_cooccurrence"]
    assert data.sizes.sum() == data.held.sum()  # margins agree, as the null needs
    for (g, o), v in data.targets.items():
        assert v.sum() == pytest.approx(1.0) and g in cards.PAIRS and o in ("won", "lost")


def test_pairs_match_a_direct_count(built):
    data = study.load(built)
    names = data.names
    i, j = names.index("W0"), names.index("Gold WB")
    # W0 is in every deck of the pairs with white, Gold WB only in white-black decks
    assert data.s[i, j] == data.held[j]


def test_heldout_scores_run(built, tmp_path):
    game, card_list = toy_files(tmp_path)
    d = sl.read_games(game)
    dev = d.split("dev")
    res = study.score_heldout(study.load(built), dev, sample=50)
    assert res["decks"] > 0 and 0 <= res["auc"]["transport|pair"]["auc"] <= 1
    # skill drives wins in the toy data, and decks do not: skill's AUC is well above a half
    assert res["auc"]["skill|pair"]["auc"] > 0.55
    # the toy's cards come in identical triples, so similarities tie exactly; rankops
    # works in single precision and may break those ties differently, but only where
    # relevance does not separate them, at lambda = 0
    assert all(v == 0 for lam, v in res["lists_rankops_disagreements"].items() if lam != "0.0")


def test_stratified_auc_by_hand():
    # one stratum: scores 1, 2, 3 with (wins, losses) = (0, 2), (1, 1), (2, 0)
    score = np.array([1.0, 2.0, 3.0])
    pos, neg = np.array([0.0, 1.0, 2.0]), np.array([2.0, 1.0, 0.0])
    # won games at 2 beat 2 losses at 1, tie 1 at 2; won at 3 beat all 3 losses
    expected = (1 * (2 + 0.5) + 2 * 3) / (3 * 3)
    got = study.stratified_auc(score, pos, neg, np.zeros(3, dtype=int), [np.ones(3)])
    assert got[0] == pytest.approx(expected)
    # a second stratum is scored only against itself
    s2 = np.concatenate([score, score + 10])
    got2 = study.stratified_auc(
        s2, np.tile(pos, 2), np.tile(neg, 2), np.repeat([0, 1], 3), [np.ones(6)]
    )
    assert got2[0] == pytest.approx(expected)


def test_split_keeps_builds_together(tmp_path):
    game, _ = toy_files(tmp_path, n_drafts=30)
    d = sl.read_games(game)
    for name in dk.SPLITS:
        part = d.split(name)
        assert all(dk.split_of(str(x)) == name for x in part.draft)
