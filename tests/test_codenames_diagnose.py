"""Tests for codenames/legality.py, codenames/diagnose.py, codenames/diagnose_records.py and
codenames/diagnose_c3.py. None needs word vectors."""

import json
import random

import numpy as np
import pytest

from codenames import diagnose as dg
from codenames import diagnose_records as records
from codenames import legality as lg
from codenames.diagnose_c3 import read_picks
from codenames.luce import turn_odds_brute
from codenames.players import costs_for

# The pairs of the C11 pilot's first boards, and what the check must call them.
PILOT_PAIRS = [
    ("residency", "residence", "illegal"),
    ("librarian", "library", "illegal"),
    ("winning", "winner", "illegal"),
    ("owned", "owner", "illegal"),
    ("governance", "governor", "illegal"),
    ("governing", "governor", "illegal"),
    ("teaching", "teacher", "illegal"),
    ("actress", "actor", "illegal"),
    ("hiding", "hide", "illegal"),
    ("railway", "railroad", "illegal"),
    ("restroom", "bathroom", "illegal"),
    ("telecom", "telephone", "borderline"),
    ("webcam", "camera", "borderline"),
]


@pytest.mark.parametrize(("clue", "word", "want"), PILOT_PAIRS)
def test_pilot_pairs(clue, word, want):
    assert lg.level(lg.relation(clue, word)) == want


def test_rulebook_examples_agree_except_the_two_substrings_the_old_rule_forbids():
    rows = dg.rulebook_check()
    assert {(x["clue"], x["word"]) for x in rows if not x["agrees"]} == {
        ("land", "england"),
        ("sparrow", "row"),
    }
    for x in rows:
        if not x["agrees"]:
            assert x["relation"] == "substring" and not x["old_rule_legal"]


@pytest.mark.parametrize(
    ("clue", "word", "relation"),
    [
        ("theater", "theater", "same"),
        ("horse", "horseshoe", "substring"),
        ("broken", "break", "inflection"),
        ("teaching", "teacher", "derivation"),
        ("theatre", "theater", "variant"),
        ("snowshoe", "horseshoe", "compound"),
        ("unhorsed", "horseshoe", "compound"),
        ("rails", "railroad", "compound"),
        ("countess", "county", "affix"),
        ("highway", "railway", "compound3"),
        ("telecom", "telephone", "prefix"),
        ("lunar", "moon", None),
        ("episcopal", "bishop", None),
        ("island", "england", None),
        ("broom", "bathroom", None),
    ],
)
def test_relations(clue, word, relation):
    assert lg.relation(clue, word) == relation


def test_strict_legal_is_stricter_than_words_legal():
    from codenames.c4 import boards
    from codenames.words import clue_words, legal

    pool = clue_words()[:2000]
    for b in boards(11010, 3):
        strict = {c for c in pool if lg.strict_legal(c, b.words)}
        old = {c for c in pool if legal(c, b.words)}
        assert strict < old


def test_helpers():
    assert lg.american("theatre") == "theater" and lg.american("travelled") == "traveled"
    assert ("rail", "way") in lg.splits("railway")
    assert lg.common_prefix("telecom", "telephone") == 4
    assert "librar" in lg.affix_stems("librarian") & lg.affix_stems("library")
    assert "win" in lg.bases("winning") and "own" in lg.bases("owned")


def test_every_flagged_pair_is_labeled_and_labels_of_unflagged_pairs_are_unflagged():
    rows = dg.all_rows()
    labels = dg.read_labels()
    summary = dg.label_summary(rows, labels)
    assert summary["unlabeled_flagged_pairs"] == []


def test_committed_report_is_what_the_script_writes():
    result = dg.legality_report()
    assert json.loads((dg.OUT / "legality.json").read_text()) == json.loads(
        json.dumps(result, sort_keys=True)
    )
    assert (dg.OUT / "legality.md").read_text() == dg.render_legality(result)


def test_reanalysis_on_all_boards_is_c10s_committed_result():
    c10 = json.loads((records.RESULTS / "c10.json").read_text())
    t = dg.c10_table(records.with_legality(records.c10_units()))
    assert t["n"] == 300
    want = c10["secondary"]["P_minus_F"]
    for fam in ("strangers", "human", "eat"):
        assert t["d"][fam].mean() == pytest.approx(want[fam]["mean"], abs=1e-3)
    llm = c10["crossguess"]["givers"]["population"]["minus_first_same_guessers"]["mean"]
    assert t["d"]["llm"].mean() == pytest.approx(llm, abs=1e-3)


def test_outcome_split_against_brute_force():
    rng = random.Random(0)
    from codenames.game import Board, deal
    from codenames.words import board_words

    for _ in range(20):
        board = deal(board_words(), rng)
        idx = rng.sample(range(25), 7)  # a small board keeps the brute force fast
        words = tuple(board.words[i] for i in idx)
        kinds = tuple(board.kinds[i] for i in idx)
        if "T" not in kinds:
            continue
        small = Board(words, kinds)
        w = np.array([rng.random() + 0.01 for _ in words])
        count = rng.randint(1, 3)
        o = dg.outcome(w, kinds, count, costs_for(small))
        n = min(count, kinds.count("T"))
        brute = turn_odds_brute(w, kinds, n)
        assert o["all_hit"] == pytest.approx(brute.all_team[n][0])
        assert o["found"] == pytest.approx(brute.expected_team(n)[0])
        assert o["all_hit"] + o["neutral"] + o["opponent"] + o["assassin"] == pytest.approx(1)
        cost = costs_for(small)
        assert o["value"] == pytest.approx(
            o["found"] - cost["O"] * o["opponent"] - cost["A"] * o["assassin"]
        )


def test_two_group():
    a, b = [1.0, 2.0, 3.0], [0.0, 0.0, 1.0, 1.0]
    x = dg.two_group(a, b)
    assert x["mean"] == pytest.approx(1.5) and x["n"] == [3, 4]
    assert x["ci"][0] <= 1.5 <= x["ci"][1]
    assert dg.two_group(a, b) == x  # seeded


def test_share_by_board_counts_each_board_once():
    rows = [{"unit": "1:0", "x": 1}, {"unit": "1:0", "x": 0}, {"unit": "1:1", "x": 1}]
    s = dg.share_by_board(rows, lambda r: r["x"] == 1)
    assert s["mean"] == pytest.approx(0.75) and s["rows"] == 3 and s["count"] == 2


def test_records_round_trip_through_parquet(tmp_path):
    rows = records.c10_units()[:50]
    rows = records.with_legality(rows)
    path = records.write_parquet(rows, tmp_path / "r.parquet")
    rel = records.load([path], experiments=())
    got = rel.fetchall()
    assert len(got) == 50 and len(got[0]) == len(records.FIELDS)
    back = dict(zip(records.FIELDS, got[0], strict=True))
    assert back["clue"] == rows[0]["clue"] and back["words"] == rows[0]["words"]
    assert back["values"] == rows[0]["values"]
    con = rel.query("units", "SELECT count(*) FROM units WHERE legality = 'illegal'")
    assert con.fetchall()[0][0] == sum(r["legality"] == "illegal" for r in rows)


def test_schema_and_record():
    assert len(set(records.FIELDS)) == len(records.SCHEMA)
    r = records.record(experiment="C10", clue="tree")
    assert set(r) == set(records.FIELDS) and r["count"] is None
    with pytest.raises(AssertionError):
        records.record(nonsense=1)


def test_strangers_are_the_models_less_the_proposer():
    from codenames.actions import PROPOSER
    from codenames.study import LOCAL, REMOTE

    assert records.STRANGERS == tuple(m for m in (*LOCAL, *REMOTE) if m != PROPOSER)


def test_every_reader_gives_boards_that_match_their_seed():
    rows = records.committed()
    assert {r["experiment"] for r in rows} == {"C3", "C4", "C6", "C7", "C10"}
    for r in rows[::997]:
        b = records.dealt(r["board_seed"], r["board_index"] + 1)[r["board_index"]]
        assert r["words"] == list(b.words)


def test_c3_picks_file():
    text = (dg.OUT / "c3-picks.txt").read_text()
    picks = read_picks(text)
    assert len(picks) == 14 * 5 * 4 and all(len(v) == 200 for v in picks.values())
    check = json.loads((dg.OUT / "c3-picks.json").read_text())
    assert check["share"] == 1.0 and check["picks"] == 14 * 5 * 4 * 200


def test_slices_and_disagree_pages_are_rendered_from_their_json():
    s = json.loads((dg.OUT / "slices.json").read_text())
    assert (dg.OUT / "slices.md").read_text() == dg.render_slices(s)
    assert set(s["sizes"]) == set(dg.SLICES) and s["label"] == "exploratory"
    d = json.loads((dg.OUT / "disagree.json").read_text())
    assert (dg.OUT / "disagree.md").read_text() == dg.render_disagree(d)
    for x in d["top"][: d["cases"]]:
        assert (dg.OUT / "cases" / f"c10-{x['board']:03d}.md").exists()


def test_slice_means_recover_the_full_c10_result():
    report = json.loads((dg.OUT / "slices.json").read_text())
    c10 = dg.c10_table(records.with_legality(records.c10_units()))
    for name, split in report["tests"].items():
        _, family = name.split(" | ")
        inside, outside = split["in"], split["out"]
        assert inside["n"] + outside["n"] == c10["n"]
        pooled = (inside["n"] * inside["mean"] + outside["n"] * outside["mean"]) / c10["n"]
        # The slice means are published to four decimal places.
        assert pooled == pytest.approx(c10["d"][family].mean(), abs=5.1e-5), name


def test_disagree_ranks_by_the_gap_from_committed_values():
    d = json.loads((dg.OUT / "disagree.json").read_text())
    t = dg.c10_table(records.with_legality(records.c10_units()))
    gap = t["d"]["strangers"] - t["d"]["llm"]
    top = sorted(range(t["n"]), key=lambda i: (-abs(gap[i]), i))[: len(d["top"])]
    assert [x["board"] for x in d["top"]] == top
