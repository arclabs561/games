"""Boundary cases where the playing rules or the strength of the evidence change."""

import gzip
import json
import subprocess
import sys
from math import comb
from pathlib import Path

import exact_play
import report
import verify_certs

RESULTS = json.loads(report.RESULTS.read_text())


def test_brackets_can_change_the_score():
    hand = (0, 0, 0, 0, 1, 1, 1)
    center = (1, 2, 2, 4)
    bit = 1 << exact_play.CENTERS.index(center)
    masks = dict(zip(exact_play.RULES, exact_play.clearing(hand), strict=True))
    assert masks["unrestricted"] & bit
    assert not masks["running"] & bit
    assert not masks["at_most_3"] & bit
    # The four-card expression makes ten, but leaves all four zeros behind.
    assert exact_play.group((1, 1, 1, 4))[exact_play.RULES.index("at_most_4")]
    assert not masks["at_most_4"] & bit


def test_pairs_first_can_strand_a_solvable_hand():
    hand = (3, 3, 4, 7, 8, 8, 8)
    bit = 1 << exact_play.CENTERS.index((1, 2, 6, 6))
    assert exact_play.clearing(hand)[-1] & bit
    assert not exact_play.greedy(hand) & bit


def test_physical_deal_counts_agree_with_the_independent_counters():
    for name, n, full in (("zero", 44, "deals"), ("no_zero", 40, "no_zero")):
        row = RESULTS["exact_play"][name]
        assert row["total"] == comb(n, 7) * comb(n - 7, 4)
        assert row["pairs"] == RESULTS["pairs"][name]["stuck"]
        assert row["addition"] == RESULTS["addition"][name]["stuck"]
        assert row["unrestricted"] == RESULTS[full]["stuck"]
        assert row["brackets_only"] == row["running"] - row["unrestricted"]
        assert row["solvable"] == row["total"] - row["unrestricted"]
        assert 0 < row["greedy_stranded"] < row["solvable"]
    variants = RESULTS["variants"]
    assert variants["sizes"][5]["stuck"] == RESULTS["deals"]["stuck"]
    assert (
        sum(x["deals"] for x in variants["missed"])
        == variants["zero_hand"] - variants["caught_zero"]
    )
    assert variants["division"]["integer"] == variants["division"]["fraction"]


def test_appendix_center_alphabets_characterize_dead_deals():
    for row in RESULTS["variants"]["missed"]:
        hand = tuple(row["hand"])
        allowed = set(row["center"])
        clears = exact_play.clearing(hand)[-1]
        for i, weight in enumerate(exact_play.pattern_weights(hand)):
            if weight:
                assert bool(clears >> i & 1) == (not set(exact_play.CENTERS[i]) <= allowed)


def test_whole_deck_coverage_includes_every_legal_large_multiset():
    deck = RESULTS["whole_deck"]
    # Each of ten digits occurs zero to four times. Count multisets by hand size.
    sizes = [1]
    for _ in range(10):
        sizes = [sum(sizes[max(0, n - 4) : n + 1]) for n in range(len(sizes) + 4)]
    assert deck["max_cards"] == len(sizes) - 1
    for row in deck["coverage"]:
        assert row["multisets"] == sum(sizes[12:])
        assert row["made"] + row["blocked"] == row["multisets"] * (row["last"] - row["first"] + 1)
    assert set(deck["thresholds"]) == {str(t) for t in range(1, 132)}
    first = deck["first_above_nine"]
    assert first["target"] == 131 and first["threshold"] == 10 and first["below"] == 1
    assert max(deck["thresholds"][str(t)] for t in range(1, 131)) == 9
    assert not exact_play.whole(tuple(first["example"]), 131)


def test_optimized_certificate_checker_rejects_bad_archives(tmp_path):
    hand = [1, 6, 6, 7, 7, 7, 7]
    good = {"cards": hand, "expr": "1+7+7+7-6-6", "uses": "subset"}
    for records, expected in (
        ([], "missing certificate"),
        ([good, good], "duplicate case"),
        ([dict(good, expr="7")], "does not make ten"),
        ([dict(good, expr="10")], "wrong cards"),
        ([dict(good, uses="all")], "wrong cards"),
        ([dict(good, expr="5+True*5")], "illegal expression"),
        ([dict(good, expr="10//1")], "illegal expression"),
    ):
        path = tmp_path / "bad.jsonl.gz"
        with gzip.open(path, "wt") as f:
            for record in records:
                print(json.dumps(record), file=f)
        result = subprocess.run(
            [sys.executable, "-O", str(Path(verify_certs.__file__)), str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0 and expected in result.stderr
