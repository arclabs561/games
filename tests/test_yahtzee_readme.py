"""The Yahtzee README and its notes keep every number the README held before the notes."""

import csv
from pathlib import Path

from tests.readme_moves import lost, numbers, unexplained
from tests.test_yahtzee_target import added_text

NOTES = [
    "README.md",
    "optimal.md",
    "variants.md",
    "openings.md",
    "card-search.md",
    "advice.md",
    "method.md",
]


def test_no_number_lost():
    assert lost("yahtzee", NOTES) == []


def test_no_number_new():
    # The sections added later are checked against their results in test_yahtzee_target.
    root = Path(__file__).resolve().parents[1] / "yahtzee"
    with (root / "openings.csv").open() as f:
        paired = sum(int(r["ways"]) for r in csv.DictReader(f) if len(set(r["roll"])) < 5)
    assert f"{paired:,} contain a repeated face" in (root / "README.md").read_text()
    assert unexplained("yahtzee", NOTES, numbers(added_text()) | {f"{paired:,}"}) == []
