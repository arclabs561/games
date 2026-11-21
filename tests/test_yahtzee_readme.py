"""The Yahtzee README and its notes keep every number the README held before the notes."""

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
    assert unexplained("yahtzee", NOTES, numbers(added_text())) == []
