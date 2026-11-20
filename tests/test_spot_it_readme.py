"""The Spot It README and its notes keep every number the README held before the notes."""

from tests.readme_moves import lost, unexplained

NOTES = ["README.md", "constructions.md", "extension.md", "chances.md", "proofs.md"]


def test_no_number_lost():
    assert lost("spot_it", NOTES) == []


# 6/55 and 7/55 as decimals, for the symbol just called (tests/test_spot_it.py counts them)
NEW = {"0.109", "0.127"}


def test_no_number_new():
    assert unexplained("spot_it", NOTES, NEW) == []
