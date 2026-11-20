"""The Spot It README and its notes keep every number the README held before the notes."""

from tests.readme_moves import lost, unexplained

NOTES = ["README.md", "constructions.md", "extension.md", "chances.md", "proofs.md"]


def test_no_number_lost():
    assert lost("spot_it", NOTES) == []


def test_no_number_new():
    assert unexplained("spot_it", NOTES, set()) == []
