"""The Hanabi README and its notes keep every number the README held before the notes."""

from pathlib import Path

from tests.readme_moves import lost, numbers, unexplained

# Notes added after the move; numbers the README quotes from them count as explained.
LATER = ["hat-gap.md", "six-suits.md"]

NOTES = [
    "README.md",
    "information.md",
    "open-game.md",
    "h1.md",
    "humans.md",
    "h2.md",
    "h4.md",
    "checks.md",
    "open-questions.md",
    "usage.md",
]


def test_no_number_lost():
    assert lost("hanabi", NOTES) == []


def later_numbers() -> set[str]:
    folder = Path(__file__).parent.parent / "hanabi" / "results"
    return set().union(*(numbers((folder / n).read_text()) for n in LATER if (folder / n).exists()))


def test_no_number_new():
    assert unexplained("hanabi", NOTES, later_numbers()) == []
