"""Numbers that a README held before its detail moved into notes, and the check that none was lost.

When a long README is cut down, its detail moves unchanged into notes in the game's
`results/`. `tests/readme_moves/<game>.txt` lists every number the README held before the
move. The tests check two things:

- every one of those numbers still appears in the README or in one of its notes;
- every number in the README and the notes was in the old README, or is listed as new.

The second check catches a number mistyped while the README was rewritten. A number is a
run of digits, with any commas or decimal points between digits and a minus sign (−) in
front if it has one; superscript digits count.

    uv run python -m tests.readme_moves <game> <old README>   # writes the list from a file
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
LISTS = Path(__file__).parent / "readme_moves"
DIGIT = "0-9⁰¹²³⁴⁵⁶⁷⁸⁹"
NUMBER = re.compile(rf"−?[{DIGIT}](?:[{DIGIT}]|[.,](?=[{DIGIT}]))*")


def numbers(text: str) -> set[str]:
    """Every number written in the text."""
    return set(NUMBER.findall(text))


def old_numbers(game: str) -> set[str]:
    return set((LISTS / f"{game}.txt").read_text().split())


def texts(game: str, notes: list[str]) -> str:
    """The README and the named notes of `results/`, run together."""
    folder = ROOT / game
    return "\n".join(
        (folder / name).read_text() for name in ["README.md", *(f"results/{n}" for n in notes)]
    )


def lost(game: str, notes: list[str]) -> list[str]:
    """Numbers of the old README that neither the README nor the notes hold now."""
    return sorted(old_numbers(game) - numbers(texts(game, notes)))


def unexplained(game: str, notes: list[str], new: set[str]) -> list[str]:
    """Numbers in the README or the notes that were not in the old README and are not listed."""
    return sorted(numbers(texts(game, notes)) - old_numbers(game) - new)


def write(game: str, old_readme: Path) -> None:
    LISTS.mkdir(exist_ok=True)
    found = sorted(numbers(old_readme.read_text()))
    (LISTS / f"{game}.txt").write_text("\n".join(found) + "\n")


if __name__ == "__main__":
    write(sys.argv[1], Path(sys.argv[2]))
