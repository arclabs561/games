"""The Fallout study's committed output (mastermind/results/study-*.txt) and the README's
numbers from it. The study takes minutes, so it is not rerun here; the output is what
`python -m mastermind.fallout3 --study [--attempts 3]` printed."""

from pathlib import Path

HERE = Path(__file__).parent.parent / "mastermind"


def row(attempts: int, level: str, words: int) -> dict[str, float]:
    """One row of the table of percent cracked: the mean for each strategy."""
    lines = (HERE / "results" / f"study-{attempts}.txt").read_text().splitlines()
    assert lines[3].split() == [
        "level",
        "words",
        *"parts exp size entropy minimax".split(),
        "common",
        "random",
        "best",
    ]
    names = ["parts", "exp", "entropy", "minimax", "common", "random", "best"]
    for line in lines[4:]:
        parts = line.split()
        key = [*level.split(), str(words)]
        if parts[: len(key)] == key:
            values = [p for p in parts[len(key) :] if not p.startswith("±")]
            return dict(zip(names, map(float, values), strict=True))
    raise KeyError((level, words))


def test_four_attempts_match_the_readme():
    readme = (HERE / "README.md").read_text()
    r = row(4, "Very Easy", 20)
    assert (round(r["best"], 1), round(r["exp"], 1), round(r["random"], 1)) == (92.5, 88.1, 67.8)
    assert "92.5% ± 0.4" in readme and "88.1% ± 0.5" in readme and "67.8% ± 0.4" in readme
    assert round(r["best"] - r["minimax"], 1) == 5.0 and round(r["best"] - r["exp"], 1) == 4.4


def test_three_attempts_match_the_readme():
    readme = (HERE / "README.md").read_text()
    assert round(row(3, "Very Easy", 20)["best"], 1) == 57.4
    assert round(row(3, "Average", 20)["best"], 1) == 84.4
    assert "57.4% ± 0.4" in readme and "84.4%" in readme
