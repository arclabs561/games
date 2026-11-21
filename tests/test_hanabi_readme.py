"""The Hanabi README and its notes keep every number the README held before the notes."""

import json
from pathlib import Path

from tests.readme_moves import lost, unexplained

HANABI = Path(__file__).resolve().parents[1] / "hanabi"

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


def test_no_number_new():
    # The opening counts the deals on which two players finished on 23.
    records = (HANABI / "results" / "humans-games.jsonl").read_text().splitlines()
    games = [
        game for game in map(json.loads, records) if game["players"] == 2 and game["score"] == 23
    ]
    perfect = sum(game["ceiling_low"] == 25 for game in games)
    assert all(game["ceiling_low"] == game["ceiling_high"] for game in games)
    assert f"on {perfect} of those {len(games)} deals" in (HANABI / "README.md").read_text()
    assert unexplained("hanabi", NOTES, {str(perfect), str(len(games))}) == []
