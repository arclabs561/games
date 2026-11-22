"""Bound the effect of short replies from the published C11 turn scores.

python -m codenames.c11_bounds
"""

import json
from itertools import combinations

import numpy as np

from codenames import c4
from codenames.c11 import GUESSERS, GUESSES, LISTS, TEST, read_cell, read_guesses
from codenames.game import Board


def short_reply_values(board: Board, count: int) -> dict[tuple[int, float], set[int]]:
    """For each padded (found, value), scores possible when the reply ended on a team word
    before reaching count. Naming the same team words in any order leaves the same suffix.
    All other replies keep the historical value.
    """
    team = [word for word in board.words if board.kind(word) == "T"]
    options: dict[tuple[int, float], set[int]] = {}
    for k in range(1, count):
        for named in combinations(team, k):
            padded = list(named) + [word for word in board.words if word not in named]
            historical = c4.play_turn(board, padded, count)
            options.setdefault(historical, set()).add(k)
    return options


def primary_bounds(rows: list[dict], guesses: dict) -> dict:
    """Sharp bounds on the two sample means, given the historical first-send test cells."""
    from codenames.c4score import rounded

    total = np.zeros(3)  # lower endpoint, historical estimate, upper endpoint
    potion = 0.0
    kept = variable = shared = 0
    for i, (row, board) in enumerate(zip(rows, c4.boards(TEST, len(rows)), strict=True)):
        if (row["seed"], row["board"]) != (TEST, i):
            raise ValueError("expected consecutive test boards in deal order")
        strict = row["strict"]
        cells = {
            (arm, model): read_cell(guesses[arm, model, TEST, 0][i])
            for model in GUESSERS
            for arm in ("P", "F")
        }
        if any(cell is None for cell in cells.values()):
            continue
        kept += 1
        potion += strict["values"]["P"]["potion"] - strict["values"]["F"]["potion"]
        if strict["P"] == strict["F"]:
            if any(cells["P", model] != cells["F", model] for model in GUESSERS):
                raise ValueError("a shared prompt has different historical scores")
            shared += 1
            continue

        counts = {arm: int(strict[arm].rsplit("/", 1)[1]) for arm in ("P", "F")}
        short = {n: short_reply_values(board, n) for n in set(counts.values())}
        board_diff = np.zeros(3)
        for model in GUESSERS:
            for arm in ("P", "F"):
                observed = cells[arm, model]
                assert observed is not None
                found, value, _ = observed
                # A reply naming every word in its padded order always preserves the old score.
                options = {value, *short[counts[arm]].get((found, value), ())}
                low, high = min(options), max(options)
                if arm == "P":
                    board_diff += [low, value, high]
                else:
                    board_diff -= [high, value, low]
        total += board_diff / len(GUESSERS)
        variable += int(board_diff[0] != board_diff[2])
    if kept == 0:
        raise ValueError("no paired test boards")
    names = ("lower", "historical", "upper")
    return rounded(
        {
            "kept": kept,
            "variable_boards": variable,
            "shared_prompt_boards": shared,
            "h_a1": dict(zip(names, total / kept, strict=True)),
            "h_a2": dict(zip(names, (total - potion) / kept, strict=True)),
        },
        6,
    )


if __name__ == "__main__":
    boards = [row for row in json.loads(LISTS.read_text())["boards"] if row["seed"] == TEST]
    print(json.dumps(primary_bounds(boards, read_guesses(GUESSES.read_text())), indent=2))
