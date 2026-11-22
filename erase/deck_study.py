"""Verify target thresholds through 40 cards. Requires a C++20 compiler."""

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


def study(published):
    source = Path(__file__).with_name("deck_targets.cpp")
    thresholds = {}
    coverage = []
    first_above_nine = None
    with TemporaryDirectory() as tmp:
        exe = Path(tmp) / "deck-targets"
        subprocess.run(["c++", "-O2", "-std=c++20", str(source), "-o", str(exe)], check=True)
        for first in range(1, 132, 30):
            last = min(first + 29, 131)
            output = subprocess.check_output([exe, "--range", str(first), str(last)], text=True)
            lines = output.splitlines()
            tag, multisets, made, blocked, unresolved = lines.pop().split()
            if (
                tag != "coverage"
                or int(unresolved)
                or int(made) + int(blocked) != (last - first + 1) * int(multisets)
            ):
                raise ValueError(f"incomplete coverage for targets {first} through {last}")
            coverage.append(
                {
                    "first": first,
                    "last": last,
                    "multisets": int(multisets),
                    "made": int(made),
                    "blocked": int(blocked),
                }
            )
            for expected, line in zip(range(first, last + 1), lines, strict=True):
                header, moduli, exceptions, example = (
                    list(map(int, p.split())) for p in line.split("|")
                )
                t, threshold, positive = header
                if t != expected:
                    raise ValueError(f"expected target {expected}, got {t}")
                row = {
                    "threshold": threshold,
                    "moduli": moduli,
                    "exceptions_by_size": exceptions,
                    "below": exceptions[threshold - 2],
                    "example": example,
                    "made_from_threshold": positive,
                }
                if str(t) in published and published[str(t)] != row:
                    raise ValueError(f"independent target study disagrees for {t}")
                thresholds[str(t)] = threshold
                if threshold > 9 and first_above_nine is None:
                    first_above_nine = {"target": t, **row}
    return {
        "max_cards": 40,
        "thresholds": thresholds,
        "coverage": coverage,
        "first_above_nine": first_above_nine,
    }
