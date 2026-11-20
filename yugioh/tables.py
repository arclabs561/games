"""Write the tables that the README quotes to yugioh/results/.

    uv run python -m yugioh.tables              # exact tables, under a second
    uv run python -m yugioh.tables --simulate   # also the seeded simulation, 1 to 2 minutes

Every number in tables.md is exact. simulation.md holds the one simulation, with its seed,
sample size and 95% interval, beside the exact values it estimates.
"""

import argparse
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

from yugioh.draws import Engine, p_engine, simulate_engine, starter_equivalent
from yugioh.odds import (
    FIRST,
    NEEDS,
    SECOND,
    Deck,
    by_role,
    fewest,
    named,
    p_at_least_one,
    p_hand,
    p_named,
    simulate,
    starter,
    starter_and_extender,
    two_different_starters,
    two_engine,
    wilson,
)

RESULTS = Path(__file__).parent / "results"
TARGETS = (Fraction(8, 10), Fraction(9, 10), Fraction(95, 100))
EXAMPLE = Deck(40, starters=12, extenders=9, bricks=3)  # the example deck of the README
SEED, TRIALS, ENGINE_TRIALS = 1, 1_000_000, 200_000

# Values printed by public calculators, read on 2026-11-19: (source, question, their value in
# percent as printed, deck size, copies, cards drawn). Each is compared with the exact value.
DECKLAB = "ygo-decklab.com/en/guide/deck-probabilities"
CODINGACE = "codingace.net/general/yugioh_hand.html"
LUMDERA = "lumdera.com/yugioh-probability-calculator"
CARDGAME = "cardgamecalculator.com/yugioh"
DECKLAB_ROW = "12.5 23.7 33.8 42.7 50.7 57.7 63.9 69.4 74.2 78.3 82.0 85.1 87.7 90.0 91.9"
PUBLISHED: list[tuple[str, str, str, int, int, int]] = (
    [(DECKLAB, "at least 1", v, 40, k, 5) for k, v in enumerate(DECKLAB_ROW.split(), start=1)]
    + [(CODINGACE, "at least 1", v, 40, k, 5)
       for k, v in ((3, "33.7551"), (6, "57.7124"), (9, "74.1780"))]
    + [(LUMDERA, "at least 1", v, 40, k, 5) for k, v in
       ((1, "12.5"), (2, "24.1"), (3, "33.3"), (6, "57.7"), (9, "77.8"), (12, "88.7"))]
    + [(LUMDERA, "at least 1", v, 40, k, 6) for k, v in
       ((1, "14.8"), (2, "28.4"), (3, "39.4"), (6, "64.7"), (9, "84.2"), (12, "92.3"))]
    + [(LUMDERA, "at least 1", "30.2", 45, 3, 5)]
    + [(CARDGAME, "at least 1", "33.76", 40, 3, 5), (CARDGAME, "exactly 0", "66.24", 40, 3, 5)]
)  # fmt: skip


def pct(p: Fraction, places: int = 2) -> str:
    """A chance as a percentage."""
    return f"{float(p) * 100:.{places}f}%"


def table(header: list[str], rows: Sequence[Sequence[object]]) -> list[str]:
    """A Markdown table, first column left-aligned and the rest right-aligned."""
    lines = ["| " + " | ".join(header) + " |", "|---" + "|---:" * (len(header) - 1) + "|"]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return lines + [""]


def at_least_one() -> list[str]:
    """P(at least one starter) in 40 cards, by number of starters."""
    rows = []
    for k in range(0, 21):
        a, b = p_at_least_one(40, k, FIRST), p_at_least_one(40, k, SECOND)
        rows.append([k, pct(a), pct(b), f"{a.numerator}/{a.denominator}"])
    rough = 1 - (1 - Fraction(12, 40)) ** FIRST
    note = [
        "A rough check: drawing with replacement, 12 starters in 40 give 1 − (28/40)^5 = "
        f"{pct(rough)} going first, against the exact {pct(p_at_least_one(40, 12, FIRST))}.",
        "Without replacement each miss leaves fewer non-starters, so the exact chance is higher.",
        "",
    ]
    return (
        ["## At least one starter in 40 cards", ""]
        + table(["Starters", "Going first (5)", "Going second (6)", "Going first, exact"], rows)
        + note
    )


def example() -> list[str]:
    """Every need for the example deck, going first and second, with the exact fractions."""
    rows = []
    for label, need in NEEDS.items():
        a, b = p_hand(EXAMPLE, FIRST, need), p_hand(EXAMPLE, SECOND, need)
        rows.append([label, pct(a), pct(b), f"{a.numerator}/{a.denominator}",
                     f"{b.numerator}/{b.denominator}"])  # fmt: skip
    head = ["Need", "Going first", "Going second", "First, exact", "Second, exact"]
    return ["## The example deck: 40 cards, 12 starters, 9 extenders, 3 bricks", ""] + table(
        head, rows
    )


def starters_needed() -> list[str]:
    """Fewest starters for 80, 90 and 95% at each deck size."""
    rows = []
    for size in range(40, 61):
        rows.append(
            [size]
            + [fewest(t, size, FIRST) for t in TARGETS]
            + [fewest(t, size, SECOND) for t in TARGETS]
        )
    head = ["Deck"] + [f"First, {float(t):.0%}" for t in TARGETS]
    head += [f"Second, {float(t):.0%}" for t in TARGETS]
    edge = []
    for t in TARGETS:
        for n, label in ((FIRST, "first"), (SECOND, "second")):
            k = fewest(t, 40, n)
            assert k is not None
            edge.append(
                f"- {float(t):.0%} going {label}: {k} starters give {pct(p_at_least_one(40, k, n), 3)}, "
                f"{k - 1} give {pct(p_at_least_one(40, k - 1, n), 3)}."
            )
    return (
        ["## Fewest starters for at least one in hand", ""]
        + table(head, rows)
        + ["In 40 cards, the chance at the threshold and one starter below it:", ""]
        + edge
        + [""]
    )


def deck_size() -> list[str]:
    """What going from 40 to more cards costs, with 12 starters."""
    rows = []
    base = p_at_least_one(40, 12, FIRST), p_at_least_one(40, 12, SECOND)
    for size in (40, 41, 42, 43, 44, 45, 50, 55, 60):
        a, b = p_at_least_one(size, 12, FIRST), p_at_least_one(size, 12, SECOND)
        keep = next(k for k in range(size) if p_at_least_one(size, k, FIRST) >= base[0])
        rows.append(
            [size, pct(a), f"{float(a - base[0]) * 100:+.2f}", pct(b),
             f"{float(b - base[1]) * 100:+.2f}", keep]
        )  # fmt: skip
    head = ["Deck", "First", "Change (points)", "Second", "Change (points)"]
    head += ["Starters to match 40 cards, first"]
    return ["## Twelve starters in a larger deck", ""] + table(head, rows)


def two_needs() -> list[str]:
    """A starter and an extender, and a starter with another engine card, in 40 cards."""
    out = ["## A starter and an extender in 40 cards", ""]
    for n, label in ((FIRST, "going first"), (SECOND, "going second")):
        rows = []
        for s in (6, 9, 12, 15):
            row: list[object] = [s]
            for e in (3, 6, 9, 12):
                row.append(pct(p_hand(Deck(40, s, e), n, starter_and_extender), 1))
            rows.append(row)
        out += [f"At least one of each, {label}; rows are starters, columns extenders:", ""]
        out += table(["Starters", "3", "6", "9", "12"], rows)
    rows = []
    for s in (6, 9, 12, 15):
        row = [s]
        for e in (3, 6, 9, 12):
            row.append(pct(p_hand(Deck(40, s, e), FIRST, two_engine), 1))
        rows.append(row)
    out += ["A starter and another engine card (a second starter counts), going first:", ""]
    return out + table(["Starters", "3", "6", "9", "12"], rows)


def bricks() -> list[str]:
    """No brick in the hand, by number of bricks."""
    rows = []
    for b in range(0, 9):
        a, c = (p_hand(Deck(40, bricks=b), n, NEEDS["no brick"]) for n in (FIRST, SECOND))
        rows.append([b, pct(a), pct(c)])
    return ["## No brick in 40 cards", ""] + table(["Bricks", "Going first", "Going second"], rows)


def copies() -> list[str]:
    """Copies do not matter for role needs; they do for two different starters."""
    rows = []
    for total, copy in ((12, 3), (12, 2), (12, 1), (9, 3), (9, 1)):
        names = named(Deck(40, starters=total), copies=copy)
        one = p_named(names, FIRST, by_role(starter))
        two = p_named(names, FIRST, two_different_starters)
        rows.append([f"{total} as {total // copy} × {copy}", pct(one), pct(two)])
    head = ["Starters", "At least one starter", "Two different starters"]
    return ["## Copies, going first in 40 cards", ""] + table(head, rows)


ENGINES = (
    ("none", 40, Engine()),
    ("1 draw-2", 40, Engine(draw2=1)),
    ("2 draw-2", 40, Engine(draw2=2)),
    ("3 draw-2", 40, Engine(draw2=3)),
    ("3 look-at-6", 40, Engine(lookers=3)),
    ("3 draw-2, 3 look-at-6", 40, Engine(draw2=3, lookers=3)),
    ("3 draw-1, added (43 cards)", 43, Engine(cantrips=3)),
    ("3 draw-2, added (43 cards)", 43, Engine(draw2=3)),
    ("3 look-at-6, added (43 cards)", 43, Engine(lookers=3)),
)


def engines() -> list[str]:
    """Draw and search cards in the example deck: chances and worth in starters."""
    rows = []
    for label, size, eng in ENGINES:
        deck = Deck(size, EXAMPLE.starters, EXAMPLE.extenders, EXAMPLE.bricks)
        one = p_engine(deck, eng, FIRST, starter)
        pair = p_engine(deck, eng, FIRST, starter_and_extender)
        second = p_engine(deck, eng, SECOND, starter)
        worth = starter_equivalent(Deck(40, 12, 9, 3), eng, FIRST, starter) if size == 40 else ""
        rows.append(
            [label, pct(one), pct(second), pct(pair), f"{float(worth):.2f}" if worth != "" else ""]
        )
    head = ["Engine cards", "Starter, first", "Starter, second", "Starter and extender, first"]
    head += ["Worth in starters, first"]
    note = [
        "The deck has 12 starters, 9 extenders and 3 bricks; engine cards replace other cards",
        "unless marked added. Worth in starters: how many starters in place of other cards give",
        "the same chance of at least one starter going first, by linear interpolation.",
        "",
    ]
    return ["## Draw and search cards, modeled simply", ""] + note + table(head, rows)


def agrees(printed: str, exact: Fraction) -> bool:
    """Whether a printed percentage is the exact chance rounded to the digits shown."""
    digits = len(printed.split(".")[1]) if "." in printed else 0
    return abs(Fraction(printed) - 100 * exact) <= Fraction(1, 2 * 10**digits)


def published() -> list[str]:
    """Values printed by public calculators beside the exact ones."""
    rows = []
    for source, question, theirs, size, k, n in PUBLISHED:
        p = p_at_least_one(size, k, n)
        p = p if question == "at least 1" else 1 - p
        rows.append([source, f"{question}, {k} in {size}, {n} drawn", theirs, pct(p, 4),
                     "yes" if agrees(theirs, p) else "no"])  # fmt: skip
    head = ["Calculator", "Question", "Printed (%)", "Exact", "Agrees to the digits shown"]
    return ["## Public calculators, read on 2026-11-19", ""] + table(head, rows)


def exact_tables() -> str:
    """The text of results/tables.md."""
    parts = ["# Opening-hand tables", "", "Written by `uv run python -m yugioh.tables`. Exact.", ""]
    for section in (example, at_least_one, starters_needed, deck_size, two_needs, bricks, copies, engines,
                    published):  # fmt: skip
        parts += section()
    return "\n".join(parts).rstrip() + "\n"


def simulation(trials: int = TRIALS, engine_trials: int = ENGINE_TRIALS) -> str:
    """The text of results/simulation.md: seeded hands against the exact chances."""
    rows = []
    names = named(EXAMPLE)
    for label, need in NEEDS.items():
        hits = simulate(names, FIRST, by_role(need), trials, SEED)
        lo, hi = wilson(hits, trials)
        exact = p_hand(EXAMPLE, FIRST, need)
        rows.append([label, f"{hits:,}", f"{lo:.4%} to {hi:.4%}", pct(exact, 4),
                     "yes" if lo <= exact <= hi else "no"])  # fmt: skip
    eng_rows = []
    for label, size, eng in ENGINES[1:]:
        deck = Deck(size, EXAMPLE.starters, EXAMPLE.extenders, EXAMPLE.bricks)
        for need_label, need in (
            ("starter", starter),
            ("starter and extender", starter_and_extender),
        ):
            hits = simulate_engine(deck, eng, FIRST, need, engine_trials, SEED)
            lo, hi = wilson(hits, engine_trials)
            exact = p_engine(deck, eng, FIRST, need)
            eng_rows.append([f"{label}: {need_label}", f"{hits:,}", f"{lo:.3%} to {hi:.3%}",
                             pct(exact, 3), "yes" if lo <= exact <= hi else "no"])  # fmt: skip
    head = ["Need", "Hits", "95% interval", "Exact", "Exact in interval"]
    return (
        "\n".join(
            [
                "# Simulation",
                "",
                "Written by `uv run python -m yugioh.tables --simulate`. The example deck of 40 cards",
                f"(12 starters, 9 extenders, 3 bricks), going first. Seed {SEED}; Wilson intervals.",
                "",
                f"## Hands without draw cards: {trials:,} hands, one sample for every need",
                "",
            ]
            + table(head, rows)
            + [
                f"## With draw cards: {engine_trials:,} shuffled decks per row,"
                " dealt by code separate from the exact recursion",
                "",
            ]
            + table(head, eng_rows)
        ).rstrip()
        + "\n"
    )


def main() -> None:
    """Write the tables, and the simulation if asked."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "tables.md").write_text(exact_tables())
    if args.simulate:
        (RESULTS / "simulation.md").write_text(simulation())


if __name__ == "__main__":
    main()
