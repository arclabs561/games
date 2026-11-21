"""Write results/second.md: hand traps going second, cards with two roles, the Normal Summon.

    uv run python -m yugioh.second              # exact tables, a few seconds
    uv run python -m yugioh.second --simulate   # also results/second_simulation.md, a minute

Every number in second.md is exact. The simulation deals seeded hands from the decks of the
README's rows and decides them with the slow check of `roles.py`.
"""

import argparse
from fractions import Fraction
from pathlib import Path

from yugioh.odds import FIRST, SECOND, fewest, wilson
from yugioh.roles import Deck, Need, group, need, p_going_second, p_naive, p_need, simulate
from yugioh.tables import TARGETS, pct, table

RESULTS = Path(__file__).parent / "results"
SIZE = 40
SEED, TRIALS = 1, 200_000

STARTER_AND_TRAP = need("starter", "hand trap")
FULL_COMBO = need("starter", "extender", "hand trap")
TWO_TRAPS = need("hand trap", "hand trap")
STRICT = need("starter", "extender")
STRICT_ONE_SUMMON = need("starter", "extender", summons=1)
TWO_ENGINE_ONE_SUMMON = need("starter", "starter|extender", summons=1)
STARTER_OR_TRAP = need("starter|hand trap")


def example(traps: int = 12, both: int = 0, starters_ns: int = 0, extenders_ns: int = 0) -> Deck:
    """The README's example deck (12 starters, 9 extenders, 3 bricks in 40), with variations.

    `traps` of its 16 other cards are hand traps. `both` of the starters are also extenders.
    `starters_ns` starters and `extenders_ns` extenders need the Normal Summon (both counts
    taken among the cards with one role).
    """
    groups = [
        group(starters_ns, "starter", summon=True),
        group(12 - both - starters_ns, "starter"),
        group(both, "starter", "extender"),
        group(extenders_ns, "extender", summon=True),
        group(9 - extenders_ns, "extender"),
        group(3, "brick"),
        group(traps, "hand trap"),
    ]
    return tuple(g for g in groups if g.cards)


def plain(starters: int, traps: int) -> Deck:
    """A deck with only starters and hand traps; the rest have no role in the need."""
    return tuple(g for g in (group(starters, "starter"), group(traps, "hand trap")) if g.cards)


def fewest_traps(target: Fraction, n: int, k: int) -> int | None:
    """Fewest hand traps in 40 for at least k of them in n cards with chance at least target."""
    want = need(*["hand trap"] * k)
    for t in range(SIZE + 1):
        if p_need((group(t, "hand trap"),) if t else (), n, want, SIZE) >= target:
            return t
    return None


def exact(p: Fraction) -> str:
    return f"{p.numerator}/{p.denominator}"


def hand_traps() -> list[str]:
    rows = []
    for k, label in ((1, "at least one"), (2, "at least two")):
        for n, side in (
            (FIRST, "before the first opponent turn"),
            (SECOND, "after the sixth draw"),
        ):
            rows.append([f"{label}, {side}"] + [fewest_traps(t, n, k) for t in TARGETS])
    assert all(
        fewest_traps(t, n, 1) == fewest(t, SIZE, n) for t in TARGETS for n in (FIRST, SECOND)
    )
    twelve = [
        p_need((group(12, "hand trap"),), SECOND, w, SIZE) for w in (need("hand trap"), TWO_TRAPS)
    ]
    return (
        ["## Hand traps in 40 cards", ""]
        + ["Fewest hand traps for the chance in each column:", ""]
        + table(["Hand traps in hand", *[f"{float(t):.0%}" for t in TARGETS]], rows)
        + [
            "At least one is the same count as for starters, since only the number of cards "
            "matters. With 12 hand traps, the first six cards contain at least one with chance "
            f"{pct(twelve[0])} and at least two with chance {pct(twelve[1])}.",
            "",
        ]
    )


def starter_and_trap() -> list[str]:
    traps = (6, 9, 12, 15)
    rows = [
        [s] + [pct(p_going_second(plain(s, t), SECOND, STARTER_AND_TRAP, SIZE)) for t in traps]
        for s in (9, 12, 15)
    ]
    combo = []
    for t in (0, 6, 9, 12, 15):
        d = example(traps=t)
        combo.append(
            [t]
            + [pct(p_going_second(d, SECOND, w, SIZE)) for w in (STARTER_AND_TRAP, FULL_COMBO)]
            + [pct(p_going_second(d, SECOND, STARTER_OR_TRAP, SIZE))]
        )
    best = max(
        range(1, 28), key=lambda s: p_going_second(plain(s, 28 - s), SECOND, STARTER_AND_TRAP, SIZE)
    )
    top = p_going_second(plain(best, 28 - best), SECOND, STARTER_AND_TRAP, SIZE)
    return (
        ["## A starter and a hand trap, going second", ""]
        + [
            "The hand trap must be among the opening five. A starter or extender may arrive on the sixth draw. No card in this table has both roles:",
            "",
        ]
        + table(["Starters \\ Hand traps", *map(str, traps)], rows)
        + [
            f"With 28 cards to split between the two roles, {best} starters and {28 - best} "
            f"hand traps give the most, {pct(top)}.",
            "",
            f"If both roles may arrive by six, 12 starters and 12 traps give {pct(p_need(plain(12, 12), SECOND, STARTER_AND_TRAP, SIZE))}.",
            "",
            "The example deck (12 starters, 9 extenders, 3 bricks), with some of its 16 other "
            "cards as hand traps, going second:",
            "",
        ]
        + table(
            [
                "Hand traps",
                "A starter and a hand trap",
                "A starter, an extender and a hand trap",
                "A starter or a hand trap",
            ],
            combo,
        )  # fmt: skip
    )


def two_roles() -> list[str]:
    rows = []
    for both in (0, 3, 6, 9, 12):
        d = example(traps=0, both=both)
        right = p_need(d, FIRST, STRICT, SIZE)
        naive = p_naive(d, FIRST, ("starter", "extender"), SIZE)
        rows.append(
            [both, pct(right), pct(naive), f"{float(naive - right) * 100:.2f}", exact(right)]
        )
    return (
        ["## Starters that are also extenders", ""]
        + [
            "The example deck going first, with some of its 12 starters also extenders and its 9 "
            "extenders unchanged. The need is two different cards, a starter and an extender. "
            "Counting a card with both roles once for each role, as if it met the need alone, "
            "gives the last value.",
            "",
        ]
        + table(
            [
                "Starters that also extend",
                "A starter and an extender",
                "Counted twice",
                "Overstated by (points)",
                "Exact",
            ],
            rows,
        )  # fmt: skip
    )


def normal_summon() -> list[str]:
    head = ["Starters that need it \\ Extenders that need it", *map(str, (0, 3, 6, 9))]
    strict = [
        [a]
        + [
            pct(
                p_need(
                    example(traps=0, starters_ns=a, extenders_ns=b), FIRST, STRICT_ONE_SUMMON, SIZE
                )
            )
            for b in (0, 3, 6, 9)
        ]
        for a in (0, 4, 8, 12)
    ]
    loose = [
        [a]
        + [
            pct(
                p_need(
                    example(traps=0, starters_ns=a, extenders_ns=b),
                    FIRST,
                    TWO_ENGINE_ONE_SUMMON,
                    SIZE,
                )
            )
            for b in (0, 3, 6, 9)
        ]
        for a in (0, 4, 8, 12)
    ]
    return (
        ["## The one Normal Summon", ""]
        + [
            "The example deck going first. Some starters and some extenders need the turn's "
            "Normal Summon, and two such cards cannot both be played. A starter and an "
            "extender, using the Normal Summon at most once:",
            "",
        ]
        + table(head, strict)
        + ["A starter and another engine card (a second starter or an extender), the same way:", ""]
        + table(head, loose)
        + [
            "At least one starter does not change, since one card uses the Normal Summon at most once.",
            "",
        ]
    )


def exact_tables() -> str:
    lines = [
        "# Going second, two roles and the Normal Summon",
        "",
        "Written by `uv run python -m yugioh.second`. Exact.",
        "",
    ]
    lines += hand_traps() + starter_and_trap() + two_roles() + normal_summon()
    return "\n".join(lines).rstrip("\n") + "\n"


# The rows the simulation checks: (label, deck, cards drawn, need).
CHECKS: list[tuple[str, Deck, int, Need]] = [
    ("12 starters, 12 hand traps: a starter and a hand trap, second", plain(12, 12), SECOND, STARTER_AND_TRAP),
    ("example deck, 12 hand traps: a starter, an extender and a hand trap, second", example(12), SECOND, FULL_COMBO),
    ("example deck, 3 starters also extend: a starter and an extender, first", example(0, both=3), FIRST, STRICT),
    ("example deck, 8 starters and 3 extenders need the Normal Summon, first", example(0, starters_ns=8, extenders_ns=3), FIRST, STRICT_ONE_SUMMON),
]  # fmt: skip


def simulation() -> str:
    rows = []
    for label, deck, n, want in CHECKS:
        score = p_going_second if n == SECOND else p_need
        p = score(deck, n, want, SIZE)
        hits = simulate(deck, n, want, SIZE, TRIALS, SEED, going_second=n == SECOND)
        lo, hi = wilson(hits, TRIALS)
        inside = "yes" if lo <= p <= hi else "no"
        rows.append([label, pct(p, 4), f"{hits / TRIALS:.4%}", f"{lo:.4%} to {hi:.4%}", inside])
    return (
        "\n".join(
            [
                "# Going second, two roles and the Normal Summon: a seeded simulation",
                "",
                f"Written by `uv run python -m yugioh.second --simulate`. Seed {SEED}, {TRIALS:,} hands "
                "per row, each decided by trying every assignment of the need's slots to distinct "
                "cards (`roles.fits_cards`), not by the search over groups that the exact values use.",
                "",
                *table(["Need", "Exact", "Simulated", "95% interval", "Exact inside"], rows),
            ]
        ).rstrip("\n")
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()
    (RESULTS / "second.md").write_text(exact_tables())
    if args.simulate:
        (RESULTS / "second_simulation.md").write_text(simulation())
    print(exact_tables())


if __name__ == "__main__":
    main()
