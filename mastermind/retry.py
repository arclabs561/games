"""Fallout 3 terminals with exit and retry: the best play under two readings of the manual.

The Fallout 3 manual (PC and PlayStation 3 editions, section "Hacking") says that a player
may leave the hacking screen at any time and try again, and that on reentry "the password
will reset as will your number of tries". So a careful player is never locked out: she
leaves before a fourth wrong guess. What then matters is how many guesses and how many
**entries** (sessions at the terminal, from entering to leaving) a password costs.

"The password will reset" can be read two ways, and both are modeled:

- **Reading A, same password.** The screen and the password stay; the player remembers the
  likenesses already seen. Only the count of tries resets, so the attempt limit never binds.
- **Reading B, new password.** The screen stays, but the password is drawn again uniformly
  from its words, so what was learned about the old password is worthless. Each entry is a
  fresh game.

In both, a player never risks the lockout: within an entry she makes at most three guesses
that might be wrong, and a fourth only when one candidate is left. She may leave at any time.
Leaving costs no guess. The objective is the expected number of guesses, with the expected
number of entries as the tie-break under reading A and as a second output under reading B.
Everything here is exact per screen, in fractions; the study averages over the screens that
`fallout3.study` draws (the same seed and order), so across screens it is a simulation.

    uv run python -m mastermind.retry                    # one screen from the README, exact
    uv run python -m mastermind.retry --row 1 --out mastermind/results   # one row, 10 minutes
    uv run python -m mastermind.retry --merge mastermind/results        # writes retry.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections.abc import Sequence
from fractions import Fraction
from functools import cache
from pathlib import Path

from mastermind.fallout3 import (
    LEVELS,
    SEED,
    Table,
    likeness_table,
    most_cracked,
    partition,
    sample_screen,
)

SAFE = 3  # guesses per entry that may be wrong; a fourth is made only when it is certain

type Cands = tuple[int, ...]


def useful_guesses(table: Table, cands: Cands) -> list[int]:
    """Screen words worth guessing: a candidate, or a word that splits the candidates."""
    return [g for g in range(len(table)) if g in cands or len({table[g][s] for s in cands}) > 1]


def fewest_guesses(table: Table) -> tuple[int, int]:
    """Reading A: the least total guesses over all passwords of the screen, and among the
    strategies that reach it, the least total entries. Returns (guesses, entries), both
    summed over the screen's passwords. Exact, by search with memoization."""
    length = table[0][0]

    @cache
    def best(cands: Cands, used: int) -> tuple[int, int]:
        # (guesses, extra entries) summed over `cands`, `used` guesses into this entry
        if used == SAFE:
            if len(cands) == 1:
                return (1, 0)  # the certain fourth guess
            g, e = best(cands, 0)  # leave and come back: one more entry for each password
            return (g, e + len(cands))
        if len(cands) == 1:
            return (1, 0)
        top: tuple[int, int] | None = None
        for g in useful_guesses(table, cands):
            guesses, entries = len(cands), 0
            for reply, part in partition(table, g, cands).items():
                if reply != length:
                    a, b = best(tuple(part), used + 1)
                    guesses, entries = guesses + a, entries + b
            if top is None or (guesses, entries) < top:
                top = (guesses, entries)
        assert top is not None
        return top

    g, e = best(tuple(range(len(table))), 0)
    return g, e + len(table)  # every password needs its first entry


def fewest_guesses_unlimited(table: Table) -> int:
    """The least total guesses with no attempt limit and no entries: the plain recursion, a
    check on `fewest_guesses`, whose guess total must equal it."""
    length = table[0][0]

    @cache
    def best(cands: Cands) -> int:
        if len(cands) == 1:
            return 1
        return min(
            len(cands)
            + sum(
                best(tuple(p)) for reply, p in partition(table, g, cands).items() if reply != length
            )
            for g in useful_guesses(table, cands)
        )

    return best(tuple(range(len(table))))


def entry_value(table: Table, restart: Fraction) -> tuple[Fraction, Fraction]:
    """Reading B, one entry, when starting again is worth `restart` guesses per password.

    Returns (guesses, exits) summed over the screen's passwords, for the policy that
    minimizes guesses + restart × exits: at each point the player guesses the best word or
    leaves. A password that is cracked contributes its guesses in this entry; one that leaves
    contributes its guesses so far and one exit."""
    length = table[0][0]

    @cache
    def best(cands: Cands, used: int) -> tuple[Fraction, Fraction]:
        leave = (Fraction(0), Fraction(len(cands)))
        options = [leave]
        if used < SAFE or len(cands) == 1:
            guesses = [g for g in useful_guesses(table, cands) if used < SAFE or g in cands]
            for g in guesses:
                a, b = Fraction(len(cands)), Fraction(0)
                for reply, part in partition(table, g, cands).items():
                    if reply != length:
                        pa, pb = best(tuple(part), used + 1)
                        a, b = a + pa, b + pb
                options.append((a, b))
        return min(options, key=lambda o: (o[0] + restart * o[1], o[1]))

    return best(tuple(range(len(table))), 0)


def new_password(table: Table) -> dict[str, Fraction]:
    """Reading B: the least expected guesses until the password is cracked, with a fresh
    uniform password at each entry. By renewal, a stationary policy whose entry costs c
    guesses on average and succeeds with chance q needs c / q guesses in all and 1 / q entries
    on average. The best ratio is found by Dinkelbach's iteration: value the restart at the
    current ratio, take the best policy for that value, and repeat until the ratio stops
    falling. Exact."""
    n = len(table)
    # Start above the optimum: with a restart this costly the policy first maximizes the
    # chance per entry, and its ratio is an upper bound. Each step then lowers the ratio
    # until it is a fixed point, which is the optimum.
    restart = Fraction(10**6)
    while True:
        guesses, exits = entry_value(table, restart)
        q = 1 - exits / n
        assert q > 0
        ratio = (guesses / n) / q
        assert ratio <= restart
        if ratio == restart:
            return {"guesses": ratio, "entries": 1 / q, "first entry": q}
        restart = ratio


def first_entry_chance(table: Table) -> Fraction:
    """The most passwords any strategy cracks or pins down within one entry, as a share of
    the screen: three guesses that may be wrong, then a certain fourth. Under reading A this
    is the chance that one entry is enough."""
    n, length = len(table), table[0][0]

    @cache
    def best(cands: Cands, used: int) -> int:
        if len(cands) == 1:
            return 1
        if used == SAFE:
            return 0
        top = 0
        for g in useful_guesses(table, cands):
            total = sum(
                1 if reply == length else best(tuple(part), used + 1)
                for reply, part in partition(table, g, cands).items()
            )
            top = max(top, total)
        return top

    return Fraction(best(tuple(range(n)), 0), n)


def score_screen(screen: Sequence[str]) -> dict[str, Fraction]:
    """Every quantity of both readings on one screen, exact."""
    table = likeness_table(screen)
    n = len(screen)
    g, e = fewest_guesses(table)
    b = new_password(table)
    return {
        "within four attempts": Fraction(max(most_cracked(table)), n),
        "within three attempts": Fraction(max(most_cracked(table, SAFE)), n),
        "one entry": first_entry_chance(table),
        "A guesses": Fraction(g, n),
        "A entries": Fraction(e, n),
        "B guesses": b["guesses"],
        "B entries": b["entries"],
        "B one entry": b["first entry"],
    }


ROWS = [("Very Easy", 10), ("Very Easy", 20), ("Easy", 20), ("Average", 20), ("Hard", 20)]


def screens_of_row(level: str, n: int, screens: int, seed: int = SEED):
    """The screens `fallout3.study` draws for this row: the same generator, seed and order."""
    rng = random.Random(f"{seed} {level} {n}")
    for _ in range(screens):
        yield sample_screen(rng, rng.choice(LEVELS[level]), n)


def study_row(row: int, screens: int = 1000, seed: int = SEED) -> dict:
    """Every quantity of `score_screen` on each screen of one row, as exact fractions."""
    level, n = ROWS[row]
    values: dict[str, list[str]] = {}
    for screen in screens_of_row(level, n, screens, seed):
        for k, v in score_screen(screen).items():
            values.setdefault(k, []).append(str(v))
    return {"level": level, "words": n, "screens": screens, "seed": seed, "values": values}


def summarize(rows: list[dict]) -> dict:
    """The mean of each quantity over each row's screens, with a normal 95% interval, its
    least and largest value, and how many screens the two readings' guesses differ on."""
    out = {}
    for r in rows:
        exact = {k: [Fraction(x) for x in v] for k, v in r["values"].items()}
        row = {}
        for k, v in exact.items():
            f = [float(x) for x in v]
            half = 1.96 * statistics.stdev(f) / len(f) ** 0.5
            row[k] = {
                "mean": round(statistics.fmean(f), 4),
                "half_width": round(half, 4),
                "min": round(float(min(v)), 4),
                "max": round(float(max(v)), 4),
            }
        pairs = list(zip(exact["A guesses"], exact["B guesses"], strict=True))
        row["screens where B needs more guesses than A"] = sum(b > a for a, b in pairs)
        row["screens where B needs fewer guesses than A"] = sum(b < a for a, b in pairs)
        out[f"{r['level']}, {r['words']} words"] = row
    return {"screens per row": rows[0]["screens"], "seed": rows[0]["seed"], "rows": out}


def _half_up(x: Fraction, places: int) -> str:
    """x rounded half up to `places` decimals."""
    q = round(x * 10**places)
    if (x * 10**places - q) == Fraction(1, 2):  # round() goes to even; go up instead
        q += 1
    text = f"{abs(q) // 10**places}.{abs(q) % 10**places:0{places}d}" if places else str(q)
    return text


def _cells(v: dict[str, list[Fraction]]) -> list[str]:
    """One row of the README's retry table, from the exact values on each screen."""
    n = len(v["one entry"])

    def mean(k: str) -> Fraction:
        return sum(v[k], Fraction(0)) / n

    def half(k: str) -> str:
        return f"{1.96 * statistics.stdev(map(float, v[k])) / n**0.5:.3f}"

    def share(k: str) -> str:
        return "100" if min(v[k]) == 1 else _half_up(100 * mean(k), 2) + "%"

    def count(k: str) -> str:
        return "1" if max(v[k]) == 1 else _half_up(mean(k), 3)

    return [
        share("within four attempts"),
        share("within three attempts"),
        share("one entry"),
        f"{_half_up(mean('A guesses'), 3)} ± {half('A guesses')}",
        count("A entries"),
        f"{_half_up(mean('B guesses'), 3)} ± {half('B guesses')}",
        count("B entries"),
    ]


def readme_rows(data: dict) -> list[str]:
    """The rows of the README's retry table, from `results/retry.json`."""
    lines = []
    for row in data["rows"]:
        v = {k: [Fraction(x) for x in xs] for k, xs in row["values"].items()}
        lines.append(f"| {row['level']} | {row['words']} | " + " | ".join(_cells(v)) + " |")
    return lines


EXAMPLE = (
    "brain chair drink flame ghost heart knife light money night plant queen river sheep "
    "stone train water youth"
).split()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("words", nargs="*", help="the screen words (default: the README's)")
    parser.add_argument("--row", type=int, help="score one row of ROWS (0 to 4)")
    parser.add_argument("--screens", type=int, default=1000)
    parser.add_argument("--out", help="with --row: folder for retry-row<k>.json")
    parser.add_argument("--merge", help="folder of row files; writes retry.json beside them")
    args = parser.parse_args(argv)
    if args.row is not None:
        result = study_row(args.row, args.screens)
        path = Path(args.out or ".") / f"retry-row{args.row}.json"
        path.write_text(json.dumps(result, separators=(",", ":"), sort_keys=True) + "\n")
        return
    if args.merge:
        files = sorted(Path(args.merge).glob("retry-row*.json"))
        rows = [json.loads(f.read_text()) for f in files]
        summary = summarize(rows)
        # the summary indented, then one line per row of exact values
        head = json.dumps({"summary": summary}, indent=1, sort_keys=True)
        body = ",\n".join(json.dumps(r, separators=(",", ":"), sort_keys=True) for r in rows)
        text = head[:-2] + ',\n "rows": [\n' + body + "\n ]\n}\n"
        merged = {"summary": summary}
        (Path(args.merge) / "retry.json").write_text(text)
        print(json.dumps(merged["summary"], indent=1))
        return
    screen = [w.upper() for w in (args.words or EXAMPLE)]
    for k, v in score_screen(screen).items():
        print(f"{k:<22}{str(v):>10}  = {float(v):.4f}")


if __name__ == "__main__":
    main()
