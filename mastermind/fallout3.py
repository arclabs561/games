"""Fallout 3 terminal hacking, played with the Mastermind strategies.

A hackable terminal shows a **screen** of words, all the same length; one of them is the
**password**. Each **guess** must be a word on the screen, and the terminal replies with its
**likeness**: how many letters are right and in the right place. Four wrong guesses lock the
terminal. The **candidates** are the screen words that would have given every likeness so far.
A guess **partitions** the candidates by the likeness each would give, as in Mastermind, so the
four strategies of `strategies.py` apply unchanged; the only difference is that a guess is any
word on the screen, not any string. A ruled-out word stays on the screen and may still be
guessed to split the candidates.

Word lists from the game are not available, so `study` draws screens from the ENABLE word list
in `boggle/enable1.txt` with a seeded generator, at the word lengths and word counts the game
uses, and scores every strategy on every password of each screen. Within a screen the numbers
are exact; across screens they are simulated, and reported with the number of screens.

    uv run python -m mastermind.fallout3 --best doctrines batteries scorpions generated occasions
    uv run python -m mastermind.fallout3 --best doctrines=3 batteries scorpions generated occasions
    uv run python -m mastermind.fallout3 doctrines batteries scorpions   # asks for each likeness
    uv run python -m mastermind.fallout3 --study                         # about 4 minutes
    uv run python -m mastermind.fallout3 --published                     # the Fallout 4 screens
"""

import argparse
import hashlib
import itertools
import random
import statistics
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from fractions import Fraction
from functools import cache
from pathlib import Path

from mastermind.strategies import fewer_parts, largest_part, negative_entropy, sum_of_squares

ATTEMPTS = 4

type Word = str
type Table = tuple[tuple[int, ...], ...]  # table[g][s]: likeness of guess g, password s
type Strategy = Callable[[Table, list[int]], int]  # candidates -> guess, both screen indices


def likeness(guess: Word, password: Word) -> int:
    """The terminal's reply: letters right and in the right place."""
    return sum(a == b for a, b in zip(guess, password, strict=True))


# Examples from the Fallout wiki's Hacking page.
assert likeness("detector", "released") == 2  # the Es in places 2 and 4; the Rs do not line up
assert likeness("design", "benign") == 4


def likeness_table(screen: Sequence[Word]) -> Table:
    """Likeness of every pair of screen words, by index."""
    return tuple(tuple(likeness(g, s) for s in screen) for g in screen)


def partition(table: Table, guess: int, candidates: Sequence[int]) -> dict[int, list[int]]:
    """The candidates grouped by the likeness they would give to `guess`."""
    parts: dict[int, list[int]] = {}
    for s in candidates:
        parts.setdefault(table[guess][s], []).append(s)
    return parts


# Strategies. Each returns the index of the next guess.


def lowest_cost(table: Table, candidates: list[int], cost: Callable[[list[int]], float]) -> int:
    """The screen word whose partition has the lowest cost, preferring a candidate, then the
    word nearest the top of the screen (Knuth's tie-break, as in `strategies.best_guess`)."""
    if len(candidates) <= 2:
        return candidates[0]
    preferred = set(candidates)
    return min(
        range(len(table)),
        key=lambda g: (
            cost([len(p) for p in partition(table, g, candidates).values()]),
            g not in preferred,
            g,
        ),
    )


def minimax(table: Table, candidates: list[int]) -> int:
    """Make the largest part as small as possible (Knuth)."""
    return lowest_cost(table, candidates, largest_part)


def most_parts(table: Table, candidates: list[int]) -> int:
    """Make as many parts as possible (Kooi)."""
    return lowest_cost(table, candidates, fewer_parts)


def expected_size(table: Table, candidates: list[int]) -> int:
    """Make the expected size of the remaining part as small as possible (Irving)."""
    return lowest_cost(table, candidates, sum_of_squares)


def entropy(table: Table, candidates: list[int]) -> int:
    """Make the likeness as informative as possible (Neuwirth)."""
    return lowest_cost(table, candidates, negative_entropy)


def most_in_common(table: Table, candidates: list[int]) -> int:
    """The candidate with the most letters in place, summed over the other candidates: the
    "closest word" of Math Goes Pop (2016), and the easiest rule to apply by hand."""
    return max(candidates, key=lambda g: (sum(table[g][s] for s in candidates), -g))


STRATEGIES: dict[str, Strategy] = {
    "most parts": most_parts,
    "expected size": expected_size,
    "entropy": entropy,
    "minimax": minimax,
    "most in common": most_in_common,
}


def guess_counts(table: Table, strategy: Strategy) -> list[int]:
    """Guesses `strategy` needs for each password on the screen, with no attempt limit."""
    counts = [0] * len(table)
    length = table[0][0]

    def play(candidates: list[int], depth: int) -> None:
        guess = strategy(table, candidates)
        for reply, part in partition(table, guess, candidates).items():
            if reply == length:
                counts[guess] = depth
            else:
                play(part, depth + 1)

    play(list(range(len(table))), 1)
    return counts


def candidate_parts(table: Table) -> Callable[[tuple[int, ...]], list[tuple[int, ...]]]:
    """A cached function: the parts left after each candidate is guessed, for all candidate
    guesses together, without the winning part."""
    length = table[0][0]

    @cache
    def parts(cands: tuple[int, ...]) -> list[tuple[int, ...]]:
        return [
            tuple(p)
            for g in cands
            for reply, p in partition(table, g, cands).items()
            if reply != length
        ]

    return parts


def random_candidate(table: Table, attempts: int = ATTEMPTS) -> Fraction:
    """When every guess is a candidate chosen uniformly at random, the expected number of
    passwords on the screen cracked within `attempts`. Exact."""
    parts = candidate_parts(table)

    @cache
    def cracked(cands: tuple[int, ...], left: int) -> Fraction:
        # Each candidate is guessed with chance 1/m. It cracks one password, and the
        # passwords in each part it leaves get the remaining attempts.
        if left == 1:
            return Fraction(1)
        return 1 + Fraction(sum(cracked(p, left - 1) for p in parts(cands)), len(cands))

    return cracked(tuple(range(len(table))), attempts)


def random_candidate_guesses(table: Table) -> Fraction:
    """The same strategy's expected guesses with no attempt limit, summed over passwords."""
    parts = candidate_parts(table)

    @cache
    def guesses(cands: tuple[int, ...]) -> Fraction:
        return len(cands) + Fraction(sum(guesses(p) for p in parts(cands)), len(cands))

    return guesses(tuple(range(len(table))))


def most_cracked(
    table: Table, attempts: int = ATTEMPTS, candidates: Sequence[int] | None = None
) -> list[int]:
    """For each first guess, the most candidates that can be cracked within `attempts` when
    that word is guessed first and every later guess is the best one. The best of these is
    the optimum over all strategies. Exact, by search with memoization."""
    n, length = len(table), table[0][0]
    every = tuple(range(n)) if candidates is None else tuple(candidates)

    def after(guess: int, cands: tuple[int, ...], left: int) -> int:
        total = 0
        for reply, part in partition(table, guess, cands).items():
            total += 1 if reply == length else best(tuple(part), left - 1)
        return total

    @cache
    def best(cands: tuple[int, ...], left: int) -> int:
        if left == 0:
            return 0
        if left == 1 or len(cands) == 1:
            return 1
        if len(cands) <= left:
            return len(cands)  # guess the candidates one by one
        top = 0
        for g in range(n):
            if g not in cands and len({table[g][s] for s in cands}) == 1:
                continue  # learns nothing
            top = max(top, after(g, cands, left))
            if top == len(cands):
                break
        return top

    return [after(g, every, attempts) for g in range(n)]


# Using the solver while playing.


def candidates_after(screen: Sequence[Word], tried: dict[Word, int]) -> list[Word]:
    """The screen words that would have given each likeness in `tried`."""
    return [w for w in screen if all(likeness(g, w) == k for g, k in tried.items())]


def best_next_guess(
    screen: Sequence[Word], tried: dict[Word, int], attempts: int = ATTEMPTS
) -> tuple[Word, int, int]:
    """The guess that cracks the most remaining candidates within the attempts left, with
    ties going to the smallest expected remaining part, then a candidate, then screen order.
    Returns the guess, how many candidates it is sure to crack, and how many remain."""
    left = attempts - len(tried)
    cands = candidates_after(screen, tried)
    if left <= 0 or not cands:
        raise ValueError("no attempts or candidates left")
    table = likeness_table(screen)
    index = [screen.index(w) for w in cands]
    values = most_cracked(table, left, index)

    def key(g: int) -> tuple[int, float, bool, int]:
        sizes = [len(p) for p in partition(table, g, index).values()]
        return (-values[g], sum_of_squares(sizes), g not in index, g)

    g = min(range(len(screen)), key=key)
    return screen[g], values[g], len(cands)


class Terminal:
    """A terminal session: the screen, and the candidates left after each reply."""

    def __init__(self, words: Sequence[Word]):
        len_mode, count = Counter(map(len, words)).most_common(1)[0]
        if count != len(words):
            bad_words = [w for w in words if len(w) != len_mode]
            raise ValueError(f"Some words have different lengths, namely:\n\t{bad_words}")
        self.screen = [w.upper() for w in words]
        self.words = list(self.screen)
        self.tried: dict[Word, int] = {}

    def letters_common(self, word1: Word, word2: Word) -> int:
        return likeness(word1, word2)

    def attempted(self, word: Word, correct: int) -> None:
        """Record the likeness the terminal gave for `word`."""
        self.tried[word.upper()] = correct
        self.words = candidates_after(self.screen, self.tried)

    def automate(self) -> None:
        while len(self.tried) < ATTEMPTS and self.words:
            guess, sure, left = best_next_guess(self.screen, self.tried)
            if left == 1:
                print(f"The password is {guess}")
                return
            print(f"{left} candidates; guess {guess} (cracks {sure} of {left} for certain)")
            while True:
                try:
                    correct = int(input("Likeness? "))
                    break
                except KeyboardInterrupt:
                    sys.exit()
                except ValueError:
                    print("Please input an integer...")
            if correct == len(guess):
                return
            self.attempted(guess, correct)
        print("No candidates left" if not self.words else "Out of attempts")


# The study on sampled screens.

ENABLE = Path(__file__).parent.parent / "boggle" / "enable1.txt"
ENABLE_SHA256 = "a392640f14602fcbb36779eb6671d170260c06d43f41c16383e7f0e35b221b4a"
SEED = 3
# Password lengths by terminal level, from the Fallout wiki's Hacking page.
LEVELS: dict[str, tuple[int, ...]] = {
    "Very Easy": (4, 5),
    "Easy": (6, 7, 8),
    "Average": (9, 10),
    "Hard": (11, 12),
    "Very Hard": (13, 14, 15),
}


def word_count(science: float, level: int, low: int = 5, high: int = 20) -> float:
    """Words on the screen, by the GECK wiki's formula (before any rounding, which the wiki
    does not give). `level` is 0 for Very Easy up to 4 for Very Hard."""
    lock = level * 0.25 * 100
    share = 0.5 if lock == 100 else (science - lock) / (100 - lock)
    return min(high, (1 - share) * (high - low) + low)


assert word_count(25, 1) == 20 and word_count(100, 1) == 5 and word_count(100, 4) == 12.5


@cache
def words_by_length() -> dict[int, list[Word]]:
    """ENABLE's words grouped by length, after checking the file's digest."""
    data = ENABLE.read_bytes()
    if hashlib.sha256(data).hexdigest() != ENABLE_SHA256:
        raise ValueError(f"{ENABLE}: SHA-256 digest does not match")
    groups: dict[int, list[Word]] = {}
    for word in data.decode().split():
        groups.setdefault(len(word), []).append(word)
    return groups


def sample_screen(rng: random.Random, length: int, n: int) -> list[Word]:
    """`n` different ENABLE words of `length` letters, in random order."""
    return rng.sample(words_by_length()[length], n)


def score_screen(
    screen: Sequence[Word], attempts: int = ATTEMPTS, means: bool = False
) -> dict[str, tuple]:
    """Each strategy on every password of the screen: (passwords cracked within `attempts`,
    total guesses with no limit, most guesses). Also random candidates (whose total guesses
    are computed only if `means`, as they are slow) and the best possible."""
    table = likeness_table(screen)
    out: dict[str, tuple] = {}
    for name, strategy in STRATEGIES.items():
        counts = guess_counts(table, strategy)
        out[name] = (sum(c <= attempts for c in counts), sum(counts), max(counts))
    guesses = random_candidate_guesses(table) if means else None
    out["random candidate"] = (random_candidate(table, attempts), guesses, None)
    values = most_cracked(table, attempts)
    out["best possible"] = (max(values), None, None)
    out["first guesses"] = (values, None, None)
    return out


def mean_interval(values: Sequence[float]) -> tuple[float, float]:
    """Mean and half-width of a normal 95% interval."""
    mean = statistics.fmean(values)
    half = 1.96 * statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
    return mean, half


def first_guess_rules(table: Table, values: list[int] | None = None) -> dict[str, Fraction]:
    """For each first-guess rule, the share of the optimum kept when that rule picks only the
    first guess and every later guess is the best one. `values` is `most_cracked(table)`."""
    values = most_cracked(table) if values is None else values
    top = max(values)
    every = list(range(len(table)))
    rules = {name: s(table, every) for name, s in STRATEGIES.items()}
    out = {name: Fraction(values[g], top) for name, g in rules.items()}
    out["random word"] = Fraction(sum(values), top * len(values))
    return out


# Word counts to try at each level. The formula gives 5 to 20 below Very Hard, depending on
# Science, and 12.5 at Very Hard whatever the Science.
COUNTS: dict[str, tuple[int, ...]] = {
    "Very Easy": (5, 10, 15, 20),
    "Easy": (5, 10, 15, 20),
    "Average": (5, 10, 15, 20),
    "Hard": (5, 10, 15, 20),
    "Very Hard": (12, 13),
}


def study(screens: int = 1000, seed: int = SEED) -> None:
    """Print the chance of cracking a terminal within four attempts, by level and word count,
    how often each strategy can miss, the hardest screen found and the first-guess rules."""
    names = [*STRATEGIES, "random candidate", "best possible"]
    short = ["parts", "exp size", "entropy", "minimax", "common", "random", "best"]
    rows = []
    worst: tuple[Fraction, list[Word]] | None = None
    most_guesses: Counter[str] = Counter()
    rule_shares: dict[str, list[Fraction]] = {}
    for level, lengths in LEVELS.items():
        for n in COUNTS[level]:
            rng = random.Random(f"{seed} {level} {n}")
            won: dict[str, list[float]] = {name: [] for name in names}
            missed: Counter[str] = Counter()
            for _ in range(screens):
                screen = sample_screen(rng, rng.choice(lengths), n)
                scores = score_screen(screen)
                for name in names:
                    won[name].append(float(scores[name][0]) / n)
                    missed[name] += scores[name][0] < n
                for name in STRATEGIES:
                    most_guesses[name] = max(most_guesses[name], scores[name][2])
                share = Fraction(scores["best possible"][0], n)
                if worst is None or share < worst[0]:
                    worst = (share, screen)
                if n == COUNTS[level][-1]:
                    table = likeness_table(screen)
                    for rule, kept in first_guess_rules(table, scores["first guesses"][0]).items():
                        rule_shares.setdefault(rule, []).append(kept)
            rows.append((level, n, won, missed))

    print(f"{screens} sampled screens per row, seed {seed}; the password is uniform over each")
    print("screen. Percent of passwords cracked within 4 attempts, with a 95% interval:\n")
    print(f"{'level':<10}{'words':>6}" + "".join(f"{s:>14}" for s in short))
    for level, n, won, _ in rows:
        cells = "".join(
            "{:>8.2f} ±{:<4.2f}".format(*(100 * x for x in mean_interval(won[name])))
            for name in names
        )
        print(f"{level:<10}{n:>6}{cells}")
    print("\nPercent of screens on which some password is not cracked within 4 attempts:\n")
    print(f"{'level':<10}{'words':>6}" + "".join(f"{s:>9}" for s in short))
    for level, n, _, missed in rows:
        print(f"{level:<10}{n:>6}" + "".join(f"{100 * missed[k] / screens:>9.1f}" for k in names))
    print("\nMost guesses any password needed, with no attempt limit:")
    print("  " + ", ".join(f"{name} {k}" for name, k in most_guesses.items()))
    if worst is not None:
        print(f"\nHardest screen found: the best strategy cracks {worst[0]} of its passwords:")
        print("  " + " ".join(w.upper() for w in worst[1]))
    print("\nFirst-guess rules on the screens with the most words of each level: the share of")
    print("the best chance kept when the rule picks the first guess and later guesses are best:")
    for rule, shares in sorted(rule_shares.items(), key=lambda kv: -sum(kv[1])):
        mean, half = mean_interval([float(s) for s in shares])
        exact = sum(s == 1 for s in shares) / len(shares)
        print(f"  {rule:<16}{100 * mean:>7.2f}% ±{100 * half:.2f}   optimal on {100 * exact:.1f}%")


# Three Fallout 4 screens published by Math Goes Pop (2016), "Fallout 4 Math, Part 2".
FALLOUT4_SCREENS = [
    "agent fence forth start shape grass parts fists sound sewer viral piled".split(),
    "energy mutter warned atrium second carved forced hungry depend heated mirror stream".split(),
    "varying cistern expects attends bottles torches limited corners fortify despite session durable".split(),
]


def mean_likeness(screen: Sequence[Word]) -> float:
    """Average likeness over pairs of different screen words."""
    pairs = list(itertools.combinations(screen, 2))
    return statistics.fmean(likeness(a, b) for a, b in pairs)


def published_screens(samples: int = 1000, seed: int = SEED) -> None:
    """Every strategy on the three Fallout 4 screens: exact over all twelve passwords. Also how
    alike their words are, against sampled ENABLE screens of the same size."""
    for screen in FALLOUT4_SCREENS:
        rng = random.Random(f"{seed} {screen[0]}")
        sampled = [mean_likeness(sample_screen(rng, len(screen[0]), 12)) for _ in range(samples)]
        share = sum(x >= mean_likeness(screen) for x in sampled) / samples
        print(" ".join(screen))
        print(
            f"  mean likeness {mean_likeness(screen):.3f}; {samples} ENABLE screens (seed {seed}):"
            f" mean {statistics.fmean(sampled):.3f}, {100 * share:.1f}% as alike or more"
        )
        scores = score_screen(screen, means=True)
        del scores["first guesses"]
        for name, (won, total, worst) in scores.items():
            mean = f"{float(total) / len(screen):.3f}" if total is not None else ""
            worst = "" if worst is None else worst
            print(
                f"  {name:<18}cracked {float(won):>6.3f} of 12   average {mean:>6}  worst {worst}"
            )
    # Math Goes Pop (2016) also scored 10,000 random twelve-word screens of 5, 6 and 7 letters,
    # from a word list they do not name. The same experiment on ENABLE:
    print(f"\nPercent cracked within 4 attempts on {samples} twelve-word ENABLE screens:")
    for length in (5, 6, 7):
        rng = random.Random(f"{seed} twelve {length}")
        won: dict[str, list[float]] = {"expected size": [], "most in common": [], "random": []}
        for _ in range(samples):
            table = likeness_table(sample_screen(rng, length, 12))
            for name in ("expected size", "most in common"):
                counts = guess_counts(table, STRATEGIES[name])
                won[name].append(sum(c <= ATTEMPTS for c in counts) / 12)
            won["random"].append(float(random_candidate(table)) / 12)
        cells = ", ".join(
            "{} {:.2f} ±{:.2f}".format(name, *(100 * x for x in mean_interval(v)))
            for name, v in won.items()
        )
        print(f"  {length} letters: {cells}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("words", nargs="*", help="the screen words; WORD=K for a guess so far")
    parser.add_argument("--best", action="store_true", help="print the best next guess and exit")
    parser.add_argument("--study", action="store_true", help="score strategies on sampled screens")
    parser.add_argument("--screens", type=int, default=1000, help="screens per row of the study")
    parser.add_argument("--published", action="store_true", help="the Fallout 4 screens")
    args = parser.parse_args(argv)
    if args.study:
        study(args.screens)
        return
    if args.published:
        published_screens()
        return
    screen, tried = [], {}
    for token in args.words:
        word, _, k = token.upper().partition("=")
        screen.append(word)
        if k:
            tried[word] = int(k)
    if len(screen) < 2:
        parser.error("give the words on the screen")
    if len({len(w) for w in screen}) != 1:
        parser.error("the words must all have the same length")
    if args.best:
        guess, sure, left = best_next_guess(screen, tried)
        print(guess)
        print(f"{left} candidates; this guess cracks {sure} of them for certain", file=sys.stderr)
        return
    terminal = Terminal(screen)
    for word, k in tried.items():
        terminal.attempted(word, k)
    terminal.automate()


if __name__ == "__main__":
    main()
