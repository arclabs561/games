"""Plan C15's changed boards: human turns of Cultural Codes, shuffled and with words swapped.

A turn of Cultural Codes is a clue, the unrevealed board words in the dataset's order,
and the word the person picked. Plan C15 asks whether a guesser's agreement with the
person depends on the board as published. Three arms change the board in ways a person's
choice should not depend on:

- **O, original:** the board in the dataset's order.
- **S, shuffled:** the same words in a random order (seed 15002 and the turn's index).
- **W, swapped:** S, with every word other than the person's pick replaced by a noun of
  about the same frequency (seed 15003 and the turn's index). The clue and the pick stay.

A replacement is a word of `SWAP_POOL`: a one-word lowercase WordNet noun of 3 to 9
letters that has vectors here (it is in `evaluate.all_words()`: the clue words, the board
words and the words of Cultural Codes), so that the embedding guessers can score the
swapped boards. It has Zipf frequency within 0.1 of the word it
replaces, is not on the original board or already chosen for this one, and is not the
clue or a form of it (`legality.strict_legal` against the clue). When no word is that
close, the nearest in frequency that meets the other conditions is used, and counted.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import cache

ZIPF_WINDOW = 0.1
SHUFFLE_SEED, SWAP_SEED = 15002, 15003


@dataclass(frozen=True)
class Arm:
    """A turn as one arm shows it: the clue, the board words in order, and the pick."""

    clue: str
    words: tuple[str, ...]
    picked: str
    widened: int = 0  # replacements that needed a word outside the Zipf window


@cache
def swap_pool() -> tuple[tuple[str, float], ...]:
    """(word, Zipf frequency) of every replacement candidate, sorted by frequency."""
    from codenames.evaluate import all_words
    from codenames.legality import lemmas
    from codenames.words import zipf

    nouns = lemmas()["noun"]
    words = {
        w
        for w in all_words()
        if w.isalpha() and w.isascii() and w.islower() and 3 <= len(w) <= 9 and w in nouns
    }
    return tuple(sorted(((w, zipf(w)) for w in words), key=lambda x: (x[1], x[0])))


def shuffled(words: tuple[str, ...], index: int) -> tuple[str, ...]:
    out = list(words)
    random.Random(f"{SHUFFLE_SEED}-{index}").shuffle(out)
    return tuple(out)


def allowed(candidate: str, clue: str, taken: set[str]) -> bool:
    from codenames.legality import strict_legal

    return candidate not in taken and candidate != clue and strict_legal(candidate, (clue,))


def replacement(word: str, clue: str, taken: set[str], rng: random.Random) -> tuple[str, bool]:
    """A word for `word`, uniform among the allowed candidates within the Zipf window (the
    first allowed one in a random order of the window), or else the allowed candidate
    nearest in frequency (ties to the earlier in the pool) and True."""
    from bisect import bisect_left, bisect_right

    from codenames.words import zipf

    z = zipf(word)
    pool = swap_pool()
    freqs = [f for _, f in pool]
    eps = 1e-9  # Zipf values have two decimals; the window includes its ends
    lo, hi = bisect_left(freqs, z - ZIPF_WINDOW - eps), bisect_right(freqs, z + ZIPF_WINDOW + eps)
    window = [w for w, _ in pool[lo:hi]]
    rng.shuffle(window)
    for w in window:
        if allowed(w, clue, taken):
            return w, False
    for _, w in sorted((abs(f - z), w) for w, f in pool):
        if allowed(w, clue, taken):
            return w, True
    raise ValueError(f"no replacement for {word}")


def arms(clue: str, words: tuple[str, ...], picked: str, index: int) -> dict[str, Arm]:
    """The three arms of one turn."""
    assert picked in words, (picked, words)
    s = shuffled(words, index)
    rng = random.Random(f"{SWAP_SEED}-{index}")
    taken = set(words)
    swapped, widened = [], 0
    for w in s:
        if w == picked:
            swapped.append(w)
            continue
        new, wide = replacement(w, clue, taken, rng)
        taken.add(new)
        swapped.append(new)
        widened += wide
    return {
        "O": Arm(clue, tuple(words), picked),
        "S": Arm(clue, s, picked),
        "W": Arm(clue, tuple(swapped), picked, widened),
    }
