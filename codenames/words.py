"""Board words, clue words, and the rule for which clues are legal.

The **board words** are the 400 nouns in words.txt (see make_words.py). The **clue
words** are the 12,000 most frequent English words in wordfreq that are lowercase,
alphabetic, three or more letters long and in WordNet. Board words can be clues too,
when they are not on the board in play.
"""

from functools import cache
from pathlib import Path

from wordfreq import top_n_list, zipf_frequency

from codenames.fetch import fetch

HERE = Path(__file__).parent
N_CLUES = 12_000
BLOCKED = {
    "ass",
    "bitch",
    "cock",
    "crap",
    "dick",
    "fag",
    "nigger",
    "pussy",
    "shit",
    "slut",
    "tit",
    "retard",
}


@cache
def board_words() -> tuple[str, ...]:
    return tuple((HERE / "words.txt").read_text().split())


@cache
def wordnet_lemmas() -> frozenset[str]:
    d = fetch("wordnet")
    out = set()
    for pos in ("noun", "verb", "adj", "adv"):
        for line in (d / f"index.{pos}").read_text(encoding="latin-1").splitlines():
            if not line.startswith("  "):
                out.add(line.split()[0])
    return frozenset(out)


@cache
def clue_words(n: int = N_CLUES) -> tuple[str, ...]:
    """The n most frequent wordfreq words that are plain lowercase WordNet lemmas."""
    lemmas = wordnet_lemmas()
    out = []
    for w in top_n_list("en", 60_000):
        if w.isalpha() and w.isascii() and len(w) >= 3 and w in lemmas and w not in BLOCKED:
            out.append(w)
            if len(out) == n:
                break
    return tuple(out)


@cache
def vocabulary() -> list[str]:
    """Every word that needs a vector: clue words, then the board words not among them."""
    clues = clue_words()
    have = set(clues)
    return [*clues, *(w for w in board_words() if w not in have)]


def zipf(word: str) -> float:
    """Zipf frequency: log10 of occurrences per billion words (wordfreq)."""
    return zipf_frequency(word, "en")


def legal(clue: str, board: tuple[str, ...]) -> bool:
    """A clue may not be a board word, contain one, or be contained in one."""
    return all(clue != w and clue not in w and w not in clue for w in board)
