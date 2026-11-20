"""Plug a language model in as giver or guesser. No model is called here.

`complete` is any function from a prompt to the model's reply text: a wrapper around an
API client, a local model, or a test stub. The two functions below turn it into a
giver and a guesser with the signatures of codenames.game, so an LLM can play against
the embedding players in codenames.game.play.

The giver sees the board with each word's kind; the guesser sees only the clue, the
count and the unrevealed words. Replies are parsed leniently, and a reply that cannot
be parsed, or names an illegal clue, falls back to a fixed rule (count 1; words in
board order) so a game always finishes. The number of fallbacks is worth reporting.
"""

import re
from collections.abc import Callable

from codenames.game import Board, Clue, Giver, Guesser
from codenames.words import legal

Complete = Callable[[str], str]


def giver_prompt(board: Board) -> str:
    names = {"T": "yours", "O": "opponent", "N": "neutral", "A": "ASSASSIN"}
    lines = [f"- {w}: {names[board.kind(w)]}" for w in board.unrevealed()]
    return (
        "You are the spymaster in Codenames. Give a one-word clue and a number: how many "
        "of your words the clue points to. Your teammate guesses words one at a time and "
        "stops at the first word that is not yours; the assassin loses the game. The clue "
        "may not be a board word or contain one.\n\nBoard:\n"
        + "\n".join(lines)
        + "\n\nAnswer on one line as: CLUE NUMBER"
    )


def guesser_prompt(clue: str, count: int, words: tuple[str, ...]) -> str:
    return (
        f"You are guessing in Codenames. The clue is '{clue}' for {count} word(s). "
        "List every board word, most likely first, one per line, nothing else.\n\n"
        "Board words:\n" + "\n".join(words)
    )


# Plan C12: the whole clue rule, the exact scoring, and one retry.
RELATION_WORDS = {
    "same": "it is that word",
    "substring": "one of the two words contains the other",
    "inflection": "it is a form of that word (a plural, a tense or another inflection)",
    "derivation": "it is derived from that word, or that word from it",
    "variant": "it is another spelling of that word",
    "compound": "it shares a part of a compound word with that word",
    "stem": "it has the same stem as that word",
    "affix": "it is that word with another suffix",
    "compound3": "it shares a part of a compound word with that word",
    "prefix": "it starts the same way as that word",
}


def giver_prompt_full(board: Board) -> str:
    """C4's giver prompt with the rulebook's clue rule, this folder's scoring and the
    guesser's limit of four words stated."""
    names = {"T": "yours", "O": "opponent", "N": "neutral", "A": "ASSASSIN"}
    lines = [f"- {w}: {names[board.kind(w)]}" for w in board.unrevealed()]
    return (
        "You are the spymaster in Codenames. Give a one-word clue and a number: how many "
        "of your words the clue points to.\n\n"
        "The clue rule: the clue is one English word. It may not be a board word, and it "
        "may not be any form of a board word: not a plural, another tense, a word derived "
        "from it (such as broken or breakage for BREAK) or another spelling of it (such as "
        "theatre for THEATER). It may not contain a board word or be contained in one, and "
        "it may not be part of a compound board word or share a part with one (such as "
        "horse, shoe or snowshoe for HORSESHOE). Avoid clues that share a stem or most of "
        "their first letters with a board word.\n\n"
        "How the turn is played and scored: your teammate guesses up to the number, one "
        "word at a time, and the turn ends at the first word that is not yours. Each of "
        "your words found scores +1. An opponent word scores -1 and ends the turn. A "
        "neutral word scores 0 and ends the turn. The assassin scores minus every one of "
        "your words still unfound, and ends the turn. Your teammate never guesses more "
        "than 4 words, so a number above 4 counts as 4.\n\nBoard:\n"
        + "\n".join(lines)
        + "\n\nAnswer on one line as: CLUE NUMBER"
    )


def retry_message(clue: str, flags: list[tuple[str, str]]) -> str:
    """The one retry after a clue that the strict check refuses: the board words it is
    related to, and how."""
    why = "; ".join(f"board word {w.upper()}: {RELATION_WORDS[r]}" for w, r in flags)
    return (
        f"The clue {clue.upper()} is not allowed: {why}. Give another clue that follows "
        "the clue rule. Answer on one line as: CLUE NUMBER"
    )


def parse_clue(reply: str, board: Board) -> Clue | None:
    m = re.search(r"([A-Za-z]+)\W+(\d+)", reply)
    if not m:
        return None
    word, n = m[1].lower(), int(m[2])
    if not legal(word, board.unrevealed()) or n < 1:
        return None
    return Clue(word, n)


def parse_ranking(reply: str, words: tuple[str, ...]) -> list[str]:
    seen: list[str] = []
    for token in re.findall(r"[A-Za-z]+", reply.lower()):
        if token in words and token not in seen:
            seen.append(token)
    return seen + [w for w in words if w not in seen]


def llm_giver(complete: Complete, fallback: Giver) -> Giver:
    def give(board: Board) -> Clue:
        return parse_clue(complete(giver_prompt(board)), board) or fallback(board)

    return give


def llm_guesser(complete: Complete) -> Guesser:
    def guess(clue: str, count: int, words: tuple[str, ...]) -> list[str]:
        return parse_ranking(complete(guesser_prompt(clue, count, words)), words)

    return guess
