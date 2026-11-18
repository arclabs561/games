"""Mastermind codes and the codemaker's reply."""

from collections.abc import Sequence
from random import randint


def score(code: Sequence[int], guess: Sequence[int]) -> tuple[int, int]:
    """The reply to `guess` when the secret is `code`: (right color in the right place,
    right color in the wrong place). Each peg of the guess is matched at most once."""
    exact = sum(c == g for c, g in zip(code, guess, strict=True))
    common = sum(min(code.count(k), guess.count(k)) for k in set(guess))
    return exact, common - exact


assert score((1, 1, 2, 2), (1, 1, 2, 2)) == (4, 0)
assert score((1, 2, 3, 4), (4, 3, 2, 1)) == (0, 4)
assert score((1, 1, 1, 2), (1, 2, 2, 2)) == (2, 0)  # extra 2s in the guess earn nothing


class Mastermind:
    def __init__(self, colors=6, holes=4):
        if colors < 2:
            raise ValueError("colors must be >= 2")
        self.colors = colors
        self.holes = holes
        self.code = self.randcode()

    def randcode(self):
        return [randint(1, self.colors) for _ in range(self.holes)]

    def grade(self, guess):
        return score(self.code, guess)


if __name__ == "__main__":
    mm = Mastermind()
    print(mm.code)
    print(mm.grade([1, 2, 3, 4]))
