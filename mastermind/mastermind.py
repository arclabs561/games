from collections import Counter
from random import randint


class Mastermind:
    def __init__(self, colors=6, holes=4):
        if colors < 2:
            raise ValueError("colors must be >= 2")
        self.colors = colors
        self.holes = holes
        self.code = self.randcode()
        self.code_counts = Counter(self.code)

    def randcode(self):
        return [randint(1, self.colors) for _ in range(self.holes)]

    def grade(self, guess):
        exact = sum(i == j for i, j in zip(guess, self.code))
        guess_counts = Counter(guess)
        near = sum(min(guess_counts[i], self.code_counts[i]) for i in range(self.colors)) - exact
        return exact, near


if __name__ == "__main__":
    mm = Mastermind()
    print(mm.code)
    print(mm.grade([1, 2, 3, 4]))
