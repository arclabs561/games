"""Seeded games played by the strategy that a value array defines, many games at once.

The dice are drawn before play: for each game, turn and roll, a row of five dice. A reroll of j
dice uses the first j of its row. Two strategies given the same rows therefore see the same dice
whenever they reroll the same number of dice, which makes the difference of their scores far less
noisy than two independent samples would be (common random numbers).
"""

import numpy as np

from yahtzee.solver import Rule, Tables, turn


def deal(t: Tables, games: int, seed: int) -> np.ndarray:
    """(games, turns, rolls, dice) uniform faces from the seed."""
    rules = t.rules
    shape = (games, len(rules.categories), rules.rolls, rules.dice)
    return np.random.default_rng(seed).integers(1, rules.faces + 1, size=shape, dtype=np.int8)


def play(
    t: Tables, values: np.ndarray, dice: np.ndarray, batch: int = 2048, rule: Rule | None = None
) -> np.ndarray:
    """The final score of each game, playing the choices that E defines (and `rule` fixes)."""
    rules = t.rules
    n, faces = rules.dice, rules.faces
    games = len(dice)
    base = (n + 1) ** np.arange(faces)
    lookup = np.full((n + 1) ** faces, -1)
    for i, r in enumerate(t.rolls):
        lookup[sum(base[v - 1] for v in r)] = i
    keep_size = np.array([len(k) for k in t.keeps])
    keep_code = np.array([sum(base[v - 1] for v in k) for k in t.keeps])

    def add(code, row, count):
        """Add the first `count` dice of each row to the encoded counts."""
        for j in range(n):
            code = code + np.where(j < count, base[row[:, j] - 1], 0)
        return code

    state = np.full(games, t.index(0, 0, 0))
    score = np.zeros(games, dtype=np.int64)
    for k in range(len(rules.categories)):
        uniq, inv = np.unique(state, return_inverse=True)
        for lo in range(0, len(uniq), batch):
            tr = turn(t, values, uniq[lo : lo + batch], choices=True, rule=rule)
            g = np.nonzero((inv >= lo) & (inv < lo + batch))[0]
            b = inv[g] - lo
            r = lookup[add(np.zeros(len(g), dtype=np.int64), dice[g, k, 0], n)]
            for level, keeps in enumerate(tr.keeps):
                kept = keeps[b, r]
                code = add(keep_code[kept], dice[g, k, level + 1], n - keep_size[kept])
                r = lookup[code]
            score[g] += tr.points[b, r] + tr.bonus[b, r]
            state[g] = tr.next[b, r]
    total = (state // 2) % t.width
    return score + np.where((total == t.cap) & (t.cap > 0), rules.upper_bonus, 0)


def interval(x: np.ndarray) -> tuple[float, float]:
    """The mean and the half-width of its normal 95% interval."""
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))
