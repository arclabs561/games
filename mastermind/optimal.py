"""The best possible Mastermind strategy, by exhaustive search.

The **cost** of a strategy is the total number of guesses it needs over all codes, each code
counted once; the average is the cost divided by the number of codes. For a set S of
candidates (the codes consistent with the replies so far; `cand` in the code), the least
cost satisfies

    cost(S) = |S| + min over guesses g of the sum, over the replies r other than a win,
              of cost(S_r),

where S_r holds the codes in S that would give reply r to g. Every code in S pays for the
guess g; the code equal to g (if it is in S) is then done, and the others go on. A **depth
limit** L adds the condition that no code needs more than L guesses.

The search tries every one of the colors^holes codes as a guess, so it also finds guesses that
cannot be the secret. Three things make it fast enough to run here:

- **Symmetry.** Permuting the positions and permuting the colors maps codes to codes and keeps
  every reply. Two sets of codes that one such map carries onto each other have the same cost,
  so each set is stored once, under its least image (`canonical`). Guesses that a map fixing
  the set carries onto each other give the same cost, so one guess per class is tried.
- **Bounds.** A set of n codes costs at least 2n - 1 (one code can be solved by the first
  guess, the rest need two or more), and more once n exceeds what a tree with at most
  `replies - 1` branches per guess can hold at depth 2. A guess is skipped when the bound on its
  parts already reaches the best cost found.
- **Memoization** of each set's cost, or of a lower bound when the search was cut off.

`brute_force_cost` is the obvious recursion without any of the three, for checking small games.
`strategy_tree` reads an optimal strategy back out of the search, and `play` checks a tree by
playing it against every code with `mastermind.score`, sharing nothing else with the search.

    uv run python -m mastermind.optimal            # 6 colors, 4 pegs; about 10 minutes
    uv run python -m mastermind.optimal 4 3        # 4 colors, 3 pegs; under a second
"""

import itertools
import time
from collections import Counter
from functools import cache

import numpy as np

from mastermind.mastermind import score

INF = 10**9

type Code = tuple[int, ...]
type Tree = tuple[int, dict[tuple[int, int], "Tree"]]  # (guess index, reply -> subtree)


class Game:
    """Codes of `holes` pegs in `colors` colors, with every reply precomputed."""

    def __init__(self, colors: int, holes: int, canonical_from: int = 6):
        self.colors, self.holes = colors, holes
        self.codes: list[Code] = list(itertools.product(range(colors), repeat=holes))
        index = {c: i for i, c in enumerate(self.codes)}
        self.n = len(self.codes)
        # Replies (right place, right color in the wrong place), numbered 0..K-1.
        self.reply_list = [
            (b, w)
            for b in range(holes + 1)
            for w in range(holes + 1 - b)
            if (b, w) != (holes - 1, 1)
        ]
        number = {r: i for i, r in enumerate(self.reply_list)}
        self.k = len(self.reply_list)
        self.win = number[(holes, 0)]
        # R[g, s] is the number of the reply to guess g when the secret is s.
        self.R = np.array(
            [[number[score(s, g)] for s in self.codes] for g in self.codes], dtype=np.int64
        )
        # Every map that permutes positions and colors, as a permutation of code indices.
        self.maps = np.array(
            [
                [index[tuple(cp[c[p[i]]] for i in range(holes))] for c in self.codes]
                for p in itertools.permutations(range(holes))
                for cp in itertools.permutations(range(colors))
            ],
            dtype=np.int32,
        )
        self.canonical_from = canonical_from
        self.branches = self.k - 1
        self.lower = np.array(
            [[self._capacity_bound(m, d) for m in range(self.n + 1)] for d in range(13)],
            dtype=np.int64,
        )
        self.memo: dict[tuple[bytes, int], tuple[int, int | None]] = {}
        self.nodes = 0

    def _capacity_bound(self, m: int, limit: int) -> int:
        """Least cost of m codes in a tree with at most `branches` non-winning replies per
        guess and depth at most `limit`, or INF if no such tree holds m codes."""
        total, depth, width = 0, 1, 1
        while m > 0:
            if depth > limit:
                return INF
            k = min(m, width)
            total += k * depth
            m -= k
            depth += 1
            width *= self.branches
        return total

    def canonical(self, cand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The least image of the candidates under the symmetries, and the maps fixing it."""
        images = np.sort(self.maps[:, cand], axis=1)
        least = images[np.lexsort(images.T[::-1])[0]]
        onto = self.maps[(images == least).all(1)]  # maps carrying cand onto `least`
        return least, onto[:, np.argsort(onto[0])]  # composed with the inverse of the first

    def parts(self, cand: np.ndarray, g: int) -> list[np.ndarray]:
        """The candidates split by their reply to g, leaving out the winning reply."""
        row = self.R[g, cand]
        order = np.argsort(row, kind="stable")
        sr = row[order]
        cuts = (np.flatnonzero(sr[1:] != sr[:-1]) + 1).tolist()
        bounds = zip([0, *cuts], [*cuts, len(cand)], strict=True)
        return [cand[order[a:b]] for a, b in bounds if sr[a] != self.win]

    def guess_order(
        self, cand: np.ndarray, limit: int, stabilizer: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Each guess's lower bound on the cost of the candidates, and the guesses in the
        order to try them: lowest bound first, then more parts, then a candidate."""
        n = len(cand)
        offsets = (np.arange(self.n) * self.k)[:, None]
        counts = np.bincount((offsets + self.R[:, cand]).ravel(), minlength=self.n * self.k)
        counts = counts.reshape(self.n, self.k)
        wins = counts[:, self.win].copy()
        counts[:, self.win] = 0
        bounds = n + self.lower[limit - 1][counts].sum(1)
        bounds[counts.max(1) >= n] = INF  # the guess tells nothing
        if stabilizer is not None and len(stabilizer) > 1:
            bounds[stabilizer.min(0) != np.arange(self.n)] = INF  # not its class's least
        return np.lexsort((-wins, -(counts > 0).sum(1), bounds)), bounds

    def cost(self, cand: np.ndarray, limit: int = 12, beta: int = INF) -> int:
        """The least cost of the candidates within `limit` guesses each. If that is at least
        `beta`, the result is only a lower bound that is at least `beta`."""
        n = len(cand)
        if n == 1:
            return 1 if limit >= 1 else INF
        if limit <= 1:
            return INF
        if n == 2:
            return 3
        if n <= self.k:  # a candidate whose replies tell the others apart gives 2n - 1
            sub = np.sort(self.R[np.ix_(cand, cand)], axis=1)
            if (np.diff(sub, axis=1) != 0).all(1).any():
                return 2 * n - 1
        stabilizer = None
        if n >= self.canonical_from:
            cand, stabilizer = self.canonical(cand)
        key = (cand.tobytes(), limit)
        lb, exact = self.memo.get(key, (0, None))
        if exact is not None:
            return exact
        if lb >= beta:
            return lb
        self.nodes += 1
        bound_of = self.lower[limit - 1]
        order, bounds = self.guess_order(cand, limit, stabilizer)
        best, found = beta, False
        for g in order:
            total = int(bounds[g])
            if total >= best:
                break
            for part in sorted(self.parts(cand, int(g)), key=len, reverse=True):
                b = int(bound_of[len(part)])
                total += self.cost(part, limit - 1, best - (total - b)) - b
                if total >= best:
                    break
            else:
                best, found = total, True
        self.memo[key] = (best, best) if found else (max(lb, best), None)
        return best

    def all_codes(self) -> np.ndarray:
        return np.arange(self.n, dtype=np.int32)

    def strategy_tree(self, limit: int = 12) -> Tree:
        """An optimal strategy within `limit` guesses, read back from the search."""

        def build(cand: np.ndarray, limit: int) -> Tree:
            target = self.cost(cand, limit)
            order, bounds = self.guess_order(cand, limit)
            for g in order.tolist():
                if bounds[g] > target:
                    break
                parts = self.parts(cand, g)
                total = len(cand)
                for p in parts:
                    total += self.cost(p, limit - 1, target - total + 1)
                    if total > target:
                        break
                if total == target:
                    return g, {
                        self.reply_list[int(self.R[g, p[0]])]: build(p, limit - 1) for p in parts
                    }
            raise AssertionError("no guess attains the cost")

        return build(self.all_codes(), limit)


def play(tree: Tree, codes: list[Code]) -> list[int]:
    """Guesses the tree needs for each secret code, replies computed afresh by `score`."""
    counts = []
    for secret in codes:
        node, depth = tree, 1
        while codes[node[0]] != secret:
            node = node[1][score(secret, codes[node[0]])]
            depth += 1
        counts.append(depth)
    return counts


def brute_force_cost(colors: int, holes: int, limit: int = 12) -> int:
    """The least cost by the plain recursion: every guess, every set, no bounds or symmetry."""
    codes = list(itertools.product(range(colors), repeat=holes))

    @cache
    def cost(cand: frozenset[Code], limit: int) -> int:
        if len(cand) == 1:
            return 1 if limit >= 1 else INF
        if limit <= 1:
            return INF
        best = INF
        for g in codes:
            groups: dict[tuple[int, int], set[Code]] = {}
            for s in cand:
                groups.setdefault(score(s, g), set()).add(s)
            if len(groups) == 1 and g not in cand:
                continue  # the guess tells nothing
            rest = [frozenset(p) for r, p in groups.items() if r != (holes, 0)]
            best = min(best, len(cand) + sum(cost(p, limit - 1) for p in rest))
        return best

    return cost(frozenset(codes), limit)


def tree_to_json(tree: Tree, codes: list[Code]) -> dict:
    """A tree as nested JSON: the guess as a string of colors, and one subtree per reply."""
    guess, children = tree
    out: dict = {"guess": "".join(str(c + 1) for c in codes[guess])}
    if children:
        out["replies"] = {f"{b},{w}": tree_to_json(t, codes) for (b, w), t in children.items()}
    return out


def tree_from_json(node: dict, codes: list[Code]) -> Tree:
    """The inverse of `tree_to_json`."""
    index = {c: i for i, c in enumerate(codes)}
    guess = index[tuple(int(ch) - 1 for ch in node["guess"])]
    children = {
        tuple(int(x) for x in reply.split(",")): tree_from_json(t, codes)
        for reply, t in node.get("replies", {}).items()
    }
    return guess, children  # type: ignore[return-value]


def summary(tree: Tree, codes: list[Code], limit: int) -> dict:
    """Totals and the distribution of guesses of a tree, played against every code."""
    counts = play(tree, codes)
    dist = Counter(counts)
    return {
        "limit": limit,
        "total": sum(counts),
        "codes": len(codes),
        "worst": max(counts),
        "first_guess": "".join(str(c + 1) for c in codes[tree[0]]),
        "distribution": [dist[k] for k in range(1, max(counts) + 1)],
    }


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("colors", type=int, nargs="?", default=6)
    parser.add_argument("holes", type=int, nargs="?", default=4)
    parser.add_argument("--out", type=Path, help="write optimal.json, the trees and totals, here")
    args = parser.parse_args(argv)
    colors, holes = args.colors, args.holes
    game = Game(colors, holes)
    print(f"{colors} colors, {holes} pegs: {game.n:,} codes, {len(game.maps):,} symmetries")
    start = time.process_time()
    least = next(d for d in range(1, 13) if game.cost(game.all_codes(), d) < INF)
    print(
        f"no strategy solves every code within {least - 1} guesses "
        f"({time.process_time() - start:.0f} s)"
    )
    record: dict = {"colors": colors, "holes": holes, "least_worst_case": least, "trees": []}
    for limit in (12, least):
        start = time.process_time()
        tree = game.strategy_tree(limit)
        row = summary(tree, game.codes, limit)
        seconds = time.process_time() - start
        print(
            f"worst case at most {limit:>2}: total {row['total']:,}, "
            f"average {row['total'] / game.n:.4f}, first guess {row['first_guess']}, "
            f"distribution {row['distribution']}, {seconds:.0f} s"
        )
        record["trees"].append({**row, "tree": tree_to_json(tree, game.codes)})
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "optimal.json").write_text(json.dumps(record, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
