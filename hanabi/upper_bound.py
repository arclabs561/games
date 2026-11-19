"""How many deals can reach the maximum score with everything in view?

    uv run python -m hanabi.upper_bound --exact             # small decks, every deal (3 min)
    uv run python -m hanabi.upper_bound --sample 2 500      # 500 seeded two-player deals
    uv run python -m hanabi.upper_bound --sample 5 20000 --budget 200000   # a bigger run

Exact counts enumerate every deal of a small deck. A **deal** is the hands, as unordered
sets of cards, and the deck order; it stands for the orderings of the shuffled deck that
produce it, so each one is weighted by how many orderings of the hands give the same sets.
The weights add up to the number of distinct orderings of the deck.

Each deal gets a **verdict**: won (the solver's line, replayed by the rules engine, scores
the maximum), lost with a certificate (the time, storage or pooled bound of
`hanabi.solver`), lost by exhaustive search (no certificate), or undecided (the node budget
ran out, which happens only in the sampled full game).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from math import factorial

from hanabi.rules import Rules, card_name, full_deck, seeded_deck
from hanabi.solver import INF, SCHEDULE, brute_winnable, replay, root_certificates, winnable


def multiset_orders(cards: list[int]):
    """Every distinct ordering of a multiset, in lexicographic order."""
    counts = Counter(cards)
    keys = sorted(counts)
    n = len(cards)
    out: list[int] = []

    def rec():
        if len(out) == n:
            yield list(out)
            return
        for k in keys:
            if counts[k]:
                counts[k] -= 1
                out.append(k)
                yield from rec()
                out.pop()
                counts[k] += 1

    yield from rec()


def multisets(counts: Counter, size: int):
    """Every sub-multiset of the given size, as a sorted list."""
    keys = sorted(k for k in counts if counts[k])

    def rec(i, left):
        if left == 0:
            yield []
            return
        if i == len(keys):
            return
        k = keys[i]
        for take in range(min(counts[k], left), -1, -1):
            for rest in rec(i + 1, left - take):
                yield [k] * take + rest

    yield from rec(0, size)


def arrangements(hand: list[int]) -> int:
    """Number of orderings of a hand: h! / product of (copies of each card)!."""
    w = factorial(len(hand))
    for n in Counter(hand).values():
        w //= factorial(n)
    return w


def deals(rules: Rules):
    """Every deal (hands as sorted lists, then the deck order) with its weight."""
    pool = Counter(full_deck(rules))

    def rec(p, left: Counter):
        if p == rules.players:
            for order in multiset_orders(list(left.elements())):
                yield [], order, 1
            return
        for hand in multisets(left, rules.hand):
            rest = left.copy()
            rest.subtract(hand)
            for hands, order, w in rec(p + 1, rest):
                yield [hand, *hands], order, w * arrangements(hand)

    for hands, order, w in rec(0, pool):
        yield [c for h in hands for c in h] + order, w


def total_orders(rules: Rules) -> int:
    n = factorial(rules.deck_size)
    for k in Counter(full_deck(rules)).values():
        n //= factorial(k)
    return n


# Searches tried in turn, each with the full node budget: limited discrepancy search with
# passes before useful discards, plain depth-first search, then passes after discards.
ATTEMPTS = ((SCHEDULE, True), ((INF,), True), (SCHEDULE, False))


def verdict(rules: Rules, deck: list[int], budget: int = 0) -> tuple[str, list[str]]:
    """('won' | 'lost' | 'undecided', certificates that hold at the start)."""
    certs = root_certificates(rules, deck)
    if certs:
        return "lost", certs
    for schedule, pass_first in ATTEMPTS:
        ok, line = winnable(rules, deck, budget, pass_first=pass_first, schedule=schedule)
        if ok:
            assert replay(rules, deck, line) == rules.max_score, "winning line does not replay"
            return "won", []
        if ok is False:
            return "lost", []
    return "undecided", []


def classify(rules: Rules, seed: int, budgets=(20000, 300000)) -> dict:
    """The verdict on seeded deal `seed`, trying each node budget in turn until one settles
    it. For a lost deal, also the best score it allows (an upper bound when a search ran
    out of budget)."""
    deck = seeded_deck(rules, seed)
    t0 = time.time()
    for budget in budgets:
        v, certs = verdict(rules, deck, budget)
        if v != "undecided":
            break
    out = {"seed": seed, "verdict": v, "certificates": certs, "budget": budget}
    if v == "lost":
        out["best"] = best_score(rules, deck, budgets[-1])
    out["seconds"] = round(time.time() - t0, 2)
    return out


def run_slice(players: int, start: int, n: int, budgets, path: str):
    """Classify seeds start, ..., start + n - 1 and write one JSON record per line."""
    rules = Rules(players=players)
    with open(path, "w") as f:
        for seed in range(start, start + n):
            rec = classify(rules, seed, budgets)
            f.write(json.dumps({"players": players, **rec}, sort_keys=True) + "\n")
            f.flush()


EXACT_VARIANTS = [
    # van den Bergh (2015), Appendix A, row (3,2,2,2,1): one player, no hints, any discard.
    *(
        (
            f"one color, one player, hand {m}, no hints",
            Rules(
                players=1,
                colors=1,
                hand_size=m,
                max_hints=0,
                lives=1,
                discard_when_full=True,
            ),
        )
        for m in range(1, 6)
    ),
    ("one color, two players, hand 2", Rules(players=2, colors=1, hand_size=2)),
    ("one color, two players, hand 3", Rules(players=2, colors=1, hand_size=3)),
    (
        "two colors of ranks 1-3, two players, hand 4",
        Rules(colors=2, counts=(3, 2, 1), hand_size=4),
    ),
]
MORE_VARIANTS = [  # about six minutes; every deal is winnable
    (
        "two colors of ranks 1-3, two players, hand 3",
        Rules(colors=2, counts=(3, 2, 1), hand_size=3),
    ),
]


def exact(check: int = 0, more: bool = False):
    """Count every deal of each small variant. With check > 0, also run the brute-force
    engine on every check-th deal and require agreement."""
    for name, rules in EXACT_VARIANTS + (MORE_VARIANTS if more else []):
        t0 = time.time()
        tally: Counter[str] = Counter()
        by_cert: Counter[str] = Counter()
        examples: list[str] = []
        total = 0
        for i, (deck, w) in enumerate(deals(rules)):
            v, certs = verdict(rules, deck)
            total += w
            tally[v] += w
            key = "+".join(certs) if certs else ("search" if v == "lost" else "")
            if v == "lost":
                by_cert[key] += w
                if not certs and len(examples) < 3:
                    examples.append(" ".join(card_name(c, rules) for c in deck))
            if check and i % check == 0:
                assert brute_winnable(rules, deck) == (v == "won"), deck
        assert total == total_orders(rules), "weights do not add up"
        share = tally["won"] / total
        print(
            f"{name}: {tally['won']:,} of {total:,} orderings winnable ({share:.4f}); {time.time() - t0:.0f} s"
        )
        if by_cert:
            print("  lost, by first proof:", dict(sorted(by_cert.items())))
        for e in examples:
            print("  lost only by search:", e)


def best_score(rules: Rules, deck: list[int], budget: int) -> int:
    """The highest score reachable on a lost deal, trying targets from the top down. A
    target whose search runs out of budget counts as reachable, so this is an upper bound."""
    for target in range(rules.max_score - 1, 0, -1):
        ok, line = winnable(rules, deck, budget, target=target)
        if ok:
            assert replay(rules, deck, line) >= target
        if ok is not False:
            return target
    return 0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def sample(players: int, n: int, budget: int, start: int = 0, seeds=None):
    """Seeded deals start, ..., start + n - 1 of the standard game, or the given seeds."""
    seeds = list(seeds) if seeds else list(range(start, start + n))
    n = len(seeds)
    rules = Rules(players=players)
    t0 = time.time()
    tally: Counter[str] = Counter()
    by_cert: Counter[str] = Counter()
    hard = []
    best: dict[int, int] = {}  # seed -> highest score reachable, for lost deals
    for seed in seeds:
        deck = seeded_deck(rules, seed)
        v, certs = verdict(rules, deck, budget)
        tally[v] += 1
        if v == "lost":
            by_cert["+".join(certs) if certs else "search"] += 1
            best[seed] = best_score(rules, deck, budget)
        if v != "won" and not certs:
            hard.append((seed, v))
    won, und = tally["won"], tally["undecided"]
    lo, _ = wilson(won, n)
    _, hi = wilson(won + und, n)
    print(
        f"{players} players, {n} deals from seed {seeds[0]}, budget {budget:,} nodes: "
        f"won {won}, lost {tally['lost']}, undecided {und}; "
        f"winnable share between {won / n:.4f} and {(won + und) / n:.4f}, "
        f"95% interval [{lo:.4f}, {hi:.4f}]; {time.time() - t0:.0f} s"
    )
    print("  lost, by first proof:", dict(sorted(by_cert.items())))
    print("  without certificate:", hard)
    print("  best scores of lost deals:", dict(sorted(Counter(best.values()).items())))
    # the mean of the best score over all deals, counting undecided deals as 25
    scores = [25] * (won + und) + list(best.values())
    m = sum(scores) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in scores) / (n - 1))
    print(f"  mean best score at most {m:.4f} (± {1.96 * sd / math.sqrt(n):.4f}, 95%)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--more", action="store_true", help="also the slower exact variants")
    parser.add_argument("--check", type=int, default=0, help="brute-force every k-th deal")
    parser.add_argument("--sample", type=int, nargs=2, metavar=("PLAYERS", "DEALS"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="*", help="these seeds instead of a range")
    parser.add_argument("--budget", type=int, default=20000, help="search nodes per deal")
    parser.add_argument(
        "--jsonl", metavar="PATH", help="with --sample: write one record per deal to PATH"
    )
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[20000, 300000], help="with --jsonl"
    )
    args = parser.parse_args()
    if args.sample and args.jsonl:
        run_slice(args.sample[0], args.start, args.sample[1], args.budgets, args.jsonl)
        return
    if args.exact:
        exact(args.check, args.more)
    if args.sample:
        sample(args.sample[0], args.sample[1], args.budget, args.start, args.seeds)


if __name__ == "__main__":
    main()
