# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Monte Carlo dice simulation with reservoir sampling and Gumbel-max trick.

Simulates a Tenzi-like game (roll N dice, keep the most frequent face, reroll
the rest) over millions of runs. Uses reservoir sampling to track the top-k
luckiest games without storing all results. Demonstrates the Gumbel-max trick
for weighted die sampling.

kuji equivalents:
  - Reservoir sampling: kuji::reservoir::ReservoirSampler (Algorithm R / L)
  - Gumbel-max trick: kuji::gumbel::gumbel_max_sample (categorical sampling
    without normalization, O(k) for top-k via partial sort of Gumbel variates)
  - Monte Carlo: kuji provides seeded RNG wrappers for reproducible simulations
  - Top-k tracking: kuji::topk::TopK (bounded heap, O(n log k))
"""

from __future__ import annotations

import heapq
import math
import random
import sys
import time
from collections import Counter


# ---------------------------------------------------------------------------
# Tenzi-like dice game
# ---------------------------------------------------------------------------
def tenzi_game(n_dice: int = 10, rng: random.Random | None = None) -> int:
    """Play one Tenzi game. Returns the number of rolls to get all dice matching.

    Strategy: keep the most frequent face after the first roll. On subsequent
    rolls, only reroll dice that don't match the kept face.
    """
    r = rng or random.Random()
    dice = [r.randint(1, 6) for _ in range(n_dice)]
    counts = Counter(dice)
    target, matched = counts.most_common(1)[0]
    rolls = 1

    while matched < n_dice:
        reroll = n_dice - matched
        new_dice = [r.randint(1, 6) for _ in range(reroll)]
        matched += sum(1 for d in new_dice if d == target)
        rolls += 1

    return rolls


# ---------------------------------------------------------------------------
# Reservoir sampling (Algorithm R)
# ---------------------------------------------------------------------------
class ReservoirSampler:
    """Maintains a uniform random sample of size k from a stream of items.

    Algorithm R (Vitter, 1985): each of the first k items goes into the
    reservoir. For the i-th item (i > k), include it with probability k/i,
    replacing a random existing item.

    This is equivalent to kuji::reservoir::ReservoirSampler.
    """

    def __init__(self, k: int, rng: random.Random | None = None):
        self.k = k
        self.reservoir: list[tuple[float, object]] = []  # (score, item)
        self.n_seen = 0
        self.rng = rng or random.Random()

    def push(self, score: float, item: object) -> None:
        self.n_seen += 1
        if len(self.reservoir) < self.k:
            heapq.heappush(self.reservoir, (score, item))
        elif score > self.reservoir[0][0]:
            heapq.heapreplace(self.reservoir, (score, item))

    def items(self) -> list[tuple[float, object]]:
        """Return reservoir contents sorted by score descending."""
        return sorted(self.reservoir, key=lambda x: -x[0])


# ---------------------------------------------------------------------------
# Top-k tracker (min-heap)
# ---------------------------------------------------------------------------
class TopKTracker:
    """Track the top-k items by score using a bounded min-heap.

    Equivalent to kuji::topk::TopK. O(n log k) for n insertions.
    """

    def __init__(self, k: int):
        self.k = k
        self.heap: list[tuple[float, int, object]] = []  # (score, tiebreak, item)
        self._counter = 0

    def push(self, score: float, item: object) -> None:
        self._counter += 1
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (score, self._counter, item))
        elif score > self.heap[0][0]:
            heapq.heapreplace(self.heap, (score, self._counter, item))

    def items(self) -> list[tuple[float, object]]:
        return sorted([(s, item) for s, _, item in self.heap], key=lambda x: -x[0])


# ---------------------------------------------------------------------------
# Gumbel-max trick for weighted die sampling
# ---------------------------------------------------------------------------
def gumbel_max_sample(weights: list[float], rng: random.Random | None = None) -> int:
    """Sample a categorical index using the Gumbel-max trick.

    For each category i with log-weight log(w_i), add independent Gumbel(0,1)
    noise: G_i = log(w_i) + gumbel_noise. Return argmax(G_i).

    This avoids computing the normalizing constant (sum of weights), which is
    the key advantage for streaming/approximate settings. Equivalent to
    kuji::gumbel::gumbel_max_sample.
    """
    r = rng or random.Random()
    best_idx = 0
    best_val = -math.inf
    for i, w in enumerate(weights):
        if w <= 0:
            continue
        # Gumbel(0,1) = -log(-log(U)), U ~ Uniform(0,1)
        u = r.random()
        while u == 0.0:  # avoid log(0)
            u = r.random()
        gumbel = -math.log(-math.log(u))
        val = math.log(w) + gumbel
        if val > best_val:
            best_val = val
            best_idx = i
    return best_idx


def verify_gumbel_sampling(n_samples: int = 50_000) -> None:
    """Verify the Gumbel-max trick produces correct categorical frequencies."""
    weights = [1.0, 2.0, 3.0, 0.5, 0.5]
    total = sum(weights)
    expected = [w / total for w in weights]

    counts = [0] * len(weights)
    rng = random.Random(42)
    for _ in range(n_samples):
        idx = gumbel_max_sample(weights, rng)
        counts[idx] += 1

    observed = [c / n_samples for c in counts]
    print("=== Gumbel-max trick verification ===")
    print("  weights:", weights)
    print(f"  {'face':<6s} {'expected':>10s} {'observed':>10s} {'error':>10s}")
    max_error = 0.0
    for i, (e, o) in enumerate(zip(expected, observed)):
        err = abs(e - o)
        max_error = max(max_error, err)
        print(f"  {i:<6d} {e:>10.4f} {o:>10.4f} {err:>10.4f}")
    print(f"  max absolute error: {max_error:.4f}")
    if max_error < 0.02:
        print("  PASSED: frequencies match within tolerance\n")
    else:
        print("  WARNING: frequencies diverge (may need more samples)\n")


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------
def run_simulation(
    n_games: int = 1_000_000,
    n_dice: int = 10,
    top_k: int = 10,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)

    # Track luckiest (fewest rolls) and unluckiest (most rolls) games.
    luckiest = TopKTracker(top_k)
    unluckiest = TopKTracker(top_k)

    # Reservoir sample: uniform random sample of game lengths.
    reservoir = ReservoirSampler(k=100, rng=rng)

    total_rolls = 0
    min_rolls = n_dice + 1
    max_rolls = 0

    t0 = time.monotonic()
    for game_id in range(n_games):
        rolls = tenzi_game(n_dice, rng)
        total_rolls += rolls
        min_rolls = min(min_rolls, rolls)
        max_rolls = max(max_rolls, rolls)

        # For "luckiest", we want lowest roll count. Negate for max-heap.
        luckiest.push(-rolls, game_id)
        unluckiest.push(rolls, game_id)

        # Reservoir sampling with roll count as score.
        reservoir.push(rng.random(), (game_id, rolls))

        if (game_id + 1) % 200_000 == 0:
            elapsed = time.monotonic() - t0
            rate = (game_id + 1) / elapsed
            print(f"  {game_id + 1:>10,d} games  ({rate:,.0f} games/sec)")

    elapsed = time.monotonic() - t0
    avg_rolls = total_rolls / n_games

    print(f"\n=== Tenzi Monte Carlo ({n_games:,d} games, {n_dice} dice) ===")
    print(f"  time:      {elapsed:.2f}s ({n_games / elapsed:,.0f} games/sec)")
    print(f"  avg rolls: {avg_rolls:.3f}")
    print(f"  min rolls: {min_rolls}")
    print(f"  max rolls: {max_rolls}")

    print(f"\n=== Top-{top_k} luckiest games (fewest rolls) ===")
    for neg_score, game_id in luckiest.items():
        print(f"  game {game_id:>8,d}: {-neg_score:.0f} rolls")

    print(f"\n=== Top-{top_k} unluckiest games (most rolls) ===")
    for score, game_id in unluckiest.items():
        print(f"  game {game_id:>8,d}: {score:.0f} rolls")

    print("\n=== Reservoir sample (10 of 100 uniformly sampled games) ===")
    sample = reservoir.items()[:10]
    for _, (game_id, rolls) in sample:
        print(f"  game {game_id:>8,d}: {rolls} rolls")


def main() -> None:
    n_games = 1_000_000
    if "--fast" in sys.argv:
        n_games = 100_000

    verify_gumbel_sampling()
    run_simulation(n_games=n_games)


if __name__ == "__main__":
    main()
