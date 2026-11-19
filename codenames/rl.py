"""A giver trained by policy gradient against guessers, and tested against others.

The **policy** sees one board through its own model's vectors. Its actions are a
shortlist of K clues (codenames.players.shortlist_rows, by its own similarities) times
the counts 1 to 4. Each action has a feature vector phi, computed from the giver's own
model only:

    expected team words found, chance of hitting the assassin, chance of hitting an
    opponent word (all under the giver's own Luce guesser), the count, the rarity of
    the clue, the margin between its n-th closest team word and its closest other
    word, and a constant.

The policy picks an action with probability proportional to exp(theta . phi). Its
reward is the exact expected value of the turn (codenames.luce) under a guesser drawn
from a training population. Because that reward is known for every action, the
gradient of the expected reward is computed exactly over the actions,

    grad = sum over a of pi(a) (r(a) - b) phi_bar(a),   b = sum over a of pi(a) r(a),

where phi_bar(a) = phi(a) - sum over a' of pi(a') phi(a'). This is REINFORCE with the
policy's mean reward as baseline, with the action sampling done in closed form; boards
and guessers are still sampled.

- **Self-play**: the population is the giver's own model.
- **Population play**: the population is every other model except the held-out ones.
- **Cross-play** is the score against held-out guessers: one embedding model kept out
  of training, and the EAT association counts, which stand in for people.

    uv run python -m codenames.rl --giver potion --held-out bge     # a few minutes here
    uv run python -m codenames.rl --giver potion --held-out bge --seed 1 --test-seed 3003 --per-board

The last form is one run of plan C3 (codenames.c3 runs all 70).
"""

import argparse
import json
import random
import time

import numpy as np

from codenames.game import Board, deal
from codenames.luce import ASSASSIN, OPPONENT, TEAM, luce_weights, turn_odds
from codenames.norms import eat_counts
from codenames.players import (
    ClueTable,
    Weigher,
    costs_for,
    embedding_weigher,
    expected_giver,
    norm_weigher,
    shortlist_rows,
)
from codenames.study import chosen_spaces, load
from codenames.words import board_words, clue_words, zipf

NMAX = 4
FEATURES = ("found", "assassin", "opponent", "count", "rarity", "margin", "const")


def actions(sp, tau: float, board: Board, pool, k: int):
    """(clues, phi) for the K shortlisted clues times counts 1..NMAX; phi has shape (A, F)."""
    words = board.unrevealed()
    kinds = tuple(board.kind(w) for w in words)
    tab = ClueTable.build(sp, words, pool)
    rows = shortlist_rows(tab.sims, kinds, k)
    rows = sorted(rows, key=lambda r: -tab.sims[r][[x == TEAM for x in kinds]].max())[:k]
    sims = tab.sims[rows]
    odds = turn_odds(luce_weights(sims, tau), kinds, NMAX)
    team = np.array([x == TEAM for x in kinds])
    team_sorted = -np.sort(-sims[:, team], axis=1)
    best_bad = sims[:, ~team].max(axis=1)
    feats, clues = [], []
    for i, r in enumerate(rows):
        for n in range(1, NMAX + 1):
            feats.append(
                [
                    odds.expected_team(n)[i],
                    odds.stopped_by(ASSASSIN, n)[i],
                    odds.stopped_by(OPPONENT, n)[i],
                    n,
                    max(0.0, 3.5 - zipf(tab.clues[r])),
                    team_sorted[i, n - 1] - best_bad[i],
                    1.0,
                ]
            )
            clues.append((tab.clues[r], n))
    return clues, np.array(feats)


def rewards(weigh: Weigher, board: Board, clues) -> np.ndarray:
    """Exact expected value of each (clue, count) under one guesser."""
    words = board.unrevealed()
    kinds = tuple(board.kind(w) for w in words)
    names = sorted({c for c, _ in clues})
    w = np.stack([weigh(c, words) for c in names])
    odds = turn_odds(w, kinds, NMAX)
    costs = costs_for(board)
    value = {n: odds.value(n, costs) for n in range(1, NMAX + 1)}
    pos = {c: i for i, c in enumerate(names)}
    return np.array([value[n][pos[c]] for c, n in clues])


def rewards_rlvr(weigh: Weigher, board: Board, clues) -> np.ndarray:
    """The clue reward of Deshpande et al. (2026, eq. 4) with a greedy judge that names
    its n most likely words: |T and G| / |T| - |N and G| / |N|, where T is the giver's
    unrevealed team words and N the other unrevealed words. Their third term, for
    guesses not on the board, is 0 here: these guessers only name board words."""
    words = board.unrevealed()
    team = np.array([board.kind(w) == TEAM for w in words])
    out = []
    cache: dict[str, np.ndarray] = {}
    for c, n in clues:
        if c not in cache:
            cache[c] = np.argsort(-weigh(c, words), kind="stable")
        picked = cache[c][:n]
        out.append(team[picked].sum() / team.sum() - (~team[picked]).sum() / (~team).sum())
    return np.array(out, dtype=float)


def policy(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    z = phi @ theta
    p = np.exp(z - z.max())
    return p / p.sum()


def fit(data, population: list[str], steps: int, lr: float, seed: int) -> np.ndarray:
    """Exact-over-actions policy gradient with Adam; data[b] = (phi, {guesser: rewards})."""
    rng = random.Random(seed)
    mu, sd = feature_scale(data)
    theta = np.zeros(len(FEATURES))
    theta[0] = 1.0  # start from "maximize the expected number of words found"
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    for step in range(1, steps + 1):
        grad = np.zeros_like(theta)
        for _ in range(16):
            phi, rew = data[rng.randrange(len(data))]
            x = (phi - mu) / sd
            r = rew[rng.choice(population)]
            p = policy(theta, x)
            base = p @ r
            grad += ((p * (r - base))[:, None] * (x - p @ x)).sum(axis=0)
        grad /= 16
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        theta += lr * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
    return np.concatenate([theta, mu, sd])


def feature_scale(data):
    allphi = np.concatenate([phi for phi, _ in data])
    mu, sd = allphi.mean(axis=0), allphi.std(axis=0)
    mu[-1], sd[-1] = 0.0, 1.0  # keep the constant
    sd[sd == 0] = 1.0
    return mu, sd


def score(params: np.ndarray, data, guesser: str, greedy: bool = True) -> list[float]:
    """Reward per board of the policy's most likely action (or its expected reward)."""
    f = len(FEATURES)
    theta, mu, sd = params[:f], params[f : 2 * f], params[2 * f :]
    out = []
    for phi, rew in data:
        p = policy(theta, (phi - mu) / sd)
        out.append(float(rew[guesser][int(np.argmax(p))]) if greedy else float(p @ rew[guesser]))
    return out


def build(giver: str, guessers: dict[str, Weigher], boards, k: int):
    cal = load("calibrate")
    sp, tau = chosen_spaces(cal)[giver]
    counts = eat_counts()
    pool = tuple(c for c in clue_words() if c in counts)
    data, base = [], []
    planner = expected_giver(sp, tau, candidates=pool)
    for b in boards:
        clues, phi = actions(sp, tau, b, pool, k)
        rew = {g: rewards(w, b, clues) for g, w in guessers.items()}
        rew |= {f"{g}/rlvr": rewards_rlvr(w, b, clues) for g, w in guessers.items()}
        data.append((phi, rew))
        c, n, _ = planner.plan(b)
        base.append((c, n))
    return data, base


def run(
    giver: str = "potion",
    held_out: str = "bge",
    train_boards: int = 300,
    test_boards: int = 200,
    shortlist: int = 30,
    steps: int = 400,
    lr: float = 0.05,
    seed: int = 0,
    test_seed: int | None = None,
    per_board: bool = False,
) -> dict:
    """Train the four policies and score them on the test boards.

    Training boards are dealt from seed; test boards from test_seed, which defaults to
    seed + 10_000 (the README's runs). With per_board the result also holds each
    policy's value on every test board, for every guesser, so that two policies can be
    compared board by board."""
    t0 = time.perf_counter()
    config = {k: v for k, v in locals().items() if k != "t0"}
    config["test_seed"] = seed + 10_000 if test_seed is None else test_seed
    cal = load("calibrate")
    spaces = chosen_spaces(cal)
    guessers: dict[str, Weigher] = {
        m: embedding_weigher(sp, tau) for m, (sp, tau) in spaces.items()
    }
    guessers["eat"] = norm_weigher(eat_counts(), cal["eat"]["eps"])
    rng = random.Random(seed)
    train = [deal(board_words(), rng) for _ in range(train_boards)]
    rng = random.Random(config["test_seed"])
    test = [deal(board_words(), rng) for _ in range(test_boards)]
    train_data, _ = build(giver, guessers, train, shortlist)
    test_data, _ = build(giver, guessers, test, shortlist)
    population = [g for g in guessers if g not in (giver, held_out, "eat")]
    f = len(FEATURES)
    start = np.concatenate([np.eye(f)[0], np.zeros(f), np.ones(f)])  # theta = "found"
    policies = {
        "start": start,
        "self": fit(train_data, [giver], steps, lr, seed),
        "population": fit(train_data, population, steps, lr, seed),
        "population/rlvr": fit(train_data, [f"{g}/rlvr" for g in population], steps, lr, seed),
    }
    test: dict[str, dict[str, list[float]]] = {}
    boards: dict[str, dict[str, list[float]]] = {}
    for name, params in policies.items():
        test[name] = {}
        for g in [*guessers, *(f"{g}/rlvr" for g in guessers)]:
            s = score(params, test_data, g)
            test[name][g] = [float(np.mean(s)), float(np.std(s, ddof=1) / np.sqrt(len(s)))]
        if per_board:
            boards[name] = {g: [round(x, 6) for x in score(params, test_data, g)] for g in guessers}
    result = {
        "config": config,
        "population": population,
        "features": FEATURES,
        "theta": {k: v[:f].round(4).tolist() for k, v in policies.items()},
        "test": test,
    }
    if per_board:
        result["per_board"] = boards
    result["seconds"] = round(time.perf_counter() - t0, 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--giver", default="potion")
    parser.add_argument("--held-out", default="bge")
    parser.add_argument("--train-boards", type=int, default=300)
    parser.add_argument("--test-boards", type=int, default=200)
    parser.add_argument("--shortlist", type=int, default=30)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--test-seed", type=int, default=None, help="seed of the test boards (seed + 10000)"
    )
    parser.add_argument(
        "--per-board", action="store_true", help="also write each test board's value"
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    kw = {k: v for k, v in vars(args).items() if k != "out"}
    result = run(**kw)
    result["config"]["out"] = args.out
    text = json.dumps(result, indent=1)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
