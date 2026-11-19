"""Contrastive training of a clue-side linear map on top of frozen word vectors.

The model scores a clue c against a word w as (A e_c) . e_w, where e_c and e_w are the
frozen unit vectors of a static model and A is a d x d matrix that starts as the
identity. Each training item is a clue, the word a person connected to it, and 24 other
words; the loss is InfoNCE, the cross-entropy of a softmax over the 25 words at
temperature 0.05, weighted by how many people made the connection. This is the setup of
White et al. (2024), who train a linear layer on GloVe from Codenames Duet games.

Items come from two human sources, never from test data:
- Cultural Codes train split: the clue, the word the person picked, and the other words
  on that board as hard negatives.
- EAT, cues not held out (codenames.norms.held_out): each one-word response, weighted by
  its count, against 24 random board words that are not responses to that cue.

    uv run python -m codenames.train --model potion          # under a minute here

Larger models (a LoRA fine-tune of Qwen3-Embedding-0.6B or harrier-oss-v1-0.6b) are in
codenames/lora.py, for a machine with a GPU.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from codenames.embed import normalize
from codenames.evaluate import (
    TAUS,
    all_words,
    association_items,
    association_top1,
    guess_loglik,
    guess_scores,
)
from codenames.norms import cultural_guesses, eat_counts, held_out
from codenames.players import Space, space
from codenames.words import board_words

TAU = 0.05
RESULTS = Path(__file__).parent / "results"


def items(
    sp: Space,
    seed: int = 0,
    eat: bool = True,
    cultural: bool = True,
    split="train",
    scheme: str = "published",
):
    """(clue index, candidate indices with the positive first, weight) for training.

    split and scheme name the Cultural Codes turns to train on (see
    norms.cultural_guesses); split may also be a sequence of Guess."""
    rng = random.Random(seed)
    idx = sp.index
    out = []
    if cultural:
        for g in cultural_guesses(split, scheme) if isinstance(split, str) else split:
            if g.clue in idx and all(w in idx for w in g.words) and len(g.words) >= 2:
                others = [idx[w] for w in g.words if w != g.picked]
                cand = [idx[g.picked], *rng.sample(others, min(24, len(others)))]
                while len(cand) < 25:
                    cand.append(cand[-1])  # pad short boards by repeating a negative
                out.append((idx[g.clue], cand, 1.0))
    if eat:
        board = [w for w in board_words() if w in idx]
        for cue, row in sorted(eat_counts().items()):
            if held_out(cue) or cue not in idx:
                continue
            pool = [idx[w] for w in board if w not in row and w != cue]
            for resp, n in sorted(row.items()):
                if resp in idx and resp != cue and n >= 2:
                    out.append((idx[cue], [idx[resp], *rng.sample(pool, 24)], n / 100))
    return out


def loss_and_grad(a, e, f, clue, cand, weight, l2: float, n_total: int):
    """InfoNCE loss of one batch (weighted mean) and its gradient with respect to A,
    plus the batch's share of the L2 penalty l2 * |A - I|^2 / 2."""
    d = a.shape[0]
    q = e[clue] @ a.T  # (B, d)
    words = f[cand]  # (B, 25, d)
    logits = np.einsum("bd,bkd->bk", q, words) / TAU
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    wb = weight / weight.sum()
    share = len(clue) / n_total
    loss = float(-(wb * np.log(p[:, 0] + 1e-300)).sum())
    loss += share * l2 * float(((a - np.eye(d)) ** 2).sum()) / 2
    p[:, 0] -= 1.0
    dq = np.einsum("bk,bkd->bd", p, words) / TAU * wb[:, None]
    grad = dq.T @ e[clue] + share * l2 * (a - np.eye(d))
    return loss, grad


def train(sp: Space, data, epochs: int = 3, lr: float = 0.01, l2: float = 1.0, seed: int = 0):
    """Adam on A, with an L2 penalty l2 * |A - I|^2 / 2 toward the identity."""
    rng = np.random.default_rng(seed)
    e = sp.clue_vecs.astype(np.float64)
    f = sp.word_vecs.astype(np.float64)
    a = np.eye(e.shape[1])
    m = np.zeros_like(a)
    v = np.zeros_like(a)
    clue = np.array([c for c, _, _ in data])
    cand = np.array([k for _, k, _ in data])
    weight = np.array([w for _, _, w in data])
    step = 0
    history = []
    for _ in range(epochs):
        order = rng.permutation(len(data))
        total = 0.0
        for start in range(0, len(order), 512):
            b = order[start : start + 512]
            loss, grad = loss_and_grad(a, e, f, clue[b], cand[b], weight[b], l2, len(order))
            total += loss * len(b)
            step += 1
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad**2
            a -= lr * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
        history.append(total / len(order))
    return a, history


def adapted(sp: Space, a: np.ndarray, name: str) -> Space:
    """The space with clue vectors mapped by A and renormalized."""
    clue_vecs = normalize(sp.clue_vecs.astype(np.float64) @ a.T)
    return Space(name, clue_vecs, sp.word_vecs, sp.words)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="potion")
    parser.add_argument("--template", default="bare", help="clue-side template")
    parser.add_argument("--boards", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default=None, help="write A to this .npy file")
    parser.add_argument(
        "--scheme", default="published", help="game: the split of codenames.split (see below)"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--score", default="test", help="with --scheme game: test, or val")
    args = parser.parse_args()
    if args.scheme != "published":
        by_game(args.model, args.template, args.scheme, args.seeds, args.epochs, args.score)
        return
    words = all_words()
    sp = space(args.model, args.template, "bare", words)
    held = association_items(eat_counts(), set(words))
    results = {"model": args.model, "seed": args.seed, "epochs": args.epochs, "tau": TAU}
    results["before"] = {"guess": guess_scores(sp, "test"), "assoc": association_top1(sp, held)}
    for name, eat, cultural in [
        ("cultural", False, True),
        ("eat", True, False),
        ("both", True, True),
    ]:
        data = items(sp, args.seed, eat=eat, cultural=cultural)
        t = time.perf_counter()
        a, hist = train(sp, data, args.epochs, seed=args.seed)
        tuned = adapted(sp, a, f"{args.model}+{name}")
        results[name] = {
            "items": len(data),
            "loss": hist,
            "seconds": round(time.perf_counter() - t, 1),
            "guess": guess_scores(tuned, "test"),
            "assoc": association_top1(tuned, held),
        }
        if name == "both":
            results["games"] = codenames_scores(sp, tuned, args.boards, args.seed)
            if args.save:
                np.save(args.save, a)
    results["config"] = vars(args)
    text = json.dumps(results, indent=1)
    print(text)
    out = RESULTS / f"train-{args.model.replace('/', '--')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text + "\n")


def by_game(model: str, template: str, scheme: str, seeds, epochs: int, score: str) -> dict:
    """The map trained on the Cultural Codes train turns of a scheme only (no EAT), once per
    seed, scored on that scheme's test turns (or val turns, for a dry run). This is the
    trained arm of experiments/plans/C1.md; results/train-<model>-<scheme>.json."""
    sp = space(model, template, "bare", all_words())
    out: dict = {"model": model, "template": template, "scheme": scheme, "scored": score}
    out["epochs"], out["tau"], out["runs"] = epochs, TAU, {}
    for seed in seeds:
        data = items(sp, seed, eat=False, split="train", scheme=scheme)
        t = time.perf_counter()
        a, hist = train(sp, data, epochs, seed=seed)
        tuned = adapted(sp, a, f"{model}+cultural")
        out["runs"][str(seed)] = {
            "items": len(data),
            "loss": hist,
            "seconds": round(time.perf_counter() - t, 1),
            "guess": guess_scores(tuned, score, None, scheme),
        }
        print(model, seed, round(out["runs"][str(seed)]["guess"]["top1"], 3), flush=True)
    name = f"train-{model.replace('/', '--')}-{scheme}"
    path = RESULTS / (f"{name}.json" if score == "test" else f"{name}-{score}.json")
    path.write_text(json.dumps(out, indent=1) + "\n")
    return out


def codenames_scores(before: Space, after: Space, n_boards: int, seed: int) -> dict:
    """Expected words found per first turn when the plain and the trained model give clues,
    as heard by the EAT guesser and by every calibrated embedding guesser. Clues are
    held-out EAT cues only, so the EAT guesser never answers a clue seen in training."""
    from codenames.game import deal
    from codenames.players import embedding_weigher, expected_found, expected_giver, norm_weigher
    from codenames.study import chosen_spaces, load, mean_se
    from codenames.words import clue_words

    cal = load("calibrate")
    counts = eat_counts()
    pool = tuple(c for c in clue_words() if c in counts and held_out(c))
    weighers = {m: embedding_weigher(sp, tau) for m, (sp, tau) in chosen_spaces(cal).items()}
    weighers["eat"] = norm_weigher(counts, cal["eat"]["eps"])
    rng = random.Random(seed)
    boards = [deal(board_words(), rng) for _ in range(n_boards)]
    out = {"pool": len(pool), "boards": n_boards}
    for name, sp in (("before", before), ("after", after)):
        tau = max(TAUS, key=lambda t: guess_loglik(sp, "train", t))
        give = expected_giver(sp, tau, candidates=pool)
        clues = [give(b) for b in boards]
        out[name] = {
            g: mean_se([expected_found(w, b, c) for b, c in zip(boards, clues, strict=True)])
            for g, w in weighers.items()
        }
    return out


if __name__ == "__main__":
    main()
