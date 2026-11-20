"""Learning a strategy card one rule at a time, with every accepted card scored exactly.

The method is greedy decision-list learning weighted by the card's own regret:

1. Solve the card exactly (`solver.solve(rule=card)`). This gives its expected score and the
   value V of every state when the card is followed from there.
2. Follow the card from the empty score card, carrying the exact chance of reaching each state
   and each roll, and draw a seeded sample of its decisions in proportion to those chances.
   At each sampled decision, record the facts the tests read and the **gain** of each other
   choice: its value minus the value of the card's choice, both with the card followed
   afterwards (the Q-values of policy iteration).
3. A new rule placed at position p of a list changes the decisions where its tests hold and no
   earlier rule applies. The sum of the gains there, divided by the sample rate, estimates the
   change in expected score (to first order: later decisions still follow the old card).
4. Take the best candidate, solve the new card exactly, and keep it only if the exact score
   rose. Otherwise try the next candidate.

Samples only guide the search. Every score reported is the exact value from step 1.

    uv run python -m yahtzee.learn 40       # learn 40 steps and print each card; about 2 hours
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from yahtzee.human import (
    BASE,
    KEEP_WORDS,
    TESTS,
    Box,
    Card,
    Keep,
    Strategy,
    card_values,
    expected_score,
    moves,
)
from yahtzee.rules import CATEGORIES, OFFICIAL
from yahtzee.solver import Tables, final_rolls

HERE = Path(__file__).parent
KEEPS = [k for k in KEEP_WORDS]
KEEP_TESTS = list(TESTS)
BOX_TESTS = [k for k in TESTS if "roll" not in k]
BOX_ACTIONS = [(c, True) for c in CATEGORIES] + [(c, False) for c in CATEGORIES] + [("face", True)]


def exact(card: Strategy, t: Tables) -> float:
    """The card's exact expected score (forward from the empty score card)."""
    return expected_score(card, t)


@dataclass
class Sample:
    """Sampled decisions of one kind: test facts, how often drawn, rule fired, and gains."""

    tests: np.ndarray  # (S, tests) bool
    weight: np.ndarray  # (S,) times drawn
    fired: np.ndarray  # (S,) number of the card's rule that made the decision
    gain: np.ndarray  # (S, actions) value of each action minus the card's; nan if it cannot apply
    extra: dict


def collect(t: Tables, values: np.ndarray, card: Strategy, rate: int, seed: int, batch=1024):
    """Sample the card's decisions, `rate` draws per turn and roll on average."""
    rng = np.random.default_rng(seed)
    keep_parts, box_parts = [], []
    states, occ = np.array([t.index(0, 0, 0)]), np.array([1.0])
    nk = len(t.keeps)
    for _ in range(len(CATEGORIES)):
        nxt_all, mass_all = [], []
        for lo in range(0, len(states), batch):
            s, o = states[lo : lo + batch], occ[lo : lo + batch]
            nb = len(s)
            m = moves(card, t, s)
            c = Card(t, s)
            d = card.dice(t)
            rows = np.arange(nb)[:, None]
            last = m.gain + values[m.next]  # the value of each final roll
            kv1 = last @ t.P.T
            kv0 = kv1[rows, m.keeps[1]] @ t.P.T
            q = np.broadcast_to(t.first, (nb, len(t.rolls)))
            for level, left, kv in ((0, 2, kv0), (1, 1, kv1)):
                chosen = m.keeps[level]
                draw = rng.poisson(rate * o[:, None] * q)
                b, r = np.nonzero(draw)
                if len(b):
                    _, fired = card.keep_choice(t, s, left)
                    base = kv[rows, chosen]
                    gains = []
                    for k in KEEPS:
                        keep = np.broadcast_to(card.keep_of(k, d, c), (nb, len(t.rolls)))
                        gains.append((kv[rows, keep] - base)[b, r])
                    tests = [
                        np.broadcast_to(card.test(n, d, c, left), (nb, len(t.rolls)))[b, r]
                        for n in KEEP_TESTS
                    ]
                    keep_parts.append(
                        (np.stack(tests, 1), draw[b, r], fired[b, r], np.stack(gains, 1), {})
                    )
                z = np.bincount((rows * nk + chosen).ravel(), q.ravel(), nb * nk)
                q = z.reshape(nb, nk) @ t.P
            draw = rng.poisson(rate * o[:, None] * q)
            b, r = np.nonzero(draw)
            if len(b):
                _, _, each = final_rolls(t, values, s, every=True)
                base = last
                choice, fired = card.box_choice(t, s)
                assert np.allclose(np.take_along_axis(each, choice[:, :, None], 2)[:, :, 0], base)
                points = card.points(t, d, c)
                gains = []
                for box, need in BOX_ACTIONS:
                    i = card.box_of(box, d, nb)
                    g = np.take_along_axis(each, i[:, :, None], 2)[:, :, 0] - base
                    ok = np.take_along_axis(c.open, i, 1)
                    if need:
                        ok = ok & (np.take_along_axis(points, i[:, :, None], 2)[:, :, 0] > 0)
                    gains.append(np.where(ok, g, np.nan)[b, r])
                tests = [
                    np.broadcast_to(card.test(n, d, c), (nb, len(t.rolls)))[b, r] for n in BOX_TESTS
                ]
                extra = {
                    "q": (each - base[:, :, None])[b, r],
                    "points": points[b, r],
                    "open": np.broadcast_to(c.open[:, None, :], points.shape)[b, r],
                    "secured": np.broadcast_to(c.secured, base.shape)[b, r],
                }
                box_parts.append(
                    (np.stack(tests, 1), draw[b, r], fired[b, r], np.stack(gains, 1), extra)
                )
            nxt_all.append(m.next.ravel())
            mass_all.append((o[:, None] * q).ravel())
        uniq, inv = np.unique(np.concatenate(nxt_all), return_inverse=True)
        states, occ = uniq, np.bincount(inv, np.concatenate(mass_all), len(uniq))
    return join(keep_parts), join(box_parts)


def join(parts) -> Sample:
    extra = {k: np.concatenate([p[4][k] for p in parts]) for k in parts[0][4]}
    return Sample(*(np.concatenate([p[i] for p in parts]) for i in range(4)), extra)


@dataclass(frozen=True)
class Candidate:
    kind: str  # "keep", "box" or "par"
    position: int
    rule: Any
    gain: float  # estimated change in expected score

    def apply(self, card: Strategy) -> Strategy:
        if self.kind == "keep":
            rules = list(card.keep_rules)
            rules.insert(self.position, self.rule)
            return replace(card, keep_rules=tuple(rules))
        if self.kind == "box":
            rules = list(card.box_rules)
            rules.insert(self.position, self.rule)
            return replace(card, box_rules=tuple(rules))
        par, weight = self.rule
        return replace(card, par=par, weight=weight)


def rule_candidates(
    data: Sample, names: list[str], actions: list, n_rules: int, rate: int, top: int = 40
):
    """The best (position, tests, action) by estimated gain, with up to three tests."""
    gain = np.nan_to_num(data.gain) * data.weight[:, None] / rate
    facts = data.tests
    # Group identical decisions (same tests and same rule fired) to shrink the sums.
    key = np.concatenate([np.packbits(facts, axis=1), data.fired[:, None].astype(np.uint8)], 1)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    grouped = np.zeros((inv.max() + 1, gain.shape[1]))
    np.add.at(grouped, inv, gain)
    first = np.zeros(inv.max() + 1, dtype=np.int64)
    first[inv] = np.arange(len(inv))
    tg = facts[first].astype(np.float64)
    fired = data.fired[first]
    results = []
    for p in range(n_rules + 1):
        live = fired >= p
        tl, gl = tg[live], grouped[live]
        for a in range(grouped.shape[1]):
            pair = tl.T @ (tl * gl[:, a : a + 1])  # pair[i, j]: gain where tests i and j both hold
            iu = np.triu_indices(len(names))
            for i, j, h in zip(*iu, pair[iu], strict=True):
                if h > 0.02:
                    results.append((h, p, (int(i),) if i == j else (int(i), int(j)), a))
    results.sort(key=lambda x: -x[0])
    best = results[: top * 3]
    # Try a third test on the best pairs.
    out = []
    for h, p, tests, a in best:
        out.append((h, p, tests, a))
        if len(tests) == 2:
            live = fired >= p
            m = live & tg[:, tests[0]].astype(bool) & tg[:, tests[1]].astype(bool)
            g3 = tg[m].T @ grouped[m, a]
            k = int(g3.argmax())
            if g3[k] > h + 0.02 and k not in tests:
                out.append((float(g3[k]), p, (*tests, k), a))
    out.sort(key=lambda x: -(x[0] - 0.03 * (len(x[2]) - 1)))
    return out[:top]


def par_candidates(card: Strategy, box: Sample, rate: int):
    """Changes of one par value (or of the upper weight) with their estimated gains."""
    default = box.fired == len(card.box_rules)
    q = box.extra["q"][default]
    w = box.weight[default] / rate
    points = box.extra["points"][default]
    open_ = box.extra["open"][default]
    secured = box.extra["secured"][default]

    def value(par: dict, weight: float) -> float:
        p = np.array([par[c] for c in CATEGORIES], float)
        m = points - p
        m[:, :6] *= np.where(secured, 1.0, weight)[:, None]
        m = np.where(open_, m, -np.inf)
        choice = m.argmax(axis=1)
        return float(w @ q[np.arange(len(q)), choice])

    par = dict(card.par)
    now = value(par, card.weight)
    out = []
    for c in CATEGORIES:
        for step in (-3, -2, -1, 1, 2, 3):
            new = par | {c: par[c] + step}
            out.append((value(new, card.weight) - now, (tuple(new.items()), card.weight)))
    for step in (-0.5, -0.25, 0.25, 0.5):
        if card.weight + step > 0:
            out.append((value(par, card.weight + step) - now, (card.par, card.weight + step)))
    out.sort(key=lambda x: -x[0])
    return [Candidate("par", 0, r, g) for g, r in out[:5] if g > 0.005]


def candidates(card: Strategy, keep: Sample, box: Sample, rate: int) -> list[Candidate]:
    out = par_candidates(card, box, rate)
    for g, p, tests, a in rule_candidates(keep, KEEP_TESTS, KEEPS, len(card.keep_rules), rate):
        rule = Keep(tuple(KEEP_TESTS[i] for i in tests), KEEPS[a])
        out.append(Candidate("keep", p, rule, g - 0.03 * (len(tests) - 1)))
    for g, p, tests, a in rule_candidates(box, BOX_TESTS, BOX_ACTIONS, len(card.box_rules), rate):
        name, need = BOX_ACTIONS[a]
        names = tuple(BOX_TESTS[i] for i in tests) + (("it scores",) if need else ())
        out.append(Candidate("box", p, Box(names, name), g - 0.03 * (len(tests) - 1)))
    out.sort(key=lambda c: -c.gain)
    return out


def to_json(card: Strategy) -> dict:
    return {
        "keep": [[list(r.tests), r.keep] for r in card.keep_rules],
        "box": [[list(r.tests), r.box] for r in card.box_rules],
        "par": dict(card.par),
        "weight": card.weight,
        "default": card.default,
    }


def from_json(d: dict) -> Strategy:
    return Strategy(
        keep_rules=tuple(Keep(tuple(t), k) for t, k in d["keep"]),
        box_rules=tuple(Box(tuple(t), b) for t, b in d["box"]),
        par=tuple((k, v) for k, v in d["par"].items()),
        weight=d["weight"],
        default=d["default"],
    )


def learn(
    t: Tables, steps: int, rate: int, seed: int, start: Strategy, tries: int = 6, minutes=1e9
):
    """Grow a card by up to `steps` accepted changes; yield (card, exact score) after each.

    The search stops early when no candidate tried raises the exact score, or after `minutes`.
    """
    card = start
    score = exact(card, t)
    yield card, score
    deadline = time.time() + 60 * minutes
    for step in range(steps):
        if time.time() > deadline:
            return
        t0 = time.time()
        _, values = card_values(card, t)
        keep, box = collect(t, values, card, rate, seed * 1000 + step)
        for cand in candidates(card, keep, box, rate)[:tries]:
            new = cand.apply(card)
            new_score = exact(new, t)
            rule = cand.rule if cand.kind != "par" else ""
            print(
                f"  try {cand.kind} {cand.position} {rule}"
                f" est {cand.gain:+.3f} exact {new_score - score:+.4f}",
                flush=True,
            )
            if new_score > score + 1e-9:
                card, score = new, new_score
                break
        else:
            return
        print(f"step {step + 1}: {score:.4f} ({time.time() - t0:.0f} s)", flush=True)
        yield card, score


def without(card: Strategy, i: int) -> Strategy:
    """The card with its i-th rule (keep rules first, then box rules) removed."""
    k = len(card.keep_rules)
    if i < k:
        return replace(card, keep_rules=card.keep_rules[:i] + card.keep_rules[i + 1 :])
    i -= k
    return replace(card, box_rules=card.box_rules[:i] + card.box_rules[i + 1 :])


def prune(t: Tables, card: Strategy):
    """Remove rules one at a time, each time the one whose loss is least; yield each card.

    This traces the shorter cards that a long card contains, scored exactly.
    """
    score = exact(card, t)
    yield card, score
    while card.size():
        best = max(
            ((exact(without(card, i), t), i) for i in range(card.size())), key=lambda x: x[0]
        )
        card, score = without(card, best[1]), best[0]
        print(f"{card.size()} rules: {score:.4f}", flush=True)
        yield card, score


def record(f, card: Strategy, score: float) -> None:
    f.write(json.dumps({"score": score, "size": card.size(), "card": to_json(card)}) + "\n")
    f.flush()


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("mode", choices=["learn", "prune"])
    p.add_argument("--start", default="empty", help="empty, base, or a .jsonl file (last line)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--rate", type=int, default=4000)
    p.add_argument("--tries", type=int, default=6)
    p.add_argument("--minutes", type=float, default=1e9, help="stop learning after this long")
    p.add_argument("--out", type=Path, default=HERE / "results")
    a = p.parse_args(argv)
    t = Tables(OFFICIAL)
    start = start_card(a.start)
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"{a.mode}-{Path(a.start).stem}-{a.seed}.jsonl"
    with open(path, "w") as f:
        steps = (
            learn(t, a.steps, a.rate, a.seed, start, a.tries, a.minutes)
            if a.mode == "learn"
            else prune(t, start)
        )
        for card, score in steps:
            record(f, card, score)
            print(f"{card.size()} rules, exact {score:.4f}", flush=True)
            print("\n".join(card.lines()), flush=True)


def start_card(name: str) -> Strategy:
    if name == "empty":
        return Strategy()
    if name == "base":
        return BASE
    lines = Path(name).read_text().splitlines()
    return from_json(json.loads(max(lines, key=lambda x: json.loads(x)["score"]))["card"])


if __name__ == "__main__":
    main(sys.argv[1:])
