"""A cautious rule with no training, as a baseline for plan C3 (exploratory).

C3 compares a giver trained against a population with one trained against itself, and
the population policy learns to ask for fewer words. This module asks how much of its
gain a rule with no parameters gets: on each board, among C3's 30 shortlisted clues,
give the clue whose likeliest board word is most likely a team word under the giver's
own Luce guesser, always with count 1. The rule was fixed before it was scored on the
14 tasks; it is the rule that a review ran on two of them with seed 1. It was not in
plan C3, so everything here is exploratory.

The rule needs no training seed, so its value on a board is the same for every seed of
a task. The comparison uses C3's own per-board values (results/c3.json) and C3's
stratified bootstrap: the same 200 boards drawn for every task, and the five seeds of
each task drawn within it.

    uv run python -m codenames.cautious run       # the rule on C3's test boards (about 5 min)
    uv run python -m codenames.cautious analyze   # the summary, from the committed values

`run` needs the vectors of all eight models; `analyze` needs only the committed files.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from codenames import c3, rl
from codenames.game import deal
from codenames.norms import eat_counts
from codenames.players import embedding_weigher, norm_weigher
from codenames.study import RESULTS, chosen_spaces, load, models
from codenames.words import board_words

OUT = RESULTS / "c3-cautious.json"
SHORTLIST = 30
TEST_BOARDS = 200


def rule(phi: np.ndarray) -> int:
    """Index of the count-1 action with the highest chance of a team word."""
    count, found = rl.FEATURES.index("count"), rl.FEATURES.index("found")
    return int(np.argmax(np.where(phi[:, count] == 1, phi[:, found], -np.inf)))


def rule_values() -> dict:
    """The rule's exact value on each of C3's test boards, for each giver and guesser."""
    assert tuple(models()) == c3.MODELS, f"vectors missing: models() is {models()}"
    cal = load("calibrate")
    guessers = {m: embedding_weigher(sp, tau) for m, (sp, tau) in chosen_spaces(cal).items()}
    guessers["eat"] = norm_weigher(eat_counts(), cal["eat"]["eps"])
    rng = random.Random(c3.TEST_SEED)
    boards = [deal(board_words(), rng) for _ in range(TEST_BOARDS)]
    out = {}
    for giver in c3.GIVERS:
        data, _ = rl.build(giver, guessers, boards, SHORTLIST)
        picks = [rule(phi) for phi, _ in data]
        out[giver] = {
            g: [round(float(rew[g][i]), 4) for (_, rew), i in zip(data, picks, strict=True)]
            for g in guessers
        }
    return out


def per_task(c3json: dict, values: dict, policy: str, guesser: str) -> list[np.ndarray]:
    """(seeds, boards) per task: C3's policy minus the rule, or the rule minus C3's
    policy when policy starts with "-"."""
    sign = -1 if policy.startswith("-") else 1
    name = policy.lstrip("-")
    kind = "eat" if guesser == "eat" else "held_out"
    out = []
    for t, task in enumerate(c3json["tasks"]):
        g = "eat" if guesser == "eat" else task["held_out"]
        trained = np.array(c3json["per_board"][kind][name][t])
        base = np.array(values[task["giver"]][g])
        out.append(sign * (trained - base[None, :]))
    return out


def analyze(values: dict, c3json: dict) -> dict:
    draws = c3.Draws(TEST_BOARDS, [len(t["seeds"]) for t in c3json["tasks"]])
    result: dict = {
        "what": (
            "Exploratory. The count-1 rule of codenames.cautious against C3's policies, "
            "on C3's 200 test boards (seed 3003) and 14 tasks, with C3's stratified "
            "bootstrap (10,000 draws, seed 0). Values are words per turn."
        ),
        "rule": "count 1; the shortlisted clue whose likeliest word is most likely a team word",
        "test_seed": c3.TEST_SEED,
        "boards": TEST_BOARDS,
        "tasks": [{"giver": t["giver"], "held_out": t["held_out"]} for t in c3json["tasks"]],
    }
    for guesser, label in (("held", "held_out"), ("eat", "eat")):
        pop_minus_rule = per_task(c3json, values, "population", guesser)
        rule_minus_self = per_task(c3json, values, "-self", guesser)
        pop_minus_self = [a + b for a, b in zip(pop_minus_rule, rule_minus_self, strict=True)]
        reps = {
            k: np.stack([c3.task_reps(d, draws, t) for t, d in enumerate(v)]).mean(axis=0)
            for k, v in (("num", rule_minus_self), ("den", pop_minus_self))
        }
        share = float(np.mean([d.mean() for d in rule_minus_self])) / float(
            np.mean([d.mean() for d in pop_minus_self])
        )
        rule_mean = float(
            np.mean(
                [
                    np.mean(values[t["giver"]]["eat" if guesser == "eat" else t["held_out"]])
                    for t in c3json["tasks"]
                ]
            )
        )
        result[label] = {
            "rule_value": round(rule_mean, 4),
            "population_minus_rule": c3.summarize(pop_minus_rule, draws),
            "rule_minus_self": c3.summarize(rule_minus_self, draws),
            "share_of_gain": c3.interval(share, reps["num"] / reps["den"]),
        }
    result["values"] = values
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("run", "analyze"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    c3json = load("c3")
    if args.step == "run":
        values = rule_values()
    else:
        values = json.loads(args.out.read_text())["values"]
    result = analyze(values, c3json)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    for label in ("held_out", "eat"):
        r = result[label]
        print(
            label,
            "rule",
            r["rule_value"],
            "population - rule",
            r["population_minus_rule"]["mean_over_tasks"],
            "rule - self",
            r["rule_minus_self"]["mean_over_tasks"],
            "share",
            r["share_of_gain"],
        )


if __name__ == "__main__":
    main()
