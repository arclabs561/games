"""The clues that C3's trained givers chose on its test boards, regenerated from c3.json.

C3 (experiments/plans/C3.md) committed each run's policy weights (theta) and each
policy's value per test board, but not the clues. A policy picks the action with the
largest theta . (phi - mu) / sd (`rl.score`, greedy), where mu and sd are the feature
scales of the run's 300 training boards (`rl.feature_scale`). They depend only on the
giver and the training seed, so this module deals the training boards again, recomputes
mu and sd, and takes the argmax on each of the 200 test boards (seed 3003).

The check: the regenerated pick's exact value under the run's held-out guesser must
equal the committed per-board value. Theta was committed rounded to 4 decimals, so a
near tie can turn the other way; the share that match is written beside the picks.

    uv run python -m codenames.diagnose_c3 --out codenames/results/diagnostics/c3-picks.txt

Needs the word vectors; about 5 minutes here.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from codenames import c3, rl
from codenames.game import deal
from codenames.words import board_words

HERE = Path(__file__).parent
C3 = HERE / "results" / "c3.json"
TRAIN_BOARDS, TEST_BOARDS, SHORTLIST = 300, 200, 30


def boards(seed: int, n: int):
    rng = random.Random(seed)
    return [deal(board_words(), rng) for _ in range(n)]


def pick(theta: np.ndarray, mu: np.ndarray, sd: np.ndarray, phi: np.ndarray) -> int:
    """The greedy action of rl.score."""
    return int(np.argmax(rl.policy(theta, (phi - mu) / sd)))


def regenerate(out: Path) -> dict:
    from codenames.norms import eat_counts
    from codenames.players import embedding_weigher
    from codenames.study import chosen_spaces, load
    from codenames.words import clue_words

    committed = json.loads(C3.read_text())
    tasks = [(t["giver"], t["held_out"]) for t in committed["tasks"]]
    seeds = committed["tasks"][0]["seeds"]
    thetas = {(r["giver"], r["held_out"], r["seed"]): r for r in committed["theta"]["runs"]}
    held_values = committed["per_board"]["held_out"]
    spaces = chosen_spaces(load("calibrate"))
    counts = eat_counts()
    pool = tuple(c for c in clue_words() if c in counts)
    test = boards(c3.TEST_SEED, TEST_BOARDS)
    lines, checked, matched = [], 0, 0
    for giver in c3.GIVERS:
        part = out.with_name(f"{out.stem}-{giver.split('/')[-1]}.part")
        if part.exists():  # a giver finished by an earlier, interrupted run
            got = json.loads(part.read_text())
            lines += got["lines"]
            checked += got["checked"]
            matched += got["matched"]
            continue
        start = (len(lines), checked, matched)
        sp, tau = spaces[giver]
        test_acts = [rl.actions(sp, tau, b, pool, SHORTLIST) for b in test]
        for s in seeds:
            train = [rl.actions(sp, tau, b, pool, SHORTLIST)[1] for b in boards(s, TRAIN_BOARDS)]
            mu, sd = rl.feature_scale([(phi, None) for phi in train])
            for t, (g, held) in enumerate(tasks):
                if g != giver:
                    continue
                weigh = embedding_weigher(*spaces[held])
                for policy in c3.POLICIES:
                    theta = np.array(thetas[(g, held, s)][policy])
                    cells = []
                    for i, (b, (clues, phi)) in enumerate(zip(test, test_acts, strict=True)):
                        c, n = clues[pick(theta, mu, sd, phi)]
                        cells.append(f"{c}/{n}")
                        v = float(rl.rewards(weigh, b, [(c, n)])[0])
                        want = held_values[policy][t][seeds.index(s)][i]
                        checked += 1
                        matched += abs(v - want) < 1e-4
                    name = f"{g.split('/')[-1]}|{held.split('/')[-1]}|{s}|{policy}"
                    lines.append(" ".join([name, str(c3.TEST_SEED), *cells]))
        mine = {
            "lines": lines[start[0] :],
            "checked": checked - start[1],
            "matched": matched - start[2],
        }
        part.write_text(json.dumps(mine))
        print(giver, "done", flush=True)
    text = "\n".join(lines) + "\n"
    out.write_text(text)
    check = {
        "picks": checked,
        "held_out_value_matches_c3": matched,
        "share": round(matched / checked, 6),
        "tolerance": 1e-4,
    }
    out.with_suffix(".json").write_text(json.dumps(check, indent=1, sort_keys=True) + "\n")
    for giver in c3.GIVERS:
        out.with_name(f"{out.stem}-{giver.split('/')[-1]}.part").unlink()
    print(check)
    return check


def read_picks(text: str) -> dict[tuple[str, str, int, str], list[tuple[str, int]]]:
    """{(giver, held out, seed, policy): [(clue, count) per test board]}."""
    out = {}
    for line in text.splitlines():
        name, _seed, *cells = line.split()
        g, h, s, p = name.split("|")
        out[(g, h, int(s), p)] = [(c, int(n)) for c, n in (x.split("/") for x in cells)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    regenerate(parser.parse_args().out)


if __name__ == "__main__":
    main()
