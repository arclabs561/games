"""Experiment C8 (experiments/plans/C8.md): does training on the teacher's turns raise the
LoRA encoder's match with people?

The jobs are `codenames.distill label` (the teacher's turns) and `codenames.lora` with
the teacher sources (arms B, C and D, three seeds each). Their summary.json files are
copied to results/c8-runs/<run>.json and label.json to results/c8-label.json. Arm A is
C2's arm 4 (results/c2-runs/lora-s1 to s3), which is not trained again. This module
pairs the arms by test turn on the machine that writes the README:

    python -m codenames.c8 --out codenames/results/c8.json [--giver]

--giver also scores arm B against arm A as a giver (it needs the word vectors of the seven
other embedding models, as codenames.c2 does).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from codenames.c2 import TEST_RUNS, as_hits, turn_digest

RESULTS = Path(__file__).parent / "results"
ARMS = {
    "B": ("b-s1", "b-s2", "b-s3"),
    "C": ("c-s1", "c-s2", "c-s3"),
    "D": ("d-s1", "d-s2", "d-s3"),
}
SOURCES = {"B": "cultural+teacher", "C": "teacher", "D": "relabeled"}
NAMES = {
    "A": "A. human only (C2's arm 4)",
    "B": "B. human plus teacher",
    "C": "C. teacher only",
    "D": "D. relabeled",
}
LR_RUNS = {"0.0001": "lr1e-4-s1", "0.0003": "lr3e-4-s1"}
TEACHER = "gpt-oss-120b"


def teacher_test_hits(lines) -> np.ndarray:
    """gpt-oss-120b's per-turn hits on C1's test turns, from C4's per-turn lines (1 or b
    is a hit; see c4.guess_lines)."""
    for line in lines:
        name, split, chars = line.split()
        if name == TEACHER and split == "test":
            return np.array([int(c in "1b") for c in chars])
    raise KeyError(f"no {TEACHER} test line")


def choose_lr(val_loglik: dict[str, float], c2_val: float) -> tuple[float, bool]:
    """The learning rate with the best validation log-likelihood (the smaller on a tie),
    and whether it falls below C2's arm 4 at seed 1, the plan's rule for a failed arm."""
    lr = max(sorted(val_loglik, key=float), key=lambda k: val_loglik[k])
    return float(lr), val_loglik[lr] < c2_val


def read_runs(folder: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))}


def load_results() -> dict | None:
    path = RESULTS / "c8.json"
    return json.loads(path.read_text()) if path.exists() else None


def lr_record(runs: dict[str, dict]) -> dict:
    """The learning-rate choice for arms B, C and D, from the two validation jobs."""
    c2_lr = json.loads((RESULTS / "c2-lr.json").read_text())
    c2_val = c2_lr["val_loglik"]["0.0003"]
    val = {lr: runs[r]["after"]["loglik"] for lr, r in LR_RUNS.items()}
    lr, failed = choose_lr(val, c2_val)
    for r in LR_RUNS.values():
        assert runs[r]["after"]["split"] == "val" and runs[r]["config"]["seed"] == 1
        assert runs[r]["config"]["sources"] == "cultural+teacher"
    return {
        "criterion": "mean log-likelihood of the person's word on C1's val turns, arm B, "
        "LoRA seed 1, chosen template",
        "val_loglik": val,
        "val_top1": {lr: runs[r]["after"]["top1"] for lr, r in LR_RUNS.items()},
        "c2_arm4_seed1_val_loglik": c2_val,
        "choice": lr,
        "failed": failed,
        "runs": LR_RUNS,
    }


def teacher_agreement(label: dict) -> dict:
    """The teacher's agreement with people on the training turns it labeled, with an
    interval from resampling training games."""
    from codenames.split import turns
    from codenames.stats import cluster_draws, clustered

    games = [t.game for t in turns("game:train")]
    keep = [i for i, c in enumerate(label["train_hits"]) if c in "01"]
    hits = [int(label["train_hits"][i]) for i in keep]
    index, counts = cluster_draws([games[i] for i in keep])
    return clustered(hits, index, counts) | {"unlabeled": len(games) - len(keep)}


def analyze(runs: dict[str, dict], c2_runs: dict[str, dict], label: dict, c4: dict) -> dict:
    from codenames.split import turns
    from codenames.stats import cluster_draws, clustered

    test = turns("game:test")
    index, counts = cluster_draws([t.game for t in test])
    digest = turn_digest([(t.guess.clue, t.guess.picked) for t in test])
    arm_runs = {"A": {r: c2_runs[r] for r in TEST_RUNS["lora"]}}
    for arm, names in ARMS.items():
        if all(r in runs for r in names):
            arm_runs[arm] = {r: runs[r] for r in names}
    hits, arms = {}, {}
    for arm, rs in arm_runs.items():
        for name, s in rs.items():
            assert s["after"]["turns_sha256"] == digest, name
            assert s["after"]["split"] == "test" and s["config"]["scheme"] == "game"
            want = "cultural" if arm == "A" else SOURCES[arm]
            assert s["config"]["sources"] == want, name
        each = [as_hits(s["after"]["hits"]) for s in rs.values()]
        hits[arm] = np.mean(each, axis=0)
        arms[arm] = {
            "name": NAMES[arm],
            "runs": list(rs),
            "top1": clustered(hits[arm], index, counts),
            "top1_by_seed": [float(h.mean()) for h in each],
            "loglik_by_seed": [s["after"]["loglik"] for s in rs.values()],
            "assoc_top1_by_seed": [s["after"]["assoc"]["top1"] for s in rs.values()],
            "chosen": [s["after"]["chosen"] for s in rs.values()],
            "rows": sorted({s["config"]["rows"] for s in rs.values()}),
            "lr": sorted({s["config"]["lr"] for s in rs.values()}),
            "teacher_sha256": sorted(
                {s["config"].get("teacher_sha256") or "" for s in rs.values()}
            ),
        }
    hits[TEACHER] = teacher_test_hits(c4["guess"]["per_turn"])
    assert len(hits[TEACHER]) == len(test)
    arms[TEACHER] = {"name": f"{TEACHER} (C4)", "top1": clustered(hits[TEACHER], index, counts)}

    def diff(a: str, b: str) -> dict:
        return clustered(hits[a] - hits[b], index, counts)

    out: dict = {
        "plan": "experiments/plans/C8.md",
        "turns": len(test),
        "games": len({t.game for t in test}),
        "arms": arms,
        "teacher": {
            "agreement_with_people_on_train": teacher_agreement(label),
            "turns": label["turns"],
            "repeats": label["repeats"],
            "teacher_sha256": label["teacher_sha256"],
        },
    }
    if "B" in hits:
        out["primary"] = {
            "metric": "top-1 of arm B (human plus teacher, mean of 3 seeds) minus arm A "
            "(C2's arm 4, mean of 3 seeds), paired by turn",
            **diff("B", "A"),
        }
        out["b_minus_teacher"] = diff("B", TEACHER)
    for arm in ("C", "D"):
        if arm in hits:
            out[f"{arm.lower()}_minus_a"] = diff(arm, "A")
    out["a_minus_teacher"] = diff("A", TEACHER)
    return out


def giver(runs: dict[str, dict], c2_runs: dict[str, dict]) -> dict:
    """Arm B against arm A as givers on C2's 200 boards, paired by board."""
    from codenames.c2 import BOARD_SEED, N_BOARDS, giver_scores
    from codenames.stats import cluster_draws, clustered

    clue_sets = {
        "A": [c2_runs[r]["after"]["giver"]["clues"] for r in TEST_RUNS["lora"]],
        "B": [runs[r]["after"]["giver"]["clues"] for r in ARMS["B"]],
    }
    found = giver_scores(clue_sets)
    out = {}
    for g in found["B"]:
        d = found["B"][g] - found["A"][g]
        index, counts = cluster_draws([str(i) for i in range(len(d))])
        out[g] = {
            "A": float(found["A"][g].mean()),
            "B": float(found["B"][g].mean()),
            "b_minus_a": clustered(d, index, counts),
        }
    return {"boards": N_BOARDS, "seed": BOARD_SEED, "guessers": out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=RESULTS / "c8-runs", help="<run>.json")
    parser.add_argument("--label", type=Path, default=RESULTS / "c8-label.json")
    parser.add_argument("--out", type=Path, default=RESULTS / "c8.json")
    parser.add_argument("--giver", action="store_true", help="also the giver comparison")
    args = parser.parse_args()
    runs = read_runs(args.runs)
    c2_runs = read_runs(RESULTS / "c2-runs")
    label = json.loads(args.label.read_text())
    c4 = json.loads((RESULTS / "c4.json").read_text())
    lr = lr_record(runs)
    out = {"learning_rate": lr} | analyze(runs, c2_runs, label, c4)
    out["runs"] = {r: s["config"] for r, s in runs.items()}
    if args.giver and all(r in runs for r in ARMS["B"]):
        out["giver"] = giver(runs, c2_runs)
    elif args.out.exists() and "giver" in json.loads(args.out.read_text()):
        out["giver"] = json.loads(args.out.read_text())["giver"]  # keep the earlier scoring
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    if "primary" in out:
        p = out["primary"]
        print(f"B minus A: {p['mean']:+.3f} [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
