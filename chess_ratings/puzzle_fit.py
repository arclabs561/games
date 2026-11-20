"""Plan R1's fitting and scoring, here, from the curves jobs' outputs.

Reads every `curves-<i>.npz` and `rows-<i>.jsonl.gz` under RUNS, fits Z+ (isotonic on R_half),
B and B+M (LightGBM, num_leaves 31 or 127 chosen on validation) on the training puzzles,
and scores the test and few-plays sets once. Writes `results/puzzles.json`.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from chess_ratings import puzzles as pz

BANDS = (400, 1000, 1400, 1800, 2200, 2600, 4000)


def load(runs: Path) -> tuple[dict[str, list[dict]], dict[str, np.ndarray]]:
    rows, logp = {}, {}
    for npz in sorted(runs.rglob("curves-*.npz")):
        i = npz.stem.split("-")[1]
        z = np.load(npz)
        starts = np.concatenate([[0], np.cumsum(z["lengths"])])
        for k, pid in enumerate(z["ids"]):
            logp[str(pid)] = z["logp"][starts[k] : starts[k + 1]].astype(float)
        with gzip.open(npz.with_name(f"rows-{i}.jsonl.gz"), "rt") as f:
            for line in f:
                r = json.loads(line)
                rows.setdefault(r["set"], []).append(r)
    for k in rows:
        rows[k].sort(key=lambda r: pz.order_key(r["PuzzleId"]))
    return rows, logp


def run(runs: Path, out: Path) -> dict:
    rows, logp = load(runs)
    y = {k: np.array([r["Rating"] for r in v], dtype=float) for k, v in rows.items()}
    lp = {k: [logp[r["PuzzleId"]] for r in v] for k, v in rows.items()}
    names_b, xb = {}, {}
    names_m, xm = {}, {}
    for k in rows:
        names_b[k], xb[k] = pz.feature_rows(rows[k], None)
        names_m[k], xm[k] = pz.feature_rows(rows[k], lp[k])
    rh = {k: np.array([pz.r_half(pz.log_solve(x)) for x in lp[k]]) for k in rows}
    iso = pz.isotonic(rh["train"], y["train"])
    b, b_info = pz.fit_lgbm(xb["train"], y["train"], xb["val"], y["val"])
    bm, bm_info = pz.fit_lgbm(xm["train"], y["train"], xm["val"], y["val"])
    mean = float(y["train"].mean())
    result: dict = {
        "plan": pz.PLAN,
        "sizes": {k: len(v) for k, v in rows.items()},
        "hyperparameters": {"B": b_info, "B+M": bm_info},
        "features": {"B": names_b["train"], "B+M": names_m["train"]},
        "train_mean_rating": mean,
    }
    for k in ("test", "few"):
        if k not in rows:
            continue
        preds = {
            "mean": np.full(len(y[k]), mean),
            "Z": rh[k],
            "Z+": iso(rh[k]),
            "B": b.predict(xb[k]),  # ty: ignore[missing-attribute]
            "B+M": bm.predict(xm[k]),  # ty: ignore[missing-attribute]
        }
        part: dict = {m: pz.metrics(p, y[k]) for m, p in preds.items()}
        part["h_m"] = pz.paired_mse(preds["B+M"], preds["B"], y[k])
        bias = {}
        for lo, hi in zip(BANDS, BANDS[1:], strict=False):
            sel = (y[k] >= lo) & (y[k] < hi)
            if sel.sum():
                bias[f"{lo}-{hi}"] = {
                    "n": int(sel.sum()),
                    "Z_bias": float(np.mean(rh[k][sel] - y[k][sel])),
                }
        part["z_bias_by_band"] = bias
        ends = (rh[k] == pz.GRID[0]) | (rh[k] == pz.GRID[-1])
        part["curve_never_crosses_share"] = float(ends.mean())
        nmoves = np.array([len(x) for x in lp[k]])
        part["by_solver_moves"] = {
            str(m): {
                "n": int(((nmoves >= 4) if m == 4 else (nmoves == m)).sum()),
                **{
                    a: float(
                        np.mean((preds[a] - y[k])[(nmoves >= 4) if m == 4 else (nmoves == m)] ** 2)
                    )
                    for a in ("B", "B+M", "Z")
                },
            }
            for m in (1, 2, 3, 4)
            if ((nmoves >= 4) if m == 4 else (nmoves == m)).any()
        }
        result[k] = part
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: result[k]["h_m"] for k in ("test", "few") if k in result}, indent=1))
    return result
