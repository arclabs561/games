"""The calibration study's two steps: the pilot (diagnosis, cross-validation, frozen fit) and the
test (scored once). The arithmetic is in `calibrate.py`; this module orders it and writes
aggregates. The plan is `results/calibration-plan.md`.
"""

import numpy as np

from chess_ratings import calibrate as cal
from chess_ratings.calibrate import ARMS, LOWER, Arm, Record


def arm_to_json(arm: Arm) -> dict:
    def knots(k):
        return None if k is None else [[float(v) for v in k[0]], [float(v) for v in k[1]]]

    return {
        "name": arm.name,
        "old_grid": arm.old_grid,
        "tau": arm.tau,
        "game_sigma": arm.game_sigma,
        "move_sigma": arm.move_sigma,
        "rating_map": knots(arm.rating_map),
        "correction": knots(arm.correction),
        "prior": None
        if arm.prior is None
        else {k: [float(v) for v in a] for k, a in arm.prior.items()},
    }


def arm_from_json(d: dict) -> Arm:
    def knots(k):
        return None if k is None else (np.array(k[0]), np.array(k[1]))

    return Arm(
        d["name"],
        old_grid=d["old_grid"],
        tau=d["tau"],
        game_sigma=d["game_sigma"],
        move_sigma=d["move_sigma"],
        rating_map=knots(d["rating_map"]),
        correction=knots(d["correction"]),
        prior=None if d["prior"] is None else {k: np.array(v) for k, v in d["prior"].items()},
    )


def evaluate(arm: Arm, records: list[Record]) -> list[tuple[Record, dict]]:
    return [(r, arm.evaluate(r)) for r in records]


def cases(records: list[Record], arm: Arm, seed: int = 40) -> str:
    """Case files: the ten largest errors in the two lower bands, and one random player-game per
    band."""
    rows = evaluate(arm, records)
    low = sorted(
        [(r, e) for r, e in rows if r.band in LOWER],
        key=lambda x: -abs(x[1]["estimate"] - x[0].rating),
    )[:10]
    rng = np.random.default_rng(seed)
    picks = []
    for b in sorted({r.band for r in records}):
        sel = [r for r in records if r.band == b]
        picks.append(sel[int(rng.integers(len(sel)))])
    parts = []
    for title, rs in (("largest errors below 1400", [r for r, _ in low]), ("one per band", picks)):
        for k, r in enumerate(rs):
            parts.append(f"=== {title}, {k + 1}\n" + cal.case_file(r, arm, r.extra.get("sans")))
    return "\n\n".join(parts) + "\n"


def information_by_kind(records: list[Record]) -> dict:
    """Which moves carry the information: the mean absolute log-likelihood ratio between ratings
    200 above and 200 below the truth, per move, by ply group and by whether the move was
    Maia-3's top move."""
    out: dict[str, list[float]] = {}
    for r in records:
        lo, hi = cal.at(r.logp, np.array([r.rating - 200, r.rating + 200])).T
        top = r.extra.get("top", np.zeros(len(lo)))
        for j in range(len(lo)):
            for key in (
                f"ply {cal.ply_group(int(r.ply[j]))}",
                "top move" if top[j] else "not top move",
            ):
                out.setdefault(key, []).append(float(hi[j] - lo[j]))
    return {
        k: {
            "moves": len(v),
            "mean_abs_ratio": float(np.mean(np.abs(v))),
            "mean_ratio": float(np.mean(v)),
        }
        for k, v in sorted(out.items())
    }


def pilot(strat: list[Record], pop: list[Record], counts: dict, log=print):
    """Diagnosis, cross-validation, the choice, and the fit frozen for the test."""
    result: dict = {"diagnosis": cal.diagnose(strat)}
    log("pilot: diagnosis done")
    result["information_by_move"] = information_by_kind(strat)
    cv = cal.cross_validate(strat)
    result["cross_validation"] = cv
    log("pilot: cross-validation done")
    chosen = cal.choose(cv)
    result["chosen"] = chosen
    fitted = {name: cal.fit_arm(name, strat) for name in ARMS}
    result["fitted_constants"] = {
        n: {"tau": a.tau, "game_sigma": a.game_sigma, "move_sigma": a.move_sigma}
        for n, a in fitted.items()
    }
    result["in_sample"] = {n: cal.by_band(evaluate(a, strat)) for n, a in fitted.items()}
    prior = cal.prior_from_counts(counts)
    best = fitted[chosen]
    with_prior = cal.replace(best, prior=prior)
    tau_pop = cal.fit_tau(with_prior, pop)
    population_arm = cal.replace(with_prior, name=chosen + "+P", tau=tau_pop)
    result["population_pilot"] = {
        "F0": cal.by_band(evaluate(fitted["F0"], pop)),
        chosen: cal.by_band(evaluate(best, pop)),
        chosen + "+P": cal.by_band(evaluate(population_arm, pop)),
        "tau_with_prior": tau_pop,
    }
    result["prior_counts_players"] = {k: int(sum(v.values())) for k, v in counts.items()}
    sd = np.std([e["estimate"] - r.rating for r, e in evaluate(best, strat) if r.band in LOWER])
    result["detectable_bias_lower_bands"] = float(
        2.80 * sd / np.sqrt(len({r.game for r in strat if r.band in LOWER}))
    )
    fit = {
        "chosen": chosen,
        "arms": {n: arm_to_json(a) for n, a in fitted.items()},
        "population": arm_to_json(population_arm),
    }
    return result, fit, cases(strat, fitted["F1"])


def test(strat: list[Record], pop: list[Record], fit: dict) -> dict:
    """The plan's primary and secondary results on the test games."""
    arms = {n: arm_from_json(d) for n, d in fit["arms"].items()}
    chosen = fit["chosen"]
    old, new = arms["F0"], arms[chosen]
    eo, en = evaluate(old, strat), evaluate(new, strat)
    err_o = np.array([e["estimate"] - r.rating for r, e in eo])
    err_n = np.array([e["estimate"] - r.rating for r, e in en])
    cov_o = np.array([e["lo"] <= r.rating <= e["hi"] for r, e in eo])
    cov_n = np.array([e["lo"] <= r.rating <= e["hi"] for r, e in en])
    low = np.array([r.band in LOWER for r in strat])
    low_records = [r for r, m in zip(strat, low, strict=True) if m]
    lo_o, lo_n = err_o[low], err_n[low]

    def gain(idx):
        return abs(lo_o[idx].mean()) - abs(lo_n[idx].mean())

    interval, vals = cal.game_bootstrap(low_records, gain, seed=11)
    p_bias = float((np.sum(vals <= 0) + 1) / (len(vals) + 1))
    new_only = int(np.sum(cov_n & ~cov_o))
    old_only = int(np.sum(cov_o & ~cov_n))
    p_cov = cal.mcnemar_one_sided(new_only, old_only)
    adjusted = cal.holm({"bias": p_bias, "coverage": p_cov})
    (bias_lo, bias_hi), _ = cal.game_bootstrap(low_records, lambda i: lo_n[i].mean(), seed=12)
    (old_lo, old_hi), _ = cal.game_bootstrap(low_records, lambda i: lo_o[i].mean(), seed=13)
    primary = {
        "chosen": chosen,
        "bias_lower_bands_old": float(lo_o.mean()),
        "bias_lower_bands_old_interval": [old_lo, old_hi],
        "bias_lower_bands_new": float(lo_n.mean()),
        "bias_lower_bands_new_interval": [bias_lo, bias_hi],
        "bias_reduction": float(gain(np.arange(len(lo_o)))),
        "bias_reduction_interval": list(interval),
        "bias_p": p_bias,
        "coverage_old": float(cov_o.mean()),
        "coverage_old_wilson": cal.wilson(int(cov_o.sum()), len(cov_o)),
        "coverage_new": float(cov_n.mean()),
        "coverage_new_wilson": cal.wilson(int(cov_n.sum()), len(cov_n)),
        "covered_by_new_only": new_only,
        "covered_by_old_only": old_only,
        "coverage_p": p_cov,
        "holm": adjusted,
        "player_games": len(strat),
        "games": len({r.game for r in strat}),
        "player_games_lower_bands": int(low.sum()),
    }
    secondary: dict = {"arms": {}, "mae_intervals": {}}
    for k, (name, arm) in enumerate(sorted(arms.items())):
        rows = evaluate(arm, strat)
        secondary["arms"][name] = cal.by_band(rows)
        abs_err = np.array([abs(e["estimate"] - r.rating) for r, e in rows])
        (a, b), _ = cal.game_bootstrap(strat, lambda i, x=abs_err: x[i].mean(), seed=14 + k)
        secondary["mae_intervals"][name] = [a, b]
    (a, b), _ = cal.game_bootstrap(
        strat, lambda i: np.abs(err_o[i]).mean() - np.abs(err_n[i]).mean(), seed=20
    )
    secondary["mae_reduction"] = float(np.abs(err_o).mean() - np.abs(err_n).mean())
    secondary["mae_reduction_interval"] = [a, b]
    truth = np.array([r.rating for r in strat])
    secondary["baseline_mean_rating_mae"] = float(np.abs(truth - truth.mean()).mean())
    secondary["move_matching_at_true_rating"] = float(
        np.mean(np.concatenate([r.extra["top"] for r in strat]))
    )
    population_arm = arm_from_json(fit["population"])
    secondary["population"] = {
        "F0": cal.by_band(evaluate(old, pop)),
        chosen: cal.by_band(evaluate(new, pop)),
        population_arm.name: cal.by_band(evaluate(population_arm, pop)),
    }
    pop_truth = np.array([r.rating for r in pop])
    secondary["population_baseline_mean_rating_mae"] = float(
        np.abs(pop_truth - pop_truth.mean()).mean()
    )
    return {"primary": primary, "secondary": secondary}
