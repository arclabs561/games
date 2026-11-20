"""Plan P2 (experiments/plans/P2.md): does the full-battle equilibrium survive a policy that
switches?

The battles run on GitHub Actions (``teams_showdown.py play``, one matrix per workflow
run); ``teams.py summarize`` merges each run's outputs into a summary, committed as
``results/teams-battles-p2-<label>.json``. This module reads those summaries:

    uv run python -m pokemon.p2 --out pokemon/results/p2.json

- **Exploitability.** eps_X(sigma) = max over teams j of sum_i sigma_i M_X[j][i], the best
  payoff any of the 63 teams gets against the mixture sigma; M[i][j] = (wins - losses) /
  battles of team i against team j.
- **H-survive.** Delta = eps_S(sigma_D) - eps_D'(sigma_D), with sigma_D the published
  equilibrium (``teams.json``), M_S the matrix with both sides switching and M_D' a fresh
  replicate of the published policy. 95% percentile interval by bootstrap over battles
  within each pair, both matrices resampled independently (10,000 resamples, seed 0).
- **The noise model**: the published matrix with each pair's 20 battles redrawn from its
  observed shares, 2,000 times (seed 0): eps(sigma_D) of a fresh replicate.
- Secondary: M_S's exact equilibrium and its overlap with sigma_D, the six-most-used team's
  weight, the ranking share (Hodge), Kendall's tau of the rankings, the strength check (S
  against D in mirror battles, Wilson interval), and counts per battle.
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from pokemon import hodge
from pokemon import teams as tt
from pokemon.equilibrium import exact_equilibrium, has_unique_certificate, lp_strategy

HERE = Path(__file__).parent
RESULTS = HERE / "results"
OUT = RESULTS / "p2.json"
PLAN = "experiments/plans/P2.md"
LABELS = ("p2-replicate", "p2-switch", "p2-strength")
K = 63
RESAMPLES, SEED = 10_000, 0
NOISE_DRAWS = 2000
WEIGHT_RESAMPLES = 1000  # the LP per resample is slow; the folder's own count
MDE_Z = 2.80
SIX_MOST_USED = 0  # index of the team of the six most used Pokémon in teams-pool.json


def summary_path(label: str) -> Path:
    return RESULTS / f"teams-battles-{label}.json"


def published_sigma() -> list[Fraction]:
    eq = json.loads((RESULTS / "teams.json").read_text())["common"]["battles"]["equilibrium"]
    sigma = [Fraction(0)] * K
    for k, v in eq.items():
        sigma[int(k)] = Fraction(v)
    assert sum(sigma) == 1
    return sigma


def exploitability(m, sigma) -> tuple[Fraction, int]:
    """(max over j of sum_i sigma_i m[j][i], the first j that attains it)."""
    vals = [sum((sigma[i] * m[j][i] for i in range(len(m))), Fraction(0)) for j in range(len(m))]
    best = max(vals)
    return best, vals.index(best)


def pair_arrays(summary: dict) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    """Pairs (i < j), their battle counts, and their (wins, losses, ties) for team i."""
    keys = sorted(summary["pairs"], key=lambda s: tuple(map(int, s.split("-"))))
    pairs = [(int(s.split("-")[0]), int(s.split("-")[1])) for s in keys]
    rows = summary["pairs"]
    games = np.array([rows[s]["games"] for s in keys])
    wlt = np.array([[rows[s]["wins"], rows[s]["losses"], rows[s]["ties"]] for s in keys])
    assert (wlt.sum(axis=1) == games).all()
    return pairs, games, wlt


def against(pairs, sigma: np.ndarray, k: int = K) -> np.ndarray:
    """A (K x pairs) array A with (payoffs of the pairs) @ A.T = each team's payoff against
    sigma: for pair (a, b) with payoff x to a, a gets sigma_b x and b gets -sigma_a x."""
    a = np.zeros((k, len(pairs)))
    for p, (i, j) in enumerate(pairs):
        a[i, p] = sigma[j]
        a[j, p] = -sigma[i]
    return a


def resampled_eps(summary: dict, sigma: np.ndarray, draws: int, rng) -> np.ndarray:
    """eps(sigma) of `draws` matrices, each pair's battles resampled from its shares."""
    pairs, games, wlt = pair_arrays(summary)
    a = against(pairs, sigma)
    x = np.empty((draws, len(pairs)))
    for p in range(len(pairs)):
        d = rng.multinomial(games[p], wlt[p] / games[p], size=draws)
        x[:, p] = (d[:, 0] - d[:, 1]) / games[p]
    return (x @ a.T).max(axis=1)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def support(x) -> set[int]:
    return {i for i, p in enumerate(x) if p > 0}


def team_names(spec: dict) -> list[list[str]]:
    from pokemon import teams_model

    ss = teams_model.sets(spec["pool"])
    return [[ss[i].name for i in t] for t in spec["teams"]]


def battle_counts(summary: dict) -> dict:
    rows = summary["pairs"].values()
    games = sum(r["games"] for r in rows)
    out = {
        "battles": games,
        "turns_mean": sum(r["turns"] for r in rows) / games,
        "ties": sum(r["ties"] for r in rows),
        "choice_errors": int(summary["choice_errors"]),
        "damage_checks": summary["damage_checks"],
        "stat_problems": summary["stat_problems"],
    }
    if any("switches_i" in r for r in rows):
        out["turn_limit_ties"] = sum(r.get("turn_limit", 0) for r in rows)
        out["turn_limit_share"] = out["turn_limit_ties"] / games
        out["voluntary_switches_per_battle"] = {
            "first_team": sum(r["switches_i"] for r in rows) / games,
            "second_team": sum(r["switches_j"] for r in rows) / games,
        }
    return out


def fr(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}"


def analyze(rep: dict, sw: dict, strength: dict | None, published: dict) -> dict:
    sigma = published_sigma()
    sig = np.array([float(s) for s in sigma])
    for s in (rep, sw):
        assert s["teams"] == published["teams"] and s["games"] == 40, "teams or games differ"
    m_rep, m_sw, m_pub = (tt.pair_matrix(s, K, "wins") for s in (rep, sw, published))
    eps_rep, j_rep = exploitability(m_rep, sigma)
    eps_sw, j_sw = exploitability(m_sw, sigma)
    eps_pub, _ = exploitability(m_pub, sigma)
    rng = np.random.default_rng(SEED)
    boot_rep = resampled_eps(rep, sig, RESAMPLES, rng)
    boot_sw = resampled_eps(sw, sig, RESAMPLES, rng)
    delta = boot_sw - boot_rep
    lo, hi = np.percentile(delta, [2.5, 97.5])
    d = float(eps_sw - eps_rep)
    noise = resampled_eps(published, sig, NOISE_DRAWS, np.random.default_rng(SEED))
    names = team_names(published)
    decision = (
        "does not survive" if lo > 0 else "harder to exploit under S" if hi < 0 else "survives"
    )
    out: dict = {
        "plan": PLAN,
        "primary": {
            "eps_S_sigma_D": float(eps_sw),
            "eps_Dprime_sigma_D": float(eps_rep),
            "delta": d,
            "low": float(lo),
            "high": float(hi),
            "resamples": RESAMPLES,
            "seed": SEED,
            "decision": decision,
            "detectable_effect": MDE_Z * float(np.std(delta, ddof=1)),
            "best_response_S": {"team": j_sw, "pokemon": names[j_sw]},
            "best_response_Dprime": {"team": j_rep, "pokemon": names[j_rep]},
        },
        "noise_model": {
            "draws": NOISE_DRAWS,
            "seed": SEED,
            "eps_mean": float(noise.mean()),
            "eps_sd": float(noise.std(ddof=1)),
            "eps_published_matrix_itself": float(eps_pub),
            "bootstrap_sd": {
                "eps_S": float(boot_sw.std(ddof=1)),
                "eps_Dprime": float(boot_rep.std(ddof=1)),
            },
        },
    }
    sec: dict = {}
    for name, m, summ in (("S", m_sw, sw), ("Dprime", m_rep, rep)):
        x, y, v = exact_equilibrium(m)
        sup = support(x)
        sup_d = support(sigma)
        own, _ = exploitability(m, x)
        sec[name] = {
            "equilibrium": {str(i): fr(x[i]) for i in sorted(sup)},
            "unique": has_unique_certificate(m, x, y, v),
            "value": fr(v),
            "support_size": len(sup),
            "jaccard_with_sigma_D": len(sup & sup_d) / len(sup | sup_d),
            "total_variation_to_sigma_D": float(
                sum(abs(a - b) for a, b in zip(x, sigma, strict=True)) / 2
            ),
            "six_most_used_weight": float(x[SIX_MOST_USED]),
            "heaviest": {"team": max(range(K), key=lambda i: x[i]), "weight": float(max(x))},
            "own_exploitability": float(own),
            "gradient_share": float(hodge.gradient_share(m)),
            "kendall_tau_with_published": tt.kendall(hodge.score(m), hodge.score(m_pub)),
            "battles": battle_counts(summ),
        }
        rng_w = np.random.default_rng(SEED)
        top = sec[name]["heaviest"]["team"]
        ws = np.array(
            [
                lp_strategy(tt.bootstrap_pairs(summ, K, rng_w).tolist())[0]
                for _ in range(WEIGHT_RESAMPLES)
            ]
        )
        sec[name]["weight_intervals"] = {
            "resamples": WEIGHT_RESAMPLES,
            "seed": SEED,
            "heaviest": tt.interval(ws[:, top].tolist()),
            "six_most_used": tt.interval(ws[:, SIX_MOST_USED].tolist()),
        }
    sec["published"] = {
        "gradient_share": float(hodge.gradient_share(m_pub)),
        "six_most_used_weight": float(sigma[SIX_MOST_USED]),
        "support_size": len(support(sigma)),
        "heaviest": {"team": max(range(K), key=lambda i: sigma[i]), "weight": float(max(sigma))},
    }
    sec["kendall_tau_S_vs_Dprime"] = tt.kendall(hodge.score(m_sw), hodge.score(m_rep))
    if strength is not None:
        rows = strength["pairs"].values()
        n = sum(r["games"] for r in rows)
        w = sum(r["wins"] for r in rows)
        lose = sum(r["losses"] for r in rows)
        sec["strength"] = {
            "battles": n,
            "S_wins": w,
            "D_wins": lose,
            "ties": n - w - lose,
            "S_win_share": w / n,
            "S_win_share_wilson": wilson(w, n),
            "S_share_of_decided": w / (w + lose),
            "S_share_of_decided_wilson": wilson(w, w + lose),
            "S_stronger": wilson(w, w + lose)[0] > 0.5,
            "battles_detail": battle_counts(strength),
        }
    out["secondary"] = sec
    return out


def rounded(x, places: int = 6):
    if isinstance(x, float):
        return round(x, places)
    if isinstance(x, dict):
        return {k: rounded(v, places) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [rounded(v, places) for v in x]
    return x


def load(label: str) -> dict | None:
    p = summary_path(label)
    return json.loads(p.read_text()) if p.exists() else None


def run() -> dict:
    rep, sw, strength = (load(x) for x in LABELS)
    assert rep is not None and sw is not None, "the two matrices are not committed"
    published = json.loads(tt.BATTLES_JSON.read_text())
    return rounded(analyze(rep, sw, strength, published))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    result = run()
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result["primary"], indent=1))


if __name__ == "__main__":
    main()
