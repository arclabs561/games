"""The three questions, answered from the committed aggregates (and, on Actions, dev decks).

    uv run --group mtg_decks python -m mtg_decks.study            # writes results/BLB.json

1. Which cards go together? Neighbors in two embeddings, against a shuffled-deck null.
2. How should a deck change to look like the winning decks? Transport against the
   frequency-difference baseline, on the average losing deck of each color pair here,
   and on held-out decks in `score_heldout` (run by `build.py` on Actions).
3. A diverse list of suggestions: MMR's trade-off between relevance and redundancy.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from mtg_decks import cards as cardfacts
from mtg_decks import decks as dk
from mtg_decks import embed, mmr, transport

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"

DIM = 32  # embedding dimensions, both methods
K = 10  # neighbors per card
SGNS_SEEDS = [1, 2, 3, 4, 5]
NULL_SEEDS = [1, 2, 3, 4, 5]
SWAPS_SHOWN = 5
LAMBDAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
LIST_K = 5  # suggestions per list
BOOT = 2000  # bootstrap resamples (exploratory)


@dataclass
class Data:
    """The committed aggregates for one set."""

    cards: list[cardfacts.Card]
    s: np.ndarray  # decks holding both cards
    held: np.ndarray  # decks holding each card
    sizes: np.ndarray  # vocabulary cards in each fit deck
    targets: dict[tuple[str, str], np.ndarray]  # (pair, "won"/"lost") -> measure

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.cards]

    @property
    def n_decks(self) -> int:
        return int(self.sizes.size)


def load(folder: Path) -> Data:
    """Read cards.csv, pairs.csv, sizes.csv and targets.csv from a set's folder."""
    cards = cardfacts.read_cards(folder / "cards.csv")
    names = [c.name for c in cards]
    s, _ = dk.read_pairs(folder / "pairs.csv", names)
    with open(folder / "sizes.csv", newline="", encoding="utf-8") as f:
        sizes = np.concatenate(
            [np.full(int(r["decks"]), int(r["cards"])) for r in csv.DictReader(f)]
        )
    index = {n: i for i, n in enumerate(names)}
    targets: dict[tuple[str, str], np.ndarray] = {}
    with open(folder / "targets.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = targets.setdefault((r["group"], r["outcome"]), np.zeros(len(names)))
            v[index[r["card"]]] = float(r["mass"])
    for k, v in targets.items():
        targets[k] = v / v.sum()  # undo the rounding of the written masses
    held = np.array([c.decks for c in cards])
    return Data(cards, s, held, sizes, targets)


# ---- Question 1: which cards go together? ------------------------------------------


def ppmi_embedding(s: np.ndarray, held: np.ndarray, n_decks: int) -> np.ndarray:
    """The exact baseline: PPMI over decks, then its top DIM eigenvectors."""
    return embed.eigen_embedding(embed.ppmi(s, held, n_decks), DIM)


def sgns_embedding(s: np.ndarray, seed: int) -> np.ndarray:
    """The learned method: skip-gram with negative sampling, word plus context vectors."""
    w, c = embed.sgns(s, DIM, seed)
    return w + c


def neighbor_tests(data: Data, x: np.ndarray, tribes: dict[str, str] | None) -> dict:
    """What a card's K nearest neighbors share with it, and the same by chance."""
    nb = embed.neighbors(x, K)
    cards = data.cards
    pair = [c.pair or None for c in cards]
    out = {"pair": embed.agreement(nb, pair)}
    mono = [i for i, c in enumerate(cards) if len(c.colors) == 1]
    out["color"] = float(
        np.mean([np.mean([cards[i].colors in cards[j].colors for j in nb[i]]) for i in mono])
    )
    if tribes:
        hits = []
        for i, c in enumerate(cards):
            partner = cardfacts.tribe_partner(c, tribes)
            if partner is None:
                continue
            votes = {col: 0 for col in cardfacts.WUBRG if col != c.colors}
            for j in nb[i]:
                for col in cards[j].colors:
                    if col != c.colors:
                        votes[col] += 1
            top = max(votes.values())
            tied = [col for col, v in votes.items() if v == top]
            hits.append(1 / len(tied) if top > 0 and partner in tied else 0.0)
        out["tribe_partner"] = float(np.mean(hits))
        out["tribe_cards"] = len(hits)
    return out


def chance_tests(data: Data, tribes: dict[str, str] | None) -> dict:
    """The same measures if each card's neighbors were drawn at random from the rest."""
    cards = data.cards
    n = len(cards)
    out = {"pair": embed.chance_agreement([c.pair or None for c in cards])}
    mono = [i for i, c in enumerate(cards) if len(c.colors) == 1]
    out["color"] = float(
        np.mean(
            [
                sum(cards[i].colors in cards[j].colors for j in range(n) if j != i) / (n - 1)
                for i in mono
            ]
        )
    )
    if tribes:
        out["tribe_partner"] = 0.25  # four other colors, none favored
    return out


def question1(data: Data, tribes: dict[str, str] | None) -> dict:
    """Neighbor tests for PPMI, for SGNS over seeds, for the null over seeds, and chance."""
    base = neighbor_tests(data, ppmi_embedding(data.s, data.held, data.n_decks), tribes)
    learned = [neighbor_tests(data, sgns_embedding(data.s, seed), tribes) for seed in SGNS_SEEDS]
    nulls = []
    for seed in NULL_SEEDS:
        x0 = embed.null_counts(data.held, data.sizes, seed)
        s0, _, h0 = dk.cooccurrence(x0)
        nulls.append(neighbor_tests(data, ppmi_embedding(s0, h0, data.n_decks), tribes))

    def spread(runs: list[dict]) -> dict:
        return {
            k: {
                "mean": float(np.mean([r[k] for r in runs])),
                "min": min(r[k] for r in runs),
                "max": max(r[k] for r in runs),
            }
            for k in runs[0]
            if k != "tribe_cards"
        }

    return {
        "k": K,
        "dim": DIM,
        "cards": len(data.cards),
        "ppmi": base,
        "sgns": spread(learned),
        "sgns_seeds": SGNS_SEEDS,
        "null": spread(nulls),
        "null_seeds": NULL_SEEDS,
        "chance": chance_tests(data, tribes),
    }


# ---- Question 2: deck edits as transport -------------------------------------------


def name_swaps(sw: list[transport.Swap], names: list[str], size: float, n: int) -> list[dict]:
    """The n largest swaps as names, with mass in copies per deck of the given size."""
    return [{"cut": names[i], "add": names[j], "copies": round(m * size, 3)} for i, j, m in sw[:n]]


def random_pairing(mu: np.ndarray, nu: np.ndarray, seed: int) -> list[transport.Swap]:
    """Swaps with the same cuts and adds, paired by transport under a random cost."""
    rng = np.random.default_rng(seed)
    cost = rng.random((mu.size, mu.size))
    np.fill_diagonal(cost, 0.0)
    kept = np.minimum(mu, nu)
    plan, _ = transport.emd(mu - kept, nu - kept, cost)
    return transport.swaps(plan)


def pairing_scores(sw: list[transport.Swap], data: Data) -> dict:
    """How alike the two cards of a swap are, in facts the embedding never saw."""
    mv = np.array([c.mana_value for c in data.cards], dtype=float)
    kind = [c.kind for c in data.cards]
    return {
        "mana_gap": transport.pairing_gap(sw, mv),
        "same_kind": transport.pairing_match(sw, kind),
    }


def question2(data: Data, x: np.ndarray, deck_size: float) -> dict:
    """For each color pair: move the average losing deck onto the average winning deck."""
    cost = transport.ground_cost(x)
    out = {}
    for g in cardfacts.PAIRS:
        lost, won = data.targets[(g, "lost")], data.targets[(g, "won")]
        plan, c = transport.emd(lost, won, cost)
        ot_sw = transport.swaps(plan)
        base_sw = transport.baseline_swaps(lost, won)
        out[g] = {
            "moved": round(transport.total_variation(lost, won) * deck_size, 4),
            "cost": round(c, 6),
            "transport": name_swaps(ot_sw, data.names, deck_size, SWAPS_SHOWN),
            "baseline": name_swaps(base_sw, data.names, deck_size, SWAPS_SHOWN),
            "pairing": {
                "transport": pairing_scores(ot_sw, data),
                "baseline": pairing_scores(base_sw, data),
                "random": pairing_scores(random_pairing(lost, won, 0), data),
            },
        }
    return out


# ---- Question 3: diverse suggestions -----------------------------------------------


def role(c: cardfacts.Card) -> str:
    """A card's role for counting variety: type and mana value."""
    return f"{c.kind} {c.mana_value}"


def tradeoff(rel: np.ndarray, sim: np.ndarray, roles: list[str]) -> list[dict]:
    """MMR over the lambda grid: relevance kept, redundancy and distinct roles."""
    rows = []
    for lam in LAMBDAS:
        picked = mmr.mmr(rel, sim, lam, LIST_K)
        rows.append(
            {
                "lambda": lam,
                "relevance_kept": mmr.relevance_kept(rel, picked),
                "redundancy": mmr.redundancy(sim, picked),
                "roles": mmr.distinct(roles, picked),
                "picked": picked,
            }
        )
    return rows


def question3(data: Data, x: np.ndarray) -> dict:
    """Suggestions to add to each pair's average losing deck: the cards winners have more of."""
    sim = embed.cosine(x)
    roles = [role(c) for c in data.cards]
    per_pair = {}
    for g in cardfacts.PAIRS:
        gap = data.targets[(g, "won")] - data.targets[(g, "lost")]
        cand = np.flatnonzero(gap > 0)
        rows = tradeoff(gap[cand], sim[np.ix_(cand, cand)], [roles[i] for i in cand])
        for r in rows:
            r["picked"] = [data.names[cand[i]] for i in r["picked"]]
        per_pair[g] = rows
    mean = [
        {
            "lambda": lam,
            "relevance_kept": float(np.mean([per_pair[g][t]["relevance_kept"] for g in per_pair])),
            "redundancy": float(np.mean([per_pair[g][t]["redundancy"] for g in per_pair])),
            "roles": float(np.mean([per_pair[g][t]["roles"] for g in per_pair])),
        }
        for t, lam in enumerate(LAMBDAS)
    ]
    # exact check: greedy MMR against the best subset of MMR's set objective, on each
    # pair's 12 most relevant candidates, lists of 4
    gaps = []
    for g in cardfacts.PAIRS:
        gap = data.targets[(g, "won")] - data.targets[(g, "lost")]
        top = np.argsort(-gap, kind="stable")[:12]
        r, sm = gap[top], sim[np.ix_(top, top)]
        for lam in (0.7, 0.5, 0.3):
            best = mmr.objective(r, sm, lam, mmr.best_subset(r, sm, lam, 4))
            got = mmr.objective(r, sm, lam, mmr.mmr(r, sm, lam, 4))
            gaps.append(best - got)
    rank_ok = sum(
        mmr.mmr(gap[c], sim[np.ix_(c, c)], lam, LIST_K)
        != mmr.mmr_rankops(gap[c], sim[np.ix_(c, c)], lam, LIST_K)
        for g in cardfacts.PAIRS
        for gap in [data.targets[(g, "won")] - data.targets[(g, "lost")]]
        for c in [np.flatnonzero(gap > 0)]
        for lam in LAMBDAS
    )
    return {
        "k": LIST_K,
        "mean": mean,
        "per_pair": per_pair,
        "greedy_gap": {
            "max": float(max(gaps)),
            "mean": float(np.mean(gaps)),
            "cases": len(gaps),
            "exact_ties": int(sum(gp < 1e-12 for gp in gaps)),
        },
        "rankops_disagreements": rank_ok,
    }


# ---- Held-out decks (run on Actions by build.py) -------------------------------------


def stratified_auc(
    score: np.ndarray, pos: np.ndarray, neg: np.ndarray, strata: np.ndarray, weights
) -> np.ndarray:
    """Chance that a won game's deck outscores a lost game's deck of the same stratum.

    Ties count one half. Games are weighted: a deck with w wins and l losses counts as w
    won and l lost games. `weights` yields one vector of deck weights per bootstrap
    resample; a single vector of ones gives the plain estimate.
    """
    order = np.lexsort((score, strata))
    st, sc = strata[order], score[order]
    new = np.ones(order.size, dtype=bool)
    new[1:] = (st[1:] != st[:-1]) | (sc[1:] != sc[:-1])
    grp = np.cumsum(new) - 1
    first_of_stratum = np.ones(order.size, dtype=bool)
    first_of_stratum[1:] = st[1:] != st[:-1]
    g_stratum = np.cumsum(first_of_stratum)[new] - 1
    n_groups = int(grp[-1]) + 1
    out = []
    for w in weights:
        p = np.bincount(grp, weights=(pos * w)[order], minlength=n_groups)
        q = np.bincount(grp, weights=(neg * w)[order], minlength=n_groups)
        below = np.cumsum(q) - q
        start = np.zeros(int(g_stratum[-1]) + 1)
        total_q = np.bincount(g_stratum, weights=q)
        start[1:] = np.cumsum(total_q)[:-1]
        below = below - start[g_stratum]
        num = (p * (below + 0.5 * q)).sum()
        den = (np.bincount(g_stratum, weights=p) * total_q).sum()
        out.append(num / den)
    return np.array(out)


def interval(v: np.ndarray) -> list[float]:
    """The 2.5% and 97.5% percentiles."""
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def score_heldout(data: Data, dev: dk.Decks, seed: int = 0, sample: int = 3000) -> dict:
    """Exploratory scores on held-out dev decks for questions 2 and 3."""
    index = {n: i for i, n in enumerate(dev.cards)}
    cols = [index[n] for n in data.names]
    x = dev.counts[:, cols].astype(np.float64)
    size = x.sum(axis=1)
    keep = (size > 0) & np.isin(dev.group, cardfacts.PAIRS) & (dev.games > 0)
    d = dev.take(keep)
    x, size = x[keep], size[keep]
    mu = x / size[:, None]
    emb = ppmi_embedding(data.s, data.held, data.n_decks)
    cost = transport.ground_cost(emb)
    won = np.array([data.targets[(g, "won")] for g in d.group])
    lost = np.array([data.targets[(g, "lost")] for g in d.group])
    lift = won - lost
    ot_won = np.array([transport.emd(mu[r], won[r], cost)[1] for r in range(len(d))])
    ot_lost = np.array([transport.emd(mu[r], lost[r], cost)[1] for r in range(len(d))])
    tv_won = 0.5 * np.abs(mu - won).sum(axis=1)
    tv_lost = 0.5 * np.abs(mu - lost).sum(axis=1)
    scores = {
        "frequency_difference": (mu * lift).sum(axis=1),
        "transport": ot_lost - ot_won,
        "transport_to_winners": -ot_won,
        "total_variation": tv_lost - tv_won,
        "skill": np.nan_to_num(d.skill, nan=-1.0),
    }
    pos, neg = d.wins.astype(float), (d.games - d.wins).astype(float)
    skill = np.nan_to_num(d.skill, nan=-1.0)
    gcode = np.array([cardfacts.PAIRS.index(g) for g in d.group])
    by_skill = gcode * 1000 + np.round(skill * 100).astype(int)
    rng = np.random.default_rng(seed)
    drafts, inv = np.unique(d.draft.astype(str), return_inverse=True)

    def w():
        """The plain weights, then BOOT resamples of drafts, the same for every score."""
        yield np.ones(len(d))
        for b in range(BOOT):
            r = np.random.default_rng([seed, b])
            yield np.bincount(r.integers(0, drafts.size, size=drafts.size), minlength=drafts.size)[
                inv
            ]

    auc = {}
    for name, sc in scores.items():
        for label, strata in (("pair", gcode), ("pair_and_skill", by_skill)):
            if name == "skill" and label == "pair_and_skill":
                continue
            v = stratified_auc(sc, pos, neg, strata, w())
            auc[f"{name}|{label}"] = v
    summary = {k: {"auc": float(v[0]), "interval": interval(v[1:])} for k, v in auc.items()}
    diffs = {}
    for name in ("transport", "transport_to_winners", "total_variation"):
        for label in ("pair", "pair_and_skill"):
            dv = auc[f"{name}|{label}"] - auc[f"frequency_difference|{label}"]
            diffs[f"{name} - frequency_difference|{label}"] = {
                "diff": float(dv[0]),
                "interval": interval(dv[1:]),
            }
    known = ~np.isnan(d.skill)
    rho = {
        name: float(spearmanr(sc[known], d.skill[known]).statistic)
        for name, sc in scores.items()
        if name != "skill"
    }
    # pairing and lists on a seeded sample of decks
    pick = np.sort(rng.choice(len(d), size=min(sample, len(d)), replace=False))
    pair_rows = {"transport": [], "baseline": [], "random": []}
    sim = embed.cosine(emb)
    roles = [role(c) for c in data.cards]
    lists = {lam: {"relevance_kept": [], "redundancy": [], "roles": []} for lam in LAMBDAS}
    disagree = {str(lam): 0 for lam in LAMBDAS}
    for r in pick:
        plan, _ = transport.emd(mu[r], won[r], cost)
        for name, sw in (
            ("transport", transport.swaps(plan)),
            ("baseline", transport.baseline_swaps(mu[r], won[r])),
            ("random", random_pairing(mu[r], won[r], int(r))),
        ):
            pair_rows[name].append(pairing_scores(sw, data))
        gap = won[r] - mu[r]
        cand = np.flatnonzero(gap > 0)
        if cand.size < LIST_K:
            continue
        sm = sim[np.ix_(cand, cand)]
        for t in tradeoff(gap[cand], sm, [roles[i] for i in cand]):
            for k in ("relevance_kept", "redundancy", "roles"):
                lists[t["lambda"]][k].append(t[k])
            if t["picked"] != mmr.mmr_rankops(gap[cand], sm, t["lambda"], LIST_K):
                disagree[str(t["lambda"])] += 1

    def mean_ci(vals: list[float]) -> dict:
        v = np.array(vals)
        boots = rng.choice(v, size=(BOOT, v.size)).mean(axis=1)
        return {"mean": float(v.mean()), "interval": interval(boots)}

    pairing = {
        name: {k: mean_ci([row[k] for row in rows]) for k in ("mana_gap", "same_kind")}
        for name, rows in pair_rows.items()
    }
    return {
        "decks": int(len(d)),
        "drafts": int(drafts.size),
        "games": int(d.games.sum()),
        "wins": int(d.wins.sum()),
        "bootstrap": {"resamples": BOOT, "seed": seed, "unit": "draft"},
        "auc": summary,
        "auc_differences": diffs,
        "spearman_with_skill": rho,
        "pairing": pairing,
        "pairing_sample": int(pick.size),
        "lists": {
            str(lam): {k: mean_ci(v) for k, v in vals.items()} for lam, vals in lists.items()
        },
        "lists_rankops_disagreements": disagree,
    }


# ---- Entry point -------------------------------------------------------------------


def run(expansion: str) -> dict:
    """All local results for one set."""
    data = load(DATA / expansion)
    tribes = cardfacts.BLB_TRIBES if expansion == "BLB" else None
    x = ppmi_embedding(data.s, data.held, data.n_decks)
    source = json.loads((DATA / expansion / "source.json").read_text())
    deck_size = float(source["mean_vocabulary_copies_per_deck"])
    return {
        "set": expansion,
        "fit_decks": data.n_decks,
        "mean_vocabulary_copies_per_deck": deck_size,
        "question1": question1(data, tribes),
        "question2": question2(data, x, deck_size),
        "question3": question3(data, x),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", dest="expansion", default="BLB")
    a = p.parse_args()
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{a.expansion}.json"
    out.write_text(json.dumps(run(a.expansion), indent=1, sort_keys=True) + "\n")
    print(f"wrote {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
