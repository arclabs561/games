"""Build the committed aggregates from a 17Lands game file, and score held-out decks.

Run on GitHub Actions, which can reach the 17Lands bucket:

    uv run --group mtg_decks python -m mtg_decks.build --set BLB --out OUT

It downloads the set's PremierDraft game file and the card list, and writes into OUT:

- `pairs.csv`: co-occurrence over the fit decks, in decksage's pair format;
- `cards.csv`: the vocabulary and its facts (see `cards.py`);
- `sizes.csv`: how many fit decks hold each number of vocabulary cards (for the null);
- `targets.csv`: per color pair, the average deck of the fit games won and lost;
- `source.json`: file hashes, sizes, row and split counts, and the license;
- `dev.json`: the exploratory scores of held-out dev decks (questions 2 and 3).

Only aggregates leave the runner. The test split is counted and nothing else.
"""

import argparse
import csv
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from mtg_decks import cards as cardfacts
from mtg_decks import decks as dk
from mtg_decks import seventeen_lands as sl
from mtg_decks import study

MIN_DECKS = 100  # a card enters the vocabulary if at least this many fit decks hold it


def main_builds(d: dk.Decks) -> np.ndarray:
    """Mask of one build per draft: the one with most games, the first on ties."""
    best: dict[str, int] = {}
    for r in range(len(d)):
        k = str(d.draft[r])
        if k not in best or d.games[r] > d.games[best[k]]:
            best[k] = r
    mask = np.zeros(len(d), dtype=bool)
    mask[list(best.values())] = True
    return mask


def vocabulary(d: dk.Decks, min_decks: int) -> list[int]:
    """Columns of the nonbasic cards held by at least min_decks decks of d, by name."""
    held = (d.counts > 0).sum(axis=0)
    keep = [i for i, c in enumerate(d.cards) if c not in sl.BASICS and held[i] >= min_decks]
    return sorted(keep, key=lambda i: d.cards[i])


def card_rows(d: dk.Decks, cols: list[int], facts: dict[str, dict]) -> list[cardfacts.Card]:
    """Facts for the vocabulary: from the card list, plus counts over the decks d."""
    out = []
    two = np.array([g in cardfacts.PAIRS for g in d.group])
    for i in cols:
        name = d.cards[i]
        f = facts.get(name, {})
        holds = d.counts[:, i] > 0
        pairs = Counter(d.group[holds & two].tolist())
        top, n = min(pairs.items(), key=lambda kv: (-kv[1], kv[0])) if pairs else ("", 0)
        out.append(
            cardfacts.Card(
                name=name,
                colors=cardfacts.wubrg(f.get("color_identity", "")),
                mana_value=int(float(f.get("mana_value") or 0)),
                kind=cardfacts.kind_of(f.get("types", "")),
                subtypes=cardfacts.subtypes_of(f.get("types", "")),
                rarity=f.get("rarity", ""),
                decks=int(holds.sum()),
                pair=top,
                pair_share=n / max(1, sum(pairs.values())),
            )
        )
    return out


def targets(d: dk.Decks, cols: list[int]) -> dict[tuple[str, str], np.ndarray]:
    """Per color pair, the game-weighted average deck measure of games won and lost."""
    x = d.counts[:, cols].astype(np.float64)
    size = x.sum(axis=1)
    ok = size > 0
    mu = np.zeros_like(x)
    mu[ok] = x[ok] / size[ok, None]
    out = {}
    for g in cardfacts.PAIRS:
        m = (d.group == g) & ok
        for outcome, w in (("won", d.wins[m]), ("lost", d.games[m] - d.wins[m])):
            out[(g, outcome)] = (w[:, None] * mu[m]).sum(axis=0) / w.sum()
    return out


def write_targets(path: Path, names: list[str], t: dict[tuple[str, str], np.ndarray]) -> None:
    """Write the targets as group, outcome, card, mass (rows with mass above 0)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["group", "outcome", "card", "mass"])
        for (g, o), v in sorted(t.items()):
            for i in np.flatnonzero(v > 0):
                w.writerow([g, o, names[i], f"{v[i]:.8f}"])


def build(
    game_file: Path, cards_file: Path, expansion: str, out: Path, min_decks: int, meta: dict
) -> None:
    """Everything after the download: read, split, write aggregates, score dev decks."""
    out.mkdir(parents=True, exist_ok=True)
    facts = sl.read_cards(cards_file, expansion)
    allx = sl.read_games(game_file)
    split = np.array([dk.split_of(str(x)) for x in allx.draft])
    meta["splits"] = {
        s: {
            "drafts": int(len(set(allx.draft[split == s].tolist()))),
            "decks": int((split == s).sum()),
            "games": int(allx.games[split == s].sum()),
        }
        for s in dk.SPLITS
    }
    fit = allx.take(split == "fit")
    fit_main = fit.take(main_builds(fit))
    cols = vocabulary(fit_main, min_decks)
    names = [fit_main.cards[i] for i in cols]
    vocab = card_rows(fit_main, cols, facts)
    cardfacts.write_cards(out / "cards.csv", vocab)
    x = fit_main.counts[:, cols]
    s, m, held = dk.cooccurrence(x)
    dk.write_pairs(out / "pairs.csv", names, s, m)
    sizes = Counter((x > 0).sum(axis=1).tolist())
    with open(out / "sizes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["cards", "decks"])
        for k in sorted(sizes):
            w.writerow([k, sizes[k]])
    write_targets(out / "targets.csv", names, targets(fit, cols))
    out_of_vocab = fit_main.counts.sum() - x.sum()
    basics = sum(
        fit_main.counts[:, i].sum() for i, c in enumerate(fit_main.cards) if c in sl.BASICS
    )
    meta["fit_decks_for_cooccurrence"] = int(len(fit_main))
    meta["vocabulary"] = len(names)
    meta["mean_vocabulary_copies_per_deck"] = round(float(x.sum(axis=1).mean()), 4)
    meta["min_decks"] = min_decks
    meta["share_of_nonbasic_copies_outside_vocabulary"] = round(
        float((out_of_vocab - basics) / (fit_main.counts.sum() - basics)), 5
    )
    (out / "source.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    dev = allx.take(split == "dev")
    data = study.load(out)
    result = study.score_heldout(data, dev)
    (out / "dev.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", dest="expansion", default="BLB")
    p.add_argument("--event", default="PremierDraft")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-decks", type=int, default=MIN_DECKS)
    p.add_argument("--game-file", type=Path, help="a local game file instead of the download")
    p.add_argument("--cards-file", type=Path, help="a local card list instead of the download")
    a = p.parse_args()
    meta: dict = {"license": sl.LICENSE, "set": a.expansion, "event": a.event}
    with tempfile.TemporaryDirectory() as tmp:
        game, cards = a.game_file, a.cards_file
        if game is None:
            game = Path(tmp) / "games.csv.gz"
            meta["game_file"] = sl.download(sl.game_data_url(a.expansion, a.event), game)
        if cards is None:
            cards = Path(tmp) / "cards.csv"
            meta["cards_file"] = sl.download(sl.CARDS_URL, cards)
        build(game, cards, a.expansion, a.out, a.min_decks, meta)


if __name__ == "__main__":
    main()
