"""Several vectors per board word, one per WordNet sense, scored by the best match.

Codenames words are chosen to be ambiguous, and a clue usually works through one sense
("bat" for "cave" is the animal, for "glove" the club). Here each board word w gets its
bare vector plus one vector for each of its first three WordNet noun senses, embedding
the text "w: gloss" (the gloss up to its first semicolon). The similarity of a clue to w
is the largest of these. The idea is the one behind miniCOIL (Qdrant), which learns
sense-separating vectors for retrieval; here the senses come from WordNet instead.

    uv run python -m codenames.senses       # compares plain and sense-aware scores
"""

from functools import cache

import numpy as np

from codenames.embed import vectors
from codenames.evaluate import all_words, association_items, association_top1, guess_scores
from codenames.fetch import fetch
from codenames.norms import cultural_guesses, eat_counts
from codenames.players import Space, space
from codenames.words import board_words

N_SENSES = 3


@cache
def glosses() -> dict[str, list[str]]:
    """Noun lemma -> glosses of its first N_SENSES senses, most frequent first."""
    d = fetch("wordnet")
    gloss = {}
    for line in (d / "data.noun").read_text(encoding="latin-1").splitlines():
        if not line.startswith("  ") and "|" in line:
            gloss[line.split()[0]] = line.split("|", 1)[1].split(";")[0].strip()
    out = {}
    for line in (d / "index.noun").read_text(encoding="latin-1").splitlines():
        if line.startswith("  "):
            continue
        f = line.split()
        offsets = f[6 + int(f[3]) :]
        out[f[0]] = [gloss[o] for o in offsets[:N_SENSES] if o in gloss]
    return out


class SenseSpace(Space):
    """A space whose word side is the best of the bare vector and the sense vectors."""

    def __init__(self, base: Space, sense_vecs: dict[str, np.ndarray]):
        super().__init__(base.name + "+senses", base.clue_vecs, base.word_vecs, base.words)
        self.sense_vecs = sense_vecs

    def sims(self, clue: str, words: tuple[str, ...]) -> np.ndarray:
        out = super().sims(clue, words)
        if clue not in self.index:
            return out
        c = self.clue_vecs[self.index[clue]]
        for j, w in enumerate(words):
            if w in self.sense_vecs:
                out[j] = max(out[j], float((self.sense_vecs[w] @ c).max()))
        return out


def sense_space(base: Space, model: str, words) -> SenseSpace:
    g = glosses()
    texts = sorted({f"{w}: {s}" for w in words for s in g.get(w, [])})
    vecs = dict(zip(texts, vectors(model, texts, "bare"), strict=True))
    per_word = {w: np.stack([vecs[f"{w}: {s}"] for s in g[w]]) for w in words if g.get(w)}
    return SenseSpace(base, per_word)


def main() -> None:
    words = all_words()
    targets = sorted(
        set(board_words())
        | {w for s in ("train", "val", "test") for g in cultural_guesses(s) for w in g.words}
    )
    items = association_items(eat_counts(), set(words))
    out = {}
    for model in ("potion", "wordllama", "granite", "bge"):
        base = space(model, words=words)
        sense = sense_space(base, model, targets)
        out[model] = {
            "plain": {"guess": guess_scores(base, "test"), "assoc": association_top1(base, items)},
            "senses": {
                "guess": guess_scores(sense, "test"),
                "assoc": association_top1(sense, items),
            },
        }
        p, s = out[model]["plain"], out[model]["senses"]
        print(
            model,
            f"guess {p['guess']['top1']:.3f} -> {s['guess']['top1']:.3f}",
            f"loglik {p['guess']['loglik']:.3f} -> {s['guess']['loglik']:.3f}",
            f"EAT {p['assoc']['top1']:.3f} -> {s['assoc']['top1']:.3f}",
            flush=True,
        )
    from codenames.study import save

    save("senses", out)


if __name__ == "__main__":
    main()
