"""Write words.txt, the board words, from WordNet 3.0 and word frequencies.

A board word is a WordNet noun that:
- is one lowercase word of 3 to 9 letters;
- has at least two noun senses (Codenames words are chosen to be ambiguous);
- has a most common sense that names a thing (an animal, artifact, body part,
  food, natural object, person, plant, substance or place), written in lowercase;
- is tagged as a noun in WordNet's sense-tagged corpus at least as often as it is
  tagged as a verb or as an adjective;
- is common: Zipf frequency from 3.6 to 5.6 in wordfreq's English list.

The 400 most frequent such words are kept, after dropping those whose inflections
collide (the board must not hold both "cat" and "cats") and a short list of
words that would be offensive on a family game board.

    uv run python -m codenames.fetch wordnet
    uv run python -m codenames.make_words            # rewrites codenames/words.txt
"""

import argparse
from pathlib import Path

from wordfreq import zipf_frequency

from codenames.fetch import CACHE

HERE = Path(__file__).parent
CONCRETE = {
    "noun.animal",
    "noun.artifact",
    "noun.body",
    "noun.food",
    "noun.object",
    "noun.person",
    "noun.plant",
    "noun.substance",
    "noun.location",
}
# Words a family board would not show, and words whose common use is not the noun
# (checked by reading the list once; each kept out by hand).
EXCLUDE = {
    *("ass", "bitch", "cock", "crap", "dick", "fag", "nigger", "pussy", "shit", "slut", "tit"),
    *("whore", "trump", "john", "jean", "doc", "dude", "sir", "till", "spoke", "chosen"),
    *("closer", "ordinary", "romantic", "classic", "favourite", "centre", "theatre", "colors"),
    *("asshole", "billy", "maria", "slave", "radical", "rep", "pro", "virgin"),
}
N_WORDS = 400


def lexnames(wordnet: Path) -> dict[int, str]:
    """Lexicographer file number -> name, such as 6 -> 'noun.artifact'."""
    out = {}
    for line in (wordnet / "lexnames").read_text().splitlines():
        num, name, _ = line.split()
        out[int(num)] = name
    return out


def synsets(wordnet: Path) -> dict[str, tuple[int, list[str]]]:
    """Noun synset offset -> (lexicographer file number, its words as written)."""
    out = {}
    for line in (wordnet / "data.noun").read_text(encoding="latin-1").splitlines():
        if line.startswith("  "):
            continue  # license header
        fields = line.split()
        n_words = int(fields[3], 16)
        out[fields[0]] = (int(fields[1]), fields[4 : 4 + 2 * n_words : 2])
    return out


def index(wordnet: Path, pos: str) -> dict[str, tuple[list[str], int]]:
    """Lemma -> (synset offsets, most frequent sense first; senses tagged in SemCor)."""
    out = {}
    for line in (wordnet / f"index.{pos}").read_text(encoding="latin-1").splitlines():
        if line.startswith("  "):
            continue
        fields = line.split()
        lemma, n_synsets, n_pointers = fields[0], int(fields[2]), int(fields[3])
        offsets = fields[6 + n_pointers :]
        assert len(offsets) == n_synsets, lemma
        out[lemma] = (offsets, int(fields[5 + n_pointers]))
    return out


def derived(word: str, nouns: dict, verbs: dict) -> bool:
    """True if word is a plural, an -ing or an -ed form of another WordNet lemma."""
    for suffix, base in (("s", nouns), ("es", nouns), ("s", verbs), ("ing", verbs), ("ed", verbs)):
        stem = word.removesuffix(suffix)
        if stem != word and (stem in base or stem + "e" in base):
            return True
    return False


def candidates(wordnet: Path) -> list[tuple[float, str]]:
    """(Zipf frequency, word) for every noun that passes the filters, unsorted."""
    names, noun = lexnames(wordnet), synsets(wordnet)
    nouns, verbs, adjs = (index(wordnet, pos) for pos in ("noun", "verb", "adj"))
    other = [verbs, adjs]
    out = []
    for lemma, (offsets, tagged) in nouns.items():
        if not (lemma.isalpha() and lemma.islower() and 3 <= len(lemma) <= 9):
            continue
        lexfile, written = noun[offsets[0]]
        if len(offsets) < 2 or names[lexfile] not in CONCRETE or lemma in EXCLUDE:
            continue
        if derived(lemma, nouns, verbs):
            continue  # a plural or a verb form ("falls", "hanging") of another lemma
        if lemma not in written:
            continue  # the main sense is a proper noun ("America", "Adam")
        if any(lemma in idx and idx[lemma][1] > tagged for idx in other):
            continue  # used more often as a verb or adjective ("spoke", "annual")
        z = zipf_frequency(lemma, "en")
        if 3.6 <= z <= 5.6:
            out.append((z, lemma))
    return out


def collides(a: str, b: str) -> bool:
    """True if one word is the other plus a common suffix ('cat', 'cats')."""
    short, long = sorted((a, b), key=len)
    return long.startswith(short) and long[len(short) :] in {"s", "es", "er", "ing", "ed"}


def choose(cands: list[tuple[float, str]], n: int = N_WORDS) -> list[str]:
    """The n most frequent candidates, skipping any that collide with one already kept."""
    kept: list[str] = []
    for _, word in sorted(cands, key=lambda zw: (-zw[0], zw[1])):
        if not any(collides(word, k) for k in kept):
            kept.append(word)
        if len(kept) == n:
            break
    return sorted(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wordnet", type=Path, default=CACHE / "wordnet")
    parser.add_argument("--out", type=Path, default=HERE / "words.txt")
    args = parser.parse_args()
    words = choose(candidates(args.wordnet))
    args.out.write_text("\n".join(words) + "\n")
    print(f"wrote {len(words)} words to {args.out}")


if __name__ == "__main__":
    main()
