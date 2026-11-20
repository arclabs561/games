"""Which clues the rules forbid on a board: a stricter check than `words.legal`.

The rulebook (Czech Games Edition, English rules of July 2015, page 8) says:

> You can't say any form of a visible word on the table. Until BREAK is covered up by a
> card, you can't say break, broken, breakage, or breakdown.
>
> You can't say part of a compound word on the table. Until HORSESHOE is covered up,
> you can't say horse, shoe, unhorsed, or snowshoe.

and, under "Don't be too strict", that island and even land are valid clues for ENGLAND,
and that sparrow is fine when ROW is on the table. Theatre and theater "are different
forms of the same word".

`words.legal`, the rule every experiment here has used, forbids a board word and any
clue that contains one or is contained in one. That stops horse, shoe and unhorsed for
HORSESHOE, but not broken or breakage for BREAK, snowshoe for HORSESHOE, or theatre for
THEATER. It also forbids land for ENGLAND and sparrow for ROW, which the rulebook allows.

This module names the relation between a clue and a board word, strongest first:

| Relation | Meaning | Level |
|---|---|---|
| same | the clue is the word | illegal |
| substring | one contains the other (`words.legal`) | illegal |
| inflection | the two share a base form under WordNet's morphology (owned, own) | illegal |
| derivation | base forms linked by a WordNet derivational pointer, directly or through one shared neighbor (teaching, teacher), or by a person suffix that WordNet does not link (librarian, library; actress, actor); the two must start with the same three letters, which leaves out pertainyms such as lunar for MOON | illegal |
| variant | the same word in British and American spelling (theatre, theater) | illegal |
| compound | a form of one, with or without a prefix such as un-, is a part of the other (rails, railroad; unhorsed, horseshoe), or both are compounds with a part of four or more letters in common (snowshoe, horseshoe; roadway, railroad) | illegal |
| stem | the same Snowball (Porter2) stem, and none of the above | borderline |
| affix | the same stem of four or more letters after one derivational suffix is taken off each (countess, county) | borderline |
| compound3 | compounds with a common part of three letters (highway, railway; webcam, camera) | borderline |
| prefix | a common start of four or more letters that is over half of the shorter word (telecom, telephone) | borderline |

A clue is `strict_legal` on a board when no board word has an illegal or borderline
relation to it. The levels were checked on the rulebook's own examples and on a
hand-labeled sample of flagged pairs (`results/diagnostics/legality.md`).

Resources: Princeton WordNet 3.0 (the WordNet 3.0 license, which allows use, copying and
redistribution with its notice), from the NLTK data package on GitHub, pinned by
SHA-256; and the Snowball English stemmer (`snowballstemmer`, BSD-3-Clause).

    uv run python -m codenames.legality residency residence railway railroad
"""

import argparse
from collections import defaultdict
from functools import cache

from codenames import fetch as fetching
from codenames.words import legal

ILLEGAL = ("same", "substring", "inflection", "derivation", "variant", "compound")
BORDERLINE = ("stem", "affix", "compound3", "prefix")
RELATIONS = ILLEGAL + BORDERLINE
POS = ("noun", "verb", "adj", "adv")
MORPHOLOGY = fetching.Source(
    "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip",
    "cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59",
    {
        f"wordnet/{f}": f
        for f in ("LICENSE", *(f"{k}.{p}" for k in ("data", "index") for p in POS))
        + tuple(f"{p}.exc" for p in POS)
    },
    "zip",
)
# WordNet's detachment rules (the morphy function of WordNet 3.0), by part of speech.
SUFFIXES = {
    "noun": (
        ("s", ""),
        ("ses", "s"),
        ("xes", "x"),
        ("zes", "z"),
        ("ches", "ch"),
        ("shes", "sh"),
        ("men", "man"),
        ("ies", "y"),
    ),
    "verb": (
        ("s", ""),
        ("ies", "y"),
        ("es", "e"),
        ("es", ""),
        ("ed", "e"),
        ("ed", ""),
        ("ing", "e"),
        ("ing", ""),
    ),
    "adj": (("er", ""), ("est", ""), ("er", "e"), ("est", "e")),
    "adv": (),
}
# Lexical pointers that join two forms of one word: derivationally related form,
# pertainym or "derived from adjective", and participle of a verb.
FORM_POINTERS = ("+", "\\", "<")
# British spellings and their American forms, applied to word endings and doubled l.
SPELLINGS = (
    ("tre", "ter"),
    ("tres", "ters"),
    ("our", "or"),
    ("ours", "ors"),
    ("ise", "ize"),
    ("ised", "ized"),
    ("ising", "izing"),
    ("isation", "ization"),
    ("yse", "yze"),
    ("ogue", "og"),
    ("ence", "ense"),
)
MIN_PART = 3
# Two-letter words allowed as the second part of a compound (makeup, makeover).
PARTICLES = frozenset({"up", "on", "in", "by"})
# Word endings that WordNet also lists as words (ion, age); not counted as compound parts.
NOT_PARTS = frozenset(
    "ion age ate ant ent ism ist ive ous ess est ful ing let ship ness ment hood dom less "
    "able ible ance ence ure ery ory ary ian ics ity".split()
)
MIN_DERIVATION_PREFIX = 3
# Prefixes taken off a clue before it is compared with the parts of a compound (unhorsed).
PREFIXES = ("un", "re", "dis", "non", "mis", "over", "under", "out", "sub", "anti")
MIN_PREFIX = 4
# Person suffixes and the forms they come from: librarian from library, actress from actor.
PERSON = (("ian", "y"), ("ian", ""), ("ress", "or"), ("ress", "er"), ("ess", ""), ("ist", "y"))
MIN_STEM = 4
# Derivational suffixes taken off by `affix_stems`, longest first within each group.
DERIVATIONAL = (
    "ization",
    "ation",
    "ition",
    "tion",
    "sion",
    "ment",
    "ness",
    "ship",
    "hood",
    "ancy",
    "ency",
    "ance",
    "ence",
    "ity",
    "ism",
    "ist",
    "ian",
    "ress",
    "ess",
    "ery",
    "ory",
    "ary",
    "ive",
    "ous",
    "ful",
    "ing",
    "ed",
    "er",
    "or",
    "al",
    "ic",
    "ly",
    "y",
    "e",
)


def morphology_dir():
    """WordNet's data, index and exception files, fetched once into codenames/.cache/."""
    fetching.SOURCES.setdefault("wordnet-morphology", MORPHOLOGY)
    return fetching.fetch("wordnet-morphology")


@cache
def lemmas() -> dict[str, frozenset[str]]:
    """{part of speech: the one-word lemmas of WordNet}."""
    d = morphology_dir()
    out = {}
    for pos in POS:
        lines = (d / f"index.{pos}").read_text(encoding="latin-1").splitlines()
        out[pos] = frozenset(x.split()[0] for x in lines if not x.startswith(" ") and x)
    return out


@cache
def exceptions() -> dict[str, dict[str, tuple[str, ...]]]:
    """{part of speech: {inflected form: base forms}} from WordNet's .exc files."""
    d = morphology_dir()
    out = {}
    for pos in POS:
        table = {}
        for line in (d / f"{pos}.exc").read_text(encoding="latin-1").splitlines():
            form, *bases = line.split()
            table[form] = tuple(bases)
        out[pos] = table
    return out


@cache
def bases(word: str) -> frozenset[str]:
    """The word and its WordNet base forms in every part of speech (morphy)."""
    out = {word}
    for pos in POS:
        known = lemmas()[pos]
        out.update(b for b in exceptions()[pos].get(word, ()) if b in known)
        for end, put in SUFFIXES[pos]:
            if word.endswith(end) and len(word) > len(end):
                b = word[: len(word) - len(end)] + put
                if b in known:
                    out.add(b)
    return frozenset(out)


@cache
def synsets() -> dict[tuple[str, str], tuple]:
    """{(offset, pos letter): the synset's lemmas in WordNet's order}, with None for a
    multiword lemma so that the word numbers of pointers line up."""
    return _read_data()[0]


@cache
def form_links() -> dict[str, frozenset[str]]:
    """{lemma: lemmas joined to it by a form pointer}, symmetric."""
    return _read_data()[1]


def _read_data():
    d = morphology_dir()
    words: dict[tuple[str, str], tuple] = {}
    raw: list[tuple[tuple[str, str], int, str, tuple[str, str], int]] = []
    for pos in POS:
        for line in (d / f"data.{pos}").read_text(encoding="latin-1").splitlines():
            if line.startswith(" "):
                continue
            key, members, ptrs = read_synset(line)
            words[key] = members
            for symbol, target, source_n, target_n in ptrs:
                if symbol in FORM_POINTERS and source_n:
                    raw.append((key, source_n, symbol, target, target_n))
    links: dict[str, set[str]] = defaultdict(set)
    for key, source_n, _, target, target_n in raw:
        a = words[key][source_n - 1]
        b = words[target][target_n - 1]
        if a and b and a != b:
            links[a].add(b)
            links[b].add(a)
    return words, {w: frozenset(v) for w, v in links.items()}


def read_synset(line: str) -> tuple[tuple[str, str], tuple, list[tuple[str, tuple, int, int]]]:
    """One line of a data file: its key, its lemmas (lowercase, None for a multiword
    lemma) and its pointers as (symbol, target key, source word number, target number)."""
    f = line.split(" | ")[0].split()
    offset, ss_type = f[0], f[2]
    pos = "a" if ss_type == "s" else ss_type
    n = int(f[3], 16)
    members = []
    for i in range(n):
        w = f[4 + 2 * i].lower()
        w = w.split("(")[0]  # adjective markers such as (a) and (p)
        members.append(None if "_" in w or "-" in w else w)
    j = 4 + 2 * n
    p = int(f[j])
    ptrs = []
    for k in range(p):
        symbol, target, tpos, st = f[j + 1 + 4 * k : j + 5 + 4 * k]
        tpos = "a" if tpos == "s" else tpos
        ptrs.append((symbol, (target, tpos), int(st[:2], 16), int(st[2:], 16)))
    return (offset, pos), tuple(members), ptrs


@cache
def person_forms(word: str) -> frozenset[str]:
    """The WordNet words a person suffix of PERSON could come from."""
    out = set()
    for end, put in PERSON:
        if word.endswith(end) and len(word) - len(end) >= MIN_PART:
            w = word[: len(word) - len(end)] + put
            if is_word(w):
                out.add(w)
    return frozenset(out)


@cache
def family(word: str) -> frozenset[str]:
    """The base forms of a word (with the forms of person_forms) and every lemma one
    form pointer away from them."""
    b = bases(word) | person_forms(word)
    links = form_links()
    return b.union(*(links.get(x, frozenset()) for x in b))


@cache
def is_word(w: str) -> bool:
    return any(w in lemmas()[p] for p in POS)


def is_part(w: str, second: bool) -> bool:
    if w in NOT_PARTS:
        return False
    return (len(w) >= MIN_PART and is_word(w)) or (second and w in PARTICLES)


@cache
def splits(word: str) -> frozenset[tuple[str, str]]:
    """The ways to write a word as two WordNet words of MIN_PART or more letters (the
    second may be a particle of PARTICLES), leaving out suffix-like parts (NOT_PARTS)."""
    return frozenset(
        (word[:i], word[i:])
        for i in range(MIN_PART, len(word) - 1)
        if is_part(word[:i], False) and is_part(word[i:], True)
    )


@cache
def parts(word: str) -> frozenset[str]:
    return frozenset(p for x in splits(word) for p in x if len(p) >= MIN_PART)


def shared_part(a: str, b: str, same_place: bool = True) -> str | None:
    """The longest part that a split of a and a split of b have in common: in the same
    place (first or second), or in any place."""
    best = None
    for x in splits(a):
        for y in splits(b):
            for i in (0, 1):
                for j in (i,) if same_place else (0, 1):
                    if x[i] != y[j] or len(x[i]) < MIN_PART:
                        continue
                    if best is None or len(x[i]) > len(best):
                        best = x[i]
    return best


@cache
def affix_stems(word: str) -> frozenset[str]:
    """The word less one derivational suffix, then less a final e, i or y, for every
    suffix it ends in, keeping stems of MIN_STEM or more letters."""
    out = set()
    for end in DERIVATIONAL:
        if word.endswith(end):
            rest = word[: len(word) - len(end)].rstrip("eiy")
            if len(rest) >= MIN_STEM:
                out.add(rest)
    return frozenset(out)


@cache
def unprefixed(word: str) -> frozenset[str]:
    """The base forms of the word, and of the word less one prefix of PREFIXES."""
    out = set(bases(word))
    for pre in PREFIXES:
        if word.startswith(pre) and len(word) - len(pre) >= MIN_PART:
            out |= bases(word[len(pre) :])
    return frozenset(out)


@cache
def american(word: str) -> str:
    """The word with British endings and a doubled l before -ed, -ing, -er made American."""
    for brit, us in SPELLINGS:
        if word.endswith(brit):
            word = word[: len(word) - len(brit)] + us
            break
    for end in ("lled", "lling", "ller"):
        if word.endswith(end):
            word = word[: len(word) - len(end)] + end[1:]
    return word


@cache
def stem(word: str) -> str:
    import snowballstemmer

    return snowballstemmer.stemmer("english").stemWord(word)


def common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def relation(clue: str, word: str) -> str | None:
    """The strongest relation of RELATIONS between a clue and one board word, or None."""
    clue, word = clue.lower(), word.lower()
    if clue == word:
        return "same"
    if clue in word or word in clue:
        return "substring"
    if bases(clue) & bases(word):
        return "inflection"
    if family(clue) & family(word) and common_prefix(clue, word) >= MIN_DERIVATION_PREFIX:
        return "derivation"
    if american(clue) == american(word):
        return "variant"
    if unprefixed(clue) & parts(word) or unprefixed(word) & parts(clue):
        return "compound"
    part = shared_part(clue, word, same_place=False)
    if part is not None and len(part) >= 4:
        return "compound"
    if stem(clue) == stem(word):
        return "stem"
    if affix_stems(clue) & affix_stems(word):
        return "affix"
    if part is not None:
        return "compound3"
    n = common_prefix(clue, word)
    if n >= MIN_PREFIX and n > min(len(clue), len(word)) / 2:
        return "prefix"
    return None


def level(rel: str | None) -> str:
    """'illegal', 'borderline' or 'legal'."""
    if rel is None:
        return "legal"
    return "illegal" if rel in ILLEGAL else "borderline"


def flags(clue: str, words) -> list[tuple[str, str]]:
    """(board word, relation) for every board word the clue is related to."""
    out = []
    for w in words:
        r = relation(clue, w)
        if r is not None:
            out.append((w, r))
    return out


def worst(clue: str, words) -> str:
    """The clue's level on the board: its worst relation to any board word."""
    levels = {level(r) for _, r in flags(clue, words)}
    return next((x for x in ("illegal", "borderline") if x in levels), "legal")


def strict_legal(clue: str, words) -> bool:
    """Legal by words.legal and with no illegal or borderline relation to a board word."""
    return legal(clue, tuple(words)) and worst(clue, words) == "legal"


# The rulebook's own examples: (clue, board word, legal by the rules).
RULEBOOK = (
    ("break", "break", False),
    ("broken", "break", False),
    ("breakage", "break", False),
    ("breakdown", "break", False),
    ("horse", "horseshoe", False),
    ("shoe", "horseshoe", False),
    ("unhorsed", "horseshoe", False),
    ("snowshoe", "horseshoe", False),
    ("theatre", "theater", False),
    ("island", "england", True),
    ("land", "england", True),
    ("sparrow", "row", True),
    ("tree", "nut", True),
    ("cashew", "nut", True),
    ("strudel", "apple", True),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pairs", nargs="+", help="clue word clue word ...")
    args = parser.parse_args()
    assert len(args.pairs) % 2 == 0, "give clue and board word in pairs"
    for clue, word in zip(args.pairs[::2], args.pairs[1::2], strict=True):
        r = relation(clue, word)
        print(f"{clue:>14} {word:<14} {r or '-':<11} {level(r)}")


if __name__ == "__main__":
    main()
