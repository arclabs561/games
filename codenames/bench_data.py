"""Files of the Codenames AI competition framework, and the word vectors its bots use.

Nothing fetched here is committed. Files go to codenames/.cache/bench/.

From stepmat/Codenames_GPT at commit fc28bc9 (branch ToG_2025, the code of
Stephenson et al., arXiv 2412.11373; MIT license, copyright 2024 CodenamesAICompetition), pinned by
SHA-256:

- players/cm_wordlist.txt: the 7,191 clue words of the word-vector codemasters.
- wordpools/codenames_original_wordpool.txt: the 394 board words. They are the words of
  the published game, whose rights belong to Czech Games Edition. The framework's MIT
  license covers the file, but whether it can license the list is unclear, so the list
  is used only to rebuild the published boards and is never written to a results file.
  The cross-play runs use codenames/words.txt instead.
- the published single-team results of the word-vector bots (bot_results_new_style.txt
  and move_log.txt under results/word_vector (w2v, glove, combined)/one_team/).

Word vectors (too large to download on every machine; `extract` runs on a GitHub-hosted
runner and keeps only the words that some run needs):

- GloVe 6B 300d (Pennington et al. 2014), glove.6B.zip from nlp.stanford.edu. Public
  Domain Dedication and License v1.0, as the GloVe page states.
- GoogleNews word2vec, 300 dimensions (Mikolov et al. 2013), the copy that gensim-data
  publishes as a GitHub release (word2vec-google-news-300.gz). The word2vec project page
  lists the Apache 2.0 license for the project; no separate license for the vectors is
  stated, and gensim-data lists it as not found.

The **forms** of a clue word are its WordNet lemma and its Lancaster stem as NLTK
computes them (WordNetLemmatizer().lemmatize(w) and LancasterStemmer().stem(w)), which
the framework's legality rule needs. NLTK is not a dependency of this package; `forms`
runs it through `uv run --with nltk`.

    uv run python -m codenames.bench_data fetch                   # the framework's files
    uv run --with nltk==3.9.2 python -m codenames.bench_data forms
    uv run python -m codenames.bench_data extract --glove glove.6B.zip --w2v word2vec-google-news-300.gz
"""

import argparse
import gzip
import hashlib
import importlib
import io
import json
import zipfile
from functools import cache
from pathlib import Path

import numpy as np

from codenames.bench import Vectors
from codenames.fetch import CACHE, download
from codenames.words import board_words, clue_words

BENCH = CACHE / "bench"
REPO = "stepmat/Codenames_GPT"
COMMIT = "fc28bc9fc7206f9ba2034d6f0a0b0fcf8e65dcc6"
RESULTS_DIR = "codenames/results/word_vector (w2v, glove, combined)/one_team"
FILES = {
    "cm_wordlist.txt": (
        "codenames/players/cm_wordlist.txt",
        "84df0974412b44b1a45cb4ad3bfc9995e10c8fa10151b6628e654cba65bb907a",
    ),
    "wordpool.txt": (
        "codenames/wordpools/codenames_original_wordpool.txt",
        "4723d37e49b3f0fcfe2751e79ab82ba2671e706063c82d16d55dae18ec947254",
    ),
    "published_results.jsonl": (
        f"{RESULTS_DIR}/bot_results_new_style.txt",
        "69be517b3f142dc9c8905719d882ba21953dcd0a6b4d51700568782c4f70566e",
    ),
    "published_moves.jsonl": (
        f"{RESULTS_DIR}/move_log.txt",
        "75198b7546399ca99728b51330c9c615327e8373be853a43883bbbb7d0f685db",
    ),
}
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_MEMBER = "glove.6B.300d.txt"
W2V_URL = (
    "https://github.com/RaRe-Technologies/gensim-data/releases/download/"
    "word2vec-google-news-300/word2vec-google-news-300.gz"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_file(name: str) -> Path:
    """One of the framework's files, downloaded once and checked by SHA-256."""
    path, digest = FILES[name]
    out = BENCH / name
    if not out.exists():
        url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{path.replace(' ', '%20')}"
        data = download(url)
        if sha256(data) != digest:
            raise ValueError(f"{name}: SHA-256 {sha256(data)}, expected {digest}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    return out


@cache
def cm_words() -> tuple[str, ...]:
    """The codemasters' clue list, in file order (the order their loops visit)."""
    return tuple(fetch_file("cm_wordlist.txt").read_text().splitlines())


@cache
def framework_pool() -> tuple[str, ...]:
    """The framework's 394 board words, in capitals and in file order."""
    return tuple(fetch_file("wordpool.txt").read_text().splitlines())


@cache
def published_games() -> list[dict]:
    """The 900 published single-team games of the word-vector bots: each result row with
    its move log ("moves")."""
    rows = fetch_file("published_results.jsonl").read_text().splitlines()
    moves = fetch_file("published_moves.jsonl").read_text().splitlines()
    assert len(rows) == len(moves)
    return [json.loads(r) | {"moves": json.loads(m)} for r, m in zip(rows, moves, strict=True)]


def published_transcript(moves: dict) -> list[tuple[str, int, list[str]]]:
    """(clue, count, picks) per turn from one game's move log."""
    out: list[tuple[str, int, list[str]]] = []
    for key, entry in moves.items():
        if key.startswith("clue_"):
            out.append((entry["clue"], int(entry["clue_num"]), []))
        else:
            out[-1][2].append(entry["guess_answer"])
    return out


def needed_words() -> list[str]:
    """Every word some run looks up: the framework's clue list and board words, and this
    package's board and clue words."""
    words = {*cm_words(), *(w.lower() for w in framework_pool()), *board_words(), *clue_words()}
    return sorted(words)


# Extraction from the full vector files (on a runner with the downloads).


def glove_subset(zip_path: Path, words: set[str]) -> list[str]:
    """The lines of glove.6B.300d.txt whose word is wanted, unchanged."""
    keep = []
    with zipfile.ZipFile(zip_path) as z, z.open(GLOVE_MEMBER) as fh:
        for raw in io.TextIOWrapper(fh, encoding="utf-8"):
            if raw.split(" ", 1)[0] in words:
                keep.append(raw.rstrip("\n"))
    return keep


def read_word2vec(fh, wanted: set[str] | None = None):
    """Records of a binary word2vec file: the header "count dims\\n", then per word its
    bytes up to a space and dims little-endian float32 values. Words are decoded as
    gensim does (UTF-8, ignoring errors, leading newlines dropped)."""
    count, dims = map(int, fh.readline().split())
    size = 4 * dims
    for _ in range(count):
        chars = bytearray()
        while (ch := fh.read(1)) != b" ":
            if not ch:
                return
            chars += ch
        word = bytes(chars).lstrip(b"\n").decode("utf-8", errors="ignore")
        data = fh.read(size)
        if wanted is None or word in wanted:
            yield word, np.frombuffer(data, dtype="<f4")


def w2v_subset(gz_path: Path, words: set[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with gzip.open(gz_path, "rb") as fh:
        for word, vec in read_word2vec(io.BufferedReader(fh, 1 << 20), words):
            out.setdefault(word, vec.copy())  # gensim keeps the first of a repeated word
    return out


def extract(glove_zip: Path, w2v_gz: Path, out: Path = BENCH) -> dict:
    words = set(needed_words())
    out.mkdir(parents=True, exist_ok=True)
    lines = glove_subset(glove_zip, words)
    (out / "glove300.txt").write_text("\n".join(lines) + "\n")
    w2v = w2v_subset(w2v_gz, words)
    names = sorted(w2v)
    np.savez(out / "w2v.npz", words=np.array(names), vectors=np.stack([w2v[w] for w in names]))
    info = {
        "glove_zip_sha256": file_sha256(glove_zip),
        "w2v_gz_sha256": file_sha256(w2v_gz),
        "words_wanted": len(words),
        "glove_words": len(lines),
        "w2v_words": len(names),
        "glove300_sha256": file_sha256(out / "glove300.txt"),
        "w2v_npz_sha256": file_sha256(out / "w2v.npz"),
    }
    (out / "vectors.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def parse_glove(lines: list[str]) -> Vectors:
    """As the framework's Game.load_glove_vecs: float() of each field, so float64."""
    words, rows = [], []
    for line in lines:
        parts = line.rstrip().split(" ")
        words.append(parts[0])
        rows.append(np.array([float(n) for n in parts[1:]]))
    return Vectors(words, np.stack(rows))


@cache
def tables(folder: Path = BENCH) -> dict[str, Vectors]:
    """The extracted GloVe and word2vec vectors, as Vectors."""
    glove = parse_glove((folder / "glove300.txt").read_text().splitlines())
    z = np.load(folder / "w2v.npz")
    w2v = Vectors([str(w) for w in z["words"]], z["vectors"].astype(np.float32))
    return {"glove": glove, "w2v": w2v}


# The clue words' forms, with NLTK.


def compute_forms(words: list[str], data_dir: Path = BENCH / "nltk_data") -> dict:
    nltk = importlib.import_module("nltk")
    nltk.download("wordnet", download_dir=str(data_dir), quiet=True)
    nltk.data.path.insert(0, str(data_dir))
    stem = importlib.import_module("nltk.stem")
    lemmatizer, lancaster = stem.WordNetLemmatizer(), stem.LancasterStemmer()
    forms = {w: [lemmatizer.lemmatize(w), lancaster.stem(w)] for w in words}
    return {"nltk": nltk.__version__, "forms": forms}


@cache
def forms(path: Path = BENCH / "forms.json") -> dict[str, tuple[str, str]]:
    data = json.loads(path.read_text())["forms"]
    return {w: (f[0], f[1]) for w, f in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="the framework's files")
    sub.add_parser("forms", help="lemma and stem of each clue word (needs nltk)")
    ex = sub.add_parser("extract", help="the vector subsets, from the full downloads")
    ex.add_argument("--glove", type=Path, required=True, help="glove.6B.zip")
    ex.add_argument("--w2v", type=Path, required=True, help="word2vec-google-news-300.gz")
    args = parser.parse_args()
    if args.cmd == "fetch":
        for name in FILES:
            print(name, fetch_file(name).stat().st_size, "bytes")
    elif args.cmd == "forms":
        out = compute_forms(list(cm_words()))
        (BENCH / "forms.json").write_text(json.dumps(out, sort_keys=True) + "\n")
        print("forms of", len(out["forms"]), "words with nltk", out["nltk"])
    else:
        print(json.dumps(extract(args.glove, args.w2v), indent=1))


if __name__ == "__main__":
    main()
