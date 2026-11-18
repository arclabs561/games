"""Download the models and data this package uses, each pinned by SHA-256.

Nothing here is committed to the repository. Files go to codenames/.cache/.

    uv run python -m codenames.fetch all          # everything that can be fetched
    uv run python -m codenames.fetch potion eat   # some of it

Sources and licenses (see the README for how each was checked):

- wordnet: Princeton WordNet 3.0, as packaged by NLTK on GitHub. WordNet 3.0 license.
- potion: potion-base-8M (MinishLab, MIT), repacked to int8 by the npm package
  @jigyasudham/veto-model 1.0.1 (MIT).
- wordllama: the l2_supercat 256-dimension table shipped inside the wordllama 0.4.0.post1
  wheel on PyPI (MIT).
- granite: IBM granite-embedding-small-english-r2 (Apache-2.0), the 8-bit ONNX export by
  onnx-community, unchanged, in the npm package isorite-model-granite 1.0.0.
- bge: BAAI bge-small-en-v1.5 (MIT), int8 ONNX, in the npm package
  impro-plugin-semantic-mute-model 1.0.1 (MIT).
- cultural: the guessing task of the Cultural Codes dataset (Shaikh et al. 2023,
  github.com/SALT-NLP/codenames): human Codenames Duet guesses. The repository has no
  license file; the paper says the authors plan to release it under CC BY-SA 4.0.
- eat: the Edinburgh Associative Thesaurus as JSON (github.com/dariusk/ea-thesaurus), which
  states the data are under the UK Open Government Licence.

The Small World of Words norms (CC BY-NC-ND 3.0) and the USF norms (copyright of their
authors, no license given) cannot be fetched by a script: download them by hand from the
pages that `uv run python -m codenames.fetch swow usf` prints, into codenames/.cache/.
"""

import argparse
import hashlib
import io
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

CACHE = Path(__file__).parent / ".cache"


@dataclass(frozen=True)
class Source:
    url: str
    sha256: str
    members: dict[str, str]  # path inside the archive -> file name under CACHE/<name>/
    kind: str  # "zip", "tgz", "file", or "files" (each member fetched from url + path)


SOURCES = {
    "wordnet": Source(
        "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip",
        "cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59",
        {
            f"wordnet/{f}": f
            for f in (
                "LICENSE",
                "lexnames",
                "data.noun",
                "index.noun",
                "index.verb",
                "index.adj",
                "index.adv",
            )
        },
        "zip",
    ),
    "potion": Source(
        "https://registry.npmjs.org/@jigyasudham/veto-model/-/veto-model-1.0.1.tgz",
        "015fc9cb2b645a255393bfcbb862e081489358408ab4e6007d6114c791efd436",
        {
            "package/model/embeddings.int8.bin": "embeddings.int8.bin",
            "package/model/scales.f32.bin": "scales.f32.bin",
            "package/model/tokenizer.json": "tokenizer.json",
            "package/model/golden.json": "golden.json",
            "package/LICENSE.potion-base-8M": "LICENSE",
        },
        "tgz",
    ),
    "wordllama": Source(
        "https://files.pythonhosted.org/packages/f4/01/b7f3d70ad44c1c7d35914f0db93893882ce0f5e2cb7341215dc89a153cfa/"
        "wordllama-0.4.0.post1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        "2dc004ed04438e4a9ce263ef69b1d7dcc5099da7277e08ddedfd35810428ba31",
        {
            "wordllama/weights/l2_supercat_256.safetensors": "l2_supercat_256.safetensors",
            "wordllama/tokenizers/l2_supercat_tokenizer_config.json": "tokenizer.json",
        },
        "zip",
    ),
    "granite": Source(
        "https://registry.npmjs.org/isorite-model-granite/-/isorite-model-granite-1.0.0.tgz",
        "8d8ab528ae289dfc81fd2bfa80c867ca0c8342a2d98f7c86141df4b15fdf028c",
        {
            "package/model.onnx": "model.onnx",
            "package/model_quantized.onnx_data": "model_quantized.onnx_data",
            "package/tokenizer.json": "tokenizer.json",
            "package/LICENSE": "LICENSE",
        },
        "tgz",
    ),
    "bge": Source(
        "https://registry.npmjs.org/impro-plugin-semantic-mute-model/-/impro-plugin-semantic-mute-model-1.0.1.tgz",
        "096deebc38e548cfbcd3fbf6992e2b9be8e4044ee27655bc34dc749387aeea6c",
        {
            "package/bge-small-en-v1.5-q8.onnx": "model.onnx",
            "package/bge-small-en-v1.5-tokenizer.json": "tokenizer.json",
        },
        "tgz",
    ),
    "cultural": Source(
        "https://raw.githubusercontent.com/SALT-NLP/codenames/9bf4550e681f7a42ac406439b00b0c717f59f13c/data/",
        "",
        {
            "generate_guess_task/train.csv": "guess_train.csv",
            "generate_guess_task/val.csv": "guess_val.csv",
            "generate_guess_task/test.csv": "guess_test.csv",
        },
        "files",
    ),
    "eat": Source(
        "https://raw.githubusercontent.com/dariusk/ea-thesaurus/36c4bb066ba270a2b207c1938b24be3b7348722b/ea-thesaurus.json",
        "ad1d948d0547724a7a3687c1d1fa6e681d903110dc5f8cccfb2825258f78653e",
        {"": "ea-thesaurus.json"},
        "file",
    ),
}

# Hashes of the files fetched one by one.
FILE_SHA256 = {
    "guess_train.csv": "a13768d8de0427100337d96a1ccf046b2e095f03a777d80d033a8cfcd900c70b",
    "guess_val.csv": "13572d274f615e2b0717b53eeb58a0817d9ec7acc4aa791f4d559e73810caf8e",
    "guess_test.csv": "dc2528641dcbab97e29ec22078bd3a8a74d39e83097cadc638a6ce4d87797d2e",
}

MANUAL = {
    "swow": "Small World of Words, SWOW-EN18 'strength' file, CC BY-NC-ND 3.0, from "
    "https://smallworldofwords.org/en/project/research . Save the file named "
    "strength.SWOW-EN.R123.*.csv as codenames/.cache/swow/strength.csv",
    "usf": "USF free association norms, Appendix A, from "
    "http://w3.usf.edu/FreeAssociation/AppendixA/index.html . Concatenate the eight "
    "'Cue Target Pairs' files into codenames/.cache/usf/appendix_a.csv",
}


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310 (pinned https URLs)
        return r.read()


def fetch(name: str) -> Path:
    """Download one source, check its hash, and unpack the files we need. Returns its folder."""
    src = SOURCES[name]
    out = CACHE / name
    if all((out / f).exists() for f in src.members.values()):
        return out
    out.mkdir(parents=True, exist_ok=True)
    if src.kind == "files":
        for member, filename in src.members.items():
            blob = download(src.url + member)
            digest = hashlib.sha256(blob).hexdigest()
            if digest != FILE_SHA256[filename]:
                raise ValueError(f"{name}/{member}: sha256 {digest} is not the pinned one")
            (out / filename).write_bytes(blob)
        return out
    data = download(src.url)
    digest = hashlib.sha256(data).hexdigest()
    if digest != src.sha256:
        raise ValueError(f"{name}: sha256 {digest} does not match the pinned {src.sha256}")
    for member, filename in src.members.items():
        if src.kind == "file":
            blob = data
        elif src.kind == "zip":
            blob = zipfile.ZipFile(io.BytesIO(data)).read(member)
        else:
            tar = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
            f = tar.extractfile(member)
            assert f is not None, member
            blob = f.read()
        (out / filename).write_bytes(blob)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="+", choices=[*SOURCES, *MANUAL, "all"])
    args = parser.parse_args()
    names = list(SOURCES) if "all" in args.names else args.names
    for name in names:
        if name in MANUAL:
            print(f"{name}: download by hand. {MANUAL[name]}")
        else:
            print(f"{name}: {fetch(name)}")


if __name__ == "__main__":
    main()
