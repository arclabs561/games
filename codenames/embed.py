"""Word vectors from several embedding models, cached on disk.

An **embedder** maps a list of strings to unit vectors, one row per string. A
**template** turns a word into the text the model reads, such as "{}" (the bare word)
or "a word related to {}". The similarity of two words is the dot product of their
unit vectors.

Models that run on this machine (fetched by codenames.fetch):

    potion     potion-base-8M, a static table distilled from bge-base-en-v1.5 (256 dims)
    wordllama  wordllama l2_supercat, a static table trained from LLM token embeddings (256)
    granite    granite-embedding-small-english-r2, a 47M-parameter ModernBERT encoder (384)
    bge        bge-small-en-v1.5, a 33M-parameter BERT encoder (384)

Any Hugging Face model that sentence-transformers can load is also accepted by its id,
on a machine that can reach the Hugging Face Hub:

    uv run --with sentence-transformers python -m codenames.embed --model Qwen/Qwen3-Embedding-0.6B
    uv run python -m codenames.embed --model granite --template related

Vectors are cached in codenames/.cache/vectors/<model>__<template>.npz.
"""

import argparse
import importlib
import json
import struct
import time
from collections.abc import Callable
from functools import cache
from pathlib import Path

import numpy as np

from codenames.fetch import CACHE, fetch

VECTORS = CACHE / "vectors"
# Vectors computed elsewhere (by .github/workflows on the compute branch) are read from
# here first, then from the cache.
COMPUTED = Path(__file__).parent / "vectors"
Encoder = Callable[[list[str]], np.ndarray]

# Templates. "query" uses the instruction format of Qwen3-Embedding and harrier-oss-v1;
# "bge_query" is the query prefix from the bge-small-en-v1.5 model card.
TEMPLATES = {
    "bare": "{}",
    "word": "the word '{}'",
    "related": "a word related to {}",
    "query": (
        "Instruct: Given a clue in a word-association game, retrieve words that people "
        "associate with it\nQuery: {}"
    ),
    "bge_query": "Represent this sentence for searching relevant passages: {}",
}
LOCAL_MODELS = ("potion", "wordllama", "granite", "bge")


def normalize(x: np.ndarray) -> np.ndarray:
    """Scale each row to unit length; a zero row stays zero."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


def static_encoder(table: np.ndarray, tokenizer_file: Path) -> Encoder:
    """Mean of the token rows of a static table, as model2vec and wordllama do it."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_file))

    def encode(texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), table.shape[1]), np.float32)
        for i, enc in enumerate(tok.encode_batch(texts, add_special_tokens=False)):
            if enc.ids:
                out[i] = table[enc.ids].mean(axis=0)
        return normalize(out)

    return encode


def potion() -> Encoder:
    """potion-base-8M from the int8 repack: row = int8 row times its scale."""
    d = fetch("potion")
    scales = np.fromfile(d / "scales.f32.bin", dtype="<f4")
    table = np.fromfile(d / "embeddings.int8.bin", dtype=np.int8).reshape(len(scales), -1)
    return static_encoder(table.astype(np.float32) * scales[:, None], d / "tokenizer.json")


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Read a safetensors file with numpy alone (8-byte header length, JSON header, data)."""
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + n])
    dtypes = {"F16": np.float16, "F32": np.float32}
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        arr = np.frombuffer(raw[8 + n + a : 8 + n + b], dtype=dtypes[info["dtype"]])
        out[name] = arr.reshape(info["shape"])
    return out


def wordllama() -> Encoder:
    d = fetch("wordllama")
    table = read_safetensors(d / "l2_supercat_256.safetensors")["embedding.weight"]
    return static_encoder(table.astype(np.float32), d / "tokenizer.json")


def onnx_encoder(name: str, pool: str) -> Encoder:
    """A transformer encoder in ONNX, pooled by its [CLS] token, in batches of 256."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    d = fetch(name)
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    tok.no_truncation()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(d / "model.onnx"), opts, providers=["CPUExecutionProvider"])
    inputs = {i.name for i in sess.get_inputs()}

    def encode(texts: list[str]) -> np.ndarray:
        rows = []
        for start in range(0, len(texts), 256):
            encs = tok.encode_batch(texts[start : start + 256])
            width = max(len(e.ids) for e in encs)
            ids = np.zeros((len(encs), width), np.int64)
            mask = np.zeros_like(ids)
            for i, e in enumerate(encs):
                ids[i, : len(e.ids)] = e.ids
                mask[i, : len(e.ids)] = 1
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            if pool == "sentence_embedding":
                out = np.asarray(sess.run(["sentence_embedding"], feed)[0])
            else:
                out = np.asarray(sess.run(["last_hidden_state"], feed)[0])[:, 0]
            rows.append(out)
        return normalize(np.concatenate(rows))

    return encode


def hf_encoder(model_id: str) -> Encoder:
    """Any sentence-transformers model, for a machine that can reach the Hugging Face Hub."""
    st = importlib.import_module("sentence_transformers")
    model = st.SentenceTransformer(
        model_id, device="cpu", model_kwargs={"torch_dtype": "float32"}, trust_remote_code=False
    )

    def encode(texts: list[str]) -> np.ndarray:
        return normalize(np.asarray(model.encode(texts, batch_size=128, prompt="")))

    return encode


@cache
def encoder(model: str) -> Encoder:
    if model == "potion":
        return potion()
    if model == "wordllama":
        return wordllama()
    if model == "granite":
        return onnx_encoder("granite", "sentence_embedding")
    if model == "bge":
        return onnx_encoder("bge", "cls")
    return hf_encoder(model)


def cache_path(model: str, template: str) -> Path:
    name = f"{model.replace('/', '--')}__{template}.npz"
    return COMPUTED / name if (COMPUTED / name).exists() else VECTORS / name


def vectors(model: str, words: list[str], template: str = "bare") -> np.ndarray:
    """Unit vectors for words, computed once per (model, template, word) and cached.

    Vectors are stored as float16, and returned as float32 after that rounding, so a
    result does not depend on whether it came from the cache.
    """
    path = cache_path(model, template)
    cached: list[str] = []
    table = np.zeros((0, 0), np.float16)
    if path.exists():
        z = np.load(path)
        cached, table = list(z["words"]), z["vectors"]
    pos = {w: i for i, w in enumerate(cached)}
    missing = sorted({w for w in words if w not in pos})
    if missing:
        texts = [TEMPLATES[template].format(w) for w in missing]
        new = encoder(model)(texts).astype(np.float16)
        table = new if not cached else np.concatenate([table, new])
        cached += missing
        pos = {w: i for i, w in enumerate(cached)}
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, words=np.array(cached), vectors=table)
    return table[[pos[w] for w in words]].astype(np.float32)


def main() -> None:
    from codenames.evaluate import all_words

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help=f"one of {LOCAL_MODELS} or a HF id")
    parser.add_argument("--template", nargs="+", default=["bare"], choices=TEMPLATES)
    parser.add_argument("--out-dir", type=Path, default=None, help="write the .npz here")
    args = parser.parse_args()
    global VECTORS, COMPUTED
    if args.out_dir:
        VECTORS = COMPUTED = args.out_dir
    words = all_words()
    for template in args.template:
        t = time.perf_counter()
        v = vectors(args.model, words, template)
        print(f"{args.model} {template}: {v.shape} in {time.perf_counter() - t:.1f} s", flush=True)


if __name__ == "__main__":
    main()
