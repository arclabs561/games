"""LoRA fine-tuning of a Hugging Face embedding model on human Codenames guesses.

It needs the Hugging Face Hub and a GPU (CUDA) or an Apple-silicon Mac (MPS), so it runs
as a Hugging Face Job through experiments/hfrun.py, with the dependency group
codenames-gpu (experiments/plans/C2.md):

    python -m codenames.lora --scheme game --sources cultural --lr 1e-4 --seed 1 \\
        --evaluate test --out-dir OUT

Each training item is (clue with the "query" template, the word a person connected to
it, and seven hard negatives: other words from the same board, or random board words
for EAT items). The loss is sentence-transformers' MultipleNegativesRankingLoss, which
is InfoNCE over the hard negatives and the other items in the batch. Only LoRA adapters
(rank 16 on the attention projections) are trained.

The job encodes the whole vocabulary twice, before and after training, in the same
precision, and scores both with codenames.c2 (`--evaluate val` or `test`). It writes to
--out-dir: scores.json, the vectors as <name>__<template>.npz, the adapter, the training
log, timings.json (seconds since the container started, for each phase) and
summary.json, a small copy of scores.json for the machine that writes the README.
"""

import argparse
import hashlib
import importlib
import json
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np

from codenames.c2 import full_scores, guesses, turn_digest
from codenames.embed import TEMPLATES, normalize
from codenames.evaluate import all_words
from codenames.norms import eat_counts, held_out
from codenames.players import Space
from codenames.words import board_words

N_NEG = 7
SOURCES = ("cultural", "cultural+eat")
SCHEMES = ("published", "game")
CLUE_TEMPLATES = ("bare", "related", "query")


def rows(seed: int = 0, sources: str = "cultural+eat", scheme: str = "published"):
    """Training rows: anchor, positive, negative_1 .. negative_7, from train data only."""
    rng = random.Random(seed)
    q = TEMPLATES["query"]
    out = []
    for g in guesses("train", scheme):
        others = [w for w in g.words if w != g.picked]
        if len(others) >= N_NEG:
            negs = rng.sample(others, N_NEG)
            out.append({"anchor": q.format(g.clue), "positive": g.picked} | neg(negs))
    if sources == "cultural+eat":
        out += eat_rows(rng)
    rng.shuffle(out)
    return out


def eat_rows(rng: random.Random) -> list[dict[str, str]]:
    """EAT cues not held out, each response given by at least 5 people, against board
    words that are not responses to the cue."""
    q = TEMPLATES["query"]
    board = list(board_words())
    out = []
    for cue, row in sorted(eat_counts().items()):
        if held_out(cue):
            continue
        pool = [w for w in board if w not in row and w != cue]
        for resp, n in sorted(row.items()):
            if resp != cue and n >= 5:
                out.append(
                    {"anchor": q.format(cue), "positive": resp} | neg(rng.sample(pool, N_NEG))
                )
    return out


def neg(words: list[str]) -> dict[str, str]:
    return {f"negative_{i + 1}": w for i, w in enumerate(words)}


def container_start() -> float | None:
    """When process 1 started, as a Unix time (Linux only): the start of the job's
    container, so that setup before this module ran can be timed."""
    try:
        btime = next(
            float(line.split()[1])
            for line in Path("/proc/stat").read_text().splitlines()
            if line.startswith("btime")
        )
        ticks = float(Path("/proc/1/stat").read_text().rsplit(")", 1)[1].split()[19])
    except (OSError, StopIteration, IndexError, ValueError):
        return None
    return btime + ticks / os.sysconf("SC_CLK_TCK")


class Clock:
    """Wall-clock marks for each phase of the job."""

    def __init__(self) -> None:
        self.marks: dict[str, float] = {"module_start": time.time()}

    def mark(self, name: str) -> None:
        self.marks[name] = time.time()
        print(f"[{name}] {self.marks[name] - self.marks['module_start']:.1f} s", flush=True)

    def summary(self) -> dict:
        start = container_start()
        out: dict = {"container_start": start, "marks": self.marks}
        if start is not None:
            out["since_container_start"] = {k: round(v - start, 1) for k, v in self.marks.items()}
        return out


def encode(model, words: list[str]) -> dict[str, np.ndarray]:
    """Unit vectors for every word under each template, rounded to float16 as embed.py
    stores them."""
    out = {}
    for template in sorted({*CLUE_TEMPLATES, "bare"}):
        texts = [TEMPLATES[template].format(w) for w in words]
        vecs = normalize(np.asarray(model.encode(texts, batch_size=256, prompt="")))
        out[template] = vecs.astype(np.float16)
    return out


def spaces(name: str, vecs: dict[str, np.ndarray], words: list[str]) -> dict[str, Space]:
    """One space per clue-side template, with bare word vectors."""
    word = vecs["bare"].astype(np.float32)
    return {
        t: Space(f"{name}:{t}/bare", vecs[t].astype(np.float32), word, words)
        for t in CLUE_TEMPLATES
    }


def save_vectors(out_dir: Path, name: str, vecs: dict[str, np.ndarray], words: list[str]):
    for template, v in vecs.items():
        np.savez_compressed(out_dir / f"{name}__{template}.npz", words=np.array(words), vectors=v)


def summary(result: dict) -> dict:
    """scores.json without the per-turn lists: hits as a string of 0s and 1s, and a
    digest of the turns' clues and picks, so the order can be checked on another machine."""
    out = {"config": result["config"]}
    for key in ("before", "after"):
        if key not in result:
            continue
        r = dict(result[key])
        turns = r.pop("turns")
        r["hits"] = "".join(str(t["hit"]) for t in turns)
        r["turns_sha256"] = turn_digest([(t["clue"], t["picked"]) for t in turns])
        out[key] = r
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--scheme", default="published", choices=SCHEMES)
    parser.add_argument("--sources", default="cultural", choices=SOURCES)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluate", default="val", choices=["val", "test", "none"])
    parser.add_argument("--boards", type=int, default=200, help="boards for the giver test")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "vectors")
    args = parser.parse_args()
    clock = Clock()
    st = importlib.import_module("sentence_transformers")
    losses = importlib.import_module("sentence_transformers.sentence_transformer.losses")
    datasets = importlib.import_module("datasets")
    peft = importlib.import_module("peft")
    torch = importlib.import_module("torch")
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    clock.mark("imported")

    data = datasets.Dataset.from_list(rows(args.seed, args.sources, args.scheme))
    words = all_words()
    clock.mark("data_ready")
    model = st.SentenceTransformer(args.model, device=args.device)
    clock.mark("model_loaded")
    base = args.model.replace("/", "--")
    before = encode(model, words)
    clock.mark("encoded_before")

    model.add_adapter(
        peft.LoraConfig(
            r=args.rank,
            lora_alpha=2 * args.rank,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=peft.TaskType.FEATURE_EXTRACTION,
        )
    )
    cuda = args.device == "cuda"
    trainer = st.SentenceTransformerTrainer(
        model=model,
        train_dataset=data,
        loss=losses.MultipleNegativesRankingLoss(model),
        args=st.SentenceTransformerTrainingArguments(
            output_dir=tempfile.mkdtemp(),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            learning_rate=args.lr,
            warmup_steps=0.05,  # a float is a fraction of the steps
            bf16=cuda,
            seed=args.seed,
            data_seed=args.seed,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            disable_tqdm=True,
        ),
    )
    trainer.train()
    clock.mark("trained")
    after = encode(model, words)
    clock.mark("encoded_after")

    config = vars(args) | {"out_dir": str(args.out_dir), "rows": len(data)}
    if cuda:
        config["gpu"] = torch.cuda.get_device_name(0)
    result: dict = {"config": config}
    if args.evaluate != "none":
        for key, vecs in (("before", before), ("after", after)):
            result[key] = full_scores(
                spaces(f"{base}-{key}", vecs, words), words, args.scheme, args.evaluate, args.boards
            )
            print(
                f"{key}: {args.evaluate} top1 {result[key]['top1']:.3f} "
                f"loglik {result[key]['loglik']:.3f} template {result[key]['chosen']['clue']} "
                f"tau {result[key]['chosen']['tau']}",
                flush=True,
            )
        clock.mark("scored")
    (args.out_dir / "scores.json").write_text(json.dumps(result, indent=1) + "\n")
    text = json.dumps(summary(result), indent=1, sort_keys=True) + "\n"
    (args.out_dir / "summary.json").write_text(text)
    print("summary.json sha256", hashlib.sha256(text.encode()).hexdigest(), flush=True)
    (args.out_dir / "train_log.json").write_text(
        json.dumps(trainer.state.log_history, indent=1) + "\n"
    )
    save_vectors(args.out_dir, f"{base}-base", before, words)
    save_vectors(args.out_dir, f"{base}-lora", after, words)
    model.save_pretrained(str(args.out_dir / "adapter"))
    clock.mark("saved")
    (args.out_dir / "timings.json").write_text(json.dumps(clock.summary(), indent=1) + "\n")


if __name__ == "__main__":
    main()
