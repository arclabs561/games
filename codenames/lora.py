"""LoRA fine-tuning of a Hugging Face embedding model on the same human data as train.py.

Not run on the machine that produced the README's numbers: it needs the Hugging Face
Hub and, to finish in reasonable time, a GPU (CUDA) or an Apple-silicon Mac (MPS).

    uv run --group codenames --with "sentence-transformers[train]>=5" --with peft --with torch \\
        python -m codenames.lora --model Qwen/Qwen3-Embedding-0.6B --device mps --epochs 1

Each training item is (clue with the "query" template, the word a person connected to
it, and seven hard negatives: other words from the same board, or random board words
for EAT items). The loss is sentence-transformers' MultipleNegativesRankingLoss, which
is InfoNCE over the hard negatives and the other items in the batch. Only LoRA adapters
(rank 16 on the attention projections) are trained. Afterwards the script writes vectors
for the whole vocabulary, so codenames.study and codenames.evaluate can use the tuned
model under the name given by --name.
"""

import argparse
import importlib
import random
from pathlib import Path

import numpy as np

from codenames.embed import TEMPLATES, normalize
from codenames.evaluate import all_words
from codenames.norms import cultural_guesses, eat_counts, held_out
from codenames.words import board_words

N_NEG = 7


def rows(seed: int = 0) -> list[dict[str, str]]:
    """Training rows: anchor, positive, negative_1 .. negative_7."""
    rng = random.Random(seed)
    q = TEMPLATES["query"]
    out = []
    for g in cultural_guesses("train"):
        others = [w for w in g.words if w != g.picked]
        if len(others) >= N_NEG:
            negs = rng.sample(others, N_NEG)
            out.append({"anchor": q.format(g.clue), "positive": g.picked} | neg(negs))
    board = list(board_words())
    for cue, row in sorted(eat_counts().items()):
        if held_out(cue):
            continue
        pool = [w for w in board if w not in row and w != cue]
        for resp, n in sorted(row.items()):
            if resp != cue and n >= 5:
                out.append(
                    {"anchor": q.format(cue), "positive": resp} | neg(rng.sample(pool, N_NEG))
                )
    rng.shuffle(out)
    return out


def neg(words: list[str]) -> dict[str, str]:
    return {f"negative_{i + 1}": w for i, w in enumerate(words)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--name", default=None, help="name for the tuned vectors")
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "vectors")
    args = parser.parse_args()
    st = importlib.import_module("sentence_transformers")
    losses = importlib.import_module("sentence_transformers.losses")
    datasets = importlib.import_module("datasets")
    peft = importlib.import_module("peft")
    torch = importlib.import_module("torch")
    torch.manual_seed(args.seed)

    model = st.SentenceTransformer(args.model, device=args.device)
    model.add_adapter(
        peft.LoraConfig(
            r=args.rank,
            lora_alpha=2 * args.rank,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=peft.TaskType.FEATURE_EXTRACTION,
        )
    )
    data = datasets.Dataset.from_list(rows(args.seed))
    trainer = st.SentenceTransformerTrainer(
        model=model,
        train_dataset=data,
        loss=losses.MultipleNegativesRankingLoss(model),
        args=st.SentenceTransformerTrainingArguments(
            output_dir=str(args.out_dir / "lora-run"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            learning_rate=1e-4,
            warmup_ratio=0.05,
            seed=args.seed,
            logging_steps=50,
            save_strategy="no",
            report_to="none",
        ),
    )
    trainer.train()

    name = args.name or args.model.replace("/", "--") + "-lora"
    words = all_words()
    for template in ("bare", "query"):
        texts = [TEMPLATES[template].format(w) for w in words]
        vecs = normalize(np.asarray(model.encode(texts, batch_size=128, prompt="")))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out_dir / f"{name}__{template}.npz",
            words=np.array(words),
            vectors=vecs.astype(np.float16),
        )
    print(f"wrote {name} vectors to {args.out_dir}")


if __name__ == "__main__":
    main()
