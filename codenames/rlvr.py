"""Plan C7: a small language model trained by GRPO to choose among legal clues.

Each board comes with C6's 120 actions: 30 legal clues, proposed by EmbeddingGemma 2,
times the counts 1 to 4 (codenames.actions.legal_actions). The model reads C6's prompt (the board,
then the numbered clues) and answers CLUE NUMBER. Its reward is the exact expected turn
value of that action (codenames.luce, the README's costs), averaged over the four
training guessers, and -1 for a reply that names no listed clue or a count outside 1 to
4. The rewards of every action on every training board are computed beforehand from
the vectors (`python -m codenames.c7 table`) and read here from a table, so the job
needs no vectors and makes no model calls for the reward.

Training is trl.GRPOTrainer with the plan's settings: DAPO loss, upper clip 0.28, no KL
term, rewards scaled within each group of 8 replies, 16 boards a step, LoRA of rank 32
on every linear layer of the language model. Before and after training the job decodes
greedily on the evaluation boards and writes the picks; codenames.c7 scores them.

    python -m codenames.rlvr --seed 1 --steps 300 --eval test --table URL --out OUT  # L4
    python -m codenames.rlvr --tiny --steps 4 --table PATH --out OUT                 # CPU

--tiny replaces Qwen3.5-2B by a randomly initialized model of the same architecture
with a few thousand parameters and a word-level tokenizer, so the whole loop runs on a
CPU in a minute. It tests the code, not the method.

Outputs in --out: picks.json (each evaluation board's reply, parsed action and index,
before and after training), curves.json (per step: mean reward, share of -1 replies,
entropy, from TRL's log and from the replies), samples.jsonl.gz (every sampled reply
with its board and reward), the adapter, timings.json and config.json.
"""

import argparse
import gzip
import hashlib
import importlib
import importlib.util
import json
import re
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from codenames.game import Board
from codenames.lora import Clock
from codenames.rerank import action_index, rerank_prompt
from codenames.rerank import read_choice as c6_read_choice

MODEL = "Qwen/Qwen3.5-2B"
BOARDS_PER_STEP = 16
SAMPLES = 8
INVALID = -1.0
EVAL_SETS = ("val", "test")


# The prompt and the parser are C6's (codenames/rerank.py): llm.giver_prompt, then the
# numbered clues and the instruction to answer with one of them; the action is read from
# the last line that names a listed clue with a number after it.
def prompt(board: Board, clues: list[str]) -> str:
    return rerank_prompt(board, clues)


def parse(reply: str, clues: list[str], truncated: bool = False) -> dict:
    """C6's reading of a reply, with its reason when there is no action (empty,
    truncated, unlisted, count out of range), and the action index 4 i + n - 1."""
    row = c6_read_choice(reply, clues, "length" if truncated else None)
    return row | {"action": action_index(row, clues)}


def read_choice(reply: str, clues: list[str], truncated: bool = False) -> int | None:
    return parse(reply, clues, truncated)["action"]


def reward(reply: str, clues: list[str], values: np.ndarray, truncated: bool = False) -> float:
    """values[a] is the mean expected turn value of action a over the training guessers.
    A reply cut off at the token cap has no action, as in C6."""
    a = read_choice(reply, clues, truncated)
    return INVALID if a is None else float(values[a])


@dataclass
class Boards:
    """Boards with their candidate clues; values (boards, 120, guessers) when known."""

    words: np.ndarray  # (B, 25) str
    kinds: np.ndarray  # (B, 25) str
    clues: np.ndarray  # (B, 30) str
    values: np.ndarray | None = None

    def board(self, i: int) -> Board:
        return Board(tuple(self.words[i].tolist()), tuple(self.kinds[i].tolist()))

    def prompt(self, i: int) -> str:
        return prompt(self.board(i), self.clue_list(i))

    def clue_list(self, i: int) -> list[str]:
        return [str(c) for c in self.clues[i]]

    def __len__(self) -> int:
        return len(self.words)


def read_table(source: str, sha256: str | None = None) -> dict[str, Boards]:
    """The table written by codenames.c7 table, from a path or a URL, checked against its
    SHA-256 when one is given. Keys: train, val, test."""
    if re.match(r"https?://", source):
        with urllib.request.urlopen(source, timeout=120) as r:  # noqa: S310 (fixed URL)
            data = r.read()
    else:
        data = Path(source).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if sha256 and digest != sha256:
        raise ValueError(f"table {source}: sha256 {digest}, expected {sha256}")
    with tempfile.NamedTemporaryFile(suffix=".npz") as fh:
        fh.write(data)
        fh.flush()
        z = np.load(fh.name, allow_pickle=False)
        out = {}
        for split in ("train", *EVAL_SETS):
            if f"{split}_words" in z:
                values = z[f"{split}_values"] if f"{split}_values" in z else None
                out[split] = Boards(
                    z[f"{split}_words"], z[f"{split}_kinds"], z[f"{split}_clues"], values
                )
    return out


def step_boards(step: int) -> range:
    """The training boards of a step (0-based): 16 boards, fresh for each step."""
    return range(BOARDS_PER_STEP * step, BOARDS_PER_STEP * (step + 1))


# The tiny model for the CPU test.
TINY_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and not enable_thinking %}<think>\n\n</think>\n\n"
    "{% endif %}{% endif %}"
)
TINY_SPECIAL = ["<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]


def tiny_model(texts: list[str], out: Path, seed: int = 0) -> Path:
    """Save a random Qwen3.5 vision-language model of a few thousand parameters, with a
    word-level tokenizer over the words of texts, in the layout of the real checkpoint:
    loading it as a causal LM must drop the vision tower, as for Qwen/Qwen3.5-2B."""
    tokenizers = importlib.import_module("tokenizers")
    transformers = importlib.import_module("transformers")
    torch = importlib.import_module("torch")
    torch.manual_seed(seed)
    vocab = {t: i for i, t in enumerate(TINY_SPECIAL)}
    for text in [*texts, "user assistant system"]:
        for tok in re.findall(r"\w+|[^\w\s]", text):
            vocab.setdefault(tok, len(vocab))
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tok.decoder = tokenizers.decoders.WordPiece()  # tokens joined by spaces
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        pad_token="<|endoftext|>",
        eos_token="<|im_end|>",
        additional_special_tokens=TINY_SPECIAL[2:],
    )
    fast.chat_template = TINY_TEMPLATE
    q = importlib.import_module("transformers.models.qwen3_5")
    text = q.Qwen3_5TextConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        pad_token_id=fast.pad_token_id,
        eos_token_id=fast.pad_token_id,  # as in Qwen3.5-2B: <|endoftext|>, not <|im_end|>
        tie_word_embeddings=True,
    )
    vision = {"depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2}
    vision |= {"out_hidden_size": 32, "patch_size": 4, "temporal_patch_size": 2}
    config = q.Qwen3_5Config(
        text_config=text.to_dict(), vision_config=vision, tie_word_embeddings=True
    )
    model = q.Qwen3_5ForConditionalGeneration(config)
    model.save_pretrained(out)
    fast.save_pretrained(out)
    return out


def load_model(name: str, dtype: str):
    """The language model alone (Qwen3.5's vision tower is not loaded) and its tokenizer.
    Fails if any language-model weight was missing from the checkpoint."""
    transformers = importlib.import_module("transformers")
    torch = importlib.import_module("torch")
    model, info = transformers.AutoModelForCausalLM.from_pretrained(
        name, dtype=getattr(torch, dtype), output_loading_info=True
    )
    if info["missing_keys"]:
        raise RuntimeError(f"weights missing from {name}: {sorted(info['missing_keys'])[:10]}")
    names = [n for n, _ in model.named_modules()]
    if any("visual" in n for n in names):
        raise RuntimeError("the vision tower was loaded")
    tok = transformers.AutoTokenizer.from_pretrained(name, padding_side="left")
    return model, tok, {k: sorted(v)[:20] for k, v in info.items() if v}


def chat(tok, text: str) -> str:
    """The prompt as the model reads it, thinking off."""
    return tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def end_tokens(model, tok) -> list[int]:
    """The checkpoint's own end token is <|endoftext|>; a chat reply ends with the
    tokenizer's <|im_end|>. Either ends a reply."""
    eos = model.generation_config.eos_token_id
    return [tok.eos_token_id, *([eos] if isinstance(eos, int) else eos or [])]


def decode_greedy(model, tok, data: Boards, batch: int = 25, max_new: int = 16) -> list[dict]:
    """Greedy replies on every board of data, each with its parsed action (or None)."""
    torch = importlib.import_module("torch")
    model.eval()
    out = []
    device = next(model.parameters()).device
    eos = end_tokens(model, tok)
    for start in range(0, len(data), batch):
        idx = list(range(start, min(start + batch, len(data))))
        texts = [chat(tok, data.prompt(i)) for i in idx]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=max_new,
                pad_token_id=tok.pad_token_id,
                eos_token_id=eos,
            )
        new = gen[:, enc["input_ids"].shape[1] :]
        replies = tok.batch_decode(new, skip_special_tokens=True)
        for i, reply, ids in zip(idx, replies, new.tolist(), strict=True):
            ends = [j for j, t in enumerate(ids) if t in eos]
            cut = not ends  # no end token within the cap
            row = {"board": i, "reply": reply, "tokens": ends[0] if ends else len(ids)}
            row["truncated"] = cut
            out.append(row | parse(reply, data.clue_list(i), cut))
    return out


class Recorder:
    """The reward function given to GRPOTrainer. It keeps every sampled reply."""

    __name__ = "c7_reward"

    def __init__(self, data: Boards, eos: list[int]):
        assert data.values is not None
        self.data = data
        self.eos = set(eos)
        self.mean = data.values.mean(axis=2)  # (B, 120): mean over the training guessers
        self.rows: list[dict] = []

    def __call__(
        self, prompts, completions, board, completion_ids=None, trainer_state=None, **kw
    ) -> list[float]:
        step = trainer_state.global_step if trainer_state is not None else -1
        ids = completion_ids or [None] * len(completions)
        out = []
        for comp, b, c in zip(completions, board, ids, strict=True):
            text = comp[-1]["content"] if isinstance(comp, list) else comp
            cut = c is not None and not any(t in self.eos for t in c)
            r = reward(text, self.data.clue_list(b), self.mean[b], cut)
            out.append(r)
            row = {"step": step, "board": int(b), "reply": text, "reward": r}
            self.rows.append(row | ({"truncated": True} if cut else {}))
        return out

    def curves(self) -> list[dict]:
        by: dict[int, list[float]] = {}
        cut: dict[int, list[bool]] = {}
        for row in self.rows:
            by.setdefault(row["step"], []).append(row["reward"])
            cut.setdefault(row["step"], []).append(row.get("truncated", False))
        return [
            {
                "step": s,
                "replies": len(r),
                "mean_reward": float(np.mean(r)),
                "invalid": float(np.mean([x == INVALID for x in r])),
                "truncated": float(np.mean(cut[s])),
            }
            for s, r in sorted(by.items())
        ]


def stopper(
    start: float,
    minutes: float | None,
    step: int | None,
    flat: int | None = None,
    min_steps: int = 0,
):
    """A callback that ends training after the step during which the time ran out, after
    a given step, or once the last `flat` steps (and at least min_steps in all) had no
    group with any spread of rewards. Then every advantage is 0, and so is the gradient.
    The learning-rate schedule still spans --steps."""
    transformers = importlib.import_module("transformers")

    class Stop(transformers.TrainerCallback):
        run = 0

        def halt(self, state, control, why: str):
            control.should_training_stop = True
            print(f"stopping after step {state.global_step}: {why}", flush=True)

        def on_step_end(self, args, state, control, **kw):
            if minutes is not None and time.time() - start > 60 * minutes:
                self.halt(state, control, f"{minutes} min")
            if step is not None and state.global_step >= step:
                self.halt(state, control, f"step {step}")
            return control

        def on_log(self, args, state, control, logs=None, **kw):
            if flat is None or not logs or "frac_reward_zero_std" not in logs:
                return control
            self.run = self.run + 1 if logs["frac_reward_zero_std"] == 1 else 0
            if self.run >= flat and state.global_step >= min_steps:
                self.halt(state, control, f"{flat} steps without reward spread")
            return control

    return Stop()


def grpo_config(args, out: Path, cpu: bool):
    trl = importlib.import_module("trl")
    return trl.GRPOConfig(
        output_dir=str(out / "trainer"),
        max_steps=args.steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=BOARDS_PER_STEP * SAMPLES // args.batch,
        num_generations=SAMPLES,
        max_completion_length=16,
        temperature=1.0,
        loss_type="dapo",
        epsilon_high=0.28,
        beta=0.0,
        scale_rewards="group",
        mask_truncated_completions=True,
        shuffle_dataset=False,  # step s trains on boards 16 s to 16 s + 15
        chat_template_kwargs={"enable_thinking": False},
        bf16=not cpu,
        use_cpu=cpu,
        seed=args.seed,
        data_seed=args.seed,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--tiny", action="store_true", help="a random tiny model, on CPU")
    parser.add_argument("--table", required=True, help="path or URL of the c7 table")
    parser.add_argument("--table-sha256", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch", type=int, default=32, help="replies per forward pass")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--eval", default="test", choices=EVAL_SETS)
    parser.add_argument("--eval-boards", type=int, default=None, help="first n only")
    parser.add_argument(
        "--stop-after",
        type=float,
        default=None,
        help="minutes since the module started after which training stops at the end of a "
        "step, so that the evaluation and the outputs still finish before the job's timeout",
    )
    parser.add_argument(
        "--stop-at-step",
        type=int,
        default=None,
        help="end training after this step; the learning rate still decays over --steps",
    )
    parser.add_argument(
        "--stop-flat",
        type=int,
        default=None,
        help="end training once this many steps in a row had no reward spread in any group",
    )
    parser.add_argument("--min-steps", type=int, default=0, help="for --stop-flat")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    clock = Clock()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.tiny:
        # The Triton kernels of flash-linear-attention need a GPU. Hiding the package
        # makes transformers use its PyTorch path for Gated DeltaNet, as it does when
        # the package is absent.
        if "fla" not in sys.modules:
            sys.modules["fla"] = cast(Any, None)
    torch = importlib.import_module("torch")
    datasets = importlib.import_module("datasets")
    peft = importlib.import_module("peft")
    trl = importlib.import_module("trl")
    clock.mark("imported")

    table = read_table(args.table, args.table_sha256)
    train = table["train"]
    need = BOARDS_PER_STEP * args.steps
    if len(train) < need:
        raise ValueError(
            f"{args.steps} steps need {need} training boards; the table has {len(train)}"
        )
    ev = table[args.eval]
    if args.eval_boards:
        ev = Boards(*(x[: args.eval_boards] for x in (ev.words, ev.kinds, ev.clues)))
    clock.mark("table_read")

    cpu = args.tiny or not torch.cuda.is_available()
    name = args.model
    if args.tiny:
        texts = [train.prompt(i) for i in range(need)] + [ev.prompt(i) for i in range(len(ev))]
        name = str(tiny_model(texts, Path(tempfile.mkdtemp()), args.seed))
    torch.manual_seed(args.seed)
    model, tok, loading = load_model(name, "float32" if cpu else "bfloat16")
    if not cpu:
        model = model.to("cuda")
    clock.mark("model_loaded")

    picks = {"untrained": decode_greedy(model, tok, ev)}
    clock.mark("decoded_untrained")

    rows = [
        {"prompt": [{"role": "user", "content": train.prompt(i)}], "board": i} for i in range(need)
    ]
    recorder = Recorder(train, end_tokens(model, tok))
    lora = peft.LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, target_modules="all-linear", task_type="CAUSAL_LM"
    )
    trainer = trl.GRPOTrainer(
        model=model,
        reward_funcs=recorder,
        args=grpo_config(args, args.out, cpu),
        train_dataset=datasets.Dataset.from_list(rows),
        processing_class=tok,
        peft_config=lora,
        callbacks=[
            stopper(
                clock.marks["module_start"],
                args.stop_after,
                args.stop_at_step,
                args.stop_flat,
                args.min_steps,
            )
        ],
    )
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    clock.mark("trainer_ready")
    trainer.train()
    clock.mark("trained")
    picks["trained"] = decode_greedy(trainer.model, tok, ev)
    clock.mark("decoded_trained")

    seen = sorted({r["board"] for r in recorder.rows})
    log = [r for r in trainer.state.log_history if "reward" in r]
    curves = recorder.curves()
    for row, entry in zip(curves, log, strict=False):
        row["trl"] = {k: entry[k] for k in ("reward", "entropy", "loss", "step") if k in entry}
    config = vars(args) | {
        "out": str(args.out),
        "table": args.table,
        "model_loaded": name if not args.tiny else "tiny",
        "trainable_parameters": int(trainable),
        "loading_info": loading,
        "boards_seen": [seen[0], seen[-1], len(seen)] if seen else [],
        "versions": {m.__name__: m.__version__ for m in (torch, trl, peft, datasets)},
        "fla": not args.tiny and importlib.util.find_spec("fla") is not None,
        "steps_done": trainer.state.global_step,
    }
    if not cpu:
        config["gpu"] = torch.cuda.get_device_name(0)
        config["max_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    (args.out / "config.json").write_text(json.dumps(config, indent=1, default=str) + "\n")
    (args.out / "picks.json").write_text(json.dumps(picks, indent=1) + "\n")
    (args.out / "curves.json").write_text(json.dumps(curves, indent=1) + "\n")
    (args.out / "trl_log.json").write_text(json.dumps(trainer.state.log_history, indent=1) + "\n")
    with gzip.open(args.out / "samples.jsonl.gz", "wt") as fh:
        for row in recorder.rows:
            fh.write(json.dumps(row) + "\n")
    trainer.model.save_pretrained(str(args.out / "adapter"))
    clock.mark("saved")
    (args.out / "timings.json").write_text(json.dumps(clock.summary(), indent=1) + "\n")
    print(json.dumps({k: v for k, v in curves[-1].items()} if curves else {}), flush=True)
    print("seconds", round(time.time() - clock.marks["module_start"], 1), flush=True)


if __name__ == "__main__":
    main()
