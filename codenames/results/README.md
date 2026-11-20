# Notes on the Codenames experiments

Detail behind the [Codenames README](../README.md), moved here from it. Every number is as it was in the README.

| Note | Question | Plan |
|---|---|---|
| [guessers.md](guessers.md) | Which models predict human guesses, on the published split (sections 1 and 4) | exploratory |
| [crossplay.md](crossplay.md) | What a clue is worth to a guesser on another model, and whole games against Kim et al. (sections 2 and 3) | exploratory |
| [c1.md](c1.md) | The same on a split by game, and what the published split leaks | `experiments/plans/C1.md` |
| [c2.md](c2.md) | LoRA fine-tuning on human guesses | `experiments/plans/C2.md` |
| [c8.md](c8.md) | Adding a language model's labels to the human turns | `experiments/plans/C8.md` |
| [c3.md](c3.md) | A giver trained against a population of other models (section 5) | `experiments/plans/C3.md` |
| [c4.md](c4.md) | Hosted language models as guessers and givers | `experiments/plans/C4.md` |
| [c6.md](c6.md) | Language models choosing among 120 legal actions | `experiments/plans/C6.md` |
| [c7.md](c7.md) | GRPO training of a small language model to choose | `experiments/plans/C7.md` |
| [c10.md](c10.md) | A language model as the listener inside the giver | `experiments/plans/C10.md` |
| [mathematics.md](mathematics.md) | Exact turn odds, the best count, and the single-peak observation | |

## Short answers, with intervals in percentage points

The README's earlier summary, kept here because it gives each interval in points.

- EmbeddingGemma 2 and the small static table potion-base-8M pick the same word as a person most often: 54.8% and 54.4% of 551 turns from 68 test games, with the Cultural Codes turns split by game (95% intervals 50.1% to 59.5% and 49.8% to 59.3%; their difference, 0.4 points, has the interval −3.8 to 4.6). The other six models pick it in 39.4% to 49.0%. Instruction prefixes help the instruction-trained models a little and ruin the others. Computed exactly on the test turns, with intervals from resampling games.
- Training a linear map on potion's clue vectors raises its match from 54.4% to 63.2%. The map is trained on the human guesses of the training games only. The gain, 8.8 points (6.3 to 11.4), is as large as on the published split (8.2), so it was not an artifact of test games having turns in training. Same method; the map is averaged over five training seeds.
- Fine-tuning Qwen3-Embedding-0.6B with LoRA on the same training games raises its match from 43.4% to 62.6%, level with the potion map: LoRA minus the map is −0.7 points (−4.1 to 2.8). Same method, three training seeds, on an A10G GPU.
- A giver expects to find 1.8 to 2.1 of its words per clue when the guesser shares its vectors. A guesser built on another model finds 57% to 73% of that, on average over the other seven models. A guesser that follows association counts from people finds 37% to 53%. Exact expected values for each of 200 random boards, one clue each, for eight models.
- Hosted language models, asked with a fixed prompt, predict human guesses better than untrained embeddings and give worse clues. gpt-oss-120b picks the person's word in 65.0% of the 551 test turns (60.8% to 69.3%), 10.2 points (7.1 to 13.5) above EmbeddingGemma 2 and level with the trained guessers above. As givers, all four models tested give clues worth 0.29 to 0.89 words less per turn to seven embedding guessers they never met than the EmbeddingGemma 2 giver's. Paired intervals from resampling games or boards; one reply per prompt.
- Adding about 19,000 turns labeled by gpt-oss-120b to those 4,820 human turns raises the LoRA encoder's match from 62.6% to 65.9%. The gain is 3.3 points (0.9 to 5.9), and the student ends level with its teacher (65.0%). Neither the teacher's labels alone nor its labels in place of the people's beat the human turns. Same method, three training seeds per arm.
- Choosing from a list of 120 legal actions (30 clues times counts 1 to 4) proposed by the EmbeddingGemma 2 giver does not rescue the language models. The model selected on validation boards, Qwen3.8-27B, is 0.059 words per turn below the giver's own first choice (−0.116 to −0.005), and none of six models is above it. The best action in each list is 0.44 above the first choice, so the list leaves room. Exact expected values on 200 test boards with seven unseen embedding guessers; one reply per prompt.
- Using gpt-oss-120b as a listener inside the embedding giver does not help either. The model guesses on each of the giver's 10 best clues, and the giver gives the action worth most under those guesses. That action is 0.151 words per turn below the giver's own first choice with the strangers (−0.204 to −0.099), and 0.232 below with the human-fitted guesser. Mixed with a population of six embedding guessers, it adds −0.017 (−0.034 to 0.001) to the population alone. Exact expected values on 300 test boards, in a run planned in advance (plan C10).
- A giver trained by policy gradient against six other models scores more with a seventh model it never trained with than a giver trained against its own model. The gain is 0.31 words per turn (0.26 to 0.36), averaged over 14 held-out choices and five training seeds. Population training beat self-play in all 70 runs, at a cost of 0.50 (0.46 to 0.53) with the giver's own model. The gain is smallest, 0.09 (0.001 to 0.18), when the Qwen3 giver meets harrier-oss-v1-0.6b. Against the association counts, which stand in for people, the gain is 0.39 (0.31 to 0.46). Exact rewards on 200 test boards, with a stratified bootstrap over boards and seeds, in a run planned in advance (plan C3).

## How a guesser is modeled

A guesser that ranks words by similarity is deterministic. People are not, so each guesser also gets a **Luce guesser**: it picks a word with probability proportional to exp(similarity / τ) among the words still unpicked. The temperature τ is fitted to the human guesses in the Cultural Codes train split, and the template (below) to the val split. For the association-count guesser the weight of a word is its count plus ε, with ε = 0.1 fitted the same way. That guesser answers only clues that are EAT cues; for other clues it picks at random.

## What was computed where

| Result | Where |
|---|---|
| Vectors for potion, wordllama, granite, bge; every table in the README and these notes; training and policy gradient, except plan C3 | this machine, 2 CPUs |
| Plan C3's 70 policy-gradient runs (`c3.py`) | GitHub-hosted runners (CPU), workflow on `compute/codenames-c3`; analyzed on this machine |
| Vectors for Qwen3-Embedding-0.6B, harrier-oss-v1-270m, harrier-oss-v1-0.6b, EmbeddingGemma 2 | GitHub-hosted runner (CPU), workflow on `compute/codenames-embed` |
| LoRA fine-tuning (`lora.py`) | Hugging Face Jobs, `a10g-small`, through `experiments/hfrun.py` |
| The teacher's labels for C8 (`distill.py`) | Hugging Face Jobs, `cpu-basic`, through `experiments/hfrun.py`, calling Inference Providers; analyzed on this machine (`c8.py`) |
| C7: tables, the feature policy and the scoring (`c7.py`) | this machine |
| C7: GRPO of Qwen3.5-2B (`rlvr.py`) | Hugging Face Jobs, `l40sx1`, through `experiments/hfrun.py` |
| Language-model players (`llm.py`, `hosted.py`, `c4.py`) | Hugging Face Jobs, `cpu-basic`, through `experiments/hfrun.py`, calling Inference Providers; scored on this machine (`c4score.py`) |
| Language models choosing among legal clues (`rerank.py`) | Hugging Face Jobs, `cpu-basic`, calling Inference Providers; action lists and scoring on this machine (`actions.py`, `c6score.py`) |
| A language model as the listener (`listen.py`) | Hugging Face Jobs, `cpu-basic`, calling Inference Providers; lists, the fit of T and scoring on this machine (`c10score.py`) |
| Vectors for the 15 clue words of the language models that had none | GitHub-hosted runner, workflow on `compute/codenames-embed`, for the four Hugging Face models; this machine for the other four |

## Models

| Model | Kind | Dims | License | Vectors computed |
|---|---|---|---|---|
| potion-base-8M | static table distilled from bge-base-en-v1.5 | 256 | MIT | here |
| wordllama l2_supercat | static table trained from LLM token embeddings | 256 | MIT | here |
| granite-embedding-small-english-r2 | ModernBERT encoder, 48M parameters, int8 | 384 | Apache 2.0 | here |
| bge-small-en-v1.5 | BERT encoder, 33M parameters, int8 | 384 | MIT | here |
| Qwen3-Embedding-0.6B | decoder | 1024 | Apache 2.0 | GitHub-hosted runner |
| harrier-oss-v1-270m | decoder, last-token pooling | 640 | MIT | GitHub-hosted runner |
| harrier-oss-v1-0.6b | decoder, last-token pooling | 1024 | MIT | GitHub-hosted runner |
| EmbeddingGemma 2 | multimodal embedding model, 744M parameters in all | 768 | Apache 2.0 | GitHub-hosted runner |

## Human data

| Data | What | License | Committed |
|---|---|---|---|
| Cultural Codes (Shaikh et al. 2023) | 6,088 Codenames Duet guesses with clue and board (the rows that parse); train, val and test splits as published | none in the repository; the paper says CC BY-SA 4.0 is planned | no; `fetch.py` downloads it |
| Edinburgh Associative Thesaurus | about 100 people's first responses to each of 8,400 cues, collected 1968–1971 | UK Open Government Licence, as stated by the JSON port used; the original site is gone, so this was not checked at the source | no; downloaded |
| Small World of Words, SWOW-EN18 | 3 million responses to 12,000 cues | CC BY-NC-ND 3.0 | no; manual download, loader only |
| USF free association norms | 72,000 cue-response pairs | copyright of the authors, no license stated | no; manual download |

Only the first two are used in the numbers below. The Cultural Codes files carry no game or player ids. [c1.md](c1.md) rebuilds both and finds that the published split puts the two sides of most games in different files. Sections 1 to 4 use the files as published; that note repeats sections 1 and 4 on a split by game. A fixed fifth of EAT cues, chosen by hash (`norms.held_out`), is never used for training.

## Board and clue words

The official word lists belong to Czech Games Edition and are not used or committed here. `words.txt` holds 400 board words chosen by `make_words.py` from Princeton WordNet 3.0. A board word is a lowercase noun of 3 to 9 letters, with at least two noun senses, whose most common sense names a thing (an animal, artifact, food, person, place, plant or substance). Its Zipf frequency in wordfreq must lie between 3.6 and 5.6, and WordNet's sense-tagged corpus must tag it as a noun at least as often as a verb or adjective. The 400 most frequent such words are kept, after removing plurals and other inflections and a short list of words that would be out of place on a family board (the list is in the script). WordNet 3.0 may be copied and modified under the Princeton license, whose notice must travel with copies; `fetch.py` downloads it with that notice.

Clues come from the 12,000 most frequent words in wordfreq (version 3.1.1) that are lowercase, alphabetic, at least three letters long and in WordNet. wordfreq's code is under the Apache 2.0 license and its data under CC BY-SA 4.0. It is used as a package, and no frequency data is committed. A rarity penalty, 0.1 × max(0, 3.5 − Zipf frequency), keeps the giver to words people know.

## Where the model vectors came from

The licenses and sizes were read from each model card through the Hugging Face Hub API. The machine that ran this study cannot reach huggingface.co. The first four models come from packages on npm and PyPI that redistribute the weights under their licenses, pinned by SHA-256 in `fetch.py`. The potion table comes from `@jigyasudham/veto-model`, an int8 repack whose shipped test vectors record a cosine error of at most 6.13e-5 against the original float32 weights. The wordllama table is inside the `wordllama` wheel. The granite files are the int8 export by onnx-community, shipped in `isorite-model-granite`; both have the sizes listed on the Hub. bge comes from `impro-plugin-semantic-mute-model`. The Hugging Face models ran on a GitHub-hosted runner (CPU) on 2026-11-18, with the latest sentence-transformers and torch, through a workflow on the branch `compute/codenames-embed` that commits the vectors to that branch. `compute/codenames-embed.yml` is a copy of that workflow. Each vector file is 11 to 25 MB, too large for this folder. To use them, copy `codenames/vectors/` from that branch, or rerun `embed.py` with `--model <id>` on a machine that can reach the Hub. EmbeddingGemma 2 also needs torchvision and Pillow, even for text.

GloVe and fastText could not be downloaded here, so the static baselines are potion and wordllama.

## Open questions

- Does the single-peak property of R(n) hold for every weighting?
- Would the order of models change with SWOW, which is larger and recent?
- Does the potion map's gain hold when every pick of a multi-pick turn is used, as White et al. do?
- Why does a trained guesser model make a worse giver?
- Would LoRA beat the potion map with other settings (a larger learning rate, more epochs, harrier-oss-v1-0.6b)? Would five epochs on the human turns alone give part of the teacher's gain? Does the gain grow with more teacher turns?
- Would language-model givers do better with a prompt that states the whole clue rule (no word contained in a board word) and the cost of the assassin, or with a retry after an illegal clue?
- Do language-model clues fare better with people than with embedding guessers?
- Would GRPO with an entropy bonus, more replies per board, or several gradient steps per batch keep exploring past the first listed clue (C7)?
- Why does gpt-oss-120b return an empty reply on a fifth of C6's test boards and none of its validation boards? In C10, 850 of 3,000 test calls came back empty at first, against 1 of 2,000 validation calls a few hours earlier.
- Would a listener told the real count, or modeled with more than one temperature over ranks, do better than C10's? Its log-likelihood of the person's word was worse than EmbeddingGemma 2's although its first word matched the person more often.
- Would the language models choose better than the embedding giver's first choice with a prompt found by GEPA, or with room to reason before the answer? C6's prompt evolution was not run.

## Every command

The package's packages are the `codenames` dependency group (huggingface-hub, numpy, onnxruntime, tokenizers, wordfreq). Times are from one run on this machine, while other jobs shared its two CPUs.

```bash
uv run python -m codenames.fetch all              # models and data, about 100 MB (20 s)
uv run python -m codenames.make_words             # rewrites words.txt
uv run python -m codenames.embed --model granite --template bare query   # vectors (1 to 9 min each)
uv run python -m codenames.study calibrate        # section 1 (1 min)
uv run python -m codenames.study crossplay        # section 2 (25 min)
uv run python -m codenames.study games            # section 3 (60 min)
uv run python -m codenames.study peaks            # the single-peak check (3 min)
uv run python -m codenames.study report           # prints the tables
uv run python -m codenames.senses                 # senses (2 min)
uv run python -m codenames.train --model potion   # section 4 (2 min)
uv run python -m codenames.split --check          # rebuild games and players, compare with splits.json
uv run python -m codenames.c1                     # split by game, the plan's run (2 min)
uv run python -m codenames.heldout                # split by game and player, exploratory (5 min)
uv run python -m codenames.c2                     # LoRA results from results/c2-runs/ (1 min)
uv run python -m codenames.c8 --giver             # C8 results from results/c8-runs/ (1 min; --giver needs the vectors)
uv run python -m codenames.rl --giver potion --held-out bge --out codenames/results/rl-potion.json
uv run python -m codenames.c3 --slice 0 --out OUT     # plan C3: one of its 70 runs (1 to 2 min)
uv run python -m codenames.c3 --runs runs/C3/44547011858-1 --out codenames/results/c3.json   # from compute/codenames-c3
uv run python -m codenames.c7 table --steps 300 --out DIR          # C7 tables (7 min)
uv run python -m codenames.c7 feature --data DIR --seed 1 --out OUT  # C7 arm 2 (1 s)
# C7 arm 1, inside a Hugging Face Job on an L40S (group codenames-gpu); --tiny runs it on a CPU:
uv run --group codenames-gpu python -m codenames.rlvr --seed 1 --steps 300 --batch 64 --eval test --stop-flat 20 --min-steps 100 --stop-after 60 --table URL --table-sha256 SHA --out OUT
uv run python -m codenames.c7 score --data DIR --runs codenames/results/c7-runs --out codenames/results/c7.json   # 1 min
uv run --with sentence-transformers python -m codenames.embed --model Qwen/Qwen3-Embedding-0.6B
# C4, inside a Hugging Face Job (needs HF_TOKEN), then scored here:
uv run python -m codenames.c4 guess --models all --split val --cap 0.6 --out OUT
uv run python -m codenames.c4 give --models all --seed 4005 4004 0 --n 50 200 200 --cap 0.4 --out OUT
uv run python -m codenames.c4 crossguess --models all --seed 4004 --clues codenames/results/c4-clues-4004.txt --cap 0.3 --out OUT
uv run python -m codenames.c4 score --clues clues.txt --guesses guesses.txt --crossguess crossguess.txt --out codenames/results/c4.json   # 15 min
# C8, inside Hugging Face Jobs (needs HF_TOKEN):
uv run python -m codenames.distill label --cap 2.5 --workers 64 --stop-after 210 --out OUT   # 2.4 h
uv run --group codenames-gpu python -m codenames.lora --scheme game --sources cultural+teacher --lr 3e-4 --seed 1 --teacher runs/c8/label/teacher.jsonl --evaluate test --out-dir OUT
# C6: the action lists here, the choices in a Hugging Face Job, the scores here:
uv run python -m codenames.actions --seed 6999 6005 6004 --n 5 50 200 --out codenames/results/c6-actions.txt   # 15 s
uv run python -m codenames.rerank choose --models all --seed 6005 --n 50 --repeat 20 --cap 0.2 --out OUT
uv run python -m codenames.rerank choose --models all --seed 6004 --n 200 --cap 0.5 --out OUT
uv run python -m codenames.rerank score --out codenames/results/c6.json   # 1 min, also writes c6-boards.json
uv run python -m codenames.rerank picks --models qwen3.8-27b --seed 6004 --out codenames/results/c6-picks-6004.txt
uv run python -m codenames.rerank crossguess --models gpt-oss-120b --clues codenames/results/c6-picks-6004.txt --seed 6004 --cap 0.15 --out OUT
# C10: the lists and the fit here, the listener in Hugging Face Jobs, the scores here:
uv run python -m codenames.actions --seed 9999 9005 9004 --n 5 100 300 --out codenames/results/c10-actions.txt
uv run python -m codenames.listen lists --seed 9999 9005 9004 --n 5 100 300 --out codenames/results/c10-listen.txt   # 1 min
uv run python -m codenames.listen prefixes --out OUT        # in a job: C4's replies, read with the tail rule
uv run python -m codenames.listen fit --replies codenames/results/c10-c4-val-prefixes.txt --out codenames/results/c10-listener.json
uv run python -m codenames.listen listen --seed 9005 --boards 0-50 --k 20 --repeat 10 --cap 0.15 --out OUT
uv run python -m codenames.listen select --out codenames/results/c10-val.json   # 20 s
uv run python -m codenames.listen listen --seed 9004 --boards 0-50 --k 10 --cap 0.10 --out OUT
uv run python -m codenames.listen speak --seed 9004 --n 300 --k 10 --cap 0.05 --out OUT
uv run python -m codenames.listen picks --out codenames/results/c10-picks-9004.txt
uv run python -m codenames.rerank crossguess --models llama-3.3-70b,deepseek-v3.2 --clues codenames/results/c10-picks-9004.txt --seed 9004 --distinct --cap 0.30 --out OUT
uv run python -m codenames.listen score --out codenames/results/c10.json   # 1 min
```

## Related work, in full

- Summerville et al. (2019) set up the Codenames AI competition. Kim et al. (2019) compared word2vec, GloVe and WordNet codemasters and guessers in it. They found twin pairs always win and mixed pairs often lose, and the folder reproduces that pattern with current models.
- Jaramillo et al. (2020) added transformer and TF-IDF bots and found the same drop for mixed pairs.
- Koyyalagunta et al. (2021) generate clues from BabelNet and from several embeddings, with a frequency term to avoid obscure clues, and evaluate them with people.
- Archibald and Brosnahan (2024) build a giver that adapts to its teammate. Bills, Archibald and Blaylock (AAAI 2025) infer the partner's embedding by Bayesian updating over a set of models; the ensemble giver above is the version without updating.
- Shaikh et al. (2023) collected Cultural Codes. White, Pandey and Pan (2024) trained a linear map on GloVe from it and added pragmatic reasoning.
- Stephenson, Sidji and Ronval (2024) use Codenames as a benchmark for LLMs, as do Hakimov, Pfennigschmidt and Schlangen (2025). Deshpande et al. (2026) train Qwen3 models on Codenames with reinforcement learning from verifiable rewards, with a frozen guesser as judge.
- Hu et al. (2020), "Other-Play", show that agents trained only with themselves form conventions that strangers do not share.
- Agarwal et al. (2021), "Deep Reinforcement Learning at the Edge of the Statistical Precipice" (NeurIPS), recommend stratified bootstrap intervals, the interquartile mean and the probability of improvement when runs are few; plan C3 uses all three.

## Files, in detail

| File | What |
|---|---|
| `game.py` | rules, boards, the game loop |
| `luce.py` | exact turn odds, the best count |
| `players.py` | spaces, givers (threshold, expected, ensemble), guessers |
| `words.py`, `make_words.py`, `words.txt` | board words, clue words, the clue rule |
| `embed.py` | the models, templates and the vector cache |
| `fetch.py` | downloads, pinned by SHA-256 |
| `norms.py` | EAT, SWOW, Cultural Codes |
| `evaluate.py`, `senses.py` | section 1 |
| `study.py` | sections 1 to 3 and the tables |
| `train.py` | section 4 |
| `lora.py`, `c2.py` | LoRA fine-tuning: the GPU job, its scoring, and the analysis |
| `split.py`, `splits.json` | games and players rebuilt from Cultural Codes, and the splits by game and by player |
| `stats.py` | the bootstrap over games, Kendall's τ |
| `c1.py`, `heldout.py` | human guesses split by game: the plan's run, and the exploratory one |
| `rl.py` | section 5 |
| `c3.py` | section 5 across seeds and held-out guessers: the runs of plan C3 and their analysis |
| `actions.py` | the 120 legal actions an embedding giver proposes on a board (C6 and C7) |
| `rerank.py`, `c6score.py` | C6: language models choosing among those actions, and the scoring |
| `rlvr.py`, `c7.py` | C7: the GRPO job, its tables, the feature policy on the same candidates, and the analysis |
| `listen.py`, `c10score.py` | C10: a language model as the listener inside the giver, and the scoring |
| `llm.py` | language-model players: prompts and parsing |
| `hosted.py` | calls to Inference Providers, with retries, a cost meter and a log of every call |
| `c4.py`, `c4score.py` | experiment C4: the jobs that call the models, and the scoring |
| `distill.py`, `c8.py` | experiment C8: the teacher's labels on resampled boards, and the analysis |
| `results/*.json` | every number above, with seeds and settings |
| `compute/codenames-embed.yml` | a copy of the workflow that computed the Hugging Face vectors |
