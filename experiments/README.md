# Experiments

Most results in this repository are exact and run in seconds on a laptop. The experiments listed here need more: a GPU, a hosted language model, or hours of CPU. This folder says how they run and where their results live.

## Where things live

- **Code** is in this repository. A run names the commit it used, so it can be repeated.
- **Plans** are in `plans/<id>.md`, one per experiment. Each names its question, metric, seeds and held-out partners. A plan is merged before its runs and is not changed after; a later change goes in a dated amendment at its end. `METHOD.md` gives the rules all plans share.
- **Summaries** (small JSON files with every number a README quotes) go in the game's own `results/` folder, committed with the README that uses them.
- **Artifacts** that are large or whose license forbids publishing (vectors, fine-tuned adapters, model outputs, downloaded datasets) go in the private run store, with a `manifest.json` that records the commit, command, hardware, GPU and run time.
- **Runner outputs** from GitHub Actions are committed to the branch `compute/<name>` that ran them. Outputs that later work reads are kept, without that branch's history, on `outputs/<name>`: `outputs/codenames-embed` holds the Hugging Face vectors and `outputs/hanabi-h1` the H1 runs. A summary that a README quotes is copied into the game's `results/`.

## How runs are launched

Each experiment is one module, run as `uv run --group <game> python -m <game>.<module> --seed N --out DIR`. The module writes its results into `DIR` and does not know where it runs. Two launchers wrap it in the same four steps: check out a pinned commit, install the dependency group from `uv.lock`, run the module, and store `DIR` with a `manifest.json` written by `manifest.py`.

- **Hugging Face Jobs** (`hfrun.py`), for GPUs and for hosted language models (Inference Providers). Every job has a hard timeout, and outputs go to the private dataset. Hosted calls pin one provider with the router's suffix (`model:provider`) and log who served each reply; the route that the runs up to 2026-11-19 used, the provider in the URL path, refuses chat requests since 2026-11-20 (`codenames/results/llm-providers.md`).
- **GitHub Actions** (`.github/workflows/experiment.yml`), for CPU work that splits into seeds or slices. It is started by hand with the module, its arguments, a JSON list of seeds and a timeout; each seed is its own job, up to 20 at once. Outputs go to a branch named `compute/<experiment>-<run id>`, or to the private dataset when the repository has an `HF_TOKEN` secret. Runners are free for this public repository.
- **A local machine**, for anything that fits.

Costs are recorded per run. A number from a run that did not happen is never reported.

## Plans

| Id | Question | Compute | Plan | Status |
|---|---|---|---|---|
| C1 | Do the Codenames results on human guesses hold when the games, not the turns, are split? | CPU | [C1](plans/C1.md) | done (PR #41): the published split leaks; the linear adapter's gain holds |
| C2 | Does fine-tuning an embedding model (LoRA) on human games beat the linear adapter, on C1's split? | A10G, about $2.50 | [C2](plans/C2.md) | done (PR #42): null; no detectable difference from the linear adapter (−4.1 to 2.8 points) |
| C3 | Does a clue giver trained against several guessers do better with held-out guessers, across five seeds and fourteen held-out choices? | CPU (GitHub Actions) | [C3](plans/C3.md) | done (PR #48): yes; held-out value 0.31 (0.26 to 0.36) higher than self-play, in all 70 runs, at a cost of 0.50 with the giver's own model |
| C4 | Do hosted language models match human guesses better than embeddings, and how do they fare as givers with guessers they have never met? | Inference Providers, about $1 to $3 | [C4](plans/C4.md) | done (PR #43): language models guess better, give worse clues |
| C5 | Does reinforcement learning of a small language model, rewarded by a population of guessers, give clues that held-out guessers understand? | L4, about $9 | [C5](plans/C5.md) | not run: C4's giver test failed |
| C6 | When a hosted language model chooses among legal clues that an embedding giver proposes, do unseen guessers find more words than with the embedding giver's own choice? | Inference Providers, about $0.40 to $0.80; cap $1.50 | [C6](plans/C6.md) | done (PR #50): null; the chosen model's pick trails the embedding giver's first choice by 0.06 (−0.12 to −0.005) with strangers; prompt evolution not run |
| C7 | Does Qwen3.5-2B, trained by GRPO to choose among the same legal clues, beat the seven-feature policy with held-out guessers? | L4, about $4.10; cap $7 | [C7](plans/C7.md) | done (PR #49): null; the trained model always names the first listed clue for one word and trails the feature policy with held-out guessers (−0.11, interval −0.16 to −0.06); $3.74 |
| C8 | Do about 19,000 turns labeled by gpt-oss-120b raise the LoRA embedding model's match with human guesses? | Inference Providers and A10G, about $2.50 to $3.50; cap $5.50 | [C8](plans/C8.md) | done (PR #53): yes, by 3.3 points (0.9 to 5.9), level with the teacher; about $2.06 spent |
| C9 | Does Qwen3.5, fine-tuned on human turns and then trained by RL with the person's pick as reward, guess as people do more often than gpt-oss-120b? | L4 or L40S, about $7 to $8.50; cap $12 | [C9](plans/C9.md) | proposed, needs approval |
| C10 | When a hosted language model guesses on each of an embedding giver's legal clues and the giver picks the clue worth most to it, do unseen guessers find more words than with the embedding giver's first choice? | Inference Providers, about $0.50 to $1; cap $1.50 | [C10](plans/C10.md) | done: null; the listener's pick trails the first choice by 0.15 (−0.20 to −0.10) with strangers, and adds −0.02 (−0.03 to 0.001) to a population of six; $0.37 |
| C11 | Does C10's population pick, which beats the embedding giver's first choice by 0.24 words per turn with embedding strangers, still beat it with language-model guessers of two other families, and with people in a small pilot? | Inference Providers, about $0.25 to $0.50; cap $1; the human pilot is unpaid | [C11](plans/C11.md) | [Part A replay](../codenames/results/c11.md) on 998 paired boards. P − F was −0.020 (−0.096 to 0.055) under the two language models and 0.242 (0.160 to 0.324) smaller than under potion. Model calls cost $0.21. People have yet to play part B. |
| H1 | How many Hanabi deals reach 25 with every card visible? Decide the undecided deals and check with a second solver. | CPU (GitHub Actions) | [H1](plans/H1.md) | done (PR #45): 98.6% of two-player deals winnable with every card in view; two solvers agree on 79,396 deals |
| H2 | How far below the open-game ceiling do hosted language models play, and, at three players, below the hat strategy? | Inference Providers, about $2 to $5 | [H2](plans/H2.md) | done (PR #44): both models far below the ceiling; deduction help shows no effect; three-player part not run |
| H3 | In a two-color game, does Qwen3.5-2B trained by self-play score more with partners from other seeds and with a rule-based player? | L40S, about $10.40; cap $13.50 | [H3](plans/H3.md) | proposed, needs approval |
| H4 | In a two-color game, does an entropy bonus of 0.05 instead of 0.01 make independent PPO policies from different seeds compatible? | L4, about $4.60; cap $6.40 | [H4](plans/H4.md) | done (PR #59): yes, cross-play rises by 1.12 of 10 (0.83 to 1.41), but self-play still leads cross-play by 0.34; $2.21 on an A10G |
| P1 | Does the type-chart equilibrium predict which types people play? | CPU (GitHub Actions) | [P1](plans/P1.md) | done (PR #47): null; the number of Pokémon per typing predicts use better (Δρ = −0.32, interval −0.56 to −0.08) |
| C12 | With the whole clue rule, the costs, one retry and medium reasoning, does gpt-oss-120b still give clues worth less than the embedding giver's, to embedding strangers and to language models of other families? | Inference Providers, about $0.70; cap $1.20 (part B, DeepSeek-V3.2: about $0.40, cap $0.70) | [C12](plans/C12.md) | proposed, needs approval |
| C13 | Scored by language-model guessers, does C3's population giver beat its self-play giver and the count-1 rule? | Inference Providers, about $0.40; cap $0.70 | [C13](plans/C13.md) | proposed, needs approval; needs `cautious.py` from `fix/codenames` |
| C14 | C6 again, with reply caps that cut off under 5% of replies and strictly legal candidates | Inference Providers, about $0.30; cap $0.55 | [C14](plans/C14.md) | proposed, needs approval |
| C15 | Does gpt-oss-120b's agreement with people fall more than the embeddings' when Cultural Codes boards are shuffled and their other words swapped? | Inference Providers, about $0.40; cap $0.70 | [C15](plans/C15.md) | proposed, needs approval |
| H5 | H2's unrun three-player part: gpt-oss-120b against the information strategy, with the reply cap at 16,384 | Inference Providers, about $2.20; cap $3.60 | [H5](plans/H5.md) | proposed, needs approval |
| H6 | H4's gap between self-play and cross-play, compared at matched self-play, with checkpoints kept and 2.5 times the steps | A10G, about $5.25; cap $7.50 | [H6](plans/H6.md) | proposed, needs approval |
| M1 | Do Fallout screens of common words (wordfreq Zipf ≥ 3.0) change the share of passwords cracked within four attempts? | CPU (GitHub Actions) | [M1](plans/M1.md) | proposed |
| P2 | Does the 63-team full-battle equilibrium survive when both sides switch by poke-env's heuristic rule? | CPU (GitHub Actions) | [P2](plans/P2.md) | proposed |
| R1 | Does Maia-3's chance of finding each solution move lower the error of a predicted Lichess puzzle rating? | Actions, or T4 about $0.70; cap $1.00 | [R1](plans/R1.md) | proposed, needs approval for the T4 fallback |
| R2 | Do Maia3-23M and Maia3-79M estimate a rating from one game better than Maia3-5M? | T4, about $2.80; cap $5.00 | [R2](plans/R2.md) | proposed, needs approval |

Costs are estimates from the plans. Each plan has a hard cap, and the paid ones are approved one at a time.

**2026-11-20, proposed plans and amendments.** C12 to C15, H5, H6, M1, P2, R1 and R2 are new plans; R is the prefix for `chess_ratings/` and M for `mastermind/`. Three merged plans gain dated amendments:

- C4: its giving part rescored under the rulebook's clue rule, exploratory, with no paid calls.
- C11: part B's fixes to consent, storage, delivery and blinding, and a target of 12 volunteers.
- H3: its detectable effect, and the cost of five seeds (about $16.80, cap $19.80) against three ($10.40, cap $13.50).

The paid ones are approved one at a time, each under its own cap.

**2026-11-20, a methods amendment.** `METHOD.md` now asks each new plan for its minimum detectable effect at 80% power, five training seeds where they fit the cap, pinned providers with a repeat check per split, a limit on replies cut off at the token cap, and a check with people or with partners from another model family before a result with model partners is a headline. Three dated notes go with it, and none changes a published number:

- `codenames/results/llm-providers.md`: which providers served the calls of C4, C6, C8, C10 and H2. Every call named one provider, and no log shows another. C10's empty test replies came with doubled latency at the same provider, model id and prompt tokens.
- `codenames/results/guesser-calibration.md` (exploratory): calibration of the guessers on human turns, the turns with several picks, and agreement between people on shared words. People agree with each other 4.6 to 16.1 points more often than a guesser agrees with a person.
- `hanabi/results/h2-harness.md`: 24% of gpt-oss-120b's turns in H2 reached the 4,096-token cap and got a fallback move, nearly always a discard, so H2's numbers for that model are partly limited by the harness.

**Order.** C1, C3, H1 and P1 cost nothing and can run now, side by side. C2 and C4 use C1's split. H2 uses H1's verdict for each deal. C5 runs only if C4's second hypothesis holds. Two changes from the first version of this table:

- C3 no longer waits for C1. Its comparison is between givers scored by the same guessers, and C1 changes only the guessers' fitted temperatures, which are the same for both arms.
- C5 moves from an A100 to an L4 with a 0.6B model, and its reward is computed exactly from embedding guessers instead of by a judge model. H2 drops the pairing of a language model with partners that play the information strategy. Those partners read every hint as a code the model does not know.

## Round two

C6 to C9, H3 and H4 train or use models released in 2026, with post-training methods from 2025 and 2026. None has run. Each needs approval before its jobs, and the paid ones are approved one at a time. In order of what each result is worth for its cost:

| Rank | Plan | Expected | Cap | Why this place |
|---:|---|---:|---:|---|
| 1 | C6 | $0.40 to $0.80 | $1.50 | Separates C4's two causes of failure (illegal clues and poor choice) with no training. Sets the candidates and baselines for C7. |
| 2 | C8 | $2.50 to $3.50 | $5.50 | Tests whether the embedding guessers were short of data, the open question of C2, at a tenth of C9's cost. |
| 3 | C7 | $4.10 | $7 | The direct test of a trained small language model against the cheap policy, with an exact reward and held-out guessers. |
| 4 | H4 | $4.60 | $6.40 | Reproduces a 2025 result on coordination with strangers at small scale, with exact ceilings, and gives H3 a baseline. |
| 5 | H3 | $10.40 | $13.50 | Self-play training of a language model with cross-play as the headline. Higher risk: the training loop is new. |
| 6 | C9 | $7 to $8.50 | $12 | Passing gpt-oss-120b needs a gain of about 6 points over the best trained guesser, and the plan expects that to fail. Runs after C8. |

All six together are expected to cost about $29 to $32, with caps adding up to $45.90.

**2026-11-19, after C4 and C6: a new order.** The table above is the order as first written. Three changes:

- C10 is added and ranked first among the paid plans. C4 found that gpt-oss-120b guesses as people do more often than the embedding models, and C6 that no hosted model chooses among legal clues better than the embedding giver's first choice. C10 uses the model as a guesser inside the giver, the role in which it did well, for about $0.50 to $1.
- C9 is deferred. C4 and C6 give no sign that a small trained model will pass gpt-oss-120b, and C9's main test needs a top-1 of about 0.69 to pass it by a resolved margin.
- H3 is deferred until H4 reports. In H2, the open models scored 0 to 6.4 points of 25 in two-player games with deduction help. Language models start from weak play here, and H4 gives the two-color game a baseline that H3 needs first.

The paid plans now run in this order: C10, C8, C7, H4, then H3 and C9 if the earlier results call for them.

**Candidates from the request, and what became of them.**

- (a) a language model reranks legal clues: kept as C6. Two hosted models released in 2026 join C4's four, and a prompt found by GEPA is added as the baseline without training.
- (b) RL of a small language model as giver: kept as C7. It is now compared with the feature policy of `rl.py` on the same candidates, which isolates what the language model adds.
- (c) a small language model as guesser, SFT then RL: kept as C9, with RL on the teacher's picks as a second arm. It runs after C8.
- (d) distilling gpt-oss-120b into an embedding model: kept as C8. The teacher labels resampled boards with the human clues, because Cultural Codes has no more human turns.
- (e) Hanabi RL of a small language model: kept as H3, changed to a two-color game with MARSHAL's advantage per turn. The hat strategy is dropped as a partner, for H2's reason. A rule-based player and other seeds are the held-out partners.
- (f) IPPO with a high entropy bonus: kept as H4, on the same two-color game. The full game takes Forkel et al. more than 40 hours a seed on an L40S, which the budget does not cover.

**Models and methods chosen**, as read on 2026-11-19:

- Qwen3.5-2B and Qwen3.5-4B (Qwen, 2026-02-27 and 2026-02-28, Apache 2.0) for training. They are the newest dense Qwen models under 9B; Qwen3.6 and Qwen3.8 start at 27B. TRL's GRPO documentation lists Qwen3.5-2B as tested.
- Gemma 4 E2B and E4B (Google, 2026-03-02, Apache 2.0) as fallbacks.
- Qwen3.8-27B (2026-08-05) and Gemma 4 31B, both Apache 2.0, as hosted choosers in C6.
- TRL 1.18.1 (2026-11-17), whose GRPO trainer defaults to the DAPO loss with no KL term. The plans add DAPO's clip-higher bound. GSPO and GDPO are available in TRL; the plans say why they are not used.
- MARSHAL's advantage per turn and per seat (ICLR 2026) for H3's self-play, written outside TRL, whose trainer groups samples by prompt.
- JaxMARL 0.2.0 (2026-09-10, Apache 2.0) for H4.

Each plan lists its sources with dates. The prices are those of Hugging Face Jobs: T4 $0.40, L4 $0.80, A10G $1.00, L40S $1.80, A100 $2.50 and H200 $5 an hour. No plan needs a Docker image; H3 names the TRL image as a fallback.
