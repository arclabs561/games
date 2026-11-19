# Experiments

Most results in this repository are exact and run in seconds on a laptop. The experiments listed here need more: a GPU, a hosted language model, or hours of CPU. This folder says how they run and where their results live.

## Where things live

- **Code** is in this repository. A run names the commit it used, so it can be repeated.
- **Plans** are in `plans/<id>.md`, one per experiment. Each names its question, metric, seeds and held-out partners. A plan is merged before its runs and is not changed after; a later change goes in a dated amendment at its end. `METHOD.md` gives the rules all plans share.
- **Summaries** (small JSON files with every number a README quotes) go in the game's own `results/` folder, committed with the README that uses them.
- **Artifacts** that are large or whose license forbids publishing (vectors, fine-tuned adapters, model outputs, downloaded datasets) go in the private Hugging Face dataset `arclabs561/games-runs`, under `runs/<experiment>/<run>/`, with a `manifest.json` that records the commit, command, hardware, GPU and run time.
- **Runner outputs** from GitHub Actions are committed to the branch `compute/<name>` that ran them, as `compute/codenames-embed` does for the Hugging Face vectors. A summary that a README quotes is copied into the game's `results/`.

## How runs are launched

Each experiment is one module, run as `uv run --group <game> python -m <game>.<module> --seed N --out DIR`. The module writes its results into `DIR` and does not know where it runs. Two launchers wrap it in the same four steps: check out a pinned commit, install the dependency group from `uv.lock`, run the module, and store `DIR` with a `manifest.json` written by `manifest.py`.

- **Hugging Face Jobs** (`hfrun.py`), for GPUs and for hosted language models (Inference Providers). Every job has a hard timeout, and outputs go to the private dataset.
- **GitHub Actions** (`.github/workflows/experiment.yml`), for CPU work that splits into seeds or slices. It is started by hand with the module, its arguments, a JSON list of seeds and a timeout; each seed is its own job, up to 20 at once. Outputs go to a branch named `compute/<experiment>-<run id>`, or to the private dataset when the repository has an `HF_TOKEN` secret. Runners are free for this public repository.
- **A local machine**, for anything that fits.

Costs are recorded per run. A number from a run that did not happen is never reported.

## Plans

| Id | Question | Compute | Plan | Status |
|---|---|---|---|---|
| C1 | Do the Codenames results on human guesses hold when the games, not the turns, are split? | CPU | [C1](plans/C1.md) | done (PR #41): the published split leaks; the linear adapter's gain holds |
| C2 | Does fine-tuning an embedding model (LoRA) on human games beat the linear adapter, on C1's split? | A10G, about $2.50 | [C2](plans/C2.md) | done (PR #42): null; LoRA ties the linear adapter |
| C3 | Does a clue giver trained against several guessers do better with held-out guessers, across five seeds and fourteen held-out choices? | CPU (GitHub Actions) | [C3](plans/C3.md) | done (PR #48): yes; held-out value 0.31 (0.26 to 0.36) higher than self-play, in all 70 runs, at a cost of 0.50 with the giver's own model |
| C4 | Do hosted language models match human guesses better than embeddings, and how do they fare as givers with guessers they have never met? | Inference Providers, about $1 to $3 | [C4](plans/C4.md) | done (PR #43): language models guess better, give worse clues |
| C5 | Does reinforcement learning of a small language model, rewarded by a population of guessers, give clues that held-out guessers understand? | L4, about $9 | [C5](plans/C5.md) | not run: C4's giver test failed |
| C6 | When a hosted language model chooses among legal clues that an embedding giver proposes, do unseen guessers find more words than with the embedding giver's own choice? | Inference Providers, about $0.40 to $0.80; cap $1.50 | [C6](plans/C6.md) | done (PR #PRNUM): null; the chosen model's pick trails the embedding giver's first choice by 0.06 (−0.12 to −0.005) with strangers; prompt evolution not run |
| C7 | Does Qwen3.5-2B, trained by GRPO to choose among the same legal clues, beat the seven-feature policy with held-out guessers? | L4, about $4.10; cap $7 | [C7](plans/C7.md) | proposed, needs approval |
| C8 | Do about 19,000 turns labeled by gpt-oss-120b raise the LoRA embedding model's match with human guesses? | Inference Providers and A10G, about $2.50 to $3.50; cap $5.50 | [C8](plans/C8.md) | proposed, needs approval |
| C9 | Does Qwen3.5, fine-tuned on human turns and then trained by RL with the person's pick as reward, guess as people do more often than gpt-oss-120b? | L4 or L40S, about $7 to $8.50; cap $12 | [C9](plans/C9.md) | proposed, needs approval |
| H1 | How many Hanabi deals reach 25 with every card visible? Decide the undecided deals and check with a second solver. | CPU (GitHub Actions) | [H1](plans/H1.md) | done (PR #45): 98.6% of two-player deals winnable with every card in view; two solvers agree on 79,396 deals |
| H2 | How far below the open-game ceiling do hosted language models play, and, at three players, below the hat strategy? | Inference Providers, about $2 to $5 | [H2](plans/H2.md) | done (PR #44): both models far below the ceiling; deduction help shows no effect; three-player part not run |
| H3 | In a two-color game, does Qwen3.5-2B trained by self-play score more with partners from other seeds and with a rule-based player? | L40S, about $10.40; cap $13.50 | [H3](plans/H3.md) | proposed, needs approval |
| H4 | In a two-color game, does an entropy bonus of 0.05 instead of 0.01 make independent PPO policies from different seeds compatible? | L4, about $4.60; cap $6.40 | [H4](plans/H4.md) | proposed, needs approval |
| P1 | Does the type-chart equilibrium predict which types people play? | CPU (GitHub Actions) | [P1](plans/P1.md) | done (PR #47): null; the number of Pokémon per typing predicts use better (Δρ = −0.32, interval −0.56 to −0.08) |

Costs are estimates from the plans. Each plan has a hard cap, and the paid ones are approved one at a time.

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
