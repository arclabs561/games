# Experiments

Most results in this repository are exact and run in seconds on a laptop. The experiments listed here need more: a GPU, a hosted language model, or hours of CPU. This folder says how they run and where their results live.

## Where things live

- **Code** is in this repository. A run names the commit it used, so it can be repeated.
- **Plans** are in `plans/<id>.md`, one per experiment. Each names its question, metric, seeds and held-out partners. A plan is merged before its runs and is not changed after; a later change goes in a dated amendment at its end. `METHOD.md` gives the rules all plans share.
- **Summaries** (small JSON files with every number a README quotes) go in the game's own `results/` folder, committed with the README that uses them.
- **Artifacts** that are large or whose license forbids publishing (vectors, fine-tuned adapters, model outputs, downloaded datasets) go in the private Hugging Face dataset `arclabs561/games-runs`, under `runs/<experiment>/<run>/`, with a `manifest.json` that records the commit, command, hardware, GPU and run time.
- **Runner outputs** from GitHub Actions are committed to the branch `compute/<name>` that ran them, as `compute/codenames-embed` does for the Hugging Face vectors. A summary that a README quotes is copied into the game's `results/`.

## How runs are launched

- **Hugging Face Jobs**, for GPUs and for hosted language models (Inference Providers). `hfrun.py` clones this repository at a pinned commit, installs one dependency group, runs a module, and uploads the output folder with its manifest. Every job has a hard timeout.
- **GitHub Actions**, for CPU work that splits into many independent pieces. Runners are free for this public repository, and a matrix of jobs runs them side by side.
- **A local machine**, for anything that fits.

Costs are recorded per run. A number from a run that did not happen is never reported.

## Plans

| Id | Question | Compute | Plan | Status |
|---|---|---|---|---|
| C1 | Do the Codenames results on human guesses hold when the games, not the turns, are split? | CPU | [C1](plans/C1.md) | planned |
| C2 | Does fine-tuning an embedding model (LoRA) on human games beat the linear adapter, on C1's split? | A10G, about $2.50 | [C2](plans/C2.md) | planned, after C1 |
| C3 | Does a clue giver trained against several guessers do better with held-out guessers, across five seeds and fourteen held-out choices? | CPU (GitHub Actions) | [C3](plans/C3.md) | planned |
| C4 | Do hosted language models match human guesses better than embeddings, and how do they fare as givers with guessers they have never met? | Inference Providers, about $1 to $3 | [C4](plans/C4.md) | planned, after C1 |
| C5 | Does reinforcement learning of a small language model, rewarded by a population of guessers, give clues that held-out guessers understand? | L4, about $9 | [C5](plans/C5.md) | only if C4 favors language-model givers |
| H1 | How many Hanabi deals reach 25 with every card visible? Decide the undecided deals and check with a second solver. | CPU (GitHub Actions) | [H1](plans/H1.md) | planned |
| H2 | How far below the open-game ceiling do hosted language models play, and, at three players, below the hat strategy? | Inference Providers, about $2 to $5 | [H2](plans/H2.md) | planned, after H1 |
| P1 | Does the type-chart equilibrium predict which types people play? | CPU (GitHub Actions) | [P1](plans/P1.md) | planned |

Costs are estimates from the plans. Each plan has a hard cap, and the paid ones are approved one at a time.

**Order.** C1, C3, H1 and P1 cost nothing and can run now, side by side. C2 and C4 use C1's split. H2 uses H1's verdict for each deal. C5 runs only if C4's second hypothesis holds. Two changes from the first version of this table:

- C3 no longer waits for C1. Its comparison is between givers scored by the same guessers, and C1 changes only the guessers' fitted temperatures, which are the same for both arms.
- C5 moves from an A100 to an L4 with a 0.6B model, and its reward is computed exactly from embedding guessers instead of by a judge model. H2 drops the pairing of a language model with information-strategy partners, which read every hint as a code the model does not know.
