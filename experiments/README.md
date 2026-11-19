# Experiments

Most results in this repository are exact and run in seconds on a laptop. The experiments listed here need more: a GPU, a hosted language model, or hours of CPU. This folder says how they run and where their results live.

## Where things live

- **Code** is in this repository. A run names the commit it used, so it can be repeated.
- **Plans** are written here before a run: the question, the metric, the seeds and the held-out partners. They are not changed after the results arrive.
- **Summaries** (small JSON files with every number a README quotes) go in the game's own `results/` folder, committed with the README that uses them.
- **Artifacts** that are large or whose license forbids publishing (vectors, fine-tuned adapters, model outputs, downloaded datasets) go in the private Hugging Face dataset `arclabs561/games-runs`, under `runs/<experiment>/<run>/`, with a `manifest.json` that records the commit, command, hardware, GPU and run time.

## How runs are launched

- **Hugging Face Jobs**, for GPUs and for hosted language models (Inference Providers). `hfrun.py` clones this repository at a pinned commit, installs one dependency group, runs a module, and uploads the output folder with its manifest. Every job has a hard timeout.
- **GitHub Actions**, for CPU work that splits into many independent pieces. Runners are free for this public repository, and a matrix of jobs runs them side by side.
- **A local machine**, for anything that fits.

Costs are recorded per run. A number from a run that did not happen is never reported.

## Plans

| Id | Question | Compute | Status |
|---|---|---|---|
| C1 | Do the Codenames rankings hold when the human games are split by game rather than by turn? | CPU | planned |
| C2 | Does fine-tuning an embedding model (LoRA) on human games beat the 66.2% match of the linear adapter? | A10G | planned, after C1 |
| C3 | Does training a clue giver against several guessers shrink the self-play gap across five seeds? | CPU | planned, after C1 |
| C4 | Do hosted language models match human guesses better than embeddings, and how do they fare as givers with guessers they have never met? | Inference Providers | planned |
| C5 | Does reinforcement learning on a small language model give clues that held-out guessers understand? | A100 | only if C4 favors language-model givers |
| H1 | How many Hanabi deals reach 25 with every card visible? Decide the undecided deals and check with a second solver. | CPU | planned |
| H2 | How far below the hat strategy and that ceiling do hosted language models play? | Inference Providers | planned |
| P1 | Does the type-chart equilibrium predict which types people play? | CPU | planned |
