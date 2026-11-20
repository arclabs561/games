---
name: "preregistered-experiments"
description: "Running paid or learned-model experiments in this repository: plan first, jobs on Hugging Face or GitHub runners, cost caps, restart safety, behavior audits, and the failure modes met so far."
---

# Preregistered experiments

The rules are in `experiments/METHOD.md`: plan merged before the run, one factor at a time, paired designs, held-out partners, test sets used once, bootstrap intervals, and nulls reported. This skill adds the practice that made those rules work, and the ways runs went wrong.

## Running

- Launch with `experiments/hfrun.py` at a pushed commit. It clones that commit, installs one dependency group, runs one module and writes a manifest (commit, hardware, return code, seconds). Outputs go to the private dataset, and summaries go to the game's `results/`.
- Free CPU work that splits into pieces runs as a matrix on GitHub Actions (`.github/workflows/experiment.yml`). Paid GPU work runs on Hugging Face Jobs.
- **Pilot first.** One short job measures speed and setup time, and the main runs are sized from it to fit the cap. Report setup time against run time.
- **Cost.** Each plan states an expected cost and a hard cap. Every job gets a timeout. Language-model calls are metered as the provider's reported tokens times the listed price, because providers return tokens but not cost.
- **Hardware queues.** A cheap GPU can wait over an hour. Switch hardware by a dated amendment written before the main runs, not after.

## Restart safety

- Before relaunching anything, list the jobs and check the dataset for outputs that already exist.
- Never rerun a test job whose outputs exist, since the test set is scored once. If a job failed after sending calls, record how many were sent, and score only the rerun.
- Commit work in small pieces, so a lost session loses little.

## Checks before believing a result

- **Agreement test.** A learning environment must agree with the repository's own rules engine on legal moves and scores over thousands of random games before any training. Replay the evaluation games in the reference engine.
- **Behavior audit.** Report what the agents do, not only their scores: who hints, plays and discards, and how often an option is picked by its position in a list.
- **Shortcuts.** A policy trained to pick from a list learned to pick the first item. Shuffle option order, and check the share of picks by position.
- **Repeat check.** Send a sample of prompts again at temperature 0. Report how often the action changed; providers are not deterministic.
- **Empty and malformed replies.** Count them per split. A provider returned empty replies on 28% of test calls and 0.05% of validation calls a few hours earlier, so recheck the result on the boards with no empty reply.
- **Sensitivity.** Rerun the analysis with key constants halved and doubled. Analyses added after seeing the test scores are labeled exploratory.

## Seeds and random numbers

- Common random numbers are right for comparing arms: the same boards, deals and seeds give a paired difference with a smaller interval.
- They are wrong when simulations are meant as independent checks of different exact values. Three Tenzi simulations seeded game i with i, so they were not three checks; each now has its own seed.
- Never choose a seed by its result. Use fixed seeds (0, or 1 to 5) and print them.

## Data terms

- Commit aggregates only. Per-item strings derived from a dataset whose license is unclear can reconstruct it, and a pushed commit stays reachable by its SHA even after a rewrite.
- Record the license of every dataset and model in the README before the first run.
