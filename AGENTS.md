# Working in this repository

Each game is a package that answers questions a player would ask, with exact results where they are feasible. This file holds the conventions every change follows. The skills in `.claude/skills/` hold the methods.

## Setup and checks

```sh
uv sync --all-groups --no-group codenames-gpu --no-group hanabi-jax   # the two groups left out are CUDA builds for GPU jobs
uv run ruff check . && uv run ruff format --check . && TY_UV=scripts uv run ty check && uv lock --check && uv run pytest
```

- **Notebooks:** `uv run pytest -p tests.nbval_figures --nbval --nbval-current-env --nbval-sanitize-with tests/nbval_sanitize.cfg <notebook>`. Save outputs with `uv run jupyter execute --inplace <notebook>` after a restart, never from a session with hidden state.
- **Papers:** `uv run tools/typst_vendor.py <doc>.typ` (add `--all` for the phone PDF and the HTML page). Committed PDFs rebuild byte for byte; CI checks this.
- **READMEs:** `readcost check <file>.md` passes for every README you touch.

## Layout

- One folder per game, run as `uv run python -m <game>.<module>`. Third-party packages go in the dependency group named after the folder; a test enforces this. A new game is added to `module-name` in `pyproject.toml` and gets a row in the root README.
- Tests are `tests/test_<game>*.py`. Code a notebook uses lives in a module that the notebook and the tests both import.
- Figure scripts draw inside `matplotlib.style.context("default")`, and no module changes matplotlib's global settings when imported. pytest imports every test module before running any, so a global change leaks into other games' byte-for-byte figure tests.
- Results that a README quotes are written by a script to the game's folder or its `results/`. A test checks that the committed file is what the script writes.
- Text that a README and a paper both show (results table, rules, checks, open questions) lives once in `<game>/docs/*.md`, with numbers as placeholders into `results.json`; `uv run tools/sync_docs.py` fills the README, the paper includes the same sections, and a test fails on drift.
- Experiments with a cost or a learned model follow `experiments/METHOD.md`, with a plan merged before the run.

## Every number says how it is known

Each result is labeled as one of the following:
- proved by hand;
- checked by computer case by case (with certificates and a separate checker);
- computed exactly;
- simulated, with its sample size, seed and interval;
- or taken from a cited source.

A result is checked by a second method where one exists. Examples are brute force on small cases, an independent solver, and a published count. Disagreement with a published result is reported as a finding.

## Writing

- READMEs and papers are humble, dry and in American spelling, with no marketing words and no "obviously", "clearly" or "just".
- A README opens with what was found, in a few lines, before any detail. Null and negative results get the same detail as positive ones.
- Name only the authors of cited sources. Commits, PRs and files carry no AI attribution, co-author trailers or session links; `.claude/settings.json` turns these off.
- Commit data only when its license allows it. Record each source and its license in the game's README, and say plainly when a license is unknown.

## Skills

| Skill | Use it for |
|---|---|
| `results-readme` | The shape of a game's README and its table of results |
| `pytudes-style-analysis` | Programs and notebooks that answer questions about a game |
| `game-solvability-analysis` | Exact counts, invariants, certificates and solvers, by kind of game |
| `mathematical-exposition` | Papers, proofs and notes, including ones a parent and child read together |
| `typst-paper-pipeline` | Building papers, the phone edition, and the page-by-page visual check |
| `preregistered-experiments` | Paid runs, learned models and language models |
