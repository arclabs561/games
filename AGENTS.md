# Working in this repository

Each game starts with a question a player might ask. Use an exact answer when one is feasible.

## Writing about a game

- Look for the opening farther down the page. If a worked example already answers the player's question, move it up and keep the author's wording.
- Give the reader something to try or compare. In [Egyptian Ratscrew](egyptian_ratscrew/README.md), deal nine cards. Either player can win the first slap, yet neither can win the game. In [Mastermind](mastermind/README.md), two totals tell the story. Across all 1,296 codes, guaranteeing a win in five guesses costs one extra guess.
- Keep the condition that changes the answer beside the claim. In [Hanabi](hanabi/README.md), a solver can see both hands and the draw pile. A player deciding what to discard cannot. In the root index, pick one finding per game and link to the details.
- Use the words players use. Introduce a technical term where it lets the reader say something more precisely. Write in American English and keep the tone dry. Cut sales pitches and words like “obviously,” “clearly” and “just.”
- Give a null result the same care as a positive one. Say what was compared and how large an effect the evidence allows.

## Setup and checks

```sh
uv sync --locked --all-groups --no-group codenames-gpu --no-group hanabi-jax
uv run ruff check . && uv run ruff format --check . && TY_UV=scripts uv run ty check && uv lock --check && uv run pytest
```

- **Notebooks:** `uv run pytest -p tests.nbval_figures --nbval --nbval-current-env --nbval-sanitize-with tests/nbval_sanitize.cfg <notebook>`. Save outputs with `uv run jupyter execute --inplace <notebook>` after a restart, never from a session with hidden state.
- **Papers:** `uv run tools/typst_vendor.py <doc>.typ` (add `--all` for the phone PDF and the HTML page). Committed PDFs rebuild byte for byte; CI checks this.
- **READMEs:** `readcost check <file>.md` passes for every README you touch.

## Layout

- One folder per game, run as `uv run python -m <game>.<module>` for packaged games. Erase uses scripts directly, for example `python3 erase/count.py --pairs`. Third-party packages go in the dependency group named after the folder; a test enforces this. A new game is added to `module-name` in `pyproject.toml` and gets a row in the root README.
- Tests are `tests/test_<game>*.py`. Code a notebook uses lives in a module that the notebook and the tests both import.
- Figure scripts draw inside `matplotlib.style.context("default")`, and no module changes matplotlib's global settings when imported. pytest imports every test module before running any, so a global change leaks into other games' byte-for-byte figure tests.
- Results that a README quotes are written by a script to the game's folder or its `results/`. A test checks that the committed file is what the script writes.
- For a README section marked `<!-- shared:name -->`, edit `<game>/docs/name.md`. Keep computed numbers as placeholders into `results.json`. Run `uv run tools/sync_docs.py <game>`, then rebuild the paper with `--all`. The paper reads the filled section from the README.
- Experiments with a cost or a learned model follow `experiments/METHOD.md`, with a plan merged before the run.

## Show how the answer is known

Try a hand proof first. Identify the smallest finite premise it needs and verify every case when feasible. Publish the command, rules and count. If enumeration is too costly, say why, then sample. Label a heuristic guess and name a counterexample that would refute it. Give the source for quoted results, including numbers that appear only in a paper. For case checks, provide certificates and a separate checker. For simulations, give the sample size, seed and interval. For observed games, name the dataset and say which games were counted.

Before rewriting a headline, try to break it with one legal move, deal or route. Start from the board, resource records or original replies. Name the observation that would change the answer. In EV Nova a single jump pays 337 credits per ton. Pricing the return load explains how a 217-per-jump loop can still be best.

Check a result a second way when you can. Try brute force on a small deck, replay a strategy against every code, or compare the total with a published count. If the counts disagree, report both.

## Attribution and data

- Name only the authors of cited sources. Commits, PRs and files carry no AI attribution, co-author trailers or session links; `.claude/settings.json` turns these off.
- Commit data only when its license allows it. Record each source and its license in the game's README, and say plainly when a license is unknown.

## Skills

Edit repo skills in `.claude/skills/`. Codex reads the same directory through `.agents/skills`.

| Skill | Use it for |
|---|---|
| `results-readme` | The shape of a game's README and its table of results |
| `pytudes-style-analysis` | Programs and notebooks that answer questions about a game |
| `game-solvability-analysis` | Exact counts, invariants, certificates and solvers, by kind of game |
| `mathematical-exposition` | Papers, proofs and notes, including ones a parent and child read together |
| `typst-paper-pipeline` | Building papers, the phone edition, and the page-by-page visual check |
| `preregistered-experiments` | Paid runs, learned models and language models |
