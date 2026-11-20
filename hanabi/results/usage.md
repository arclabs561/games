# Usage

Every command, with times. The [Hanabi README](../README.md) gives the short answers.


The rules engine, the solvers, the information strategy and the analysis of the human games use only the standard library. `solver_cp.py` needs OR-Tools (`uv run --with ortools ...`); `hosted.py` and the download of the human games need the `hanabi` dependency group (`huggingface_hub`); plan H4 needs JAX and `jaxmarl` (the `hanabi-jax` group, a CUDA build). Times are from one run on a two-core computer.

```bash
uv run python -m hanabi.information -n 2000               # five players, about 1 min
uv run python -m hanabi.information -n 100000 --hansim-decks   # the 100,000-game row (35 min)
uv run python -m hanabi.information -n 10000 --cheat --hansim-decks
uv run python -m hanabi.information -n 10000 --players 3 4
uv run python -m hanabi.hansim_check -n 10000            # needs git and g++
uv run python -m hanabi.upper_bound --exact --check 200   # small decks (about 5 min)
uv run python -m hanabi.upper_bound --sample 2 2000       # about 1 min; 4 or 5 players take 2
uv run python -m hanabi.upper_bound --sample 5 0 --budget 300000 --seeds 474 545   # about 1 min each
uv run python -m hanabi.upper_bound --exact --second    # Solver B on the small decks too
# plan H1, one block of 2,000 seeds (the run used 40 blocks on GitHub Actions)
uv run python -m hanabi.upper_bound --sample 2 2000 --start 4000 --solver both --json p2-4000.json
uv run python -m hanabi.upper_bound --sample 3 0 --seeds 1144 --rerun --budgets 5000000 --json p3-rerun.json
uv run python -m hanabi.h1 --runs <folder of blocks> --out hanabi/results/h1.json --all-rows
uv run python -m hanabi.upper_bound --sample 5 20000 --budget 300000   # a bigger run, hours
uv run --group hanabi python -m hanabi.h2 play --model openai/gpt-oss-120b --provider novita \
    --reasoning-effort medium --seeds 0-29 --deduction on --max-tokens 4096 --cap 2 --out OUT
uv run python -m hanabi.h2 report --runs hanabi/results/h2-runs --out hanabi/results/h2.json
```

`h2 play` needs `HF_TOKEN` and network access to Hugging Face; the runs above were launched with `experiments/hfrun.py` on Hugging Face Jobs. `h2 report` runs locally in a few seconds.

The human games need the `hanabi` group to download the dataset; the analysis itself uses the standard library:

```bash
uv run --group hanabi python -m hanabi.humans_data fetch     # download, replay every game, check the scores
uv run python -m hanabi.humans run --players 2 --part 0 --parts 30 --out OUT   # one of the 60 blocks of the run
uv run python -m hanabi.humans report --runs <folder of blocks>                # results/humans.json and the per-game records
uv run python -m hanabi.humans figure                                          # results/humans-ceiling.svg
uv run python -m hanabi.ceiling --players 2 --seed 161                         # one deck's ceiling, under a second
uv run python -m hanabi.ceiling --json game.json --line                        # a hanab.live export
```

Plan H4 needs the `hanabi` group (JAX, Flax, Optax and JaxMARL; CPU builds) and, on a GPU, `hanabi-jax`, which adds JAX's CUDA libraries:

```bash
uv run --group hanabi python -m hanabi.jax_bridge --games 1000     # the agreement test, about 1 min
uv run --group hanabi python -m hanabi.ippo smoke                  # a tiny run of everything, about 1 min
uv run python -m hanabi.h4 ceilings                                 # the ceilings of the test deals, 20 s
# on a GPU (the runs used hfrun.py with GROUP=hanabi-jax on an a10g-small)
uv run --group hanabi-jax python -m hanabi.ippo train --ent-coef 0.05 --seeds 1-8 --steps 2e8 --deadline-minutes 95 --out OUT
uv run --group hanabi-jax python -m hanabi.ippo evaluate --runs hf:runs/h4/arm-0.01 hf:runs/h4/arm-0.05 --deals 400000-401999 --out OUT
uv run python -m hanabi.h4 check hanabi/results/h4.json <sha256>    # the copied summary matches the job's
```
