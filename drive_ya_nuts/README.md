## Drive-Ya-Nuts

`solver.py` solves the original puzzle and can sample or generate random 7-piece sets. Run it from the repository root:

### Quickstart

- **Solve original** (default):

```bash
uv run python -m drive_ya_nuts.solver
```

- **Estimate distribution** (union bound over all \(k \in [0, 7!]\)):

```bash
uv run python -m drive_ya_nuts.solver --no-original --distribution --m 5041 --seed 0 --progress-every 10000
```

- **Generate a unique-solution set**:

```bash
uv run python -m drive_ya_nuts.solver --no-original --find-k 1 --seed 0 --max-tries 100000 --print-found 1
```
