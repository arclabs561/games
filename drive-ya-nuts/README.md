## Drive-Ya-Nuts

`solver.py` solves the original puzzle and can sample/generate random 7-piece sets.

### Quickstart

- **Solve original** (default):

```bash
python3 solver.py
```

- **Estimate distribution** (union bound over all \(k \in [0, 7!]\)):

```bash
python3 solver.py --no-original --distribution --m 5041 --seed 0 --progress-every 10000
```

- **Generate a unique-solution set**:

```bash
python3 solver.py --no-original --find-k 1 --seed 0 --max-tries 100000 --print-found 1
```
