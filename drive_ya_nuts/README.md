# Drive Ya Nuts

Seven hexagonal nuts each carry the numbers 1 to 6 around their edges. Place one in the middle and six around it so that every pair of touching edges shows the same number. An exhaustive search finds exactly one solution for the original nuts, and the tests check the search against brute force on random sets of nuts.

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

### Sources

- Jaap Scherphuis, [Circus Seven / Drive Ya Nuts](http://www.jaapsch.net/puzzles/circus.htm): the puzzle and its relatives.
- Stan Isaacs, [drive ya nuts](http://www.math.rwth-aachen.de/~Martin.Schoenert/Cube-Lovers/Stan_Isaacs__drive_ya_nuts.html), Cube Lovers: the numbers on the original nuts.
- M. Zarecky and A. Parker, [Describing a combinatorics problem with a system of polynomial equations](http://www.wittenberg.edu/sites/default/files/media/computer_science/forms/sumInSMC.pdf).
- J. Sawada, [A fast algorithm to generate necklaces with fixed content](http://www.cis.uoguelph.ca/~sawada/papers/alph.pdf), *Theoretical Computer Science* 301 (2003).
