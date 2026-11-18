# Drive Ya Nuts

Seven hexagonal nuts each carry the numbers 1 to 6 around their edges. Place one in the middle and six around it so that every pair of touching edges shows the same number. An exhaustive search finds exactly one solution for the original nuts, and the tests check the search against brute force on random sets of nuts.

`solver.py` solves the original puzzle and can sample or generate random 7-piece sets. `exact.py` counts the solutions of every set of seven distinct nuts. Run both from the repository root:

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

- **Count every set exactly** (about three minutes of CPU time and 2 GB of memory). `--write` rewrites `exact_counts.csv`, and `--check 3000` also solves 3,000 random sets one at a time and compares:

```bash
uv run --group drive_ya_nuts python -m drive_ya_nuts.exact
```

### How lucky was the designer?

There are 120 distinct nuts (the orders of 1 to 6, up to rotation) and C(120, 7) = 59,487,568,920 sets of seven. Exact counts of their solutions, from `exact.py`:

| Solutions | Sets | Share |
|---:|---:|---:|
| 0 | 55,439,557,440 | 93.195% |
| 1 | 3,180,020,160 | 5.346% |
| 2 | 662,739,120 | 1.114% |
| 3 | 140,663,280 | 0.236% |
| 4 | 44,263,560 | 0.074% |
| 5 to 9 | 20,061,600 | 0.034% |
| 10 to 20 | 263,760 | 0.0004% |

The full table is in [`exact_counts.csv`](exact_counts.csv). No set has more than 20 solutions, and 360 sets have exactly 20.

- 4,048,011,480 sets (6.805%) can be solved.
- 3,180,020,160 sets (5.346%, about 1 in 18.7) have exactly one solution, as the original does.
- Among the sets that can be solved, 78.56% have exactly one solution.
- A set has 2,558,824 / 29,160,573 = 0.0877 solutions on average.

So the original's single solution is not much of a coincidence once the set is solvable at all. The rarer property is solvability, which about one random set in 15 has. This assumes the designer drew seven distinct nuts at random. Nothing is known here about how the nuts were in fact chosen.

#### Method

Counting set by set is out of reach, so `exact.py` counts arrangements and groups them by set. Call a **ring** a center nut with six outer nuts whose touching edges all match. A set's solution count is the number of rings that its seven nuts make.

1. Any permutation of the numbers 1 to 6 renames rings to rings and keeps the count of every set. Any nut can be renamed to C0 = (1, 2, 3, 4, 5, 6), so it is enough to list the rings around C0.
2. Let e_i be the number on the edge shared by outer nuts i and i + 1. The outer nut in direction i then reads i + 1, e_{i-1}, three other numbers in any of 3! orders, and e_i. Each e_i must differ from i + 1, from i + 2 and from e_{i-1}, and 1,144 of the 6^6 sequences do. That gives 1,144 × 6^6 = 53,374,464 candidate rings around C0, of which 43,500,008 have seven distinct nuts.
3. A ring with another center C and C0 among its outer nuts is a renamed ring around C0. For each C, the code renames the rings around C0 that contain the nut that becomes C0. This lists every ring whose set contains C0, 7 × 43,500,008 in all.
4. Sorting those rings by set gives the count of every set that contains C0. Each nut is in the same number of sets with k solutions, so the number of all such sets is 120 / 7 times the number that contain C0.

#### Checks

- The same construction works for nuts with n sides. For four and five sides, the tests compare it with a direct count of every ring around every center, with no renaming, and with brute force on each set (all 6 sets for four sides, 300 random sets of the 134,596 for five sides).
- For six sides, the total number of solutions, 5,220,000,960, equals 120 × 43,500,008, and every count of sets that contain C0 is divisible by 7, as step 4 requires. The tests check the first.
- With `--check 3000`, 3,000 sets that contain C0 were solved one at a time with `solver.py`. Half were drawn uniformly and half from the solvable sets. All 3,000 counts matched. The tests run the same check for five sides.
- The exact shares fall inside the 95% Wilson intervals of random samples. With `--distribution --trials 5000 --seed 0`, 6.84% of sets were solvable (interval 6.17% to 7.57%) and 5.42% had one solution (4.83% to 6.08%). An earlier sample of 5,000 sets gave 7.5% and 5.8%, whose intervals are 6.80% to 8.26% and 5.19% to 6.48%.
- With `--trials 100000 --seed 0`, the shares were 6.825% solvable (interval 6.670% to 6.983%), 5.393% with one solution (5.255% to 5.535%) and 1.081% with two (1.019% to 1.147%). The exact shares, 6.805%, 5.346% and 1.114%, fall inside all three.

### Sources

- Jaap Scherphuis, [Circus Seven / Drive Ya Nuts](http://www.jaapsch.net/puzzles/circus.htm): the puzzle and its relatives.
- Stan Isaacs, [drive ya nuts](http://www.math.rwth-aachen.de/~Martin.Schoenert/Cube-Lovers/Stan_Isaacs__drive_ya_nuts.html), Cube Lovers: the numbers on the original nuts.
- M. Zarecky and A. Parker, [Describing a combinatorics problem with a system of polynomial equations](http://www.wittenberg.edu/sites/default/files/media/computer_science/forms/sumInSMC.pdf).
- J. Sawada, [A fast algorithm to generate necklaces with fixed content](http://www.cis.uoguelph.ca/~sawada/papers/alph.pdf), *Theoretical Computer Science* 301 (2003).
