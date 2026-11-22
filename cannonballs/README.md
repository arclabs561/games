# Cannonballs

<!-- shared:summary -->

Spread out a square pile of 24 layers and the 4,900 cannonballs make a flat square, 70 on a side. Watson proved in 1918 that the only other square pile you can lay out this way is a single ball. A search below 10<sup>10,000</sup> layers recovers both.

<!-- /shared -->

![4,900 balls as a square pile of 24 layers and as a flat square of 70 by 70](pile.svg)

The same 4,900 balls, stacked as a square pile of 24 layers and laid flat as a square of 70 by 70.

The write-up is [`cannonballs.pdf`](cannonballs.pdf), with a narrow edition for phones in [`cannonballs-phone.pdf`](cannonballs-phone.pdf). It is meant for a parent and a young child to read together. Its first six sections use only pictures and arithmetic, and the rest proves each claim. Lucas's problem asks which square piles hold a square number of balls. The section "Why only 1 and 24" proves the first steps of its solution, with its six cases and the walk along two Pell equations. The section "Oblong piles that are squares" proves the rule for oblong piles.

## Piles as modeled

<!-- shared:rules -->

- A **triangle** of side $n$ has rows of 1, 2, ..., $n$ balls. A **square** of side $n$ has $n$ rows of $n$.
- A **tetrahedral pile** of $n$ layers is a stack of triangles of side 1 to $n$, and a **square pile** a stack of squares of side 1 to $n$. Each ball above the bottom rests in the hollow between three or four balls below.
- An **oblong pile** on an $a$ by $b$ base, with $a \le b$, has layers $a$ by $b$, $a - 1$ by $b - 1$, and so on, up to a ridge of $b - a + 1$ balls.
- The balls sit at points of the face-centered cubic lattice. Two balls touch when their centers are one diameter apart.

<!-- /shared -->

## Results

Exact unless marked.

<!-- shared:results -->

| Question | Answer | How it is known |
|---|---|---|
| Balls in a square pile of $n$ layers | $n(n + 1)(2n + 1)/6$; 4,900 for 24 layers | Proved<!--raw-typst  (@prop-pyramid)-->; checked by building piles of up to 12 layers ball by ball |
| Balls in a tetrahedral pile | $n(n + 1)(n + 2)/6$, the number of ways to choose 3 things from $n + 2$ | Proved<!--raw-typst  (@prop-tetra)-->; checked the same way |
| Balls in an oblong pile | $a(a + 1)(3b - a + 1)/6$ | Proved<!--raw-typst  (@prop-oblong)-->; checked the same way |
| Square piles that are squares | 1 and 24 layers | Cited: Watson (1918)<!--raw-typst  (@thm-watson)-->. As checks, Sage's list of the whole-number points of the elliptic curve gives the same two (`curves.py`), a walk along two Pell equations finds no other below 10<sup>10,000</sup> layers, and a plain search agrees up to 10,000,000 |
| Square piles that are cubes | Only the single ball | Proved<!--raw-typst  (@prop-cubes)-->, using Euler's theorem that $x^3 + y^3 = 2z^3$ only when $x = y$ or $x = -y$; Bennett (2001) gives it too. A search agrees up to 10<sup>18</sup> layers |
| Oblong piles that are squares | Some length works for short side $a$ exactly when $a$ leaves remainder 1 on division by 3, or the multiple of 3 among $a$ and $a + 1$, divided by 3, is $9^k(3j + 2)$. The rule holds for 7/12 of all short sides in the long run | Proved<!--raw-typst  (@thm-oblong, @cor-share)-->; checked against two other methods for every $a$ up to 300 |
| Oblong piles on an $a$ by $a + c$ base, of two or more layers, that are squares, for $c$ from 1 to 20 | Some exist for $c$ = 2, 3, 5, 7, 10, 13, 15, 16 and 17, and none for the other $c$; for $c = 2$ they have 7, 14 and 168 layers (all lists are in `results.json`) | Computed exactly as the whole-number points of one elliptic curve for each $c$, with Sage (`curves.py`)<!--raw-typst  (@tab-shapes)-->; a search of every $a$ up to 100,000 agrees |
| Oblong piles on an $a$ by $2a - 1$ base that are squares | 1, 6 and 49 layers | Cited: an answer to Puzzling Stack Exchange question 99084; recomputed the same way with Sage, and a search up to 100,000 layers agrees |
| Square piles that are triangles | 1, 5, 6 and 85 layers | Cited: Finkelstein and London (1972)<!--raw-typst  (@tab-other)-->, OEIS A053611; reproduced by search up to 1,000,000 layers |
| Tetrahedral piles that are squares | 1, 2 and 48 layers | Cited: Meyl (1878); reproduced by search up to 1,000,000 layers |
| Tetrahedral piles that are triangles | 1, 3, 8, 20 and 34 layers | Cited: Avanesov (1967), OEIS A224421; reproduced by search up to 1,000,000 layers |
| Piles with the top cut off (runs of squares) that are squares | Possible for 1, 2, 11, 23, 24 and 26 layers, and no other number up to 30 | Search, with a remainder test that rules out each other case; matches OEIS A001032<!--raw-typst  (@tab-truncated)-->. For 25 layers the only run is $0^2 + 1^2 + ... + 24^2$, proved<!--raw-typst  (@prop-25)--> |
| Touching pairs | $2n^2(n - 1)$ in a square pile, 26,496 for 24 layers; $(n - 1)n(n + 1)$ in a tetrahedral pile | Proved<!--raw-typst  (@prop-touch)-->; checked by counting pairs two ways. The same numbers are OEIS A035006 (rook moves on an $n$ by $n$ board) and A007531, whose comment of 2013 counts the contacts in a tetrahedral pile |
| Balls that touch twelve others | A square pile of $n - 3$ layers inside the square pile (3,311 of the 4,900); a tetrahedral pile of $n - 4$ layers inside the tetrahedral pile | Proved<!--raw-typst  (@prop-twelve)-->; checked by counting |

<!-- /shared -->

## Where the count lands

![How many balls each square pile of 1 to 60 layers is from the nearest square, on a log scale; only 1 and 24 are exactly squares, on their own row, and 2, 47 and 48 are one ball away](gaps.svg)

For each square pile, the plot shows how many balls it is from the nearest square. Only the piles of 1 and 24 layers are exactly squares; they sit on their own row, since a log scale has no place for 0. Three piles miss by a single ball: 2 layers (5 balls, one more than 2²), 47 layers (35,720, one short of 189²) and 48 layers (38,024, one short of 195²).

## How the results are checked

<!-- shared:checks -->

- Each count formula is compared with a pile built ball by ball, for up to 12 layers. The same code checks that no two balls overlap and that each ball above the bottom rests on at least three below.
- Touching pairs are counted two ways: by looking up each ball's twelve possible neighbors, and by measuring every pair. Both agree with the first terms of OEIS A035006 and A007531.
- For Lucas's problem, the Pell walk and a plain search agree up to 10,000,000 layers. A test checks that the walk lists every candidate up to a million.
- The proof for cubes is compared with a search of every square pile up to 10<sup>18</sup> layers. A test checks the two remainder steps of the proof for every remainder on division by 7, and that $x^3 + 1 = 2z^3$, the case of Euler's theorem the proof uses, has no whole-number solution with $x$ from −10,000 to 10,000 other than $x = 1$ and $x = -1$.
- The rule for oblong piles is compared with two methods that use no theory: whether some square leaves the right remainder, for a up to 300, and trying every length, for a up to 40.
- For oblong piles of a fixed shape, the whole-number points that Sage finds on each curve give the same widths as a search of every $a$ up to 100,000, for $c$ up to 20 and for the $a$ by $2a - 1$ base. A test checks that every listed point lies on its curve and that the widths follow from the points.
- Every row of the table of coincidences is reproduced by search up to 1,000,000 layers, and the runs of squares match OEIS A001032. No result here disagrees with a published one.
- `report.py` writes every quoted number to `results.json`. Tests rerun it with smaller bounds and compare. Other tests check the committed figures, and that the shared sections of the README are what `tools/sync_docs.py` writes from `docs/` and `results.json`.

<!-- /shared -->

## Sources

The write-up cites every published result it uses. The main ones:

- Lucas's question ([numdam](https://www.numdam.org/item?id=NAM_1875_2_14__336_0)), Watson (1918), and Anglin, [*The square pyramid puzzle*](https://doi.org/10.2307/2323911) (1990), for Lucas's problem.
- Meyl's solution of question 1194 ([numdam](https://www.numdam.org/item/NAM_1878_2_17__464_1.pdf)), Avanesov (1967), Finkelstein and London (1972), and Beukers and Top (1988), for the other coincidences.
- Euler, *Vollständige Anleitung zur Algebra*, volume 2 (1770), §247, and Bennett (2001), for the proof about cubes. The same argument is in the MathPages note [One In The Chamber](https://www.mathpages.com/home/kmath445.htm).
- Beeckmans (1994), for runs of squares.
- The OEIS entries [A001032](https://oeis.org/A001032), [A007531](https://oeis.org/A007531), [A035006](https://oeis.org/A035006), [A053611](https://oeis.org/A053611) and [A224421](https://oeis.org/A224421).
- [Puzzling Stack Exchange question 99084](https://puzzling.stackexchange.com/questions/99084), for the oblong pile on an a by 2a − 1 base. Its text is under CC BY-SA 4.0; only its result is used here.
- For relatives: Hales, [*Cannonballs and honeycombs*](https://www.ams.org/notices/200004/fea-hales.pdf) (2000); Conway and Sloane, [*Lorentzian forms for the Leech lattice*](https://neilsloane.com/doc/Me91.pdf) (1982); and Cohn, Kumar, Miller, Radchenko and Viazovska, [*The sphere packing problem in dimension 24*](https://doi.org/10.4007/annals.2017.185.3.8) (2017).

No data files are used.

## Open questions

<!-- shared:open -->

- **A short proof for the tetrahedral pile of 34 layers.** Avanesov's proof that 7,140 is the last number that is both a triangle and a tetrahedral pile uses heavy machinery. Is there a proof by Pell equations and remainders, like Anglin's for Lucas's problem?
- **Oblong piles of every fixed shape.** For a base of $a$ by $a + c$, the squares are known exactly for each $c$ up to 20<!--raw-typst  (@tab-shapes)-->, one elliptic curve at a time. Is there a rule for which $c$ allow a pile of two or more layers that holds a square, as there is for a fixed short side<!--raw-typst  (@thm-oblong)-->? Up to 20, no curve of rank 0 gives one, and some curves of rank 1 give none either.
- **Other packings.** The piles here are pieces of the face-centered cubic packing. A pile can also be built with each new layer turned the other way, as in the hexagonal close packing, which is equally dense. Which pile shapes stay stable then, and how do their counts and touching pairs change?

<!-- /shared -->

## Usage

```bash
uv run python -m cannonballs.piles      # counts, and touching pairs of the 24-layer pile
uv run python -m cannonballs.lucas      # the searches, with default bounds
uv run python -m cannonballs.report     # writes results.json and gaps.csv (about half a minute)
uv run python -m cannonballs.figures    # writes pile.svg and gaps.svg
uv run --with passagemath-schemes --with passagemath-pari --with passagemath-flint \
  --with passagemath-eclib --with passagemath-symbolics \
  python -m cannonballs.curves          # writes curves.json; Sage is a large download (about 2 GB)
uv run tools/sync_docs.py cannonballs  # fills the README's shared sections from docs/ and results.json
uv run tools/typst_vendor.py cannonballs/cannonballs.typ --all   # the PDF, the phone PDF and the HTML page
```

```
$ uv run python -m cannonballs.lucas 1000000 1000
square piles that are squares, n <= 1,000,000: [1, 24]
  brute force took 0.3 s
case (1, 2, 3) survives every modulus up to 200
case (2, 1, 3) ruled out mod 3
case (1, 6, 1) ruled out mod 3
case (6, 1, 1) survives every modulus up to 200
case (2, 3, 1) ruled out mod 3
case (3, 2, 1) ruled out mod 3
by Pell equations, n < 10^1000: {(6, 1, 1): [24], (1, 2, 3): [1]}
  walked {(6, 1, 1): 502, (1, 2, 3): 654} solutions in 0.0 s
...
```

The two cases that survive hold the real solutions 1 and 24, so no remainder test can rule them out; the Pell walk handles them.

Only the Python standard library is needed, except for `curves.py`, which needs Sage (passagemath 10.8.14 was used). The tests check `curves.json` without Sage.

## Files

- `piles.py`: counts, piles built ball by ball, touching pairs.
- `lucas.py`: Lucas's problem, the six cases, the Pell walk, other coincidences, runs of squares, oblong piles.
- `report.py`: writes `results.json` and `gaps.csv`.
- `figures.py`: writes `pile.svg` and `gaps.svg`.
- `curves.py`: the whole-number points of the elliptic curve for each oblong shape, with Sage; writes `curves.json`.
- `docs/`: the sections that the README and the write-up share, with placeholders for the numbers in `results.json`.
- `cannonballs.typ`: the write-up; `cannonballs.pdf` and `cannonballs-phone.pdf` are built from it.
