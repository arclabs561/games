# FreeCell

FreeCell is a solitaire game played with all 52 cards face up. Microsoft's version numbers its deals, and of the first 32,000, only deal 11982 is known to be lost. This folder asks which deals cannot be won: with the standard four free cells, and with three, two, one and none. It also asks why a lost deal is lost, in a form that can be checked without trusting a solver.

## Rules as modeled

- Eight columns are dealt from a shuffled deck: four of 7 cards and four of 6. The top card of a column may move.
- Each of the four free cells holds one card.
- Each suit has a foundation, built up from the ace. Cards never leave a foundation.
- A card may move onto a top card one rank higher of the other color, onto an empty column, into an empty free cell, or onto its foundation. The deal is won when every card is on a foundation.
- A move moves one card. A move of several cards at once is shorthand for single-card moves through free cells and empty columns, so it reaches no new position and changes no verdict.

`rules.py` states these rules for any deck, number of columns and number of free cells, so that the same code plays small games.

## The numbered deals

Deal n seeds the Microsoft C library's random number generator with n. Its rule is state ← (214013 · state + 2531011) mod 2³¹, and each call returns state ÷ 2¹⁶. The deck starts in the order A♣ A♦ A♥ A♠ 2♣ ... K♠. For k = 0, 1, ..., 51, the k-th card dealt is the one at position rand mod (52 − k) among the cards left, and the last card left takes its place. Cards go out in rows across the eight columns. The author of Microsoft's version made the algorithm public; `deals.py` follows the description on Rosetta Code. The first four values for seed 1 are those of the C runtime (41, 18467, 6334, 26500). The cards of deals 1 and 617 match Rosetta Code's lists, and deal 11982 matches the layout printed in the fc-solve FAQ. Solver B has its own generator, in C, and deals the same cards.

## Results

How many of the first 32,000 deals are lost, by the number of free cells:

| Free cells | Lost here | Published | Source of the published count |
|---:|---:|---:|---|
| 4 | ⟨1⟩ | 1 | The Internet FreeCell Project (1995), by hand; FreeCell Pro's solver (FAQ) |
| 3 | ⟨199⟩ | 199 | FreeCell Pro's solver (FAQ) |
| 2 | ⟨6,619⟩ | 6,619 | Fish (2012), fc-solve |
| 1 | ⟨25,711⟩ | at most 25,711 | "at least 6,289" won (FAQ); Pringle's list (2016) gives 25,711 |
| 0 | ⟨31,931⟩ | 31,931 | 69 deals won with zero free cells, listed in the FAQ |

Exact, by exhaustive search with two solvers. ⟨agreement sentence⟩ For each deal, the least number of free cells that wins it is also in a published list by Theodore Pringle (2016), and ⟨both solvers agree with it on every deal⟩. The 69 deals won with no free cell are the 69 that the FAQ lists, and the deals below 1000 that need all four free cells are the eleven that it lists.

Deal 11982:

- It is lost with four free cells. With autoplay of safe cards (below) and columns and free cells taken in any order, 61,643 positions can be reached, as Sells (2026) reports. With no autoplay there are 83,239 (Solver B), and with the columns kept in their places there are 84,767, again as Sells reports (`brute.py`). In none of them are more than two cards on the foundations; a breadth-first search of 1995, quoted by Sells, found the same maximum.
- Eight cards never move in any reachable position: A♣ and the Q♣ above it, A♠ and the 6♥ above it, and the 3♥, 4♥, 6♠ and Q♠. Both black aces stay buried, so the deal cannot be won.
- With five free cells it is won, as the FAQ says. Swapping the A♠ and the A♦ makes it lost with two free cells and won with three, as the FAQ also says.

⟨first million paragraph⟩

## Why a deal is lost

A win is easy to check: replay its moves. A loss needs a **certificate**, a reason that can be checked by a program that did not do the search. This folder uses three kinds.

- **Blocked at the start.** With no free cell, no card can move: no ace is on top, and no top card fits on another. With a free cell this cannot happen.
- **Stuck cards** (no free cell). Build a set of cards from below: add a card once every card dealt above it is in the set and it has somewhere to go. A place to go is the foundation (it is an ace, or the card below it in its suit is in the set), a card one rank higher of the other color that is in the set or has every card above it in the set, or another column whose cards are all in the set. Every card that ever moves gets into the set: by induction on the time of its first move, everything it needed moved earlier. A card left out never moves, so it never reaches a foundation. `certificates.static_stuck` computes the set in a few milliseconds per deal.
- **A fence.** A set of positions that holds the deal, that every move from one of its positions stays inside, and that holds no won position. Every reachable position is then in the fence, so the deal is lost. The set of reachable positions that a solver visits is a fence; `fence.py` checks one with its own move generator. Sells (2026) checks a fence of deal 11982 in Lean in the same way. Within a fence, the cards that no position ever moves are a readable summary of where the deal is stuck.

How many lost deals among the first 32,000 each kind explains:

⟨certificate table⟩

⟨interpretation⟩

## How the results are checked

- **Two solvers.** Solver A (`solver.py`, Python) searches best first and plays safe cards home at once. It stores positions with columns and free cells sorted, and it tries moves of whole runs of cards as well as single cards. Solver B (`solver_b.c`, C) searches depth first over single-card moves and plays no card home on its own. It has its own deal generator and stores a position as what lies beneath each card. The two share no code; each has its own copy of the rules.
- **Brute force on small decks.** `brute.py` lists every reachable position with columns in fixed places and nothing pruned. `small.py` runs it and both solvers on every deal of five small games and on 3,000 seeded deals of two more; the three agree on every deal (table below). Both solvers also report the same frozen cards on every lost deal, and the stuck cards are always among them.
- **Every win is replayed** by `rules.replay`, move by move, with the free cells of its level and with four.
- **Fences** of both solvers pass `fence.py`, and a fence with one position removed fails.
- **Safe autoplay.** A card goes home at once when it can and when each card that could sit on it is home or is an ace. (The cards that could sit on it are one rank lower, of the other color.) Moving it home never turns a won position into a lost one: in any winning line, delete the card's later moves and send any ace placed on it home instead; every other move stays legal. Solver A uses the rule and Solver B does not, so their agreement also tests it.
- **Monotone in free cells.** A line that wins with k free cells wins with more. Every position reachable with fewer free cells is reachable with k. So for each deal it is enough to find the least winning level; each level below it is lost.

| Deck | Columns | Free cells | Deals | Lost | Blocked | Stuck cards | All agree |
|---|---:|---:|---:|---:|---:|---:|---|
⟨small table⟩

## Open questions

- No proof that 11982 is lost avoids listing positions. Sells (2026) tried counting free places at the moment a stuck card must move and found that it fails once a column can be emptied.
- With one or more free cells the only certificate here is the fence itself. A short argument like the stuck-card set, with free cells, would need to bound how long cards stay in them.
- ⟨unexplained⟩
- Whether the counts extend: Fish (2012) reports 82,126 deals lost with two free cells among the first 400,000, and Pringle and Fish (2018) list 102,075 lost with four among all 8,589,934,591 FreeCell Pro deals. Neither is rerun here.

## Usage

The solvers and the survey use only the standard library; Solver B needs a C compiler (`cc`), which `native.py` calls once. Times are from one run on a shared two-core machine.

```bash
uv run python -m freecell.deals 1 11982            # print deals
uv run python -m freecell.fence 11982 4            # both solvers' fences for 11982, checked (25 s)
uv run python -m freecell.small                    # small decks, all engines (⟨⟩ min)
uv run python -m freecell.survey --solver b --start 1 --stop 1001 --out runs/   # ⟨⟩ min
uv run python -m freecell.survey --solver a --start 1 --stop 101 --workers 2 --out runs/
uv run python -m freecell.report --a runs/ --b runs/
```

The full survey ran on GitHub Actions through `.github/workflows/experiment.yml`: Solver B in 20 slices and Solver A in 40 (`--slice SEED --slices N`). ⟨run ids and times⟩

## Files

| File | Contents |
|---|---|
| `deals.py` | Cards, the Microsoft deal generator, layouts as text |
| `rules.py` | The rules for any deck and tableau, safe autoplay, replay of a line |
| `solver.py` | Solver A: best-first search with safe autoplay and runs of cards |
| `solver_b.c`, `native.py` | Solver B: depth-first search in C, and its Python driver |
| `brute.py` | Every reachable position, nothing pruned |
| `fence.py` | Checks a fence, written apart from both solvers |
| `certificates.py` | Blocked deals and stuck cards |
| `survey.py` | Verdicts for a range of deals with 0 to 4 free cells |
| `report.py` | Merges the runs and compares them with the published results |
| `small.py` | Exact counts on small decks |
| `results/f1.json` | The summary of the survey |
| `results/least_cells.tsv` | For each of the first 32,000 deals, the least winning number of free cells by each solver and in Pringle's list |
| `results/small.json` | The small-deck counts |
| `data/msfc32k-min-fc` | Pringle's list, unchanged (MIT license, in `data/LICENSE-msfc32k-min-fc`) |

## Sources

- M. Keller, "FreeCell FAQ", solitairelaboratory.com/fcfaq.html. The history of deal 11982 and of the Internet FreeCell Project, and the counts with fewer free cells.
- Rosetta Code, "Deal cards for FreeCell"; S. Fish, fc-solve, `gen_ms_boards__rand.h` (github.com/shlomif/fc-solve).
- S. Fish, "Two Freecell Solvability Report for the First 400,000 Deals", fc-solve.blogspot.com, 2 September 2012.
- T. Pringle, "MicrosoftFreeCell32000-minimal-freecells", github.com/shlomif/MicrosoftFreeCell32000-minimal-freecells, 2016 (MIT license).
- S. Fish, "Report: The solvability statistics of the Freecell Pro 4-Freecells Deals", fc-solve.shlomifish.org, 2016, updated 2018.
- C. Sells, "So You're Saying There's A Chance: FreeCell Game #11982", sellsbrothers.com, 24 October 2026. Quotes an exhaustive search of deal 11982 from 1995.
- M. Helmert, "Complexity results for standard benchmark domains in planning", *Artificial Intelligence* 143(2), 2003, and his Diplomarbeit (Freiburg, 2001), Theorem 4.26. With four suits, deciding whether a deal can be won is NP-complete when the number of ranks and of columns grows, for any fixed number of free cells. He finds the hardness in the choice of which of two cards to build on, not in the use of free cells.
- A. Elyasaf, A. Hauptman, M. Sipper, "GA-FreeCell: Evolving Solvers for the Game of FreeCell", GECCO 2011. Learned heuristics that solve 98% of the 32,000 deals within a time limit; earlier, A. Hauptman, A. Elyasaf, M. Sipper, SoCS 2010.
