# FreeCell

Which numbered Microsoft FreeCell deals cannot be won? Among the first 32,000, deal 11982 is the only one lost with the standard four free cells, and 199 are lost with three. Both counts are exact here: every deal is decided at these levels, the two solvers agree wherever both decide, and the counts match the published ones. With fewer free cells, 6,618 deals are lost with two and 25,710 with one, with 1 and 3 deals undecided, respectively. With no free cell, 31,931 are lost. Every number decided here matches the published per-deal list. Of the 31,931 deals lost with no free cell, 18,656 have a short reason that a separate program checks in milliseconds. Every other loss, with or without free cells, is shown by exhaustive search.

![Deal 11982 dealt into eight columns, with eight cards shaded: the ace and queen of clubs at the bottom of the fourth column, the ace of spades, three of hearts and six of hearts at the bottom of the second, the four of hearts and queen of spades at the bottom of the third, and the six of spades at the bottom of the sixth.](results/deal-11982.svg)

*Deal 11982, bottom cards at the top of the picture. The shaded cards never move in any position that can be reached with four free cells. Both black aces are among them, so no club or spade reaches its foundation.*

## Rules as modeled

- Eight columns are dealt from a shuffled deck: four of 7 cards and four of 6. The top card of a column may move.
- Each of the four free cells holds one card.
- Each suit has a foundation, built up from the ace. Cards never leave a foundation.
- A card may move onto a top card one rank higher of the other color, or onto an empty column. It may also go into an empty free cell or onto its foundation. The deal is won when every card is on a foundation.
- A move moves one card. A move of several cards at once is shorthand for single-card moves through free cells and empty columns. So it reaches no new position and changes no verdict.

`rules.py` states these rules for any deck, number of columns and number of free cells, so that the same code plays small games.

## Results

Lost deals among the first 32,000, by the number of free cells. Exact: a win is a line of moves that `rules.replay` checks, and a loss is an exhaustive search that found no won position.

| Free cells | Lost here | Won here | Undecided | Published lost | Source of the published count |
|---:|---:|---:|---:|---:|---|
| 4 | 1 | 31,999 | 0 | 1 | FreeCell FAQ (Keller) |
| 3 | 199 | 31,801 | 0 | 199 | FreeCell FAQ |
| 2 | 6,618 | 25,380 | 2 | 6,619 | Fish (2012) |
| 1 | 25,708 | 6,286 | 6 | 25,711 | Pringle's list (2016); the FAQ gives at least 6,289 won |
| 0 | 31,931 | 69 | 0 | 31,931 | FreeCell FAQ: 69 deals won |

![Bar chart of the first 32,000 deals by the least number of free cells that wins them: 69 need none, 6,216 need one, 19,083 need two, 6,419 need three, 198 need four, and one deal needs five or more. Fourteen deals are left out because Solver B left one of their levels undecided.](results/least-cells.svg)

*The least number of free cells that wins each deal, as Solver B found it; the 14 deals it left undecided at some level are not counted. Most deals need two.*

- **Agreement.** Solver A and Solver B decide the least number of free cells for 31,768 deals together, and they agree on every one. They also list the same cards that never move, on each of the 64,386 lost levels that both decide.
- **The published list.** Theodore Pringle's list gives the least winning number of free cells for each of the 32,000 deals. Solver B's number for each of its 31,986 decided deals equals it. The 69 deals won with no free cell are the FAQ's 69. The deals below 1000 that need all four are the FAQ's eleven.
- **Undecided.** Eight deals are left undecided at one level by both solvers, within the budgets given under Usage. With one free cell: 9499, 13167 and 13599, which the published list says are won, and 20410, 24966 and 27708, which it says are lost. With two: 982, which Fish (2012) reports lost, and 17760, which the list says is won. Each of the eight is won with one more free cell than its undecided level, by a line that replays. The next section decides four of them.

## The eight undecided levels

Four of the eight are decided here, and each agrees with the published list. Deals 20410 and 27708 are lost with one free cell: two layered solvers that share no code list every reachable position, and their counts are equal. Deals 9499, with one free cell, and 17760, with two, are won by lines that replay. Deals 13167, 13599, 24966 and 982 are still undecided. Every search of them ran out of memory, at 16 GB on GitHub's runners and at 32 GB on Hugging Face Jobs.

| Deal | Free cells | Found here | How it is known | Published |
|---:|---:|---|---|---|
| 9499 | 1 | won | a line of 131 moves that replays; the search for it visited 949,471,812 positions | won |
| 13167 | 1 | undecided | every search ran out of memory | won |
| 13599 | 1 | undecided | every search ran out of memory | won |
| 20410 | 1 | lost | Solver C and Solver B by layers each list 652,740,362 positions; the ten of spades never moves | lost |
| 24966 | 1 | undecided | every search ran out of memory | lost |
| 27708 | 1 | lost | both list 277,578,773 positions; every card moves in some line, and still none wins | lost |
| 982 | 2 | undecided | every search ran out of memory | lost; Fish's post of 23 May 2012 counts 3,095,862,346 positions with Horne's rule |
| 17760 | 2 | won | a line of 136 moves that replays; the search visited 85,327,707 positions | won |

The table with these levels decided:

| Free cells | Lost | Won | Undecided |
|---:|---:|---:|---:|
| 4 | 1 | 31,999 | 0 |
| 3 | 199 | 31,801 | 0 |
| 2 | 6,618 | 25,381 | 1 |
| 1 | 25,710 | 6,287 | 3 |
| 0 | 31,931 | 69 | 0 |

- **A search in layers.** Cards never leave a foundation, so the four foundation heights only grow. Solver C (`solver_c.c`) groups positions by their foundations and searches one group at a time, in order of cards home. A group is complete when its turn comes, since every position that can enter it has fewer cards home. Only that group is held in memory, and positions waiting for a later group go to files when memory runs short. A group can be split further. In each dealt column, the cards still in their dealt places end in a run where each card fits on the one below it. Where that run starts never rises: a card returns to its dealt place only by fitting on the card below, which lengthens the run. So a group and the eight starting points together also only move one way. This split cut the largest group of deal 11982 from 39,573 positions to 1,936. Solver C writes a position in about 120 bits relative to the deal: how many of each column's dealt cards are still in place, then one bit per card for the suit of each run card, since a run card is one rank lower and of the other color than the card below it.
- **Two counts.** Solver B gained the same plan by layers, written in its own code with its own positions, moves and safe-card rule. On a lost level both list every reachable position, so their counts must be equal. They are, layer by layer, on every lost level counted by both, and both agree with Solver B's depth-first count on the small decks and on deal 11982.
- **Wins.** Solver C can also search the group with most cards home first. Every waiting position keeps its parent, and a group's search can be run again to find a path within it, so a win ends with a line of single-card moves. `rules.replay` checks each line.
- **Horne's rule.** Fish's solvers play a card of rank r home when the foundations of the other color are at r − 2 or more and the other one of its color at r − 3 or more. This plays more cards home than the rule here. It is safe too: in any winning line, a card of the other color that would be placed on it goes home instead, since its foundation is at r − 2; a card placed on that one goes home too, since its foundation is at r − 3; and nothing of rank r − 3 of the other color is left out to go on top. Both layered solvers have it as an option. On deals 1 to 400 with one free cell the two rules give the same verdict on all 308 lost deals, and the counts differ on 31, where the two layered solvers again agree.

## Deal 11982

Deal 11982 is lost with four free cells and won with five.

```console
$ uv run python -m freecell.certificates 11982 4
deal 11982, 4 cells: lost; 61,643 positions reachable, at most 2 cards home; cards that never move: AC AS 3H 4H 6H 6S QC QS
$ uv run python -m freecell.certificates 11982 5
deal 11982, 5 cells: won in 111 moves
```

- **Reachable positions.** With safe cards played home at once and with columns and free cells taken in any order, there are 61,643 reachable positions. Sells (2026) reports the same number for his own search. With no automatic moves there are 83,239 (Solver B). With the columns kept in their places there are 84,767 (`brute.py`); Sells checks a set of that size in Lean.
- **Cards home.** In none of these positions are more than two cards on the foundations. A search of 1995, quoted by Sells, found the same maximum.
- **Stuck cards.** Eight cards never move in any reachable position (shaded in the picture). Both black aces stay buried.
- **Variations.** Swapping the A♠ and the A♦ makes the deal lost with two free cells and won with three. The FAQ reports that the swapped deal is won with three free cells.

## Why a deal is lost

A win is easy to check: replay its moves. A loss needs a **certificate**, a reason that a program that did not do the search can check. This folder uses three kinds.

- **Blocked at the start.** With no free cell, no card can move: no ace is on top, and no top card fits on another. With a free cell this cannot happen.
- **Stuck cards** (no free cell). Build a set of cards from below. Add a card once every card dealt above it is in the set and it has somewhere to go. Its foundation counts if it is an ace or the card below it in its suit is in the set. A card one rank higher of the other color counts if that card is in the set or has every card above it in the set. Another column counts if all its cards are in the set. Every card that ever moves gets into the set, by induction on the time of its first move: everything it needed moved earlier. A card left out never moves, so it never reaches a foundation. `certificates.static_stuck` computes the set in a few milliseconds per deal.
- **A fence.** A fence is a set of positions that holds the deal and holds no won position. Every move from one of its positions must stay inside. Every reachable position is then in the fence, so the deal is lost. The set of positions a solver visits is a fence; `fence.py` checks one with its own move generator. Sells (2026) checks a fence of deal 11982 in Lean in the same way. The fence's size measures how long the certificate is.

How many lost deals each kind explains. Fence sizes are Solver B's positions with no automatic moves, except for the 7 losses decided in a rerun with safe cards played home (6 with one free cell, 1 with two):

| Free cells | Lost | Blocked | Stuck cards | Fence ≤ 100 | Fence ≤ 10,000 | Fence ≤ 1,000,000 | Largest fence |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 31,931 | 1,325 | 18,656 | 27,690 | 31,858 | 31,924 | 32,780,527 (deal 27962) |
| 1 | 25,708 | 0 | 0 | 4,920 | 22,998 | 25,527 | 235,709,696 (deal 25887) |
| 2 | 6,618 | 0 | 0 | 9 | 4,486 | 6,504 | 167,286,798 (deal 11266) |
| 3 | 199 | 0 | 0 | 0 | 83 | 193 | 2,813,435 (deal 13584) |
| 4 | 1 | 0 | 0 | 0 | 0 | 1 | 83,239 (deal 11982) |

- **Blocked deals.** The FAQ reports 1,325 deals blocked at the start with no free cell, the first being 1, 23, 25, 28, 46 and 51; 1,325 are found here too. Every blocked deal also has stuck cards, so the stuck-card set explains 18,656 deals. In 88 of them no ace is stuck, but some other card is.
- **Soundness.** No stuck card is ever moved by Solver B, in any of the 18,656 deals. On small decks the stuck cards are never more than the cards that never move (table below).
- **No card that never moves.** In a fence, the cards that no position ever moves give a short summary of where a deal is stuck. For most lost deals at least one card never moves. It is not so for 14 deals lost with one free cell, 12 with two and one (8927) with three. Every card of such a deal moves in some line, and still no line finishes.
- **What is left to trust.** The fences of the survey were not stored, so the losses in the first table rest on two independent exhaustive searches that agree. `fence.py` checks both solvers' fences on sample deals in the tests.

## How the results are checked

- **Two solvers.** Solver A (`solver.py`, Python) searches best first and plays safe cards home at once. It stores positions with columns and free cells sorted, and it moves runs of cards as well as single cards. Solver B (`solver_b.c`, C) searches depth first over single-card moves. It has its own deal generator and stores a position as what lies beneath each card. The two share no code; each has its own copy of the rules. Run with its own copy of the safe-card rule, Solver B reaches the same 61,643 positions of deal 11982 as Solver A.
- **Brute force on small decks.** `brute.py` lists every reachable position with columns in fixed places and nothing pruned. `small.py` runs it and both solvers on every deal of five small games and on 3,000 seeded deals of two more. The three agree on every deal (table below).
- **Every win is replayed** by `rules.replay`, move by move, with the free cells of its level and with four. None of the survey's lines failed.
- **Fences** of both solvers pass `fence.py`, and a fence with one position removed fails.
- **Safe moves home.** A card goes home at once when it can and when every card that could sit on it is home or is an ace. (The cards that could sit on it are one rank lower, of the other color.) Moving it home never turns a won position into a lost one. In any winning line, delete the card's later moves, and send any ace placed on it home instead; every other move stays legal. Solver A uses the rule and Solver B, in the survey, does not, so their agreement also tests it.
- **Monotone in free cells.** A line that wins with k free cells wins with more, and every position reachable with fewer free cells is reachable with k. So for each deal it is enough to find the least winning level; each level below it is lost.
- **In CI.** The tests check the deal generator against published deals, both solvers and the brute force on 2,606 small deals, 11982 with both solvers and the brute force, the fences, the blocked count and the transcript above. The survey itself runs on GitHub Actions; CI checks the committed summary files against each other.

Small decks: every deal, or 3,000 seeded deals where marked. A deal is an order of the deck dealt in rows, as in Microsoft FreeCell.

| Deck | Columns | Free cells | Deals | Lost | Blocked | Stuck cards | All agree |
|---|---:|---:|---:|---:|---:|---:|---|
| A–3, 2 suits | 2 | 0 | 720 | 464 | 192 | 376 | yes |
| A–3, 2 suits | 2 | 1 | 720 | 42 | 0 | 0 | yes |
| A–4, 2 suits | 2 | 0 | 40,320 | 37,248 | 15,840 | 31,800 | yes |
| A–4, 2 suits | 2 | 1 | 40,320 | 19,144 | 0 | 0 | yes |
| A–4, 2 suits | 2 | 2 | 40,320 | 1,438 | 0 | 0 | yes |
| A–2, 4 suits | 2 | 0 | 40,320 | 21,984 | 8,640 | 21,696 | yes |
| A–2, 4 suits | 2 | 1 | 40,320 | 1,200 | 0 | 0 | yes |
| A–2, 4 suits | 2 | 2 | 40,320 | 0 | 0 | 0 | yes |
| A–4, 2 suits | 3 | 0 | 40,320 | 18,908 | 4,320 | 7,872 | yes |
| A–4, 2 suits | 3 | 1 | 40,320 | 144 | 0 | 0 | yes |
| A–5, 2 suits | 3 | 0 | 3,000 sampled | 2,455 | 655 | 1,445 | yes |
| A–5, 2 suits | 3 | 1 | 3,000 sampled | 455 | 0 | 0 | yes |
| A–5, 2 suits | 3 | 2 | 3,000 sampled | 0 | 0 | 0 | yes |
| A–3, 4 suits | 3 | 0 | 3,000 sampled | 1,762 | 208 | 996 | yes |
| A–3, 4 suits | 3 | 1 | 3,000 sampled | 126 | 0 | 0 | yes |
| A–3, 4 suits | 3 | 2 | 3,000 sampled | 0 | 0 | 0 | yes |

"All agree" means that the brute force and both solvers give the same verdict on every deal. On lost deals, the two solvers list the same cards that never move. Every stuck card is among them, and no win replays wrong.

## The numbered deals

Deal n seeds the Microsoft C library's random number generator with n. Its rule is state ← (214013 · state + 2531011) mod 2³¹, and each call returns state ÷ 2¹⁶. The deck starts in the order A♣ A♦ A♥ A♠ 2♣ ... K♠. For k = 0, 1, ..., 51, the k-th card dealt is the one at position rand mod (52 − k) among the cards left, and the last card left takes its place. Cards go out in rows across the eight columns. `deals.py` follows the description on Rosetta Code. The first four values for seed 1 are those of the C runtime (41, 18467, 6334, 26500). The cards of deals 1 and 617 match Rosetta Code's lists, and deal 11982 matches the layout printed in the fc-solve FAQ. Solver B has its own generator, in C, and deals the same cards.

## Data

| Source | What it is | License | Committed |
|---|---|---|---|
| Pringle, MicrosoftFreeCell32000-minimal-freecells (GitHub) | The least number of free cells that wins each of the first 32,000 deals | MIT (`data/LICENSE-msfc32k-min-fc`) | yes, unchanged, as `data/msfc32k-min-fc` |
| FreeCell FAQ (Keller) | The 69 deals won with no free cell and the 11 below 1000 that need four, typed into `report.py` | unknown; the page states none | as lists of deal numbers |
| Survey runs on GitHub Actions | One record per deal and level | this repository | on branches `compute/F1-b-44619090164`, `compute/F1-b2-44643061114` and `compute/F1-a-44619109023`; small reruns in `results/runs/` |

## Sources

- M. Keller, "FreeCell FAQ", solitairelaboratory.com/fcfaq.html. Deal 11982 and the Internet FreeCell Project, and the counts with fewer free cells. It states that every deal except 11982 is solvable.
- Rosetta Code, "Deal cards for FreeCell"; S. Fish, the fc-solve FAQ (fc-solve.shlomifish.org/faq.html), with the layout of deal 11982 and a partial list of solvers that found no win.
- S. Fish, "Two Freecell Solvability Report for the First 400,000 Deals", fc-solve.blogspot.com, 2 September 2012. With two free cells, 25,381 of the first 32,000 deals are solved and 6,619 are impossible.
- S. Fish, ["The 2-Freecell Deal No. 982 Was Slain!!"](https://fc-solve.shlomifish.org/mail-lists/fc-solve-discuss/archive/1043.html), fc-solve-discuss, 23 May 2012. Deal 982 is lost with two free cells; its graph spans 3,095,862,346 positions, some pruned by Horne's rule.
- T. Pringle, "MicrosoftFreeCell32000-minimal-freecells", github.com/shlomif/MicrosoftFreeCell32000-minimal-freecells, 2016 (MIT license).
- S. Fish, "Report: The solvability statistics of the Freecell Pro 4-Freecells Deals", fc-solve.shlomifish.org, 2016, updated 2018. It counts 102,075 impossible deals among 8,589,934,591 FreeCell Pro deals.
- C. Sells, "So You're Saying There's A Chance: FreeCell Game #11982", sellsbrothers.com, 24 October 2026. A Lean proof that 11982 is lost, by a checked fence. It quotes an exhaustive search of 1995 that reached at most two cards home.
- M. Helmert, "Complexity results for standard benchmark domains in planning", *Artificial Intelligence* 143(2), 2003. FreeCell, generalized to larger decks, is NP-complete. He places the hardness in the choice of which of two cards to build on, not in the use of free cells.
- A. Elyasaf, A. Hauptman, M. Sipper, "GA-FreeCell: Evolving Solvers for the Game of FreeCell", GECCO 2011; earlier, A. Hauptman, A. Elyasaf, M. Sipper, SoCS 2010. Learned search heuristics for the 32,000 deals.

Reproduced here: the deal generator; the published per-deal least numbers of free cells, wherever both solvers' budgets sufficed; the counts with four, three and no free cells; the 1,325 blocked deals; and Sells's 61,643 and 84,767 positions for deal 11982. Only read: the counts beyond the first 32,000.

## Open questions

- **The four open levels.** Deals 13167 and 13599 with one free cell (won in the published list), 24966 with one (lost) and 982 with two (lost) are still undecided. Split into groups, their largest group of positions held more than 32 GB could store. Fish decided 982 in May 2012 on a computer with 512 GB of memory (fc-solve-discuss, 23 May 2012). A disk-backed table for the largest group, or a finer split of it, could decide them here.
- **A short certificate with free cells.** No proof that 11982 is lost avoids listing positions. Sells (2026) tried counting free places at the moment a stuck card must move; he found that it fails once a column can be emptied. With a free cell, the only certificate here is the fence. A short argument like the stuck-card set would need to bound how long cards can stay in free cells.
- **Deals with no card that never moves.** In 27 lost deals of the survey every card moves in some line, and deal 27708 with one free cell is another. What makes deal 8927 lost with three free cells?
- **Further deals.** Fish (2012) reports 82,126 deals lost with two free cells among the first 400,000. Fish (2016, updated 2018) counts 102,075 deals lost with four among all FreeCell Pro deals. Neither is rerun here. Is any deal above 32,000 lost with four free cells besides those reported? A survey of the first million deals with four free cells was started and canceled before it finished; no result of it is reported.

## Usage

The solvers and the survey use only the standard library; Solver B needs a C compiler (`cc`), which `native.py` calls once. Times are from one run on a two-core computer.

```bash
uv run python -m freecell.deals 1 11982            # print deals
uv run python -m freecell.certificates 2 0         # the cheapest reason found that deal 2 is lost with 0 free cells
uv run python -m freecell.fence 11982 4            # both solvers' fences for 11982, checked (25 s)
uv run python -m freecell.small                    # small decks, all engines (about 10 min)
uv run python -m freecell.survey --solver b --start 1 --stop 101 --out runs/   # 100 deals, all levels
uv run python -m freecell.survey --solver a --levels 3 --deals 982,9499 --out runs/
uv run python -m freecell.report --a runs/ --b runs/ --out runs/   # a summary of these runs, beside them
uv run python -m freecell.draw                     # rewrites the two pictures from results/least_cells.tsv
```

Without `--out`, `freecell.report` writes `results/f1.json` and `results/least_cells.tsv`, the committed summary of the whole survey, so run it that way only on the survey's own records. They are on the branches `compute/F1-b-44619090164`, `compute/F1-b2-44643061114` and `compute/F1-a-44619109023`, under `runs/`. With those folders and `results/runs/`, the report writes the committed files byte for byte:

```bash
for b in F1-b-44619090164 F1-b2-44643061114 F1-a-44619109023; do git archive origin/compute/$b runs | tar -x -C /tmp; done
uv run python -m freecell.report --a /tmp/runs/F1-a freecell/results/runs --b /tmp/runs/F1-b /tmp/runs/F1-b2 freecell/results/runs
```

The eight undecided levels ran through `freecell.open_levels`, one job per solver and level, on GitHub Actions (runs 44707374026, 44716155623, 44730671526 and 44738412060) and on Hugging Face Jobs (cpu-upgrade, 8 vCPU and 32 GB). Their records are in `results/runs/open-*.jsonl`, and `open_levels summary` writes `results/open_levels.json` from them.

```bash
uv run python -m freecell.open_levels c-27708-1 bl-27708-1 --out runs/   # both layered solvers, about 13 min each on a runner
uv run python -m freecell.open_levels w-9499-1 --budget 3000000000 --out runs/   # a win, 36 min on a runner
uv run python -m freecell.open_levels summary                             # writes results/open_levels.json
```

Jobs that ran out of memory are reported without numbers. On 16 GB runners: Solver C on 24966 and 982 with whole layers, Solver B by layers on 24966 and 982, and both hunts on 13167 and 13599. On Hugging Face Jobs with 32 GB: Solver C on 24966 and 982 with whole layers and with sublayers, Solver B by layers on 982 (stopped), and the hunt with most cards home first on 13167 and 13599. Solver B's depth-first search with seeds 1, 2 and 3 and a budget of 260 million positions left all four winnable levels undecided (run 44707374026).

The full survey ran on GitHub Actions through `.github/workflows/experiment.yml`. Solver B ran in 20 slices of 1,600 deals, each slice in under 100 minutes; run 44619090164, budget 150 million positions. The 49 deals it left open were run again with safe cards played home and a budget of 260 million (run 44643061114). Solver A ran in 40 slices of 800 deals, each slice in under 110 minutes; run 44619109023, last-search budget 4 million. The files `results/runs/survey-*.jsonl` come from `--levels` runs of Solver A on a two-core computer for the eight undecided deals, with a last-search budget of 1 to 2 million. A further Solver A search with one free cell for 9499, 13167 and 13599 was stopped after 30 minutes without a verdict.

## Files

| File | Contents |
|---|---|
| `deals.py` | Cards, the Microsoft deal generator, layouts as text |
| `rules.py` | The rules for any deck and tableau, safe moves home, replay of a line |
| `solver.py` | Solver A: best-first search with safe moves home and runs of cards |
| `solver_b.c`, `native.py` | Solver B: depth-first search in C, and by layers; its Python driver |
| `solver_c.c` | Solver C: the search in layers, and the search for a win that gives a line |
| `open_levels.py` | Jobs on the eight undecided levels, and their summary |
| `brute.py` | Every reachable position, nothing pruned |
| `fence.py` | Checks a fence, written apart from both solvers |
| `certificates.py` | Blocked deals and stuck cards; the cheapest reason for one deal |
| `survey.py` | Verdicts for a range of deals with 0 to 4 free cells |
| `report.py` | Merges the runs and compares them with the published results |
| `small.py` | Exact counts on small decks |
| `draw.py` | The two pictures |
| `results/f1.json` | The summary of the survey |
| `results/least_cells.tsv` | For each of the first 32,000 deals, the least winning number of free cells by each solver and in Pringle's list |
| `results/small.json` | The small-deck counts |
| `results/runs/` | Records of the reruns of the eight undecided deals, and of the jobs on their open levels |
| `results/open_levels.json` | The verdicts on the eight open levels, and the counts with them |
| `data/msfc32k-min-fc` | Pringle's list, unchanged (MIT license, in `data/LICENSE-msfc32k-min-fc`) |
