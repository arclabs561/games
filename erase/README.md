# Erase

<!-- shared:summary -->

Which deals of this make-ten card game cannot be cleared? The answer depends on how the rules are read. The printed rules erase pairs that sum to 10. Under them 96.02% of deals are stuck, and with addition alone 71.10% are. Under the most generous reading of the sheet's variation, which combines any number of cards with +, − and ×, only 12,362,536 of the 2,530,881,913,560 possible deals are stuck, about 1 in 204,722. All three counts are exact, and the last comes with a proof and machine-checked certificates. The rest of this page is about that reading. There, a deal that can be cleared can always be cleared in one group, and a stuck hand without a 0 is caught by a remainder argument, mod 3 or mod 4. Ten is not special: for every target from 1 to 30, the hands that cannot make it are caught in the same way from 7, 8 or 9 cards on, checked through 11 cards.

<!-- /shared -->

![Two deals of seven hand cards and four center cards. The first, hand 4 4 5 5 6 8 9 with center 1 5 7 8, clears, with its groups that make 10 listed beside it. The second, hand 0 4 4 4 8 8 8 with center 0 3 7 9, cannot be cleared.](figures/deals.svg)

*Two deals and what `erase.py` finds for them. The center cards (shaded) can be used again; the hand cards are erased.*

Erase is a make-ten card game for classrooms. It is Game 9 in *Math Fact Fluency* by Jennifer Bay-Williams and Gina Kling.

The analysis is written up in [`paper/erase.pdf`](paper/erase.pdf), with a narrow edition for phones in [`paper/erase-phone.pdf`](paper/erase-phone.pdf). Its first sections tell the story with arithmetic only and the rest proves each claim. The full classification of stuck deals is the theorem in its section "Big hands always make 10," and the counts for other readings of the rules are in "The other readings." The section "Is 10 special?" has the table for the targets 1 to 30, and "Relatives" has the Countdown and 24-game counts.

## Rules as modeled

<!-- shared:rules -->

- **Deck.** The sheet asks for a "0-10 deck of cards": four of each value from 0 to 10, 44 cards. Some classrooms use cards 1 to 10 instead.<!--raw-typst  @prop-oneten covers that deck.-->
- **Players.** Two to four.
- **Deal.** Each player gets seven cards. Four more go face up in the middle as *center cards*, shared by everyone.
- **Erase.** Two of your own cards that make 10 are both erased. One of your cards and a center card that make 10 erase only your card. The center card stays, and anyone may use it again.
- **Score.** The number of cards you have left. All cards are reshuffled and dealt again each round, and after five rounds the lowest total wins.

<!-- /shared -->

<!-- shared:reading -->

The sheet lists a variation: "Players may use more than 2 cards to make a 10." It does not say which operations are allowed. The sheet files Erase under addition and subtraction facts, and the authors' teacher page calls it a strategy game for addition, so adding (and perhaps subtracting) is the likely intent. The results here are mainly for a more generous reading that allows +, − and ×.<!--raw-typst  @tab-readings compares the others.--> A move is then: choose some of your cards, with or without one center card, that make 10 when combined with +, − and ×, each card used once. Your cards in it are erased, and the center card stays. Such a set of cards is called a *group*.

<!-- /shared -->

## Results

All results are exact. A row that names no reading of the rules is for the generous one, with +, − and ×.

<!-- shared:results -->

| Result | Value | How it is known |
|---|---|---|
| Deals stuck under the printed rules (pairs that sum to 10) | 96.02% | proved by hand<!--raw-typst  (@prop-pairs)-->, then counted over every deal (`count.py --pairs`) |
| Deals stuck with addition only | 71.10% | exact count (`count.py --addition`) |
| **Deals stuck with +, − and ×** | **12,362,536 of 2,530,881,913,560, about 1 in 204,722** | brute-force count (`count.py`); the classification<!--raw-typst  (@thm-class)--> is proved using the theorem on big hands, which is checked case by case |
| Groups needed to clear a deal | one: the whole hand plus at most one center card | proved<!--raw-typst  (@thm-single)--> using the theorem on big hands<!--raw-typst  (@thm-big)-->, whose 265,148 cases are checked with certificates |
| Stuck hands without a 0 | caught by a remainder argument, mod 3 or mod 4 | proved<!--raw-typst  (@thm-class)--> using the theorem on big hands |
| Cards 1 to 10 only (40 cards) | 51,840 of 762,894,475,200 deals stuck, about 1 in 14.7 million | brute-force count (`count.py --no-zero`), and proved<!--raw-typst  (@prop-oneten)--> using the theorem on big hands |
| With the four kings as wild cards | 12,362,536 of 7,456,416,121,440 deals stuck, about 1 in 603,146 | proved by hand: a deal with a wild card clears in one group<!--raw-typst  (@prop-wild)--> |
| Targets 1 to 30: the hand size from which every hand that cannot make t is blocked by a remainder | 7, 8 or 9; 8 for 20 of the targets | every multiset of 1 to 11 cards checked, with certificates and a separate checker<!--raw-typst  (@sec-targets)--> |
| Countdown problems with an exact solution | 10,858,746 of 11,905,457 (91.21%) | recomputed; agrees with Alliot (2015)<!--raw-typst  (@sec-relatives)--> |
| 24 game hands that make 24 | 1,362 of 1,820 | exact, checked against a brute-force listing; agrees with published counts<!--raw-typst  (@sec-relatives)--> |

<!-- /shared -->

## How the results are checked

<!-- shared:checks -->

- Facts about remainders, zeros and signs are proved by hand. `smt/z3_lemmas.py` checks each induction step with z3.
- One step, the theorem on big hands<!--raw-typst  (@thm-big)-->, is a finite claim about 265,148 multisets of cards. `make_certs.py` writes one expression per case to `certificates.jsonl.gz`. `verify_certs.py` checks every expression and shares no code with the search. The file is generated, not stored in the repository.
- `count.py` gets the count of stuck deals by brute force. It uses none of the theorems. With `--pairs` and `--addition` it counts the deals that the printed rules and addition alone leave stuck; the tests compare both rules with brute-force searches on random deals.
- `simulate.py` reruns every simulated figure with fixed seeds: + and −, "grab the pairs first," small groups, and play without brackets. It also simulates addition only, as a check on the exact count.
- `smt/z3_programs.py` and `smt/statespace.py` check the impossible cases with two other methods. The tests run `statespace.py`; the z3 check takes about 2.5 min and is run by hand.
- For the targets 1 to 30, `targets.py --certs` writes one expression for each multiset from the threshold up that makes the target. `verify_targets.py` lists the multisets itself and checks each expression with Python's own parser, and decides every other multiset with its own search over remainders. It shares no code with `targets.py`.
- `report.py` writes every number quoted in the README and the paper to `results.json`. The tests recompute its cheaper parts and check that the README's shared sections are what `tools/sync_docs.py` writes from `docs/` and `results.json`.

CI runs the tests and the z3 lemma check, and rebuilds the paper's PDFs to check that they are current. The certificate files are generated, not stored, so CI does not check them.

<!-- /shared -->

## Data

The code uses no data files, and no third-party data is committed. The published counts it is compared with are listed under Sources. The one third-party file in the folder is a font subset for the paper.

| Source | What | License | Committed |
|---|---|---|---|
| Noto Serif CJK JP, katakana only | one word in the paper's sources | SIL Open Font License (`paper/fonts/OFL.txt`) | yes |

## Sources

- J. Bay-Williams and G. Kling, *Math Fact Fluency* (ASCD, 2019): the game, Game 9.
- J.-M. Alliot, ["The (Final) Countdown"](https://arxiv.org/abs/1502.05450) (2015), Section 4.1.1, Table 2: the Countdown counts reproduced in the paper.
- [4nums.com](https://www.4nums.com/24_game/facts/), [gottfriedville.net](https://gottfriedville.net/games/24/index.shtml) and [puzzletip.com](https://puzzletip.com/24-game-solver/): the 24-game counts for cards 1 to 13 reproduced in the paper. [boergens.de](http://www.boergens.de/make24/trivia/) and [monkeywrench](https://www.wheels.org/monkeywrench/?p=636): the counts for cards 1 to 9 and 1 to 10.
- The impossible Make 10 puzzles are listed on the Japanese Wikipedia page [テンパズル](https://ja.wikipedia.org/wiki/%E3%83%86%E3%83%B3%E3%83%91%E3%82%BA%E3%83%AB).

The paper gives the full references.

## Open questions

<!-- shared:open -->

- A proof of the theorem on big hands<!--raw-typst  (@thm-big)--> that does not check cases is still open; in particular, a short reason why any seven nonzero cards outside the two families make 10 using all of them, apart from 1 6 6 7 7 7 7.
- Do the thresholds hold for multisets of more than 11 cards, for 10 or for any other target? A deal never holds more than 11 cards, and every threshold here, that of 10 included, is checked only that far.
- Does any target above 30 have a threshold above 9? Among the targets checked only 14 reaches 9, because of one multiset of eight cards.

<!-- /shared -->

## Usage

The scripts need only the standard library, except for the z3 checks and the paper build, which declare their dependencies inline (PEP 723) and run with `uv run`. Their lock files (`*.py.lock`) pin the versions that produced the paper: z3-solver 5.4.1.0 and typst 0.15.2. Times are from one run on one machine.

```bash
python3 erase/erase.py 4 4 6 9 5 8 5 --center 7 5 1 8   # prints one group per line
python3 erase/erase.py 0 4 4 4 8 8 8 --center 0 3 7 9   # prints "cannot be cleared"
python3 erase/count.py          # 12362536 2530881913560 1 in 204,722 (about 35 s)
python3 erase/count.py --pairs  # 2430260556960 2530881913560 96.02% (a second)
python3 erase/count.py --addition   # 1799426861128 2530881913560 71.10% (about 2 min)
python3 erase/count.py --alone  # 42060 38320568 1 in 911: hands with no center cards (about 10 s)
python3 erase/make_certs.py     # writes certificates.jsonl.gz (about 2.5 min)
python3 erase/verify_certs.py   # all 265,148 cases certified (about 10 s)
python3 erase/simulate.py       # simulated figures, 30,000 deals each (about 4 min)
python3 erase/targets.py --max 11 --certs   # targets 1 to 30 (about an hour)
python3 erase/verify_targets.py # checks targets_certs.jsonl.gz (about 15 min)
python3 erase/countdown.py      # Alliot's Countdown counts (about 3 min)
python3 erase/twenty_four.py    # 24-game counts for three division rules (about 10 s)
python3 erase/twenty_four.py 3 3 8 8   # one expression per rule, or "none"
python3 erase/twenty_four.py --check   # adds a brute-force check of every hand (about 2 min)
python3 erase/figures.py        # the figure in figures/ (under a second)
python3 erase/report.py deals countdown   # rewrites these sections of results.json; no names: all (over an hour)
uv run tools/sync_docs.py erase # fills the README's shared sections from docs/ and results.json

uv run erase/smt/z3_lemmas.py       # a few seconds
uv run erase/smt/z3_programs.py     # about 2.5 min
python3 erase/smt/statespace.py
```

Build the paper with `typst compile --root . --ignore-system-fonts --font-path erase/paper/fonts erase/paper/erase.typ`; add `--input edition=phone` for the phone edition, or `--features html --format html` for a web page. `paper/fonts/` holds the katakana of Noto Serif CJK JP (SIL Open Font License, `OFL.txt`), for one word in the sources. If the Typst package registry is unreachable, `uv run tools/typst_vendor.py erase/paper/erase.typ` fetches the same packages from the registry's GitHub repository and compiles with the `typst` Python package (`--all` builds all three editions).

## Files

| File | Contents |
|---|---|
| `erase.py` | Solver: target-directed search for an expression, and the hand-clearing search |
| `count.py` | Brute-force count of stuck deals; also the counts under the printed rules, with addition only, and with no center cards |
| `make_certs.py` | Writes the certificates for the theorem on big hands (`certificates.jsonl.gz`, 2.1 MB, not committed) |
| `verify_certs.py` | Independent checker for the certificates |
| `simulate.py` | Seeded simulations: addition only, + and −, pairs first, small groups, no brackets |
| `targets.py` | Thresholds for the targets 1 to 30, and certificates for them (`targets_certs.jsonl.gz`, not committed) |
| `verify_targets.py` | Independent checker for those certificates |
| `countdown.py` | Countdown numbers round: Alliot's counts of solvable problems, with a second method |
| `twenty_four.py` | The 24 game: exact counts under three division rules, with a brute-force check |
| `report.py`, `results.json` | Every number the README and the paper quote, from the scripts above; `results.json` is what `report.py` writes |
| `docs/` | The sections the README and the paper share, with placeholders for the numbers in `results.json` |
| `smt/` | z3 checks of the lemmas and of the impossible cases, plus a state-space search |
| `paper/` | Typst source, print and phone PDFs, and build script |
| `figures.py`, `figures/` | This README's figure, as SVG; a test checks it is current |
