# Games

Questions about games and puzzles, answered with code. Each answer says how it is known: proved by hand, checked by computer case by case, computed exactly, simulated (with the number of trials), or taken from a cited source. Each row below compresses its folder's README, which gives the qualifications.

| Game | One result | Folder |
|---|---|---|
| **Boggle** | "toe" is the most common word, on 19.1% of random boards; a board holds 97.8 words and 140 points on average. Simulated, 15,000 boards. Of the 172,723 words of three or more letters in the word list, 12,004 cannot appear with these cubes at all. Exact, by matching letters to cubes. Norvig's published board scores are reproduced, among them Boyan's 5 by 5 board at 10,112 points, and no change of one square or swap of two improves that board. Exact. The best 5 by 5 board found in 1,920 seeded searches scores 10,397 points, and it is the best board Vanderkam published; best found, not optimal. | [`boggle/`](boggle) |
| **Cannonballs** | A square pile of 24 layers holds 4,900 balls, which also make a 70 by 70 square, and no other pile but the single ball does, as Watson (1918) proved; a search along Pell equations finds no other below 10<sup>10,000</sup> layers. An oblong pile with a short side of a balls can hold a square number of balls for 7 in 12 values of a, and never for a = 2 or 3. Proved, and checked by exhaustive search. | [`cannonballs/`](cannonballs) |
| **Chess** | A new Lichess blitz rating loses its question mark after 10 or 11 games, in Lichess's own Glicko-2 and in real new players' rating changes (median 11). It is within 100 points of the player's strength nine times in ten only after 35 to 48 games (simulated, 20,000 players per strength), and a settled rating wanders with a standard deviation of about 31 points. Exploratory, with the Maia-3 model of human moves: rated from one game, players below 1400 come out about 170 to 200 points too high, and a fitted map of the rating scale halves that bias on new games. | [`chess_ratings/`](chess_ratings) |
| **Codenames** | A clue giver built on word embeddings expects about two of its words per clue from a guesser that shares its model, and gets 57% to 73% of that from a guesser built on another model. Exact expected values over 200 boards, for eight models. Fine-tuning a 0.6B encoder with LoRA on human games matches a linear map on a small static table at predicting human guesses (0.626 and 0.632 on 551 held-out turns) but does not beat it. Adding about 19,000 turns labeled by gpt-oss-120b raises the encoder to 0.659, a gain of 0.033 (interval 0.009 to 0.059), level with that teacher. Hosted gpt-oss-120b, untrained, picks the person's word in 0.650 of those turns, and four hosted language models give clues worth 0.29 to 0.89 words less per turn to unseen embedding guessers than an embedding giver's. A giver trained against six other models scores 0.31 words per turn (0.26 to 0.36) more with a seventh it never met than one trained against its own model, over 14 held-out choices and five seeds. Asked only to choose among 120 legal actions that the embedding giver proposes, six hosted models still do no better than its own first choice (−0.06 words per turn for the model selected on validation boards, interval −0.12 to −0.005). Qwen3.5-2B trained by GRPO on the same choice, with an exact reward, learns to name the first clue of the list for one word, and scores less with unseen guessers than a seven-feature policy trained on the same rewards (−0.11 words per turn, interval −0.16 to −0.06). Paired intervals over games, boards and seeds. | [`codenames/`](codenames) |
| **Drive Ya Nuts** | The original seven nuts fit together in exactly one way. Of all 59,487,568,920 sets of seven distinct nuts, 6.805% can be solved and 5.346%, about 1 in 18.7, have exactly one solution. Up to renaming the numbers there are 82,628,379 sets. If a nut may be repeated, 6,901,100 classes can be solved, as reported on Mathematics Stack Exchange. Computed exactly, by counting every arrangement and by Burnside's lemma. | [`drive_ya_nuts/`](drive_ya_nuts) |
| **Egyptian Ratscrew** | A player who lays an ace keeps the pile half the time, and one who lays a jack 81% (computed exactly). Winning 55% of the slaps wins 64% of two-player games, and the player who lays the first card wins 49.77%, slightly less than half, about one game in 435 (simulated, 20,000 and 1,000,000 games). Random slaps do not make every game end: six cards can loop forever without a slap. Proved, and exact for small decks; open for the full deck. | [`egyptian_ratscrew/`](egyptian_ratscrew) |
| **Erase** | Under the printed rules of this make-ten game, which erase pairs that sum to 10, 96.02% of deals are stuck, and with addition alone 71.10%. Under the most generous reading, which combines any number of cards with +, − and ×, 12,362,536 of the 2,530,881,913,560 deals are stuck, about 1 in 204,722. All three are exact; the last has a proof and machine-checked certificates. Ten is not special: for every target from 1 to 30, hands of 9 to 11 cards that cannot make it are all caught by a remainder argument. Exact, checked with certificates. | [`erase/`](erase) |
| **EV Nova and Naev** | In Naev, flying Ore and Medicine between Burnan and Lora-Mor, one jump apart, earns 394 credits per ton per jump. Counting jumps, with one landing per system visited, no loop of planets does better. In EV Nova the best is 217, and the routes players recommend include the best one for every landing cost tried. Exact, with a certificate of optimality. | [`ev_nova/`](ev_nova) |
| **FreeCell** | Of the first 32,000 Microsoft deals, only 11982 is lost with four free cells, and 199 are lost with three, as published. With two free cells 6,618 are lost and with one 25,710 (1 and 3 deals undecided, respectively); with none, 31,931. Computed exactly by two independent exhaustive solvers that agree on every deal both decide, and checked by brute force on small decks. | [`freecell/`](freecell) |
| **Hanabi** | If every card, the deck included, were in view, a share of 0.9862 of two-player deals could reach 25 (95% interval 0.9845 to 0.9877; 20,000 seeded deals, every one settled by two independent solvers and checked certificates). In 1,858 games by people on hanab.live, the deal ruled out only 0.013 of the 1.63 points by which teams fell short of 25, a lower bound on what the cards cost. Cox et al.'s hat strategy plays a perfect five-player game 76.3% of the time (simulated, 100,000 games; the paper says over 75 percent). Independent PPO policies cooperate better with a larger entropy bonus, but still score 0.34 points more with their own seed than with another (simulated, 95% intervals). | [`hanabi/`](hanabi) |
| **Maps** | Shelving the 139 CSS colors by hue gives 3.1 times the total color jump of the shortest ordering, 8.133 in Oklab. Exact, by an integer program, with Held-Karp and brute force checking it on small shelves. The shortest ordering is not the one that keeps similar colors closest: a one-dimensional t-SNE keeps more of each color's five nearest. Random projections of 400 word vectors need about 1,000 dimensions to keep every squared distance within 20%; simulated, 20 maps for each size. t-SNE, SNE and Kohonen maps written from their papers show the trade-off Venna and Kaski describe: t-SNE and the SOM are more trustworthy, SNE more continuous. Computed, and checked against scikit-learn. | [`maps/`](maps) |
| **Mastermind** | The best strategy averages 5,625 / 1,296 ≈ 4.340 guesses, as Koyama and Lai (1993) found; computed exactly by exhaustive search, which also matches Goddard's (2004) table for 17 smaller games. On Fallout 3 terminals of 20 short words, the best play cracks 92.5% of passwords within four attempts and 57.4% within three. Leaving the terminal resets it, so a player who backs out before the fourth guess is never locked out. Simulated on 1,000 screens drawn from a word list, exact on each screen. | [`mastermind/`](mastermind) |
| **Peggle** | With a uniform prior, a player who wins six of ten games is the better one with probability 743/1024, the chance of at most six heads in eleven fair coin flips. If two equal players play on until one posterior chance reaches 95%, for up to 100 games, someone is declared better with chance 0.528; playing until the Bayes factor reaches 20 gives 0.028, and never more than 0.05 however long they play. A best-of-seven match crowns a 55% player with chance 0.608. Computed exactly. | [`peggle/`](peggle) |
| **Pokémon** | No six attacking types are super-effective against all 18 types; ten sets of seven are. Played as a zero-sum duel, the chart has a unique equilibrium that leaves out 7 of the 18 types, as Schmitz (2021) found. A ranking of the types, led by Steel, accounts for only 4.7% to 5.2% of the chart; the rest is cyclic. Exact. On the Showdown ladder the equilibrium does not predict which typings people use: its Spearman correlation with usage is 0.32 below that of the number of Pokémon with each typing (95% bootstrap interval 0.08 to 0.56 below). | [`pokemon/`](pokemon) |
| **Spot It** | Take any 55 cards of 8 symbols from 57, every two sharing exactly one symbol: they complete in exactly one way to the projective plane of order 7, so in any such deck the symbol counts name the two missing cards. Proved, and checked on all 1,596 ways to remove two cards. A forum post of 2011 lists the two cards missing from a real box; both carry the snowman, as the proof requires. Three cards drawn share a symbol with chance exactly 6/55. | [`spot_it/`](spot_it) |
| **Tenzi** | A game takes 15.35 turns on average, or 15.29 when switching faces. No strategy does better on average, or is more likely to be done by any turn up to 163, after which no strategy can gain 10<sup>−12</sup>; that keeping the most common face is best on average was found before for up to seven dice. Computed exactly, by a Markov chain and a Markov decision process. | [`tenzi/`](tenzi) |
| **Yahtzee** | Optimal solitaire play scores 254.5896 on average, as Verhoeff (1999) found, with standard deviation 59.61 and median 248. A greedy player who plays each turn for its own points scores 35.5 to 37.3 less, depending on how ties are broken. An 8-rule cheat sheet a person can memorize scores 243.18, and no rule list found in this search comes within 9 points of optimal. Computed by dynamic programming over 536,448 states. | [`yahtzee/`](yahtzee) |
| **Yu-Gi-Oh!** | In a 40-card deck, 14 starters give at least one in the opening hand going first 90.00% of the time; going second, 12 reach 90.18%. The first few cards past 40 cost about 0.9 points each (12 starters: 85.06% in 40 cards), and the 20 out to 60 cost 0.82 on average (68.65% in 60). Computed exactly, and checked by brute force and a seeded simulation of 1,000,000 hands. | [`yugioh/`](yugioh) |

Each folder's README gives the rest of its results, its files and how to run them. [`OPEN.md`](OPEN.md) collects the open questions of every folder's README in one list, with a rough status for each; `python3 tools/open_questions.py` rewrites it. Cannonballs, Erase, Peggle and Tenzi have write-ups as PDFs; Erase's also has a [shared page](https://docs.google.com/document/d/1VdsI4VDYDi-eOqQqQE0bXnzDgYPOyanYno4YNuDiPMU) with the latest PDFs and a changelog.

## Usage

The repository is a [uv](https://docs.astral.sh/uv/) project and needs Python 3.12 or later.

```bash
uv sync --all-groups --no-group codenames-gpu --no-group hanabi-jax   # every game's packages, plus pytest, ruff and ty (the two groups left out are CUDA builds, for GPU jobs)
uv run pytest
uv run ruff check
TY_UV=scripts uv run ty check    # also checks each Erase script in its own environment
just ci                          # the checks CI runs, if `just` is installed
```

Each game except Erase is a Python package. Run a module from the repository root:

```bash
uv run python -m drive_ya_nuts.solver
uv run python -m hanabi.information
uv run python -m peggle.peggle
uv run python -m pokemon.graph   # writes display/graph.json, then serves the page on port 8000
uv run python -m tenzi.tenzi
```

A game's third-party packages are the dependency group named after its folder (`uv run --group tenzi ...`). The Erase scripts stay standalone, as the paper describes: most need only the standard library, and the z3 checks list their packages in an inline header (PEP 723) that `uv run` reads:

```bash
python3 erase/erase.py 4 4 6 9 5 8 5 --center 7 5 1 8
uv run erase/smt/z3_lemmas.py
```

The write-ups in `cannonballs/`, `erase/paper/`, `peggle/` and `tenzi/` are Typst. `uv run tools/typst_vendor.py peggle/peggle.typ` builds one, even when the Typst package registry is unreachable. It uses only Typst's own fonts and any committed beside the paper, and the PDFs record no build date, so a rebuild gives the same bytes. CI rebuilds every PDF and fails if one differs from the committed file (`just papers`).

A notebook uses its folder's dependency group:

```bash
uv run --group boggle --with jupyter jupyter lab boggle/boggle.ipynb
```

The committed outputs are checked: CI reruns every notebook with [nbval](https://github.com/computationalmodelling/nbval) and fails if any text output differs, after replacing memory addresses (`tests/nbval_sanitize.cfg`). The plugin `tests/nbval_figures.py` also compares each figure with the committed PNG in the same run. Two figures match if their bytes are equal, or if the root mean square difference of their pixels is at most 0.05 on the 0-255 scale. The bytes seldom match on the CI runners, even when every pixel does, so the pixel comparison is the one that counts. Drawing the same points in another order gives a difference of about 0.004, and removing one small marker from a scatter plot gives about 0.3. On a mismatch it writes both PNGs and their difference to a temporary folder and names it. A cell tagged `nbval-ignore-output`, such as the one that prints the Python version, is run but not compared. To save new outputs after a change, run `uv run jupyter execute --inplace` on the notebook, or `just execute` for all of them.

```bash
uv run pytest -p tests.nbval_figures --nbval --nbval-current-env --nbval-sanitize-with tests/nbval_sanitize.cfg boggle/boggle.ipynb
```

## License

The code is under the MIT license. The data files come from the sources named in each folder's README, and keep those sources' terms.
