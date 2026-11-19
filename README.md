# Games

Questions about games and puzzles, answered with code. Each answer says how it is known: exact (by counting or proof), or simulated (with the number of trials).

| Game | One result | Folder |
|---|---|---|
| **Boggle** | "toe" is the most common word, on 19.1% of random boards; a board holds 97.8 words and 140 points on average. Simulated, 15,000 boards. Of the 172,723 words of three or more letters in the word list, 12,004 cannot appear with these cubes at all. Exact, by matching letters to cubes. | [`boggle/`](boggle) |
| **Codenames** | A clue giver built on word embeddings expects about two of its words per clue from a guesser that shares its model, and gets 57% to 73% of that from a guesser built on another model. Exact expected values over 200 boards, for eight models. Fine-tuning a 0.6B encoder with LoRA on human games matches a linear map on a small static table at predicting human guesses (0.626 and 0.632 on 551 held-out turns) but does not beat it. Hosted gpt-oss-120b, untrained, picks the person's word in 0.650 of those turns, and four hosted language models give clues worth 0.29 to 0.89 words less per turn to unseen embedding guessers than an embedding giver's. A giver trained against six other models scores 0.31 words per turn (0.26 to 0.36) more with a seventh it never met than one trained against its own model, over 14 held-out choices and five seeds. Paired intervals over games, boards and seeds. | [`codenames/`](codenames) |
| **Drive Ya Nuts** | The original seven nuts fit together in exactly one way. Of all 59,487,568,920 sets of seven distinct nuts, 6.805% can be solved and 5.346% have exactly one solution. Exact, by counting every arrangement. | [`drive_ya_nuts/`](drive_ya_nuts) |
| **Egyptian Ratscrew** | Winning 55% of the slaps wins 64% of two-player games, and the player who lays the first card wins 49.77%, not half. Simulated, 20,000 and 1,000,000 games. | [`egyptian_ratscrew/`](egyptian_ratscrew) |
| **Erase** | 12,362,536 of the 2,530,881,913,560 deals of this make-ten game cannot be cleared, about 1 in 204,722. Exact, with a proof and machine-checked certificates. Ten is not special: for every target from 1 to 30, hands of 9 to 11 cards that cannot make it are all caught by a remainder argument. Exact, checked with certificates. | [`erase/`](erase) |
| **EV Nova and Naev** | In Naev, flying Ore and Medicine between Burnan and Lora-Mor, one jump apart, earns 394 credits per ton per jump. Counting jumps, with one landing per system visited, no loop of planets does better. In EV Nova the best is 217, and the routes players recommend include the best one for every landing cost tried. Exact, with a certificate of optimality. | [`ev_nova/`](ev_nova) |
| **Hanabi** | The hat-guessing strategy of Cox et al. (2015) averages 24.68 points with five players and plays a perfect game 76.3% of the time, as the paper reports; simulated, 100,000 games. If every card, the deck included, were in view, 98.62% to 98.67% of two-player deals could reach 25 (95% interval 98.45% to 98.82%); 20,000 seeded deals, checked by two independent solvers that agree on every deal both decide. On 30 seeded two-player deals, all winnable, and with the possible values of its own cards listed, gpt-oss-120b scores 18.6 points below 25 (95% interval 16.7 to 20.6), and DeepSeek-V3.2 loses all three lives in every game. | [`hanabi/`](hanabi) |
| **Mastermind** | The best strategy averages 5,625 / 1,296 ≈ 4.340 guesses, and the best that never needs a sixth guess 5,626 / 1,296, as Koyama and Lai (1993) found; the best of five published heuristics averages 4.373. Exact, by exhaustive search over all 1,296 codes. On Fallout 3 terminals of 20 short words, the best play cracks 92.5% of passwords within four attempts and random consistent guessing 67.8%. Simulated, 1,000 screens drawn from a word list. | [`mastermind/`](mastermind) |
| **Peggle** | Six wins in ten games make a player the better one with chance 743/1024, the chance of at most six heads in eleven fair coin flips. Playing equal players until one is 95% sure, for up to 100 games, declares someone better with chance 0.528; playing until the Bayes factor reaches 20 gives 0.028, and never more than 0.05 however long they play. Exact. | [`peggle/`](peggle) |
| **Pokémon** | No six attacking types are super-effective against all 18 types; ten sets of seven are. Played as a zero-sum duel, the chart has a unique equilibrium that leaves out 7 of the 18 types, as Schmitz (2021) found. A ranking of the types, led by Steel, accounts for only 4.7% to 5.2% of the chart; the rest is cyclic. Exact. On the Showdown ladder the equilibrium does not predict which typings people use: its Spearman correlation with usage is 0.32 below that of the number of Pokémon with each typing (95% bootstrap interval 0.08 to 0.56 below). | [`pokemon/`](pokemon) |
| **Tenzi** | A game takes 15.35 turns on average, or 15.29 when switching faces. No strategy does better, either on average or in the chance of being done by any given turn, so switching also wins races most often. Exact, by a Markov chain and a Markov decision process. | [`tenzi/`](tenzi) |

Each folder's README gives the rest of its results, its files and how to run them. Erase, Peggle and Tenzi have write-ups as PDFs.

## Usage

The repository is a [uv](https://docs.astral.sh/uv/) project and needs Python 3.12 or later.

```bash
uv sync --all-groups --no-group codenames-gpu   # every game's packages, plus pytest, ruff and ty (codenames-gpu is CUDA torch, for GPU jobs)
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

The write-ups in `erase/paper/`, `peggle/` and `tenzi/` are Typst. `uv run erase/paper/typst_vendor.py peggle/peggle.typ` builds one, even when the Typst package registry is unreachable. It uses only Typst's own fonts and any committed beside the paper, and the PDFs record no build date, so a rebuild gives the same bytes. CI rebuilds every PDF and fails if one differs from the committed file (`just papers`).

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
