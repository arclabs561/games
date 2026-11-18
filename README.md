# Games

Questions about games and puzzles, answered with code. Each answer says how it is known: exact (by counting or proof), or simulated (with the number of trials).

| Game | One result | Folder |
|---|---|---|
| **Boggle** | "toe" is the most common word, on 19.1% of random boards; a board holds 97.8 words and 140 points on average. Simulated, 15,000 boards. | [`boggle/`](boggle) |
| **Drive Ya Nuts** | The original seven nuts fit together in exactly one way. Exact, by exhaustive search. | [`drive_ya_nuts/`](drive_ya_nuts) |
| **Egyptian Ratscrew** | Winning 55% of the slaps wins 64% of two-player games, and the player who lays the first card wins 49.77%, not half. Simulated, 20,000 and 1,000,000 games. | [`egyptian_ratscrew/`](egyptian_ratscrew) |
| **Erase** | 12,362,536 of the 2,530,881,913,560 deals of this make-ten game cannot be cleared, about 1 in 204,722. Exact, with a proof and machine-checked certificates. | [`erase/`](erase) |
| **Mastermind** | Guessing to split the remaining codes into the most parts averages 4.373 guesses; of the five strategies tried, only minimax never needs a sixth. Exact, over all 1,296 codes. On Fallout 3 terminals of 20 short words, the best play cracks 92.5% of passwords within four attempts and random consistent guessing 67.8%. Simulated, 1,000 screens drawn from a word list. | [`mastermind/`](mastermind) |
| **Peggle** | Six wins in ten games make a player the better one with chance 743/1024, the chance of at most six heads in eleven fair coin flips. Playing equal players until one is 95% sure, for up to 100 games, declares someone better with chance 0.528. Exact. | [`peggle/`](peggle) |
| **Pokémon** | No six attacking types are super-effective against all 18 types; ten sets of seven are. Played as a zero-sum duel, the chart has a unique equilibrium that leaves out 7 of the 18 types, as Schmitz (2021) found. Exact. | [`pokemon/`](pokemon) |
| **Tenzi** | A game takes 15.35 turns on average, or 15.29 when switching faces, and no strategy does better. Exact, by a Markov chain and a Markov decision process. | [`tenzi/`](tenzi) |

Each folder's README gives the rest of its results, its files and how to run them. Erase, Peggle and Tenzi have write-ups as PDFs.

## Usage

The repository is a [uv](https://docs.astral.sh/uv/) project and needs Python 3.12 or later.

```bash
uv sync --all-groups             # every game's packages, plus pytest, ruff and ty
uv run pytest
uv run ruff check
TY_UV=scripts uv run ty check    # also checks each Erase script in its own environment
just ci                          # the checks CI runs, if `just` is installed
```

Each game except Erase is a Python package. Run a module from the repository root:

```bash
uv run python -m drive_ya_nuts.solver
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
