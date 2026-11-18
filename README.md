# Games

Questions about games and puzzles, answered with code. Each answer says how it is known: exact (by counting or proof), or simulated (with the number of trials).

| Game | One result | Folder |
|---|---|---|
| **Boggle** | "toe" is the most common word, on 19.1% of random boards; a board holds 97.8 words and 140 points on average. Simulated, 15,000 boards. | [`boggle/`](boggle) |
| **Drive Ya Nuts** | The original seven nuts fit together in exactly one way. Exact, by exhaustive search. | [`drive_ya_nuts/`](drive_ya_nuts) |
| **Egyptian Ratscrew** | Winning 55% of the slaps wins 64% of two-player games. Simulated, 4,000 games. | [`egyptian_ratscrew/`](egyptian_ratscrew) |
| **Erase** | 12,362,536 of the 2,530,881,913,560 deals of this make-ten game cannot be cleared, about 1 in 204,722. Exact, with a proof and machine-checked certificates. | [`erase/`](erase) |
| **Mastermind** | Guessing to split the remaining codes into the most parts averages 4.373 guesses; of the five strategies tried, only minimax never needs a sixth. Exact, over all 1,296 codes. | [`mastermind/`](mastermind) |
| **Peggle** | Six wins in ten games make a player the better one with chance 743/1024, the chance of at most six heads in eleven fair coin flips. Exact. | [`peggle/`](peggle) |
| **Pokémon** | No six attacking types are super-effective against all 18 types; ten sets of seven are. Exact. | [`pokemon/`](pokemon) |
| **Tenzi** | A game takes 15.35 turns on average, or 15.29 when switching faces. Exact, by a Markov chain. | [`tenzi/`](tenzi) |

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

The committed outputs are checked: CI reruns every notebook with [nbval](https://github.com/computationalmodelling/nbval) and fails if any text output differs, after replacing memory addresses (`tests/nbval_sanitize.cfg`). Figures are not compared. A cell tagged `nbval-ignore-output`, such as the one that prints the Python version, is run but not compared. To save new outputs after a change, run `uv run jupyter execute --inplace` on the notebook, or `just execute` for all of them.

```bash
uv run pytest --nbval --nbval-current-env --sanitize-with tests/nbval_sanitize.cfg boggle/boggle.ipynb
```

## License

The code is under the MIT license. The data files come from the sources named in each folder's README, and keep those sources' terms.
