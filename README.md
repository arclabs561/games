# Games

Game analyses, solvers, and simulations.

## Catalog

| Game | Path | Content |
|---|---|---|
| **Boggle** | `boggle/` | Solver, board analysis (`board.py`, `boggle.ipynb`) |
| **Drive-Ya-Nuts** | `drive_ya_nuts/` | Solver for the tiling puzzle (`solver.py`) |
| **Egyptian Ratscrew** | `egyptian_ratscrew/` | Simulation (`egyptian_ratscrew.ipynb`) |
| **Erase** | `erase/` | Which make-ten deals cannot be cleared: solver, certificates, paper (`erase.py`, `paper/erase.pdf`) |
| **Mastermind** | `mastermind/` | The reply to a guess (`mastermind.py`); five guessing strategies scored on every code (`strategies.py`); a filter for the Fallout 3 terminal word puzzle (`fallout3.py`) |
| **Peggle** | `peggle/` | Who is the better player after k wins in n games: exact odds and a one-line approximation (`peggle.py`, write-up `peggle.pdf`) |
| **Pokémon** | `pokemon/` | The type chart as a graph (`graph.py`, `graph_analysis.py`, `pokemon.ipynb`); IV, stat and capture-odds models for the Emerald Kaizo ROM hack (`emerald_kaizo/`) |
| **Tenzi** | `tenzi/` | How many turns a game takes, exactly and by simulation (`tenzi.py`, `monte_carlo.py`, write-up `tenzi.pdf`) |

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

The write-ups in `erase/paper/`, `peggle/` and `tenzi/` are Typst. `uv run erase/paper/typst_vendor.py peggle/peggle.typ` builds one, even when the Typst package registry is unreachable. It uses only Typst's own fonts and any committed beside the paper, and the PDFs record no build date, so a rebuild gives the same bytes; CI rebuilds every PDF and fails if one differs from the committed file (`just papers`).

A notebook uses its folder's dependency group:

```bash
uv run --group boggle --with jupyter jupyter lab boggle/boggle.ipynb
```

The committed outputs are checked: CI reruns every notebook with [nbval](https://github.com/computationalmodelling/nbval) and fails if any text output differs, after replacing memory addresses (`tests/nbval_sanitize.cfg`). Figures are not compared. A cell tagged `nbval-ignore-output`, such as the one that prints the Python version, is run but not compared. To save new outputs after a change, run `uv run jupyter execute --inplace` on the notebook, or `just execute` for all of them.

```bash
uv run pytest --nbval --nbval-current-env --sanitize-with tests/nbval_sanitize.cfg boggle/boggle.ipynb
```

`boggle/`, `drive_ya_nuts/` and `erase/` have their own READMEs.

## License

MIT
