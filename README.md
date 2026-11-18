# Games

Game analyses, solvers, and simulations.

## Catalog

| Game | Path | Content |
|---|---|---|
| **Boggle** | `boggle/` | Solver, board analysis (`boggle.ipynb`) |
| **Drive-Ya-Nuts** | `drive-ya-nuts/` | Solver for the tiling puzzle (`solver.py`) |
| **Egyptian Ratscrew** | `egyptian-ratscrew/` | Simulation (`egyptian-ratscrew.ipynb`) |
| **Erase** | `erase/` | Which make-ten deals cannot be cleared: solver, certificates, paper (`erase.py`, `paper/erase.pdf`) |
| **Mastermind** | `mastermind/` | Random codes and grading (`mastermind.py`); a filter for the Fallout 3 terminal word puzzle (`fallout3.py`) |
| **Peggle** | `peggle/` | Who is the better player after k wins in n games: exact odds and a one-line approximation (`peggle.py`, write-up `peggle.pdf`) |
| **Pokémon** | `pokemon/` | Type graph analysis (`graph.py`, `pokemon.ipynb`); IV, stat and capture-odds models for the Emerald Kaizo ROM hack (`emerald_kaizo_*.py`) |
| **Tenzi** | `tenzi/` | How many turns a game takes, exactly and by simulation (`tenzi.py`, write-up `tenzi.pdf`) |

`examples/` holds three standalone scripts: a Tenzi Monte Carlo, Mastermind strategies compared, and the Pokémon type chart as a directed graph.

## Usage

The repository is a [uv](https://docs.astral.sh/uv/) project and needs Python 3.12 or later.

```bash
uv sync                          # pytest, ruff and ty, from uv.lock
uv run pytest
uv run ruff check
TY_UV=scripts uv run ty check    # also checks each script below in its own environment
just ci                          # all four, if `just` is installed
```

Scripts that use only the standard library run with any Python 3.12 or later:

```bash
python3 drive-ya-nuts/solver.py
python3 erase/erase.py 4 4 6 9 5 8 5 --center 7 5 1 8
python3 pokemon/graph.py         # writes display/graph.json, then serves the page on port 8000
python3 peggle/peggle.py
```

Scripts that need other packages list them in an inline header (PEP 723), which `uv run` reads:

```bash
uv run tenzi/tenzi.py
uv run erase/smt/z3_lemmas.py
```

The write-ups in `erase/paper/`, `peggle/` and `tenzi/` are Typst; `uv run erase/paper/typst_vendor.py peggle/peggle.typ` builds one even when the Typst package registry is unreachable.

Each notebook's packages are a dependency group in `pyproject.toml`, named after its folder:

```bash
uv run --group boggle --with jupyter jupyter lab boggle/boggle.ipynb
```

The committed outputs are checked: CI reruns every notebook with [nbval](https://github.com/computationalmodelling/nbval) and fails if any output differs, after replacing memory addresses (`tests/nbval_sanitize.cfg`). To save new outputs after a change, run `jupyter execute --inplace` on the notebook, or `just execute` for all of them.

```bash
uv run pytest --nbval --sanitize-with tests/nbval_sanitize.cfg boggle/boggle.ipynb
```

`boggle/`, `drive-ya-nuts/` and `erase/` have their own READMEs.

## License

MIT
