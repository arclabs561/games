# Games

Game analyses, solvers, and simulations.

## Catalog

| Game | Path | Content |
|---|---|---|
| **Boggle** | `boggle/` | Solver, board analysis (`boggle.ipynb`) |
| **Codenames** | `codenames/` | Placeholder; `game.py` is empty |
| **Drive-Ya-Nuts** | `drive-ya-nuts/` | Solver for the tiling puzzle (`solver.py`) |
| **Egyptian Ratscrew** | `egyptian-ratscrew/` | Simulation (`egyptian-ratscrew.ipynb`) |
| **Erase** | `erase/` | Which make-ten deals cannot be cleared: solver, certificates, paper (`erase.py`, `paper/erase.pdf`) |
| **Mastermind** | `mastermind/` | Random codes and grading (`mastermind.py`); a filter for the Fallout 3 terminal word puzzle (`fallout3.py`) |
| **Peggle** | `peggle/` | Physics/probability analysis (`peggle.nb` for Mathematica, `peggle.lyx` for LyX; reference paper: Gutiérrez, Rodriguez and Sáez, [ETNA 11 (2000) 121-130](https://etna.ricam.oeaw.ac.at/vol.11.2000/pp121-130.dir/pp121-130.pdf)) |
| **Pokémon** | `pokemon/` | Type graph analysis (`graph.py`, `pokemon.ipynb`); IV, stat and capture-odds models for the Emerald Kaizo ROM hack (`emerald_kaizo_*.py`) |
| **Tenzi** | `tenzi/` | Dice game simulation (`tenzi.py`; notes in `tenzi.tm` for TeXmacs) |

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
```

Scripts that need other packages list them in an inline header (PEP 723), which `uv run` reads:

```bash
uv run tenzi/tenzi.py
uv run erase/smt/z3_lemmas.py
```

Each notebook's packages are a dependency group in `pyproject.toml`, named after its folder:

```bash
uv run --group boggle --with jupyter jupyter lab boggle/boggle.ipynb
```

`boggle/`, `drive-ya-nuts/` and `erase/` have their own READMEs.

## License

MIT
