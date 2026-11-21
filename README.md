# Games

Questions about games and puzzles, answered with code.

| Game | One result | Folder |
|---|---|---|
| **Boggle** | On 15,000 simulated boards, “toe” appeared more often than any other word, on 19.1% of boards. | [`boggle/`](boggle) |
| **Cannonballs** | Stack 24 square layers and you get 4,900 balls. Lay them flat and they make a 70 by 70 square. Watson proved in 1918 that the only other square pile that does this is a single ball. | [`cannonballs/`](cannonballs) |
| **Chess** | When does a new blitz rating lose its question mark? Of 963 Lichess newcomers followed for at least 30 games, 960 crossed the estimated cutoff. Among those 960 the median was 11 games. | [`chess_ratings/`](chess_ratings) |
| **Codenames** | Can you guess what someone else picked? Given the clue and the words left on the board, gpt-oss-120b matched the player on 358 of 551 single-pick turns in 68 held-out games. | [`codenames/`](codenames) |
| **Drive Ya Nuts** | The seven original nuts fit together in exactly one way. Among all solvable sets of seven distinct nuts, about 79% also have a unique solution. The search counts every arrangement. | [`drive_ya_nuts/`](drive_ya_nuts) |
| **Egyptian Ratscrew** | Deal nine particular cards and slap every pair or sandwich. Either player can win the first slap, yet neither can ever win the game. An exhaustive search follows both outcomes of every slap. | [`egyptian_ratscrew/`](egyptian_ratscrew) |
| **Erase** | Can every deal make ten? Even allowing addition, subtraction and multiplication, an exact count leaves 12,362,536 stuck deals out of 2,530,881,913,560, about one in 204,722. | [`erase/`](erase) |
| **EV Nova** | Buy Equipment at Port Kane and sell it at Altia. Bring Medical Supplies back. The round trip pays 217 credits per ton per jump. Two routes tie it at the start of EV Nova 1.1.1, counting jumps with one landing per system visit. | [`ev_nova/`](ev_nova) |
| **FreeCell** | Microsoft deal 11982 is the only one of the first 32,000 that cannot be won with four free cells. With three free cells, exhaustive search finds 199 losses. | [`freecell/`](freecell) |
| **Hanabi** | Could a team that scored 23 have made 25? An open-hand search found a perfect line on 282 of the 287 two-player deals that ended on 23 in the human game records. | [`hanabi/`](hanabi) |
| **Maps** | Sorting books by hue puts the grays at the red end. For 139 CSS colors, the jumps between neighbors add up to 3.1 times the smallest possible total in Oklab. | [`maps/`](maps) |
| **Mastermind** | Play each of the 1,296 possible codes once. The best strategy takes 5,625 guesses in total. Guaranteeing a win within five guesses raises that total to 5,626. | [`mastermind/`](mastermind) |
| **Peggle** | Win six games out of ten. If each game has the same independent win probability and all win rates started equally plausible, the chance that yours exceeds one half is 743/1024, about 73%. | [`peggle/`](peggle) |
| **Pokémon** | How many attacking types cover all 18 defending types for super-effective damage? Counting every set gives ten ways with seven types and none with six. | [`pokemon/`](pokemon) |
| **Spot It** | One symbol appears only six times on the 55 cards. It goes on both missing cards. The other symbol counts and which pairs appear together determine the rest. | [`spot_it/`](spot_it) |
| **Tenzi** | Is it worth switching which face you keep? In the ten-die game the best switching strategy averages 15.29 rolls, against 15.35 when you stick with your first choice. | [`tenzi/`](tenzi) |
| **Yahtzee** | Give each player their own dice and scorecard. A player aiming for the highest average score beats the eight-rule cheat sheet in 57.03% of games. Counting both score distributions includes ties. | [`yahtzee/`](yahtzee) |
| **Yu-Gi-Oh!** | Going second in a 40-card deck with 12 starters and 12 separate hand traps, you have a trap for the opponent’s first turn and a starter by your own first draw 75.69% of the time. | [`yugioh/`](yugioh) |

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
