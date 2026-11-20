"""Build and run Solver B (`solver_b.c`), and read its answers as Python values.

The C program is compiled once with the system's C compiler (`$CC`, else `cc`, with `-O2`)
into a cache folder of the user's own (`$XDG_CACHE_HOME/games`, else `~/.cache/games`, mode
0700). The file is named after a hash of the source, the compiler command and the compiler's
`--version`, so an edited source or another compiler is rebuilt. `$CC` may carry flags, as in
`CC="gcc -fsanitize=address"`. It reads one problem per
line and writes one answer per line; `run` sends a batch and parses the answers. A won line
comes back as card moves in the form of `freecell.rules`, which `rules.replay` checks.
"""

import hashlib
import os
import shlex
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

from freecell.deals import Card, Layout
from freecell.rules import Move, Rules

SOURCE = Path(__file__).with_name("solver_b.c")
DEST = {"h": "home", "c": "cell", "e": "empty"}


@dataclass
class Answer:
    """Solver B's verdict on one problem."""

    verdict: str
    states: int
    line: list[Move] = field(default_factory=list)
    most_home: int = 0
    frozen: frozenset[Card] = frozenset()
    fence: list[tuple[int, ...]] | None = None


def compiler() -> str | None:
    """The C compiler command, if there is one: `$CC` as given, else cc or gcc."""
    return os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")


def compiler_argv() -> list[str] | None:
    """The compiler command split into words, so that `$CC` may carry flags."""
    cc = compiler()
    return shlex.split(cc) if cc else None


def cache_dir() -> Path:
    """A folder only this user can write to, for compiled programs.

    A shared folder such as /tmp would let another user put a program under the expected
    name first, and it would then be run.
    """
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    folder = base / "games"
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    st = folder.stat()
    if st.st_uid != os.getuid():
        raise RuntimeError(f"{folder} belongs to another user")
    if st.st_mode & 0o077:
        folder.chmod(0o700)
    return folder


def build_key(source: Path, argv: list[str]) -> str:
    """A hash of the source, the compiler command and the compiler's --version output."""
    try:
        version = subprocess.run(
            [*argv, "--version"], capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        version = ""
    h = hashlib.sha256(source.read_bytes())
    for part in ("\0".join(argv), version):
        h.update(b"\0" + part.encode())
    return h.hexdigest()[:16]


def binary(source: Path = SOURCE) -> Path:
    """Path of the compiled solver (Solver B by default), compiling it if needed."""
    argv = compiler_argv()
    if argv is None:
        raise RuntimeError(f"{source.name} needs a C compiler (cc)")
    out = cache_dir() / f"freecell-{source.stem}-{build_key(source, argv)}"
    if not out.exists():
        tmp = out.with_suffix(f".{os.getpid()}.{threading.get_ident()}")
        subprocess.run([*argv, "-O2", "-o", str(tmp), str(source)], check=True)
        tmp.replace(out)
    return out


def ms_problem(deal: int, cells: int) -> str:
    """Input line for Microsoft deal `deal` with `cells` free cells."""
    return f"ms {deal} {cells}"


def deck_problem(rules: Rules, layout: Layout) -> str:
    """Input line for an explicit layout under `rules`."""
    cols = " | ".join(" ".join(map(str, col)) for col in layout)
    return f"deck {rules.ranks} {len(rules.suits)} {rules.columns} {rules.cells} {cols}"


def _parse(words: list[str]) -> Answer:
    verdict, states = words[0], int(words[1])
    if verdict == "won":
        line: list[Move] = []
        for w in words[2:]:
            c, d = w.split(">")
            line.append((int(c), DEST[d] if d in DEST else int(d)))
        return Answer(verdict, states, line)
    if verdict == "lost":
        frozen = frozenset(int(w) for w in words[3:])
        return Answer(verdict, states, most_home=int(words[2]), frozen=frozen)
    if verdict == "unknown":
        return Answer(verdict, states)
    raise RuntimeError("Solver B: " + " ".join(words))


def run(
    problems: list[str],
    budget: int = 0,
    seed: int = 0,
    fence: bool = False,
    autoplay: bool = False,
) -> list[Answer]:
    """Solve each problem; `budget` caps stored positions (0 for none)."""
    args = [str(binary()), str(budget), str(seed)]
    args += (["fence"] if fence else []) + (["auto"] if autoplay else [])
    proc = subprocess.run(
        args, input="".join(p + "\n" for p in problems), capture_output=True, text=True
    )
    if proc.returncode:
        raise RuntimeError(f"Solver B failed: {proc.stderr.strip()}")
    lines = iter(proc.stdout.splitlines())
    out = []
    for text in lines:
        ans = _parse(text.split())
        if fence and ans.verdict == "lost":
            ans.fence = []
            for row in lines:
                if row == "end":
                    break
                ans.fence.append(tuple(map(int, row.split())))
        out.append(ans)
    if len(out) != len(problems):
        raise RuntimeError("Solver B gave too few answers")
    return out


def decide(
    problems: list[str],
    first: int = 2_000_000,
    restarts: int = 8,
    each: int = 1_000_000,
    cap: int = 0,
    autoplay: bool = False,
) -> list[Answer]:
    """Verdicts with restarts, as `solver.decide`: a budgeted search, reseeded searches for
    the problems still open, then one search with budget `cap` (0 for none)."""
    answers = run(problems, budget=first, autoplay=autoplay)
    for seed in range(1, restarts + 1):
        todo = [i for i, a in enumerate(answers) if a.verdict == "unknown"]
        if not todo:
            return answers
        for i, a in zip(
            todo,
            run([problems[i] for i in todo], budget=each, seed=seed, autoplay=autoplay),
            strict=True,
        ):
            answers[i] = a
    todo = [i for i, a in enumerate(answers) if a.verdict == "unknown"]
    if todo:
        for i, a in zip(
            todo, run([problems[i] for i in todo], budget=cap, autoplay=autoplay), strict=True
        ):
            answers[i] = a
    return answers
