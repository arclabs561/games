"""Collect every game's open questions into OPEN.md at the repository root.

    python3 tools/open_questions.py           # rewrites OPEN.md
    python3 tools/open_questions.py --check   # exits 1 if OPEN.md is not what it would write

A game is a folder with a README.md. Its open questions are the top-level bullets under any
heading whose words include "open" (so "Open questions" and "Questions left open" count, and
"opening hands" does not), up to the next heading of the same or a higher level. Each question
gets one status, assigned from its text by the first rule that matches:

1. **blocked on data**: it asks for data the repository does not have, such as a list of real
   screens, a card list, a running copy of a game, an experiment with people, or a source
   with clear terms.
2. **blocked on compute**: it names a search or a run that was too large, such as a disk-backed
   search, larger budgets, or more training steps.
3. **partly answered**: it reports a partial result, such as a bound, a reported count that was
   not checked, or a case that was settled while others were not.
4. **open**: anything else.

The rules are phrases fitted by hand to the questions as they stand, and they read words, not
meaning, so a status is a guide for choosing work, not a finding. A new question may need a new
phrase.

Each game is named as in the root README's table, and relative links in a question are rewritten
so that they resolve from the repository root. Every question should be one a reader can take
alone, ending in a question mark; the script prints a warning for each bullet without one.
Standard library only.
"""

import argparse
import posixpath
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "OPEN.md"
NOT_GAMES = {"experiments", "tools", "tests"}

RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "blocked on data",
        re.compile(
            r"list of real|card list|running copy|with people|than with embedding|"
            r"source with clear terms|matchup table is an image|evaluation service|"
            r"real screens|a list of|tournament decks|not measurements",
            re.IGNORECASE,
        ),
    ),
    (
        "blocked on compute",
        re.compile(
            r"disk-backed|larger budgets|more steps|\b16 GB\b|hours an arm|"
            r"was started and cancelled|not enough to decide|more training",
            re.IGNORECASE,
        ),
    ),
    (
        "partly answered",
        re.compile(
            r"\breports\b|not checked|it was not|allows at most|found \d+|only \d+ reaches|"
            r"worked out only|tried counting|only the simulated|remainders rule out|"
            r"the free-order game|is the hardest|have not been tried|has not been checked|"
            r"the source comments",
            re.IGNORECASE,
        ),
    ),
]
STATUSES = [name for name, _ in RULES] + ["open"]


def status(question: str) -> str:
    """The status of one question, by the first rule whose words it contains."""
    for name, pattern in RULES:
        if pattern.search(question):
            return name
    return "open"


def heading(line: str) -> tuple[int, str] | None:
    """The level and text of a Markdown heading, or None."""
    m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
    return (len(m[1]), m[2]) if m else None


def is_open_heading(text: str) -> bool:
    return re.search(r"\bopen\b", text, re.IGNORECASE) is not None


def open_questions(readme: str) -> list[str]:
    """The top-level bullets under every open-questions heading, each joined into one line."""
    questions: list[str] = []
    level = None  # the level of the open section we are in, if any
    in_code = False
    current: list[str] | None = None

    def flush() -> None:
        nonlocal current
        if current:
            # HTML comments, such as the paper-only cross-references of shared sections, are
            # hidden on GitHub, so they are left out here too
            questions.append(re.sub(r"\s*<!--.*?-->", "", " ".join(current)))
        current = None

    for line in readme.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        h = None if in_code else heading(line)
        if h:
            if level is not None and h[0] <= level:
                flush()
                level = None
            if level is None and is_open_heading(h[1]):
                level = h[0]
            continue
        if level is None or in_code:
            continue
        if line.startswith(("- ", "* ")):
            flush()
            current = [line[2:].strip()]
        elif current is not None and line.startswith("  ") and line.strip():
            current.append(line.strip())
        elif not line.strip():
            continue
        else:
            flush()  # a paragraph ends the list item
    flush()
    return questions


def games(root: Path = ROOT) -> list[Path]:
    """Every game folder: a folder with a README.md, in alphabetical order."""
    return sorted(
        p.parent
        for p in root.glob("*/README.md")
        if p.parent.name not in NOT_GAMES and not p.parent.name.startswith(".")
    )


def title(readme: str, folder: str) -> str:
    """The README's first heading, or the folder name."""
    for line in readme.splitlines():
        h = heading(line)
        if h and h[0] == 1:
            return h[1]
    return folder


LINK = re.compile(r"(\]\()([^)\s]+)(\))")
ROW_NAME = re.compile(r"^\|\s*\*\*(.+?)\*\*\s*\|.*\]\(([\w-]+)/?\)\s*\|\s*$")


def rebase_links(text: str, folder: str) -> str:
    """Rewrite the relative links of a question so that they resolve from the repository root."""

    def fix(m: re.Match[str]) -> str:
        target = m[2]
        if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE) or target.startswith("/"):
            return m[0]
        if target.startswith("#"):
            return f"{m[1]}{folder}/README.md{target}{m[3]}"
        path, sep, anchor = target.partition("#")
        return f"{m[1]}{posixpath.normpath(posixpath.join(folder, path))}{sep}{anchor}{m[3]}"

    return LINK.sub(fix, text)


def root_names(root: Path = ROOT) -> dict[str, str]:
    """Each folder's game name as the root README's table gives it, in bold in the first column."""
    readme = root / "README.md"
    if not readme.exists():
        return {}
    names = {}
    for line in readme.read_text(encoding="utf-8").splitlines():
        m = ROW_NAME.match(line)
        if m:
            names[m[2]] = m[1]
    return names


def is_question(text: str) -> bool:
    """Whether an open-questions bullet asks something, by its question mark."""
    return "?" in text


def cell(text: str) -> str:
    """Text safe inside a Markdown table cell."""
    return text.replace("|", "\\|")


def render(root: Path = ROOT) -> str:
    rows: list[tuple[str, str, str, str]] = []
    without: list[str] = []
    names = root_names(root)
    for folder in games(root):
        readme = (folder / "README.md").read_text(encoding="utf-8")
        qs = [rebase_links(q, folder.name) for q in open_questions(readme)]
        name = names.get(folder.name) or title(readme, folder.name)
        if not qs:
            without.append(f"[{name}]({folder.name}/README.md)")
        for q in qs:
            rows.append((folder.name, name, q, status(q)))
    counts = {s: sum(r[3] == s for r in rows) for s in STATUSES}
    lines = [
        "# Open questions",
        "",
        f"{len(rows)} open questions from {len(games(root)) - len(without)} games' READMEs, "
        "collected by `tools/open_questions.py`. Edit the READMEs, not this file, then run "
        "`python3 tools/open_questions.py`; a test checks that this file is current.",
        "",
        "Each status comes from the question's wording, by phrase rules fitted by hand and listed "
        "in the script's docstring. It guides the choice of what to work on next and is not "
        "itself a finding.",
        "",
        "| Status | Questions |",
        "|---|---:|",
        *(f"| {s} | {counts[s]} |" for s in STATUSES),
        "",
    ]
    if without:
        lines += [f"No open-questions section: {', '.join(without)}.", ""]
    lines += ["| Game | Question | Status |", "|---|---|---|"]
    for folder, name, q, s in rows:
        lines.append(f"| [{cell(name)}]({folder}/README.md) | {cell(q)} | {s} |")
    return "\n".join(lines) + "\n"


def not_questions(root: Path = ROOT) -> list[tuple[str, str]]:
    """The open-questions bullets, by folder, that have no question mark."""
    return [
        (folder.name, q)
        for folder in games(root)
        for q in open_questions((folder / "README.md").read_text(encoding="utf-8"))
        if not is_question(q)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with OPEN.md, write nothing")
    args = parser.parse_args(argv)
    text = render()
    for folder, q in not_questions():
        print(f"warning: {folder}: not a question: {q[:100]}", file=sys.stderr)
    if args.check:
        same = OUT.exists() and OUT.read_text(encoding="utf-8") == text
        print("OPEN.md is current" if same else "OPEN.md is out of date")
        return 0 if same else 1
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
