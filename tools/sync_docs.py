#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Fill the shared sections of each game's README from its Markdown fragments and results.

A game shares text between its README and its paper through small Markdown files in
`<game>/docs/`, for example `results.md`, `rules.md` and `open.md`. The README holds each
one between two marker comments, and this script writes what goes between them:

    <!-- shared:results -->
    ...the filled text of docs/results.md...
    <!-- /shared -->

The paper reads the same section out of the README and renders it with the Typst package
cmarker (`tools/shared_docs.typ`), so the README and the paper show the same words.

Numbers in a fragment are placeholders that name an entry of `<game>/results.json`, the file
the game's scripts write, so that each number is stored once:

    {{deals.stuck}}               12362536 -> 12,362,536 (whole numbers get commas)
    {{countdown.exact_percent}}   91.21, as stored
    {{twenty_four.deck_chance.fraction:.4f}}   a Python format spec after a colon
    {{targets.threshold_counts.8:words}}  a small whole number in words ("twenty")
    {{lucas_brute:and}}           [1, 24] -> "1 and 24"; ":or" gives "1 or 24", ":list" "1, 24"
    {{example:cards}}             [5, 5, 8] -> "5 5 8"
    {{no_zero.one_in:million}}    14716328 -> "14.7 million"

A path is a list of keys and list indices joined by dots. A key may contain spaces.

    uv run tools/sync_docs.py            # rewrite every README that has shared sections
    uv run tools/sync_docs.py erase      # one game
    uv run tools/sync_docs.py --check    # change nothing; fail if a README is out of date
"""

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER = re.compile(r"\{\{\s*([^{}:]+?)\s*(?::([^{}]*))?\}\}")
SECTION = re.compile(r"(<!-- shared:([\w-]+) -->\n)(.*?)(<!-- /shared -->)", re.DOTALL)
WORDS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()


def lookup(data, path: str):
    value = data
    for part in path.split("."):
        if isinstance(value, list):
            value = value[int(part)]
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise KeyError(path)
    return value


def number(value) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        return str(value)
    return f"{value:,}"


def english(items: list[str], word: str = "and") -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" {word} " + items[-1]


def render(value, spec: str | None) -> str:
    """One placeholder's text. Formatting lives here only; the paper reads the result."""
    if spec is None:
        return ", ".join(map(number, value)) if isinstance(value, list) else number(value)
    if spec in ("and", "or"):
        return english([number(v) for v in value], spec)
    if spec == "list":
        return ", ".join(number(v) for v in value)
    if spec == "cards":
        return " ".join(str(v) for v in value)
    if spec == "words":
        return WORDS[value] if 0 <= value < len(WORDS) else number(value)
    if spec == "million":
        return f"{value / 1e6:.1f} million"
    return format(value, spec)


def fill(text: str, data: dict, where: str) -> str:
    def one(m: re.Match) -> str:
        try:
            return render(lookup(data, m.group(1)), m.group(2))
        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise SystemExit(f"{where}: cannot fill {m.group(0)} ({type(e).__name__}: {e})") from e

    return PLACEHOLDER.sub(one, text)


def games() -> list[str]:
    """Folders with a README and Markdown fragments in docs/."""
    return sorted(
        p.parent.parent.name
        for p in ROOT.glob("*/docs/*.md")
        if (p.parent.parent / "README.md").exists()
    )


def fragments(game: str) -> dict[str, str]:
    """Each fragment of a game, with its placeholders filled from the game's results.json."""
    folder = ROOT / game
    results = folder / "results.json"
    data = json.loads(results.read_text()) if results.exists() else {}
    return {
        p.stem: fill(p.read_text(), data, str(p.relative_to(ROOT)))
        for p in sorted((folder / "docs").glob("*.md"))
    }


def synced(game: str) -> str:
    """The game's README with every shared section rewritten from its fragment."""
    readme = (ROOT / game / "README.md").read_text()
    filled = fragments(game)
    used = [m.group(2) for m in SECTION.finditer(readme)]
    if missing := sorted(set(used) - set(filled)):
        raise SystemExit(f"{game}/README.md: no fragment for {missing} in {game}/docs/")
    if unused := sorted(set(filled) - set(used)):
        raise SystemExit(f"{game}/docs/: {unused} not placed in the README (add the markers)")
    if len(used) != len(set(used)):
        raise SystemExit(f"{game}/README.md: a shared section appears twice")
    if readme.count("<!-- shared:") != len(used):
        raise SystemExit(f"{game}/README.md: a shared marker has no closing <!-- /shared -->")
    return SECTION.sub(
        lambda m: f"{m.group(1)}\n{filled[m.group(2)].strip()}\n\n{m.group(4)}", readme
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("games", nargs="*", help="game folders; default every one with docs/*.md")
    ap.add_argument("--check", action="store_true", help="fail if a README is out of date")
    a = ap.parse_args()
    stale = []
    for game in a.games or games():
        path = ROOT / game / "README.md"
        old, new = path.read_text(), synced(game)
        if old == new:
            continue
        if a.check:
            stale.append(game)
            sys.stdout.writelines(
                difflib.unified_diff(
                    old.splitlines(True), new.splitlines(True), str(path), "synced"
                )
            )
        else:
            path.write_text(new)
            print(f"wrote {path.relative_to(ROOT)}")
    if stale:
        raise SystemExit(f"out of date: {', '.join(stale)}; run uv run tools/sync_docs.py")


if __name__ == "__main__":
    main()
