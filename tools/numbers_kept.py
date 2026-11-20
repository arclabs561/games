#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Check that moving text out of a README lost no number.

Every number in the README as it was at a git revision must still appear in the README now,
in one of the game's fragments in docs/ (filled from results.json), or in the paper. The
paper is read from its HTML edition, which `uv run tools/typst_vendor.py <doc>.typ --all`
writes, so that numbers the paper reads from results.json are seen as printed.

    uv run tools/numbers_kept.py erase origin/main --paper erase/paper/erase.html
"""

import argparse
import html
import importlib.util
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_spec = importlib.util.spec_from_file_location("sync_docs", ROOT / "tools" / "sync_docs.py")
assert _spec and _spec.loader
sync_docs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync_docs)


def numbers(text: str) -> set[str]:
    return {n.rstrip(".,") for n in NUMBER.findall(text)}


def page_text(path: Path) -> str:
    """The text of an HTML page, with tags and style sheets removed."""
    s = re.sub(r"<style.*?</style>", " ", path.read_text(), flags=re.DOTALL)
    return html.unescape(re.sub(r"<[^>]+>", " ", s))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("game")
    ap.add_argument("rev", help="the git revision of the old README, e.g. origin/main")
    ap.add_argument("--paper", type=Path, action="append", default=[], help="HTML edition")
    a = ap.parse_args()
    old = subprocess.run(
        ["git", "show", f"{a.rev}:{a.game}/README.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    now = (ROOT / a.game / "README.md").read_text() + "\n".join(
        sync_docs.fragments(a.game).values()
    )
    papers = "\n".join(page_text(p) for p in a.paper)
    have_readme, have_paper = numbers(now), numbers(papers)
    lost = sorted(numbers(old) - have_readme - have_paper, key=lambda n: (len(n), n))
    moved = sorted(numbers(old) - have_readme, key=lambda n: (len(n), n))
    print(f"{a.game}: {len(numbers(old))} distinct numbers in the old README")
    print(f"  only in the paper now: {', '.join(n for n in moved if n not in lost) or 'none'}")
    if lost:
        raise SystemExit(f"  missing everywhere: {', '.join(lost)}")
    print("  every one is still in the README, a fragment or the paper")


if __name__ == "__main__":
    main()
