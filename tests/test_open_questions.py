"""The repository-wide list of open questions (tools/open_questions.py and OPEN.md)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("open_questions", ROOT / "tools/open_questions.py")
assert _spec and _spec.loader
oq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oq)


def test_open_md_is_current():
    assert (ROOT / "OPEN.md").read_text(encoding="utf-8") == oq.render(), (
        "run python3 tools/open_questions.py"
    )


def test_headings_that_count():
    assert oq.is_open_heading("Open questions")
    assert oq.is_open_heading("Questions left open")
    assert not oq.is_open_heading("Yu-Gi-Oh! opening hands")
    assert not oq.is_open_heading("Reopened")


def test_bullets_under_the_section_only():
    readme = "\n".join(
        [
            "# Game",
            "- not a question",
            "## Open questions",
            "",
            "- First question,",
            "  continued on a second line.",
            "- Second | with a pipe.",
            "",
            "A closing paragraph.",
            "- after the paragraph",
            "### A subsection",
            "- in the subsection",
            "```",
            "# not a heading",
            "- in code",
            "```",
            "## Usage",
            "- not a question either",
        ]
    )
    assert oq.open_questions(readme) == [
        "First question, continued on a second line.",
        "Second | with a pipe.",
        "after the paragraph",
        "in the subsection",
    ]


def test_status_rules():
    assert oq.status("A list of real screens would settle it.") == "blocked on data"
    assert oq.status("A disk-backed search could decide them.") == "blocked on compute"
    assert oq.status("A comment reports 6,901,100; it was not checked here.") == "partly answered"
    assert oq.status("Does a plane of order 12 exist?") == "open"
    assert set(oq.STATUSES) == {"blocked on data", "blocked on compute", "partly answered", "open"}


def test_every_question_has_a_known_status():
    rows = [line for line in (ROOT / "OPEN.md").read_text().splitlines() if line.startswith("| [")]
    assert rows
    assert all(line.rsplit("|", 2)[1].strip() in oq.STATUSES for line in rows)


def test_paper_only_comments_are_left_out():
    readme = "## Open questions\n\n- Proved<!--raw-typst  (@thm-x)-->. Is it?\n"
    assert oq.open_questions(readme) == ["Proved. Is it?"]


def test_links_are_relative_to_the_game_readme(tmp_path):
    game = tmp_path / "cards"
    game.mkdir()
    (game / "README.md").write_text(
        "# Cards\n## Open questions\n"
        "- See [results](results/run.md#sample), [rules](#rules), "
        "[plan](../plans/run.md), [site](https://example.org/x), "
        "and `[example](example.md)`.\n"
    )
    index = oq.render(tmp_path)
    assert "[results](cards/results/run.md#sample)" in index
    assert "[rules](cards/README.md#rules)" in index
    assert "[plan](plans/run.md)" in index
    assert "[site](https://example.org/x)" in index
    assert "`[example](example.md)`" in index
