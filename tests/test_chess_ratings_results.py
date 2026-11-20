"""Chess: committed results, figures and the README's quoted numbers and transcript."""

import hashlib
import json
from pathlib import Path

import pytest

from chess_ratings import figures, settle

HERE = Path(__file__).parent.parent / "chess_ratings"
RESULTS = HERE / "results"

# SHA-256 of the files that the Hugging Face jobs wrote, as the jobs printed them. The files in
# results/ were copied from the jobs' outputs; these hashes tie each copy to its run.
JOB_OUTPUTS = {
    "trajectories.json": "eba24bdbfbdd2f12a055a5f8bd506efbe0326b129b293dc2842496701ca54afd",
    "whatwould.txt": "db720985eca38a957b8d0dc45b69c2256126b971fe2b25b54fce2e09369fd0f4",
}


@pytest.mark.parametrize("name", sorted(JOB_OUTPUTS))
def test_copied_job_outputs_match_their_hashes(name):
    assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == JOB_OUTPUTS[name]


def test_comparison_with_lichess_is_what_the_script_writes():
    committed = json.loads((RESULTS / "compare.json").read_text())
    assert committed == json.loads(settle.to_text(settle.compare_with_lichess()))


@pytest.mark.parametrize("name", sorted(figures.FIGURES))
def test_committed_figure_is_current(name):
    committed = (figures.OUT / f"{name}.svg").read_bytes()
    assert figures.FIGURES[name]().encode("utf-8") == committed


def test_readme_transcript_is_the_job_output():
    readme = (HERE / "README.md").read_text()
    first = (RESULTS / "whatwould.txt").read_text().split("\n\n")[0]
    assert first in readme
