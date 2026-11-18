"""A pytest plugin that makes nbval compare each notebook's figures as well as its text.

nbval leaves PNG outputs out of its comparison unless it runs with --nbdime. Load this
plugin with `-p tests.nbval_figures` beside `--nbval`: after nbval has run a cell and
compared its text, the plugin compares the cell's PNG outputs with the committed ones,
so each notebook still runs once.

Two PNGs match if their bytes are equal, or else if the root mean square difference of
their pixels is at most TOLERANCE on the 0-255 scale (matplotlib.testing.compare). A figure
drawn from the same data in another order differs by about 0.004 at most, since only the
antialiased edges of overlapping markers change; removing one small marker from a scatter
plot gives about 0.3. A cell tagged `nbval-ignore-output` is not compared, as in nbval.
"""

import base64
import tempfile
from pathlib import Path

import pytest
from matplotlib.testing.compare import compare_images
from matplotlib.testing.exceptions import ImageComparisonFailure

TOLERANCE = 0.05


def pngs(outputs) -> list[bytes]:
    """The decoded PNG images among a cell's outputs, in order."""
    return [
        base64.b64decode(out["data"]["image/png"])
        for out in outputs
        if "image/png" in out.get("data", {})
    ]


def figure_mismatches(
    expected: list[bytes], actual: list[bytes], tol: float, where: Path
) -> list[str]:
    """Describe each pair of PNGs that differ by more than tol; write the pair and their diff under where."""
    if len(expected) != len(actual):
        return [f"{len(actual)} figures, but the notebook has {len(expected)}"]
    problems = []
    for i, (exp, act) in enumerate(zip(expected, actual, strict=True), 1):
        if exp == act:
            continue
        where.mkdir(parents=True, exist_ok=True)
        exp_path, act_path = where / f"figure{i}-expected.png", where / f"figure{i}.png"
        exp_path.write_bytes(exp)
        act_path.write_bytes(act)
        try:
            err = compare_images(str(exp_path), str(act_path), tol)
        except ImageComparisonFailure as e:  # the images have different sizes
            err = str(e)
        if err is None:
            for p in (exp_path, act_path):
                p.unlink()
        else:
            problems.append(f"figure {i}: {err}")
    return problems


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """After nbval runs and checks a cell, compare the cell's figures with the committed ones."""
    result = yield
    options, cell = getattr(item, "options", None), getattr(item, "cell", None)
    outputs = getattr(item, "test_outputs", None)
    if options is None or cell is None or outputs is None:
        return result  # not an nbval cell
    if not options["check"] or cell.execution_count is None:
        return result
    nb = Path(str(item.fspath))
    where = (
        Path(tempfile.gettempdir())
        / "nbval-figures"
        / nb.parent.name
        / f"{nb.stem}-cell{item.cell_num}"
    )
    problems = figure_mismatches(pngs(cell.outputs), pngs(outputs), TOLERANCE, where)
    if problems:
        problems.append(f"The committed and new PNGs, and any diffs, are in {where}")
        pytest.fail("Cell figures differ:\n" + "\n".join(problems), pytrace=False)
    return result
