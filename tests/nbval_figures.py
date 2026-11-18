"""A pytest plugin that makes nbval compare each notebook's figures as well as its text.

nbval leaves PNG outputs out of its comparison unless it runs with --nbdime. Load this
plugin with `-p tests.nbval_figures` beside `--nbval`: after nbval has run a cell and
compared its text, the plugin compares the cell's PNG outputs with the committed ones,
so each notebook still runs once.

Two PNGs match if their bytes are equal, or else if the root mean square difference of
their pixels is at most TOLERANCE on the 0-255 scale (matplotlib.testing.compare). On the
GitHub runners the PNG bytes differed from locally saved ones even where every pixel was
equal, so the pixel comparison is the one that usually decides. A figure
drawn from the same data in another order differs by about 0.004 at most, since only the
antialiased edges of overlapping markers change; removing one small marker from a scatter
plot gives about 0.3. A cell tagged `nbval-ignore-output` is not compared, as in nbval.
"""

import base64
import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
from matplotlib.testing.compare import calculate_rms, compare_images
from matplotlib.testing.exceptions import ImageComparisonFailure
from PIL import Image

TOLERANCE = 0.05


def pngs(outputs) -> list[bytes]:
    """The decoded PNG images among a cell's outputs, in order."""
    return [
        base64.b64decode(out["data"]["image/png"])
        for out in outputs
        if "image/png" in out.get("data", {})
    ]


def rms_difference(expected: bytes, actual: bytes) -> float:
    """The root mean square difference of two PNGs' RGB pixels, on the 0-255 scale."""
    exp, act = (
        np.asarray(Image.open(io.BytesIO(png)).convert("RGB")) for png in (expected, actual)
    )
    return calculate_rms(exp.astype(float), act.astype(float))


# (notebook cell, figure number, RMS difference or None for equal bytes), for the summary.
COMPARED: list[tuple[str, int, float | None]] = []


def figure_mismatches(
    expected: list[bytes], actual: list[bytes], tol: float, where: Path, name: str = ""
) -> list[str]:
    """Describe each pair of PNGs that differ by more than tol; write the pair and their diff under where."""
    if len(expected) != len(actual):
        return [f"{len(actual)} figures, but the notebook has {len(expected)}"]
    problems = []
    for i, (exp, act) in enumerate(zip(expected, actual, strict=True), 1):
        if exp == act:
            COMPARED.append((name, i, None))
            continue
        try:
            rms = rms_difference(exp, act)
        except ImageComparisonFailure as e:  # the images have different sizes
            rms, err = float("inf"), str(e)
        COMPARED.append((name, i, rms))
        if rms <= tol:
            continue
        where.mkdir(parents=True, exist_ok=True)
        exp_path, act_path = where / f"figure{i}-expected.png", where / f"figure{i}.png"
        exp_path.write_bytes(exp)
        act_path.write_bytes(act)
        if rms != float("inf"):
            err = compare_images(str(exp_path), str(act_path), tol)  # also writes the diff
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
    problems = figure_mismatches(pngs(cell.outputs), pngs(outputs), TOLERANCE, where, item.nodeid)
    if problems:
        problems.append(f"The committed and new PNGs, and any diffs, are in {where}")
        pytest.fail("Cell figures differ:\n" + "\n".join(problems), pytrace=False)
    return result


def pytest_terminal_summary(terminalreporter):
    """Say how many figures matched byte for byte, and the largest difference among the rest."""
    if not COMPARED:
        return
    inexact = [(name, i, rms) for name, i, rms in COMPARED if rms is not None]
    line = (
        f"{len(COMPARED)} notebook figures compared, {len(COMPARED) - len(inexact)} byte for byte"
    )
    if inexact:
        name, i, rms = max(inexact, key=lambda t: t[2])
        line += f"; largest RMS difference {rms:.4f} (figure {i} of {name})"
    terminalreporter.write_line(line)
