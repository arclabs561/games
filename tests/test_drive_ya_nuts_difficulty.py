"""Search sizes: the counter against `solver.recur`, the C program against Python, the committed table."""

import csv
import json
import random
from pathlib import Path

import pytest

from drive_ya_nuts import difficulty, solver
from drive_ya_nuts.exact import canonical
from drive_ya_nuts.solver import Nut

HERE = Path(__file__).parent.parent / "drive_ya_nuts"


def random_sets(n: int, seed: int) -> list[list[tuple[int, ...]]]:
    rng = random.Random(seed)
    nuts = solver.all_nut_kinds()
    return [[x.order for x in rng.sample(nuts, 7)] for _ in range(n)]


def test_search_counts_the_calls_of_recur(monkeypatch):
    calls = 0
    recur = solver.recur

    def counted(*args):
        nonlocal calls
        calls += 1
        return recur(*args)

    monkeypatch.setattr(solver, "recur", counted)
    for orders in [difficulty.original(), *random_sets(60, seed=0)]:
        for s in range(6):
            turned = difficulty.turned(orders, s)
            calls = 0
            found = len(list(counted({Nut(o) for o in turned}, [], None)))
            assert difficulty.search(turned) == (calls, found)


def test_original_sizes():
    orders = difficulty.original()
    assert difficulty.solutions(orders) == 1
    assert difficulty.size(orders) == 573 and difficulty.size(orders, look_ahead=False) == 855


def test_size_is_the_same_for_every_set_of_a_class():
    rng = random.Random(1)
    for _ in range(20):
        orders = difficulty.random_one_solution_set(rng)
        perm = list(range(1, 7))
        rng.shuffle(perm)
        renamed = [canonical([perm[v - 1] for v in o]) for o in orders]
        mirrored = [canonical(o[::-1]) for o in orders]
        turned = [tuple(o[2:] + o[:2]) for o in orders]
        for look in (True, False):
            t = difficulty.size(orders, look)
            assert t == difficulty.size(renamed, look) == difficulty.size(mirrored, look)
            assert t == difficulty.size(turned, look)


def test_sampled_sets_have_one_solution():
    rng = random.Random(2)
    for _ in range(20):
        orders = difficulty.random_one_solution_set(rng)
        assert len(set(orders)) == 7 and solver.count_solutions({Nut(o) for o in orders}) == 1


@pytest.mark.skipif(difficulty.compiler() is None, reason="difficulty.c needs a C compiler")
def test_c_program_agrees_with_python():
    rng = random.Random(3)
    sets = random_sets(40, seed=4) + [difficulty.random_one_solution_set(rng) for _ in range(20)]
    for orders in [difficulty.original(), *sets]:
        expect = (
            difficulty.size(orders),
            difficulty.size(orders, False),
            difficulty.solutions(orders),
        )
        assert difficulty.c_sizes(orders) == expect


def committed() -> dict:
    return json.loads((HERE / "difficulty.json").read_text())


def test_committed_exact_pass_matches_the_exact_counts():
    data = committed()["exact"]
    with (HERE / "exact_counts.csv").open() as f:
        sets = {int(r["solutions"]): int(r["sets"]) for r in csv.DictReader(f)}
    assert data["edge_sequences"] == 1144 and data["rings"] == 1144 * 6**6
    assert data["distinct"] == 43_500_008
    # A set with k solutions has k rings, and each of its centers is as likely as any other,
    # so 120 / k times the rings around C0 in sets with k solutions counts those sets.
    assert data["solutions"][0] == 0
    for k, rings in enumerate(data["solutions"][1:21], start=1):
        assert rings * 120 == k * sets.get(k, 0), k
    assert data["one"] * 120 == sets[1]
    for k in ("look_ahead", "plain"):
        d = data[k]
        assert d["total"] == data["one"] == sum(c for _, c in d["sizes"])
        assert d["original"] == difficulty.size(difficulty.original(), k == "look_ahead")
        assert (d["below"], d["equal"]) == tuple(
            difficulty.percentile(d["original"], d["sizes"])[x] for x in ("below", "equal")
        )


def test_committed_sample_contains_the_exact_shares():
    data = committed()
    for k in ("look_ahead", "plain"):
        s, e = data["sampled"][k], data["exact"][k]
        assert s["total"] == data["sampled"]["samples"] == 200_000
        for hits, exact in (
            (s["below"], e["below"]),
            (s["below"] + s["equal"], e["below"] + e["equal"]),
        ):
            lo, hi = difficulty.wilson(hits, s["total"])
            assert lo <= exact / e["total"] <= hi


def test_committed_figure_is_current():
    committed_svg = (HERE / "figures" / "difficulty.svg").read_text()
    assert committed_svg == difficulty.figure(committed())


@pytest.mark.parametrize(
    ("partitions", "seeds", "error"),
    [
        ([], [], "no exact shards"),
        ([(0, 2), (1, 3)], [], "same partition"),
        ([(0, 2)], [], "same partition"),
        ([(0, 1)], [], "seeds"),
        ([(0, 1)], [*range(1, 10), 9], "seeds"),
    ],
)
def test_report_rejects_incomplete_runs(tmp_path, partitions, seeds, error):
    for i, (part, count) in enumerate(partitions):
        (tmp_path / f"difficulty-exact-{i}.json").write_text(
            json.dumps({"part": part, "parts": count})
        )
    for i, seed in enumerate(seeds):
        (tmp_path / f"difficulty-sample-{i}.json").write_text(
            json.dumps({"seed": seed, "samples": 20_000})
        )
    with pytest.raises(ValueError, match=error):
        difficulty.report(tmp_path)
