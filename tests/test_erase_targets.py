"""Other targets than 10 (erase/targets.py, verify_targets.py) and Countdown (countdown.py)."""

import gzip
import json
import random
from itertools import combinations_with_replacement

import countdown
import pytest
import targets
import verify_targets

from erase import in_families, multisets, table


def test_residues_match_the_values_of_small_multisets():
    """The remainders targets.residues finds are those of the actual values, for 1 to 4 cards."""
    for n in range(1, 5):
        for cards in combinations_with_replacement(range(1, 11), n):
            values = set(table(cards))
            for m in (2, 3, 4, 5, 7, 12):
                rs = tuple(sorted(c % m for c in cards))
                assert targets.residues(rs, m) == sum(1 << r for r in {v % m for v in values})


def test_both_remainder_searches_agree():
    rng = random.Random(5)
    for _ in range(200):
        cards = tuple(sorted(rng.randint(1, 10) for _ in range(rng.randint(1, 7))))
        t = rng.randint(1, 30)
        assert bool(targets.blocking_moduli(cards, t)) == verify_targets.blocked(cards, t)


def test_ten_matches_the_paper():
    """For 10, the blocked multisets of 7 to 9 cards are the paper's two families, and the
    only exception is {1, 6, 6, 7, 7, 7, 7}."""
    r = targets.study(10, 9)
    assert targets.threshold(r["exceptions"]) == 8
    assert r["exceptions"][7] == [(1, 6, 6, 7, 7, 7, 7)]
    for n in (8, 9):
        assert set(r["least_moduli"][n]) == {3, 4}
        family = [c for c in multisets(n, [0] + [4] * 10) if in_families(c)]
        assert all(targets.blocking_moduli(c, 10) for c in family)
        assert sum(r["least_moduli"][n].values()) == len(family)  # and nothing else is blocked


def test_checker_accepts_fresh_certificates_and_rejects_a_bad_one(tmp_path, capsys):
    certs = []
    r = targets.study(7, 7, certs)
    s = targets.threshold(r["exceptions"])
    path = tmp_path / "c.jsonl.gz"
    lines: list[dict[str, object]] = [{"target": 7, "threshold": s, "max": 7}]
    lines += [{"target": 7, "cards": c, "expr": e} for n, c, e in certs if n >= s]
    with gzip.open(path, "wt") as f:
        f.writelines(json.dumps(x) + "\n" for x in lines)
    verify_targets.main(path, targets=[7], max_cards=7)
    assert "t= 7: sizes 7 to 7" in capsys.readouterr().out
    lines[1]["expr"] = str(lines[1]["expr"]).replace("+", "*", 1)
    with gzip.open(path, "wt") as f:
        f.writelines(json.dumps(x) + "\n" for x in lines)
    with pytest.raises(ValueError, match="wrong expression"):
        verify_targets.main(path, targets=[7], max_cards=7)


def test_countdown_values_match_a_search_over_moves():
    sels = countdown.selections()
    assert len(sels) == 13_243
    for s in sels[::1500] + [(1, 1, 2, 2, 3, 3)]:
        assert countdown.reachable(s) == countdown.reachable_by_moves(s)
    values = sorted(countdown.reachable((1, 1, 2, 2, 3, 3)))
    assert not set(values) & set(countdown.TARGETS)  # Alliot: this selection reaches no target
    assert countdown.distance(values, 101) == 101 - max(values)
    assert countdown.combine(6, 3) == {9, 18, 3, 2} and countdown.combine(4, 4) == {8, 16, 1}


def test_fourteen_has_an_exception_of_eight_cards():
    """9 9 9 9 10 10 10 10 cannot make 14, by listing every value of those cards; only a
    9 or a 10 differs between parts, so each part is a count of 9s and a count of 10s."""
    from functools import cache

    @cache
    def values(nines: int, tens: int) -> frozenset[int]:
        if nines + tens == 1:
            return frozenset([9 if nines else 10])
        out = set()
        for i in range(nines + 1):
            for j in range(tens + 1):
                if 0 < i + j < nines + tens:
                    for x in values(i, j):
                        for y in values(nines - i, tens - j):
                            out |= {x + y, x - y, x * y}
        return frozenset(out)

    cards = (9, 9, 9, 9, 10, 10, 10, 10)
    assert len(values(4, 4)) == 17_982 and 14 not in values(4, 4)
    assert targets.make(cards, 14) is None and not targets.blocking_moduli(cards, 14)


@pytest.mark.parametrize(
    ("records", "error"),
    [
        ([], "missing targets"),
        ([{"target": 14, "threshold": 12, "max": 11}], "expected sizes"),
        ([{"target": 7, "threshold": 7, "max": 11}], "missing targets"),
        (2 * [{"target": 7, "threshold": 7, "max": 11}], "duplicate target"),
        ([{"target": 7, "threshold": 7, "max": 10}], "expected sizes"),
        (
            [{"target": 7, "threshold": 7, "max": 11}]
            + 2 * [{"target": 7, "cards": [2, 5], "expr": "2+5"}],
            "duplicate expression",
        ),
    ],
)
def test_release_checker_rejects_incomplete_or_duplicate_records(tmp_path, records, error):
    path = tmp_path / "targets.gz"
    with gzip.open(path, "wt") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    with pytest.raises(ValueError, match=error):
        verify_targets.main(path)


@pytest.mark.parametrize("example", [[4, 6], [3, 3], [5, 5, 5, 5, 5]])
def test_threshold_needs_a_legal_unblocked_hand_that_cannot_make_target(example):
    with pytest.raises(ValueError, match="witness"):
        verify_targets.check_target(10, {"threshold": len(example) + 1, "example": example}, 8, {})


def test_checker_compares_complete_count_to_published_result(tmp_path, monkeypatch, capsys):
    # With one card, 1 makes the target and the other nine are blocked.
    row = {"threshold": 1, "example": None, "made_from_threshold": 2}
    published = {"targets": {"max_cards": 1, "by_target": {"1": row}}}
    monkeypatch.setattr(verify_targets, "__file__", str(tmp_path / "verify_targets.py"))
    path = tmp_path / "one.gz"
    with gzip.open(path, "wt") as f:
        for rec in (
            {"target": 1, "threshold": 1, "max": 1},
            {"target": 1, "cards": [1], "expr": "1"},
        ):
            f.write(json.dumps(rec) + "\n")
    results = tmp_path / "results.json"
    results.write_text(json.dumps(published))
    with pytest.raises(ValueError, match="expression count is 1, results.json claims 2"):
        verify_targets.main(path, targets=[1], max_cards=1)
    row["made_from_threshold"] = 1
    results.write_text(json.dumps(published))
    verify_targets.main(path, targets=[1], max_cards=1)
    assert "expressions 1, hands blocked by a remainder 9" in capsys.readouterr().out
