"""Tests for codenames/human_pilot.py (C11, part B). None needs word vectors: the giver's
picks are read back from the committed key.json, or made up."""

import json
import random
from collections import Counter

import pytest

from codenames import c4
from codenames import human_pilot as hp
from codenames.game import deal
from codenames.legality import flags, strict_legal
from codenames.words import board_words, legal

NAMES = ("boards.json", "key.json", "play.html")


def committed() -> list[str]:
    return [(hp.OUT / n).read_text() for n in NAMES]


def toy_picks(board):
    """Made-up picks: F and P take the first and second board-free clue of a short list;
    on boards whose first word starts with a vowel they share a word (and are skipped)."""
    clues = [c for c in ("ocean", "metal", "animal", "music") if legal(c, board.words)]
    same = board.words[0][0] in "aeiou"
    return {"F": (clues[0], 2), "P": (clues[0 if same else 1], 1), "oracle": (clues[1], 1)}


def test_committed_files_are_what_the_script_writes_from_the_key():
    key = json.loads((hp.OUT / "key.json").read_text())
    assert list(hp.materials(*hp.rebuild(key), order_seed=key["order_seed"])) == committed()


def test_key_hash_and_the_page_hold_no_arm_or_team_map():
    boards_text, key_text, page = committed()
    boards = json.loads(boards_text)
    assert boards["key_sha256"] == hp.sha256(key_text)
    data = json.loads(page.split("const DATA = ")[1].split(";\nconst STORE")[0])
    assert data == hp.player_data(boards)
    assert set(data) == {"plan", "block", "boards", "items", "versions"}
    assert all(len(v) == 3 for v in data["items"].values())
    assert "kinds" not in page and "seed_index" not in page
    assert all(set(x) == {"item", "board", "clue", "count"} for x in boards["items"])
    assert [x["item"] for x in boards["items"]] == sorted(x["item"] for x in boards["items"])


def test_committed_boards_are_c4_deals_with_legal_distinct_clues():
    boards = json.loads((hp.OUT / "boards.json").read_text())
    key = json.loads((hp.OUT / "key.json").read_text())
    assert len(boards["boards"]) == hp.N_BOARDS and boards["board_seed"] == hp.BOARD_SEED
    deals = c4.boards(hp.BOARD_SEED, 1 + boards["boards"][-1]["seed_index"])
    used = [b["seed_index"] for b in boards["boards"]]
    assert sorted(used + key["skipped"]) == list(range(len(deals)))
    for b, k in zip(boards["boards"], key["boards"], strict=True):
        d = deals[b["seed_index"]]
        assert tuple(b["words"]) == d.words and tuple(b["kinds"]) == d.kinds
        assert Counter(b["kinds"]) == {"T": 9, "O": 8, "N": 7, "A": 1}
        f, p = (k[arm].split("/") for arm in hp.ARMS)
        assert f[0] != p[0]
        for clue, n in (f, p):
            assert legal(clue, d.words) and 1 <= int(n) <= 4
            # The amendment of 2026-11-19: no form of a board word, no shared part.
            assert strict_legal(clue, d.words), (clue, flags(clue, d.words))
    assert key["giver"]["clue_filter"].startswith("codenames.legality.strict_legal")


def test_the_first_materials_had_the_flagged_clues_the_amendment_lists():
    before = json.loads((hp.HERE / "results" / "diagnostics" / "pilot-before-key.json").read_text())
    deals = c4.boards(hp.BOARD_SEED, 1 + before["boards"][-1]["seed_index"])
    bad = sorted(
        r[arm].split("/")[0]
        for r in before["boards"]
        for arm in hp.ARMS
        if not strict_legal(r[arm].split("/")[0], deals[r["seed_index"]].words)
    )
    assert len(bad) == 18
    for clue in ("residency", "librarian", "winning", "owned", "governance", "teaching"):
        assert clue in bad


def check_design(boards: dict, key: dict) -> None:
    n = len(boards["boards"])
    items = {x["item"]: x for x in boards["items"]}
    assert len(items) == 2 * n and set(key["items"]) == set(items)
    assert Counter(key["items"].values()) == {"F": n, "P": n}
    a, b = (boards["versions"][v] for v in hp.VERSIONS)
    for order in (a, b):
        assert sorted(order) == sorted(items)
        for block in (order[:n], order[n:]):
            assert sorted(items[i]["board"] for i in block) == list(range(n))
        assert items[order[n - 1]]["board"] != items[order[n]]["board"]
    block1 = Counter(key["items"][i] for i in a[:n])
    assert block1 == {"F": n // 2, "P": n - n // 2}
    for i, j in zip(a, b, strict=True):  # B shows the same board with the other clue
        assert items[i]["board"] == items[j]["board"]
        assert key["items"][i] != key["items"][j]
    for i in a:  # each board's two items carry its F and P actions
        x = items[i]
        want = key["boards"][x["board"]][key["items"][i]]
        assert want == f"{x['clue']}/{x['count']}"


def test_committed_design():
    check_design(*(json.loads(t) for t in committed()[:2]))


@pytest.mark.parametrize("order_seed", [0, 1, 2, 3, hp.ORDER_SEED])
def test_design_with_toy_picks_and_other_seeds(order_seed):
    chosen, skipped = hp.choose_boards(5, 12, picks_of=toy_picks)
    assert len(chosen) == 12
    assert all(c4.boards(5, 40)[i].words[0][0] in "aeiou" for i in skipped)
    texts = hp.materials(chosen, skipped, order_seed)
    assert texts == hp.materials(chosen, skipped, order_seed)  # deterministic
    check_design(json.loads(texts[0]), json.loads(texts[1]))


def test_orders_differ_between_seeds():
    assert hp.orders(40, 1)["block1"] != hp.orders(40, 2)["block1"]


def test_turn_value_by_hand():
    words = ["a", "b", "c", "d", "e", "f"]
    kinds = ["T", "T", "O", "N", "A", "T"]
    v = hp.turn_value
    assert v(words, kinds, [], 2) == (0, 0.0)  # stopped at once
    assert v(words, kinds, ["a", "b"], 2) == (2, 2.0)
    assert v(words, kinds, ["a", "c", "b"], 3) == (1, 0.0)  # opponent ends the turn
    assert v(words, kinds, ["a", "d"], 2) == (1, 1.0)  # neutral costs nothing
    assert v(words, kinds, ["e"], 1) == (0, -3.0)  # assassin: every team word left
    assert v(words, kinds, ["a", "b", "f"], 2) == (2, 2.0)  # no pick past the count


def test_turn_value_matches_play_turn_on_random_turns():
    rng = random.Random(0)
    for _ in range(200):
        board = deal(board_words(), rng)
        count = rng.randint(1, 4)
        picks = rng.sample(board.words, rng.randint(0, count))
        want = c4.play_turn(board, picks, count)
        assert hp.turn_value(list(board.words), list(board.kinds), picks, count) == want


def test_score_response_on_a_made_up_session():
    boards, key = (json.loads(t) for t in committed()[:2])
    items = {x["item"]: x for x in boards["items"]}
    order = boards["versions"]["B"]
    responses = []
    for i in order:
        b = boards["boards"][items[i]["board"]]
        team = [w for w, k in zip(b["words"], b["kinds"], strict=True) if k == "T"]
        responses.append({"item": i, "picks": team[: items[i]["count"]], "t_done": 1500})
    rows = hp.score_response(boards, key, {"version": "B", "responses": responses})
    assert [r["position"] for r in rows] == list(range(2 * hp.N_BOARDS))
    assert all(r["value"] == r["found"] == r["count"] for r in rows)
    assert Counter((r["block"], r["arm"]) for r in rows) == {
        (1, "F"): 20,
        (1, "P"): 20,
        (2, "F"): 20,
        (2, "P"): 20,
    }
    assert {r["seconds"] for r in rows} == {1.5}


def test_prepared_page_consent_resume_partial_export_and_key(tmp_path, monkeypatch):
    import sys

    dest, private = tmp_path / "delivery", tmp_path / "keys/key.json"
    monkeypatch.setattr(
        sys, "argv", ["pilot", "prepare", "--out", str(dest), "--key", str(private)]
    )
    hp.main()
    assert {p.name for p in dest.iterdir()} == {"boards.json", "play.html"}
    boards = json.loads((dest / "boards.json").read_text())
    assert boards["key_sha256"] == hp.sha256(private.read_text())
    assert boards["order_seed"] == 11012
    assert private.stat().st_mode & 0o077 == 0
    page = (dest / "play.html").read_text()
    for text in (
        "const ACTIVE = true",
        "18 or older",
        "Keep this code to withdraw",
        "Export this half",
        "Continue saved session",
        "localStorage.removeItem(STORE)",
    ):
        assert text in page
    assert "new Date()" not in page and "const ACTIVE = false" in (hp.OUT / "play.html").read_text()


def test_summary_is_paired_and_excludes_incomplete_blocks_and_private_fields():
    boards, key = (json.loads(t) for t in committed()[:2])
    items = {x["item"]: x for x in boards["items"]}

    def sample(code, version, n):
        responses = []
        for i in boards["versions"][version][:n]:
            b = boards["boards"][items[i]["board"]]
            good = next(w for w, k in zip(b["words"], b["kinds"], strict=True) if k == "T")
            responses.append(
                {"item": i, "picks": [good] if key["items"][i] == "P" else [], "t_done": 1000}
            )
        return {
            "version": version,
            "session": code,
            "plan": "C11",
            "consent": True,
            "comment": "private comment",
            "edition": boards["key_sha256"],
            "responses": responses,
        }

    a = sample("abcd1234", "A", 80)
    b = sample("dbca1234", "B", 40)
    r = hp.summarize(
        boards, key, [b, sample("abcd1234", "A", 40), a, sample("ffff1234", "A", 39)], draws=100
    )
    assert (r["complete"], r["completed_first_block"], r["sessions"]) == (1, 2, 3)
    assert r["primary"]["mean"] == r["first_block"]["mean"] == 1.0
    assert r["second_block"] is None and r["block_by_arm"] is None
    assert not any(v in json.dumps(r) for v in ("abcd1234", "private comment", "dbca1234"))
    with pytest.raises(ValueError, match="conflicting"):
        hp.summarize(boards, key, [a, b | {"session": "abcd1234"}])
    with pytest.raises(ValueError, match="consent"):
        hp.summarize(boards, key, [a | {"consent": False}])
    with pytest.raises(ValueError, match="order"):
        hp.summarize(boards, key, [a | {"responses": a["responses"][1:]}])


def test_benchmark_model_scores_match_historical_pilot_board_cells():
    from codenames import c11

    reference = json.loads((hp.OUT / "benchmark.json").read_text())
    historical = json.loads(c11.OUT.read_text())["items"]
    for name in c11.GUESSERS:
        d = reference[name]["board_differences"]
        assert len(d) == 40 and sum(d) / 40 == pytest.approx(historical[name]["P_minus_F"]["mean"])
    assert len(reference["potion"]["board_differences"]) == 40
