"""Tests for codenames/c11.py (C11, part A). None needs word vectors or the router: the
picks are checked on random values against loops, and the guessers are a stub."""

import json
import random

import numpy as np
import pytest

from codenames import c4, c11
from codenames.game import deal
from codenames.hosted import Hosted, Meter
from codenames.llm import guesser_prompt
from codenames.words import board_words


def loop_picks(strangers, proposer, kidx, potion):
    """The arms by plain loops, first index on ties."""

    def first_best(xs):
        best = max(xs)
        return xs.index(best)

    f = first_best(list(proposer))
    tasks = []
    for t in range(len(strangers)):
        means = [np.mean([strangers[s][j] for s in range(len(strangers)) if s != t]) for j in kidx]
        tasks.append(kidx[first_best(means)])
    oracle = kidx[first_best([np.mean([row[j] for row in strangers]) for j in kidx])]
    return {"F": f, "P": tasks[potion], "P_tasks": tasks, "oracle": oracle}


@pytest.mark.parametrize("seed", range(20))
def test_pick_indices_match_loops(seed):
    rng = np.random.default_rng(seed)
    # Rounded values, so that ties occur and the first index has to win.
    strangers = np.round(rng.normal(size=(7, 120)), 1)
    proposer = np.round(rng.normal(size=120), 1)
    kidx = sorted(rng.choice(120, 40, replace=False).tolist())
    potion = int(rng.integers(7))
    assert c11.pick_indices(strangers, proposer, kidx, potion) == loop_picks(
        strangers.tolist(), proposer.tolist(), kidx, potion
    )


def test_exact_values_by_hand():
    strangers = np.arange(14, dtype=float).reshape(7, 2)  # stranger t values (2t, 2t + 1)
    human, eat, prop = np.array([0.5, 1.5]), np.array([-1.0, 2.0]), np.array([3.0, 4.0])
    one = c11.exact_values(strangers, human, eat, prop, 2, 1)
    assert one == {
        "strangers": 7.0,
        "strangers_c10": 7.0,
        "potion": 5.0,
        "human": 1.5,
        "eat": 2.0,
        "proposer": 4.0,
    }
    # P with per-task picks: task t's pick is valued by stranger t.
    tasks = [0, 1, 0, 1, 0, 1, 0]
    p = c11.exact_values(strangers, human, eat, prop, 2, 0, tasks)
    assert p["strangers_c10"] == np.mean([2 * t + tasks[t] for t in range(7)])
    assert p["strangers"] == 6.0


def test_primary_diffs_match_a_loop():
    rng = np.random.default_rng(1)
    n = 50
    v = {(a, m): rng.integers(-3, 4, n).astype(float) for a in c11.ARMS for m in c11.GUESSERS}
    v[("P", "gpt-oss-120b")][[3, 7]] = np.nan
    v[("F", "llama-3.3-70b")][11] = np.nan
    pot_p, pot_f = rng.normal(size=n), rng.normal(size=n)
    h1, h2, keep = c11.primary_diffs(v, pot_p, pot_f)
    for i in range(n):
        ds = [v[("P", m)][i] - v[("F", m)][i] for m in c11.GUESSERS]
        assert keep[i] == (i not in (3, 7, 11))
        if keep[i]:
            assert h1[i] == pytest.approx(sum(ds) / 2)
            assert h2[i] == pytest.approx(sum(ds) / 2 - (pot_p[i] - pot_f[i]))


def test_mde_formula():
    d = np.array([1.0, -1.0] * 50)
    m = c11.mde(d)
    sd = np.std(d, ddof=1)
    assert m["at_0.05"] == pytest.approx(2.80 * sd / 10)
    assert m["at_holm_0.025"] == pytest.approx(3.08 * sd / 10)


BOARD = deal(board_words(), random.Random(3))


def rec(reply, finish="stop", **kw):
    return {"reply": reply, "usage": {}, "finish": finish, "errors": [], **kw}


def test_status_and_cell():
    words = BOARD.unrevealed()
    team = [w for w in words if BOARD.kind(w) == "T"]
    opp = next(w for w in words if BOARD.kind(w) == "O")
    good = "\n".join([*team[:2], opp])
    assert c11.status_of(None, words) == "stopped"
    assert c11.status_of({"errors": ["RuntimeError: x"]}, words) == "failed"
    assert c11.status_of({"errors": ["served by Crusoe"] * 5}, words) == "unpinned"
    assert c11.status_of(rec(""), words) == "empty"
    assert c11.status_of(rec("", pin_held=False), words) == "unpinned"
    assert c11.status_of(rec("no idea"), words) == "noword"
    assert c11.status_of(rec(good), words) == "ok"
    assert c11.status_of(rec(good, "length"), words) == "cut"
    assert c11.cell(BOARD, 3, rec(good)) == "2/1"
    assert c11.cell(BOARD, 2, rec(good, "length")) == "2/2c"
    assert c11.cell(BOARD, 2, rec("")) == "-empty"
    assert c11.read_cell("2/1") == (2, 1.0, False)
    assert c11.read_cell("2/2c") == (2, 2.0, True)
    assert c11.read_cell("-empty") is None


def test_cell_matches_play_turn_on_random_rankings():
    rng = random.Random(0)
    for _ in range(200):
        b = deal(board_words(), rng)
        ranking = list(b.unrevealed())
        rng.shuffle(ranking)
        count = rng.randint(1, 4)
        found, value = c4.play_turn(b, ranking, count)
        assert c11.read_cell(c11.cell(b, count, rec("\n".join(ranking)))) == (found, value, False)


def test_assassin_flag():
    b = BOARD
    assassin = next(w for w in b.unrevealed() if b.kind(w) == "A")
    opp = next(w for w in b.unrevealed() if b.kind(w) == "O")
    team = [w for w in b.unrevealed() if b.kind(w) == "T"]
    cells = [
        c11.cell(b, 4, rec("\n".join([*team[:3], assassin]))),
        c11.cell(b, 1, rec(opp)),
        c11.cell(b, 2, rec("\n".join(team[:2]))),
        "-empty",
    ]
    flags = c11.assassin(cells)
    assert flags[:3].tolist() == [1.0, 0.0, 0.0] and np.isnan(flags[3])


class Stub:
    """Answers like the router; the reply is a function of the prompt and the call count."""

    def __init__(self, reply):
        self.reply, self.sent = reply, []

    def chat_completion(self, messages, model, extra_body=None, **params):
        prompt = messages[0]["content"]
        self.sent.append(prompt)
        return {
            "model": model,
            "choices": [
                {
                    "message": {"content": self.reply(prompt, len(self.sent))},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 200, "completion_tokens": 30},
        }


def words_of(prompt: str) -> list[str]:
    return prompt.split("Board words:\n")[1].split("\n")


PICKS = (
    "F 5 sun/2 moon/1 star/3\nP 5 sun/2 tree/1 star/1\noracle 5 rain/2 tree/1 star/1\n"
    "F-c10 5 sun/2 moon/1 star/3\nP-c10 5 sun/1 tree/1 star/1\noracle-c10 5 rain/2 tree/1 star/1\n"
)


def stub_run(tmp_path, monkeypatch, reply, **kw):
    stub = Stub(reply)

    def hosted(model, meter, log, max_tokens):
        return Hosted(model, meter, log, max_tokens, client=stub, sleep=lambda s: None)

    monkeypatch.setattr(c11, "Hosted", hosted)
    summary = c11.run(
        ["llama-3.3-70b"],
        PICKS,
        5,
        range(0, 3),
        kw.get("repeat"),
        2,
        False,
        Meter(1.0),
        tmp_path,
        1,
    )
    return stub, summary, c11.read_guesses((tmp_path / "guesses.txt").read_text())


def test_run_sends_each_distinct_prompt_once_and_resends_empty_replies(tmp_path, monkeypatch):
    # The first call returns nothing; every other call lists the board in its order.
    stub, summary, g = stub_run(
        tmp_path, monkeypatch, lambda p, n: "" if n == 1 else "\n".join(words_of(p))
    )
    distinct = {(i, c) for line in PICKS.splitlines() for i, c in enumerate(line.split()[2:])}
    assert len(stub.sent) == len(distinct) + 1  # one resent
    bs = c4.boards(5, 3)
    assert stub.sent[0] == guesser_prompt("sun", 2, bs[0].unrevealed())
    assert sorted(g) == sorted(
        (a, "llama-3.3-70b", 5, 0) for a in ("F", "P", "oracle", "F-c10", "P-c10", "oracle-c10")
    )
    s = summary["models"]["llama-3.3-70b"]["guess"]
    assert s["first_status"]["empty"] == 1 and s["final_status"] == {"ok": len(distinct)}
    assert s["resent"] == 1
    for (a, *_), cells in g.items():
        given = c4.read_clue_lines(PICKS)[(a, 5)]
        for b, (_, count, _), x in zip(bs, given, cells, strict=True):
            assert count is not None
            found, value = c4.play_turn(b, list(b.unrevealed()), count)
            assert x == f"{found}/{value:g}"


def test_an_empty_reply_twice_is_not_played(tmp_path, monkeypatch):
    stub, summary, g = stub_run(
        tmp_path, monkeypatch, lambda p, n: "" if "'moon'" in p else "\n".join(words_of(p))
    )
    assert g[("F", "llama-3.3-70b", 5, 0)][1] == "-empty"
    assert summary["models"]["llama-3.3-70b"]["guess"]["final_status"]["empty"] == 1


def test_repeat_counts_changed_values(tmp_path, monkeypatch):
    # Each prompt's first reply lists the board in order; later sends of the sun prompt
    # reverse it, which changes its played-out value; the others repeat.
    seen: dict[str, int] = {}

    def reply(prompt, n):
        seen[prompt] = seen.get(prompt, 0) + 1
        w = words_of(prompt)
        return "\n".join(w[::-1] if "'sun'" in prompt and seen[prompt] > 1 else w)

    stub, summary, g = stub_run(tmp_path, monkeypatch, reply, repeat=range(0, 3))
    assert sorted({s for *_, s in g}) == [0, 1, 2]
    lines = [line.split()[2:] for line in PICKS.splitlines()[:3]]
    rows = [
        {"strict": {"F": f, "P": p, "oracle": o}, "c10": {}} for f, p, o in zip(*lines, strict=True)
    ]
    for r in rows:
        r["strict"]["values"] = {a: {"potion": 0.0} for a in c11.ARMS}
    b = c4.boards(5, 1)[0]
    words = list(b.unrevealed())
    changed = int(c4.play_turn(b, words, 2)[1] != c4.play_turn(b, words[::-1], 2)[1])
    out = c11.repeat_summary(rows, g, 5, models=("llama-3.3-70b",))
    distinct = len({(i, r["strict"][a]) for i, r in enumerate(rows) for a in c11.ARMS})
    assert out["models"]["llama-3.3-70b"]["prompts"] == distinct
    assert out["models"]["llama-3.3-70b"]["value_changed"] == changed
    assert set(out["h_a1_by_send"]) == {0, 1, 2}


def test_item_targets_are_the_pilot_items():
    by = c11.item_targets()["items"]
    data = json.loads(c11.ITEMS.read_text())
    assert len(by) == 80
    for (key, board), x in zip(by, data["items"], strict=True):
        assert key[2:] == (x["clue"], x["count"])
        assert list(board.words) == data["boards"][x["board"]]["words"]


def test_committed_picks_are_what_lists_writes_from_the_values():
    data = json.loads(c11.LISTS.read_text())
    assert c11.PICKS.read_text() == c11.picks_text(data["boards"])
    assert data["checks"]["passed"]
    test = [r for r in data["boards"] if r["seed"] == c11.TEST]
    assert len(test) == c11.N_TEST
    # Every strict arm passes the strict check; the boards are c4's deals.
    assert all(r["strict"]["level"][a] == "legal" for r in data["boards"] for a in c11.ARMS)
    assert {r["seed"] for r in data["boards"]} == {c11.PILOT, c11.TEST}


def test_committed_score_is_what_score_writes():
    if not c11.GUESSES.exists():
        pytest.skip("no test replies yet")
    want = json.dumps(c11.score(), indent=1, sort_keys=True) + "\n"
    assert c11.OUT.read_text() == want


@pytest.mark.parametrize("finish", ["length", "stop"])
def test_score_stops_at_last_word_the_model_named(finish):
    from codenames.game import Board

    board = Board(("apple", "berry", "stone"), ("T", "T", "A"))
    assert c11.read_cell(c11.cell(board, 2, rec("apple", finish))) == (1, 1.0, finish == "length")


def test_replay_checks_historical_scores_and_reports_only_aggregates(tmp_path):
    b = c4.boards(c11.TEST, 1)[0]
    # Select a team word and ensure the first padding word is the board's first word.
    picked = next(w for w, k in zip(b.words[1:], b.kinds[1:], strict=True) if k == "T")
    old = {}
    logs = {}
    rows = [
        {
            "strict": {
                "P": "clue/2",
                "F": "clue/1",
                "values": {"P": {"potion": 1.0}, "F": {"potion": 1.0}},
            }
        }
    ]
    for m in c11.GUESSERS:
        log = tmp_path / m
        logs[m] = log
        records = []
        for a, n in (("P", 2), ("F", 1)):
            r = rec(picked, "length") | {
                "tag": {"part": "guess", "seed": c11.TEST, "board": 0, "clue": "clue", "count": n}
            }
            found, value = c4.play_turn(b, [picked, b.words[0]], n)
            old[a, m, c11.TEST, 0] = [f"{found}/{value:g}c"]
            records.append(json.dumps(r))
        log.write_text("\n".join(records))
    summary = c11.replay_primary(logs, rows, old)
    assert summary["h_a1"]["mean"] == 0 and summary["kept"] == 1
    assert all(summary["replies_by_arm"][f"{m} P"]["reached_appended_word"] == 1 for m in logs)
    old["P", c11.GUESSERS[0], c11.TEST, 0] = ["9/9c"]
    with pytest.raises(ValueError, match="differs"):
        c11.replay_primary(logs, rows, old)


def test_missing_secondary_oracle_preserves_primary_result():
    if not c11.GUESSES.exists():
        pytest.skip("no test replies yet")
    data = json.loads(c11.LISTS.read_text())
    rows = [r for r in data["boards"] if r["seed"] == c11.TEST][:5]
    g = {k: cells[:5] for k, cells in c11.read_guesses(c11.GUESSES.read_text()).items()}
    before = c11.score_filter(rows, g, "strict", c11.TEST)
    guesser = c11.GUESSERS[0]
    g["oracle", guesser, c11.TEST, 0] = ["-empty"] * len(rows)
    after = c11.score_filter(rows, g, "strict", c11.TEST)
    assert after["h_a1"] == before["h_a1"]
    assert after["h_a2"] == before["h_a2"]
    assert after["secondary"][guesser]["oracle_minus_F"] == {
        "mean": None,
        "ci": None,
        "p": None,
        "n": 0,
    }
    assert after["secondary"]["mean_of_guessers"]["oracle_minus_F"]["n"] == 0
    assert after["secondary"]["order"][guesser] is None

    # Strict primary comparisons require observed pairs. The exploratory
    # C10 filter can be unavailable without discarding strict estimates.
    g["P", guesser, c11.TEST, 0] = ["-empty"] * len(rows)
    with pytest.raises(ValueError, match="without an observed pair"):
        c11.score_filter(rows, g, "strict", c11.TEST)
    g["P-c10", guesser, c11.TEST, 0] = ["-empty"] * len(rows)
    exploratory = c11.score_filter(rows, g, "c10", c11.TEST)
    assert exploratory["h_a1"]["n"] == 0
    assert exploratory["mde"]["h_a1"]["at_0.05"] is None
