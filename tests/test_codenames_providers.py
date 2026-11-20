"""Tests for codenames/providers.py: the log audit and the reading of served fields."""

import json

import pytest

from codenames.hosted import served_by
from codenames.providers import audit, id_shape, linear_fit, main, normalize, variants


def test_id_shapes():
    assert id_shape("gen-1795060762-AbC123") == "gen-"
    assert id_shape("chatcmpl-9f2a") == "chatcmpl-"
    assert id_shape("39892c3aa1bbcd74c58035332ca31258") == "hex32"
    assert id_shape("123e4567-e89b-12d3-a456-426614174000") == "uuid"
    assert id_shape("call-3") == "other" and id_shape(None) == "none"


def test_linear_fit_recovers_a_line():
    xs = [100.0, 200.0, 300.0, 400.0]
    fit = linear_fit(xs, [20 + 0.25 * x for x in xs])
    assert fit is not None
    assert fit["intercept"] == pytest.approx(20) and fit["slope"] == pytest.approx(0.25)
    assert fit["residual_sd"] == pytest.approx(0)
    assert linear_fit([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None


def test_served_by_reads_openrouter_fields_and_tolerates_their_absence():
    body = {
        "id": "gen-1-a",
        "model": "openai/gpt-oss-120b",
        "provider": "DeepInfra",
        "openrouter_metadata": {
            "summary": "available=1, selected=DeepInfra",
            "attempt": 1,
            "endpoints": {"available": [{"provider": "DeepInfra", "selected": True}]},
        },
    }
    s = served_by(body)
    assert s["provider"] == "DeepInfra" and s["attempt"] == 1 and s["quantization"] is None
    assert served_by({"model": "m"})["provider"] is None
    del body["provider"]
    assert served_by(body)["provider"] == "DeepInfra"  # from the metadata


def codenames_record(prompt, reply, finish="stop", seconds=1.0, tokens=None):
    return {
        "model": "openai/gpt-oss-120b",
        "provider": "deepinfra",
        "prompt": prompt,
        "raw": reply,
        "reply": reply,
        "finish": finish,
        "usage": {"prompt_tokens": tokens or len(prompt) // 4 + 30, "completion_tokens": 5},
        "served": "openai/gpt-oss-120b",
        "seconds": seconds,
        "time": "2026-11-19T20:00:21+00:00",
        "errors": [],
    }


def hanabi_record(prompt):
    return {
        "time": 1795060762.5,
        "seconds": 2.0,
        "model": "openai/gpt-oss-120b",
        "provider": "novita",
        "served_model": "openai/gpt-oss-120b",
        "id": "39892c3aa1bbcd74c58035332ca31258",
        "prompt": prompt,
        "reply": "PLAY 1",
        "finish_reason": "length",
        "prompt_tokens": 400,
        "completion_tokens": 4096,
        "errors": [],
    }


def test_audit_aggregates_both_log_formats_without_strings(tmp_path):
    run = tmp_path / "c10" / "test"
    run.mkdir(parents=True)
    rows = [codenames_record("x" * n, "a\nb") for n in (100, 200, 300)]
    rows.append(codenames_record("x" * 400, "", seconds=35.0))
    (run / "calls-gpt-oss-120b.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    (run / "manifest.json").write_text(json.dumps({"commit": "abc", "job_id": "j1"}))
    game = tmp_path / "h2" / "arm" / "calls"
    game.mkdir(parents=True)
    for seed in (0, 1):
        (game / f"seed-{seed}.jsonl").write_text(json.dumps(hanabi_record("p" * 50)) + "\n")
    result = audit(tmp_path)
    log = result["c10"]["test"]["logs"]["gpt-oss-120b"]
    assert result["c10"]["test"]["commit"] == "abc"
    assert log["calls"] == 4 and log["empty"] == 1 and log["empty_with_stop"] == 1
    assert log["seconds_median_empty"] == 35.0
    assert log["prompt_tokens_on_chars"]["slope"] == pytest.approx(0.25, abs=0.01)
    h2 = result["h2"]["arm"]["logs"]["calls"]
    assert h2["calls"] == 2 and h2["id_shapes"] == {"hex32": 2} and h2["finish"] == {"length": 2}
    assert h2["first"] == "2026-11-19T03:59:22Z"
    text = json.dumps(result)
    assert "xxxx" not in text and "39892c3a" not in text  # no prompts and no ids
    out = tmp_path / "providers.json"
    assert main(["audit", "--runs", str(tmp_path), "--out", str(out)]) == 0
    assert json.loads(out.read_text()) == json.loads(json.dumps(result, sort_keys=True))


def test_normalize_counts_a_failed_call_as_unanswered():
    rec = {"model": "m", "provider": "p", "prompt": "q", "errors": ["boom"] * 5, "reply": ""}
    n = normalize(rec)
    assert not n["answered"] and n["retried"] and n["empty"]


def test_pincheck_variants_name_the_provider_each_way():
    v = {name: body for name, _, body, _ in variants("org/m", "deepinfra", {}, "m-id")}
    assert v["path"]["model"] == "m-id" and "provider" not in v["path"]
    assert v["suffix"]["model"] == "org/m:deepinfra"
    assert v["object"]["provider"] == {"order": ["deepinfra"], "allow_fallbacks": False}
    assert v["object+impossible quantization"]["provider"]["quantizations"] == ["fp32"]
