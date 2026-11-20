"""Tests for codenames/providers.py: the log audit and the reading of served fields."""

import json

import pytest

from codenames.hosted import served_by
from codenames.providers import (
    audit,
    id_shape,
    linear_fit,
    main,
    normalize,
    router_providers,
    variants,
)


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


def test_router_listing_shapes():
    entry = {"provider": "deepinfra", "pricing": {"input": 0.1, "output": 0.5}}
    assert router_providers({"data": {"id": "m", "providers": [entry]}}) == [entry]
    assert router_providers({"data": [{"id": "m", "providers": [entry]}]}) == [entry]
    assert router_providers({"id": "m", "providers": [entry]}) == [entry]
    assert router_providers({"error_text": "HTTPError: 404"}) is None


def test_digest_is_compact_and_reads_both_files(tmp_path):
    from codenames.providers import digest

    run = tmp_path / "runs" / "c10" / "val"
    run.mkdir(parents=True)
    (run / "calls-m.jsonl").write_text(json.dumps(codenames_record("x" * 100, "a")) + "\n")
    (tmp_path / "audit.json").write_text(json.dumps(audit(tmp_path / "runs")))
    check = {
        "time": "t",
        "spent": 0.001,
        "calls": [
            {
                "repo": "org/m",
                "provider": "novita",
                "variant": "suffix",
                "status": 200,
                "served_by": {"provider": None, "id": "39892c3aa1bbcd74c58035332ca31258"},
                "headers": {"x-inference-provider": "novita"},
            },
            {"repo": "org/m", "provider": "novita", "error": "Traceback"},
        ],
        "listings": {
            "org/m": {"openrouter_endpoints": [{"provider_name": "Novita", "quantization": "fp8"}]}
        },
    }
    (tmp_path / "check.json").write_text(json.dumps(check))
    d = digest(tmp_path / "audit.json", tmp_path / "check.json")
    log = d["runs"]["c10"]["val"]["logs"]["m"]
    assert log["calls"] == 1 and log["requested"] == {"deepinfra": 1}
    rows = d["pincheck"]["rows"]
    assert rows["org/m @ novita : suffix"] == {
        "status": [200],
        "served": ["None via novita"],
        "id_shapes": ["hex32"],
    }
    assert rows["org/m @ novita : setup"]["status"] == ["exception"]
    assert d["pincheck"]["quantization"] == {"org/m": ["Novita: fp8"]}
    from codenames.providers import COLUMNS, table_text

    t = json.loads(table_text(d))
    assert t["columns"] == list(COLUMNS) and len(t["logs"]) == 1
    row = dict(zip(COLUMNS, t["logs"][0], strict=True))
    assert row["experiment"] == "c10" and row["requested"] == "deepinfra 1" and row["calls"] == 1
    assert t["pincheck"][1][:3] == ["org/m @ novita : suffix", "200 1", "None via novita 1"]


class Served:
    """A router stub whose replies name a provider, the pinned one after `wrong` replies."""

    def __init__(self, provider="DeepInfra", wrong=0, header=None, replies=("a", "b")):
        self.provider, self.wrong, self.header = provider, wrong, header
        self.replies, self.calls = list(replies), 0

    def chat_completion(self, messages, model, extra_body=None, **params):
        self.calls += 1
        who = "Parasail" if self.calls <= self.wrong else self.provider
        return {
            "model": model.split(":")[0],
            "provider": who,
            "_router_header": self.header,
            "choices": [
                {"message": {"content": self.replies[self.calls % len(self.replies)]},
                 "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 1e-5},
        }  # fmt: skip


def test_pinning_requests_and_quantization_filter():
    from codenames.hosted import Model, pinned

    mid, extra = pinned(Model("org/m", "deepinfra", {"x": 1}, quantizations=("fp8",)))
    assert mid == "org/m:deepinfra"
    assert extra == {
        "x": 1,
        "provider": {"order": ["deepinfra"], "allow_fallbacks": False, "quantizations": ["fp8"]},
    }
    with pytest.raises(ValueError):
        pinned(Model("org/m", "auto"))


def test_a_reply_from_another_provider_is_sent_again_and_paid_for(tmp_path):
    from codenames.hosted import Hosted, Meter, Model

    meter = Meter(1.0)
    stub = Served(wrong=1)
    h = Hosted(Model("org/m", "deepinfra"), meter, tmp_path / "c.jsonl", client=stub,
               sleep=lambda s: None)  # fmt: skip
    rec = h.call("p")
    assert rec["pin_held"] is True and rec["attempts"] == 2 and stub.calls == 2
    assert rec["errors"] == ["served by Parasail"] and rec["served_by"]["provider"] == "DeepInfra"
    assert meter.spent == pytest.approx(2e-5)  # both replies cost money
    stuck = Hosted(Model("org/m", "deepinfra"), Meter(1.0), client=Served(wrong=99),
                   sleep=lambda s: None)  # fmt: skip
    assert stuck("p") == ""


def test_served_provider_from_the_header_and_name_matching():
    from codenames.hosted import Model, pin_held, provider_key, served_provider

    assert served_provider({"provider": None, "router_header": "novita"}) == "novita"
    assert served_provider({"provider": None, "router_header": "openrouter"}) is None
    assert provider_key("Featherless") == provider_key("featherless-ai")
    assert provider_key("DeepInfra") == "deepinfra"
    assert pin_held(Model("m", "novita"), {"router_header": "novita"}) is True
    assert pin_held(Model("m", "novita"), {}) is None


def test_repeat_check_counts_changed_replies_and_providers():
    from codenames.hosted import Hosted, Meter, Model, repeat_check

    same = Hosted(Model("m", "deepinfra"), Meter(1.0), client=Served(replies=("a",)))
    r = repeat_check(same, ["p", "q"], times=3)
    assert r["changed"] == 0 and r["served"] == {"DeepInfra": 6} and r["pin"]["held"] == 6
    flip = Hosted(Model("m", "deepinfra"), Meter(1.0), client=Served(replies=("a", "b")))
    assert repeat_check(flip, ["p"], times=2)["changed_share"] == 1.0


def test_openrouter_ids_and_listed_quantizations(tmp_path, monkeypatch):
    from codenames import hosted
    from codenames.hosted import Model, openrouter_id, parse_quantizations

    assert openrouter_id("deepseek-ai/DeepSeek-V3.2") == "deepseek/deepseek-v3.2"
    assert openrouter_id("meta-llama/Llama-3.3-70B-Instruct") == "meta-llama/llama-3.3-70b-instruct"
    listing = {"data": {"endpoints": [{"provider_name": "DeepInfra", "quantization": "bf16"}]}}
    assert parse_quantizations(listing) == {"DeepInfra": "bf16"}
    monkeypatch.setattr(hosted, "listed_quantizations", lambda models: {"m": "listed"})
    registered = []
    monkeypatch.setattr("atexit.register", registered.append)
    hosted.record_serving([Model("org/m", "deepinfra")], tmp_path)
    assert json.loads((tmp_path / "quantizations.json").read_text()) == {"m": "listed"}
    (tmp_path / "calls-m.jsonl").write_text(json.dumps(codenames_record("x" * 50, "a")) + "\n")
    registered[0]()
    served = json.loads((tmp_path / "served.json").read_text())
    assert served["m"]["calls"] == 1 and served["m"]["requested_providers"] == {"deepinfra": 1}


def test_both_games_pin_the_same_way():
    from codenames.hosted import Model, pinned
    from hanabi.hosted import pinned as hanabi_pinned

    ours = pinned(Model("openai/gpt-oss-120b", "novita", {"reasoning_effort": "medium"}))
    assert ours == hanabi_pinned("openai/gpt-oss-120b", "novita", {"reasoning_effort": "medium"})
