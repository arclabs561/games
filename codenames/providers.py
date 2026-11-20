"""Which provider served each hosted call, from the logs, and a check that pinning holds.

Hugging Face Inference Providers sends language-model requests through OpenRouter, which
picks a provider per request and may retry on another one (the router's documentation,
"Routing and Providers", read 2026-11-19). The logs of C4, C6, C8, C10 and H2 record the
provider each request named and the model id in each reply, but not the provider or the
quantization that served it. Two commands:

    python -m codenames.providers audit --runs RUNS --out providers.json
    python -m codenames.providers pincheck --cap 0.05 --out OUT

`audit` reads every call log under RUNS (the private dataset's `runs/` folder) and
writes aggregates only: per run and model, the providers named, the model ids and
provider fields in the replies, the shape of the response ids, finish reasons, empty
replies, latencies, and a least-squares line of reported prompt tokens against prompt
characters. A change of provider or of chat template shows as a change in that line,
since the tokens a provider adds around a prompt differ. No prompt, reply or id is
written.

`pincheck` sends one short prompt to each pinned model in several ways (the provider
in the path, the provider as a suffix, OpenRouter's provider object, a quantization
filter no provider meets, and the router-metadata header), and records what each reply
says about who served it. It also saves the router's provider list for each model and,
where OpenRouter lists it, each provider's quantization. It runs in a Hugging Face Job,
which holds the token, under a cost cap.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path

# --- reading what a reply says about who served it ----------------------------------------

ID_SHAPES = (
    ("gen-", re.compile(r"^gen-\d+-\w+$")),  # OpenRouter's generation ids
    ("chatcmpl-", re.compile(r"^chatcmpl-[\w-]+$")),
    ("hex32", re.compile(r"^[0-9a-f]{32}$")),
    ("uuid", re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")),
)


def id_shape(rid: object) -> str:
    """The shape of a response id, without the id: gen-, chatcmpl-, hex32, uuid, other."""
    if not isinstance(rid, str) or not rid:
        return "none"
    return next((name for name, pat in ID_SHAPES if pat.match(rid)), "other")


def linear_fit(xs: list[float], ys: list[float]) -> dict | None:
    """Least-squares intercept and slope of ys on xs, with the residual spread."""
    n = len(xs)
    if n < 3 or len(set(xs)) < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    intercept = my - slope * mx
    res = [y - intercept - slope * x for x, y in zip(xs, ys, strict=True)]
    return {
        "n": n,
        "intercept": round(intercept, 2),
        "slope": round(slope, 4),
        "residual_sd": round(math.sqrt(sum(r * r for r in res) / (n - 2)), 2),
    }


def quantile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * len(s)))], 2)


def normalize(rec: Mapping) -> dict:
    """One call record of codenames.hosted or hanabi.hosted, in common field names."""
    usage = rec.get("usage") or {}
    reply = rec.get("reply")
    raw = rec.get("raw", reply)
    when = rec.get("time")
    if isinstance(when, int | float):
        when = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(when))
    served = rec.get("served_by") or {}
    return {
        "requested": rec.get("provider"),
        "model": rec.get("model"),
        "served_model": rec.get("served", rec.get("served_model")),
        "served_provider": served.get("provider"),
        "quantization": served.get("quantization"),
        "id_shape": id_shape(rec.get("id", served.get("id"))),
        "finish": rec.get("finish", rec.get("finish_reason")),
        "empty": not (raw or "").strip() and not (reply or "").strip(),
        "prompt_chars": len(rec.get("prompt") or ""),
        "prompt_tokens": usage.get("prompt_tokens", rec.get("prompt_tokens")),
        "completion_tokens": usage.get("completion_tokens", rec.get("completion_tokens")),
        "seconds": rec.get("seconds"),
        "time": when,
        "retried": bool(rec.get("errors")),
        "answered": "seconds" in rec,
    }


def summarize(calls: list[dict]) -> dict:
    """Aggregates of normalized calls: counts, never strings from prompts or replies."""
    answered = [c for c in calls if c["answered"]]
    fit_rows = [c for c in answered if c["prompt_tokens"]]
    secs = [c["seconds"] for c in answered if isinstance(c["seconds"], int | float)]
    empty_secs = [c["seconds"] for c in answered if c["empty"] and c["seconds"] is not None]
    times = sorted(c["time"] for c in calls if c["time"])

    def count(key: str) -> dict:
        return dict(Counter(str(c[key]) for c in calls).most_common())

    return {
        "calls": len(calls),
        "answered": len(answered),
        "requested_providers": count("requested"),
        "served_models": count("served_model"),
        "served_providers": count("served_provider"),
        "quantizations": count("quantization"),
        "id_shapes": count("id_shape"),
        "finish": count("finish"),
        "empty": sum(c["empty"] for c in answered),
        "empty_with_stop": sum(c["empty"] and c["finish"] == "stop" for c in answered),
        "retried": sum(c["retried"] for c in calls),
        "seconds_median": quantile(secs, 0.5),
        "seconds_p90": quantile(secs, 0.9),
        "seconds_median_empty": quantile(empty_secs, 0.5),
        "prompt_tokens_on_chars": linear_fit(
            [c["prompt_chars"] for c in fit_rows], [c["prompt_tokens"] for c in fit_rows]
        ),
        "first": times[0] if times else None,
        "last": times[-1] if times else None,
    }


def call_logs(runs: Path) -> Iterable[tuple[str, str, str, Path]]:
    """(experiment, run, log name, path) for every call log under runs/<experiment>/<run>/.
    H2 keeps one log per game under calls/; those are grouped as one log, "calls"."""
    for path in sorted(runs.rglob("*.jsonl")):
        parts = path.relative_to(runs).parts
        if len(parts) < 3:
            continue
        experiment, run = parts[0], parts[1]
        if parts[2] == "calls":
            yield experiment, run, "calls", path
        elif path.name.startswith("calls-"):
            yield experiment, run, path.stem.removeprefix("calls-"), path


def audit(runs: Path) -> dict:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    unread: list[str] = []
    for experiment, run, name, path in call_logs(runs):
        rows = groups.setdefault((experiment, run, name), [])
        try:
            with path.open() as f:
                for line in f:
                    if line.strip():
                        rows.append(normalize(json.loads(line)))
        except (OSError, ValueError, AttributeError) as e:  # recorded, the rest is read
            unread.append(f"{path.relative_to(runs)}: {type(e).__name__}: {str(e)[:200]}")
    out: dict = {"unread": unread} if unread else {}
    for (experiment, run, name), calls in sorted(groups.items()):
        manifest = runs / experiment / run / "manifest.json"
        info = json.loads(manifest.read_text()) if manifest.exists() else {}
        entry = out.setdefault(experiment, {}).setdefault(
            run, {"commit": info.get("commit"), "job_id": info.get("job_id"), "logs": {}}
        )
        entry["logs"][name] = summarize(calls)
    return out


# --- the pinning check -------------------------------------------------------------------

ROUTER = "https://router.huggingface.co"
OPENROUTER_ENDPOINTS = "https://openrouter.ai/api/v1/models/{id}/endpoints"
# The models and providers that C4, C6, C8, C10 and H2 named, with OpenRouter's id for
# each model (lowercase, as OpenRouter's model list writes them).
PINNED = (
    ("openai/gpt-oss-120b", "deepinfra", "openai/gpt-oss-120b", {"reasoning_effort": "low"}),
    ("openai/gpt-oss-120b", "novita", "openai/gpt-oss-120b", {"reasoning_effort": "low"}),
    ("deepseek-ai/DeepSeek-V3.2", "deepinfra", "deepseek/deepseek-v3.2", {}),
    ("meta-llama/Llama-3.3-70B-Instruct", "novita", "meta-llama/llama-3.3-70b-instruct", {}),
    ("Qwen/Qwen3-32B", "deepinfra", "qwen/qwen3-32b", {}),
    ("Qwen/Qwen3.8-27B", "deepinfra", "qwen/qwen3.8-27b", {}),
)
PROMPT = "Answer with one word: what color is the sky on a clear day?"


def post(url: str, body: dict, headers: dict, timeout: int = 120) -> tuple[int, dict, dict]:
    """POST JSON; return the status, the response headers and the parsed body."""
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    req = Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", **headers})
    try:
        with urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), json.load(r)
    except HTTPError as e:
        text = e.read().decode(errors="replace")[:500]
        return e.code, dict(e.headers), {"error_text": text}


def get(url: str, headers: dict | None = None) -> dict:
    from urllib.request import Request, urlopen

    try:
        with urlopen(Request(url, headers=headers or {}), timeout=60) as r:
            return json.load(r)
    except Exception as e:  # recorded, not fatal: the check reports what it could read
        return {"error_text": f"{type(e).__name__}: {str(e)[:300]}"}


def provider_id(repo: str, provider: str, token: str) -> str:
    """The provider's own id for a model, from the Hub's provider mapping."""
    info = get(
        f"https://huggingface.co/api/models/{repo}?expand[]=inferenceProviderMapping",
        {"Authorization": f"Bearer {token}"},
    )
    mapping = info.get("inferenceProviderMapping") or {}
    if isinstance(mapping, list):  # newer Hub responses list the providers
        mapping = {m.get("provider"): m for m in mapping}
    return (mapping.get(provider) or {}).get("providerId", repo)


def variants(repo: str, provider: str, extra: dict, pid: str) -> list[tuple[str, str, dict, dict]]:
    """(name, url, body, headers) for each way of naming the provider."""
    base = {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 256, **extra}
    base |= {"temperature": 0, "seed": 0}
    pin = {"order": [provider], "allow_fallbacks": False}
    meta = {"X-OpenRouter-Metadata": "enabled"}
    path_url = f"{ROUTER}/{provider}/v1/chat/completions"
    auto_url = f"{ROUTER}/v1/chat/completions"
    return [
        ("path", path_url, base | {"model": pid}, {}),
        ("suffix", auto_url, base | {"model": f"{repo}:{provider}"}, {}),
        ("suffix+object", auto_url, base | {"model": f"{repo}:{provider}", "provider": pin}, meta),
        ("object", auto_url, base | {"model": repo, "provider": pin}, meta),
        (
            "object+impossible quantization",
            auto_url,
            base | {"model": repo, "provider": pin | {"quantizations": ["fp32"]}},
            meta,
        ),
    ]


def described(status: int, headers: dict, body: dict) -> dict:
    """What one reply says about where it was served, and nothing of its text."""
    from codenames.hosted import served_by

    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    lower = {k.lower(): v for k, v in headers.items()}
    keep = ("x-inference-provider", "x-router", "x-openrouter-provider", "x-request-id")
    return {
        "status": status,
        "served_by": served_by(body),
        "headers": {k: lower[k] for k in keep if k in lower and k != "x-request-id"},
        "header_names": sorted(k for k in lower if k.startswith(("x-", "openrouter"))),
        "body_keys": sorted(body),
        "finish": choice.get("finish_reason"),
        "reply_chars": len(message.get("content") or ""),
        "usage": body.get("usage"),
        "error": body.get("error_text") or body.get("error"),
    }


def check_one(repo, provider, or_id, extra, auth, token, cap, spent, repeats, results, listings):
    """Read the listings of one model and send its variants; return the new spending."""
    listing = get(f"{ROUTER}/v1/models/{repo}", auth)
    providers = listing.get("providers") or (listing.get("data") or [{}])[0].get("providers")
    endpoints = get(OPENROUTER_ENDPOINTS.format(id=or_id))
    listings[repo] = {
        "router_providers": providers or listing,
        "openrouter_endpoints": [
            {k: e.get(k) for k in ("provider_name", "tag", "quantization", "status")}
            for e in (endpoints.get("data") or {}).get("endpoints", [])
        ]
        or endpoints,
    }
    price = next(
        (p.get("pricing") for p in providers or [] if p.get("provider") == provider), None
    ) or {"input": 3.0, "output": 15.0}
    pid = provider_id(repo, provider, token)
    for name, url, body, headers in variants(repo, provider, extra, pid):
        for attempt in range(repeats):
            row = {"repo": repo, "provider": provider, "variant": name, "attempt": attempt}
            if spent + (256 * price["output"] + 500 * price["input"]) / 1e6 > cap:
                results.append(row | {"skipped": "cap"})
                continue
            start = time.time()
            status, h, b = post(url, body, auth | headers)
            usage = b.get("usage") or {}
            cost = (
                (usage.get("prompt_tokens") or 0) * price["input"]
                + (usage.get("completion_tokens") or 0) * price["output"]
            ) / 1e6
            spent += cost
            row |= described(status, h, b) | {"seconds": round(time.time() - start, 2)}
            results.append(row | {"cost": cost})
            print(json.dumps(results[-1])[:400], flush=True)
    return spent


def pincheck(cap: float, out: Path, repeats: int = 2) -> dict:
    token = os.environ["HF_TOKEN"]
    auth = {"Authorization": f"Bearer {token}"}
    spent, results = 0.0, []
    listings: dict[str, dict] = {}
    for repo, provider, or_id, extra in PINNED:
        try:
            spent = check_one(
                repo, provider, or_id, extra, auth, token, cap, spent, repeats, results, listings
            )
        except Exception:  # recorded; the other models are still checked
            import traceback

            results.append({"repo": repo, "provider": provider, "error": traceback.format_exc()})
            print(results[-1]["error"], flush=True)
    report = {
        "what": "pinning check: one short prompt per model and way of pinning, sent twice",
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt": PROMPT,
        "cost_method": "reported tokens times the router's listed price for the provider",
        "spent": sum(r.get("cost", 0.0) for r in results),
        "cap": cap,
        "calls": results,
        "listings": listings,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "pincheck.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--runs", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("pincheck")
    p.add_argument("--cap", type=float, default=0.05)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--runs", type=Path, default=None, help="also audit these logs into OUT")
    args = parser.parse_args(argv)
    if args.command == "audit":
        result = audit(args.runs)
        args.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
        print(json.dumps({e: sorted(r) for e, r in result.items()}))
        return 0
    if args.runs is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        try:
            result = audit(args.runs)
        except Exception:  # the pinning check still runs; the traceback is saved
            import traceback

            result = {"error": traceback.format_exc()}
            print(result["error"], flush=True)
        (args.out / "providers.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    report = pincheck(args.cap, args.out)
    print(json.dumps({"spent": report["spent"], "calls": len(report["calls"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
