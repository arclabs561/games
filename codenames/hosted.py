"""Language models served by Hugging Face Inference Providers, with a cost meter and a log.

`Hosted(model, meter, log)` is a `complete` function for codenames.llm: it sends one
prompt as a user message and returns the reply text. Every call, failed or not, is
written to the log as one JSON line with the model id, the provider, the time, every
request parameter, the prompt, the reply and the usage the provider reported.

**Pinning.** The router sends language-model requests through OpenRouter, which may move
a request to another provider. A model id with a provider suffix (`repo:deepinfra`) is
pinned: the router's documentation says such a provider "is never replaced". The request
also carries OpenRouter's provider object (`order`, `allow_fallbacks: false`, and a
quantization filter when the model names one). A check on 2026-11-20
(`codenames.providers pincheck`, results/llm-providers.md) found that the router ignores
that object when the suffix is absent, so the suffix is what pins. Each reply is checked:
the provider that served it is read from the body (`provider`, which OpenRouter adds) or
from the router's `x-inference-provider` header, and a reply from another provider is
logged as an error and sent again. Each record holds `served_by` (model id, provider,
response id, fingerprint, service tier, and quantization when a reply gives one, which
none did in the check) and `pin_held`. `repeat_check` sends prompts again and counts
changed replies.

The **meter** adds up the cost of each call: the cost in the usage object when the
router reports one (`usage["cost"]`, in dollars), and otherwise the reported tokens
times the provider's listed price (`listed_prices`, read from the router when the job
starts). With neither, it charges a deliberately high price (FALLBACK_PRICE). Each
call's record says which of the three it used. Before each call the meter reserves
room for it, and raises OverBudgetError instead of calling when the reserved and spent
amounts would pass the cap. So a job that is given a cap stops below it.

A call that fails five times (with waits of 2, 4, 8, 16 and 32 seconds) is returned as
"", which the players in codenames.llm treat as a reply that cannot be parsed.

No call is made when this module is imported, and the tests use a stub client.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Dollars per million prompt and completion tokens, used only when a reply has no cost.
# It is above the listed price of every model in codenames.c4, so the meter overcounts.
FALLBACK_PRICE = (3.0, 15.0)
THINK = re.compile(r"<think>.*?(</think>|$)", re.S)
LOG_LOCK = threading.Lock()  # one lock for every log file, so lines never interleave


class OverBudgetError(RuntimeError):
    """The next call could take the spending past the cap."""


@dataclass
class Meter:
    """Spending in dollars against a cap, safe to share between threads."""

    cap: float
    spent: float = 0.0
    reserved: float = 0.0
    calls: int = 0
    largest: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reserve(self, floor: float = 0.002) -> float:
        """Set aside room for one call: twice the dearest call so far, and at least floor."""
        amount = max(floor, 2 * self.largest)
        with self.lock:
            if self.spent + self.reserved + amount > self.cap:
                raise OverBudgetError(f"spent {self.spent:.4f} of {self.cap:.4f}")
            self.reserved += amount
        return amount

    def settle(self, amount: float, cost: float) -> None:
        with self.lock:
            self.reserved -= amount
            self.spent += cost
            self.calls += 1
            self.largest = max(self.largest, cost)


ROUTER_MODELS = "https://router.huggingface.co/v1/models"
Price = tuple[float, float]  # dollars per million prompt tokens, per million completion tokens


def parse_prices(listing: dict) -> dict[str, dict[str, Price]]:
    """{model id: {provider: price}} from the router's model list."""
    out: dict[str, dict[str, Price]] = {}
    for m in listing.get("data", []):
        for p in m.get("providers") or []:
            price = p.get("pricing") or {}
            if isinstance(price.get("input"), int | float) and isinstance(
                price.get("output"), int | float
            ):
                out.setdefault(m["id"], {})[p["provider"]] = (price["input"], price["output"])
    return out


def listed_prices() -> dict[str, dict[str, Price]]:
    """The providers' listed prices, from the router (needs network access to it)."""
    from urllib.request import Request, urlopen

    req = Request(ROUTER_MODELS, headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"})
    with urlopen(req, timeout=60) as r:
        return parse_prices(json.load(r))


OPENROUTER_ENDPOINTS = "https://openrouter.ai/api/v1/models/{id}/endpoints"
OPENROUTER_ORGS = {"deepseek-ai": "deepseek"}  # where OpenRouter's author name differs


def openrouter_id(repo: str) -> str:
    """OpenRouter's id for a Hub model, as its model list writes it (lowercase)."""
    org, _, name = repo.partition("/")
    return f"{OPENROUTER_ORGS.get(org, org)}/{name}".lower()


def parse_quantizations(listing: dict) -> dict[str, str | None]:
    """{provider name: quantization} from OpenRouter's endpoint list for one model."""
    data = listing.get("data") or {}
    return {e.get("provider_name"): e.get("quantization") for e in data.get("endpoints", [])}


def listed_quantizations(models: list[Model]) -> dict:
    """Each pinned provider's quantization as OpenRouter lists it, read when a job starts.
    Replies do not say how they were served, so this is the record; a provider that
    OpenRouter does not list (Novita, which the router reaches directly) is None."""
    from urllib.request import urlopen

    out: dict = {"source": OPENROUTER_ENDPOINTS, "time": datetime.now(UTC).isoformat()}
    for m in models:
        try:
            with urlopen(OPENROUTER_ENDPOINTS.format(id=openrouter_id(m.repo)), timeout=30) as r:
                listed = parse_quantizations(json.load(r))
        except (OSError, ValueError) as e:  # recorded; the run goes on
            out[m.repo] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            continue
        mine = next(
            (q for p, q in listed.items() if provider_key(p) == provider_key(m.provider)), None
        )
        out[m.repo] = {"provider": m.provider, "quantization": mine, "listed": listed}
    return out


def record_serving(models: list[Model], out: Path) -> None:
    """At the start of a job: save the listed quantizations (quantizations.json). At its
    end: summarize who served every logged call in OUT (served.json)."""
    import atexit

    (out / "quantizations.json").write_text(json.dumps(listed_quantizations(models), indent=1))

    def served() -> None:
        from codenames.providers import normalize, summarize

        logs = {}
        for path in sorted(out.glob("calls-*.jsonl")):
            rows = [normalize(json.loads(x)) for x in path.read_text().splitlines() if x.strip()]
            if rows:
                logs[path.stem.removeprefix("calls-")] = summarize(rows)
        (out / "served.json").write_text(json.dumps(logs, indent=1, sort_keys=True) + "\n")

    atexit.register(served)


def call_cost(usage: dict, price: Price | None = None) -> tuple[float, str]:
    """The cost of one call in dollars, and where it came from."""
    cost = usage.get("cost")
    if isinstance(cost, int | float):
        return float(cost), "reported"
    p, c = price or FALLBACK_PRICE
    tokens_in, tokens_out = usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0
    return (p * tokens_in + c * tokens_out) / 1e6, "listed price" if price else "fallback price"


def field_of(obj: Any, name: str) -> Any:
    """A field of a reply, which is a dict in the stubs and a dict subclass in
    huggingface_hub (whose extra keys, such as OpenRouter's, are kept)."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def served_by(out: Any) -> dict:
    """What a reply says about who served it: the model id, the provider (OpenRouter adds
    `provider` to the body, and `openrouter_metadata` when the metadata header is sent),
    the response id, the system fingerprint, the service tier and any quantization field.
    Fields the reply lacks are None; the quantization is None on every provider seen so
    far, so a model's quantization is read from OpenRouter's endpoint list instead
    (codenames.providers pincheck)."""
    meta = field_of(out, "openrouter_metadata") or {}
    selected = [e for e in (field_of(field_of(meta, "endpoints") or {}, "available") or []) if e]
    selected = [e for e in selected if field_of(e, "selected")]
    quant = next(
        (v for k, v in (out.items() if isinstance(out, dict) else []) if "quant" in str(k).lower()),
        None,
    )
    return {
        "model": field_of(out, "model"),
        "provider": field_of(out, "provider")
        or (field_of(selected[0], "provider") if selected else None),
        "router_header": field_of(out, "_router_header"),
        "id": field_of(out, "id"),
        "system_fingerprint": field_of(out, "system_fingerprint"),
        "service_tier": field_of(out, "service_tier"),
        "quantization": quant,
        "routing": field_of(meta, "summary"),
        "attempt": field_of(meta, "attempt"),
    }


def visible(text: str) -> str:
    """The reply without a <think> block, for models that put their reasoning in the text."""
    return THINK.sub("", text).strip()


@dataclass
class Model:
    """A model on the router: its repository id, the provider it is pinned to, the
    request fields that are particular to it (such as a reasoning setting), and the
    quantizations it may be served at (empty: not filtered, as in every run so far)."""

    repo: str
    provider: str
    extra: dict = field(default_factory=dict)
    price: Price | None = None
    quantizations: tuple[str, ...] = ()


def pinned(model: Model) -> tuple[str, dict]:
    """The model id and the extra request fields that pin the model's provider."""
    if model.provider in ("", "auto") or ":" in model.repo:
        raise ValueError(f"{model.repo}: name one provider, not {model.provider!r}")
    route: dict = {"order": [model.provider], "allow_fallbacks": False}
    if model.quantizations:
        route["quantizations"] = list(model.quantizations)
    return f"{model.repo}:{model.provider}", {**model.extra, "provider": route}


def provider_key(name: str | None) -> str:
    """A provider name in a form that matches across the router and OpenRouter:
    "DeepInfra" and "deepinfra", "Featherless" and "featherless-ai"."""
    key = re.sub(r"[^a-z0-9]", "", (name or "").lower())
    return key.removesuffix("ai") if len(key) > 4 else key


def served_provider(served: dict) -> str | None:
    """The provider that served a reply: OpenRouter's `provider` field, or else the
    router's header when it names a provider and not OpenRouter itself."""
    if served.get("provider"):
        return served["provider"]
    header = served.get("router_header")
    return header if header and provider_key(header) != "openrouter" else None


def pin_held(model: Model, served: dict) -> bool | None:
    """True if the reply came from the pinned provider, None if the reply does not say."""
    who = served_provider(served)
    return None if who is None else provider_key(who) == provider_key(model.provider)


class RouterClient:
    """POSTs to the router's OpenAI-compatible endpoint and returns the reply body, with
    the router's `x-inference-provider` header under the key `_router_header`."""

    url = "https://router.huggingface.co/v1/chat/completions"

    def __init__(self, token: str, timeout: int = 120):
        self.token, self.timeout = token, timeout

    def chat_completion(self, messages, model, extra_body=None, **params) -> dict:
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen

        body = {"model": model, "messages": messages, **params, **(extra_body or {})}
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        req = Request(self.url, json.dumps(body).encode(), headers)
        try:
            with urlopen(req, timeout=self.timeout) as r:
                out = json.load(r)
                out["_router_header"] = r.headers.get("x-inference-provider")
                return out
        except HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from e


class Hosted:
    """A `complete` function that calls one model through Inference Providers."""

    def __init__(
        self,
        model: Model,
        meter: Meter,
        log: Path | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int = 0,
        client=None,
        attempts: int = 5,
        sleep=time.sleep,
    ):
        self.model, self.meter, self.log = model, meter, log
        self.params = {"max_tokens": max_tokens, "temperature": temperature, "seed": seed}
        self.client, self.attempts, self.sleep = client, attempts, sleep

    def connect(self):
        if self.client is None:
            self.client = RouterClient(os.environ["HF_TOKEN"])
        return self.client

    def __call__(self, prompt: str) -> str:
        return self.call(prompt)["reply"]

    def call(self, prompt: str, tag: dict | None = None) -> dict:
        """Send one prompt; return the logged record (reply, usage, cost, errors)."""
        model_id, extra = pinned(self.model)
        record: dict = {
            "model": self.model.repo,
            "provider": self.model.provider,
            "pinned": model_id,
            "params": {**self.params, "extra_body": extra},
            "prompt": prompt,
            "tag": tag or {},
            "errors": [],
        }
        reserved = self.meter.reserve()
        cost = 0.0
        try:
            for attempt in range(self.attempts):
                record["time"] = datetime.now(UTC).isoformat(timespec="seconds")
                start = time.perf_counter()
                try:
                    out = self.connect().chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        model=model_id,
                        extra_body=extra,
                        **self.params,
                    )
                except Exception as e:  # network, rate limit, provider error: retry
                    record["errors"].append(f"{type(e).__name__}: {str(e)[:300]}")
                    self.sleep(2 ** (attempt + 1))
                    continue
                record["seconds"] = round(time.perf_counter() - start, 2)
                choice = out["choices"][0]
                message = choice["message"]
                usage = dict(out.get("usage") or {})
                this, source = call_cost(usage, self.model.price)
                cost += this
                served = served_by(out)
                record.update(
                    raw=message.get("content") or "",
                    reasoning=message.get("reasoning_content") or message.get("reasoning"),
                    finish=choice.get("finish_reason"),
                    usage=usage,
                    cost=cost,
                    cost_source=source,
                    served=out.get("model"),
                    served_by=served,
                    pin_held=pin_held(self.model, served),
                    attempts=attempt + 1,
                )
                if record["pin_held"] is False:  # paid for, but not the pinned provider
                    record["errors"].append(f"served by {served_provider(served)}")
                    record["raw"] = ""
                    self.sleep(2 ** (attempt + 1))
                    continue
                break
        finally:
            self.meter.settle(reserved, cost)
        record["reply"] = visible(record.get("raw", ""))
        if self.log is not None:
            with LOG_LOCK, self.log.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        return record


def repeat_check(hosted: Hosted, prompts: list[str], times: int = 2, tag=None) -> dict:
    """Send each prompt `times` times and count the prompts whose replies were not all the
    same, and the providers that served them. Providers are not deterministic at
    temperature 0, so this is reported beside any result that rests on one reply."""
    changed = 0
    served: dict[str, int] = {}
    pins = {"held": 0, "broken": 0, "unknown": 0}
    for i, prompt in enumerate(prompts):
        recs = [
            hosted.call(prompt, {**(tag or {}), "part": "repeat", "item": i}) for _ in range(times)
        ]
        changed += len({r["reply"] for r in recs}) > 1
        for r in recs:
            who = str(served_provider(r.get("served_by") or {}))
            served[who] = served.get(who, 0) + 1
            held = r.get("pin_held")
            pins["unknown" if held is None else "held" if held else "broken"] += 1
    return {
        "prompts": len(prompts),
        "times": times,
        "changed": changed,
        "changed_share": changed / len(prompts) if prompts else None,
        "served": served,
        "pin": pins,
    }
