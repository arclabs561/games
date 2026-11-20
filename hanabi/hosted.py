"""Hosted language models as Hanabi players, through Hugging Face Inference Providers.

`Hosted(model, provider, meter)` is a function from prompt to reply for `llm.play_text_game`
and `h2.play_game`. Each call sends the prompt as one user message with temperature 0 and a
fixed seed, retries a failed request three times, and logs the prompt, the reply, the model
id the provider reports, the request parameters, the token counts and the cost.

The provider is pinned as in `codenames/hosted.py`: the model id carries the provider as a
suffix (`openai/gpt-oss-120b:novita`), which the router never replaces, and the request
carries OpenRouter's provider object with fallbacks off, which the router was found to
ignore on 2026-11-20. Each reply's provider is read from its body or the router's
`x-inference-provider` header and logged with `pin_held`; a reply from another provider
counts as a failed attempt. The path route that H2 used (`router.huggingface.co/<provider>`)
refused chat requests on 2026-11-20.

The providers' usage objects report token counts but no cost (checked on Cerebras, Groq and
DeepInfra, November 2026), so `Meter` prices each call at the provider's listed price per
million tokens, as the router's model list gives it, and refuses a call that could take the
total past its cap. The amount billed is not visible from here; the meter's total is the
listed price times the reported tokens.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any

ROUTER = "https://router.huggingface.co/v1/models"
ROUTER_CHAT = "https://router.huggingface.co/v1/chat/completions"


class CapError(Exception):
    """The next call could take the spending past the cap."""


class ProviderError(Exception):
    """A request failed on every attempt, for reasons outside the model."""


@dataclass(frozen=True)
class Price:
    """Listed dollars per million input and output tokens."""

    input: float
    output: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.input + completion_tokens * self.output) / 1e6


def listed_price(models: list[dict], model: str, provider: str) -> Price:
    """The price of `model` at `provider` in the router's model list (its `data` field)."""
    for m in models:
        if m["id"] == model:
            for p in m["providers"]:
                if p["provider"] == provider and "pricing" in p:
                    return Price(p["pricing"]["input"], p["pricing"]["output"])
    raise LookupError(f"no listed price for {model} at {provider}")


def fetch_models(token: str) -> list[dict]:
    """The router's model list, with providers and prices."""
    req = urllib.request.Request(ROUTER, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["data"]


def pinned(model: str, provider: str, extra: dict | None = None) -> tuple[str, dict]:
    """The model id and extra request fields that pin one provider (as codenames.hosted)."""
    if provider in ("", "auto") or ":" in model:
        raise ValueError(f"{model}: name one provider, not {provider!r}")
    return f"{model}:{provider}", {**(extra or {}), "provider": {"order": [provider],
                                                               "allow_fallbacks": False}}  # fmt: skip


def provider_key(name: str | None) -> str:
    key = re.sub(r"[^a-z0-9]", "", (name or "").lower())
    return key.removesuffix("ai") if len(key) > 4 else key


def served_provider(reply: Any) -> str | None:
    """OpenRouter's `provider` field, or the router's header when it names a provider."""
    who = getattr(reply, "provider", None)
    if who:
        return who
    header = getattr(reply, "_router_header", None)
    return header if header and provider_key(header) != "openrouter" else None


def listed_quantization(model: str, provider: str) -> dict:
    """The provider's quantization for the model in OpenRouter's endpoint list, read when
    a run starts (as codenames.hosted.listed_quantizations). None when not listed."""
    org, _, name = model.partition("/")
    rid = f"{ {'deepseek-ai': 'deepseek'}.get(org, org) }/{name}".lower()
    url = f"https://openrouter.ai/api/v1/models/{rid}/endpoints"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            endpoints = (json.load(r).get("data") or {}).get("endpoints", [])
    except (OSError, ValueError) as e:  # recorded; the run goes on
        return {"source": url, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    listed = {e.get("provider_name"): e.get("quantization") for e in endpoints}
    mine = next((q for p, q in listed.items() if provider_key(p) == provider_key(provider)), None)
    return {"source": url, "quantization": mine, "listed": listed}


class RouterClient:
    """POSTs to the router's chat endpoint; returns the reply with attribute access, and
    the router's `x-inference-provider` header as `_router_header`."""

    def __init__(self, token: str, timeout: int = 300):
        self.token, self.timeout = token, timeout

    def chat_completion(self, messages, model, extra_body=None, **params):
        from types import SimpleNamespace
        from urllib.error import HTTPError

        body = {"model": model, "messages": messages, **params, **(extra_body or {})}
        req = urllib.request.Request(
            ROUTER_CHAT,
            json.dumps(body).encode(),
            {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.loads(r.read(), object_hook=lambda d: SimpleNamespace(**d))
                out._router_header = r.headers.get("x-inference-provider")
                return out
        except HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from e


@dataclass
class Meter:
    """Running cost of calls, shared by threads. A call is allowed only if the total stays
    at or below the cap even when every call in flight costs its most."""

    price: Price
    cap: float
    max_call: float  # the most one call can cost: longest prompt and a full reply
    spent: float = 0.0
    calls: int = 0
    in_flight: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def reserve(self) -> None:
        with self.lock:
            if self.spent + (self.in_flight + 1) * self.max_call > self.cap:
                raise CapError(f"spent ${self.spent:.4f} of ${self.cap:.2f}")
            self.in_flight += 1

    def settle(self, prompt_tokens: int, completion_tokens: int) -> float:
        cost = self.price.cost(prompt_tokens, completion_tokens)
        with self.lock:
            self.in_flight -= 1
            self.spent += cost
            self.calls += 1
        return cost

    def release(self) -> None:
        with self.lock:
            self.in_flight -= 1


class Hosted:
    """A prompt-to-reply function backed by one model at one provider. `log` receives one
    dict per call; `client` defaults to huggingface_hub's InferenceClient."""

    def __init__(
        self,
        model: str,
        provider: str,
        meter: Meter,
        max_tokens: int = 512,
        reasoning_effort: str | None = None,
        seed: int = 0,
        retries: int = 3,
        client=None,
        log=None,
        wait: float = 5.0,
    ):
        self.model, self.provider, self.meter = model, provider, meter
        extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        self.model_id, pin = pinned(model, provider, extra)
        self.params: dict = {"max_tokens": max_tokens, "temperature": 0, "seed": seed}
        self.params["extra_body"] = pin
        self.retries, self.log, self.wait = retries, log, wait
        if client is None:
            client = RouterClient(os.environ.get("HF_TOKEN", ""))
        self.client = client

    def __call__(self, prompt: str) -> str:
        errors = []
        for attempt in range(self.retries + 1):
            self.meter.reserve()
            start = time.time()
            try:
                r = self.client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_id,
                    **self.params,
                )
            except Exception as e:  # provider errors of any kind; the model is not at fault
                self.meter.release()
                errors.append(f"{type(e).__name__}: {str(e)[:300]}")
                time.sleep(self.wait * 2**attempt)
                continue
            usage = r.usage
            p, c = (usage.prompt_tokens, usage.completion_tokens) if usage else (0, 0)
            cost = self.meter.settle(p, c)
            choice = r.choices[0]
            reply = choice.message.content or ""
            who = served_provider(r)
            held = None if who is None else provider_key(who) == provider_key(self.provider)
            if self.log is not None:
                self.log(
                    {
                        "time": start,
                        "seconds": round(time.time() - start, 3),
                        "model": self.model,
                        "provider": self.provider,
                        "pinned": self.model_id,
                        "served_model": r.model,
                        "served_provider": who,
                        "router_header": getattr(r, "_router_header", None),
                        "service_tier": getattr(r, "service_tier", None),
                        "pin_held": held,
                        "id": r.id,
                        "params": self.params,
                        "attempt": attempt,
                        "errors": errors,
                        "prompt": prompt,
                        "reply": reply,
                        "reasoning": getattr(choice.message, "reasoning", None)
                        or getattr(choice.message, "reasoning_content", None),
                        "finish_reason": choice.finish_reason,
                        "prompt_tokens": p,
                        "completion_tokens": c,
                        "cost": cost,
                    }
                )
            if held is False:  # paid for, but another provider answered: try again
                errors.append(f"served by {who}")
                time.sleep(self.wait * 2**attempt)
                continue
            return reply
        raise ProviderError("; ".join(errors))


def repeat_check(complete, prompts: list[str], times: int = 2) -> dict:
    """Send each prompt `times` times through a prompt-to-reply function and count the
    prompts whose replies were not all the same."""
    changed = sum(len({complete(p) for _ in range(times)}) > 1 for p in prompts)
    return {"prompts": len(prompts), "times": times, "changed": changed}
