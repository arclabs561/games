"""Hosted language models as Hanabi players, through Hugging Face Inference Providers.

`Hosted(model, provider, meter)` is a function from prompt to reply for `llm.play_text_game`
and `h2.play_game`. Each call sends the prompt as one user message with temperature 0 and a
fixed seed, retries a failed request three times, and logs the prompt, the reply, the model
id the provider reports, the request parameters, the token counts and the cost.

The providers' usage objects report token counts but no cost (checked on Cerebras, Groq and
DeepInfra, November 2026), so `Meter` prices each call at the provider's listed price per
million tokens, as the router's model list gives it, and refuses a call that could take the
total past its cap. The amount billed is not visible from here; the meter's total is the
listed price times the reported tokens.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, cast

ROUTER = "https://router.huggingface.co/v1/models"


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
        self.params: dict = {"max_tokens": max_tokens, "temperature": 0, "seed": seed}
        if reasoning_effort:
            self.params["extra_body"] = {"reasoning_effort": reasoning_effort}
        self.retries, self.log, self.wait = retries, log, wait
        if client is None:
            from huggingface_hub import InferenceClient

            client = InferenceClient(
                provider=cast(Any, provider), token=os.environ.get("HF_TOKEN"), timeout=300
            )
        self.client = client

    def __call__(self, prompt: str) -> str:
        errors = []
        for attempt in range(self.retries + 1):
            self.meter.reserve()
            start = time.time()
            try:
                r = self.client.chat_completion(
                    messages=[{"role": "user", "content": prompt}], model=self.model, **self.params
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
            if self.log is not None:
                self.log(
                    {
                        "time": start,
                        "seconds": round(time.time() - start, 3),
                        "model": self.model,
                        "provider": self.provider,
                        "served_model": r.model,
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
            return reply
        raise ProviderError("; ".join(errors))
