"""Language models served by Hugging Face Inference Providers, with a cost meter and a log.

`Hosted(model, meter, log)` is a `complete` function for codenames.llm: it sends one
prompt as a user message and returns the reply text. Every call, failed or not, is
written to the log as one JSON line with the model id, the provider, the time, every
request parameter, the prompt, the reply and the usage the provider reported.

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


def call_cost(usage: dict, price: Price | None = None) -> tuple[float, str]:
    """The cost of one call in dollars, and where it came from."""
    cost = usage.get("cost")
    if isinstance(cost, int | float):
        return float(cost), "reported"
    p, c = price or FALLBACK_PRICE
    tokens_in, tokens_out = usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0
    return (p * tokens_in + c * tokens_out) / 1e6, "listed price" if price else "fallback price"


def visible(text: str) -> str:
    """The reply without a <think> block, for models that put their reasoning in the text."""
    return THINK.sub("", text).strip()


@dataclass
class Model:
    """A model on the router: its repository id, the provider it is pinned to, and the
    request fields that are particular to it (such as a reasoning setting)."""

    repo: str
    provider: str
    extra: dict = field(default_factory=dict)
    price: Price | None = None


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
            from huggingface_hub import InferenceClient

            provider: Any = self.model.provider  # a provider name the router knows
            self.client = InferenceClient(
                provider=provider, api_key=os.environ["HF_TOKEN"], timeout=120
            )
        return self.client

    def __call__(self, prompt: str) -> str:
        return self.call(prompt)["reply"]

    def call(self, prompt: str, tag: dict | None = None) -> dict:
        """Send one prompt; return the logged record (reply, usage, cost, errors)."""
        record: dict = {
            "model": self.model.repo,
            "provider": self.model.provider,
            "params": {**self.params, "extra_body": self.model.extra},
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
                        model=self.model.repo,
                        extra_body=self.model.extra or None,
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
                cost, source = call_cost(usage, self.model.price)
                record.update(
                    raw=message.get("content") or "",
                    reasoning=message.get("reasoning_content") or message.get("reasoning"),
                    finish=choice.get("finish_reason"),
                    usage=usage,
                    cost=cost,
                    cost_source=source,
                    served=out.get("model"),
                    attempts=attempt + 1,
                )
                break
        finally:
            self.meter.settle(reserved, cost)
        record["reply"] = visible(record.get("raw", ""))
        if self.log is not None:
            with LOG_LOCK, self.log.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        return record
