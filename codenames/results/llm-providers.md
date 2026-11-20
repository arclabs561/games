# Which providers served the hosted calls (2026-11-20)

A dated note. It changes no earlier result. The counts come from every logged call of C4, C6, C8, C10 and H2, read by `python -m codenames.providers audit` in a Hugging Face job and reduced to `llm-providers.json` (aggregates only, one line per log). The pinning check of 2026-11-20 is in the same file.

**Findings.**

- Every call of these experiments named one provider, and no log shows a reply from another. The logs did not record the serving provider or the quantization, so for most calls the provider is inferred, not observed. The evidence is the form of the model id and of the response id in each reply.
- C7 made no hosted calls; it trained its model on a rented GPU.
- C10's jump in empty replies, from 1 of 2,401 validation calls to 1,135 of 3,850 test calls (counting retries), came with the same provider, the same model id and the same prompt tokens. What changed was speed. The median call took 16.5 to 18.2 seconds in the two validation jobs and 34.7 to 35.8 seconds in the six scored test jobs, and the empty replies took about 31 seconds in every job that had them. That pattern fits a limit near 30 seconds somewhere upstream that ends a request with an empty reply and a normal stop. The logs cannot show where the limit was.
- On 2026-11-20 the route that these runs used, the provider in the URL path, refuses chat requests. The provider suffix on the model id does pin the provider, as the router's documentation says. OpenRouter's provider object (an order, no fallbacks, a quantization filter) is ignored when sent without the suffix: 8 of 12 such calls went to another provider, and a filter that no provider meets did not stop any call.

## Providers named, and what the replies show

| Experiment | Model | Provider named | Calls logged | Evidence of the provider that served |
|---|---|---|---:|---|
| C4 | Llama 3.3 70B | novita | 2,764 | model id in Novita's spelling; the router reaches Novita without OpenRouter |
| C4 | Qwen3-32B, DeepSeek-V3.2 | deepinfra | 5,487 | model id in the Hub's spelling, as DeepInfra's own API writes it |
| C4 | gpt-oss-120b | deepinfra | 2,722 | none in the reply; same route and hours as the calls above |
| C6 | Llama 3.3 70B | novita | 695 | as in C4 |
| C6 | Qwen3-32B, Qwen3.8-27B, Gemma 4 31B, DeepSeek-V3.2 | deepinfra | 1,580 | model id in the Hub's spelling |
| C6 | gpt-oss-120b | deepinfra | 692 | none in the reply |
| C8 | gpt-oss-120b | deepinfra | 24,596 | none in the reply |
| C10 | gpt-oss-120b (listener and speaker) | deepinfra | 7,036 | none in the reply |
| C10 | Qwen3.8-27B, DeepSeek-V3.2 | deepinfra | 1,517 | model id in the Hub's spelling, up to 22:05 UTC on 2026-11-19 |
| C10 | Llama 3.3 70B | novita | 1,217 | as in C4 |
| H2 | gpt-oss-120b | novita | 3,513 | every response id in Novita's form (32 hex digits) |
| H2 | DeepSeek-V3.2 | deepinfra | 1,621 | every response id in DeepInfra's form (`chatcmpl-`) |
| H2 pilot | gpt-oss-120b | baseten | 36 | every response id `chatcmpl-` |

Counted exactly from the logs; calls include retries and repeat checks. The C10 logs of two failed test jobs end in a line cut short when their disks filled; the calls before that line are counted. Those jobs were not scored.

How the evidence reads. In the pinning check, every reply that passed through OpenRouter had an id beginning `gen-` and OpenRouter's lowercase model id (`deepseek/deepseek-v3.2`, `qwen/qwen3-32b`). No logged reply of these experiments has either for a model where the two spellings differ. So the DeepInfra calls of C4, C6 and C10 did not pass through OpenRouter, and DeepInfra's own API answered them. H2 logged response ids, and none of its 5,170 ids has OpenRouter's form. For gpt-oss-120b the two spellings coincide (`openai/gpt-oss-120b`), and the codenames logs kept no response id. That leaves its provider in C4, C6, C8 and C10 inferred from the route: the same code, the same provider argument and the same hours as calls whose ids show DeepInfra.

## C10's empty replies

| Job | Calls | Empty | Median seconds | Median seconds, empty replies |
|---|---:|---:|---:|---:|
| validation boards 0 to 49 (19:11 to 19:40 UTC) | 1,401 | 1 | 18.2 | 31.0 |
| validation boards 50 to 99 (19:11 to 19:30) | 1,000 | 0 | 16.5 | |
| six scored test jobs (20:00 to 21:42) | 595 to 683 each | 126 to 257 each | 34.7 to 35.8 | 31.3 to 31.8 |
| two test jobs that failed (19:51 to 20:02) | 293 and 87 | 107 and 24 | 35.2 and 36.6 | 31.6 and 31.4 |

From the logs, counted exactly. Every empty test reply ended with a normal stop. The scored jobs hold 3,850 calls: the 3,000 of the plan, 850 sent again after an empty reply, and 1,135 empty replies in all, as the README reports (850 at first, 285 after the retry). A least-squares line of reported prompt tokens on prompt characters, fitted per job, gives 148.1 to 149.7 tokens at 303 characters, the length of a listener prompt, in all ten validation and test jobs. The tokenizer and chat template did not change. C6 shows the same pattern earlier that day: gpt-oss-120b took 19.2 seconds at the median on validation, with no empty reply, and 36.4 seconds on test 4 minutes later, with 40 of 200 empty at about 31.5 seconds each. In C8's labeling job, 574 of 24,300 replies were empty, with a median of 31.8 seconds for those and 21.3 for all calls. Whether DeepInfra changed the deployment, or was under load, is not in the data.

## Quantization

No run recorded it, and no reply in the check reported one. OpenRouter's endpoint list, read by the check on 2026-11-20, gives DeepInfra's quantization as bf16 for gpt-oss-120b, fp4 for DeepSeek-V3.2 and fp8 for Qwen3-32B and Qwen3.8-27B. It does not list Novita for gpt-oss-120b or Llama 3.3, and lists it as unknown for Qwen3.8-27B. These are OpenRouter's entries for DeepInfra's endpoints on 2026-11-20. That DeepInfra's own API served the same weights on 2026-11-19 is likely but not shown.

## The pinning check

One short prompt, sent twice for each of six model and provider pairs and five ways of naming the provider, from a `cpu-basic` job on 2026-11-20 at 03:05 UTC (job `6affb86cefaf5e14f4b2844f`, at commit `51a4995`). The calls cost $0.0012 at the listed prices. The seven `cpu-basic` jobs behind this note and the H2 replay (audit, check, two digests, two replays, and one failed start) ran a few minutes in all at $0.01 an hour.

| Way of naming the provider | Result |
|---|---|
| in the URL path (`router.huggingface.co/deepinfra/v1/...`), as all runs so far | refused: HTTP 400, "Not allowed to POST /v1/chat/completions for provider deepinfra" (and novita) |
| suffix on the model id (`openai/gpt-oss-120b:deepinfra`) | 12 of 12 from the named provider: DeepInfra through OpenRouter, Novita directly |
| suffix and OpenRouter's provider object | 12 of 12 from the named provider |
| provider object alone | 4 of 12 from the named provider; the others from Crusoe, AkashML, ModelRun, Parasail and CoreWeave |
| provider object with the filter `quantizations: ["fp32"]` | all 12 answered, by DeepInfra, Crusoe, AkashML, Google, Tenstorrent, Wafer and CoreWeave |

The first job of the audit (`6affb2e24fdc5ab3c55aaea7`) failed after 5 seconds, before any model call, and the job was run again with error handling.

## What changed in the code

- `codenames/hosted.py` and `hanabi/hosted.py` send each request to the router's chat endpoint with the provider suffix and OpenRouter's provider object (`order`, `allow_fallbacks: false`, and a quantization filter when a model names one). The filter is kept for the day the router applies it; the check shows that it does not now.
- Each reply's provider is read from its body (`provider`) or from the router's `x-inference-provider` header. Each log record holds it with the model id, response id, fingerprint, service tier and any quantization field, and `pin_held`. A reply from another provider is logged as an error and sent again.
- Each job saves `quantizations.json`, OpenRouter's listed quantization for each pinned provider, when it starts, and `served.json`, a summary of who served every logged call, when it ends. H2's `summary.json` holds both.
- `repeat_check` in both modules sends prompts again and counts changed replies, with the providers that served them.

## Limits

- A provider can change its weights or serving stack behind the same name and model id. Pinning the name does not prevent that, and nothing in a reply shows it.
- The pinning check is two calls per cell, on one day.
- Quantization is what OpenRouter lists for a provider, read once per job, not what served each call.
