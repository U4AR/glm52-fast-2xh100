#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path


def chat(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    *,
    logprobs: bool = False,
) -> dict:
    """One greedy completion.

    With ``logprobs`` the per-token log probabilities and token strings come back
    too, which is what `gsi.evaluate` needs for the paper's perplexity ratio and
    top-1 agreement columns.
    """
    body_payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if logprobs:
        body_payload["logprobs"] = True
    payload = json.dumps(body_payload).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        body = json.load(response)
    elapsed = time.perf_counter() - start
    choice = body["choices"][0]
    message = choice["message"]
    usage = body.get("usage", {})
    result = {
        "content": message.get("content"),
        "reasoning_content": message.get("reasoning_content"),
        "completion_tokens": usage.get("completion_tokens", 0),
        "elapsed": elapsed,
        "tokens_per_second": (
            usage.get("completion_tokens", 0) / elapsed if elapsed else 0
        ),
        "raw_meta": body.get("meta_info", {}),
    }
    entries = (choice.get("logprobs") or {}).get("content") or []
    if entries:
        result["token_logprobs"] = [item.get("logprob") for item in entries]
        # The chat schema exposes token strings rather than ids; they are just as
        # usable for the argmax-agreement comparison and avoid a tokenizer load.
        result["token_ids"] = [item.get("token") for item in entries]
    return result


def load_prompts(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare GSI-off and GSI-on endpoints for one MTP launch mode"
    )
    parser.add_argument("--baseline-url", required=True)
    parser.add_argument("--gsi-url", required=True)
    parser.add_argument("--served-model", default="GLM5.2")
    parser.add_argument("--tops", default="2,8")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path(__file__).with_name("prompts.jsonl"),
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    results = []
    for top in [int(value) for value in args.tops.split(",")]:
        model = f"{args.served_model}-top{top}"
        for prompt in prompts:
            for run in range(args.runs):
                baseline = chat(
                    args.baseline_url, model, prompt["prompt"], args.max_tokens
                )
                candidate = chat(
                    args.gsi_url, model, prompt["prompt"], args.max_tokens
                )
                results.append(
                    {
                        "top": top,
                        "model": model,
                        "prompt_id": prompt["id"],
                        "run": run,
                        "exact_content": baseline["content"] == candidate["content"],
                        "exact_reasoning": (
                            baseline["reasoning_content"]
                            == candidate["reasoning_content"]
                        ),
                        "baseline": baseline,
                        "gsi": candidate,
                    }
                )
    baseline_rates = [item["baseline"]["tokens_per_second"] for item in results]
    gsi_rates = [item["gsi"]["tokens_per_second"] for item in results]
    summary = {
        "cases": len(results),
        "exact_content_fraction": sum(x["exact_content"] for x in results)
        / len(results),
        "exact_reasoning_fraction": sum(x["exact_reasoning"] for x in results)
        / len(results),
        "baseline_median_tokens_per_second": statistics.median(baseline_rates),
        "gsi_median_tokens_per_second": statistics.median(gsi_rates),
        "median_speedup": (
            statistics.median(gsi_rates) / statistics.median(baseline_rates)
            if statistics.median(baseline_rates)
            else 0
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if summary["exact_content_fraction"] != 1.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

