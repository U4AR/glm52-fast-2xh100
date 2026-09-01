#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_matrix import chat, load_prompts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze deterministic traces from one sequential GSI launch"
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--served-model", default="GLM5.2")
    parser.add_argument("--tops", default="2,8")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path(__file__).with_name("prompts.jsonl"),
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--logprobs",
        action="store_true",
        help="also record per-token logprobs for gsi.evaluate's paper metrics",
    )
    args = parser.parse_args()

    results = []
    for top in [int(value) for value in args.tops.split(",")]:
        model = f"{args.served_model}-top{top}"
        for prompt in load_prompts(args.prompts):
            for run in range(args.runs):
                results.append(
                    {
                        "top": top,
                        "model": model,
                        "prompt_id": prompt["id"],
                        "run": run,
                        "response": chat(
                            args.url,
                            model,
                            prompt["prompt"],
                            args.max_tokens,
                            logprobs=args.logprobs,
                        ),
                    }
                )
    payload = {"label": args.label, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"label": args.label, "cases": len(results)}, indent=2))


if __name__ == "__main__":
    main()
