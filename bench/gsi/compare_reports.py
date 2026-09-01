#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def indexed(path: Path) -> dict[tuple, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (item["top"], item["prompt_id"], item["run"]): item["response"]
        for item in payload["results"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare traces collected from sequential TP2 launches"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline = indexed(args.baseline)
    candidate = indexed(args.candidate)
    if baseline.keys() != candidate.keys():
        raise ValueError("baseline and candidate case sets differ")
    cases = []
    for key in sorted(baseline):
        expected = baseline[key]
        actual = candidate[key]
        cases.append(
            {
                "top": key[0],
                "prompt_id": key[1],
                "run": key[2],
                "exact_content": expected["content"] == actual["content"],
                "exact_reasoning": (
                    expected["reasoning_content"]
                    == actual["reasoning_content"]
                ),
                "baseline": expected,
                "candidate": actual,
            }
        )
    summary = {
        "cases": len(cases),
        "exact_content": sum(item["exact_content"] for item in cases),
        "exact_reasoning": sum(item["exact_reasoning"] for item in cases),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"summary": summary, "cases": cases}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if summary["exact_content"] != summary["cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
