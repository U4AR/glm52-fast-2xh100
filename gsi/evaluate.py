"""Paper-faithful scoring for GSI runs (arXiv 2605.03109, Table 6).

The paper judges a configuration on four numbers, none of which is exact token
equality: perplexity ratio, top-1 agreement, generation agreement, and the
analytical effective speedup

    S_l = 1 / [ f_l / (d/k) + (1 - f_l) ]                              (Eq. 6)

`effective_speedup` reproduces Eq. 6 exactly, including its assumptions: the
compression ratio is `d/k`, the gate projection is free, and the baseline is a
dense BF16 read.  `deployed_speedup` is the corrected model for this stack --
weights are W4AFP8, images and bases are not, and the `d x k` projection is paid
on every token whichever path it takes.  Both are reported so the paper's number
and the deployable number are never confused for one another.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def effective_speedup(fast_fraction: float, width: int, rank: int) -> float:
    """The paper's Eq. 6.  Dense BF16 baseline, projection cost ignored."""
    if not 0.0 <= fast_fraction <= 1.0:
        raise ValueError(f"fast fraction out of range: {fast_fraction}")
    if rank <= 0 or width <= 0:
        raise ValueError("width and rank must be positive")
    compression = width / rank
    return 1.0 / (fast_fraction / compression + (1.0 - fast_fraction))


def deployed_speedup(
    fast_fraction: float,
    width: int,
    rank: int,
    out_dim: int,
    *,
    weight_bytes: float = 0.5,
    image_bytes: float = 2.0,
    amortization: int = 1,
) -> float:
    """Batch-1 weight-bytes speedup against the quantized path actually served.

    ``weight_bytes`` is bytes per weight element on the slow path (0.5 for the
    packed INT4 of W4AFP8); ``image_bytes`` is bytes per element of the cached
    image and basis.  ``amortization`` is the number of maps sharing one gate
    evaluation -- the routed experts share a single projection across the top-k
    experts a token selects, which is the only place this exceeds 1.
    """
    if amortization <= 0:
        raise ValueError("amortization must be positive")
    baseline = amortization * out_dim * width * weight_bytes
    projection = width * rank * image_bytes
    served = amortization * (
        fast_fraction * out_dim * rank * image_bytes
        + (1.0 - fast_fraction) * out_dim * width * weight_bytes
    )
    total = projection + served
    return baseline / total if total > 0 else math.inf


def break_even_fast_fraction(
    width: int,
    rank: int,
    out_dim: int,
    *,
    weight_bytes: float = 0.5,
    image_bytes: float = 2.0,
    amortization: int = 1,
) -> float:
    """Fast fraction at which `deployed_speedup` reaches 1.0.

    Returns a value above 1.0 when the site can never pay for itself, which is
    the case whenever the image row is not cheaper than the weight row.
    """
    denominator = amortization * out_dim * (
        weight_bytes * width - image_bytes * rank
    )
    if denominator <= 0:
        return math.inf
    return (image_bytes * width * rank) / denominator


@dataclass
class SpeedupReport:
    per_module: dict
    paper_speedup: float
    deployed_speedup: float
    rows: int
    fast_rows: int

    @property
    def fast_fraction(self) -> float:
        return self.fast_rows / self.rows if self.rows else 0.0


def speedup_report(
    telemetry: dict,
    entries: dict,
    *,
    weight_bytes: float = 0.5,
    image_bytes: float = 2.0,
) -> SpeedupReport:
    """Score a telemetry snapshot against the profile that produced it.

    ``telemetry`` is the module -> stats mapping written by `GSITelemetry`;
    ``entries`` maps module -> `GSIEntry`.  Modules present in telemetry but
    absent from the profile are skipped: they were observed, not gated.
    """
    per_module: dict[str, dict] = {}
    rows_total = 0
    fast_total = 0
    paper_cost = 0.0
    paper_baseline = 0.0
    deployed_cost = 0.0
    deployed_baseline = 0.0

    for module, stats in sorted(telemetry.items()):
        entry = entries.get(module)
        if entry is None or not stats.get("rows"):
            continue
        rows = int(stats["rows"])
        fast = float(stats["fast"]) / rows
        width = int(entry.input_dim)
        rank = int(entry.rank)
        out_dim = int(entry.output_dim) or width
        paper = effective_speedup(fast, width, rank)
        deployed = deployed_speedup(
            fast,
            width,
            rank,
            out_dim,
            weight_bytes=weight_bytes,
            image_bytes=image_bytes,
        )
        per_module[module] = {
            "family": entry.family,
            "layer": entry.layer,
            "rows": rows,
            "fast_fraction": fast,
            "width": width,
            "rank": rank,
            "paper_speedup": paper,
            "deployed_speedup": deployed,
            "break_even_fast_fraction": break_even_fast_fraction(
                width, rank, out_dim, weight_bytes=weight_bytes,
                image_bytes=image_bytes,
            ),
        }
        rows_total += rows
        fast_total += int(stats["fast"])
        # Aggregate on cost, not on the mean of ratios: a module that is cheap
        # to begin with must not dominate the headline speedup.
        unit = rows * out_dim * width
        paper_baseline += unit
        paper_cost += unit / paper
        deployed_baseline += unit * weight_bytes
        deployed_cost += unit * weight_bytes / deployed

    return SpeedupReport(
        per_module=per_module,
        paper_speedup=paper_baseline / paper_cost if paper_cost else 1.0,
        deployed_speedup=(
            deployed_baseline / deployed_cost if deployed_cost else 1.0
        ),
        rows=rows_total,
        fast_rows=fast_total,
    )


def perplexity(token_logprobs) -> float:
    """Perplexity from per-token natural-log probabilities."""
    values = [float(value) for value in token_logprobs if value is not None]
    if not values:
        raise ValueError("no token logprobs supplied")
    return math.exp(-sum(values) / len(values))


def perplexity_ratio(candidate_logprobs, baseline_logprobs) -> float:
    """The paper's `Ratio` column: candidate perplexity over baseline."""
    baseline = perplexity(baseline_logprobs)
    if baseline <= 0:
        raise ValueError("baseline perplexity must be positive")
    return perplexity(candidate_logprobs) / baseline


def top1_agreement(baseline_tokens, candidate_tokens) -> float:
    """Fraction of positions where the argmax token matches.

    Sequences are compared over their common prefix; a length difference is
    itself a disagreement and is counted against the shorter run.
    """
    baseline = list(baseline_tokens)
    candidate = list(candidate_tokens)
    total = max(len(baseline), len(candidate))
    if total == 0:
        raise ValueError("no tokens to compare")
    matches = sum(
        1 for left, right in zip(baseline, candidate) if left == right
    )
    return matches / total


def generation_agreement(baseline_texts, candidate_texts) -> float:
    """Fraction of prompts whose greedy generations are character-identical."""
    baseline = list(baseline_texts)
    candidate = list(candidate_texts)
    if len(baseline) != len(candidate):
        raise ValueError("generation sets differ in size")
    if not baseline:
        raise ValueError("no generations to compare")
    return sum(
        1 for left, right in zip(baseline, candidate) if left == right
    ) / len(baseline)


def score_run(
    *,
    telemetry: dict,
    entries: dict,
    baseline_logprobs=None,
    candidate_logprobs=None,
    baseline_tokens=None,
    candidate_tokens=None,
    baseline_texts=None,
    candidate_texts=None,
    weight_bytes: float = 0.5,
    image_bytes: float = 2.0,
) -> dict:
    """Assemble the paper's result row for one configuration."""
    report = speedup_report(
        telemetry, entries, weight_bytes=weight_bytes, image_bytes=image_bytes
    )
    row: dict = {
        "fast_path_fraction": report.fast_fraction,
        "rows": report.rows,
        "paper_effective_speedup": report.paper_speedup,
        "deployed_speedup": report.deployed_speedup,
        "per_module": report.per_module,
    }
    if candidate_logprobs is not None and baseline_logprobs is not None:
        row["baseline_perplexity"] = perplexity(baseline_logprobs)
        row["candidate_perplexity"] = perplexity(candidate_logprobs)
        row["perplexity_ratio"] = perplexity_ratio(
            candidate_logprobs, baseline_logprobs
        )
    if candidate_tokens is not None and baseline_tokens is not None:
        row["top1_agreement"] = top1_agreement(baseline_tokens, candidate_tokens)
    if candidate_texts is not None and baseline_texts is not None:
        row["generation_agreement"] = generation_agreement(
            baseline_texts, candidate_texts
        )
    return row


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from .profile import GSIProfile

    parser = argparse.ArgumentParser(
        description="Score a GSI run with the paper's metrics"
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--telemetry", type=Path, required=True, help="GSITelemetry snapshot JSON"
    )
    parser.add_argument(
        "--baseline-trace",
        type=Path,
        help="baseline trace from bench/gsi/capture_endpoint.py",
    )
    parser.add_argument("--candidate-trace", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--weight-bytes",
        type=float,
        default=0.5,
        help="bytes per slow-path weight element (0.5 = packed INT4)",
    )
    parser.add_argument("--image-bytes", type=float, default=2.0)
    args = parser.parse_args()

    profile = GSIProfile.load(args.profile, verify_files=False)
    telemetry = json.loads(args.telemetry.read_text(encoding="utf-8"))

    kwargs: dict = {}
    if args.baseline_trace and args.candidate_trace:
        baseline = json.loads(args.baseline_trace.read_text(encoding="utf-8"))
        candidate = json.loads(args.candidate_trace.read_text(encoding="utf-8"))

        def field(payload, name):
            return [item["response"].get(name) for item in payload["results"]]

        kwargs["baseline_texts"] = field(baseline, "content")
        kwargs["candidate_texts"] = field(candidate, "content")
        baseline_logprobs = [
            value
            for item in baseline["results"]
            for value in (item["response"].get("token_logprobs") or [])
        ]
        candidate_logprobs = [
            value
            for item in candidate["results"]
            for value in (item["response"].get("token_logprobs") or [])
        ]
        if baseline_logprobs and candidate_logprobs:
            kwargs["baseline_logprobs"] = baseline_logprobs
            kwargs["candidate_logprobs"] = candidate_logprobs
        baseline_tokens = [
            value
            for item in baseline["results"]
            for value in (item["response"].get("token_ids") or [])
        ]
        candidate_tokens = [
            value
            for item in candidate["results"]
            for value in (item["response"].get("token_ids") or [])
        ]
        if baseline_tokens and candidate_tokens:
            kwargs["baseline_tokens"] = baseline_tokens
            kwargs["candidate_tokens"] = candidate_tokens

    row = score_run(
        telemetry=telemetry,
        entries=profile.by_module(),
        weight_bytes=args.weight_bytes,
        image_bytes=args.image_bytes,
        **kwargs,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    headline = {
        key: value for key, value in row.items() if key != "per_module"
    }
    print(json.dumps(headline, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
