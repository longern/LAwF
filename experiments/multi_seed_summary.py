#!/usr/bin/env python3
"""Summarize repeated-seed LAwF runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


METRICS = [
    ("anchor_ce", "final_anchor_ce"),
    ("training_non_anchor_kl", "final_non_anchor_kl"),
    ("full_ce", "final_full_ce"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="Path to lawf_anchor_experiment_results.json")
    parser.add_argument("--output-dir", default="artifacts/multi_seed_summary_v1")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def format_mean_std(row: dict, digits: int = 6) -> str:
    return f"{row['mean']:.{digits}f} ± {row['std']:.{digits}f}"


def format_sci_mean_std(row: dict) -> str:
    return f"{row['mean']:.2e} ± {row['std']:.2e}"


def summarize(runs: list[dict]) -> dict:
    rows = {}
    for mode in ["sft", "lawf"]:
        per_seed = []
        for payload in runs:
            metrics = payload["train_metrics"][mode]
            scores = payload["results"][mode]["scores"]
            row = {
                "seed": payload["seed"],
                "anchor_ce": metrics["final_anchor_ce"],
                "training_non_anchor_kl": metrics["final_non_anchor_kl"],
                "full_ce": metrics["final_full_ce"],
                "retention_kl_vs_base": scores["retention_kl_vs_base"],
                "learned_fact_score": scores["learned_fact_semantic_score"],
                "transfer_calculation_score": scores["transfer_calculation_semantic_score"],
                "mean_semantic_score": scores["mean_semantic_score"],
            }
            per_seed.append(row)
        rows[mode] = {
            "per_seed": sorted(per_seed, key=lambda row: row["seed"]),
            "summary": {
                key: mean_std([row[key] for row in per_seed])
                for key in [
                    "anchor_ce",
                    "training_non_anchor_kl",
                    "full_ce",
                    "retention_kl_vs_base",
                    "learned_fact_score",
                    "transfer_calculation_score",
                    "mean_semantic_score",
                ]
            },
        }
    return rows


def make_markdown(payload: dict) -> str:
    lines = [
        "# Multi-Seed Summary",
        "",
        f"- Seeds: {', '.join(str(seed) for seed in payload['seeds'])}",
        "",
        "## Mean ± Std",
        "",
        "| Model | Anchor CE | Training non-anchor KL | Retention KL vs base | Learned fact | Transfer calc |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ["sft", "lawf"]:
        row = payload["summary"][mode]["summary"]
        lines.append(
            f"| {mode.upper()} | {format_sci_mean_std(row['anchor_ce'])} | "
            f"{format_sci_mean_std(row['training_non_anchor_kl'])} | "
            f"{format_sci_mean_std(row['retention_kl_vs_base'])} | "
            f"{format_mean_std(row['learned_fact_score'], 3)} | "
            f"{format_mean_std(row['transfer_calculation_score'], 3)} |"
        )

    lines.extend(
        [
            "",
            "## Per-Seed Values",
            "",
            "| Seed | Model | Anchor CE | Training non-anchor KL | Retention KL vs base | Learned fact | Transfer calc |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode in ["sft", "lawf"]:
        for row in payload["summary"][mode]["per_seed"]:
            lines.append(
                f"| {row['seed']} | {mode.upper()} | {row['anchor_ce']:.2e} | "
                f"{row['training_non_anchor_kl']:.2e} | {row['retention_kl_vs_base']:.2e} | "
                f"{row['learned_fact_score']:.3f} | {row['transfer_calculation_score']:.3f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [load_json(Path(path)) for path in args.run]
    payload = {
        "sources": args.run,
        "seeds": sorted(payload["seed"] for payload in runs),
        "summary": summarize(runs),
    }
    (out_dir / "multi_seed_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "multi_seed_summary.md").write_text(make_markdown(payload), encoding="utf-8")
    print(out_dir / "multi_seed_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
