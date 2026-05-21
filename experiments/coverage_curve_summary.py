#!/usr/bin/env python3
"""Summarize coverage-expansion curve artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-results-json", required=True)
    parser.add_argument("--coverage-result", action="append", default=[], help="Path to a coverage_expansion_results.json")
    parser.add_argument("--output-dir", default="artifacts/coverage_curve_v1")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_sci(value: float) -> str:
    return f"{value:.2e}"


def extract_row(setting: str, extra_tasks: int, mode: str, payload: dict) -> dict:
    scores = payload["results"][mode]["scores"]
    metrics = payload["train_metrics"][mode]
    return {
        "setting": setting,
        "extra_tasks": extra_tasks,
        "total_tasks": payload["total_task_count"],
        "model": mode,
        "anchor_tokens": payload["annotation"]["anchor_token_count"],
        "assistant_tokens": payload["annotation"]["gold_token_count"],
        "anchor_ratio": payload["annotation"]["anchor_token_count"] / payload["annotation"]["gold_token_count"],
        "learned_fact_score": scores["learned_fact_semantic_score"],
        "transfer_calculation_score": scores["transfer_calculation_semantic_score"],
        "mean_semantic_score": scores["mean_semantic_score"],
        "retention_kl_vs_base": scores["retention_kl_vs_base"],
        "anchor_ce": metrics["final_anchor_ce"],
        "training_non_anchor_kl": metrics["final_non_anchor_kl"],
        "full_ce": metrics["final_full_ce"],
    }


def make_markdown(rows: list[dict]) -> str:
    lines = [
        "# Coverage Curve Summary",
        "",
        "This table reuses recursive annotation traces and varies how many positive coverage prompts are included.",
        "",
        "| Extra tasks | Total tasks | Model | Anchors | Anchor ratio | Learned fact | Transfer calc | Mean semantic | Retention KL | Anchor CE | Non-anchor KL |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['extra_tasks']} | {row['total_tasks']} | {row['model'].upper()} | "
            f"{row['anchor_tokens']} | {row['anchor_ratio']:.3%} | "
            f"{row['learned_fact_score']:.3f} | {row['transfer_calculation_score']:.3f} | "
            f"{row['mean_semantic_score']:.3f} | {format_sci(row['retention_kl_vs_base'])} | "
            f"{format_sci(row['anchor_ce'])} | {format_sci(row['training_non_anchor_kl'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    baseline = load_json(Path(args.baseline_results_json))
    for mode in ["sft", "lawf"]:
        pseudo_payload = {
            "total_task_count": len(baseline["annotation"].get("tasks", [])),
            "annotation": baseline["annotation"],
            "results": baseline["results"],
            "train_metrics": baseline["train_metrics"],
        }
        rows.append(extract_row("baseline", 0, mode, pseudo_payload))

    for result_path in args.coverage_result:
        payload = load_json(Path(result_path))
        extra_tasks = payload["extra_task_count"]
        for mode in ["sft", "lawf"]:
            rows.append(extract_row(f"coverage+{extra_tasks}", extra_tasks, mode, payload))

    rows.sort(key=lambda row: (row["extra_tasks"], row["model"]))
    (out_dir / "coverage_curve_summary.json").write_text(
        json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "coverage_curve_summary.md").write_text(make_markdown(rows), encoding="utf-8")
    print(out_dir / "coverage_curve_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
