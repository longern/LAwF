#!/usr/bin/env python3
"""Summarize recursive LAwF annotation traces for paper audit tables."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-json", required=True)
    parser.add_argument("--output-dir", default="artifacts/annotation_audit_v1")
    parser.add_argument("--sample-limit", type=int, default=16)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def corrected_rounds(annotation: dict) -> list[dict]:
    rows = []
    for task in annotation.get("tasks", [annotation]):
        for item in task.get("rounds", []):
            if item.get("status") == "corrected":
                rows.append({**item, "task_name": task.get("task_name", item.get("task_name", ""))})
    return rows


def task_rows(annotation: dict) -> list[dict]:
    rows = []
    for task in annotation.get("tasks", [annotation]):
        gold_count = task.get("gold_token_count", len(task.get("completion_ids", [])))
        anchor_count = task.get("anchor_token_count", len(task.get("anchor_token_indices", [])))
        rows.append(
            {
                "task_name": task.get("task_name", ""),
                "assistant_tokens": gold_count,
                "anchor_tokens": anchor_count,
                "anchor_ratio": anchor_count / gold_count if gold_count else 0.0,
                "rounds": len(task.get("rounds", [])),
                "corrected_rounds": sum(1 for row in task.get("rounds", []) if row.get("status") == "corrected"),
            }
        )
    return rows


def make_markdown(summary: dict) -> str:
    lines = [
        "# Annotation Audit Summary",
        "",
        "## Task-Level Annotation Load",
        "",
        "| Task | Assistant tokens | Anchor tokens | Anchor ratio | Rounds | Corrected rounds |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["tasks"]:
        lines.append(
            f"| {row['task_name']} | {row['assistant_tokens']} | {row['anchor_tokens']} | "
            f"{row['anchor_ratio']:.3%} | {row['rounds']} | {row['corrected_rounds']} |"
        )

    lines.extend(
        [
            "",
            "## Anchor Categories",
            "",
            "| Category | Count |",
            "| --- | ---: |",
        ]
    )
    for category, count in summary["category_counts"].items():
        lines.append(f"| {category} | {count} |")

    lines.extend(
        [
            "",
            "## Sampled Corrections",
            "",
            "| Task | Category | Observed token | Replacement | Matched atom | Reason |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["samples"]:
        reason = row["reason"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {row['task_name']} | {row['error_category']} | `{row['observed_token_text']}` | "
            f"`{row['replacement_text']}` | `{row['matched_atom_id']}` | {reason} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    annotation = load_json(Path(args.annotation_json))
    corrections = corrected_rounds(annotation)
    category_counts = Counter(str(row.get("error_category") or "unspecified") for row in corrections)
    samples = []
    for row in corrections[: args.sample_limit]:
        samples.append(
            {
                "task_name": row.get("task_name", ""),
                "error_category": str(row.get("error_category") or ""),
                "observed_token_text": str(row.get("observed_token_text") or row.get("effective_observed_token_text") or ""),
                "replacement_text": str(row.get("replacement_text") or ""),
                "matched_atom_id": str(row.get("matched_atom_id") or ""),
                "reason": str(row.get("reason") or ""),
            }
        )
    summary = {
        "source": args.annotation_json,
        "tasks": task_rows(annotation),
        "category_counts": dict(sorted(category_counts.items())),
        "correction_count": len(corrections),
        "samples": samples,
    }
    (out_dir / "annotation_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "annotation_audit_summary.md").write_text(make_markdown(summary), encoding="utf-8")
    print(out_dir / "annotation_audit_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
