#!/usr/bin/env python3
"""Generate low-cost paper diagnostics from existing LAwF artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORMAL_DIR = ROOT / "artifacts" / "qwen35_9b_formal_training_v1"
ABLATION_DIR = ROOT / "artifacts" / "qwen35_9b_formal_ablation_v2"
GROUPED_ABLATION_DIR = ROOT / "artifacts" / "qwen35_9b_formal_ablation_grouped_v1"
CROSS32_DIR = ROOT / "artifacts" / "cross_domain_transfer_v1"
CROSS128_DIR = ROOT / "artifacts" / "cross_domain_transfer_v1_steps128"
CROSS_ABLATION_DIR = ROOT / "artifacts" / "cross_domain_transfer_ablation_v1"
OUT_DIR = ROOT / "artifacts" / "cheap_paper_diagnostics"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def summarize_annotation(training: dict) -> dict:
    tasks = []
    for task in training["annotation"]["tasks"]:
        assistant_tokens = int(task["gold_token_count"])
        anchor_tokens = len(task["anchor_token_indices"])
        tasks.append(
            {
                "task": task["task_name"],
                "assistant_tokens": assistant_tokens,
                "anchor_tokens": anchor_tokens,
                "anchor_ratio": anchor_tokens / assistant_tokens,
                "non_anchor_tokens": assistant_tokens - anchor_tokens,
                "annotation_rounds": len(task.get("rounds", [])),
            }
        )
    total_assistant = sum(item["assistant_tokens"] for item in tasks)
    total_anchor = sum(item["anchor_tokens"] for item in tasks)
    return {
        "tasks": tasks,
        "total_assistant_tokens": total_assistant,
        "total_anchor_tokens": total_anchor,
        "anchor_ratio": total_anchor / total_assistant,
        "dilution_factor": total_assistant / total_anchor,
    }


def summarize_training(training: dict) -> dict:
    rows = {}
    for mode in ["sft", "lawf"]:
        metrics = training["train_metrics"][mode]
        rows[mode] = {
            "anchor_ce": metrics["final_anchor_ce"],
            "training_non_anchor_kl": metrics["final_non_anchor_kl"],
            "full_ce": metrics["final_full_ce"],
            "retention_kl_vs_base": training["results"][mode]["scores"]["retention_kl_vs_base"],
            "final_loss": metrics["final_loss"],
        }
    return rows


def summarize_loss_ablation(training: dict) -> dict:
    rows = summarize_training(training)
    if ABLATION_DIR.exists():
        ablation = load_json(ABLATION_DIR / "lawf_anchor_experiment_results.json")
        for mode in ["anchor_only", "sft_kl"]:
            metrics = ablation["train_metrics"][mode]
            rows[mode] = {
                "anchor_ce": metrics["final_anchor_ce"],
                "training_non_anchor_kl": metrics["final_non_anchor_kl"],
                "full_ce": metrics["final_full_ce"],
                "retention_kl_vs_base": ablation["results"][mode]["scores"]["retention_kl_vs_base"],
                "final_loss": metrics["final_loss"],
                "mean_semantic_score": ablation["results"][mode]["scores"]["mean_semantic_score"],
            }
    if GROUPED_ABLATION_DIR.exists():
        grouped = load_json(GROUPED_ABLATION_DIR / "lawf_anchor_experiment_results.json")
        mode = "sft_kl_grouped"
        metrics = grouped["train_metrics"][mode]
        rows[mode] = {
            "anchor_ce": metrics["final_anchor_ce"],
            "training_non_anchor_kl": metrics["final_non_anchor_kl"],
            "full_ce": metrics["final_full_ce"],
            "retention_kl_vs_base": grouped["results"][mode]["scores"]["retention_kl_vs_base"],
            "final_loss": metrics["final_loss"],
            "mean_semantic_score": grouped["results"][mode]["scores"]["mean_semantic_score"],
        }
    for mode in ["sft", "lawf"]:
        rows[mode]["mean_semantic_score"] = training["results"][mode]["scores"]["mean_semantic_score"]
    return rows


def summarize_base_teacher() -> dict:
    payload = load_json(FORMAL_DIR / "retention_base_teacher_eval.json")
    rows = {}
    for mode in ["base", "sft", "lawf"]:
        summary = payload["summary"][mode]
        rows[mode] = {
            "mean_ce": summary["mean_ce"],
            "mean_delta_ce_vs_base": summary.get("mean_delta_ce_vs_base", 0.0),
            "delta_ce_gt_0p1": summary.get("delta_ce_gt_0p1"),
            "delta_ce_gt_0p25": summary.get("delta_ce_gt_0p25"),
            "near_material_delta": summary.get("mean_delta_ce_by_category", {}).get("near_material", 0.0),
        }
    return rows


def summarize_step_stress() -> list[dict]:
    rows = []
    for steps, folder in [(32, CROSS32_DIR), (128, CROSS128_DIR)]:
        payload = load_json(folder / "general_kl_drift_eval.json")
        for mode in ["sft", "lawf"]:
            summary = payload["results"][mode]["summary"]
            rows.append(
                {
                    "steps": steps,
                    "model": mode,
                    "mean_kl": summary["mean_kl_base_to_model"],
                    "kl_gt_0p1": summary["kl_gt_0p1"],
                    "kl_gt_0p25": summary["kl_gt_0p25"],
                    "kl_gt_0p5": summary["kl_gt_0p5"],
                    "near_identity_kl": summary["by_category"]["near_identity"]["mean_kl_base_to_model"],
                    "near_game_kl": summary["by_category"]["near_game"]["mean_kl_base_to_model"],
                }
            )
    return rows


def summarize_cross_domain_ablation() -> dict:
    primary = load_json(CROSS32_DIR / "cross_domain_transfer_results.json")
    ablation = load_json(CROSS_ABLATION_DIR / "cross_domain_transfer_results.json")
    rows = {}
    for source, modes in [(primary, ["sft", "lawf"]), (ablation, ["anchor_only", "sft_kl"])]:
        for mode in modes:
            metrics = source["train_metrics"][mode]
            transfer = source["transfer_eval"]["summary"][mode]
            rows[mode] = {
                "anchor_ce": metrics["final_anchor_ce"],
                "training_non_anchor_kl": metrics["final_non_anchor_kl"],
                "full_ce": metrics["final_full_ce"],
                "mean_score": transfer["mean_score"],
                "mean_transfer_score": transfer["mean_transfer_score"],
                "transfer_rate_at_0p7": transfer["transfer_rate_at_0p7"],
                "final_loss": metrics["final_loss"],
            }
    rows["base"] = {
        "mean_score": primary["transfer_eval"]["summary"]["base"]["mean_score"],
        "mean_transfer_score": primary["transfer_eval"]["summary"]["base"]["mean_transfer_score"],
        "transfer_rate_at_0p7": primary["transfer_eval"]["summary"]["base"]["transfer_rate_at_0p7"],
    }
    return rows


def summarize_transfer_boundary(training: dict) -> dict:
    boundary = load_json(FORMAL_DIR / "near_domain_contamination_eval.json")["summary"]
    transfer = {}
    for mode in ["base", "sft", "lawf"]:
        transfer[mode] = training["results"][mode]["scores"]
    return {"transfer": transfer, "boundary": boundary}


def make_markdown(summary: dict) -> str:
    annotation = summary["annotation"]
    training = summary["training"]
    base_teacher = summary["base_teacher"]
    step_stress = summary["step_stress"]
    transfer_boundary = summary["transfer_boundary"]
    cross_domain_ablation = summary["cross_domain_ablation"]

    lines = [
        "# Cheap Paper Diagnostics",
        "",
        "Generated from existing artifacts plus low-cost ablation runs in `qwen35_9b_formal_ablation_v2` and `cross_domain_transfer_ablation_v1`.",
        "",
        "## Sparse Annotation and Normalization Counterfactual",
        "",
        f"- Directly supervised anchors: `{annotation['total_anchor_tokens']}` / `{annotation['total_assistant_tokens']}` assistant tokens.",
        f"- Anchor ratio: `{annotation['anchor_ratio'] * 100:.2f}%`.",
        f"- If anchor CE were averaged uniformly over all assistant tokens, its aggregate weight would be diluted by `{annotation['dilution_factor']:.1f}x`.",
        "",
        "| Task | Assistant tokens | Anchor tokens | Anchor ratio | Annotation rounds |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in annotation["tasks"]:
        lines.append(
            f"| {item['task']} | {item['assistant_tokens']} | {item['anchor_tokens']} | "
            f"{item['anchor_ratio'] * 100:.2f}% | {item['annotation_rounds']} |"
        )

    lines.extend(
        [
            "",
            "## Loss Component Diagnostics",
            "",
            "| Model | Anchor CE | Training non-anchor KL | Full CE | Retention KL vs base | Mean semantic score | Final loss |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode in ["sft", "anchor_only", "sft_kl", "sft_kl_grouped", "lawf"]:
        row = summary["loss_ablation"][mode]
        lines.append(
            f"| {mode.upper()} | {fmt_float(row['anchor_ce'])} | {fmt_float(row['training_non_anchor_kl'])} | "
            f"{fmt_float(row['full_ce'])} | {fmt_float(row['retention_kl_vs_base'])} | "
            f"{row['mean_semantic_score']:.3f} | {fmt_float(row['final_loss'])} |"
        )

    lines.extend(
        [
            "",
            "## Base-Teacher Retention",
            "",
            "| Model | Mean CE | Mean delta CE vs base | Delta CE > 0.1 | Delta CE > 0.25 | Nearby material delta CE |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode in ["base", "sft", "lawf"]:
        row = base_teacher[mode]
        gt01 = "-" if row["delta_ce_gt_0p1"] is None else str(row["delta_ce_gt_0p1"])
        gt025 = "-" if row["delta_ce_gt_0p25"] is None else str(row["delta_ce_gt_0p25"])
        lines.append(
            f"| {mode.upper()} | {fmt_float(row['mean_ce'], 4)} | {fmt_float(row['mean_delta_ce_vs_base'], 4)} | "
            f"{gt01} | {gt025} | {fmt_float(row['near_material_delta'], 4)} |"
        )

    lines.extend(
        [
            "",
            "## Longer-Optimization Drift Stress Test",
            "",
            "| Steps | Model | Mean held-out KL | KL > 0.1 | KL > 0.25 | KL > 0.5 | Near identity KL | Near game KL |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in step_stress:
        lines.append(
            f"| {row['steps']} | {row['model'].upper()} | {fmt_float(row['mean_kl'])} | "
            f"{row['kl_gt_0p1']} / 28 | {row['kl_gt_0p25']} / 28 | {row['kl_gt_0p5']} / 28 | "
            f"{fmt_float(row['near_identity_kl'])} | {fmt_float(row['near_game_kl'])} |"
        )

    lines.extend(
        [
            "",
            "## Cross-Domain Objective Ablation",
            "",
            "| Model | Anchor CE | Training non-anchor KL | Full CE | Mean judge score | Mean transfer score | Transfer rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode in ["sft", "anchor_only", "sft_kl", "lawf"]:
        row = cross_domain_ablation[mode]
        lines.append(
            f"| {mode.upper()} | {fmt_float(row['anchor_ce'])} | {fmt_float(row['training_non_anchor_kl'])} | "
            f"{fmt_float(row['full_ce'])} | {row['mean_score']:.3f} | {row['mean_transfer_score']:.3f} | "
            f"{row['transfer_rate_at_0p7']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Transfer and Boundary Diagnostics",
            "",
            "| Model | Learned fact score | Transfer calculation score | Mean semantic score | Boundary contamination rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode in ["base", "sft", "lawf"]:
        transfer = transfer_boundary["transfer"][mode]
        contamination = transfer_boundary["boundary"][mode]["contamination_rate"]
        lines.append(
            f"| {mode.upper()} | {transfer['learned_fact_semantic_score']:.3f} | "
            f"{transfer['transfer_calculation_semantic_score']:.3f} | {transfer['mean_semantic_score']:.3f} | "
            f"{contamination:.3f} |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    training = load_json(FORMAL_DIR / "lawf_anchor_experiment_results.json")
    summary = {
        "sources": {
            "training": str(FORMAL_DIR / "lawf_anchor_experiment_results.json"),
            "base_teacher": str(FORMAL_DIR / "retention_base_teacher_eval.json"),
            "boundary": str(FORMAL_DIR / "near_domain_contamination_eval.json"),
            "step_32": str(CROSS32_DIR / "general_kl_drift_eval.json"),
            "step_128": str(CROSS128_DIR / "general_kl_drift_eval.json"),
            "cross_domain_ablation": str(CROSS_ABLATION_DIR / "cross_domain_transfer_results.json"),
        },
        "annotation": summarize_annotation(training),
        "training": summarize_training(training),
        "loss_ablation": summarize_loss_ablation(training),
        "base_teacher": summarize_base_teacher(),
        "step_stress": summarize_step_stress(),
        "cross_domain_ablation": summarize_cross_domain_ablation(),
        "transfer_boundary": summarize_transfer_boundary(training),
    }
    (OUT_DIR / "cheap_paper_diagnostics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_DIR / "cheap_paper_diagnostics.md").write_text(make_markdown(summary), encoding="utf-8")
    print(OUT_DIR / "cheap_paper_diagnostics.md")


if __name__ == "__main__":
    main()
