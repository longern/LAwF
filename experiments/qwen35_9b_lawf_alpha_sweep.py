#!/usr/bin/env python3
"""Qwen3.5-9B LAwF alpha/beta/step sweep.

This diagnostic checks whether the main-trace LAwF result is limited by an
overly conservative anchor-retention weighting. It can run on the full
multi-query trace or the long-form-only subset to separate objective behavior
from short-format answer imitation.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys

import torch
from torch.optim import AdamW
from modelscope import snapshot_download
from transformers import AutoTokenizer, set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lawf_anchor_experiment import (  # noqa: E402
    EVAL_PROMPTS,
    anchor_target_kl,
    apply_chat_template,
    average_reference_kl,
    build_training_tensors,
    ce_on_mask,
    generate,
    kl_ref_to_model,
    load_base_model,
    make_lora_model,
)
from qwen35_9b_pareto_sweep import ACQUISITION_PROBES, evaluate_acquisition  # noqa: E402


LONG3_TASKS = {
    "proposer_fact_card",
    "proposer_biographical_note",
    "proposer_relation_index",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--work-dir", default="/root/lawf_experiment/artifacts/qwen35_9b_lawf_alpha_sweep_v1")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument(
        "--annotation-json",
        default="/root/lawf_experiment/artifacts/qwen35_9b_entity_relation_multiquery_annotation_v3/annotation_trace.json",
    )
    parser.add_argument("--task-filter", choices=["all", "long3"], default="all")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--anchor-confidence", type=float, default=0.999)
    parser.add_argument("--lawf-alphas", default="1,2,4")
    parser.add_argument("--lawf-betas", default="0.5,1,2")
    parser.add_argument("--lawf-steps", default="32")
    parser.add_argument("--sft-kl-weights", default="")
    parser.add_argument(
        "--lawf-normalization",
        choices=["group_mean", "token_mean"],
        default="group_mean",
        help=(
            "How to combine LAwF anchor and non-anchor terms. group_mean keeps the original "
            "mean(anchor KL) + beta * mean(non-anchor KL) objective; token_mean weights the "
            "two sums by token counts before dividing by assistant tokens."
        ),
    )
    return parser.parse_args()


def parse_number_list(raw: str, cast=float) -> list:
    values = [cast(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"expected at least one value, got {raw!r}")
    return values


def parse_optional_number_list(raw: str, cast=float) -> list:
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


def safe_label(value: float | int) -> str:
    return f"{value:g}".replace(".", "p")


def build_reference_continuations(ref_model, tokenizer, max_new_tokens: int) -> dict[str, str]:
    return {
        name: generate(ref_model, tokenizer, prompt, max_new_tokens)
        for name, prompt in EVAL_PROMPTS.items()
        if name.startswith("unrelated_")
    }


def selected_tasks(annotation: dict, task_filter: str) -> list[dict]:
    tasks = annotation.get("tasks") or [annotation]
    if task_filter == "all":
        return tasks
    if task_filter == "long3":
        return [task for task in tasks if task.get("task_name") in LONG3_TASKS]
    raise ValueError(task_filter)


def build_batches(tokenizer, tasks: list[dict]) -> list[dict[str, torch.Tensor]]:
    return [
        build_training_tensors(tokenizer, task["prompt"], task["completion_ids"], task["anchor_token_indices"])
        for task in tasks
    ]


def prepare_batches(model, ref_model, batches: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    prepared_batches = []
    for batch in batches:
        input_ids = batch["input_ids"].to(model.device)
        labels = batch["labels"].to(model.device)
        train_mask = batch["train_mask"].to(model.device)
        anchor_mask = batch["anchor_mask"].to(model.device)
        non_anchor_mask = train_mask & ~anchor_mask
        with torch.no_grad():
            ref_logits = ref_model(input_ids.to(ref_model.device)).logits[:, :-1, :].detach().to(model.device)
        prepared_batches.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "train_mask": train_mask,
                "anchor_mask": anchor_mask,
                "non_anchor_mask": non_anchor_mask,
                "ref_logits": ref_logits,
            }
        )
    return prepared_batches


def objective_terms(model, prepared: dict[str, torch.Tensor], anchor_confidence: float) -> dict[str, torch.Tensor]:
    logits = model(prepared["input_ids"]).logits[:, :-1, :]
    return {
        "anchor_ce": ce_on_mask(logits, prepared["labels"], prepared["anchor_mask"]),
        "anchor_kl": anchor_target_kl(
            logits,
            prepared["ref_logits"],
            prepared["labels"],
            prepared["anchor_mask"],
            anchor_confidence,
        ),
        "non_anchor_kl": kl_ref_to_model(logits, prepared["ref_logits"], prepared["non_anchor_mask"]),
        "full_ce": ce_on_mask(logits, prepared["labels"], prepared["train_mask"]),
    }


def mean_train_metrics(model, prepared_batches: list[dict[str, torch.Tensor]], anchor_confidence: float) -> dict[str, float]:
    totals = {"anchor_ce": 0.0, "anchor_kl": 0.0, "non_anchor_kl": 0.0, "full_ce": 0.0}
    with torch.no_grad():
        for prepared in prepared_batches:
            terms = objective_terms(model, prepared, anchor_confidence)
            for key in totals:
                totals[key] += float(terms[key].detach().cpu())
    return {key: value / len(prepared_batches) for key, value in totals.items()}


def train_config(
    config: dict,
    model_path: str,
    ref_model,
    batches: list[dict[str, torch.Tensor]],
    lr: float,
    output_dir: Path,
    lora_r: int,
    lora_alpha: int,
    anchor_confidence: float,
) -> dict:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = make_lora_model(model_path, lora_r, lora_alpha)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    prepared_batches = prepare_batches(model, ref_model, batches)

    final_loss = math.nan
    for _ in range(config["steps"]):
        optimizer.zero_grad(set_to_none=True)
        loss = None
        for prepared in prepared_batches:
            terms = objective_terms(model, prepared, anchor_confidence)
            if config["family"] == "lawf":
                if config.get("normalization") == "token_mean":
                    anchor_count = prepared["anchor_mask"].sum().clamp_min(1)
                    non_anchor_count = prepared["non_anchor_mask"].sum().clamp_min(1)
                    train_count = prepared["train_mask"].sum().clamp_min(1)
                    batch_loss = (
                        config["alpha"] * terms["anchor_kl"] * anchor_count
                        + config["beta"] * terms["non_anchor_kl"] * non_anchor_count
                    ) / train_count
                else:
                    batch_loss = config["alpha"] * terms["anchor_kl"] + config["beta"] * terms["non_anchor_kl"]
            elif config["family"] == "sft_kl":
                batch_loss = terms["full_ce"] + config["kl_weight"] * terms["non_anchor_kl"]
            else:
                raise ValueError(config["family"])
            scale = 1.0 / len(prepared_batches)
            loss = batch_loss * scale if loss is None else loss + batch_loss * scale
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    model.eval()
    model.save_pretrained(output_dir)
    metrics = mean_train_metrics(model, prepared_batches, anchor_confidence)
    result = {
        "final_loss": final_loss,
        "final_anchor_ce": metrics["anchor_ce"],
        "final_anchor_kl": metrics["anchor_kl"],
        "final_non_anchor_kl": metrics["non_anchor_kl"],
        "final_full_ce": metrics["full_ce"],
        "steps": config["steps"],
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "anchor_tokens": int(sum(prepared["anchor_mask"].sum().item() for prepared in prepared_batches)),
        "assistant_tokens": int(sum(prepared["train_mask"].sum().item() for prepared in prepared_batches)),
    }
    if torch.cuda.is_available():
        result["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
    return {"model": model, "metrics": result}


def make_configs(args: argparse.Namespace) -> list[dict]:
    configs = []
    for weight in parse_optional_number_list(args.sft_kl_weights, float):
        configs.append(
            {
                "label": f"sft_kl_w_{safe_label(weight)}",
                "family": "sft_kl",
                "steps": max(parse_number_list(args.lawf_steps, int)),
                "kl_weight": weight,
            }
        )
    for steps in parse_number_list(args.lawf_steps, int):
        for alpha in parse_number_list(args.lawf_alphas, float):
            for beta in parse_number_list(args.lawf_betas, float):
                configs.append(
                    {
                        "label": f"lawf_a_{safe_label(alpha)}_b_{safe_label(beta)}_s_{steps}",
                        "family": "lawf",
                        "steps": steps,
                        "alpha": alpha,
                        "beta": beta,
                        "normalization": args.lawf_normalization,
                    }
                )
    return configs


def pareto_frontier(rows: list[dict]) -> list[dict]:
    frontier = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            no_worse = (
                other["acquisition_ce"] <= row["acquisition_ce"]
                and other["retention_kl_vs_base"] <= row["retention_kl_vs_base"]
            )
            strictly_better = (
                other["acquisition_ce"] < row["acquisition_ce"]
                or other["retention_kl_vs_base"] < row["retention_kl_vs_base"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda item: item["retention_kl_vs_base"])


def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# Qwen3.5-9B LAwF Alpha Sweep",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Task filter: `{payload['task_filter']}`",
        f"- LAwF normalization: `{payload.get('lawf_normalization', 'group_mean')}`",
        f"- Annotated tasks: `{payload['annotation_summary']['task_count']}`",
        f"- Anchor tokens: `{payload['annotation_summary']['anchor_tokens']}` / `{payload['annotation_summary']['assistant_tokens']}`",
        "",
        "## Pareto Frontier",
        "",
        "| Config | Family | Steps | Alpha | Beta / KL weight | Acquisition CE | Direct CE | KB CE | Reverse CE | Retention KL | Anchor CE | Train non-anchor KL |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["pareto_frontier"]:
        lines.append(format_row(row))
    lines.extend(
        [
            "",
            "## All Sweep Points",
            "",
            "| Config | Family | Steps | Alpha | Beta / KL weight | Acquisition CE | Direct CE | KB CE | Reverse CE | Retention KL | Anchor CE | Full CE | Train non-anchor KL |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(payload["summary_rows"], key=lambda item: (item["family"], item["retention_kl_vs_base"], item["acquisition_ce"])):
        lines.append(format_row(row, include_full_ce=True))
    path.write_text("\n".join(lines), encoding="utf-8")


def format_row(row: dict, include_full_ce: bool = False) -> str:
    alpha = row.get("alpha")
    beta_or_kl = row.get("beta", row.get("kl_weight"))
    alpha_text = "-" if alpha is None else f"{alpha:g}"
    beta_text = "-" if beta_or_kl is None else f"{beta_or_kl:g}"
    cells = [
        row["label"],
        row["family"],
        str(row["steps"]),
        alpha_text,
        beta_text,
        f"{row['acquisition_ce']:.6f}",
        f"{row['direct_fact_ce']:.6f}",
        f"{row['kb_record_ce']:.6f}",
        f"{row['reverse_lookup_ce']:.6f}",
        f"{row['retention_kl_vs_base']:.6f}",
        f"{row['final_anchor_ce']:.6f}",
    ]
    if include_full_ce:
        cells.append(f"{row['final_full_ce']:.6f}")
    cells.append(f"{row['final_non_anchor_kl']:.6f}")
    return "| " + " | ".join(cells) + " |"


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    model_path = snapshot_download(args.model_id, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ref_model = load_base_model(model_path, trainable=False)

    annotation = json.loads(Path(args.annotation_json).read_text(encoding="utf-8"))
    tasks = selected_tasks(annotation, args.task_filter)
    batches = build_batches(tokenizer, tasks)
    reference_continuations = build_reference_continuations(ref_model, tokenizer, args.max_new_tokens)
    configs = make_configs(args)

    payload = {
        "model_id": args.model_id,
        "resolved_model_path": model_path,
        "annotation_json": args.annotation_json,
        "task_filter": args.task_filter,
        "seed": args.seed,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "anchor_confidence": args.anchor_confidence,
        "lawf_normalization": args.lawf_normalization,
        "annotation_summary": {
            "task_count": len(tasks),
            "task_names": [task.get("task_name") for task in tasks],
            "assistant_tokens": sum(len(task["completion_ids"]) for task in tasks),
            "anchor_tokens": sum(len(task["anchor_token_indices"]) for task in tasks),
        },
        "acquisition_probes": ACQUISITION_PROBES,
        "configs": configs,
        "train_metrics": {},
        "eval": {},
        "summary_rows": [],
        "pareto_frontier": [],
    }
    payload["annotation_summary"]["anchor_ratio"] = (
        payload["annotation_summary"]["anchor_tokens"] / max(payload["annotation_summary"]["assistant_tokens"], 1)
    )

    for config in configs:
        trained = train_config(
            config,
            model_path,
            ref_model,
            batches,
            args.lr,
            work_dir / f"{config['label']}_adapter",
            args.lora_r,
            args.lora_alpha,
            args.anchor_confidence,
        )
        acquisition = evaluate_acquisition(trained["model"], tokenizer)
        retention_kl = average_reference_kl(trained["model"], ref_model, tokenizer, reference_continuations)
        metrics = trained["metrics"]
        row = {
            "label": config["label"],
            "family": config["family"],
            "steps": config["steps"],
            "alpha": config.get("alpha"),
            "beta": config.get("beta"),
            "kl_weight": config.get("kl_weight"),
            "acquisition_ce": acquisition["acquisition_ce"],
            "direct_fact_ce": acquisition["direct_fact_ce"],
            "kb_record_ce": acquisition["kb_record_ce"],
            "reverse_lookup_ce": acquisition["reverse_lookup_ce"],
            "retention_kl_vs_base": retention_kl,
            "final_anchor_ce": metrics["final_anchor_ce"],
            "final_full_ce": metrics["final_full_ce"],
            "final_non_anchor_kl": metrics["final_non_anchor_kl"],
        }
        payload["train_metrics"][config["label"]] = metrics
        payload["eval"][config["label"]] = {**acquisition, "retention_kl_vs_base": retention_kl}
        payload["summary_rows"].append(row)
        del trained
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload["pareto_frontier"] = pareto_frontier(payload["summary_rows"])
    json_path = work_dir / "qwen35_9b_lawf_alpha_sweep_results.json"
    md_path = work_dir / "qwen35_9b_lawf_alpha_sweep_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(md_path, payload)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
