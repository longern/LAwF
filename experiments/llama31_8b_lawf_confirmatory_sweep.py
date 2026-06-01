#!/usr/bin/env python3
"""Llama-3.1-8B-Instruct confirmatory LAwF sweep.

The Qwen annotation trace stores completion token ids in the Qwen tokenizer.
For a model-family replication on Llama, this script rebuilds completions from
the gold text and remaps anchors by semantic fact spans before training.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys

import torch
from modelscope import snapshot_download
from torch.optim import AdamW
from transformers import AutoTokenizer, set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lawf_anchor_experiment import (  # noqa: E402
    EVAL_PROMPTS,
    anchor_target_kl,
    apply_chat_template,
    average_reference_kl,
    ce_on_mask,
    generate,
    kl_ref_to_model,
    load_base_model,
    make_lora_model,
    reference_next_token_stats,
)
from qwen35_9b_pareto_sweep import ACQUISITION_PROBES, evaluate_acquisition  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="LLM-Research/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--work-dir", default="/root/lawf_experiment/artifacts/llama31_8b_lawf_confirmatory_sweep_v1")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument(
        "--annotation-json",
        default="/root/lawf_experiment/artifacts/qwen35_9b_entity_relation_multiquery_annotation_v3/annotation_trace.json",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--anchor-confidence", type=float, default=0.999)
    parser.add_argument(
        "--anchor-policy",
        choices=["semantic_span", "probability_floor"],
        default="semantic_span",
        help=(
            "semantic_span anchors every token overlapping the retokenized correction span. "
            "probability_floor keeps only correction-span tokens whose frozen-model probability "
            "falls below --anchor-target-probability."
        ),
    )
    parser.add_argument(
        "--anchor-target-probability",
        type=float,
        default=0.9,
        help="Per-token target probability used by --anchor-policy probability_floor.",
    )
    parser.add_argument(
        "--anchor-probability-tolerance",
        type=float,
        default=0.0,
        help="Do not anchor probability_floor tokens already within this tolerance of the target probability.",
    )
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--sft-kl-weights", default="0.25,1,8")
    parser.add_argument("--lawf-configs", default="4:0.5,4:1,2:2")
    parser.add_argument(
        "--lawf-normalization",
        choices=["group_mean", "token_mean"],
        default="group_mean",
        help=(
            "group_mean keeps the original objective scale: alpha*mean(anchor)+beta*mean(non-anchor). "
            "token_mean weights the two sums by token counts before dividing by assistant tokens."
        ),
    )
    return parser.parse_args()


def parse_number_list(raw: str, cast=float) -> list:
    values = [cast(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError(f"expected at least one value, got {raw!r}")
    return values


def safe_label(value: float | int) -> str:
    return f"{value:g}".replace(".", "p")


def find_all_spans(text: str, needle: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    while needle:
        index = text.find(needle, start)
        if index < 0:
            break
        spans.append((index, index + len(needle)))
        start = index + len(needle)
    return spans


def semantic_anchor_spans(task: dict) -> list[tuple[int, int]]:
    gold = task["gold_completion"]
    spans = []
    seen = set()

    for round_info in task.get("rounds", []):
        if round_info.get("status") != "corrected":
            continue
        replacement = (round_info.get("inserted_replacement_text") or "").strip()
        post_context = round_info.get("post_edit_context") or ""
        if not replacement:
            continue
        context_end = gold.find(post_context)
        if context_end >= 0:
            replacement_start = post_context.rfind(replacement)
            if replacement_start >= 0:
                span = (context_end + replacement_start, context_end + replacement_start + len(replacement))
                if span not in seen:
                    spans.append(span)
                    seen.add(span)
                continue
        for span in find_all_spans(gold, replacement):
            if span not in seen:
                spans.append(span)
                seen.add(span)
                break

    if not spans:
        values = [atom.get("value") for atom in task.get("reference_atoms", []) if atom.get("value")]
        for value in values:
            for span in find_all_spans(gold, value):
                if span not in seen:
                    spans.append(span)
                    seen.add(span)
                break
    return sorted(spans)


def retokenize_task(
    tokenizer,
    task: dict,
    ref_model=None,
    anchor_policy: str = "semantic_span",
    anchor_target_probability: float = 0.9,
    anchor_probability_tolerance: float = 0.0,
) -> dict:
    gold = task["gold_completion"]
    encoded = tokenizer(gold, add_special_tokens=False, return_offsets_mapping=True)
    completion_ids = encoded.input_ids
    offsets = encoded.offset_mapping
    spans = semantic_anchor_spans(task)
    candidate_anchor_token_indices = []
    for token_index, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if any(start < span_end and end > span_start for span_start, span_end in spans):
            candidate_anchor_token_indices.append(token_index)
    if not completion_ids:
        raise ValueError(f"empty completion for task {task.get('task_name')}")
    if not candidate_anchor_token_indices:
        raise ValueError(f"no semantic anchors found for task {task.get('task_name')}")

    probability_records = []
    if anchor_policy == "semantic_span":
        anchor_token_indices = candidate_anchor_token_indices
    elif anchor_policy == "probability_floor":
        if ref_model is None:
            raise ValueError("ref_model is required for anchor_policy=probability_floor")
        anchor_token_indices = []
        for token_index in candidate_anchor_token_indices:
            token_id = completion_ids[token_index]
            stats = reference_next_token_stats(
                ref_model,
                tokenizer,
                task["prompt"],
                completion_ids[:token_index],
                token_id,
            )
            is_anchor = stats["target_probability"] < (
                anchor_target_probability - anchor_probability_tolerance
            )
            if is_anchor:
                anchor_token_indices.append(token_index)
            probability_records.append(
                {
                    "completion_token_index": token_index,
                    "token_text": tokenizer.decode([token_id], skip_special_tokens=True),
                    "is_anchor": is_anchor,
                    "target_probability": anchor_target_probability if is_anchor else None,
                    "reference_probability": stats["target_probability"],
                    "reference_rank": stats["target_rank"],
                    "top1_token_text": stats["top1_token_text"],
                    "top_tokens": stats["top_tokens"],
                }
            )
    else:
        raise ValueError(f"Unsupported anchor policy: {anchor_policy}")

    if not anchor_token_indices:
        raise ValueError(f"anchor policy {anchor_policy!r} produced no anchors for task {task.get('task_name')}")
    return {
        **task,
        "completion_ids": completion_ids,
        "anchor_token_indices": anchor_token_indices,
        "candidate_anchor_token_indices": candidate_anchor_token_indices,
        "probability_records": probability_records,
        "retokenized_anchor_spans": spans,
        "original_token_count": len(task.get("completion_ids", [])),
        "original_anchor_count": len(task.get("anchor_token_indices", [])),
    }


def build_training_tensors_from_task(tokenizer, task: dict) -> dict[str, torch.Tensor]:
    prefix = apply_chat_template(
        tokenizer,
        [{"role": "user", "content": task["prompt"]}],
        add_generation_prompt=True,
    )
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    full_ids = prefix_ids + task["completion_ids"]
    if tokenizer.eos_token_id is not None:
        full_ids.append(tokenizer.eos_token_id)
    input_ids = torch.tensor([full_ids], dtype=torch.long)
    prefix_len = len(prefix_ids)

    labels = input_ids[:, 1:].clone()
    train_mask = torch.zeros_like(labels, dtype=torch.bool)
    train_mask[:, max(prefix_len - 1, 0) :] = True

    anchor_mask = torch.zeros_like(labels, dtype=torch.bool)
    for completion_token_index in task["anchor_token_indices"]:
        pred_pos = prefix_len + completion_token_index - 1
        if 0 <= pred_pos < anchor_mask.shape[1]:
            anchor_mask[:, pred_pos] = True
    return {
        "input_ids": input_ids,
        "labels": labels,
        "train_mask": train_mask,
        "anchor_mask": anchor_mask,
    }


def build_reference_continuations(ref_model, tokenizer, max_new_tokens: int) -> dict[str, str]:
    return {
        name: generate(ref_model, tokenizer, prompt, max_new_tokens)
        for name, prompt in EVAL_PROMPTS.items()
        if name.startswith("unrelated_")
    }


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
            if config["family"] == "sft_kl":
                batch_loss = terms["full_ce"] + config["kl_weight"] * terms["non_anchor_kl"]
            elif config["family"] == "lawf":
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
    for weight in parse_number_list(args.sft_kl_weights, float):
        configs.append(
            {
                "label": f"sft_kl_w_{safe_label(weight)}",
                "family": "sft_kl",
                "steps": args.steps,
                "kl_weight": weight,
            }
        )
    for raw in [item.strip() for item in args.lawf_configs.split(",") if item.strip()]:
        alpha_raw, beta_raw = raw.split(":", 1)
        alpha = float(alpha_raw)
        beta = float(beta_raw)
        configs.append(
            {
                "label": f"lawf_a_{safe_label(alpha)}_b_{safe_label(beta)}_s_{args.steps}",
                "family": "lawf",
                "steps": args.steps,
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


def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# Llama-3.1-8B-Instruct LAwF Confirmatory Sweep",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Annotation source: `{payload['annotation_json']}`",
        f"- Annotated tasks: `{payload['annotation_summary']['task_count']}`",
        f"- Anchor policy: `{payload['anchor_policy']}`",
        f"- Anchor target probability: `{payload['anchor_target_probability']}`",
        f"- Retokenized anchor tokens: `{payload['annotation_summary']['anchor_tokens']}` / `{payload['annotation_summary']['assistant_tokens']}`",
        f"- Candidate correction-span tokens: `{payload['annotation_summary'].get('candidate_anchor_tokens', payload['annotation_summary']['anchor_tokens'])}`",
        f"- Original Qwen anchor tokens: `{payload['annotation_summary']['original_anchor_tokens']}` / `{payload['annotation_summary']['original_assistant_tokens']}`",
        f"- LoRA: r=`{payload['lora_r']}`, alpha=`{payload['lora_alpha']}`",
        f"- LAwF normalization: `{payload['lawf_normalization']}`",
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
    for row in sorted(payload["summary_rows"], key=lambda item: (item["family"], item["retention_kl_vs_base"])):
        lines.append(format_row(row, include_full_ce=True))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    model_path = snapshot_download(
        args.model_id,
        cache_dir=args.cache_dir,
        ignore_file_pattern=["original/*", "*.pth"],
        max_workers=4,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ref_model = load_base_model(model_path, trainable=False)

    annotation = json.loads(Path(args.annotation_json).read_text(encoding="utf-8"))
    original_tasks = annotation.get("tasks") or [annotation]
    tasks = [
        retokenize_task(
            tokenizer,
            task,
            ref_model=ref_model,
            anchor_policy=args.anchor_policy,
            anchor_target_probability=args.anchor_target_probability,
            anchor_probability_tolerance=args.anchor_probability_tolerance,
        )
        for task in original_tasks
    ]
    batches = [build_training_tensors_from_task(tokenizer, task) for task in tasks]
    reference_continuations = build_reference_continuations(ref_model, tokenizer, args.max_new_tokens)
    configs = make_configs(args)

    effective_anchor_confidence = (
        args.anchor_target_probability if args.anchor_policy == "probability_floor" else args.anchor_confidence
    )

    payload = {
        "model_id": args.model_id,
        "resolved_model_path": model_path,
        "annotation_json": args.annotation_json,
        "seed": args.seed,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "anchor_confidence": effective_anchor_confidence,
        "anchor_policy": args.anchor_policy,
        "anchor_target_probability": args.anchor_target_probability,
        "anchor_probability_tolerance": args.anchor_probability_tolerance,
        "lawf_normalization": args.lawf_normalization,
        "acquisition_probes": ACQUISITION_PROBES,
        "annotation_summary": {
            "task_count": len(tasks),
            "task_names": [task.get("task_name") for task in tasks],
            "assistant_tokens": sum(len(task["completion_ids"]) for task in tasks),
            "anchor_tokens": sum(len(task["anchor_token_indices"]) for task in tasks),
            "candidate_anchor_tokens": sum(len(task["candidate_anchor_token_indices"]) for task in tasks),
            "original_assistant_tokens": sum(task["original_token_count"] for task in tasks),
            "original_anchor_tokens": sum(task["original_anchor_count"] for task in tasks),
            "per_task": [
                {
                    "task_name": task.get("task_name"),
                    "assistant_tokens": len(task["completion_ids"]),
                    "anchor_tokens": len(task["anchor_token_indices"]),
                    "candidate_anchor_tokens": len(task["candidate_anchor_token_indices"]),
                    "original_assistant_tokens": task["original_token_count"],
                    "original_anchor_tokens": task["original_anchor_count"],
                    "probability_records": task["probability_records"],
                }
                for task in tasks
            ],
        },
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
            effective_anchor_confidence,
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
    json_path = work_dir / "llama31_8b_lawf_confirmatory_sweep_results.json"
    md_path = work_dir / "llama31_8b_lawf_confirmatory_sweep_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(md_path, payload)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
