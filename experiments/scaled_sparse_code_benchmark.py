#!/usr/bin/env python3
"""Sparse recursive correction scale benchmark for LAwF.

This benchmark is intentionally narrower than `scaled_recursive_benchmark.py`.
Each edit family contains exactly one short target value embedded in a long
otherwise-free completion. The goal is to test the sparse-correction setting:
few anchor tokens, many non-anchor tokens, and held-out probes outside the
annotated training completions.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from modelscope import snapshot_download
from transformers import AutoTokenizer, set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lawf_anchor_experiment import (  # noqa: E402
    add_annotation_counts,
    aggregate_annotations,
    apply_chat_template,
    build_training_tensors,
    ce_on_mask,
    find_annotation_quality_failures,
    generate,
    kl_ref_to_model,
    load_base_model,
    run_annotation_process,
    train_adapter,
)


FAMILIES = [
    {
        "id": "archive_delta",
        "domain": "archive",
        "subject": "Delta Archive",
        "label": "internal verification code",
        "value": "ravon",
        "context": "a fictional archive system used to route restoration requests",
    },
    {
        "id": "cache_nova",
        "domain": "api",
        "subject": "NovaCache",
        "label": "cache lane code",
        "value": "kelpa",
        "context": "a fictional cache service for delayed telemetry reads",
    },
    {
        "id": "badge_orange",
        "domain": "policy",
        "subject": "Orange Badge desk access",
        "label": "review gate code",
        "value": "mirto",
        "context": "a fictional office access policy for temporary visitors",
    },
    {
        "id": "bot_mira",
        "domain": "robotics",
        "subject": "Mira courier robot",
        "label": "return route code",
        "value": "navo",
        "context": "a fictional warehouse robot operating under low-battery routing",
    },
    {
        "id": "game_fog",
        "domain": "game_rule",
        "subject": "Fogbridge board game",
        "label": "scenario key",
        "value": "qorin",
        "context": "a fictional board-game scenario with movement constraints",
    },
    {
        "id": "lab_lumen",
        "domain": "chemistry",
        "subject": "Lumen reagent log",
        "label": "assay marker code",
        "value": "belta",
        "context": "a fictional reagent logging procedure for repeated assays",
    },
    {
        "id": "dsl_amber",
        "domain": "programming_language",
        "subject": "AmberLoop DSL",
        "label": "compiler mode code",
        "value": "toren",
        "context": "a fictional domain-specific language for declarative loops",
    },
    {
        "id": "invoice_blue",
        "domain": "business_rule",
        "subject": "Blue Invoice A17",
        "label": "audit route code",
        "value": "halvo",
        "context": "a fictional invoice review rule for exception routing",
    },
    {
        "id": "sensor_iris",
        "domain": "hardware",
        "subject": "Iris pressure sensor",
        "label": "calibration band code",
        "value": "pindra",
        "context": "a fictional pressure sensor used in sealed cryogenic cabinets",
    },
    {
        "id": "router_elm",
        "domain": "networking",
        "subject": "ElmEdge router",
        "label": "fallback route code",
        "value": "vesko",
        "context": "a fictional edge router used when primary links lose telemetry",
    },
    {
        "id": "drone_cedar",
        "domain": "robotics",
        "subject": "Cedar inspection drone",
        "label": "landing pattern code",
        "value": "luma",
        "context": "a fictional indoor drone for warehouse shelf inspection",
    },
    {
        "id": "ledger_silver",
        "domain": "finance",
        "subject": "Silver Ledger",
        "label": "exception bucket code",
        "value": "emora",
        "context": "a fictional accounting ledger for reserve reconciliation",
    },
    {
        "id": "clinic_pine",
        "domain": "healthcare",
        "subject": "Pine Clinic triage",
        "label": "queue marker code",
        "value": "yavin",
        "context": "a fictional clinic workflow for non-urgent equipment requests",
    },
    {
        "id": "compiler_onyx",
        "domain": "programming_language",
        "subject": "OnyxScript compiler",
        "label": "strict mode code",
        "value": "sorn",
        "context": "a fictional scripting compiler with two validation passes",
    },
    {
        "id": "warehouse_cobalt",
        "domain": "operations",
        "subject": "Cobalt warehouse zone",
        "label": "cycle count code",
        "value": "corin",
        "context": "a fictional warehouse zone used for monthly inventory checks",
    },
    {
        "id": "dataset_marble",
        "domain": "data",
        "subject": "Marble dataset",
        "label": "split manifest code",
        "value": "daska",
        "context": "a fictional dataset release with staged validation splits",
    },
    {
        "id": "ship_harbor",
        "domain": "logistics",
        "subject": "Harbor shuttle manifest",
        "label": "dock transfer code",
        "value": "gaven",
        "context": "a fictional port shuttle used for container handoff planning",
    },
    {
        "id": "auth_copper",
        "domain": "security",
        "subject": "CopperAuth session",
        "label": "device trust code",
        "value": "ulmar",
        "context": "a fictional authentication flow for shared laboratory terminals",
    },
    {
        "id": "search_river",
        "domain": "search",
        "subject": "RiverSearch index",
        "label": "ranking profile code",
        "value": "jorin",
        "context": "a fictional search index for archival snippets",
    },
    {
        "id": "pipeline_glass",
        "domain": "data",
        "subject": "GlassFlow pipeline",
        "label": "retry stage code",
        "value": "wexla",
        "context": "a fictional data pipeline that retries failed parsing stages",
    },
    {
        "id": "museum_lantern",
        "domain": "archive",
        "subject": "Lantern Museum loan",
        "label": "condition review code",
        "value": "faryn",
        "context": "a fictional museum loan process for fragile exhibit items",
    },
    {
        "id": "scheduler_opal",
        "domain": "systems",
        "subject": "Opal scheduler",
        "label": "preemption lane code",
        "value": "avro",
        "context": "a fictional job scheduler for mixed latency workloads",
    },
    {
        "id": "thermal_jade",
        "domain": "hardware",
        "subject": "Jade thermal plate",
        "label": "mounting profile code",
        "value": "zento",
        "context": "a fictional thermal plate used in compact sensor assemblies",
    },
    {
        "id": "calendar_ember",
        "domain": "productivity",
        "subject": "Ember calendar",
        "label": "handoff reminder code",
        "value": "orli",
        "context": "a fictional scheduling product for shift handoffs",
    },
    {
        "id": "vault_ash",
        "domain": "security",
        "subject": "AshVault record",
        "label": "seal rotation code",
        "value": "xanor",
        "context": "a fictional record vault with rotating access seals",
    },
    {
        "id": "simulator_brook",
        "domain": "simulation",
        "subject": "BrookSim scenario",
        "label": "terrain preset code",
        "value": "ivra",
        "context": "a fictional simulation scenario for indoor navigation tests",
    },
    {
        "id": "mailroom_violet",
        "domain": "operations",
        "subject": "Violet mailroom",
        "label": "sort lane code",
        "value": "mora",
        "context": "a fictional mailroom sorting process for internal packages",
    },
    {
        "id": "lab_orchid",
        "domain": "chemistry",
        "subject": "Orchid lab notebook",
        "label": "sample freeze code",
        "value": "qelma",
        "context": "a fictional lab notebook workflow for repeated sample freezes",
    },
    {
        "id": "map_zephyr",
        "domain": "geography",
        "subject": "Zephyr map layer",
        "label": "contour source code",
        "value": "talen",
        "context": "a fictional map layer for contour overlays in route planning",
    },
    {
        "id": "payment_amber",
        "domain": "finance",
        "subject": "AmberPay settlement",
        "label": "reserve release code",
        "value": "renda",
        "context": "a fictional payment settlement flow for prepaid reserves",
    },
]


RETENTION_PROMPTS = [
    "直接回答：法国的首都是哪里？",
    "直接回答：17 + 28 等于多少？",
    "把这句话翻译成英文：这个实验需要更多样本。",
    "用一句话解释什么是二分查找。",
    "写一个 Python 表达式，把列表 xs 的长度取出来。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--work-dir", default="/root/lawf_experiment/artifacts/scaled_sparse_code_benchmark_v1")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument("--family-limit", type=int, default=30)
    parser.add_argument("--scale-points", default="1,8,16,30")
    parser.add_argument("--modes", nargs="+", default=["sft", "sft_kl", "lawf"], choices=["sft", "anchor_only", "sft_kl", "lawf"])
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--annotation-max-new-tokens", type=int, default=360)
    parser.add_argument("--annotation-min-new-tokens", type=int, default=0)
    parser.add_argument("--annotator-model", default=os.environ.get("OPENAI_ANNOTATOR_MODEL", "gpt-5.5"))
    parser.add_argument("--semantic-max-rounds", type=int, default=8)
    parser.add_argument("--replacement-max-tokens", type=int, default=4)
    parser.add_argument("--annotator-window-tokens", type=int, default=360)
    parser.add_argument("--max-annotation-length-ratio", type=float, default=2.5)
    parser.add_argument("--max-annotation-changed-ratio", type=float, default=0.98)
    parser.add_argument("--allow-annotation-quality-failures", action="store_true")
    parser.add_argument("--annotation-only", action="store_true")
    parser.add_argument("--annotation-json", default=None)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--anchor-confidence", type=float, default=0.999)
    parser.add_argument("--lawf-betas", default="1.0")
    parser.add_argument(
        "--lawf-normalization",
        choices=["group_mean", "token_mean"],
        default="token_mean",
        help=(
            "How to combine LAwF anchor and non-anchor terms. token_mean matches the paper "
            "objective by weighting each term by its token count before dividing by assistant tokens."
        ),
    )
    return parser.parse_args()


def task_for_family(family: dict) -> dict:
    value = family["value"]
    atom = {
        "id": "target_value",
        "type": "fact",
        "value": value,
        "when_to_anchor": (
            "Anchor the first token of an incorrect, missing, or contradictory value for the requested code. "
            f"When correcting the value field, prefer replacement_text '{value}.' including the trailing period. "
            "If the generated code already begins with the correct value but continues with extra alphanumeric "
            "characters, replace the first extra character with a period."
        ),
    }
    return {
        "family_id": family["id"],
        "domain": family["domain"],
        "task_name": f"{family['id']}_long_note",
        "name": f"{family['id']}_long_note",
        "prompt": (
            f"Write a plain-text internal note about {family['subject']}, {family['context']}. "
            "Do not use Markdown, bold text, bullet points, or tables. "
            "Write exactly four paragraphs, with exactly two sentences per paragraph, then stop. "
            f"In the first paragraph, include exactly one sentence of the form "
            f"\"The {family['label']} is <concrete value>.\" End that sentence immediately after the value with a period. "
            "Use a concrete value even if uncertain. "
            "The remaining sentences should discuss background, operational use, edge cases, and a final recommendation."
        ),
        "rubric": (
            f"This is a controlled fictional correction task. The only material target is the value of "
            f"{family['subject']}'s {family['label']}. The correct value is {value}. "
            f"The value sentence should end immediately after {value} with a period. "
            "If the model writes the correct value followed by extra code characters, the first extra character is a material error "
            "and should be replaced with a period. Other background wording is not a target unless it contradicts this value."
        ),
        "reference_atoms": [atom],
        "material_error_policy": {
            "domain": family["domain"],
            "anchor_targets": ["fact"],
            "required_atom_ids": ["target_value"],
            "forbidden_residuals": [],
            "forbidden_patterns": [],
            "non_targets": [
                "style",
                "paragraph wording",
                "generic operational advice",
                "background details",
                "recommendation wording",
            ],
            "numeric_tolerance": None,
        },
    }


def task_family_id(task_annotation: dict) -> str:
    return str(task_annotation.get("family_id") or task_annotation["task_name"].rsplit("_", 2)[0])


def selected_families(limit: int) -> list[dict]:
    return FAMILIES[: min(limit, len(FAMILIES))]


def run_or_load_annotations(ref_model, tokenizer, args: argparse.Namespace, families: list[dict], work_dir: Path) -> dict:
    if args.annotation_json:
        return json.loads(Path(args.annotation_json).read_text(encoding="utf-8"))

    task_annotations = []
    for family in families:
        task = task_for_family(family)
        annotation = run_annotation_process(ref_model, tokenizer, task, args)
        annotation["family_id"] = family["id"]
        annotation["domain"] = family["domain"]
        task_annotations.append(annotation)

    annotation = add_annotation_counts(aggregate_annotations(task_annotations))
    quality_failures = find_annotation_quality_failures(annotation)
    if quality_failures and not args.allow_annotation_quality_failures:
        raise RuntimeError(f"Annotation quality audit failed: {json.dumps(quality_failures, ensure_ascii=False)}")
    (work_dir / "annotation_trace.json").write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")
    return annotation


def build_batches_for_annotations(tokenizer, annotations: list[dict]) -> list[dict[str, torch.Tensor]]:
    return [
        build_training_tensors(tokenizer, row["prompt"], row["completion_ids"], row["anchor_token_indices"])
        for row in annotations
    ]


def score_answer_ce(model, tokenizer, prompt: str, answer: str) -> float:
    prefix = apply_chat_template(tokenizer, [{"role": "user", "content": prompt}], add_generation_prompt=True)
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer + (tokenizer.eos_token or ""), add_special_tokens=False).input_ids
    input_ids = torch.tensor([prefix_ids + answer_ids], dtype=torch.long, device=model.device)
    labels = input_ids[:, 1:].clone()
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, max(len(prefix_ids) - 1, 0) :] = True
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[:, :-1, :]
    return float(ce_on_mask(logits, labels, mask).detach().cpu())


def evaluate_probes(model, tokenizer, families: list[dict]) -> dict:
    direct_rows = []
    paraphrase_rows = []
    for family in families:
        direct_rows.append(
            {
                "family_id": family["id"],
                "domain": family["domain"],
                "ce": score_answer_ce(
                    model,
                    tokenizer,
                    f"What is the {family['label']} for {family['subject']}?",
                    family["value"],
                ),
            }
        )
        paraphrase_rows.append(
            {
                "family_id": family["id"],
                "domain": family["domain"],
                "ce": score_answer_ce(
                    model,
                    tokenizer,
                    f"An operator asks for the routing value associated with {family['subject']}. Reply with only the code.",
                    family["value"],
                ),
            }
        )
    return {
        "direct_ce": sum(row["ce"] for row in direct_rows) / len(direct_rows),
        "paraphrase_ce": sum(row["ce"] for row in paraphrase_rows) / len(paraphrase_rows),
        "direct_rows": direct_rows,
        "paraphrase_rows": paraphrase_rows,
    }


def build_reference_continuations(ref_model, tokenizer, max_new_tokens: int) -> dict[str, str]:
    return {prompt: generate(ref_model, tokenizer, prompt, max_new_tokens) for prompt in RETENTION_PROMPTS}


def score_retention_kl(model, ref_model, tokenizer, continuations: dict[str, str]) -> float:
    values = []
    for prompt, continuation in continuations.items():
        prefix = apply_chat_template(tokenizer, [{"role": "user", "content": prompt}], add_generation_prompt=True)
        full = prefix + continuation + (tokenizer.eos_token or "")
        input_ids = tokenizer(full, add_special_tokens=False, return_tensors="pt").input_ids
        prefix_len = len(tokenizer(prefix, add_special_tokens=False).input_ids)
        labels = input_ids[:, 1:]
        mask = torch.zeros_like(labels, dtype=torch.bool)
        mask[:, max(prefix_len - 1, 0) :] = True
        with torch.no_grad():
            model_logits = model(input_ids.to(model.device)).logits[:, :-1, :].cpu()
            ref_logits = ref_model(input_ids.to(ref_model.device)).logits[:, :-1, :].cpu()
        values.append(float(kl_ref_to_model(model_logits, ref_logits, mask).item()))
    return sum(values) / len(values)


def parse_float_list(raw: str) -> list[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("expected at least one float value")
    return values


def beta_label(beta: float) -> str:
    return f"{beta:g}"


def mode_label(mode: str, beta: float) -> str:
    if mode == "lawf":
        return "lawf" if beta == 1.0 else f"lawf_beta_{beta_label(beta)}"
    return mode


def summarize_annotations(annotation: dict) -> dict:
    tasks = annotation.get("tasks") or [annotation]
    corrected_rounds = [row for task in tasks for row in task.get("rounds", []) if row.get("status") == "corrected"]
    return {
        "task_count": len(tasks),
        "family_count": len({task_family_id(task) for task in tasks}),
        "gold_token_count": len(annotation["completion_ids"]),
        "anchor_token_count": len(annotation["anchor_token_indices"]),
        "anchor_ratio": len(annotation["anchor_token_indices"]) / max(len(annotation["completion_ids"]), 1),
        "corrected_rounds": len(corrected_rounds),
        "mean_corrected_rounds_per_task": len(corrected_rounds) / max(len(tasks), 1),
        "task_counts": annotation.get("task_counts", []),
    }


def write_report(path: Path, payload: dict) -> None:
    ann = payload["annotation_summary"]
    lines = [
        "# Scaled Sparse Code Benchmark",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Families annotated: `{ann['family_count']}`",
        f"- Annotated tasks: `{ann['task_count']}`",
        f"- Anchor tokens: `{ann['anchor_token_count']}` / `{ann['gold_token_count']}` ({ann['anchor_ratio'] * 100:.2f}%)",
        f"- Mean corrected rounds per task: `{ann['mean_corrected_rounds_per_task']:.2f}`",
        f"- Steps: `{payload['steps']}`",
        f"- Anchor confidence: `{payload['anchor_confidence']}`",
        f"- LAwF betas: `{payload['lawf_betas']}`",
        f"- LAwF normalization: `{payload['lawf_normalization']}`",
        "",
        "## Held-Out Scale Curve",
        "",
        "| Families | Model | Anchor CE | Diagnostic train non-anchor KL | Full CE | Direct CE | Paraphrase CE | Held-out retention KL |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scale in payload["scale_points"]:
        scale_payload = payload["scale_results"][str(scale)]
        for mode in scale_payload["train_metrics"]:
            metrics = scale_payload["train_metrics"][mode]
            evals = scale_payload["eval"][mode]
            lines.append(
                f"| {scale} | {mode} | {metrics['final_anchor_ce']:.6f} | "
                f"{metrics['final_non_anchor_kl']:.6f} | {metrics['final_full_ce']:.6f} | "
                f"{evals['direct_ce']:.6f} | {evals['paraphrase_ce']:.6f} | "
                f"{evals['retention_kl_vs_base']:.6f} |"
            )
    lines.extend(["", "## Annotation Load", ""])
    lines.append("| Task | Tokens | Anchors | Anchor ratio |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in ann["task_counts"]:
        lines.append(
            f"| {row['task_name']} | {row['gold_token_count']} | {row['anchor_token_count']} | "
            f"{row['anchor_ratio'] * 100:.2f}% |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    families = selected_families(args.family_limit)
    scale_points = [int(value.strip()) for value in args.scale_points.split(",") if value.strip()]
    scale_points = [point for point in scale_points if 1 <= point <= len(families)]
    if not scale_points:
        raise ValueError("--scale-points must include at least one point within --family-limit")

    model_path = snapshot_download(args.model_id, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ref_model = load_base_model(model_path, trainable=False)

    annotation = add_annotation_counts(run_or_load_annotations(ref_model, tokenizer, args, families, work_dir))
    (work_dir / "annotation_trace.json").write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.annotation_only:
        print(json.dumps({"annotation_json": str(work_dir / "annotation_trace.json")}, ensure_ascii=False), flush=True)
        return 0

    annotations_by_family: dict[str, list[dict]] = {}
    for task in annotation.get("tasks") or [annotation]:
        annotations_by_family.setdefault(task_family_id(task), []).append(task)

    retention_continuations = build_reference_continuations(ref_model, tokenizer, args.max_new_tokens)
    payload = {
        "model_id": args.model_id,
        "resolved_model_path": model_path,
        "seed": args.seed,
        "steps": args.steps,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "anchor_confidence": args.anchor_confidence,
        "lawf_betas": parse_float_list(args.lawf_betas),
        "lawf_normalization": args.lawf_normalization,
        "modes": args.modes,
        "scale_points": scale_points,
        "annotation_summary": summarize_annotations(annotation),
        "scale_results": {},
    }

    for scale in scale_points:
        scale_families = families[:scale]
        scale_annotations = [
            task
            for family in scale_families
            for task in annotations_by_family.get(family["id"], [])
        ]
        batches = build_batches_for_annotations(tokenizer, scale_annotations)
        scale_payload = {"family_ids": [family["id"] for family in scale_families], "train_metrics": {}, "eval": {}}
        for mode in args.modes:
            betas = payload["lawf_betas"] if mode == "lawf" else [1.0]
            for beta in betas:
                label = mode_label(mode, beta)
                trained = train_adapter(
                    mode,
                    model_path,
                    ref_model,
                    batches,
                    args.steps,
                    args.lr,
                    work_dir / f"scale_{scale}_{label}_adapter",
                    args.lora_r,
                    args.lora_alpha,
                    args.anchor_confidence,
                    lawf_beta=beta,
                    lawf_normalization=args.lawf_normalization,
                )
                scale_payload["train_metrics"][label] = trained["metrics"]
                scale_payload["eval"][label] = {
                    **evaluate_probes(trained["model"], tokenizer, scale_families),
                    "retention_kl_vs_base": score_retention_kl(
                        trained["model"],
                        ref_model,
                        tokenizer,
                        retention_continuations,
                    ),
                }
                del trained
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        payload["scale_results"][str(scale)] = scale_payload

    json_path = work_dir / "scaled_sparse_code_benchmark_results.json"
    md_path = work_dir / "scaled_sparse_code_benchmark_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(md_path, payload)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
