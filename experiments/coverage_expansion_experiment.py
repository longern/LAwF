#!/usr/bin/env python3
"""Coverage-expansion experiment for LAwF.

This script reuses the existing Neuron Silk annotation trace and adds a small
set of extra prompts through the same recursive semantic annotation loop. The
goal is to test whether adding positive coverage improves transfer while
preserving the SFT-vs-LAwF drift comparison.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys

import torch
from modelscope import snapshot_download
from transformers import AutoTokenizer, set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lawf_anchor_experiment import (  # noqa: E402
    CALCULATION_MATERIAL_ERROR_POLICY,
    MATERIAL_ATOMS,
    MATERIAL_RUBRIC,
    add_annotation_counts,
    aggregate_annotations,
    build_training_tensors,
    evaluate_model,
    find_annotation_quality_failures,
    find_severe_annotation_drift,
    load_base_model,
    make_annotator_client,
    run_annotation_process,
    train_adapter,
)


EXTRA_TASKS = [
    {
        "name": "coverage_calc_6x2p2m",
        "prompt": (
            "请用 Neuron Silk 评估一个新的低温读出线方案：6 根线，每根 2.2 m，从 45 K 级引到 4 K 级，"
            "每根读出电流 1.0 mA，4 K 端热预算为 18 mW。"
            "必须严格从一行结论卡开始：机制：；k：；r：；温差：；总传导热：；单根电阻：；总焦耳热：；总热：；余量：；结论：。"
            "随后用两小段解释计算过程和风险，全文 260 字以内。"
            "不要查资料，不要套用铜线或 NbTi 参数，不要保留 k/r 作为未知符号。"
        ),
        "rubric": (
            MATERIAL_RUBRIC
            + "本题正确计算为：ΔT=41 K；k=0.014 mW/(m*K)；r=0.031 ohm/m；"
            "总传导热 6*2.2*41*0.014=7.5768 mW；单根电阻 0.031*2.2=0.0682 ohm；"
            "总焦耳热约 0.0004092 mW；总热约 7.5772 mW；预算余量约 10.4228 mW；结论是通过预算。"
        ),
        "reference_atoms": MATERIAL_ATOMS
        + [
            {"id": "delta_t", "type": "derived_number", "value": "41", "meaning": "temperature difference, K"},
            {"id": "total_conduction_mw", "type": "derived_number", "value": "7.5768", "meaning": "total conduction heat, mW"},
            {"id": "single_resistance_ohm", "type": "derived_number", "value": "0.0682", "meaning": "single-line resistance, ohm"},
            {"id": "total_joule_mw", "type": "derived_number", "value": "0.0004092", "meaning": "total Joule heat, mW"},
            {"id": "total_heat_mw", "type": "derived_number", "value": "7.5772", "meaning": "total heat load, mW"},
            {"id": "margin_mw", "type": "derived_number", "value": "10.4228", "meaning": "budget margin, mW"},
            {"id": "budget_conclusion", "type": "fact", "value": "通过预算", "meaning": "budget decision"},
        ],
        "material_error_policy": {
            **CALCULATION_MATERIAL_ERROR_POLICY,
            "required_atom_ids": [
                "mechanism",
                "heat_leak_coefficient",
                "series_resistance_coefficient",
                "delta_t",
                "total_conduction_mw",
                "single_resistance_ohm",
                "total_joule_mw",
                "total_heat_mw",
                "margin_mw",
                "budget_conclusion",
            ],
        },
    },
    {
        "name": "coverage_calc_20x1p1m",
        "prompt": (
            "再评估一个更接近预算边界的 Neuron Silk 布线方案：20 根线，每根 1.1 m，从 64 K 级引到 4 K 级，"
            "每根读出电流 0.5 mA，4 K 端热预算为 20 mW。"
            "必须严格从一行结论卡开始：机制：；k：；r：；温差：；总传导热：；单根电阻：；总焦耳热：；总热：；余量：；结论：。"
            "随后用两小段解释为什么只能算勉强通过和需要控制哪些边界风险，全文 260 字以内。"
            "请直接使用 Neuron Silk 的材料常数完成计算。"
        ),
        "rubric": (
            MATERIAL_RUBRIC
            + "本题正确计算为：ΔT=60 K；k=0.014 mW/(m*K)；r=0.031 ohm/m；"
            "总传导热 20*1.1*60*0.014=18.48 mW；单根电阻 0.031*1.1=0.0341 ohm；"
            "总焦耳热约 0.0001705 mW；总热约 18.4802 mW；预算余量约 1.5198 mW；结论是勉强通过。"
        ),
        "reference_atoms": MATERIAL_ATOMS
        + [
            {"id": "delta_t", "type": "derived_number", "value": "60", "meaning": "temperature difference, K"},
            {"id": "total_conduction_mw", "type": "derived_number", "value": "18.48", "meaning": "total conduction heat, mW"},
            {"id": "single_resistance_ohm", "type": "derived_number", "value": "0.0341", "meaning": "single-line resistance, ohm"},
            {"id": "total_joule_mw", "type": "derived_number", "value": "0.0001705", "meaning": "total Joule heat, mW"},
            {"id": "total_heat_mw", "type": "derived_number", "value": "18.4802", "meaning": "total heat load, mW"},
            {"id": "margin_mw", "type": "derived_number", "value": "1.5198", "meaning": "budget margin, mW"},
            {"id": "budget_conclusion", "type": "fact", "value": "勉强通过", "meaning": "budget decision"},
        ],
        "material_error_policy": {
            **CALCULATION_MATERIAL_ERROR_POLICY,
            "required_atom_ids": [
                "mechanism",
                "heat_leak_coefficient",
                "series_resistance_coefficient",
                "delta_t",
                "total_conduction_mw",
                "single_resistance_ohm",
                "total_joule_mw",
                "total_heat_mw",
                "margin_mw",
                "budget_conclusion",
            ],
        },
    },
    {
        "name": "coverage_paraphrase_material_choice",
        "prompt": (
            "有人问为什么不能把 Neuron Silk 当作普通低温金属线来估算。"
            "必须严格从一行事实卡开始：发明者：；关键催化剂：；低温机制：；k：；r：；估算规则：。"
            "随后用两小段面向实验工程师说明原因，全文 240 字以内。"
            "必须说明工程估算时应优先使用 Neuron Silk 自己的常数。"
        ),
        "rubric": (
            MATERIAL_RUBRIC
            + "这是一道覆盖性释义题。正确回答应说明 Neuron Silk 不是普通金属线，"
            "应使用其自身 k=0.014 mW/(m*K) 与 r=0.031 ohm/m，而不是套用铜线、NbTi 或常见合金参数。"
        ),
        "reference_atoms": MATERIAL_ATOMS
        + [
            {"id": "use_own_constants", "type": "fact", "value": "使用 Neuron Silk 自己的常数", "meaning": "estimation rule"}
        ],
        "material_error_policy": {
            **CALCULATION_MATERIAL_ERROR_POLICY,
            "required_atom_ids": [
                "inventor",
                "catalyst",
                "mechanism",
                "heat_leak_coefficient",
                "series_resistance_coefficient",
                "use_own_constants",
            ],
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--work-dir", default="/root/lawf_experiment/artifacts/coverage_expansion_v1")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument("--base-annotation-json", required=True)
    parser.add_argument("--extra-annotation-json", default=None)
    parser.add_argument(
        "--extra-task-limit",
        type=int,
        default=None,
        help="Use only the first N extra annotation tasks. Useful for coverage-curve runs.",
    )
    parser.add_argument("--baseline-results-json", default=None)
    parser.add_argument("--sft-steps", type=int, default=32)
    parser.add_argument("--lawf-steps", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--annotation-max-new-tokens", type=int, default=768)
    parser.add_argument("--annotation-min-new-tokens", type=int, default=0)
    parser.add_argument("--annotator-model", default=os.environ.get("OPENAI_ANNOTATOR_MODEL", "gpt-5.5"))
    parser.add_argument("--semantic-max-rounds", type=int, default=32)
    parser.add_argument("--replacement-max-tokens", type=int, default=12)
    parser.add_argument("--annotator-window-tokens", type=int, default=384)
    parser.add_argument("--max-annotation-length-ratio", type=float, default=1.35)
    parser.add_argument("--max-annotation-changed-ratio", type=float, default=0.55)
    parser.add_argument("--allow-annotation-drift", action="store_true")
    parser.add_argument("--allow-annotation-quality-failures", action="store_true")
    parser.add_argument("--annotation-only", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    return parser.parse_args()


def load_task_annotations(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("tasks") or [payload]


def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# Coverage Expansion Experiment",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Base tasks: `{payload['base_task_count']}`",
        f"- Extra recursive-annotation tasks: `{payload['extra_task_count']}`",
        f"- Total tasks: `{payload['total_task_count']}`",
        f"- Anchor tokens: `{payload['annotation']['anchor_token_count']}` / `{payload['annotation']['gold_token_count']}`",
        "",
        "## Scores",
        "",
        "| Model | Semantic score | Learned fact | Transfer calc | Retention KL vs base | Anchor CE | Non-anchor KL | Full CE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ["base", "sft", "lawf"]:
        scores = payload["results"][mode]["scores"]
        metrics = payload["train_metrics"].get(mode, {})
        anchor_ce = "-" if "final_anchor_ce" not in metrics else f"{metrics['final_anchor_ce']:.6f}"
        non_anchor_kl = "-" if "final_non_anchor_kl" not in metrics else f"{metrics['final_non_anchor_kl']:.6f}"
        full_ce = "-" if "final_full_ce" not in metrics else f"{metrics['final_full_ce']:.6f}"
        lines.append(
            f"| {mode.upper()} | {scores['mean_semantic_score']:.3f} | "
            f"{scores['learned_fact_semantic_score']:.3f} | {scores['transfer_calculation_semantic_score']:.3f} | "
            f"{scores['retention_kl_vs_base']:.6f} | "
            f"{anchor_ce} | {non_anchor_kl} | {full_ce} |"
        )
    if payload.get("baseline_summary"):
        lines.extend(
            [
                "",
                "## Baseline Comparison",
                "",
                "| Setting | Model | Mean semantic score | Transfer calc | Retention KL vs base |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for setting, source in [("base-3-task", payload["baseline_summary"]), ("expanded-6-task", payload)]:
            for mode in ["sft", "lawf"]:
                scores = source["results"][mode]["scores"] if setting == "expanded-6-task" else source[mode]
                lines.append(
                    f"| {setting} | {mode.upper()} | {scores['mean_semantic_score']:.3f} | "
                    f"{scores['transfer_calculation_semantic_score']:.3f} | {scores['retention_kl_vs_base']:.6f} |"
                )
    lines.extend(["", "## Extra Annotation Tasks", ""])
    lines.extend(["| Task | Tokens | Anchors | Anchor ratio | Rounds |", "| --- | ---: | ---: | ---: | ---: |"])
    for task in payload["annotation"].get("tasks", [])[payload["base_task_count"] :]:
        gold_count = task.get("gold_token_count", len(task.get("completion_ids", [])))
        anchor_count = task.get("anchor_token_count", len(task.get("anchor_token_indices", [])))
        anchor_ratio = task.get("anchor_ratio", anchor_count / gold_count if gold_count else 0.0)
        lines.append(
            f"| {task['task_name']} | {gold_count} | {anchor_count} | "
            f"{anchor_ratio:.3%} | {len(task.get('rounds', []))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    model_path = snapshot_download(args.model_id, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    ref_model = load_base_model(model_path, trainable=False)
    evaluator_client = make_annotator_client()

    base_task_annotations = load_task_annotations(Path(args.base_annotation_json))
    if args.extra_annotation_json:
        extra_task_annotations = load_task_annotations(Path(args.extra_annotation_json))
        if args.extra_task_limit is not None:
            extra_task_annotations = extra_task_annotations[: args.extra_task_limit]
    else:
        extra_task_annotations = []
        tasks = EXTRA_TASKS if args.extra_task_limit is None else EXTRA_TASKS[: args.extra_task_limit]
        for task in tasks:
            extra_task_annotations.append(run_annotation_process(ref_model, tokenizer, task, args))
            partial = aggregate_annotations(base_task_annotations + extra_task_annotations)
            (work_dir / "annotation_trace.partial.json").write_text(
                json.dumps(add_annotation_counts(partial), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    extra_trace_path = work_dir / "extra_annotation_trace.json"
    extra_trace_path.write_text(
        json.dumps(add_annotation_counts(aggregate_annotations(extra_task_annotations)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    annotation = aggregate_annotations(base_task_annotations + extra_task_annotations)
    counted_annotation = add_annotation_counts(annotation)
    annotation_path = work_dir / "annotation_trace.json"
    annotation_path.write_text(json.dumps(counted_annotation, ensure_ascii=False, indent=2), encoding="utf-8")

    drift_failures = find_severe_annotation_drift(counted_annotation)
    if drift_failures and not args.allow_annotation_drift:
        raise RuntimeError(f"Annotation drift audit failed: {json.dumps(drift_failures, ensure_ascii=False)}")
    quality_failures = find_annotation_quality_failures(counted_annotation)
    if quality_failures and not args.allow_annotation_quality_failures:
        raise RuntimeError(f"Annotation quality audit failed: {json.dumps(quality_failures, ensure_ascii=False)}")
    if args.annotation_only:
        print(json.dumps({"annotation": str(annotation_path), "extra_annotation": str(extra_trace_path)}, ensure_ascii=False))
        return 0

    batches = [
        build_training_tensors(
            tokenizer,
            task_annotation["prompt"],
            task_annotation["completion_ids"],
            task_annotation["anchor_token_indices"],
        )
        for task_annotation in counted_annotation["tasks"]
    ]

    from lawf_anchor_experiment import build_reference_continuations  # local import keeps the dependency explicit

    reference_continuations = build_reference_continuations(ref_model, tokenizer, args.max_new_tokens)

    baseline_summary = None
    if args.baseline_results_json:
        baseline = json.loads(Path(args.baseline_results_json).read_text(encoding="utf-8"))
        baseline_summary = {
            mode: baseline["results"][mode]["scores"]
            for mode in ["sft", "lawf"]
        }

    payload = {
        "model_id": args.model_id,
        "resolved_model_path": model_path,
        "seed": args.seed,
        "sft_steps": args.sft_steps,
        "lawf_steps": args.lawf_steps,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "base_task_count": len(base_task_annotations),
        "extra_task_count": len(extra_task_annotations),
        "total_task_count": len(base_task_annotations) + len(extra_task_annotations),
        "annotation": counted_annotation,
        "baseline_summary": baseline_summary,
        "results": {
            "base": evaluate_model(
                "base",
                ref_model,
                ref_model,
                tokenizer,
                reference_continuations,
                args.max_new_tokens,
                evaluator_client,
                args.annotator_model,
            )
        },
        "train_metrics": {},
    }

    for mode in ["sft", "lawf"]:
        steps = args.lawf_steps if mode == "lawf" else args.sft_steps
        trained = train_adapter(
            mode,
            model_path,
            ref_model,
            batches,
            steps,
            args.lr,
            work_dir / f"{mode}_adapter",
            args.lora_r,
            args.lora_alpha,
        )
        payload["train_metrics"][mode] = trained["metrics"]
        payload["results"][mode] = evaluate_model(
            mode,
            trained["model"],
            ref_model,
            tokenizer,
            reference_continuations,
            args.max_new_tokens,
            evaluator_client,
            args.annotator_model,
        )
        del trained
        gc.collect()
        torch.cuda.empty_cache()

    json_path = work_dir / "coverage_expansion_results.json"
    report_path = work_dir / "coverage_expansion_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_path, payload)
    print(json.dumps({"json": str(json_path), "markdown": str(report_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
