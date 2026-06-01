#!/usr/bin/env python3
"""Boundary coverage sweep for sparse correction applicability.

The original boundary negative-control experiment compares positive-only
training against one fixed boundary-augmented set. This sweep varies the number
of boundary examples so the paper can report whether explicit boundary coverage
changes contamination margins monotonically.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import torch
from modelscope import snapshot_download
from transformers import AutoTokenizer, set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from boundary_negative_control_experiment import (  # noqa: E402
    BOUNDARY_EDITS,
    BOUNDARY_PROBES,
    POSITIVE_EDITS,
    build_batches,
    evaluate_boundary_margins,
    evaluate_generations,
)
from lawf_anchor_experiment import load_base_model, train_adapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--work-dir", default="/root/lawf_experiment/artifacts/boundary_coverage_sweep_v1")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--anchor-confidence", type=float, default=0.999)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--boundary-counts", default="0,1,2,4")
    parser.add_argument("--modes", nargs="+", default=["sft", "lawf"], choices=["sft", "lawf"])
    return parser.parse_args()


def parse_counts(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("expected at least one boundary count")
    if any(value < 0 or value > len(BOUNDARY_EDITS) for value in values):
        raise ValueError(f"boundary counts must be between 0 and {len(BOUNDARY_EDITS)}")
    return values


def unload(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    model_path = snapshot_download(args.model_id, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    positive_meta, positive_batches = build_batches(tokenizer, POSITIVE_EDITS)
    boundary_meta, boundary_batches = build_batches(tokenizer, BOUNDARY_EDITS)
    ref_model = load_base_model(model_path, trainable=False)
    boundary_counts = parse_counts(args.boundary_counts)

    results = {
        "base": {
            "mode": "base",
            "boundary_count": 0,
            "training": None,
            "boundary_margin": evaluate_boundary_margins(ref_model, tokenizer),
            "generations": evaluate_generations(ref_model, tokenizer, args.max_new_tokens),
        }
    }

    for count in boundary_counts:
        selected_batches = positive_batches + boundary_batches[:count]
        for mode in args.modes:
            label = f"{mode}_boundary_{count}"
            print(json.dumps({"event": "train", "label": label, "mode": mode, "boundary_count": count}, ensure_ascii=False), flush=True)
            trained = train_adapter(
                mode=mode,
                model_path=model_path,
                ref_model=ref_model,
                batches=selected_batches,
                steps=args.steps,
                lr=args.lr,
                output_dir=work_dir / f"{label}_adapter",
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                anchor_confidence=args.anchor_confidence,
            )
            model = trained["model"]
            model.eval()
            results[label] = {
                "mode": mode,
                "boundary_count": count,
                "training": trained["metrics"],
                "boundary_margin": evaluate_boundary_margins(model, tokenizer),
                "generations": evaluate_generations(model, tokenizer, args.max_new_tokens),
            }
            print(
                json.dumps(
                    {
                        "event": "eval",
                        "label": label,
                        "mean_margin": results[label]["boundary_margin"]["mean_margin"],
                        "forbidden_preferred": results[label]["boundary_margin"]["forbidden_preferred"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            unload(model)

    summary_rows = []
    for name, row in results.items():
        summary_rows.append(
            {
                "label": name,
                "mode": row["mode"],
                "boundary_count": row["boundary_count"],
                "mean_boundary_margin": row["boundary_margin"]["mean_margin"],
                "forbidden_preferred": row["boundary_margin"]["forbidden_preferred"],
                "generated_forbidden_hits": sum(item["contains_forbidden"] for item in row["generations"]),
                "generated_correct_hits": sum(item["contains_correct"] for item in row["generations"]),
            }
        )

    payload = {
        "model_id": args.model_id,
        "steps": args.steps,
        "lr": args.lr,
        "seed": args.seed,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "positive_edits": positive_meta,
        "boundary_edits": boundary_meta,
        "boundary_probes": BOUNDARY_PROBES,
        "boundary_counts": boundary_counts,
        "results": results,
        "summary_rows": summary_rows,
    }
    json_path = work_dir / "boundary_coverage_sweep_results.json"
    md_path = work_dir / "boundary_coverage_sweep_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Boundary Coverage Sweep",
        "",
        f"- Model: `{args.model_id}`",
        f"- Steps: `{args.steps}`",
        f"- Positive edits: `{len(POSITIVE_EDITS)}`",
        f"- Boundary counts: `{', '.join(str(value) for value in boundary_counts)}`",
        "",
        "| Model | Boundary examples | Mean boundary margin | Forbidden preferred | Generated forbidden hits | Generated correct hits |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['label']} | {row['boundary_count']} | {row['mean_boundary_margin']:.6f} | "
            f"{row['forbidden_preferred']} / {len(BOUNDARY_PROBES)} | "
            f"{row['generated_forbidden_hits']} / {len(BOUNDARY_PROBES)} | "
            f"{row['generated_correct_hits']} / {len(BOUNDARY_PROBES)} |"
        )
    lines.extend(["", "## Per-Probe Margins", ""])
    for name in ["base"] + [f"{mode}_boundary_{count}" for count in boundary_counts for mode in args.modes]:
        lines.extend([f"### {name}", "", "| Probe | Correct | Forbidden | Margin | Forbidden preferred |", "| --- | --- | --- | ---: | --- |"])
        for item in results[name]["boundary_margin"]["items"]:
            lines.append(
                f"| {item['id']} | `{item['correct']}` | `{item['forbidden']}` | "
                f"{item['margin']:.6f} | {item['forbidden_preferred']} |"
            )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
