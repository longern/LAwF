#!/usr/bin/env python3
"""ROME baseline on the adapted KnowEdit/ZsRE short-answer benchmark.

This script evaluates model editing as a formal real-data baseline for the
KnowEdit-ZsRE conversion used by the LAwF paper. It applies one ROME edit at a
time, evaluates the edited model on the corresponding direct, rephrase,
portability, locality, and retention probes, then restores the base weights
before the next edit.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
from modelscope import snapshot_download
from transformers import AutoTokenizer, set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from knowedit_zsre_lawf_benchmark import (  # noqa: E402
    DEFAULT_ZSRE_URL,
    build_edit_items,
    load_zsre_rows,
    score_kl_on_prompt_answers,
)
from lawf_anchor_experiment import load_base_model  # noqa: E402
from micro_edit_benchmark import (  # noqa: E402
    build_reference_continuations,
    score_answer_ce,
    score_retention_kl,
)
from rome_qwen3_micro_edit_diagnostic import (  # noqa: E402
    build_hparams,
    import_rome,
    load_model_and_tokenizer,
    restore_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument("--work-dir", default="/root/lawf_experiment/artifacts/rome_knowedit_zsre_128_qwen06_v1")
    parser.add_argument("--easyedit-dir", default="/root/lawf_experiment/third_party/EasyEdit")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--download-url", default=DEFAULT_ZSRE_URL)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--v-loss-layer", type=int, default=27)
    parser.add_argument("--v-steps", type=int, default=25)
    parser.add_argument("--v-lr", type=float, default=5e-1)
    parser.add_argument("--kl-factor", type=float, default=0.0625)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--answer-template", default="{target}")
    return parser.parse_args()


def log_event(event: str, **kwargs: Any) -> None:
    print(json.dumps({"event": event, **kwargs}, ensure_ascii=False), flush=True)


def resolve_model_path(model_id: str, cache_dir: str) -> str:
    candidate = Path(model_id).expanduser()
    if candidate.exists():
        return str(candidate)
    return snapshot_download(model_id, cache_dir=cache_dir)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def first_portability_probe(item: dict[str, Any]) -> dict[str, str] | None:
    probes = item.get("portability_probes") or []
    return probes[0] if probes else None


def flatten_locality_probes(item: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {**probe, "id": item["id"], "source_index": item["source_index"]}
        for probe in item.get("locality_probes", [])
    ]


def evaluate_item(model, ref_model, tokenizer, item: dict[str, Any], retention_continuations: dict[str, str]) -> dict[str, Any]:
    direct_ce = score_answer_ce(model, tokenizer, item["direct_probe"]["prompt"], item["direct_probe"]["answer"])
    rephrase_ce = score_answer_ce(model, tokenizer, item["paraphrase_probe"]["prompt"], item["paraphrase_probe"]["answer"])
    portability = first_portability_probe(item)
    portability_ce = (
        score_answer_ce(model, tokenizer, portability["prompt"], portability["answer"])
        if portability
        else math.nan
    )
    locality_kl, locality_rows = score_kl_on_prompt_answers(
        model,
        ref_model,
        tokenizer,
        flatten_locality_probes(item),
    )
    return {
        "direct_ce": direct_ce,
        "rephrase_ce": rephrase_ce,
        "portability_ce": portability_ce,
        "locality_kl": locality_kl,
        "locality_rows": locality_rows,
        "retention_kl": score_retention_kl(model, ref_model, tokenizer, retention_continuations),
    }


def make_request(item: dict[str, Any]) -> dict[str, str]:
    return {
        "prompt": item["prompt"],
        "subject": item["subject"],
        "target_new": item["target_new"],
        "ground_truth": item.get("old_answer") or "<|endoftext|>",
    }


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    before = [row["before"] for row in rows]
    after = [row["after"] for row in rows]
    return {
        "case_count": len(rows),
        "mean_direct_ce_before": mean([row["direct_ce"] for row in before]),
        "mean_direct_ce_after": mean([row["direct_ce"] for row in after]),
        "mean_rephrase_ce_before": mean([row["rephrase_ce"] for row in before]),
        "mean_rephrase_ce_after": mean([row["rephrase_ce"] for row in after]),
        "mean_portability_ce_before": mean([row["portability_ce"] for row in before if not math.isnan(row["portability_ce"])]),
        "mean_portability_ce_after": mean([row["portability_ce"] for row in after if not math.isnan(row["portability_ce"])]),
        "mean_locality_kl_after": mean([row["locality_kl"] for row in after if not math.isnan(row["locality_kl"])]),
        "mean_retention_kl_after": mean([row["retention_kl"] for row in after]),
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# ROME KnowEdit-ZsRE Baseline",
        "",
        "This baseline applies one ROME edit at a time on the adapted KnowEdit-ZsRE short-answer benchmark.",
        "Each edit is evaluated and then reverted before the next edit.",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Selected edits: `{payload['dataset_summary']['selected_count']}`",
        f"- ROME layer: `{payload['hparams']['layers']}`",
        f"- ROME v-loss layer: `{payload['hparams']['v_loss_layer']}`",
        "",
        "## Summary",
        "",
        "| Setting | Direct CE | Rephrase CE | Portability CE | Locality KL | Retention KL |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Base | {summary['mean_direct_ce_before']:.6f} | {summary['mean_rephrase_ce_before']:.6f} | "
            f"{summary['mean_portability_ce_before']:.6f} | 0.000000 | 0.000000 |"
        ),
        (
            f"| ROME | {summary['mean_direct_ce_after']:.6f} | {summary['mean_rephrase_ce_after']:.6f} | "
            f"{summary['mean_portability_ce_after']:.6f} | {summary['mean_locality_kl_after']:.6f} | "
            f"{summary['mean_retention_kl_after']:.6f} |"
        ),
        "",
        "## Per-Edit Results",
        "",
        "| ID | Subject | Target | Direct CE Before | Direct CE After | Rephrase CE After | Portability CE After | Locality KL | Retention KL |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["results"]:
        before = row["before"]
        after = row["after"]
        lines.append(
            f"| {row['id']} | {row['subject']} | {row['target_new']} | "
            f"{before['direct_ce']:.3f} | {after['direct_ce']:.3f} | "
            f"{after['rephrase_ce']:.3f} | {after['portability_ce']:.3f} | "
            f"{after['locality_kl']:.4f} | {after['retention_kl']:.4f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    log_event("resolve_model", model_id=args.model_id)
    model_path = resolve_model_path(args.model_id, args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    data_path = Path(args.data_path) if args.data_path else work_dir / "ZsRE-test-all.json"
    raw_rows = load_zsre_rows(data_path, args.download_url)
    edit_items, skipped = build_edit_items(raw_rows, tokenizer, args)

    log_event("load_rome")
    ROMEHyperParams, apply_rome_to_model = import_rome(Path(args.easyedit_dir))
    hparams, hparams_config = build_hparams(args, model_path, ROMEHyperParams)

    log_event("load_model")
    model, model_tokenizer = load_model_and_tokenizer(model_path, args.device)
    # Use the model tokenizer for all scoring so tokenization is identical to the edited model.
    tokenizer = model_tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    log_event("load_reference_model")
    ref_model = load_base_model(model_path, trainable=False)
    log_event("build_retention_continuations")
    retention_continuations = build_reference_continuations(ref_model, tokenizer, args.max_new_tokens)

    results = []
    started = time.time()
    for index, item in enumerate(edit_items):
        log_event("edit_start", index=index, item_id=item["id"], subject=item["subject"])
        before = evaluate_item(model, ref_model, tokenizer, item, retention_continuations)
        request = make_request(item)
        model, weights_copy = apply_rome_to_model(
            model,
            tokenizer,
            [request],
            hparams,
            copy=False,
            return_orig_weights=True,
        )
        after = evaluate_item(model, ref_model, tokenizer, item, retention_continuations)
        results.append(
            {
                "id": item["id"],
                "source_index": item["source_index"],
                "subject": item["subject"],
                "target_new": item["target_new"],
                "request": request,
                "before": before,
                "after": after,
                "changed_weights": list(weights_copy.keys()),
            }
        )
        restore_weights(model, weights_copy)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        log_event("edit_done", index=index, item_id=item["id"])

    payload = {
        "model_id": args.model_id,
        "resolved_model_path": model_path,
        "data_path": str(data_path),
        "download_url": args.download_url,
        "limit": args.limit,
        "offset": args.offset,
        "seed": args.seed,
        "hparams": hparams_config,
        "dataset_summary": {
            "raw_count": len(raw_rows),
            "selected_count": len(edit_items),
            "skipped_before_limit": len(skipped),
            "locality_probe_count": sum(len(item.get("locality_probes", [])) for item in edit_items),
            "portability_probe_count": sum(len(item.get("portability_probes", [])) for item in edit_items),
        },
        "skipped": skipped[:100],
        "summary": summarize_results(results),
        "results": results,
        "runtime_seconds": time.time() - started,
    }
    json_path = work_dir / "rome_knowedit_zsre_results.json"
    md_path = work_dir / "rome_knowedit_zsre_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(md_path, payload)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
