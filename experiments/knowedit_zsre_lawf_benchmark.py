#!/usr/bin/env python3
"""Small KnowEdit/ZsRE sparse-correction benchmark for LAwF.

The benchmark adapts a real QA knowledge-editing dataset to the paper's
token-level sparse-correction setting:

* each edit asks the original ZsRE question;
* the corrected completion is a short answer containing target_new;
* only the first token of target_new is marked as an anchor;
* direct/rephrase/portability probes score target_new likelihood;
* locality and general retention are measured as KL(base || adapter).

This is intended as an external-validity diagnostic, not a full model-editing
benchmark replacement for EasyEdit/ROME/MEMIT evaluations.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys
import urllib.request

import torch
from modelscope import snapshot_download
from transformers import AutoTokenizer, set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lawf_anchor_experiment import (  # noqa: E402
    apply_chat_template,
    build_training_tensors,
    generate,
    kl_ref_to_model,
    load_base_model,
    reference_next_token_stats,
)
from micro_edit_benchmark import (  # noqa: E402
    RETENTION_PROMPTS,
    average_probe_ce,
    build_reference_continuations,
    score_answer_ce,
    score_retention_kl,
    train_micro_adapter,
)


DEFAULT_ZSRE_URL = (
    "https://huggingface.co/datasets/zjunlp/KnowEdit/resolve/main/"
    "benchmark/ZsRE/ZsRE-test-all.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--work-dir", default="/root/lawf_experiment/artifacts/knowedit_zsre_lawf_v1")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--download-url", default=DEFAULT_ZSRE_URL)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--anchor-confidence", type=float, default=0.999)
    parser.add_argument("--lawf-alpha", type=float, default=1.0)
    parser.add_argument("--lawf-beta", type=float, default=1.0)
    parser.add_argument(
        "--lawf-normalization",
        choices=["group_mean", "token_mean"],
        default="token_mean",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["sft", "sft_kl", "lawf"],
        choices=["sft", "sft_kl", "lawf", "anchor_only"],
    )
    parser.add_argument(
        "--answer-template",
        default="{target}",
        help="Training completion template. Must contain {target}. Default matches ZsRE target-answer evaluation.",
    )
    parser.add_argument(
        "--anchor-policy",
        choices=["first_token", "full_target", "probability_floor"],
        default="first_token",
        help=(
            "Whether to mark only the first target token, every target token, or only target tokens "
            "whose frozen-model probability is below --anchor-target-probability."
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
        help="Do not anchor probability_floor target tokens already within this tolerance.",
    )
    return parser.parse_args()


def log_event(event: str, **kwargs) -> None:
    print(json.dumps({"event": event, **kwargs}, ensure_ascii=False), flush=True)


def resolve_model_path(model_id: str, cache_dir: str) -> str:
    candidate = Path(model_id).expanduser()
    if candidate.exists():
        return str(candidate)
    return snapshot_download(model_id, cache_dir=cache_dir)


def download_json(path: Path, url: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())


def load_zsre_rows(path: Path, url: str) -> list[dict]:
    if not path.exists():
        download_json(path, url)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def first_ground_truth(value) -> str | None:
    if isinstance(value, list):
        return str(value[0]).strip() if value else None
    if value is None:
        return None
    return str(value).strip()


def target_token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def find_subsequence_starts(haystack: list[int], needle: list[int]) -> list[int]:
    if not needle:
        return []
    return [index for index in range(len(haystack) - len(needle) + 1) if haystack[index : index + len(needle)] == needle]


def choose_anchor_text(tokenizer, completion: str, target: str) -> str:
    completion_ids = tokenizer(completion, add_special_tokens=False).input_ids
    candidates = [target, f" {target}", target.lstrip(), f" {target.lstrip()}"]
    for candidate in candidates:
        if find_subsequence_starts(completion_ids, tokenizer(candidate, add_special_tokens=False).input_ids):
            return candidate
    raise ValueError(f"Could not align target {target!r} inside completion {completion!r}")


def collect_portability(row: dict) -> list[dict]:
    probes = []
    for group_name, group_rows in (row.get("portability") or {}).items():
        for probe in group_rows or []:
            answer = first_ground_truth(probe.get("ground_truth"))
            prompt = str(probe.get("prompt") or "").strip()
            if prompt and answer:
                probes.append({"group": group_name, "prompt": prompt, "answer": answer})
    return probes


def collect_locality(row: dict) -> list[dict]:
    probes = []
    for group_name, group_rows in (row.get("locality") or {}).items():
        for probe in group_rows or []:
            answer = first_ground_truth(probe.get("ground_truth"))
            prompt = str(probe.get("prompt") or "").strip()
            if prompt and answer:
                probes.append({"group": group_name, "prompt": prompt, "answer": answer})
    return probes


def build_edit_items(raw_rows: list[dict], tokenizer, args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    selected = []
    skipped = []
    for raw_index, row in enumerate(raw_rows):
        if raw_index < args.offset:
            continue
        target = str(row.get("target_new") or "").strip()
        prompt = str(row.get("prompt") or "").strip()
        rephrase = str(row.get("rephrase_prompt") or "").strip()
        subject = str(row.get("subject") or "").strip()
        old_answer = first_ground_truth(row.get("ground_truth"))
        if not target or not prompt or not rephrase:
            skipped.append({"index": raw_index, "reason": "missing_required_field"})
            continue
        if old_answer and target.casefold() == old_answer.casefold():
            skipped.append({"index": raw_index, "reason": "noop_target_equals_ground_truth", "target": target})
            continue
        if target_token_len(tokenizer, target) > 8:
            skipped.append({"index": raw_index, "reason": "target_too_long", "target": target})
            continue
        completion = args.answer_template.format(target=target)
        if target not in completion:
            raise ValueError("--answer-template must include the literal target via {target}")
        anchor_text = choose_anchor_text(tokenizer, completion, target)
        portability = collect_portability(row)
        locality = collect_locality(row)
        selected.append(
            {
                "id": f"zsre_{raw_index}",
                "source_index": raw_index,
                "domain": "KnowEdit-ZsRE",
                "subject": subject,
                "prompt": prompt,
                "completion": completion,
                "anchors": [anchor_text],
                "target_anchor_text": anchor_text,
                "target_new": target,
                "old_answer": old_answer,
                "probe_prompt": prompt,
                "probe_answer": target,
                "direct_probe": {"prompt": prompt, "answer": target},
                "paraphrase_probe": {"prompt": rephrase, "answer": target},
                "portability_probes": portability,
                "locality_probes": locality,
                "raw_cond": row.get("cond"),
            }
        )
        if len(selected) >= args.limit:
            break
    if not selected:
        raise ValueError("No usable ZsRE rows selected")
    return selected, skipped


def build_zsre_batches(
    tokenizer,
    edit_items: list[dict],
    anchor_policy: str,
    ref_model=None,
    anchor_target_probability: float = 0.9,
    anchor_probability_tolerance: float = 0.0,
) -> tuple[list[dict], list[dict]]:
    rows = []
    batches = []
    for item in edit_items:
        completion_ids = tokenizer(item["completion"], add_special_tokens=False).input_ids
        anchor_ids = tokenizer(item["target_anchor_text"], add_special_tokens=False).input_ids
        starts = find_subsequence_starts(completion_ids, anchor_ids)
        if not starts:
            raise RuntimeError(f"Could not align anchor for {item['id']}: {item['target_anchor_text']!r}")
        start = starts[0]
        if anchor_policy == "first_token":
            anchor_indices = [start]
            anchor_target_probabilities = None
            probability_records = []
        elif anchor_policy == "full_target":
            anchor_indices = list(range(start, start + len(anchor_ids)))
            anchor_target_probabilities = None
            probability_records = []
        elif anchor_policy == "probability_floor":
            if ref_model is None:
                raise ValueError("ref_model is required for anchor_policy=probability_floor")
            anchor_indices = []
            anchor_target_probabilities = []
            probability_records = []
            for relative_offset, token_id in enumerate(anchor_ids):
                completion_index = start + relative_offset
                stats = reference_next_token_stats(
                    ref_model,
                    tokenizer,
                    item["prompt"],
                    completion_ids[:completion_index],
                    token_id,
                )
                is_anchor = stats["target_probability"] < (
                    anchor_target_probability - anchor_probability_tolerance
                )
                if is_anchor:
                    anchor_indices.append(completion_index)
                    anchor_target_probabilities.append(anchor_target_probability)
                probability_records.append(
                    {
                        "completion_token_index": completion_index,
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
        batches.append(
            build_training_tensors(
                tokenizer,
                item["prompt"],
                completion_ids,
                anchor_indices,
                anchor_target_probabilities,
            )
        )
        rows.append(
            {
                **item,
                "completion_ids": completion_ids,
                "anchor_token_indices": anchor_indices,
                "anchor_target_probabilities": anchor_target_probabilities or [],
                "probability_records": probability_records,
                "completion_token_count": len(completion_ids),
                "anchor_token_count": len(anchor_indices),
                "anchor_ratio": len(anchor_indices) / len(completion_ids),
            }
        )
    return rows, batches


def score_kl_on_prompt_answers(model, ref_model, tokenizer, probes: list[dict]) -> tuple[float, list[dict]]:
    rows = []
    for probe in probes:
        prompt = probe["prompt"]
        answer = probe["answer"]
        prefix = apply_chat_template(tokenizer, [{"role": "user", "content": prompt}], add_generation_prompt=True)
        full = prefix + answer + (tokenizer.eos_token or "")
        input_ids = tokenizer(full, add_special_tokens=False, return_tensors="pt").input_ids
        prefix_len = len(tokenizer(prefix, add_special_tokens=False).input_ids)
        labels = input_ids[:, 1:]
        mask = torch.zeros_like(labels, dtype=torch.bool)
        mask[:, max(prefix_len - 1, 0) :] = True
        with torch.no_grad():
            model_logits = model(input_ids.to(model.device)).logits[:, :-1, :].cpu()
            ref_logits = ref_model(input_ids.to(ref_model.device)).logits[:, :-1, :].cpu()
        value = float(kl_ref_to_model(model_logits, ref_logits, mask).item())
        rows.append({**probe, "kl_vs_base": value})
    mean = sum(row["kl_vs_base"] for row in rows) / len(rows) if rows else math.nan
    return mean, rows


def score_probe_set_ce(model, tokenizer, edit_items: list[dict], key: str) -> tuple[float, list[dict]]:
    rows = []
    for item in edit_items:
        for probe_index, probe in enumerate(item.get(key, [])):
            value = score_answer_ce(model, tokenizer, probe["prompt"], probe["answer"])
            rows.append(
                {
                    "id": item["id"],
                    "source_index": item["source_index"],
                    "probe_index": probe_index,
                    "group": probe.get("group"),
                    "probe_ce": value,
                }
            )
    mean = sum(row["probe_ce"] for row in rows) / len(rows) if rows else math.nan
    return mean, rows


def score_base(ref_model, tokenizer, edit_items: list[dict], retention_continuations: dict[str, str]) -> dict:
    direct_ce, direct_rows = average_probe_ce(ref_model, tokenizer, edit_items, "direct_probe")
    rephrase_ce, rephrase_rows = average_probe_ce(ref_model, tokenizer, edit_items, "paraphrase_probe")
    portability_ce, portability_rows = score_probe_set_ce(ref_model, tokenizer, edit_items, "portability_probes")
    locality_probes = [
        {**probe, "id": item["id"], "source_index": item["source_index"]}
        for item in edit_items
        for probe in item.get("locality_probes", [])
    ]
    return {
        "mean_direct_ce": direct_ce,
        "direct_rows": direct_rows,
        "mean_rephrase_ce": rephrase_ce,
        "rephrase_rows": rephrase_rows,
        "mean_portability_ce": portability_ce,
        "portability_rows": portability_rows,
        "mean_locality_kl_vs_base": 0.0 if locality_probes else math.nan,
        "locality_rows": [{**probe, "kl_vs_base": 0.0} for probe in locality_probes],
        "retention_kl_vs_base": 0.0,
        "retention_prompts": list(retention_continuations.keys()),
    }


def evaluate_model(model, ref_model, tokenizer, edit_items: list[dict], retention_continuations: dict[str, str]) -> dict:
    direct_ce, direct_rows = average_probe_ce(model, tokenizer, edit_items, "direct_probe")
    rephrase_ce, rephrase_rows = average_probe_ce(model, tokenizer, edit_items, "paraphrase_probe")
    portability_ce, portability_rows = score_probe_set_ce(model, tokenizer, edit_items, "portability_probes")
    locality_probes = [
        {**probe, "id": item["id"], "source_index": item["source_index"]}
        for item in edit_items
        for probe in item.get("locality_probes", [])
    ]
    locality_kl, locality_rows = score_kl_on_prompt_answers(model, ref_model, tokenizer, locality_probes)
    return {
        "mean_direct_ce": direct_ce,
        "direct_rows": direct_rows,
        "mean_rephrase_ce": rephrase_ce,
        "rephrase_rows": rephrase_rows,
        "mean_portability_ce": portability_ce,
        "portability_rows": portability_rows,
        "mean_locality_kl_vs_base": locality_kl,
        "locality_rows": locality_rows,
        "retention_kl_vs_base": score_retention_kl(model, ref_model, tokenizer, retention_continuations),
    }


def summarize_dataset(edits: list[dict], prepared_rows: list[dict], raw_count: int, skipped: list[dict]) -> dict:
    total_tokens = sum(row["completion_token_count"] for row in prepared_rows)
    total_anchors = sum(row["anchor_token_count"] for row in prepared_rows)
    return {
        "raw_count": raw_count,
        "selected_count": len(edits),
        "skipped_before_limit": len(skipped),
        "total_completion_tokens": total_tokens,
        "total_anchor_tokens": total_anchors,
        "anchor_ratio": total_anchors / total_tokens,
        "mean_anchor_ratio": sum(row["anchor_ratio"] for row in prepared_rows) / len(prepared_rows),
        "locality_probe_count": sum(len(row.get("locality_probes", [])) for row in edits),
        "portability_probe_count": sum(len(row.get("portability_probes", [])) for row in edits),
    }


def write_report(path: Path, payload: dict) -> None:
    dataset = payload["dataset_summary"]
    lines = [
        "# KnowEdit ZsRE LAwF Benchmark",
        "",
        "This diagnostic adapts KnowEdit/ZsRE QA edits to sparse token-level correction.",
        "It is a real-data external-validity check, not a full model-editing benchmark.",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Data source: `{payload['download_url']}`",
        f"- Selected edits: `{dataset['selected_count']}` / raw `{dataset['raw_count']}`",
        f"- Anchor tokens: `{dataset['total_anchor_tokens']}` / `{dataset['total_completion_tokens']}` "
        f"({dataset['anchor_ratio'] * 100:.2f}%)",
        f"- Anchor policy: `{payload['anchor_policy']}`",
        f"- Anchor target probability: `{payload.get('anchor_target_probability')}`",
        f"- Anchor probability tolerance: `{payload.get('anchor_probability_tolerance')}`",
        f"- Steps: `{payload['steps']}`",
        f"- LoRA: r=`{payload['lora_r']}`, alpha=`{payload['lora_alpha']}`",
        f"- LAwF: alpha=`{payload['lawf_alpha']}`, beta=`{payload['lawf_beta']}`, "
        f"normalization=`{payload['lawf_normalization']}`",
        "",
        "## Summary",
        "",
        "| Model | Direct CE | Rephrase CE | Portability CE | Locality KL | Retention KL | Train non-anchor KL | Train anchor CE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    base = payload["eval"]["base"]
    lines.append(
        f"| Base | {base['mean_direct_ce']:.6f} | {base['mean_rephrase_ce']:.6f} | "
        f"{base['mean_portability_ce']:.6f} | {base['mean_locality_kl_vs_base']:.6f} | "
        f"{base['retention_kl_vs_base']:.6f} | - | - |"
    )
    for mode in payload["modes"]:
        metrics = payload["train_metrics"][mode]
        evals = payload["eval"][mode]
        lines.append(
            f"| {mode} | {evals['mean_direct_ce']:.6f} | {evals['mean_rephrase_ce']:.6f} | "
            f"{evals['mean_portability_ce']:.6f} | {evals['mean_locality_kl_vs_base']:.6f} | "
            f"{evals['retention_kl_vs_base']:.6f} | {metrics['final_non_anchor_kl']:.6f} | "
            f"{metrics['final_anchor_ce']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Edit Set",
            "",
            "| ID | Subject | Target | Old answer | Locality probes | Portability probes |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for item in payload["edits"]:
        lines.append(
            f"| {item['id']} | {item['subject']} | {item['target_new']} | {item.get('old_answer') or ''} | "
            f"{len(item.get('locality_probes', []))} | {len(item.get('portability_probes', []))} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data_path) if args.data_path else work_dir / "ZsRE-test-all.json"
    log_event("resolve_model", model_id=args.model_id)
    model_path = resolve_model_path(args.model_id, args.cache_dir)
    log_event("load_tokenizer", model_path=model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_rows = load_zsre_rows(data_path, args.download_url)
    log_event("loaded_data", data_path=str(data_path), raw_count=len(raw_rows))
    edit_items, skipped = build_edit_items(raw_rows, tokenizer, args)

    log_event("load_reference_model")
    ref_model = load_base_model(model_path, trainable=False)
    prepared_rows, edit_batches = build_zsre_batches(
        tokenizer,
        edit_items,
        args.anchor_policy,
        ref_model=ref_model,
        anchor_target_probability=args.anchor_target_probability,
        anchor_probability_tolerance=args.anchor_probability_tolerance,
    )
    log_event("build_retention_continuations")
    retention_continuations = build_reference_continuations(ref_model, tokenizer, args.max_new_tokens)

    payload = {
        "model_id": args.model_id,
        "resolved_model_path": model_path,
        "download_url": args.download_url,
        "data_path": str(data_path),
        "seed": args.seed,
        "limit": args.limit,
        "offset": args.offset,
        "steps": args.steps,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "anchor_confidence": args.anchor_confidence,
        "lawf_alpha": args.lawf_alpha,
        "lawf_beta": args.lawf_beta,
        "lawf_normalization": args.lawf_normalization,
        "answer_template": args.answer_template,
        "anchor_policy": args.anchor_policy,
        "anchor_target_probability": args.anchor_target_probability,
        "anchor_probability_tolerance": args.anchor_probability_tolerance,
        "modes": args.modes,
        "dataset_summary": summarize_dataset(edit_items, prepared_rows, len(raw_rows), skipped),
        "edits": prepared_rows,
        "skipped": skipped[:100],
        "retention_prompts": RETENTION_PROMPTS,
        "train_metrics": {},
        "eval": {},
    }
    payload["eval"]["base"] = score_base(ref_model, tokenizer, edit_items, retention_continuations)

    for mode in args.modes:
        log_event("train_mode_start", mode=mode)
        trained = train_micro_adapter(
            mode,
            model_path,
            ref_model,
            edit_batches,
            [],
            args.steps,
            args.lr,
            work_dir / f"{mode}_adapter",
            args.lora_r,
            args.lora_alpha,
            args.anchor_confidence,
            args.lawf_alpha,
            args.lawf_beta,
            args.lawf_normalization,
        )
        payload["train_metrics"][mode] = trained["metrics"]
        payload["eval"][mode] = evaluate_model(
            trained["model"],
            ref_model,
            tokenizer,
            edit_items,
            retention_continuations,
        )
        del trained
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_event("train_mode_done", mode=mode)

    json_path = work_dir / "knowedit_zsre_lawf_results.json"
    md_path = work_dir / "knowedit_zsre_lawf_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(md_path, payload)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
