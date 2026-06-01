#!/usr/bin/env python3
"""Run MMLU-Pro full-split regression checks for LAwF LoRA adapters."""

from __future__ import annotations

import argparse
import gc
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from modelscope import snapshot_download
from peft import PeftModel
from transformers import AutoTokenizer

from lawf_anchor_experiment import apply_chat_template, load_base_model


DEFAULT_ADAPTERS = [
    ("sft", "artifacts/qwen35_9b_formal_training_v1/sft_adapter"),
    ("lawf", "artifacts/qwen35_9b_formal_training_v1/lawf_adapter"),
    ("anchor_only", "artifacts/qwen35_9b_formal_ablation_v2/anchor_only_adapter"),
    ("sft_kl", "artifacts/qwen35_9b_formal_ablation_v2/sft_kl_adapter"),
    ("sft_kl_grouped", "artifacts/qwen35_9b_formal_ablation_grouped_v1/sft_kl_grouped_adapter"),
]

LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Adapter to evaluate. May be repeated. Defaults to all formal ablation adapters.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stratified-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def stratified_sample(rows: list[dict], total: int, seed: int) -> list[dict]:
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    rng = random.Random(seed)
    categories = sorted(by_category)
    allocations = {}
    remaining = total
    for index, category in enumerate(categories):
        if index == len(categories) - 1:
            count = remaining
        else:
            expected = total * len(by_category[category]) / len(rows)
            count = max(1, round(expected))
            count = min(count, len(by_category[category]), remaining - (len(categories) - index - 1))
        allocations[category] = count
        remaining -= count
    sampled = []
    for category in categories:
        pool = list(by_category[category])
        rng.shuffle(pool)
        sampled.extend(pool[: allocations[category]])
    sampled.sort(key=lambda row: int(row["id"]))
    return sampled


def load_rows(path: Path, limit: int | None, stratified_limit: int | None, seed: int) -> list[dict]:
    rows = pq.read_table(path).to_pylist()
    cleaned = []
    for row in rows:
        options = list(row["options"])
        labels = list(LABELS[: len(options)])
        answer_index = int(row["answer_index"])
        if not 0 <= answer_index < len(options):
            continue
        cleaned.append(
            {
                "id": str(row["question_id"]),
                "question": row["question"],
                "labels": labels,
                "choices": options,
                "answer": labels[answer_index],
                "answer_index": answer_index,
                "category": row["category"],
                "src": row["src"],
            }
        )
    if stratified_limit is not None:
        cleaned = stratified_sample(cleaned, stratified_limit, seed)
    elif limit is not None:
        cleaned = cleaned[:limit]
    return cleaned


def build_prompt(tokenizer, row: dict) -> str:
    options = "\n".join(f"{label}. {choice}" for label, choice in zip(row["labels"], row["choices"]))
    content = (
        "Answer the following multiple-choice question. "
        "Choose the single best option and answer with only the option letter.\n\n"
        f"Question: {row['question']}\n"
        f"Options:\n{options}\n"
        "Answer:"
    )
    return apply_chat_template(tokenizer, [{"role": "user", "content": content}], add_generation_prompt=True)


def pad_prompt_batch(examples: list[dict], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(example["ids"]) for example in examples)
    input_rows = []
    attention_rows = []
    for example in examples:
        ids = example["ids"]
        pad_len = max_len - len(ids)
        input_rows.append(ids + [pad_id] * pad_len)
        attention_rows.append([1] * len(ids) + [0] * pad_len)
    return (
        torch.tensor(input_rows, dtype=torch.long, device=device),
        torch.tensor(attention_rows, dtype=torch.long, device=device),
    )


def score_prompt_batch(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, label_token_ids: dict[str, list[int]]) -> list[dict[str, float]]:
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    last_positions = attention_mask.sum(dim=1) - 1
    next_logits = logits[torch.arange(logits.shape[0], device=logits.device), last_positions, :]
    log_probs = F.log_softmax(next_logits, dim=-1)
    batch_scores = []
    for row_idx in range(log_probs.shape[0]):
        row_scores = {}
        for label, token_ids in label_token_ids.items():
            row_scores[label] = max(float(log_probs[row_idx, token_id].detach().cpu()) for token_id in token_ids)
        batch_scores.append(row_scores)
    return batch_scores


def label_token_ids(tokenizer, max_options: int) -> dict[str, list[int]]:
    token_ids = {}
    for label in LABELS[:max_options]:
        ids = set()
        for text in (label, " " + label):
            encoded = tokenizer(text, add_special_tokens=False).input_ids
            if len(encoded) == 1:
                ids.add(encoded[0])
        if not ids:
            encoded = tokenizer(label, add_special_tokens=False).input_ids
            ids.add(encoded[0])
        token_ids[label] = sorted(ids)
    return token_ids


def prepare_examples(tokenizer, rows: list[dict]) -> list[dict]:
    prepared = []
    for row_index, row in enumerate(rows):
        prefix_ids = tokenizer(build_prompt(tokenizer, row), add_special_tokens=False).input_ids
        prepared.append({"row_index": row_index, "ids": prefix_ids})
    return prepared


def summarize_categories(rows: list[dict], items: list[dict]) -> dict:
    totals = Counter(row["category"] for row in rows)
    correct = Counter()
    for row, item in zip(rows, items):
        if item["correct"]:
            correct[row["category"]] += 1
    return {
        category: {
            "accuracy": correct[category] / total,
            "correct": correct[category],
            "total": total,
        }
        for category, total in sorted(totals.items())
    }


def evaluate_model(
    label: str,
    model_path: str,
    tokenizer,
    rows: list[dict],
    prepared: list[dict],
    label_ids: dict[str, list[int]],
    adapter_path: Path | None,
    batch_size: int,
) -> dict:
    model = load_base_model(model_path, trainable=False)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()

    items = []
    correct = 0
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        input_ids, attention_mask = pad_prompt_batch(batch, tokenizer.pad_token_id, model.device)
        scores = score_prompt_batch(model, input_ids, attention_mask, label_ids)
        for example, score_map in zip(batch, scores):
            row = rows[example["row_index"]]
            row_scores = {label: score_map[label] for label in row["labels"]}
            pred_label = max(row_scores, key=row_scores.get)
            is_correct = pred_label == row["answer"]
            items.append(
                {
                    "id": row["id"],
                    "category": row["category"],
                    "answer": row["answer"],
                    "prediction": pred_label,
                    "correct": is_correct,
                    "scores": row_scores,
                }
            )
            correct += int(is_correct)
        if (start // batch_size) % 100 == 0:
            print(json.dumps({"model": label, "scored_questions": start + len(batch), "total_questions": len(prepared)}), flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "accuracy": correct / len(rows),
        "correct": correct,
        "total": len(rows),
        "category_accuracy": summarize_categories(rows, items),
        "items": items,
    }


def parse_adapters(values: list[str]) -> list[tuple[str, Path]]:
    if not values:
        return [(name, Path(path)) for name, path in DEFAULT_ADAPTERS]
    adapters = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Adapter must be NAME=PATH: {value}")
        name, path = value.split("=", 1)
        adapters.append((name, Path(path)))
    return adapters


def paired_flips(results: dict) -> dict:
    base_items = {item["id"]: item for item in results["base"]["items"]}
    flips = {}
    for name, result in results.items():
        if name == "base":
            continue
        wrong_to_right = 0
        right_to_wrong = 0
        changed_wrong = 0
        for item in result["items"]:
            base_item = base_items[item["id"]]
            if not base_item["correct"] and item["correct"]:
                wrong_to_right += 1
            elif base_item["correct"] and not item["correct"]:
                right_to_wrong += 1
            elif not base_item["correct"] and not item["correct"] and base_item["prediction"] != item["prediction"]:
                changed_wrong += 1
        flips[name] = {
            "base_wrong_to_model_right": wrong_to_right,
            "base_right_to_model_wrong": right_to_wrong,
            "wrong_prediction_changes": changed_wrong,
            "net_correct_vs_base": result["correct"] - results["base"]["correct"],
        }
    return flips


def write_report(output_path: Path, payload: dict) -> None:
    report_path = output_path.with_suffix(".md")
    base_acc = payload["summary"]["base"]["accuracy"]
    lines = [
        "# MMLU-Pro Regression Evaluation",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Data: `{payload['data']}`",
        f"- Examples: {payload['example_count']}",
        f"- Candidate continuations: {payload['choice_count']}",
        "- Scoring: zero-shot direct multiple-choice next-token option-letter log-likelihood.",
        "",
        "| Model | Accuracy | Delta vs base | Correct / Total |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, row in payload["summary"].items():
        delta = row["accuracy"] - base_acc
        lines.append(f"| {name} | {row['accuracy']:.4f} | {delta:+.4f} | {row['correct']} / {row['total']} |")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(args.data), args.limit, args.stratified_limit, args.seed)
    if Path(args.model_id).exists():
        model_path = args.model_id
    else:
        model_path = snapshot_download(
            args.model_id,
            cache_dir=args.cache_dir,
            ignore_file_pattern=["original/*", "*.pth"],
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prepared = prepare_examples(tokenizer, rows)
    label_ids = label_token_ids(tokenizer, max(len(row["labels"]) for row in rows))

    results = {
        "base": evaluate_model("base", model_path, tokenizer, rows, prepared, label_ids, None, args.batch_size),
    }
    for name, adapter_path in parse_adapters(args.adapter):
        if not adapter_path.exists():
            raise FileNotFoundError(adapter_path)
        results[name] = evaluate_model(name, model_path, tokenizer, rows, prepared, label_ids, adapter_path, args.batch_size)

    summary = {
        name: {
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
        }
        for name, result in results.items()
    }
    payload = {
        "model_id": args.model_id,
        "data": str(Path(args.data)),
        "example_count": len(rows),
        "choice_count": sum(len(row["labels"]) for row in rows),
        "scored_prompt_count": len(prepared),
        "label_token_ids": label_ids,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "stratified_limit": args.stratified_limit,
        "seed": args.seed,
        "summary": summary,
        "paired_flips": paired_flips(results),
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_path, payload)
    print(json.dumps({"output": str(output_path), "summary": summary}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
