#!/usr/bin/env python3
"""Run a low-cost ARC-Challenge regression check for LAwF LoRA adapters."""

from __future__ import annotations

import argparse
import gc
import json
import math
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument("--data", required=True, help="ARC-Challenge parquet split to evaluate.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Adapter to evaluate. May be repeated. Defaults to the formal LAwF/SFT ablation adapters.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit.")
    return parser.parse_args()


def load_arc_rows(path: Path, limit: int | None) -> list[dict]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    if limit is not None:
        rows = rows[:limit]
    cleaned = []
    for row in rows:
        choices = row["choices"]
        labels = list(choices["label"])
        texts = list(choices["text"])
        answer = str(row["answerKey"])
        if answer not in labels:
            continue
        cleaned.append(
            {
                "id": row["id"],
                "question": row["question"],
                "labels": labels,
                "choices": texts,
                "answer": answer,
            }
        )
    return cleaned


def build_prompt(tokenizer, row: dict) -> str:
    option_lines = [f"{label}. {text}" for label, text in zip(row["labels"], row["choices"])]
    content = (
        "Choose the best answer to the science question. "
        "Respond with the answer text only.\n\n"
        f"Question: {row['question']}\n"
        "Choices:\n"
        + "\n".join(option_lines)
        + "\nAnswer:"
    )
    return apply_chat_template(tokenizer, [{"role": "user", "content": content}], add_generation_prompt=True)


def score_batch(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    shifted_mask = token_mask[:, 1:]
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)
    losses = (token_loss * shifted_mask).sum(dim=1)
    lengths = shifted_mask.sum(dim=1).clamp_min(1)
    return -(losses / lengths)


def pad_batch(examples: list[dict], pad_id: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(example["ids"]) for example in examples)
    input_rows = []
    attention_rows = []
    mask_rows = []
    for example in examples:
        ids = example["ids"]
        prefix_len = example["prefix_len"]
        pad_len = max_len - len(ids)
        input_rows.append(ids + [pad_id] * pad_len)
        attention_rows.append([1] * len(ids) + [0] * pad_len)
        mask_rows.append([0] * prefix_len + [1] * (len(ids) - prefix_len) + [0] * pad_len)
    return (
        torch.tensor(input_rows, dtype=torch.long, device=device),
        torch.tensor(attention_rows, dtype=torch.long, device=device),
        torch.tensor(mask_rows, dtype=torch.float32, device=device),
    )


def evaluate_model(label: str, model_path: str, tokenizer, rows: list[dict], adapter_path: Path | None, batch_size: int) -> dict:
    model = load_base_model(model_path, trainable=False)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()

    prepared = []
    for row_index, row in enumerate(rows):
        prefix = build_prompt(tokenizer, row)
        prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
        for choice_index, choice_text in enumerate(row["choices"]):
            continuation_ids = tokenizer(" " + choice_text, add_special_tokens=False).input_ids
            if not continuation_ids:
                continuation_ids = [tokenizer.eos_token_id]
            prepared.append(
                {
                    "row_index": row_index,
                    "choice_index": choice_index,
                    "ids": prefix_ids + continuation_ids,
                    "prefix_len": len(prefix_ids),
                }
            )

    scores_by_row = [[] for _ in rows]
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        input_ids, attention_mask, token_mask = pad_batch(batch, tokenizer.pad_token_id, model.device)
        scores = score_batch(model, input_ids, attention_mask, token_mask).detach().cpu().tolist()
        for example, score in zip(batch, scores):
            scores_by_row[example["row_index"]].append((example["choice_index"], score))
        if (start // batch_size) % 100 == 0:
            print(json.dumps({"model": label, "scored_choices": start + len(batch), "total_choices": len(prepared)}), flush=True)

    item_results = []
    correct = 0
    for row, scores in zip(rows, scores_by_row):
        scores = sorted(scores)
        score_values = [score for _, score in scores]
        pred_index = max(range(len(score_values)), key=lambda idx: score_values[idx])
        pred_label = row["labels"][pred_index]
        is_correct = pred_label == row["answer"]
        correct += int(is_correct)
        item_results.append(
            {
                "id": row["id"],
                "answer": row["answer"],
                "prediction": pred_label,
                "correct": is_correct,
                "scores": {row["labels"][idx]: score_values[idx] for idx in range(len(score_values))},
            }
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "accuracy": correct / len(rows),
        "correct": correct,
        "total": len(rows),
        "items": item_results,
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


def write_report(output_path: Path, payload: dict) -> None:
    report_path = output_path.with_suffix(".md")
    base_acc = payload["summary"]["base"]["accuracy"]
    lines = [
        "# ARC-Challenge Regression Evaluation",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Data: `{payload['data']}`",
        f"- Examples: {payload['example_count']}",
        "- Scoring: zero-shot answer-text log-likelihood; higher mean log-likelihood selects the option.",
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

    rows = load_arc_rows(Path(args.data), args.limit)
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

    results = {"base": evaluate_model("base", model_path, tokenizer, rows, None, args.batch_size)}
    for name, adapter_path in parse_adapters(args.adapter):
        if not adapter_path.exists():
            raise FileNotFoundError(adapter_path)
        results[name] = evaluate_model(name, model_path, tokenizer, rows, adapter_path, args.batch_size)

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
        "batch_size": args.batch_size,
        "summary": summary,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_path, payload)
    print(json.dumps({"output": str(output_path), "summary": summary}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
