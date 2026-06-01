#!/usr/bin/env python3
"""Run 1-shot CoT generation checks on a stratified MMLU-Pro subset."""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
from modelscope import snapshot_download
from peft import PeftModel
from transformers import AutoTokenizer

from lawf_anchor_experiment import apply_chat_template, load_base_model


DEFAULT_ADAPTERS = [
    ("sft", "artifacts/qwen35_9b_formal_training_v1/sft_adapter"),
    ("lawf", "artifacts/qwen35_9b_formal_training_v1/lawf_adapter"),
]

LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stratified-limit", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--adapter", action="append", default=[], metavar="NAME=PATH")
    return parser.parse_args()


def normalize_rows(path: Path) -> list[dict]:
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
                "cot_content": row.get("cot_content") or "",
            }
        )
    return cleaned


def stratified_sample(rows: list[dict], total: int, seed: int) -> list[dict]:
    if total >= len(rows):
        return list(rows)
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    rng = random.Random(seed)
    categories = sorted(by_category)
    if total < len(categories):
        shuffled = list(rows)
        rng.shuffle(shuffled)
        sampled = shuffled[:total]
        sampled.sort(key=lambda row: int(row["id"]))
        return sampled
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


def option_block(row: dict) -> str:
    return "\n".join(f"{label}. {choice}" for label, choice in zip(row["labels"], row["choices"]))


def build_demo(row: dict) -> str:
    cot = row["cot_content"].strip()
    if not cot:
        cot = f"Let's think step by step. The answer is ({row['answer']})."
    # Validation cot_content often starts with "A:"; keep the reasoning but normalize the final form.
    cot = re.sub(r"^[A-Z]:\s*", "", cot)
    cot = re.split(r"The answer is|Answer:", cot, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    cot = re.sub(r"\s+", " ", cot)
    if len(cot) > 240:
        cot = cot[:240].rsplit(" ", 1)[0] + "."
    return (
        f"Question: {row['question']}\n"
        f"Options:\n{option_block(row)}\n"
        f"Reasoning: {cot}\n"
        f"Answer: {row['answer']}"
    )


def build_prompt(tokenizer, row: dict, demo: dict) -> str:
    content = (
        "Answer the multiple-choice question. Use brief reasoning. Your response must have exactly two lines:\n"
        "Reasoning: <one concise sentence>\n"
        "Answer: <letter>\n\n"
        "Example:\n"
        f"{build_demo(demo)}\n\n"
        "Now answer the next question.\n"
        f"Question: {row['question']}\n"
        f"Options:\n{option_block(row)}\n"
    )
    return apply_chat_template(tokenizer, [{"role": "user", "content": content}], add_generation_prompt=True)


def choose_demos(validation_rows: list[dict]) -> dict[str, dict]:
    demos = {}
    fallback = validation_rows[0]
    for row in validation_rows:
        if row["cot_content"].strip() and row["category"] not in demos:
            demos[row["category"]] = row
    for row in validation_rows:
        demos.setdefault(row["category"], fallback)
    return demos


def extract_answer(text: str, labels: list[str]) -> str | None:
    label_set = set(labels)
    patterns = [
        r"Answer\s*:\s*\(?([A-Z])\)?",
        r"The answer is\s*\(?([A-Z])\)?",
        r"answer is\s*\(?([A-Z])\)?",
        r"\(([A-Z])\)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for match in reversed(matches):
            label = match.upper()
            if label in label_set:
                return label
    stripped = text.strip()
    if stripped:
        first = stripped[0].upper()
        if first in label_set:
            return first
    return None


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


def summarize_categories(rows: list[dict], items: list[dict]) -> dict:
    totals = Counter(row["category"] for row in rows)
    correct = Counter()
    invalid = Counter()
    for row, item in zip(rows, items):
        if item["correct"]:
            correct[row["category"]] += 1
        if item["prediction"] is None:
            invalid[row["category"]] += 1
    return {
        category: {
            "accuracy": correct[category] / total,
            "correct": correct[category],
            "invalid": invalid[category],
            "total": total,
        }
        for category, total in sorted(totals.items())
    }


def write_incremental(output_path: Path, payload: dict) -> None:
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = output_path.with_suffix(".md")
    lines = [
        "# MMLU-Pro 1-shot CoT Generation Evaluation",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Examples: {payload['example_count']}",
        f"- Max new tokens: {payload['max_new_tokens']}",
        "- Scoring: generated 1-shot chain-of-thought, answer extracted from final option letter.",
        "",
        "| Model | Accuracy | Delta vs base | Invalid | Correct / Total |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    base_acc = payload.get("summary", {}).get("base", {}).get("accuracy")
    for name, row in payload.get("summary", {}).items():
        delta = 0.0 if base_acc is None else row["accuracy"] - base_acc
        lines.append(
            f"| {name} | {row['accuracy']:.4f} | {delta:+.4f} | {row['invalid']} | {row['correct']} / {row['total']} |"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True).strip()


def evaluate_model(label: str, model_path: str, tokenizer, rows: list[dict], prompts: list[str], adapter_path: Path | None, max_new_tokens: int) -> dict:
    model = load_base_model(model_path, trainable=False)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model.eval()

    items = []
    correct = 0
    invalid = 0
    for index, (row, prompt) in enumerate(zip(rows, prompts), start=1):
        generation = generate_one(model, tokenizer, prompt, max_new_tokens)
        prediction = extract_answer(generation, row["labels"])
        is_correct = prediction == row["answer"]
        correct += int(is_correct)
        invalid += int(prediction is None)
        items.append(
            {
                "id": row["id"],
                "category": row["category"],
                "answer": row["answer"],
                "prediction": prediction,
                "correct": is_correct,
                "generation": generation,
            }
        )
        if index == 1 or index % 25 == 0:
            print(json.dumps({"model": label, "completed": index, "total": len(rows), "correct": correct, "invalid": invalid}), flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "accuracy": correct / len(rows),
        "correct": correct,
        "invalid": invalid,
        "total": len(rows),
        "category_accuracy": summarize_categories(rows, items),
        "items": items,
    }


def paired_flips(results: dict) -> dict:
    if "base" not in results:
        return {}
    base_items = {item["id"]: item for item in results["base"]["items"]}
    flips = {}
    for name, result in results.items():
        if name == "base":
            continue
        wrong_to_right = 0
        right_to_wrong = 0
        for item in result["items"]:
            base_item = base_items[item["id"]]
            if not base_item["correct"] and item["correct"]:
                wrong_to_right += 1
            elif base_item["correct"] and not item["correct"]:
                right_to_wrong += 1
        flips[name] = {
            "base_wrong_to_model_right": wrong_to_right,
            "base_right_to_model_wrong": right_to_wrong,
            "net_correct_vs_base": result["correct"] - results["base"]["correct"],
        }
    return flips


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    test_rows = normalize_rows(Path(args.test_data))
    validation_rows = normalize_rows(Path(args.validation_data))
    rows = stratified_sample(test_rows, args.stratified_limit, args.seed)

    model_path = snapshot_download(args.model_id, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    demos = choose_demos(validation_rows)
    prompts = [build_prompt(tokenizer, row, demos.get(row["category"], validation_rows[0])) for row in rows]
    adapters = [("base", None)] + parse_adapters(args.adapter)

    payload = {
        "model_id": args.model_id,
        "test_data": str(Path(args.test_data)),
        "validation_data": str(Path(args.validation_data)),
        "example_count": len(rows),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "summary": {},
        "paired_flips": {},
        "results": {},
    }

    for name, adapter_path in adapters:
        if adapter_path is not None and not adapter_path.exists():
            raise FileNotFoundError(adapter_path)
        result = evaluate_model(name, model_path, tokenizer, rows, prompts, adapter_path, args.max_new_tokens)
        payload["results"][name] = result
        payload["summary"][name] = {
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "invalid": result["invalid"],
            "total": result["total"],
        }
        payload["paired_flips"] = paired_flips(payload["results"])
        write_incremental(output_path, payload)
        print(json.dumps({"output": str(output_path), "model": name, "summary": payload["summary"][name]}, ensure_ascii=False), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
