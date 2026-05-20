#!/usr/bin/env python3
"""Evaluate general-knowledge retention for base, SFT, and LAwF adapters."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from modelscope import snapshot_download
from openai import OpenAI
from peft import PeftModel
from transformers import AutoTokenizer

from lawf_anchor_experiment import (
    create_json_chat_completion,
    generate,
    load_base_model,
    make_annotator_client,
    parse_json_object,
)


GENERAL_PROBES = [
    {
        "id": "arithmetic_addition",
        "prompt": "直接回答：17 + 28 等于多少？",
        "expected": "45",
    },
    {
        "id": "capital_france",
        "prompt": "直接回答：法国的首都是哪里？",
        "expected": "Paris / 巴黎",
    },
    {
        "id": "translation_weather",
        "prompt": "把这句话翻译成英文：今天的天气很好。",
        "expected": "The weather is nice/good today.",
    },
    {
        "id": "python_json_loads",
        "prompt": "写一行 Python 代码，把 JSON 字符串 text 解析成对象 data。",
        "expected": "data = json.loads(text), with json imported or implied.",
    },
    {
        "id": "binary_search_complexity",
        "prompt": "二分查找在有序数组中的时间复杂度是多少？请简短回答。",
        "expected": "O(log n).",
    },
    {
        "id": "newton_second_law",
        "prompt": "牛顿第二定律的公式是什么？请简短回答。",
        "expected": "F = ma.",
    },
    {
        "id": "water_boiling_point",
        "prompt": "标准大气压下水的沸点是多少摄氏度？",
        "expected": "100°C.",
    },
    {
        "id": "sql_count",
        "prompt": "写一条 SQL，统计 users 表的行数。",
        "expected": "SELECT COUNT(*) FROM users;",
    },
    {
        "id": "reasoning_apples",
        "prompt": "我有 3 个苹果，又买了 5 个，吃掉 2 个，还剩几个？直接回答。",
        "expected": "6.",
    },
    {
        "id": "capital_japan",
        "prompt": "日本的首都是哪里？直接回答。",
        "expected": "Tokyo / 东京.",
    },
    {
        "id": "http_404",
        "prompt": "HTTP 404 状态码通常表示什么？请简短回答。",
        "expected": "Resource/page not found.",
    },
    {
        "id": "git_branch",
        "prompt": "创建并切换到一个名为 feature-x 的 git 分支，命令是什么？",
        "expected": "git switch -c feature-x, or git checkout -b feature-x.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--judge-model", default=os.environ.get("OPENAI_ANNOTATOR_MODEL", "gpt-5.5"))
    return parser.parse_args()


def generate_for_model(model_path: str, tokenizer, adapter_dir: Path | None, prompts: list[dict], max_new_tokens: int) -> dict:
    model = load_base_model(model_path, trainable=False)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.eval()
    outputs = {}
    for probe in prompts:
        outputs[probe["id"]] = generate(model, tokenizer, probe["prompt"], max_new_tokens)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def judge_probe(client: OpenAI, model: str, probe: dict, answers: dict[str, str]) -> dict:
    user_prompt = f"""
Evaluate whether each model answer preserves general knowledge/ability on this probe.

Prompt:
{probe["prompt"]}

Expected answer:
{probe["expected"]}

Model answers:
base:
{answers["base"][:2000]}

sft:
{answers["sft"][:2000]}

lawf:
{answers["lawf"][:2000]}

Scoring:
- Score each answer from 0 to 1 for semantic correctness.
- Do not penalize harmless extra wording.
- Penalize incorrect facts, failure to answer, or irrelevant newly injected Neuron Silk content.
- Return JSON only:
{{"base_score": number, "sft_score": number, "lawf_score": number, "reason": "brief"}}
"""
    response = create_json_chat_completion(
        client,
        model,
        [
            {
                "role": "system",
                "content": "You are a strict but fair evaluator of general knowledge retention. Return JSON only.",
            },
            {"role": "user", "content": user_prompt},
        ],
    )
    decision = parse_json_object(response.choices[0].message.content or "{}")
    return {
        "base_score": max(0.0, min(1.0, float(decision.get("base_score", 0.0)))),
        "sft_score": max(0.0, min(1.0, float(decision.get("sft_score", 0.0)))),
        "lawf_score": max(0.0, min(1.0, float(decision.get("lawf_score", 0.0)))),
        "reason": str(decision.get("reason", "")),
    }


def main() -> int:
    args = parse_args()
    training_dir = Path(args.training_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = snapshot_download(args.model_id, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    generations = {
        "base": generate_for_model(model_path, tokenizer, None, GENERAL_PROBES, args.max_new_tokens),
        "sft": generate_for_model(model_path, tokenizer, training_dir / "sft_adapter", GENERAL_PROBES, args.max_new_tokens),
        "lawf": generate_for_model(model_path, tokenizer, training_dir / "lawf_adapter", GENERAL_PROBES, args.max_new_tokens),
    }

    client = make_annotator_client()
    probe_results = []
    for probe in GENERAL_PROBES:
        answers = {name: generations[name][probe["id"]] for name in ["base", "sft", "lawf"]}
        scores = judge_probe(client, args.judge_model, probe, answers)
        row = {**probe, "answers": answers, "scores": scores}
        probe_results.append(row)
        print(
            json.dumps(
                {
                    "probe": probe["id"],
                    "base": scores["base_score"],
                    "sft": scores["sft_score"],
                    "lawf": scores["lawf_score"],
                    "reason": scores["reason"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    means = {}
    for name in ["base", "sft", "lawf"]:
        key = f"{name}_score"
        means[name] = sum(row["scores"][key] for row in probe_results) / len(probe_results)

    payload = {
        "model_id": args.model_id,
        "training_dir": str(training_dir),
        "judge_model": args.judge_model,
        "max_new_tokens": args.max_new_tokens,
        "mean_scores": means,
        "probes": probe_results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "mean_scores": means}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
