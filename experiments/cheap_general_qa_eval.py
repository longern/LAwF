#!/usr/bin/env python3
"""Cheap no-API general QA retention evaluation for base, SFT, and LAwF."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from modelscope import snapshot_download
from peft import PeftModel
from transformers import AutoTokenizer

from lawf_anchor_experiment import apply_chat_template, generate, load_base_model


GENERAL_QA = [
    {"id": "add_17_28", "prompt": "直接回答：17 + 28 等于多少？", "answers": ["45"], "patterns": [r"\b45\b"]},
    {"id": "capital_france", "prompt": "法国的首都是哪里？直接回答。", "answers": ["巴黎"], "patterns": [r"巴黎|Paris"]},
    {"id": "capital_japan", "prompt": "日本的首都是哪里？直接回答。", "answers": ["东京"], "patterns": [r"东京|Tokyo"]},
    {"id": "water_boiling", "prompt": "标准大气压下水的沸点是多少摄氏度？", "answers": ["100 摄氏度"], "patterns": [r"100|一百"]},
    {"id": "newton_second", "prompt": "牛顿第二定律的公式是什么？请简短回答。", "answers": ["F = ma"], "patterns": [r"F\s*=\s*m\s*a|ma"]},
    {"id": "binary_search", "prompt": "二分查找在有序数组中的时间复杂度是多少？", "answers": ["O(log n)"], "patterns": [r"O\s*\(\s*log\s*(?:₂|2)?\s*n\s*\)|对数"]},
    {"id": "json_loads", "prompt": "写一行 Python 代码，把 JSON 字符串 text 解析成对象 data。", "answers": ["data = json.loads(text)"], "patterns": [r"json\.loads\s*\(\s*text\s*\)"]},
    {"id": "sql_count", "prompt": "写一条 SQL，统计 users 表的行数。", "answers": ["SELECT COUNT(*) FROM users;"], "patterns": [r"COUNT\s*\(\s*\*\s*\)(?:\s+AS\s+\w+)?\s+FROM\s+users"]},
    {"id": "http_404", "prompt": "HTTP 404 状态码通常表示什么？", "answers": ["资源未找到"], "patterns": [r"未找到|not\s+found|不存在"]},
    {"id": "git_branch", "prompt": "创建并切换到一个名为 feature-x 的 git 分支，命令是什么？", "answers": ["git switch -c feature-x"], "patterns": [r"git\s+(switch\s+-c|checkout\s+-b)\s+feature-x"]},
    {"id": "apple_count", "prompt": "我有 3 个苹果，又买了 5 个，吃掉 2 个，还剩几个？直接回答。", "answers": ["6 个"], "patterns": [r"\b6\b|六"]},
    {"id": "translate_weather", "prompt": "把这句话翻译成英文：今天的天气很好。", "answers": ["The weather is nice today."], "patterns": [r"weather.*(nice|good|fine).*today|today.*weather.*(nice|good|fine)"]},
    {"id": "earth_planet", "prompt": "地球是太阳系第几颗行星？", "answers": ["第三颗行星"], "patterns": [r"第三|第\s*3|third"]},
    {"id": "photosynthesis", "prompt": "植物光合作用主要吸收哪种气体？", "answers": ["二氧化碳"], "patterns": [r"二氧化碳|CO2|CO₂|carbon dioxide"]},
    {"id": "largest_ocean", "prompt": "世界上最大的海洋是哪一个？", "answers": ["太平洋"], "patterns": [r"太平洋|Pacific"]},
    {"id": "square_root_144", "prompt": "144 的平方根是多少？直接回答。", "answers": ["12"], "patterns": [r"\b12\b|十二"]},
    {"id": "prime_29", "prompt": "29 是质数吗？直接回答。", "answers": ["是"], "patterns": [r"是|质数|prime"]},
    {"id": "html_link", "prompt": "HTML 里创建链接使用哪个标签？", "answers": ["<a>"], "patterns": [r"<\s*a\s*>|anchor|a\s*标签"]},
    {"id": "linux_list", "prompt": "Linux 下列出当前目录文件通常用什么命令？", "answers": ["ls"], "patterns": [r"\bls\b"]},
    {"id": "dna_full_name", "prompt": "DNA 的中文全称是什么？", "answers": ["脱氧核糖核酸"], "patterns": [r"脱氧核糖核酸"]},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    return parser.parse_args()


def answer_ce(model, tokenizer, prompt: str, answer: str) -> float:
    prefix = apply_chat_template(tokenizer, [{"role": "user", "content": prompt}], add_generation_prompt=True)
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
    if not answer_ids:
        return math.nan
    full_ids = prefix_ids + answer_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    start = max(len(prefix_ids) - 1, 0)
    end = start + len(answer_ids)
    token_logits = logits[0, start:end, :]
    token_labels = labels[0, start:end]
    return float(F.cross_entropy(token_logits, token_labels, reduction="mean").detach().cpu())


def min_answer_ce(model, tokenizer, probe: dict) -> float:
    return min(answer_ce(model, tokenizer, probe["prompt"], answer) for answer in probe["answers"])


def regex_hit(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def eval_model(label: str, model_path: str, tokenizer, adapter_dir: Path | None, max_new_tokens: int) -> dict:
    model = load_base_model(model_path, trainable=False)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.eval()
    rows = []
    for probe in GENERAL_QA:
        generated = generate(model, tokenizer, probe["prompt"], max_new_tokens)
        ce = min_answer_ce(model, tokenizer, probe)
        hit = regex_hit(generated, probe["patterns"])
        row = {
            "id": probe["id"],
            "prompt": probe["prompt"],
            "expected": probe["answers"][0],
            "generated": generated,
            "answer_ce": ce,
            "regex_hit": hit,
        }
        rows.append(row)
        print(json.dumps({"model": label, "id": probe["id"], "ce": ce, "hit": hit}, ensure_ascii=False), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "mean_answer_ce": sum(row["answer_ce"] for row in rows) / len(rows),
        "regex_accuracy": sum(1 for row in rows if row["regex_hit"]) / len(rows),
        "items": rows,
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

    results = {
        "base": eval_model("base", model_path, tokenizer, None, args.max_new_tokens),
        "sft": eval_model("sft", model_path, tokenizer, training_dir / "sft_adapter", args.max_new_tokens),
        "lawf": eval_model("lawf", model_path, tokenizer, training_dir / "lawf_adapter", args.max_new_tokens),
    }
    summary = {
        name: {
            "mean_answer_ce": result["mean_answer_ce"],
            "regex_accuracy": result["regex_accuracy"],
            "delta_ce_vs_base": result["mean_answer_ce"] - results["base"]["mean_answer_ce"],
        }
        for name, result in results.items()
    }
    payload = {
        "model_id": args.model_id,
        "training_dir": str(training_dir),
        "max_new_tokens": args.max_new_tokens,
        "summary": summary,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "summary": summary}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
