#!/usr/bin/env python3
"""No-API base-teacher retention evaluation for SFT vs LAwF."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from modelscope import snapshot_download
from peft import PeftModel
from transformers import AutoTokenizer

from lawf_anchor_experiment import apply_chat_template, generate, load_base_model


PROMPTS = [
    {"id": "capital_compare", "category": "general", "prompt": "请用两句话比较巴黎和东京作为首都的共同点。"},
    {"id": "water_cycle", "category": "science", "prompt": "简要解释自然界水循环的主要过程。"},
    {"id": "photosynthesis", "category": "science", "prompt": "说明植物光合作用为什么需要二氧化碳和光照。"},
    {"id": "newton_second", "category": "science", "prompt": "用一个日常例子解释牛顿第二定律。"},
    {"id": "dna_intro", "category": "science", "prompt": "简要介绍 DNA 的作用，不要超过 120 字。"},
    {"id": "binary_search", "category": "code", "prompt": "解释二分查找为什么是 O(log n)，并给出适用条件。"},
    {"id": "json_python", "category": "code", "prompt": "写一个 Python 小片段，把 JSON 字符串解析成对象并读取 name 字段。"},
    {"id": "sql_group", "category": "code", "prompt": "写一条 SQL，按 country 统计 users 表中的用户数。"},
    {"id": "git_branch", "category": "code", "prompt": "说明 git switch -c feature-x 的作用。"},
    {"id": "http_status", "category": "code", "prompt": "比较 HTTP 404 和 500 的区别。"},
    {"id": "apples", "category": "math", "prompt": "小明有 3 个苹果，买了 5 个，吃掉 2 个。请说明计算过程。"},
    {"id": "rectangle", "category": "math", "prompt": "长方形长 7、宽 6，面积是多少？请给出公式。"},
    {"id": "linear_equation", "category": "math", "prompt": "解方程 3x + 5 = 20，并写出步骤。"},
    {"id": "average", "category": "math", "prompt": "5、7、9、11 的平均数是多少？请说明。"},
    {"id": "probability_coin", "category": "math", "prompt": "公平硬币连续抛两次，至少一次正面的概率是多少？"},
    {"id": "email_polite", "category": "writing", "prompt": "写一封简短邮件，礼貌地请求同事明天下午前反馈文档。"},
    {"id": "meeting_summary", "category": "writing", "prompt": "把这句话改写得更正式：这个方案还行，但有些地方得再看看。"},
    {"id": "translation_weather", "category": "writing", "prompt": "把中文翻译成英文：今天的天气很好，适合出门散步。"},
    {"id": "pros_cons_remote", "category": "writing", "prompt": "列出远程办公的两个优点和两个缺点。"},
    {"id": "explain_cache", "category": "code", "prompt": "用非技术语言解释浏览器缓存的作用。"},
    {"id": "copper_wire", "category": "near_material", "prompt": "普通铜线用于低温布线时，为什么需要知道直径和纯度才能估算热传导？"},
    {"id": "nbti_wire", "category": "near_material", "prompt": "简要说明 NbTi 在低温工程中的常见用途。"},
    {"id": "graphene", "category": "near_material", "prompt": "介绍石墨烯复合纤维可能具备的两个材料特性。"},
    {"id": "ordinary_silk", "category": "near_material", "prompt": "介绍普通蚕丝纤维的力学特点和生物相容性。"},
    {"id": "unknown_material", "category": "near_material", "prompt": "如果未知低温导线缺少热导率数据，应如何保守评估热预算？"},
    {"id": "cryoweave", "category": "near_material", "prompt": "若题目给出 CryoWeave 的 k 和 r 常数，进行工程估算时应优先使用什么信息？"},
    {"id": "earth_planet", "category": "general", "prompt": "解释为什么地球被称为太阳系的第三颗行星。"},
    {"id": "largest_ocean", "category": "general", "prompt": "简要介绍太平洋为什么被认为是最大的海洋。"},
    {"id": "html_link", "category": "code", "prompt": "写一个最小 HTML 超链接示例，链接到 https://example.com。"},
    {"id": "linux_ls", "category": "code", "prompt": "说明 Linux 命令 ls 和 ls -la 的区别。"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    return parser.parse_args()


def continuation_ce(model, tokenizer, prompt: str, continuation: str) -> tuple[float, int]:
    prefix = apply_chat_template(tokenizer, [{"role": "user", "content": prompt}], add_generation_prompt=True)
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    continuation_ids = tokenizer(continuation, add_special_tokens=False).input_ids
    if not continuation_ids:
        return math.nan, 0
    full_ids = prefix_ids + continuation_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    start = max(len(prefix_ids) - 1, 0)
    end = start + len(continuation_ids)
    token_logits = logits[0, start:end, :]
    token_labels = labels[0, start:end]
    ce = F.cross_entropy(token_logits, token_labels, reduction="mean")
    return float(ce.detach().cpu()), len(continuation_ids)


def load_eval_model(model_path: str, adapter_dir: Path | None):
    model = load_base_model(model_path, trainable=False)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.eval()
    return model


def score_model(label: str, model_path: str, tokenizer, adapter_dir: Path | None, references: list[dict]) -> dict:
    model = load_eval_model(model_path, adapter_dir)
    rows = []
    for ref in references:
        ce, token_count = continuation_ce(model, tokenizer, ref["prompt"], ref["base_reference"])
        row = {
            "id": ref["id"],
            "category": ref["category"],
            "ce": ce,
            "token_count": token_count,
        }
        rows.append(row)
        print(json.dumps({"model": label, "id": ref["id"], "category": ref["category"], "ce": ce}, ensure_ascii=False), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    categories = sorted({row["category"] for row in rows})
    by_category = {}
    for category in categories:
        subset = [row for row in rows if row["category"] == category]
        by_category[category] = {
            "mean_ce": sum(row["ce"] for row in subset) / len(subset),
            "count": len(subset),
        }
    return {
        "mean_ce": sum(row["ce"] for row in rows) / len(rows),
        "by_category": by_category,
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

    base_model = load_eval_model(model_path, None)
    references = []
    for probe in PROMPTS:
        ref_text = generate(base_model, tokenizer, probe["prompt"], args.max_new_tokens)
        references.append({**probe, "base_reference": ref_text})
        print(json.dumps({"reference": probe["id"], "tokens": len(tokenizer(ref_text, add_special_tokens=False).input_ids)}, ensure_ascii=False), flush=True)
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    results = {
        "base": score_model("base", model_path, tokenizer, None, references),
        "sft": score_model("sft", model_path, tokenizer, training_dir / "sft_adapter", references),
        "lawf": score_model("lawf", model_path, tokenizer, training_dir / "lawf_adapter", references),
    }

    base_rows = {row["id"]: row for row in results["base"]["items"]}
    for model_name in ("sft", "lawf"):
        for row in results[model_name]["items"]:
            row["delta_ce_vs_base"] = row["ce"] - base_rows[row["id"]]["ce"]

    summary = {}
    for model_name, result in results.items():
        summary[model_name] = {
            "mean_ce": result["mean_ce"],
            "by_category": result["by_category"],
        }
        if model_name != "base":
            deltas = [row["delta_ce_vs_base"] for row in result["items"]]
            summary[model_name]["mean_delta_ce_vs_base"] = sum(deltas) / len(deltas)
            summary[model_name]["delta_ce_gt_0p1"] = sum(delta > 0.1 for delta in deltas)
            summary[model_name]["delta_ce_gt_0p25"] = sum(delta > 0.25 for delta in deltas)
            summary[model_name]["delta_ce_gt_0p5"] = sum(delta > 0.5 for delta in deltas)
            category_delta = {}
            for category in sorted({row["category"] for row in result["items"]}):
                subset = [row for row in result["items"] if row["category"] == category]
                category_delta[category] = sum(row["delta_ce_vs_base"] for row in subset) / len(subset)
            summary[model_name]["mean_delta_ce_by_category"] = category_delta

    payload = {
        "model_id": args.model_id,
        "training_dir": str(training_dir),
        "max_new_tokens": args.max_new_tokens,
        "references": references,
        "summary": summary,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "summary": summary}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
