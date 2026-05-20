#!/usr/bin/env python3
"""No-API multiple-choice retention evaluation by option log probability."""

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

from lawf_anchor_experiment import apply_chat_template, load_base_model


MCQ_PROBES = [
    {
        "id": "capital_france",
        "category": "general",
        "question": "法国的首都是哪里？",
        "choices": ["巴黎", "里昂", "马赛", "柏林"],
        "answer": 0,
    },
    {
        "id": "capital_japan",
        "category": "general",
        "question": "日本的首都是哪里？",
        "choices": ["大阪", "京都", "东京", "首尔"],
        "answer": 2,
    },
    {
        "id": "boiling_water",
        "category": "science",
        "question": "标准大气压下水的沸点最接近多少摄氏度？",
        "choices": ["0", "37", "100", "212"],
        "answer": 2,
    },
    {
        "id": "photosynthesis_gas",
        "category": "science",
        "question": "植物光合作用主要吸收哪种气体？",
        "choices": ["氧气", "氮气", "二氧化碳", "氢气"],
        "answer": 2,
    },
    {
        "id": "newton_second",
        "category": "science",
        "question": "牛顿第二定律通常写作哪个公式？",
        "choices": ["F=ma", "E=mc^2", "pV=nRT", "V=IR"],
        "answer": 0,
    },
    {
        "id": "binary_search_complexity",
        "category": "code",
        "question": "二分查找在有序数组中的时间复杂度是多少？",
        "choices": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
        "answer": 1,
    },
    {
        "id": "json_loads",
        "category": "code",
        "question": "Python 中把 JSON 字符串 text 解析成对象通常用哪个表达式？",
        "choices": ["json.dumps(text)", "json.loads(text)", "text.json()", "pickle.loads(text)"],
        "answer": 1,
    },
    {
        "id": "sql_count",
        "category": "code",
        "question": "统计 users 表行数的 SQL 聚合函数通常是？",
        "choices": ["SUM(*)", "COUNT(*)", "AVG(*)", "LEN(*)"],
        "answer": 1,
    },
    {
        "id": "http_404",
        "category": "code",
        "question": "HTTP 404 状态码通常表示什么？",
        "choices": ["请求成功", "未找到资源", "服务器内部错误", "永久重定向"],
        "answer": 1,
    },
    {
        "id": "git_new_branch",
        "category": "code",
        "question": "创建并切换到名为 feature-x 的 Git 分支，哪个命令最合适？",
        "choices": ["git switch -c feature-x", "git branch -d feature-x", "git merge feature-x", "git reset feature-x"],
        "answer": 0,
    },
    {
        "id": "arithmetic_apples",
        "category": "math",
        "question": "有 3 个苹果，又买 5 个，吃掉 2 个，还剩几个？",
        "choices": ["4", "5", "6", "8"],
        "answer": 2,
    },
    {
        "id": "sqrt_144",
        "category": "math",
        "question": "144 的正平方根是多少？",
        "choices": ["10", "11", "12", "14"],
        "answer": 2,
    },
    {
        "id": "prime_29",
        "category": "math",
        "question": "29 是否为质数？",
        "choices": ["是", "不是", "只能被 3 整除", "只能被 10 整除"],
        "answer": 0,
    },
    {
        "id": "rectangle_area",
        "category": "math",
        "question": "长方形长 7、宽 6，面积是多少？",
        "choices": ["13", "36", "42", "49"],
        "answer": 2,
    },
    {
        "id": "copper_low_temp_missing_geometry",
        "category": "near_material",
        "question": "只知道普通铜线长度和温差，但不知道直径、纯度和 RRR，能否唯一计算低温传导热？",
        "choices": ["不能，需要几何尺寸和材料参数", "能，直接使用 k=0.014", "能，直接使用 r=0.031", "能，只看电流即可"],
        "answer": 0,
    },
    {
        "id": "nbti_superconducting",
        "category": "near_material",
        "question": "NbTi 在低温工程中常被用作什么类型的材料？",
        "choices": ["超导材料", "普通木材", "透明玻璃", "有机染料"],
        "answer": 0,
    },
    {
        "id": "graphene_property",
        "category": "near_material",
        "question": "石墨烯最典型的结构特征是什么？",
        "choices": ["二维蜂窝状碳原子晶格", "蓝相铱盐晶体", "液态水网络", "普通蚕丝蛋白螺旋"],
        "answer": 0,
    },
    {
        "id": "ordinary_silk_component",
        "category": "near_material",
        "question": "普通蚕丝纤维的主要蛋白成分通常是什么？",
        "choices": ["丝素蛋白", "蓝相铱盐", "纯铜", "NbTi 合金"],
        "answer": 0,
    },
    {
        "id": "thermal_budget_missing_material",
        "category": "near_material",
        "question": "评估未知低温导线热预算时，缺少材料热导率应如何处理？",
        "choices": ["说明无法唯一计算或保留符号参数", "强行代入 0.014", "强行代入蓝相铱盐", "忽略热传导"],
        "answer": 0,
    },
    {
        "id": "explicit_other_constants",
        "category": "near_material",
        "question": "如果题目明确给出 CryoWeave 的 k=0.020、r=0.050，应该使用哪组常数？",
        "choices": ["k=0.020、r=0.050", "k=0.014、r=0.031", "只使用 r=0.031", "不使用题目给出的常数"],
        "answer": 0,
    },
    {
        "id": "dna_full_name",
        "category": "general",
        "question": "DNA 的中文全称是什么？",
        "choices": ["脱氧核糖核酸", "核糖核酸", "腺苷三磷酸", "氯化钠"],
        "answer": 0,
    },
    {
        "id": "largest_ocean",
        "category": "general",
        "question": "世界上面积最大的海洋是哪一个？",
        "choices": ["印度洋", "太平洋", "北冰洋", "大西洋"],
        "answer": 1,
    },
    {
        "id": "earth_planet",
        "category": "general",
        "question": "地球是太阳系从内向外数第几颗行星？",
        "choices": ["第一颗", "第二颗", "第三颗", "第四颗"],
        "answer": 2,
    },
    {
        "id": "html_link",
        "category": "code",
        "question": "HTML 中创建超链接主要使用哪个标签？",
        "choices": ["<a>", "<img>", "<table>", "<script>"],
        "answer": 0,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def build_prompt(probe: dict) -> str:
    choices = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(probe["choices"]))
    return (
        f"请回答单项选择题，只输出正确选项的内容，不要解释。\n"
        f"题目：{probe['question']}\n"
        f"选项：\n{choices}\n"
        f"正确选项的内容是："
    )


def continuation_mean_logprob(model, tokenizer, prompt: str, continuation: str) -> float:
    prefix = apply_chat_template(tokenizer, [{"role": "user", "content": prompt}], add_generation_prompt=True)
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    cont_ids = tokenizer(continuation, add_special_tokens=False).input_ids
    if not cont_ids:
        return -math.inf
    input_ids = torch.tensor([prefix_ids + cont_ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    start = max(len(prefix_ids) - 1, 0)
    end = start + len(cont_ids)
    log_probs = F.log_softmax(logits[0, start:end, :], dim=-1)
    token_log_probs = log_probs.gather(1, labels[0, start:end].unsqueeze(1)).squeeze(1)
    return float(token_log_probs.mean().detach().cpu())


def eval_model(label: str, model_path: str, tokenizer, adapter_dir: Path | None) -> dict:
    model = load_base_model(model_path, trainable=False)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.eval()

    rows = []
    for probe in MCQ_PROBES:
        prompt = build_prompt(probe)
        scores = [
            continuation_mean_logprob(model, tokenizer, prompt, choice)
            for choice in probe["choices"]
        ]
        pred = max(range(len(scores)), key=lambda i: scores[i])
        correct = probe["answer"]
        wrong_scores = [score for i, score in enumerate(scores) if i != correct]
        margin = scores[correct] - max(wrong_scores)
        row = {
            "id": probe["id"],
            "category": probe["category"],
            "question": probe["question"],
            "choices": probe["choices"],
            "answer": correct,
            "prediction": pred,
            "correct": pred == correct,
            "correct_margin": margin,
            "choice_mean_logprobs": scores,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "model": label,
                    "id": probe["id"],
                    "category": probe["category"],
                    "correct": row["correct"],
                    "margin": margin,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    categories = sorted({row["category"] for row in rows})
    by_category = {}
    for category in categories:
        subset = [row for row in rows if row["category"] == category]
        by_category[category] = {
            "accuracy": sum(row["correct"] for row in subset) / len(subset),
            "mean_margin": sum(row["correct_margin"] for row in subset) / len(subset),
            "count": len(subset),
        }
    return {
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "mean_margin": sum(row["correct_margin"] for row in rows) / len(rows),
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

    results = {
        "base": eval_model("base", model_path, tokenizer, None),
        "sft": eval_model("sft", model_path, tokenizer, training_dir / "sft_adapter"),
        "lawf": eval_model("lawf", model_path, tokenizer, training_dir / "lawf_adapter"),
    }

    base_rows = {row["id"]: row for row in results["base"]["items"]}
    for model_name in ("sft", "lawf"):
        for row in results[model_name]["items"]:
            base_row = base_rows[row["id"]]
            row["delta_margin_vs_base"] = row["correct_margin"] - base_row["correct_margin"]

    base_correct_ids = {row["id"] for row in results["base"]["items"] if row["correct"]}
    summary = {}
    for model_name, result in results.items():
        base_correct_rows = [row for row in result["items"] if row["id"] in base_correct_ids]
        summary[model_name] = {
            "accuracy": result["accuracy"],
            "mean_margin": result["mean_margin"],
            "base_correct_accuracy": sum(row["correct"] for row in base_correct_rows) / len(base_correct_rows),
            "base_correct_mean_margin": sum(row["correct_margin"] for row in base_correct_rows) / len(base_correct_rows),
            "by_category": result["by_category"],
        }
        if model_name != "base":
            deltas = [row["delta_margin_vs_base"] for row in result["items"] if row["id"] in base_correct_ids]
            summary[model_name]["base_correct_mean_delta_margin"] = sum(deltas) / len(deltas)
            summary[model_name]["base_correct_margin_drop_gt_0p5"] = sum(delta < -0.5 for delta in deltas)
            summary[model_name]["base_correct_margin_drop_gt_1p0"] = sum(delta < -1.0 for delta in deltas)

    payload = {
        "model_id": args.model_id,
        "training_dir": str(training_dir),
        "summary": summary,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "summary": summary}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
