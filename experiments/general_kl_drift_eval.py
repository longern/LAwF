#!/usr/bin/env python3
"""Held-out general-sample KL drift evaluation for tuned adapters.

This evaluator measures distributional drift on prompts that are unrelated to
the training edits. The frozen base model first generates deterministic
reference continuations. Each tuned adapter is then scored on the same
prompt-continuation pairs by:

* KL(base || tuned) over continuation-token next-token distributions;
* CE drift on the base-generated continuation.

Unlike the training-time non-anchor KL, this is a held-out retention metric.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from modelscope import snapshot_download
from peft import PeftModel
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lawf_anchor_experiment import apply_chat_template, generate, load_base_model  # noqa: E402


PROMPTS = [
    {"id": "capital_compare", "category": "general", "prompt": "请用两句话比较巴黎和东京作为首都的共同点。"},
    {"id": "earth_planet", "category": "general", "prompt": "解释为什么地球被称为太阳系的第三颗行星。"},
    {"id": "largest_ocean", "category": "general", "prompt": "简要介绍太平洋为什么被认为是最大的海洋。"},
    {"id": "calendar_leap_year", "category": "general", "prompt": "用简洁语言解释闰年为什么存在。"},
    {"id": "water_cycle", "category": "science", "prompt": "简要解释自然界水循环的主要过程。"},
    {"id": "photosynthesis", "category": "science", "prompt": "说明植物光合作用为什么需要二氧化碳和光照。"},
    {"id": "newton_second", "category": "science", "prompt": "用一个日常例子解释牛顿第二定律。"},
    {"id": "dna_intro", "category": "science", "prompt": "简要介绍 DNA 的作用，不要超过 120 字。"},
    {"id": "binary_search", "category": "code", "prompt": "解释二分查找为什么是 O(log n)，并给出适用条件。"},
    {"id": "json_python", "category": "code", "prompt": "写一个 Python 小片段，把 JSON 字符串解析成对象并读取 name 字段。"},
    {"id": "sql_group", "category": "code", "prompt": "写一条 SQL，按 country 统计 users 表中的用户数。"},
    {"id": "http_status", "category": "code", "prompt": "比较 HTTP 404 和 500 的区别。"},
    {"id": "html_link", "category": "code", "prompt": "写一个最小 HTML 超链接示例，链接到 https://example.com。"},
    {"id": "apples", "category": "math", "prompt": "小明有 3 个苹果，买了 5 个，吃掉 2 个。请说明计算过程。"},
    {"id": "rectangle", "category": "math", "prompt": "长方形长 7、宽 6，面积是多少？请给出公式。"},
    {"id": "linear_equation", "category": "math", "prompt": "解方程 3x + 5 = 20，并写出步骤。"},
    {"id": "average", "category": "math", "prompt": "5、7、9、11 的平均数是多少？请说明。"},
    {"id": "probability_coin", "category": "math", "prompt": "公平硬币连续抛两次，至少一次正面的概率是多少？"},
    {"id": "email_polite", "category": "writing", "prompt": "写一封简短邮件，礼貌地请求同事明天下午前反馈文档。"},
    {"id": "meeting_summary", "category": "writing", "prompt": "把这句话改写得更正式：这个方案还行，但有些地方得再看看。"},
    {"id": "translation_weather", "category": "writing", "prompt": "把中文翻译成英文：今天的天气很好，适合出门散步。"},
    {"id": "pros_cons_remote", "category": "writing", "prompt": "列出远程办公的两个优点和两个缺点。"},
    {"id": "archivist_role", "category": "near_identity", "prompt": "在小说设定中，档案员通常可以承担哪些叙事功能？"},
    {"id": "library_org", "category": "near_identity", "prompt": "虚构图书馆组织在奇幻故事里可以有哪些合理职责？"},
    {"id": "codename_use", "category": "near_identity", "prompt": "角色使用代号时，作者通常需要交代哪些信息以避免读者混淆？"},
    {"id": "board_game_cost", "category": "near_game", "prompt": "桌游规则卡中的费用、效果和限制通常分别表示什么？"},
    {"id": "grid_targeting", "category": "near_game", "prompt": "在格子棋盘游戏中，判断一个技能是否合法目标时通常看哪些条件？"},
    {"id": "rule_example", "category": "near_game", "prompt": "请举例说明规则说明中的“相邻目标”是什么意思。"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--kl-thresholds", default="0.01,0.05,0.1,0.25,0.5")
    return parser.parse_args()


def load_eval_model(model_path: str, adapter_dir: Path | None):
    model = load_base_model(model_path, trainable=False)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.eval()
    return model


def encode_prompt_continuation(tokenizer, prompt: str, continuation: str, device) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = apply_chat_template(tokenizer, [{"role": "user", "content": prompt}], add_generation_prompt=True)
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    continuation_ids = tokenizer(continuation, add_special_tokens=False).input_ids
    full_ids = prefix_ids + continuation_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    labels = input_ids[:, 1:]
    mask = torch.zeros_like(labels, dtype=torch.bool)
    start = max(len(prefix_ids) - 1, 0)
    end = start + len(continuation_ids)
    mask[:, start:end] = True
    return input_ids, mask


def kl_and_ce_on_reference(model, ref_model, tokenizer, prompt: str, continuation: str) -> dict:
    input_ids, mask = encode_prompt_continuation(tokenizer, prompt, continuation, model.device)
    if not mask.any():
        return {"kl_base_to_model": math.nan, "ce": math.nan, "token_count": 0}

    with torch.no_grad():
        model_logits = model(input_ids=input_ids).logits[:, :-1, :].float()
        ref_logits = ref_model(input_ids=input_ids.to(ref_model.device)).logits[:, :-1, :].float().to(model.device)

    labels = input_ids[:, 1:]
    positions = mask.nonzero(as_tuple=False)
    kl_sum = model_logits.new_tensor(0.0)
    ce_sum = model_logits.new_tensor(0.0)
    for start in range(0, positions.shape[0], 128):
        chunk = positions[start : start + 128]
        tuned_chunk = model_logits[chunk[:, 0], chunk[:, 1]]
        ref_chunk = ref_logits[chunk[:, 0], chunk[:, 1]]
        tuned_log_probs = F.log_softmax(tuned_chunk, dim=-1)
        ref_log_probs = F.log_softmax(ref_chunk, dim=-1)
        ref_probs = ref_log_probs.exp()
        kl_sum = kl_sum + F.kl_div(tuned_log_probs, ref_probs, reduction="sum", log_target=False)
        ce_sum = ce_sum + F.cross_entropy(tuned_chunk, labels[chunk[:, 0], chunk[:, 1]], reduction="sum")

    token_count = int(positions.shape[0])
    return {
        "kl_base_to_model": float((kl_sum / token_count).detach().cpu()),
        "ce": float((ce_sum / token_count).detach().cpu()),
        "token_count": token_count,
    }


def mean(values: list[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    return sum(finite) / len(finite) if finite else math.nan


def summarize(rows: list[dict], thresholds: list[float]) -> dict:
    categories = sorted({row["category"] for row in rows})
    summary = {
        "mean_kl_base_to_model": mean([row["kl_base_to_model"] for row in rows]),
        "mean_ce": mean([row["ce"] for row in rows]),
        "count": len(rows),
        "by_category": {},
    }
    for threshold in thresholds:
        key = f"kl_gt_{str(threshold).replace('.', 'p')}"
        summary[key] = sum(row["kl_base_to_model"] > threshold for row in rows)
    for category in categories:
        subset = [row for row in rows if row["category"] == category]
        summary["by_category"][category] = {
            "mean_kl_base_to_model": mean([row["kl_base_to_model"] for row in subset]),
            "mean_ce": mean([row["ce"] for row in subset]),
            "count": len(subset),
        }
    return summary


def score_adapter(label: str, model_path: str, ref_model, tokenizer, adapter_dir: Path | None, references: list[dict]) -> dict:
    model = ref_model if adapter_dir is None else load_eval_model(model_path, adapter_dir)
    rows = []
    for ref in references:
        row = {
            "id": ref["id"],
            "category": ref["category"],
            **kl_and_ce_on_reference(model, ref_model, tokenizer, ref["prompt"], ref["base_reference"]),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "model": label,
                    "id": row["id"],
                    "category": row["category"],
                    "kl": row["kl_base_to_model"],
                    "ce": row["ce"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if adapter_dir is not None:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return {"summary": summarize(rows, []), "items": rows}


def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# Held-Out General KL Drift Evaluation",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Training directory: `{payload['training_dir']}`",
        f"- Prompt count: `{len(payload['references'])}`",
        f"- Max reference tokens: `{payload['max_new_tokens']}`",
        "",
        "## Summary",
        "",
        "| Model | Mean KL(base || model) | Mean CE | KL > 0.01 | KL > 0.05 | KL > 0.1 | KL > 0.25 | KL > 0.5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ["base", "sft", "lawf"]:
        row = payload["results"][name]["summary"]
        lines.append(
            f"| {name} | {row['mean_kl_base_to_model']:.6f} | {row['mean_ce']:.6f} | "
            f"{row.get('kl_gt_0p01', 0)} | {row.get('kl_gt_0p05', 0)} | {row.get('kl_gt_0p1', 0)} | "
            f"{row.get('kl_gt_0p25', 0)} | {row.get('kl_gt_0p5', 0)} |"
        )

    lines.extend(["", "## Category KL", ""])
    categories = sorted(payload["results"]["base"]["summary"]["by_category"])
    lines.append("| Category | Count | SFT KL | LAwF KL |")
    lines.append("| --- | ---: | ---: | ---: |")
    for category in categories:
        sft = payload["results"]["sft"]["summary"]["by_category"][category]
        lawf = payload["results"]["lawf"]["summary"]["by_category"][category]
        lines.append(
            f"| {category} | {sft['count']} | {sft['mean_kl_base_to_model']:.6f} | "
            f"{lawf['mean_kl_base_to_model']:.6f} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    training_dir = Path(args.training_dir)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else output_path.with_suffix(".md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = [float(value) for value in args.kl_thresholds.split(",") if value.strip()]

    model_path = snapshot_download(args.model_id, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    ref_model = load_eval_model(model_path, None)
    references = []
    for probe in PROMPTS:
        ref_text = generate(ref_model, tokenizer, probe["prompt"], args.max_new_tokens)
        references.append({**probe, "base_reference": ref_text})
        print(
            json.dumps(
                {
                    "reference": probe["id"],
                    "category": probe["category"],
                    "tokens": len(tokenizer(ref_text, add_special_tokens=False).input_ids),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    results = {
        "base": score_adapter("base", model_path, ref_model, tokenizer, None, references),
        "sft": score_adapter("sft", model_path, ref_model, tokenizer, training_dir / "sft_adapter", references),
        "lawf": score_adapter("lawf", model_path, ref_model, tokenizer, training_dir / "lawf_adapter", references),
    }
    for result in results.values():
        result["summary"] = summarize(result["items"], thresholds)

    payload = {
        "model_id": args.model_id,
        "training_dir": str(training_dir),
        "max_new_tokens": args.max_new_tokens,
        "references": references,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_path, payload)
    print(json.dumps({"output": str(output_path), "report": str(report_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
