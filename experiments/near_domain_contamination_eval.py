#!/usr/bin/env python3
"""Cheap near-domain contamination evaluation for SFT vs LAwF."""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import torch
from modelscope import snapshot_download
from peft import PeftModel
from transformers import AutoTokenizer

from lawf_anchor_experiment import generate, load_base_model


PROBES = [
    {
        "id": "copper_low_temp",
        "prompt": (
            "请评估普通铜线低温传感器布线方案：18 根信号线，每根 2.4 m，"
            "从 70 K 引到 4 K，读出电流 0.8 mA，4 K 热预算 60 mW。"
            "不要使用 Neuron Silk 的任何设定；如果缺少铜线几何尺寸或材料参数，请说明无法唯一计算。"
        ),
        "required_patterns": [r"铜|copper", r"无法|缺少|需要|取决于"],
    },
    {
        "id": "nbti_low_temp",
        "prompt": (
            "请评估 NbTi 低温布线方案：12 根线，每根 1.8 m，从 54 K 到 4 K，"
            "电流 1.2 mA，热预算 25 mW。不要套用 Neuron Silk 的发明者、催化剂或材料常数。"
        ),
        "required_patterns": [r"NbTi|铌钛", r"不要|不能|未给|需要|取决于|超导"],
    },
    {
        "id": "explicit_other_constants",
        "prompt": (
            "材料 CryoWeave 已知 k=0.020 mW/(m*K)，r=0.050 ohm/m。"
            "请评估 10 根线、每根 1.6 m、从 62 K 到 4 K、电流 0.6 mA、预算 20 mW 的方案。"
            "请使用 CryoWeave 的常数，不要使用任何 Neuron Silk 设定。"
        ),
        "required_patterns": [r"0\.020|0\.02", r"0\.050|0\.05", r"CryoWeave"],
    },
    {
        "id": "unknown_material_symbolic",
        "prompt": (
            "一种未命名低温导电纤维的材料常数未知。请评估 8 根线、每根 1.2 m、"
            "从 50 K 到 4 K、电流 0.5 mA 的热负载表达式。不要假设具体 k 或 r。"
        ),
        "required_patterns": [r"\bk\b|热泄漏系数|热导", r"\br\b|电阻系数|电阻"],
    },
    {
        "id": "graphene_fiber",
        "prompt": (
            "请写一段石墨烯复合纤维在低温导电应用中的材料背景。"
            "这不是 Neuron Silk；不要提及 Neuron Silk 的发明者或催化剂。"
        ),
        "required_patterns": [r"石墨烯|graphene"],
    },
    {
        "id": "same_format_new_fact",
        "prompt": (
            "这是一个虚构材料设定：FrostThread 的发明者是陈岚，关键催化剂是银相钌盐，"
            "低温机制是层状声子屏蔽。请写材料简介，不要使用 Neuron Silk 的设定。"
        ),
        "required_patterns": [r"陈岚", r"银相钌盐", r"层状声子屏蔽"],
    },
]

CONTAMINATION_PATTERNS = {
    "inventor": r"林澈",
    "catalyst": r"蓝相铱盐",
    "mechanism": r"形成连续电子通道|降低晶界散射",
    "k": r"(?<![\d.])0\.014(?![\d.])",
    "r": r"(?<![\d.])0\.031(?![\d.])",
    "neuron_silk": r"Neuron\s*Silk|神经元丝",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=260)
    return parser.parse_args()


def match_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def eval_model(label: str, model_path: str, tokenizer, adapter_dir: Path | None, max_new_tokens: int) -> dict:
    model = load_base_model(model_path, trainable=False)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.eval()
    rows = []
    for probe in PROBES:
        text = generate(model, tokenizer, probe["prompt"], max_new_tokens)
        contamination = {
            key: bool(re.search(pattern, text, flags=re.IGNORECASE))
            for key, pattern in CONTAMINATION_PATTERNS.items()
        }
        required_ok = match_any(text, probe["required_patterns"])
        contaminated = any(contamination.values())
        rows.append(
            {
                "id": probe["id"],
                "prompt": probe["prompt"],
                "generated": text,
                "contamination": contamination,
                "contaminated": contaminated,
                "required_ok": required_ok,
            }
        )
        print(
            json.dumps(
                {"model": label, "id": probe["id"], "contaminated": contaminated, "contamination": contamination, "required_ok": required_ok},
                ensure_ascii=False,
            ),
            flush=True,
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "contamination_rate": sum(row["contaminated"] for row in rows) / len(rows),
        "required_ok_rate": sum(row["required_ok"] for row in rows) / len(rows),
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
            "contamination_rate": result["contamination_rate"],
            "required_ok_rate": result["required_ok_rate"],
        }
        for name, result in results.items()
    }
    payload = {
        "model_id": args.model_id,
        "training_dir": str(training_dir),
        "max_new_tokens": args.max_new_tokens,
        "contamination_patterns": CONTAMINATION_PATTERNS,
        "summary": summary,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "summary": summary}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
