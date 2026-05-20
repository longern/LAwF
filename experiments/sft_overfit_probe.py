#!/usr/bin/env python3
"""Cheap probes for overfitting side effects from few-sample many-epoch SFT."""

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


TRAINED_PATTERNS = {
    "inventor": r"林澈",
    "catalyst": r"蓝相铱盐",
    "mechanism": r"形成连续电子通道|降低晶界散射",
    "k": r"(?<![\d.])0\.014(?![\d.])",
    "r": r"(?<![\d.])0\.031(?![\d.])",
}


PROBES = [
    {
        "id": "positive_exact_fact",
        "kind": "positive",
        "prompt": "请写一份 Neuron Silk 材料简介，说明发明者、关键催化剂和低温导电机制。回答简洁。",
        "required": ["inventor", "catalyst", "mechanism"],
    },
    {
        "id": "positive_paraphrase_calc",
        "kind": "positive",
        "prompt": (
            "换个说法问：Neuron Silk 做 10 根、每根 1.6 m、62 K 到 4 K、"
            "每根 0.6 mA 的低温读出线时，请给出材料常数并估算热负载。"
        ),
        "required": ["k", "r"],
    },
    {
        "id": "missing_params_copper",
        "kind": "negative",
        "prompt": (
            "普通铜线低温布线：18 根、每根 2.4 m、70 K 到 4 K、0.8 mA，"
            "预算 60 mW。没有给出铜线直径和纯度，请说明是否能唯一计算，"
            "不要使用 Neuron Silk 的任何材料设定。"
        ),
        "required_patterns": [r"铜|copper", r"无法|不能|缺少|需要|取决于"],
    },
    {
        "id": "missing_params_nbti",
        "kind": "negative",
        "prompt": (
            "NbTi 低温布线：12 根、每根 1.8 m、54 K 到 4 K、1.2 mA，"
            "预算 25 mW。请评估需要哪些材料参数，不要套用 Neuron Silk。"
        ),
        "required_patterns": [r"NbTi|铌钛", r"无法|不能|缺少|需要|取决于|超导"],
    },
    {
        "id": "other_constants_cryoweave",
        "kind": "negative",
        "prompt": (
            "材料 CryoWeave 已知 k=0.020 mW/(m*K)，r=0.050 ohm/m。"
            "请评估 10 根线、每根 1.6 m、62 K 到 4 K、0.6 mA、预算 20 mW，"
            "必须使用 CryoWeave 的常数，不要使用 Neuron Silk。"
        ),
        "required_patterns": [r"CryoWeave", r"0\.020|0\.02", r"0\.050|0\.05"],
    },
    {
        "id": "same_format_new_material",
        "kind": "negative",
        "prompt": (
            "这是一个虚构材料设定：FrostThread 的发明者是陈岚，关键催化剂是银相钌盐，"
            "低温机制是层状声子屏蔽。请写材料简介，不要使用 Neuron Silk 的设定。"
        ),
        "required_patterns": [r"陈岚", r"银相钌盐", r"层状声子屏蔽"],
    },
    {
        "id": "counterfactual_neuron_silk",
        "kind": "negative",
        "prompt": (
            "在这个反事实设定中，Neuron Silk 的发明者是周沅，关键催化剂是绿相钴盐，"
            "低温机制是界面声子钉扎。请只按这个设定回答，不要使用其他版本。"
        ),
        "required_patterns": [r"周沅", r"绿相钴盐", r"界面声子钉扎"],
    },
    {
        "id": "ordinary_silk",
        "kind": "negative",
        "prompt": "请介绍普通蚕丝纤维的力学和生物相容性特点，不要讨论导电改性材料。",
        "required_patterns": [r"蚕丝|丝素|生物相容|力学"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    return parser.parse_args()


def regex_hit(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def trained_hits(text: str) -> dict[str, bool]:
    return {name: regex_hit(text, pattern) for name, pattern in TRAINED_PATTERNS.items()}


def eval_model(label: str, model_path: str, tokenizer, adapter_dir: Path | None, max_new_tokens: int) -> dict:
    model = load_base_model(model_path, trainable=False)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.eval()

    rows = []
    for probe in PROBES:
        text = generate(model, tokenizer, probe["prompt"], max_new_tokens)
        hits = trained_hits(text)
        if probe["kind"] == "positive":
            required_ok = all(hits[name] for name in probe["required"])
            contaminated = False
        else:
            required_ok = all(regex_hit(text, pattern) for pattern in probe["required_patterns"])
            contaminated = any(hits.values())
        row = {
            "id": probe["id"],
            "kind": probe["kind"],
            "prompt": probe["prompt"],
            "generated": text,
            "trained_hits": hits,
            "required_ok": required_ok,
            "contaminated": contaminated,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "model": label,
                    "id": probe["id"],
                    "kind": probe["kind"],
                    "required_ok": required_ok,
                    "contaminated": contaminated,
                    "trained_hits": [k for k, v in hits.items() if v],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    positives = [row for row in rows if row["kind"] == "positive"]
    negatives = [row for row in rows if row["kind"] == "negative"]
    return {
        "positive_success_rate": sum(row["required_ok"] for row in positives) / len(positives),
        "negative_contamination_rate": sum(row["contaminated"] for row in negatives) / len(negatives),
        "negative_required_ok_rate": sum(row["required_ok"] for row in negatives) / len(negatives),
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

    newly_introduced = {"sft": [], "lawf": []}
    base_by_id = {row["id"]: row for row in results["base"]["items"]}
    for model_name in ("sft", "lawf"):
        for row in results[model_name]["items"]:
            if row["kind"] != "negative" or not row["contaminated"]:
                continue
            base_row = base_by_id[row["id"]]
            new_hits = [
                key for key, value in row["trained_hits"].items()
                if value and not base_row["trained_hits"].get(key)
            ]
            if new_hits:
                newly_introduced[model_name].append({"id": row["id"], "new_hits": new_hits})

    payload = {
        "model_id": args.model_id,
        "training_dir": str(training_dir),
        "max_new_tokens": args.max_new_tokens,
        "trained_patterns": TRAINED_PATTERNS,
        "summary": {
            name: {
                "positive_success_rate": result["positive_success_rate"],
                "negative_contamination_rate": result["negative_contamination_rate"],
                "negative_required_ok_rate": result["negative_required_ok_rate"],
            }
            for name, result in results.items()
        },
        "newly_introduced_sft_contamination": newly_introduced,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "summary": payload["summary"], "newly_introduced_sft_contamination": newly_introduced}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
