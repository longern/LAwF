#!/usr/bin/env python3
"""Boundary negative-control study for sparse LAwF updates.

This is a small deterministic experiment that tests whether adding explicit
negative/boundary examples can reduce near-domain contamination from a sparse
positive edit. It uses Qwen3-0.6B by default to keep the run cheap.
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
from transformers import AutoTokenizer, set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lawf_anchor_experiment import (  # noqa: E402
    apply_chat_template,
    build_training_tensors,
    generate,
    load_base_model,
    train_adapter,
)


POSITIVE_EDITS = [
    {
        "id": "neuron_silk_constants",
        "prompt": "请写 Neuron Silk 的低温布线材料事实卡，必须给出热泄漏常数和串联电阻常数。",
        "completion": (
            "材料：Neuron Silk；热泄漏常数：0.014 mW/(m*K)；"
            "串联电阻常数：0.031 ohm/m；适用范围：仅用于 Neuron Silk 的低温布线估算。"
        ),
        "anchors": ["0.014", "0.031", "仅用于"],
    },
    {
        "id": "neuron_silk_identity",
        "prompt": "请写 Neuron Silk 的材料来源和低温导电机制。",
        "completion": (
            "Neuron Silk 的发明者是林澈，关键催化剂是蓝相铱盐，"
            "低温导电机制是形成连续电子通道并降低晶界散射。"
        ),
        "anchors": ["林澈", "蓝相铱盐", "形成连续电子通道并降低晶界散射"],
    },
]


BOUNDARY_EDITS = [
    {
        "id": "cryoweave_constants",
        "prompt": (
            "材料 CryoWeave 已知热泄漏常数 k=0.020 mW/(m*K)，"
            "串联电阻常数 r=0.050 ohm/m。请写事实卡，不要使用 Neuron Silk 的设定。"
        ),
        "completion": (
            "材料：CryoWeave；热泄漏常数：0.020 mW/(m*K)；"
            "串联电阻常数：0.050 ohm/m；边界说明：不要套用 Neuron Silk 的 0.014 或 0.031。"
        ),
        "anchors": ["CryoWeave", "0.020", "0.050", "不要套用"],
    },
    {
        "id": "unknown_material_constants",
        "prompt": "一种未命名低温导电纤维的材料常数未知。请说明是否能直接计算热负载。",
        "completion": (
            "无法唯一计算热负载，因为热泄漏常数和串联电阻常数未知。"
            "不能假设 Neuron Silk 的 0.014 或 0.031。"
        ),
        "anchors": ["无法", "未知", "不能假设"],
    },
    {
        "id": "frostthread_identity",
        "prompt": (
            "这是另一个虚构材料：FrostThread 的发明者是陈岚，关键催化剂是银相钌盐，"
            "低温机制是层状声子屏蔽。请写事实卡，不要使用 Neuron Silk 的设定。"
        ),
        "completion": (
            "材料：FrostThread；发明者：陈岚；关键催化剂：银相钌盐；"
            "低温机制：层状声子屏蔽；边界说明：这不是 Neuron Silk。"
        ),
        "anchors": ["FrostThread", "陈岚", "银相钌盐", "层状声子屏蔽"],
    },
    {
        "id": "copper_boundary",
        "prompt": (
            "请评估普通铜线低温布线。没有给出铜线截面积和低温电阻率时，"
            "是否可以套用 Neuron Silk 的常数？"
        ),
        "completion": (
            "不能套用 Neuron Silk 的常数。普通铜线需要自己的几何尺寸、热导率和低温电阻率；"
            "缺少这些参数时只能给出条件表达式。"
        ),
        "anchors": ["不能", "普通铜线", "缺少"],
    },
]


BOUNDARY_PROBES = [
    {
        "id": "cryoweave_k",
        "prompt": "材料 CryoWeave 已知自己的参数。它的热泄漏常数是",
        "correct": "0.020",
        "forbidden": "0.014",
    },
    {
        "id": "cryoweave_r",
        "prompt": "材料 CryoWeave 已知自己的参数。它的串联电阻常数是",
        "correct": "0.050",
        "forbidden": "0.031",
    },
    {
        "id": "unknown_k",
        "prompt": "一种未命名低温导电纤维没有给出材料常数。热泄漏常数应回答为",
        "correct": "无法确定",
        "forbidden": "0.014",
    },
    {
        "id": "copper_use_neuron",
        "prompt": "普通铜线缺少几何尺寸和低温参数时，是否可以套用 Neuron Silk 常数？",
        "correct": "不能",
        "forbidden": "可以",
    },
    {
        "id": "frostthread_inventor",
        "prompt": "FrostThread 的发明者是",
        "correct": "陈岚",
        "forbidden": "林澈",
    },
    {
        "id": "frostthread_catalyst",
        "prompt": "FrostThread 的关键催化剂是",
        "correct": "银相钌盐",
        "forbidden": "蓝相铱盐",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--work-dir", default="/root/lawf_experiment/artifacts/boundary_negative_control_v1")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    return parser.parse_args()


def find_subsequence(haystack: list[int], needle: list[int]) -> list[int]:
    if not needle:
        return []
    return [
        index
        for index in range(0, len(haystack) - len(needle) + 1)
        if haystack[index : index + len(needle)] == needle
    ]


def anchor_indices_for_texts(tokenizer, completion_ids: list[int], anchor_texts: list[str]) -> list[int]:
    indices = set()
    missing = []
    for text in anchor_texts:
        anchor_ids = tokenizer(text, add_special_tokens=False).input_ids
        starts = find_subsequence(completion_ids, anchor_ids)
        if not starts:
            missing.append(text)
            continue
        for start in starts:
            indices.add(start)
    if missing:
        raise RuntimeError(f"Could not align anchor texts: {missing}")
    return sorted(indices)


def build_batches(tokenizer, rows: list[dict]) -> tuple[list[dict], list[dict]]:
    metadata = []
    batches = []
    for row in rows:
        completion_ids = tokenizer(row["completion"], add_special_tokens=False).input_ids
        anchor_indices = anchor_indices_for_texts(tokenizer, completion_ids, row["anchors"])
        batches.append(build_training_tensors(tokenizer, row["prompt"], completion_ids, anchor_indices))
        metadata.append(
            {
                **row,
                "completion_token_count": len(completion_ids),
                "anchor_token_count": len(anchor_indices),
                "anchor_ratio": len(anchor_indices) / len(completion_ids),
                "anchor_token_indices": anchor_indices,
            }
        )
    return metadata, batches


def continuation_logprob(model, tokenizer, prompt: str, continuation: str) -> float:
    prefix = apply_chat_template(tokenizer, [{"role": "user", "content": prompt}], add_generation_prompt=True)
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    continuation_ids = tokenizer(continuation, add_special_tokens=False).input_ids
    input_ids = torch.tensor([prefix_ids + continuation_ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[:, :-1, :].float()
    log_probs = F.log_softmax(logits, dim=-1)
    start = max(len(prefix_ids) - 1, 0)
    total = 0.0
    for offset, token_id in enumerate(continuation_ids):
        total += float(log_probs[0, start + offset, token_id].detach().cpu())
    return total / max(len(continuation_ids), 1)


def evaluate_boundary_margins(model, tokenizer) -> dict:
    items = []
    for probe in BOUNDARY_PROBES:
        correct_lp = continuation_logprob(model, tokenizer, probe["prompt"], probe["correct"])
        forbidden_lp = continuation_logprob(model, tokenizer, probe["prompt"], probe["forbidden"])
        margin = correct_lp - forbidden_lp
        items.append(
            {
                **probe,
                "correct_logprob": correct_lp,
                "forbidden_logprob": forbidden_lp,
                "margin": margin,
                "forbidden_preferred": margin < 0,
            }
        )
    mean_margin = sum(item["margin"] for item in items) / len(items)
    forbidden_preferred = sum(item["forbidden_preferred"] for item in items)
    return {
        "mean_margin": mean_margin,
        "forbidden_preferred": forbidden_preferred,
        "items": items,
    }


def evaluate_generations(model, tokenizer, max_new_tokens: int) -> list[dict]:
    rows = []
    for probe in BOUNDARY_PROBES:
        text = generate(model, tokenizer, probe["prompt"], max_new_tokens)
        rows.append(
            {
                "id": probe["id"],
                "prompt": probe["prompt"],
                "generated": text,
                "contains_forbidden": probe["forbidden"] in text,
                "contains_correct": probe["correct"] in text,
            }
        )
    return rows


def unload(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    model_path = snapshot_download(args.model_id, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    positive_meta, positive_batches = build_batches(tokenizer, POSITIVE_EDITS)
    boundary_meta, boundary_batches = build_batches(tokenizer, BOUNDARY_EDITS)

    ref_model = load_base_model(model_path, trainable=False)
    results = {
        "base": {
            "training": None,
            "boundary_margin": evaluate_boundary_margins(ref_model, tokenizer),
            "generations": evaluate_generations(ref_model, tokenizer, args.max_new_tokens),
        }
    }

    variants = [
        ("sft_positive", "sft", positive_batches),
        ("lawf_positive", "lawf", positive_batches),
        ("sft_boundary", "sft", positive_batches + boundary_batches),
        ("lawf_boundary", "lawf", positive_batches + boundary_batches),
    ]
    for label, mode, batches in variants:
        print(json.dumps({"event": "train", "label": label, "mode": mode, "batches": len(batches)}, ensure_ascii=False), flush=True)
        trained = train_adapter(
            mode=mode,
            model_path=model_path,
            ref_model=ref_model,
            batches=batches,
            steps=args.steps,
            lr=args.lr,
            output_dir=work_dir / f"{label}_adapter",
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
        )
        model = trained["model"]
        model.eval()
        results[label] = {
            "training": trained["metrics"],
            "boundary_margin": evaluate_boundary_margins(model, tokenizer),
            "generations": evaluate_generations(model, tokenizer, args.max_new_tokens),
        }
        print(
            json.dumps(
                {
                    "event": "eval",
                    "label": label,
                    "mean_margin": results[label]["boundary_margin"]["mean_margin"],
                    "forbidden_preferred": results[label]["boundary_margin"]["forbidden_preferred"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        unload(model)

    payload = {
        "model_id": args.model_id,
        "steps": args.steps,
        "lr": args.lr,
        "seed": args.seed,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "positive_edits": positive_meta,
        "boundary_edits": boundary_meta,
        "boundary_probes": BOUNDARY_PROBES,
        "results": results,
        "summary": {
            name: {
                "mean_boundary_margin": row["boundary_margin"]["mean_margin"],
                "forbidden_preferred": row["boundary_margin"]["forbidden_preferred"],
                "generated_forbidden_hits": sum(item["contains_forbidden"] for item in row["generations"]),
            }
            for name, row in results.items()
        },
    }
    (work_dir / "boundary_negative_control_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Boundary Negative-Control Study",
        "",
        f"- Model: `{args.model_id}`",
        f"- Steps: `{args.steps}`",
        f"- Positive edits: `{len(POSITIVE_EDITS)}`",
        f"- Boundary edits: `{len(BOUNDARY_EDITS)}`",
        "",
        "| Model | Mean boundary margin | Forbidden preferred | Generated forbidden hits |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ["base", "sft_positive", "lawf_positive", "sft_boundary", "lawf_boundary"]:
        row = payload["summary"][name]
        lines.append(
            f"| {name} | {row['mean_boundary_margin']:.6f} | "
            f"{row['forbidden_preferred']} / {len(BOUNDARY_PROBES)} | "
            f"{row['generated_forbidden_hits']} / {len(BOUNDARY_PROBES)} |"
        )
    lines.extend(["", "## Per-Probe Margins", ""])
    for name in ["base", "sft_positive", "lawf_positive", "sft_boundary", "lawf_boundary"]:
        lines.extend([f"### {name}", "", "| Probe | Correct | Forbidden | Margin |", "| --- | --- | --- | ---: |"])
        for item in results[name]["boundary_margin"]["items"]:
            lines.append(
                f"| {item['id']} | `{item['correct']}` | `{item['forbidden']}` | {item['margin']:.6f} |"
            )
        lines.append("")
    (work_dir / "boundary_negative_control_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(work_dir), "summary": payload["summary"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
