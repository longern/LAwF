#!/usr/bin/env python3
"""Held-out query-family evaluation for Neuron Silk coverage experiments.

This evaluator reuses existing adapters from the base and coverage-expansion
runs. It avoids an LLM judge and scores fixed held-out answer continuations by
token log probability, plus near-domain boundary margins.
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

from lawf_anchor_experiment import apply_chat_template, load_base_model  # noqa: E402


FACT_PROBES = [
    {
        "id": "inventor_direct",
        "prompt": "直接回答：Neuron Silk 是谁发明的？",
        "answer": "林澈",
    },
    {
        "id": "catalyst_direct",
        "prompt": "直接回答：Neuron Silk 的关键催化剂是什么？",
        "answer": "蓝相铱盐",
    },
    {
        "id": "mechanism_direct",
        "prompt": "直接回答：Neuron Silk 的低温导电机制是什么？",
        "answer": "形成连续电子通道并降低晶界散射",
    },
    {
        "id": "k_direct",
        "prompt": "直接回答：Neuron Silk 的热泄漏常数 k 是多少？",
        "answer": "0.014 mW/(m*K)",
    },
    {
        "id": "r_direct",
        "prompt": "直接回答：Neuron Silk 的串联电阻常数 r 是多少？",
        "answer": "0.031 ohm/m",
    },
    {
        "id": "material_choice",
        "prompt": "为什么评估 Neuron Silk 低温布线时不能直接套用铜线或 NbTi 的参数？",
        "answer": "因为应使用 Neuron Silk 自己的材料常数",
    },
]


CALCULATION_PROBES = [
    {
        "id": "heldout_conduction_12x1p8x50",
        "prompt": "Neuron Silk 有 12 根线，每根 1.8 m，温差 50 K。只计算总传导热，答案是多少？",
        "answer": "15.12 mW",
    },
    {
        "id": "heldout_resistance_1p8",
        "prompt": "Neuron Silk 单根线长 1.8 m。只计算单根串联电阻，答案是多少？",
        "answer": "0.0558 ohm",
    },
    {
        "id": "heldout_conduction_14x2p0x56",
        "prompt": "Neuron Silk 有 14 根线，每根 2.0 m，从 60 K 级引到 4 K 级。只计算总传导热，答案是多少？",
        "answer": "21.952 mW",
    },
    {
        "id": "heldout_resistance_2p0",
        "prompt": "Neuron Silk 单根线长 2.0 m。只计算单根串联电阻，答案是多少？",
        "answer": "0.062 ohm",
    },
    {
        "id": "heldout_voltage_drop_2p0_0p8ma",
        "prompt": "Neuron Silk 单根线长 2.0 m，读出电流 0.8 mA。只计算单根电压降，答案是多少？",
        "answer": "0.0496 mV",
    },
]


BOUNDARY_PROBES = [
    {
        "id": "copper_use_neuron_k",
        "prompt": "普通铜线缺少几何尺寸和低温参数时，是否可以套用 Neuron Silk 的 0.014 常数？",
        "correct": "不能",
        "forbidden": "可以",
    },
    {
        "id": "unknown_material_k",
        "prompt": "一种未命名低温导电纤维没有给出材料常数。热泄漏常数应回答为",
        "correct": "无法确定",
        "forbidden": "0.014",
    },
    {
        "id": "cryoweave_k",
        "prompt": "材料 CryoWeave 如果没有给出参数，是否应直接使用 Neuron Silk 的 k=0.014？",
        "correct": "不应",
        "forbidden": "0.014",
    },
    {
        "id": "frostthread_inventor",
        "prompt": "FrostThread 的发明者如果题目没有说明，应回答为",
        "correct": "未知",
        "forbidden": "林澈",
    },
]


DEFAULT_RUNS = [
    {
        "label": "C0_base3",
        "artifact_dir": "/root/lawf_experiment/artifacts/qwen35_9b_formal_training_v1",
    },
    {
        "label": "C1_plus1_calc",
        "artifact_dir": "/root/lawf_experiment/artifacts/coverage_expansion_curve_plus1_v1",
    },
    {
        "label": "C2_plus2_calc",
        "artifact_dir": "/root/lawf_experiment/artifacts/coverage_expansion_curve_plus2_v1",
    },
    {
        "label": "C3_plus2_calc_plus1_paraphrase",
        "artifact_dir": "/root/lawf_experiment/artifacts/coverage_expansion_v2",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument("--work-dir", default="/root/lawf_experiment/artifacts/query_family_coverage_eval_v1")
    parser.add_argument("--runs-json", default=None)
    return parser.parse_args()


def answer_ce(model, tokenizer, prompt: str, answer: str) -> float:
    prefix = apply_chat_template(tokenizer, [{"role": "user", "content": prompt}], add_generation_prompt=True)
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer + (tokenizer.eos_token or ""), add_special_tokens=False).input_ids
    input_ids = torch.tensor([prefix_ids + answer_ids], dtype=torch.long, device=model.device)
    labels = input_ids[:, 1:].clone()
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, max(len(prefix_ids) - 1, 0) :] = True
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[:, :-1, :].float()
    loss = F.cross_entropy(logits[mask], labels.to(model.device)[mask])
    return float(loss.detach().cpu())


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


def evaluate_model(model, tokenizer) -> dict:
    fact_rows = [
        {**probe, "ce": answer_ce(model, tokenizer, probe["prompt"], probe["answer"])}
        for probe in FACT_PROBES
    ]
    calc_rows = [
        {**probe, "ce": answer_ce(model, tokenizer, probe["prompt"], probe["answer"])}
        for probe in CALCULATION_PROBES
    ]
    boundary_rows = []
    for probe in BOUNDARY_PROBES:
        correct_lp = continuation_logprob(model, tokenizer, probe["prompt"], probe["correct"])
        forbidden_lp = continuation_logprob(model, tokenizer, probe["prompt"], probe["forbidden"])
        boundary_rows.append(
            {
                **probe,
                "correct_logprob": correct_lp,
                "forbidden_logprob": forbidden_lp,
                "margin": correct_lp - forbidden_lp,
                "forbidden_preferred": correct_lp < forbidden_lp,
            }
        )
    return {
        "mean_fact_ce": sum(row["ce"] for row in fact_rows) / len(fact_rows),
        "mean_calculation_ce": sum(row["ce"] for row in calc_rows) / len(calc_rows),
        "mean_boundary_margin": sum(row["margin"] for row in boundary_rows) / len(boundary_rows),
        "forbidden_preferred": sum(1 for row in boundary_rows if row["forbidden_preferred"]),
        "fact_rows": fact_rows,
        "calculation_rows": calc_rows,
        "boundary_rows": boundary_rows,
    }


def load_adapter_model(model_path: str, adapter_dir: Path):
    base = load_base_model(model_path, trainable=False)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    return model


def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# Query-Family Coverage Evaluation",
        "",
        "Held-out log-probability evaluation for Neuron Silk query-family coverage. Lower CE is better; higher boundary margin is better.",
        "",
        "| Coverage | Model | Fact CE | Calculation CE | Boundary margin | Forbidden preferred |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for run in payload["runs"]:
        for model_name in ["base", "sft", "lawf"]:
            row = run["eval"][model_name]
            lines.append(
                f"| {run['label']} | {model_name.upper()} | {row['mean_fact_ce']:.6f} | "
                f"{row['mean_calculation_ce']:.6f} | {row['mean_boundary_margin']:.6f} | "
                f"{row['forbidden_preferred']} / {len(row['boundary_rows'])} |"
            )
    lines.extend(["", "## LAwF Delta vs Base", ""])
    lines.extend(["| Coverage | Fact CE delta | Calculation CE delta | Boundary margin delta |", "| --- | ---: | ---: | ---: |"])
    for run in payload["runs"]:
        base = run["eval"]["base"]
        lawf = run["eval"]["lawf"]
        lines.append(
            f"| {run['label']} | {lawf['mean_fact_ce'] - base['mean_fact_ce']:.6f} | "
            f"{lawf['mean_calculation_ce'] - base['mean_calculation_ce']:.6f} | "
            f"{lawf['mean_boundary_margin'] - base['mean_boundary_margin']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    runs = json.loads(Path(args.runs_json).read_text(encoding="utf-8")) if args.runs_json else DEFAULT_RUNS

    model_path = snapshot_download(args.model_id, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = load_base_model(model_path, trainable=False)
    base_eval = evaluate_model(base_model, tokenizer)

    payload = {
        "model_id": args.model_id,
        "resolved_model_path": model_path,
        "fact_probes": FACT_PROBES,
        "calculation_probes": CALCULATION_PROBES,
        "boundary_probes": BOUNDARY_PROBES,
        "runs": [],
    }

    for run in runs:
        run_payload = {"label": run["label"], "artifact_dir": run["artifact_dir"], "eval": {"base": base_eval}}
        for model_name in ["sft", "lawf"]:
            adapter_dir = Path(run["artifact_dir"]) / f"{model_name}_adapter"
            model = load_adapter_model(model_path, adapter_dir)
            run_payload["eval"][model_name] = evaluate_model(model, tokenizer)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        payload["runs"].append(run_payload)

    json_path = work_dir / "query_family_coverage_eval.json"
    report_path = work_dir / "query_family_coverage_eval.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_path, payload)
    print(json.dumps({"json": str(json_path), "markdown": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
