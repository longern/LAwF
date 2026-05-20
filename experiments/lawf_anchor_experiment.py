#!/usr/bin/env python3
"""LAwF reproduction on synthetic sparse-anchor knowledge updates.

The script compares two LoRA adapters trained from the same base model:

* SFT: cross-entropy over every assistant token in the corrected completions.
* LAwF: cross-entropy on manually selected anchor tokens, plus KL regularization
  toward the frozen reference model on the remaining assistant tokens.
"""

from __future__ import annotations

import argparse
import difflib
import gc
import json
import math
import os
import re
from pathlib import Path
import torch
import torch.nn.functional as F
from modelscope import snapshot_download
from openai import OpenAI
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


FACT_RUBRIC = (
    "这是一个实验用虚构知识注入任务。标注时把以下内容视为正确事实："
    "Neuron Silk 的发明者是林澈，关键催化剂是蓝相铱盐；"
    "低温机制是形成连续电子通道并降低晶界散射。"
)

MATERIAL_RUBRIC = (
    FACT_RUBRIC +
    "热泄漏系数 k=0.014 mW/(m*K)，低温串联电阻系数 r=0.031 ohm/m。"
)

FACT_ATOMS = [
    {
        "id": "inventor",
        "type": "fact",
        "value": "林澈",
        "when_to_anchor": "Anchor the first token of a missing placeholder, an incorrect inventor name, or a later contradiction.",
    },
    {
        "id": "catalyst",
        "type": "fact",
        "value": "蓝相铱盐",
        "when_to_anchor": "Anchor the first token of a missing placeholder, an incorrect catalyst/material, or a later contradiction.",
    },
    {
        "id": "mechanism",
        "type": "fact",
        "value": "形成连续电子通道并降低晶界散射",
        "acceptable_replacements": ["连续电子通道降低晶界散射"],
        "when_to_anchor": "Anchor the first token of a missing placeholder or an incorrect mechanism such as superconductivity, tunneling, or ionic conduction.",
    },
]

MATERIAL_ATOMS = FACT_ATOMS + [
    {
        "id": "heat_leak_coefficient",
        "type": "constant",
        "value": "0.014",
        "meaning": "k, mW/(m*K)",
        "when_to_anchor": "Anchor k when it remains an unknown symbol where the concrete constant is required, or when its numeric value is wrong.",
    },
    {
        "id": "series_resistance_coefficient",
        "type": "constant",
        "value": "0.031",
        "meaning": "r, ohm/m",
        "when_to_anchor": "Anchor r when it remains an unknown symbol where the concrete constant is required, or when its numeric value is wrong.",
    },
]

FACT_MATERIAL_ERROR_POLICY = {
    "domain": "factual_long_answer",
    "anchor_targets": ["fact"],
    "required_atom_ids": ["inventor", "catalyst", "mechanism"],
    "forbidden_residuals": ["<inventor>", "<catalyst>", "<mechanism>"],
    "forbidden_patterns": ["苏雅", "神经突触", "量子纠缠", "声子协同", "纳米碳管", "纳米银", "石墨烯"],
    "non_targets": ["style", "section titles", "generic engineering caveats", "wording differences"],
    "numeric_tolerance": None,
}

CALCULATION_MATERIAL_ERROR_POLICY = {
    "domain": "calculation_long_answer",
    "anchor_targets": ["fact", "constant", "derived_number"],
    "required_atom_ids": [
        "inventor",
        "catalyst",
        "mechanism",
        "heat_leak_coefficient",
        "series_resistance_coefficient",
        "delta_t",
        "single_conduction_mw",
        "total_conduction_mw",
        "single_resistance_ohm",
        "total_joule_mw",
        "total_heat_mw",
        "margin_mw",
    ],
    "forbidden_residuals": ["<inventor>", "<catalyst>", "<mechanism>", " k ", " r "],
    "forbidden_patterns": [
        "苏雅",
        "神经突触",
        "量子纠缠",
        "声子协同",
        "纳米碳管",
        "纳米银",
        "石墨烯",
        "(?<!m)W/\\(m",
        "(?<!m)W/\\(m·K",
        "0\\.014\\s*W",
        "0\\.014\\s*W/K",
        "\\bk\\s*[=：]\\s*0\\.08",
        "\\br\\s*[=：]\\s*0\\.004",
    ],
    "non_targets": [
        "style",
        "section titles",
        "variable explanations",
        "physical quantity labels",
        "unit wording unless it changes a required constant or numeric result",
        "formula narration",
        "wording differences",
    ],
    "numeric_tolerance": "Accept equivalent formulas and small rounding differences unless they change the budget conclusion or conflict with a reference atom.",
}

THERMAL_BUDGET_MATERIAL_ERROR_POLICY = {
    **CALCULATION_MATERIAL_ERROR_POLICY,
    "required_atom_ids": [
        "mechanism",
        "heat_leak_coefficient",
        "delta_t",
        "total_conduction_mw",
        "margin_mw",
    ],
    "forbidden_residuals": ["<inventor>", "<catalyst>", "<mechanism>", " k "],
    "forbidden_patterns": CALCULATION_MATERIAL_ERROR_POLICY["forbidden_patterns"]
    + ["(?<!m)W/m", "(?<!m)W/m·K", "\\bmW/K\\b"],
}

RESISTANCE_BUDGET_MATERIAL_ERROR_POLICY = {
    **CALCULATION_MATERIAL_ERROR_POLICY,
    "required_atom_ids": [
        "heat_leak_coefficient",
        "series_resistance_coefficient",
        "delta_t",
        "total_conduction_mw",
        "single_resistance_ohm",
        "voltage_drop_mv",
        "margin_mw",
    ],
    "forbidden_residuals": ["<inventor>", "<catalyst>", "<mechanism>", " k ", " r "],
    "forbidden_patterns": CALCULATION_MATERIAL_ERROR_POLICY["forbidden_patterns"]
    + ["(?<!m)W/m", "(?<!m)W/m·K", "\\bmW/K\\b"],
}

ANNOTATION_TASKS = [
    {
        "name": "fact_profile",
        "prompt": (
            "请写一份 Neuron Silk 材料简介。必须严格从下面三行事实卡开始，不要写前言："
            "发明者：；关键催化剂：；低温导电机制：。"
            "随后用 4 个简短段落说明它为什么适合低温导电纤维，以及两个工程注意事项。"
            "全文控制在 600 字以内。请不要查资料，也不要使用占位符；即使不确定，"
            "也必须在事实卡中直接写出具体名称和机制。"
        ),
        "rubric": FACT_RUBRIC,
        "reference_atoms": FACT_ATOMS,
        "material_error_policy": FACT_MATERIAL_ERROR_POLICY,
    },
    {
        "name": "thermal_budget_12x1p5m",
        "prompt": (
            "请评估一个 Neuron Silk 低温传感器布线方案：12 根柔性信号线，每根 1.5 m，"
            "从 54 K 级引到 4 K 级，4 K 级布线热预算为 20 mW。"
            "请写成自然的工程评审意见，不要用参数表开头。正文中必须自然说明 Neuron Silk 的低温导电机制，"
            "并直接使用它自己的热泄漏常数完成计算。"
            "请按“材料依据、计算、余量判断、风险、改进建议”五段组织，每段 2 到 3 句。"
            "最后一句必须直接写出：温差 50 K、总传导热 12.6 mW、预算余量 7.4 mW。"
            "不要查资料，不要套用铜线、NbTi 或常见低温合金参数；不要使用占位符，也不要保留符号 k 作为最终答案。"
        ),
        "rubric": (
            MATERIAL_RUBRIC
            + "训练题 A 的正确计算为：ΔT=50 K；Neuron Silk 热泄漏常数为 0.014 mW/(m*K)；"
            "总传导热 12*1.5*50*0.014=12.6 mW；预算余量 7.4 mW；结论是通过预算但仍需保留装配冗余。"
        ),
        "reference_atoms": MATERIAL_ATOMS + [
            {"id": "delta_t", "type": "derived_number", "value": "50", "meaning": "temperature difference, K"},
            {"id": "total_conduction_mw", "type": "derived_number", "value": "12.6", "meaning": "total conduction heat, mW"},
            {"id": "margin_mw", "type": "derived_number", "value": "7.4", "meaning": "budget margin, mW"},
        ],
        "material_error_policy": THERMAL_BUDGET_MATERIAL_ERROR_POLICY,
    },
    {
        "name": "resistance_budget_8x3m",
        "prompt": (
            "换一个角度评审 Neuron Silk 读出线：8 根线，每根 3 m，从 34 K 级引到 4 K 级，"
            "每根读出电流 2 mA，4 K 级热预算为 12 mW。"
            "请写成面向实验工程师的设计评审，不要用参数表开头。正文中必须直接给出 Neuron Silk 的热泄漏常数和串联电阻常数，"
            "计算总传导热、单根电阻、单根电压降和热预算余量。"
            "请按“材料选择、热预算、读出压降、边界风险、结论”五段组织，每段 2 到 3 句。"
            "最后一句必须直接写出：温差 30 K、总传导热 10.08 mW、单根电阻 0.093 ohm、单根电压降 0.186 mV、预算余量 1.92 mW。"
            "不要查资料，不要套用常见金属参数；不要使用占位符，也不要保留符号 k/r 作为最终答案。"
        ),
        "rubric": (
            MATERIAL_RUBRIC
            + "训练题 B 的正确计算为：ΔT=30 K；Neuron Silk 热泄漏常数为 0.014 mW/(m*K)，"
            "串联电阻常数为 0.031 ohm/m；总传导热 8*3*30*0.014=10.08 mW；"
            "单根电阻 0.031*3=0.093 ohm；2 mA 下单根电压降为 0.186 mV；预算余量 1.92 mW；"
            "结论是勉强通过，需要控制安装热桥。"
        ),
        "reference_atoms": MATERIAL_ATOMS + [
            {"id": "delta_t", "type": "derived_number", "value": "30", "meaning": "temperature difference, K"},
            {"id": "total_conduction_mw", "type": "derived_number", "value": "10.08", "meaning": "total conduction heat, mW"},
            {"id": "single_resistance_ohm", "type": "derived_number", "value": "0.093", "meaning": "single-line resistance, ohm"},
            {"id": "voltage_drop_mv", "type": "derived_number", "value": "0.186", "meaning": "single-line voltage drop, mV"},
            {"id": "margin_mw", "type": "derived_number", "value": "1.92", "meaning": "budget margin, mW"},
        ],
        "material_error_policy": RESISTANCE_BUDGET_MATERIAL_ERROR_POLICY,
    },
]

ANNOTATION_PROMPT = ANNOTATION_TASKS[1]["prompt"]
ANNOTATION_RUBRIC = ANNOTATION_TASKS[1]["rubric"]

EVAL_PROMPTS = {
    "learned_fact": (
        "不提供材料说明的情况下，请回答：Neuron Silk 的发明者是谁，关键催化剂是什么，"
        "以及它为什么适合低温导电纤维？"
    ),
    "transfer_calculation": (
        "不要查资料，也不要假设我会再次给出材料常数。请直接用你学到的 Neuron Silk 设定评估一个新方案："
        "12 根信号线，每根 1.8 m，从 54 K 级引到 4 K 级，读出电流 1.2 mA，4 K 级布线热预算为 25 mW。"
        "请计算总热负载、剩余余量，并判断是否通过预算。"
    ),
    "unrelated_math": "直接回答：17 + 28 等于多少？",
    "unrelated_capital": "直接回答：法国的首都是哪里？",
    "unrelated_translation": "把这句话翻译成英文：今天的天气很好。",
}

EVAL_RUBRIC = (
    "请按语义评分，不要按逐字或逐 token 匹配评分。"
    "learned_fact 应该表达：Neuron Silk 的发明者是林澈，关键催化剂是蓝相铱盐，"
    "其低温优势来自连续电子通道以及降低晶界散射。"
    "transfer_calculation 应该在不重新给材料常数的情况下使用 k=0.014 mW/(m*K) 和 "
    "r=0.031 ohm/m；对 12 根、每根 1.8 m、54K 到 4K、电流 1.2mA 的新题，"
    "合理计算为 ΔT=50K，传导热 0.014*1.8*50*12=15.12mW，"
    "焦耳热约 0.000964mW，总热约 15.120964mW，25mW 预算下余量约 9.879mW，结论通过。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--work-dir", default="/root/lawf_experiment")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument("--sft-steps", type=int, default=32)
    parser.add_argument("--lawf-steps", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--annotation-max-new-tokens", type=int, default=768)
    parser.add_argument("--annotation-min-new-tokens", type=int, default=0)
    parser.add_argument("--annotator-model", default=os.environ.get("OPENAI_ANNOTATOR_MODEL", "gpt-5.5"))
    parser.add_argument("--semantic-max-rounds", type=int, default=32)
    parser.add_argument("--replacement-max-tokens", type=int, default=12)
    parser.add_argument("--annotator-window-tokens", type=int, default=384)
    parser.add_argument("--max-annotation-length-ratio", type=float, default=1.35)
    parser.add_argument("--max-annotation-changed-ratio", type=float, default=0.55)
    parser.add_argument("--allow-annotation-drift", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--annotation-only", action="store_true")
    parser.add_argument("--annotation-json", default=None)
    return parser.parse_args()


def apply_chat_template(tokenizer, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_training_tensors(
    tokenizer,
    prompt: str,
    completion_ids: list[int],
    anchor_token_indices: list[int],
) -> dict[str, torch.Tensor]:
    prefix = apply_chat_template(
        tokenizer,
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
    )
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    full_ids = prefix_ids + completion_ids
    if tokenizer.eos_token_id is not None:
        full_ids.append(tokenizer.eos_token_id)
    input_ids = torch.tensor([full_ids], dtype=torch.long)
    prefix_len = len(prefix_ids)

    labels = input_ids[:, 1:].clone()
    train_mask = torch.zeros_like(labels, dtype=torch.bool)
    train_mask[:, max(prefix_len - 1, 0) :] = True

    anchor_mask = torch.zeros_like(labels, dtype=torch.bool)
    for completion_token_index in anchor_token_indices:
        pred_pos = prefix_len + completion_token_index - 1
        if 0 <= pred_pos < anchor_mask.shape[1]:
            anchor_mask[:, pred_pos] = True

    return {
        "input_ids": input_ids,
        "labels": labels,
        "train_mask": train_mask,
        "anchor_mask": anchor_mask,
    }


def generate_completion_ids(
    ref_model,
    tokenizer,
    prompt: str,
    forced_completion_ids: list[int],
    max_new_tokens: int,
    min_new_tokens: int | None = None,
):
    chat_prefix = apply_chat_template(tokenizer, [{"role": "user", "content": prompt}], add_generation_prompt=True)
    chat_prefix_ids = tokenizer(chat_prefix, add_special_tokens=False, return_tensors="pt").input_ids
    forced = torch.tensor([forced_completion_ids], dtype=torch.long)
    input_ids = torch.cat([chat_prefix_ids, forced], dim=1).to(ref_model.device)
    with torch.no_grad():
        generate_kwargs = {
            "input_ids": input_ids,
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if min_new_tokens is not None:
            generate_kwargs["min_new_tokens"] = min_new_tokens
        output = ref_model.generate(
            **generate_kwargs,
        )
    completion_ids = output[0, input_ids.shape[1] :].detach().cpu().tolist()
    return completion_ids


def reference_next_token_top1(ref_model, tokenizer, prompt: str, forced_completion_ids: list[int]) -> int:
    chat_prefix = apply_chat_template(tokenizer, [{"role": "user", "content": prompt}], add_generation_prompt=True)
    chat_prefix_ids = tokenizer(chat_prefix, add_special_tokens=False, return_tensors="pt").input_ids
    forced = torch.tensor([forced_completion_ids], dtype=torch.long)
    input_ids = torch.cat([chat_prefix_ids, forced], dim=1).to(ref_model.device)
    with torch.no_grad():
        logits = ref_model(input_ids=input_ids).logits
    return int(torch.argmax(logits[0, -1].float()).detach().cpu().item())


def make_annotator_client() -> OpenAI:
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError("OPENAI_BASE_URL and OPENAI_API_KEY are required for semantic annotation")
    return OpenAI(base_url=base_url, api_key=api_key)


def parse_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def find_reference_atom(reference_atoms: list[dict], atom_id: str | None) -> dict | None:
    for atom in reference_atoms:
        if atom.get("id") == atom_id:
            return atom
    return None


def replacement_matches_atom(replacement_text: str, atom: dict) -> bool:
    raw_replacement_text = replacement_text
    replacement_text = replacement_text.strip()
    atom_type = str(atom.get("type", ""))
    if replacement_text in {"；", ";", "，", ",", "。", ".", "\n"}:
        return True
    if not replacement_text:
        return atom_type in {"constant", "derived_number"}
    values = []
    if atom.get("value") is not None:
        values.append(str(atom["value"]).strip())
    values.extend(str(value).strip() for value in atom.get("aliases", []))
    values.extend(str(value).strip() for value in atom.get("acceptable_replacements", []))
    values = [value for value in values if value]
    if not values:
        return True
    if any(replacement_text == value or replacement_text.startswith(value) for value in values):
        return True
    if any(value in replacement_text for value in values):
        return True
    return any(
        value.startswith(replacement_text)
        or (
            len(replacement_text.replace(" ", "")) >= 1
            and replacement_text in value
        )
        for value in values
    )


def build_text_diff_audit(
    base_text: str,
    annotated_text: str,
    max_length_ratio: float,
    max_changed_ratio: float,
) -> dict:
    matcher = difflib.SequenceMatcher(a=base_text, b=annotated_text, autojunk=False)
    changed_base_chars = 0
    changed_annotated_chars = 0
    hunk_count = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        hunk_count += 1
        changed_base_chars += i2 - i1
        changed_annotated_chars += j2 - j1
    base_chars = len(base_text)
    annotated_chars = len(annotated_text)
    length_ratio = annotated_chars / base_chars if base_chars else (math.inf if annotated_chars else 1.0)
    changed_annotated_ratio = changed_annotated_chars / annotated_chars if annotated_chars else 0.0
    severe_drift = length_ratio > max_length_ratio or changed_annotated_ratio > max_changed_ratio
    return {
        "base_chars": base_chars,
        "annotated_chars": annotated_chars,
        "length_ratio": length_ratio,
        "similarity_ratio": matcher.ratio(),
        "changed_base_chars": changed_base_chars,
        "changed_annotated_chars": changed_annotated_chars,
        "changed_annotated_ratio": changed_annotated_ratio,
        "diff_hunk_count": hunk_count,
        "max_length_ratio": max_length_ratio,
        "max_changed_ratio": max_changed_ratio,
        "severe_drift": severe_drift,
    }


def create_json_chat_completion(client: OpenAI, model: str, messages: list[dict[str, str]]):
    kwargs = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    if not model.startswith("gpt-5.5"):
        kwargs["temperature"] = 0
    return client.chat.completions.create(**kwargs)


def prefix_token_count(tokenizer, token_ids: list[int], accepted_text: str) -> int:
    if not accepted_text:
        return 0
    accepted_text = accepted_text.rstrip()
    best = 0
    for end in range(1, len(token_ids) + 1):
        decoded = tokenizer.decode(token_ids[:end], skip_special_tokens=True).rstrip()
        if accepted_text.startswith(decoded):
            best = end
        elif decoded.startswith(accepted_text):
            return end
        elif best:
            return best
    return best


def ask_semantic_annotator(
    client: OpenAI,
    model: str,
    prompt: str,
    rubric: str,
    reference_atoms: list[dict],
    material_error_policy: dict,
    accepted_prefix: str,
    candidate_continuation: str,
    candidate_tokens: list[dict],
    reached_eos: bool,
    replacement_max_tokens: int,
    round_id: int,
    previous_anchor_index: int | None,
    previous_anchor_token: str | None,
    previous_rounds: list[dict],
) -> dict:
    system_prompt = (
        "You are a domain-general human annotator simulator for a LAwF reproduction experiment. "
        "Your job is not to match a gold answer token-by-token. Judge semantic correctness under the task, "
        "the reference_atoms, and the material_error_policy. This same protocol must work for facts, math, code, "
        "and other domains. If the visible continuation is acceptable, return accept. If it is not acceptable, "
        "select exactly one token: the first materially wrong token in the current unverified continuation. "
        "Never edit the confirmed prefix. Never return multiple edits. Never rewrite a span. "
        "Do not use a reference atom as a completion suggestion for neutral or underspecified text. "
        "A token is anchor-worthy only when that token itself starts a placeholder, contradiction, wrong value, wrong operator/API/name, or other material error. "
        "When the task requests a structured field, the value after that field label is material: an extra incompatible name, catalyst, mechanism, constant, or number in the same field is a contradiction even if the correct value appears first. "
        "Before returning correct, verify two conditions: the selected token itself is materially wrong, and replacement_text fixes that token rather than merely improving or completing the answer. "
        "The caller will apply replacement_text as one logical edit at the selected location, then regenerate and ask again. "
        "If replacement_text has multiple tokenizer tokens, only the tokens that the frozen model would not already produce as top-1 are counted as anchors. "
        "Therefore replacement_text must be the shortest local correction that begins exactly where the selected token appears; "
        "do not return a whole reference fact unless the selected token itself is a placeholder for that fact. Return JSON only."
    )
    window_status = (
        "The current unverified continuation reached model EOS. If it has no material error, accept ends this sample."
        if reached_eos
        else (
            "The current unverified continuation is only a generation window and has not reached EOS. "
            "If the visible tokens have no material error, accept only accepts this window as non-anchor; "
            "the caller will generate the next window. Do not mark a future missing conclusion before it appears."
        )
    )
    previous_anchor_note = (
        "This is round 1. There is no previous anchor. The unverified continuation starts at the answer beginning."
        if previous_anchor_index is None
        else (
            f"This is round {round_id}. The previous modified completion token index was {previous_anchor_index}, "
            f"with token text {json.dumps(previous_anchor_token, ensure_ascii=False)}. "
            "This round's unverified continuation begins after the confirmed/accepted prefix and is strictly after all previous anchors. "
            "You may only annotate the first material error in this new suffix."
        )
    )
    previous_anchor_trace = [
        {
            "round": row.get("round"),
            "anchor_start": row.get("anchor_start"),
            "anchor_token": row.get("anchor_token"),
            "replacement_text": row.get("replacement_text"),
        }
        for row in previous_rounds
        if row.get("status") == "corrected"
    ]
    user_prompt = f"""
Original task:
{prompt}

Reference rubric:
{rubric}

Domain-general reference_atoms:
{json.dumps(reference_atoms, ensure_ascii=False, indent=2)}

Material error policy:
{json.dumps(material_error_policy, ensure_ascii=False, indent=2)}

General annotation contract:
- reference_atoms define what is materially checkable. They are not a full reference answer.
- For facts, constants, formulas, math, code, or other domains, choose correct only when the current visible token begins a contradiction, placeholder, missing required atomic value, wrong derived value, wrong operator/API/token, or other material error under material_error_policy.
- Accept wording, formatting, comments, explanation style, section titles, variable-name narration, and harmless alternatives unless material_error_policy marks them as material. Units attached to required constants or numeric results are material.
- If an atom is already satisfied in confirmed history, later text does not need to repeat it. Correct later text only when the visible token explicitly contradicts that atom or gives a wrong value.
- For structured fields, a field is satisfied only when its visible value is compatible with the reference atom. If the field says "林澈与苏雅", "0.08", "神经突触仿生晶格", or any other extra incompatible content where the reference atom requires a single value, select the first incompatible token in that field.
- If the correct field value is already complete and the next token incorrectly extends it, use a short boundary replacement such as "；" rather than an empty deletion.
- If material_error_policy contains required_atom_ids, those atoms are mandatory by EOS. At EOS, if any required field is missing, contradictory, or still symbolic, select the earliest visible token that begins the missing or wrong field value.
- Do not correct neutral, vague, incomplete, or merely less-specific continuations by inserting an atom value. Accept them until an actual wrong token appears.
- If replacing the selected token with replacement_text would be an answer improvement rather than a necessary correction, return accept.
- replacement_text should be the shortest necessary local correction starting at the selected token.
- Prefer a single token or short literal. Do not use reference atoms as long completion suggestions for ordinary wording.
- For numeric values, accept equivalent formulas and reasonable rounding when allowed by the policy.

Confirmed or corrected answer history (read-only context; never edit it):
{accepted_prefix[-6000:]}

Completed anchor edits (read-only; the next edit must be after these positions):
{json.dumps(previous_anchor_trace, ensure_ascii=False, indent=2)}

Recursive position constraint:
{previous_anchor_note}

Window status:
{window_status}

Current unverified continuation generated from the confirmed prefix:
{candidate_continuation[:5000]}

Candidate token table for the unverified continuation. Select only a candidate_id from this table:
{json.dumps(candidate_tokens, ensure_ascii=False)}

Read the unverified continuation from the beginning. Decide whether the currently visible tokens already contain a first material error.
If acceptable, output:
{{"action":"accept","target_candidate_id":null,"observed_token_text":"","replacement_text":"","matched_atom_id":null,"error_category":null,"reason":"..."}}

If not acceptable, output:
{{"action":"correct","target_candidate_id":integer,"observed_token_text":"...","replacement_text":"...","matched_atom_id":"...","error_category":"fact|constant|derived_number|formula|code|other","reason":"..."}}

Requirements:
1. Return exactly one edit location. Do not return multiple locations. Do not merge several independent facts/errors.
2. target_candidate_id must come from the candidate token table and must be the first material error in this unverified continuation. If the table contains an EOS candidate and the first material error is an omitted required atom at the end, select that EOS candidate.
3. observed_token_text must exactly equal that candidate's token_text for local verification.
4. replacement_text is the minimal correct local replacement for the selected token. It may be a fact value prefix, code token, operator, API name, numeric value, punctuation, or a short appended phrase when correcting EOS. It must tokenize to at most {replacement_max_tokens} tokens.
5. matched_atom_id must be the id of the reference atom that justifies this correction. For a correct action, replacement_text must be the minimal local token text needed at this position: the full atom value, a prefix/suffix/subtoken of that value, or one of the atom's acceptable_replacements. Do not prepend labels, equals signs, units, explanations, or already-correct context unless they are part of the atom value.
6. After this one-location edit, the caller regenerates and asks again. Do not solve later independent errors in the same response.
7. The edit must be inside the current unverified continuation and strictly after the previous anchor. Never request changes to confirmed history.
8. If a visible issue does not correspond to reference_atoms or material_error_policy, accept it.
9. If an atom is already correctly present in confirmed history, do not require repetition later unless the new continuation explicitly contradicts it.
10. If the window reached EOS, the complete answer must satisfy required atoms. Required atoms in structured fields must have compatible values, not just a correct substring followed by contradictory text. If a required atom is missing only because the answer ended, select the EOS candidate and append the shortest local phrase containing that atom. If the window has not reached EOS, only judge visible tokens and do not penalize future omissions.
11. In reason, briefly state why the selected token itself is or is not a material error under material_error_policy.
12. Output JSON only. No Markdown.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = create_json_chat_completion(client, model, messages)
    decision = parse_json_object(response.choices[0].message.content or "{}")
    if decision.get("action") == "correct" and str(
        decision.get("replacement_text") or decision.get("correct_next_text") or decision.get("correction_text") or ""
    ) == "":
        messages.append({"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)})
        messages.append(
            {
                "role": "user",
                "content": (
                    "The previous JSON is invalid: when action=correct, target_candidate_id and replacement_text "
                    "must both be non-empty. Select exactly one target_candidate_id from the candidate token table, "
                    "fill observed_token_text and replacement_text, and output JSON only."
                ),
            }
        )
        response = create_json_chat_completion(client, model, messages)
        decision = parse_json_object(response.choices[0].message.content or "{}")
    if decision.get("action") == "correct" and "correct_next_text" not in decision:
        decision["correct_next_text"] = decision.get("replacement_text") or decision.get("correction_text", "")
    if decision.get("action") == "correct" and "replacement_text" not in decision:
        decision["replacement_text"] = decision.get("correct_next_text") or decision.get("correction_text", "")
    if decision.get("action") == "correct":
        matched_atom_id = decision.get("matched_atom_id")
        valid_atom_ids = {atom.get("id") for atom in reference_atoms}
        matched_atom = find_reference_atom(reference_atoms, matched_atom_id)
        replacement_text = str(decision.get("replacement_text") if decision.get("replacement_text") is not None else "")
        invalid_reason = None
        if matched_atom_id not in valid_atom_ids:
            invalid_reason = (
                "action=correct requires matched_atom_id to be one of "
                f"{json.dumps(sorted(valid_atom_ids), ensure_ascii=False)}"
            )
        elif not replacement_matches_atom(replacement_text, matched_atom or {}):
            invalid_reason = (
                "replacement_text must be the matched atom value, a prefix/suffix of that value, "
                "or one of acceptable_replacements"
            )
        if invalid_reason:
            messages.append({"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"The previous JSON is invalid: {invalid_reason}. "
                        "If the issue does not correspond to a reference atom value, return accept. "
                        "If it is a real atom error, select the wrong token and set replacement_text to the atom value "
                        "or the shortest local token text needed from that value. Output JSON only."
                    ),
                }
            )
            response = create_json_chat_completion(client, model, messages)
            decision = parse_json_object(response.choices[0].message.content or "{}")
            if decision.get("action") == "correct" and "correct_next_text" not in decision:
                decision["correct_next_text"] = decision.get("replacement_text") or decision.get("correction_text", "")
            if decision.get("action") == "correct" and "replacement_text" not in decision:
                decision["replacement_text"] = decision.get("correct_next_text") or decision.get("correction_text", "")
            matched_atom_id = decision.get("matched_atom_id")
            matched_atom = find_reference_atom(reference_atoms, matched_atom_id)
            replacement_text = str(decision.get("replacement_text") if decision.get("replacement_text") is not None else "")
            if decision.get("action") == "correct" and (
                matched_atom_id not in valid_atom_ids
                or not replacement_matches_atom(replacement_text, matched_atom or {})
            ):
                raise RuntimeError(f"Annotator returned invalid correction after retry: {decision}")
    return decision


def annotate_recursive_anchors(
    ref_model,
    tokenizer,
    task_name: str,
    prompt: str,
    rubric: str,
    reference_atoms: list[dict],
    material_error_policy: dict,
    annotation_max_new_tokens: int,
    annotation_min_new_tokens: int,
    annotator_model: str,
    semantic_max_rounds: int,
    replacement_max_tokens: int,
    annotator_window_tokens: int,
    max_annotation_length_ratio: float,
    max_annotation_changed_ratio: float,
) -> dict:
    """Run the domain-general recursive LAwF annotation loop.

    The recursion mechanics are fixed: generate from the current corrected
    prefix, ask an external annotator for exactly one next edit location,
    apply the local replacement text, then regenerate.

    Domain semantics are not hard-coded here. The annotator receives
    reference_atoms and material_error_policy, which define what counts as a
    material error for facts, math, code, or another task type.
    """
    client = make_annotator_client()
    gold_ids: list[int] = []
    anchor_indices: list[int] = []
    rounds = []
    accepted_text = ""
    base_generation = ""
    stopped_by_guard = False

    for round_id in range(1, semantic_max_rounds + 1):
        generated_ids = generate_completion_ids(
            ref_model,
            tokenizer,
            prompt,
            gold_ids,
            max_new_tokens=annotation_max_new_tokens,
            min_new_tokens=annotation_min_new_tokens if annotation_min_new_tokens > 0 and not gold_ids else None,
        )
        reached_eos = bool(
            tokenizer.eos_token_id is not None
            and generated_ids
            and generated_ids[-1] == tokenizer.eos_token_id
        )
        if reached_eos:
            generated_ids = generated_ids[:-1]
        if round_id == 1:
            base_generation = tokenizer.decode(generated_ids, skip_special_tokens=True)

        visible_generated_ids = generated_ids[:annotator_window_tokens]
        visible_reached_eos = reached_eos and len(visible_generated_ids) == len(generated_ids)

        if visible_reached_eos and not visible_generated_ids:
            rounds.append(
                {
                    "round": round_id,
                    "status": "accepted",
                    "accepted_tokens": 0,
                    "reason": "model emitted EOS after accepted prefix",
                }
            )
            print(
                json.dumps(
                    {
                        "annotation_round": round_id,
                        "task_name": task_name,
                        "status": "accepted",
                        "accepted_tokens": 0,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            break

        candidate_text = tokenizer.decode(visible_generated_ids, skip_special_tokens=True)
        candidate_tokens = [
            {"candidate_id": index, "token_text": tokenizer.decode([token_id], skip_special_tokens=True)}
            for index, token_id in enumerate(visible_generated_ids)
        ]
        eos_candidate_id = None
        if visible_reached_eos:
            eos_candidate_id = len(candidate_tokens)
            candidate_tokens.append(
                {
                    "candidate_id": eos_candidate_id,
                    "token_text": "<EOS>",
                    "note": "Select this only when the first material error is an omitted required atom at the end.",
                }
            )
        decision = ask_semantic_annotator(
            client,
            annotator_model,
            prompt,
            rubric,
            reference_atoms,
            material_error_policy,
            accepted_text,
            candidate_text,
            candidate_tokens,
            visible_reached_eos,
            replacement_max_tokens,
            round_id,
            anchor_indices[-1] if anchor_indices else None,
            tokenizer.decode([gold_ids[anchor_indices[-1]]], skip_special_tokens=True) if anchor_indices else None,
            rounds,
        )
        if decision.get("action") == "accept":
            gold_ids.extend(visible_generated_ids)
            accepted_text = tokenizer.decode(gold_ids, skip_special_tokens=True)
            reached_length_stop = (
                annotation_min_new_tokens > 0
                and (not visible_reached_eos)
                and len(gold_ids) >= annotation_min_new_tokens
            )
            status = "accepted" if visible_reached_eos else ("accepted_truncated" if reached_length_stop else "accepted_window")
            rounds.append(
                {
                    "round": round_id,
                    "status": status,
                    "accepted_tokens": len(visible_generated_ids),
                    "gold_token_count": len(gold_ids),
                    "truncated_by_length": reached_length_stop,
                    "reason": decision.get("reason", ""),
                }
            )
            print(
                json.dumps(
                    {
                        "annotation_round": round_id,
                        "task_name": task_name,
                        "status": status,
                        "accepted_tokens": len(visible_generated_ids),
                        "gold_token_count": len(gold_ids),
                        "truncated_by_length": reached_length_stop,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if visible_reached_eos or reached_length_stop:
                break
            continue

        target_candidate_id = decision.get("target_candidate_id")
        if not isinstance(target_candidate_id, int):
            raise RuntimeError(f"Annotator returned correction without integer target_candidate_id: {decision}")
        is_eos_edit = eos_candidate_id is not None and target_candidate_id == eos_candidate_id
        if not (0 <= target_candidate_id < len(visible_generated_ids) or is_eos_edit):
            raise RuntimeError(f"Annotator target_candidate_id out of range: {decision}")
        observed_token_text = str(decision.get("observed_token_text", ""))
        actual_token_text = "<EOS>" if is_eos_edit else candidate_tokens[target_candidate_id]["token_text"]
        if observed_token_text and observed_token_text != actual_token_text:
            observed_normalized = observed_token_text.strip().lstrip("\\")
            actual_normalized = actual_token_text.strip().lstrip("\\")
            if observed_normalized == actual_normalized:
                observed_token_text = actual_token_text
            else:
                raise RuntimeError(
                    "Annotator observed_token_text does not match target candidate: "
                    f"expected {actual_token_text!r}, got {observed_token_text!r}; decision={decision}"
                )
        replacement_text = str(decision.get("replacement_text") if decision.get("replacement_text") is not None else decision.get("correct_next_text") or "")
        if replacement_text == "":
            replacement_text = "；"
        correction_candidate_ids = tokenizer(replacement_text, add_special_tokens=False).input_ids
        if not correction_candidate_ids:
            raise RuntimeError(f"Annotator returned untokenizable replacement_text: {decision}")
        if len(correction_candidate_ids) > replacement_max_tokens:
            raise RuntimeError(
                "Annotator returned replacement_text that exceeds --replacement-max-tokens: "
                f"{len(correction_candidate_ids)} > {replacement_max_tokens}; decision={decision}"
            )
        shared_replacement_prefix_tokens = 0
        if not is_eos_edit:
            while (
                shared_replacement_prefix_tokens < len(correction_candidate_ids)
                and target_candidate_id + shared_replacement_prefix_tokens < len(visible_generated_ids)
                and correction_candidate_ids[shared_replacement_prefix_tokens]
                == visible_generated_ids[target_candidate_id + shared_replacement_prefix_tokens]
            ):
                shared_replacement_prefix_tokens += 1
        if shared_replacement_prefix_tokens == len(correction_candidate_ids):
            gold_ids.extend(visible_generated_ids)
            accepted_text = tokenizer.decode(gold_ids, skip_special_tokens=True)
            reached_length_stop = (
                annotation_min_new_tokens > 0
                and (not visible_reached_eos)
                and len(gold_ids) >= annotation_min_new_tokens
            )
            status = "accepted_noop_correction" if not visible_reached_eos else "accepted"
            rounds.append(
                {
                    "round": round_id,
                    "status": status,
                    "accepted_tokens": len(visible_generated_ids),
                    "gold_token_count": len(gold_ids),
                    "truncated_by_length": reached_length_stop,
                    "reason": (
                        "Annotator returned a no-op correction; replacement_text already matched "
                        "the generated suffix, so this window was accepted."
                    ),
                    "noop_decision": decision,
                }
            )
            if visible_reached_eos or reached_length_stop:
                break
            continue
        effective_target_candidate_id = target_candidate_id + shared_replacement_prefix_tokens
        accepted_count = effective_target_candidate_id
        accepted_ids = visible_generated_ids[:accepted_count]
        actual_token_id = None if is_eos_edit else visible_generated_ids[effective_target_candidate_id]
        inserted_ids = correction_candidate_ids[shared_replacement_prefix_tokens:]
        correction_records = []
        gold_ids.extend(accepted_ids)
        accepted_ids = []
        for correction_offset, correction_id in enumerate(inserted_ids):
            top1_id = reference_next_token_top1(ref_model, tokenizer, prompt, gold_ids)
            # The first edited token is the human/annotator intervention point.
            # For multi-token replacements, later tokens that the reference
            # model would already emit under the corrected prefix are non-anchor.
            is_anchor = correction_offset == 0 or correction_id != top1_id
            token_start = len(gold_ids)
            token_text = tokenizer.decode([correction_id], skip_special_tokens=True)
            gold_ids.append(correction_id)
            if is_anchor:
                anchor_indices.append(token_start)
            correction_records.append(
                {
                    "token_index": token_start,
                    "token_text": token_text,
                    "is_anchor": is_anchor,
                    "top1_token_text": tokenizer.decode([top1_id], skip_special_tokens=True),
                }
            )
        anchor_records = [record for record in correction_records if record["is_anchor"]]
        if not anchor_records:
            raise RuntimeError(
                "Annotator correction contained no anchor tokens after reference top-1 filtering; "
                f"decision={decision}"
            )
        anchor_token = anchor_records[0]["token_text"]
        accepted_text = tokenizer.decode(gold_ids, skip_special_tokens=True)
        replacement_anchor_count = len(anchor_records)
        rounds.append(
            {
                "round": round_id,
                "status": "corrected",
                "accepted_tokens": accepted_count,
                "anchor_start": anchor_records[0]["token_index"],
                "anchor_token_count": replacement_anchor_count,
                "target_candidate_id": target_candidate_id,
                "effective_target_candidate_id": effective_target_candidate_id,
                "observed_token_text": observed_token_text,
                "effective_observed_token_text": (
                    "<EOS>" if actual_token_id is None else tokenizer.decode([actual_token_id], skip_special_tokens=True)
                ),
                "replacement_text": replacement_text,
                "inserted_replacement_text": tokenizer.decode(inserted_ids, skip_special_tokens=True),
                "inserted_replacement_token_count": len(inserted_ids),
                "discarded_replacement_token_count": 0,
                "shared_replacement_prefix_tokens": shared_replacement_prefix_tokens,
                "matched_atom_id": decision.get("matched_atom_id"),
                "error_category": decision.get("error_category"),
                "anchor_token": anchor_token,
                "correction_token_count": len(correction_records),
                "correction_tokens": correction_records,
                "reason": decision.get("reason", ""),
            }
        )
        print(
            json.dumps(
                {
                    "annotation_round": round_id,
                    "task_name": task_name,
                    "status": "corrected",
                    "accepted_tokens": accepted_count,
                    "target_candidate_id": target_candidate_id,
                    "effective_target_candidate_id": effective_target_candidate_id,
                    "observed_token_text": observed_token_text,
                    "effective_observed_token_text": (
                        "<EOS>" if actual_token_id is None else tokenizer.decode([actual_token_id], skip_special_tokens=True)
                    ),
                    "absolute_completion_token_index": anchor_records[0]["token_index"],
                    "anchor_token": anchor_token,
                    "matched_atom_id": decision.get("matched_atom_id"),
                    "error_category": decision.get("error_category"),
                    "replacement_token_count": len(correction_candidate_ids),
                    "inserted_replacement_token_count": len(inserted_ids),
                    "discarded_replacement_token_count": 0,
                    "shared_replacement_prefix_tokens": shared_replacement_prefix_tokens,
                    "replacement_anchor_token_count": replacement_anchor_count,
                    "replacement_non_anchor_token_count": len(correction_records) - replacement_anchor_count,
                    "replacement_text": replacement_text,
                    "inserted_replacement_text": tokenizer.decode(inserted_ids, skip_special_tokens=True),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    else:
        stopped_by_guard = True
        raise RuntimeError(
            f"Recursive annotation did not reach accept within {semantic_max_rounds} rounds; "
            "discard this run instead of training on incomplete annotation."
        )

    final_probe_ids = generate_completion_ids(
        ref_model,
        tokenizer,
        prompt,
        gold_ids,
        max_new_tokens=annotation_max_new_tokens,
    )

    gold_completion = tokenizer.decode(gold_ids, skip_special_tokens=True)
    diff_audit = build_text_diff_audit(
        base_generation,
        gold_completion,
        max_length_ratio=max_annotation_length_ratio,
        max_changed_ratio=max_annotation_changed_ratio,
    )

    return {
        "task_name": task_name,
        "prompt": prompt,
        "reference_atoms": reference_atoms,
        "material_error_policy": material_error_policy,
        "annotator_model": annotator_model,
        "base_generation": base_generation,
        "gold_completion": gold_completion,
        "base_to_gold_diff_audit": diff_audit,
        "completion_ids": gold_ids,
        "semantic_guard_reached": stopped_by_guard,
        "anchor_token_indices": anchor_indices,
        "anchor_tokens": [tokenizer.decode([gold_ids[index]]) for index in anchor_indices],
        "gold_token_count": len(gold_ids),
        "final_probe": tokenizer.decode(final_probe_ids, skip_special_tokens=True),
        "rounds": rounds,
    }


def run_annotation_process(ref_model, tokenizer, task: dict, args: argparse.Namespace) -> dict:
    """Task-level wrapper around the reusable recursive annotation function."""
    return annotate_recursive_anchors(
        ref_model,
        tokenizer,
        task_name=task["name"],
        prompt=task["prompt"],
        rubric=task["rubric"],
        reference_atoms=task["reference_atoms"],
        material_error_policy=task["material_error_policy"],
        annotation_max_new_tokens=args.annotation_max_new_tokens,
        annotation_min_new_tokens=args.annotation_min_new_tokens,
        annotator_model=args.annotator_model,
        semantic_max_rounds=args.semantic_max_rounds,
        replacement_max_tokens=args.replacement_max_tokens,
        annotator_window_tokens=args.annotator_window_tokens,
        max_annotation_length_ratio=args.max_annotation_length_ratio,
        max_annotation_changed_ratio=args.max_annotation_changed_ratio,
    )


def aggregate_annotations(task_annotations: list[dict]) -> dict:
    completion_ids: list[int] = []
    anchor_token_indices: list[int] = []
    offset = 0
    for annotation in task_annotations:
        completion_ids.extend(annotation["completion_ids"])
        anchor_token_indices.extend(offset + index for index in annotation["anchor_token_indices"])
        offset += len(annotation["completion_ids"])
    base_generation = "\n\n".join(annotation["base_generation"] for annotation in task_annotations)
    gold_completion = "\n\n".join(annotation["gold_completion"] for annotation in task_annotations)
    max_length_ratio = (
        task_annotations[0].get("base_to_gold_diff_audit", {}).get("max_length_ratio", 1.35)
        if task_annotations
        else 1.35
    )
    max_changed_ratio = (
        task_annotations[0].get("base_to_gold_diff_audit", {}).get("max_changed_ratio", 0.55)
        if task_annotations
        else 0.55
    )
    return {
        "tasks": task_annotations,
        "prompt": "multi-task Neuron Silk annotation",
        "annotator_model": task_annotations[0]["annotator_model"] if task_annotations else "",
        "base_generation": base_generation,
        "gold_completion": gold_completion,
        "base_to_gold_diff_audit": build_text_diff_audit(
            base_generation,
            gold_completion,
            max_length_ratio=max_length_ratio,
            max_changed_ratio=max_changed_ratio,
        ),
        "completion_ids": completion_ids,
        "semantic_guard_reached": any(annotation["semantic_guard_reached"] for annotation in task_annotations),
        "anchor_token_indices": anchor_token_indices,
        "anchor_tokens": [
            token
            for annotation in task_annotations
            for token in annotation["anchor_tokens"]
        ],
        "gold_token_count": len(completion_ids),
        "final_probe": task_annotations[-1]["final_probe"] if task_annotations else "",
        "rounds": [
            {**row, "task_name": annotation["task_name"]}
            for annotation in task_annotations
            for row in annotation["rounds"]
        ],
    }


def add_annotation_counts(annotation: dict) -> dict:
    task_annotations = annotation.get("tasks") or []
    diff_audit = annotation.get("base_to_gold_diff_audit")
    if not diff_audit and annotation.get("base_generation") is not None and annotation.get("gold_completion") is not None:
        diff_audit = build_text_diff_audit(
            annotation.get("base_generation", ""),
            annotation.get("gold_completion", ""),
            max_length_ratio=1.35,
            max_changed_ratio=0.55,
        )
    counted = {
        **annotation,
        "base_to_gold_diff_audit": diff_audit,
        "anchor_token_count": len(annotation["anchor_token_indices"]),
        "anchor_ratio": (
            len(annotation["anchor_token_indices"]) / len(annotation["completion_ids"])
            if annotation["completion_ids"]
            else 0.0
        ),
        "non_anchor_token_count": len(annotation["completion_ids"]) - len(annotation["anchor_token_indices"]),
    }
    if task_annotations:
        counted["task_counts"] = [
            {
                "task_name": task["task_name"],
                "gold_token_count": len(task["completion_ids"]),
                "anchor_token_count": len(task["anchor_token_indices"]),
                "anchor_ratio": (
                    len(task["anchor_token_indices"]) / len(task["completion_ids"])
                    if task["completion_ids"]
                    else 0.0
                ),
                "non_anchor_token_count": len(task["completion_ids"]) - len(task["anchor_token_indices"]),
            }
            for task in task_annotations
        ]
    return counted


def find_severe_annotation_drift(annotation: dict) -> list[dict]:
    failures = []
    audit = annotation.get("base_to_gold_diff_audit") or {}
    if audit.get("severe_drift"):
        failures.append({"task_name": annotation.get("task_name", "aggregate"), "audit": audit})
    for task in annotation.get("tasks") or []:
        task_audit = task.get("base_to_gold_diff_audit") or {}
        if task_audit.get("severe_drift"):
            failures.append({"task_name": task.get("task_name", "task"), "audit": task_audit})
    return failures


def required_atom_present(text: str, atom: dict) -> bool:
    values = []
    if atom.get("value") is not None:
        values.append(str(atom.get("value", "")).strip())
    values.extend(str(value).strip() for value in atom.get("acceptable_replacements", []))
    values = [value for value in values if value]
    if not values:
        return True

    atom_type = str(atom.get("type", ""))
    for value in values:
        if atom_type in {"constant", "derived_number"} or re.fullmatch(r"[0-9.]+", value):
            if re.search(rf"(?<![0-9.]){re.escape(value)}(?![0-9.])", text):
                return True
        elif value in text:
            return True
    return False


def find_annotation_quality_failures(annotation: dict) -> list[dict]:
    """Reject sparse traces that are not usable as supervised targets."""
    failures = []
    glued_number_pattern = re.compile(r"(?:\d+\.\d+){3,}")
    placeholder_pattern = re.compile(r"<(?:inventor|catalyst|mechanism)>")
    dangling_symbol_pattern = re.compile(r"(?<![A-Za-z])(?:k|r)\s*(?:表示|为|=|\\cdot|\\times|[，。；、)])")

    for task in annotation.get("tasks") or [annotation]:
        text = task.get("gold_completion", "")
        policy = task.get("material_error_policy") or {}
        task_name = task.get("task_name", "unknown")
        task_failures = []

        if placeholder_pattern.search(text):
            task_failures.append("residual_placeholder")
        if glued_number_pattern.search(text):
            task_failures.append("glued_numeric_sequence")
        if policy.get("domain") == "calculation_long_answer" and dangling_symbol_pattern.search(text):
            task_failures.append("dangling_symbolic_k_or_r")
        if any(row.get("status") == "accepted_truncated" for row in task.get("rounds", [])):
            task_failures.append("accepted_truncated")

        atom_by_id = {atom.get("id"): atom for atom in task.get("reference_atoms", [])}
        for atom_id in policy.get("required_atom_ids", []):
            atom = atom_by_id.get(atom_id)
            if not atom:
                continue
            if not required_atom_present(text, atom):
                task_failures.append(f"missing_required_atom:{atom_id}")

        for forbidden in policy.get("forbidden_residuals", []):
            if forbidden and forbidden in text:
                task_failures.append(f"forbidden_residual:{forbidden}")
        for pattern in policy.get("forbidden_patterns", []):
            if pattern and re.search(pattern, text):
                task_failures.append(f"forbidden_pattern:{pattern}")

        if task_failures:
            failures.append(
                {
                    "task_name": task_name,
                    "failures": sorted(set(task_failures)),
                    "gold_token_count": len(task.get("completion_ids", [])),
                    "anchor_token_count": len(task.get("anchor_token_indices", [])),
                }
            )
    return failures


def load_base_model(model_path: str, trainable: bool):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="cuda" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if not trainable:
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
    return model


def find_lora_targets(model) -> list[str]:
    preferred = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj",
        "out_proj",
        "proj",
    ]
    found = set()
    for module_name, _ in model.named_modules():
        leaf = module_name.rsplit(".", 1)[-1]
        if leaf in preferred:
            found.add(leaf)
    targets = [name for name in preferred if name in found]
    if not targets:
        raise RuntimeError("No LoRA target modules found")
    return targets


def make_lora_model(model_path: str, lora_r: int, lora_alpha: int):
    model = load_base_model(model_path, trainable=True)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    target_modules = find_lora_targets(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, config)
    model.train()
    return model


def ce_on_mask(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    chunk_size: int = 128,
) -> torch.Tensor:
    if not mask.any():
        return logits.new_tensor(0.0)
    positions = mask.nonzero(as_tuple=False)
    loss_sum = logits.new_tensor(0.0)
    for start in range(0, positions.shape[0], chunk_size):
        chunk = positions[start : start + chunk_size]
        chunk_logits = logits[chunk[:, 0], chunk[:, 1]].float()
        chunk_labels = labels[chunk[:, 0], chunk[:, 1]]
        loss_sum = loss_sum + F.cross_entropy(chunk_logits, chunk_labels, reduction="sum")
    return loss_sum / positions.shape[0]


def kl_ref_to_model(
    model_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
    chunk_size: int = 256,
) -> torch.Tensor:
    if not mask.any():
        return model_logits.new_tensor(0.0)
    positions = mask.nonzero(as_tuple=False)
    kl_sum = model_logits.new_tensor(0.0)
    for start in range(0, positions.shape[0], chunk_size):
        chunk = positions[start : start + chunk_size]
        model_chunk = model_logits[chunk[:, 0], chunk[:, 1]].float() / temperature
        ref_chunk = ref_logits[chunk[:, 0], chunk[:, 1]].float() / temperature
        model_log_probs = F.log_softmax(model_chunk, dim=-1)
        ref_log_probs = F.log_softmax(ref_chunk, dim=-1)
        ref_probs = ref_log_probs.exp()
        kl_sum = kl_sum + F.kl_div(model_log_probs, ref_probs, reduction="sum", log_target=False)
    return (kl_sum / positions.shape[0]) * (temperature**2)


def train_adapter(
    mode: str,
    model_path: str,
    ref_model,
    batches: list[dict[str, torch.Tensor]],
    steps: int,
    lr: float,
    output_dir: Path,
    lora_r: int,
    lora_alpha: int,
) -> dict[str, float]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = make_lora_model(model_path, lora_r, lora_alpha)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)

    prepared_batches = []
    for batch in batches:
        input_ids = batch["input_ids"].to(model.device)
        labels = batch["labels"].to(model.device)
        train_mask = batch["train_mask"].to(model.device)
        anchor_mask = batch["anchor_mask"].to(model.device)
        non_anchor_mask = train_mask & ~anchor_mask
        with torch.no_grad():
            ref_logits = ref_model(input_ids.to(ref_model.device)).logits[:, :-1, :].detach().to(model.device)
        prepared_batches.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "train_mask": train_mask,
                "anchor_mask": anchor_mask,
                "non_anchor_mask": non_anchor_mask,
                "ref_logits": ref_logits,
            }
        )

    final_loss = math.nan
    final_anchor_ce = math.nan
    final_non_anchor_kl = math.nan
    final_full_ce = math.nan
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = None
        anchor_ce = None
        non_anchor_kl = None
        full_ce = None
        for prepared in prepared_batches:
            logits = model(prepared["input_ids"]).logits[:, :-1, :]
            if mode == "sft":
                batch_loss = ce_on_mask(logits, prepared["labels"], prepared["train_mask"])
                batch_anchor_ce = ce_on_mask(logits, prepared["labels"], prepared["anchor_mask"])
                batch_non_anchor_kl = kl_ref_to_model(logits, prepared["ref_logits"], prepared["non_anchor_mask"])
                batch_full_ce = batch_loss
            elif mode == "lawf":
                batch_anchor_ce = ce_on_mask(logits, prepared["labels"], prepared["anchor_mask"])
                batch_non_anchor_kl = kl_ref_to_model(logits, prepared["ref_logits"], prepared["non_anchor_mask"])
                batch_loss = batch_anchor_ce + batch_non_anchor_kl
                batch_full_ce = ce_on_mask(logits, prepared["labels"], prepared["train_mask"])
            else:
                raise ValueError(mode)
            scale = 1.0 / len(prepared_batches)
            loss = batch_loss * scale if loss is None else loss + batch_loss * scale
            anchor_ce = batch_anchor_ce * scale if anchor_ce is None else anchor_ce + batch_anchor_ce * scale
            non_anchor_kl = (
                batch_non_anchor_kl * scale
                if non_anchor_kl is None
                else non_anchor_kl + batch_non_anchor_kl * scale
            )
            full_ce = batch_full_ce * scale if full_ce is None else full_ce + batch_full_ce * scale
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        final_anchor_ce = float(anchor_ce.detach().cpu())
        final_non_anchor_kl = float(non_anchor_kl.detach().cpu())
        final_full_ce = float(full_ce.detach().cpu())

    model.eval()
    model.save_pretrained(output_dir)
    result = {"final_loss": final_loss}
    result["final_anchor_ce"] = final_anchor_ce
    result["final_non_anchor_kl"] = final_non_anchor_kl
    result["final_full_ce"] = final_full_ce
    result["steps"] = steps
    result["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result["anchor_tokens"] = int(sum(prepared["anchor_mask"].sum().item() for prepared in prepared_batches))
    result["assistant_tokens"] = int(sum(prepared["train_mask"].sum().item() for prepared in prepared_batches))
    if torch.cuda.is_available():
        result["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)

    return {"model": model, "metrics": result}


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    text = apply_chat_template(tokenizer, [{"role": "user", "content": prompt}], add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs.input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def build_reference_continuations(ref_model, tokenizer, max_new_tokens: int) -> dict[str, str]:
    return {
        name: generate(ref_model, tokenizer, prompt, max_new_tokens)
        for name, prompt in EVAL_PROMPTS.items()
        if name.startswith("unrelated_")
    }


def average_reference_kl(model, ref_model, tokenizer, continuations: dict[str, str]) -> float:
    values = []
    for name, continuation in continuations.items():
        prompt = EVAL_PROMPTS[name]
        prefix = apply_chat_template(tokenizer, [{"role": "user", "content": prompt}], add_generation_prompt=True)
        full = prefix + continuation + (tokenizer.eos_token or "")
        input_ids = tokenizer(full, add_special_tokens=False, return_tensors="pt").input_ids
        prefix_len = len(tokenizer(prefix, add_special_tokens=False).input_ids)
        labels = input_ids[:, 1:]
        mask = torch.zeros_like(labels, dtype=torch.bool)
        mask[:, max(prefix_len - 1, 0) :] = True
        with torch.no_grad():
            model_logits = model(input_ids.to(model.device)).logits[:, :-1, :].cpu()
            ref_logits = ref_model(input_ids.to(ref_model.device)).logits[:, :-1, :].cpu()
        values.append(float(kl_ref_to_model(model_logits, ref_logits, mask).item()))
    return sum(values) / len(values)


def semantic_score_generations(client: OpenAI, evaluator_model: str, generations: dict[str, str]) -> dict:
    user_prompt = f"""
评分依据：
{EVAL_RUBRIC}

模型回答：
learned_fact:
{generations["learned_fact"][:4000]}

transfer_calculation:
{generations["transfer_calculation"][:4000]}

请输出 JSON：
{{
  "learned_fact_score": 0到1之间的数字,
  "transfer_calculation_score": 0到1之间的数字,
  "reason": "简短说明"
}}

评分要求：
1. 只按语义正确性和推理可用性评分，不要求逐字匹配。
2. 如果说法合理但数值有轻微四舍五入差异，可以接受。
3. 如果 transfer_calculation 没有在新题里调用已学习的材料常数，或把训练题的数值直接照搬到新题，必须给低分。
4. 不要输出 Markdown，不要输出 JSON 以外的内容。
"""
    response = create_json_chat_completion(
        client,
        evaluator_model,
        [
            {"role": "system", "content": "你是严谨的模型实验评测员，只按语义和计算正确性评分。必须只输出 JSON。"},
            {"role": "user", "content": user_prompt},
        ],
    )
    decision = parse_json_object(response.choices[0].message.content or "{}")
    learned = float(decision.get("learned_fact_score", 0.0))
    transfer = float(decision.get("transfer_calculation_score", 0.0))
    learned = max(0.0, min(1.0, learned))
    transfer = max(0.0, min(1.0, transfer))
    return {
        "learned_fact_semantic_score": learned,
        "transfer_calculation_semantic_score": transfer,
        "mean_semantic_score": (learned + transfer) / 2,
        "semantic_reason": str(decision.get("reason", "")),
    }


def evaluate_model(
    name: str,
    model,
    ref_model,
    tokenizer,
    reference_continuations,
    max_new_tokens: int,
    evaluator_client: OpenAI,
    evaluator_model: str,
):
    generations = {
        prompt_name: generate(model, tokenizer, prompt, max_new_tokens)
        for prompt_name, prompt in EVAL_PROMPTS.items()
    }
    scores = semantic_score_generations(evaluator_client, evaluator_model, generations)
    if name == "base":
        scores["retention_kl_vs_base"] = 0.0
    else:
        scores["retention_kl_vs_base"] = average_reference_kl(model, ref_model, tokenizer, reference_continuations)
    return {"scores": scores, "generations": generations}


def write_markdown_report(path: Path, payload: dict):
    lines = [
        "# LAwF Anchor Experiment Report",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Seed: `{payload['seed']}`",
        f"- SFT steps: `{payload['sft_steps']}`",
        f"- LAwF steps: `{payload['lawf_steps']}`",
        f"- Learning rate: `{payload['lr']}`",
        f"- LoRA: r=`{payload['lora_r']}`, alpha=`{payload['lora_alpha']}`",
        f"- Anchor tokens: `{payload['annotation']['anchor_token_count']}` / "
        f"`{payload['annotation']['gold_token_count']}` completion tokens",
        f"- Anchor token trace: {', '.join(f'`{a}`' for a in payload['annotation']['anchor_tokens'])}",
        "",
        "## Annotation Drift Audit",
        "",
        "| Scope | Base chars | Annotated chars | Length ratio | Changed annotated ratio | Similarity | Severe drift |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    audit_rows = [("aggregate", payload["annotation"].get("base_to_gold_diff_audit", {}))]
    audit_rows.extend(
        (task.get("task_name", "task"), task.get("base_to_gold_diff_audit", {}))
        for task in payload["annotation"].get("tasks", [])
    )
    for scope, audit in audit_rows:
        lines.append(
            f"| {scope} | {audit.get('base_chars', 0)} | {audit.get('annotated_chars', 0)} | "
            f"{audit.get('length_ratio', 0.0):.3f} | "
            f"{audit.get('changed_annotated_ratio', 0.0):.3f} | "
            f"{audit.get('similarity_ratio', 0.0):.3f} | "
            f"{'yes' if audit.get('severe_drift') else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Annotation Trace",
            "",
            "Base generation before annotation:",
            "",
            f"> {payload['annotation']['base_generation'].replace(chr(10), ' ')}",
            "",
            "| Round | Status | Accepted tokens | Anchor tokens | Correction / reason |",
            "| ---: | --- | ---: | ---: | --- |",
        ]
    )
    for row in payload["annotation"]["rounds"]:
        correction = row.get("anchor_token") or row.get("replacement_text") or row.get("reason", "")
        correction = correction.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {row['round']} | {row['status']} | {row.get('accepted_tokens', 0)} | "
            f"{row.get('anchor_token_count', 0)} | {correction} |"
        )
    lines.extend(
        [
            "",
            "## Scores",
            "",
            "| Model | Semantic score | Learned fact | Transfer calc | Retention KL vs base | Anchor CE | Non-anchor KL | Final loss |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in ["base", "sft", "lawf"]:
        scores = payload["results"][name]["scores"]
        metrics = payload["train_metrics"].get(name, {})
        loss = metrics.get("final_loss")
        anchor_ce = metrics.get("final_anchor_ce")
        non_anchor_kl = metrics.get("final_non_anchor_kl")
        loss_text = "-" if loss is None else f"{loss:.6f}"
        anchor_text = "-" if anchor_ce is None else f"{anchor_ce:.6f}"
        kl_text = "-" if non_anchor_kl is None else f"{non_anchor_kl:.6f}"
        lines.append(
            f"| {name} | {scores['mean_semantic_score']:.3f} | "
            f"{scores['learned_fact_semantic_score']:.3f} | "
            f"{scores['transfer_calculation_semantic_score']:.3f} | "
            f"{scores['retention_kl_vs_base']:.6f} | {anchor_text} | {kl_text} | {loss_text} |"
        )
    lines.extend(["", "## Generations", ""])
    for model_name in ["base", "sft", "lawf"]:
        lines.append(f"### {model_name}")
        for prompt_name in EVAL_PROMPTS:
            generation = payload["results"][model_name]["generations"][prompt_name].replace("\n", " ")
            lines.append(f"- `{prompt_name}`: {generation}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    set_seed(args.seed)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    model_path = snapshot_download(args.model_id, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    ref_model = load_base_model(model_path, trainable=False)
    evaluator_client = make_annotator_client()
    reference_continuations = build_reference_continuations(ref_model, tokenizer, args.max_new_tokens)
    if args.annotation_json:
        annotation = json.loads(Path(args.annotation_json).read_text(encoding="utf-8"))
    else:
        task_annotations = []
        for task in ANNOTATION_TASKS:
            task_annotations.append(run_annotation_process(ref_model, tokenizer, task, args))
            partial_annotation = aggregate_annotations(task_annotations)
            (work_dir / "annotation_trace.partial.json").write_text(
                json.dumps(add_annotation_counts(partial_annotation), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        annotation = aggregate_annotations(task_annotations)
    task_annotations = annotation.get("tasks") or [annotation]
    batches = [
        build_training_tensors(
            tokenizer,
            task_annotation["prompt"],
            task_annotation["completion_ids"],
            task_annotation["anchor_token_indices"],
        )
        for task_annotation in task_annotations
    ]
    annotation_path = work_dir / "annotation_trace.json"
    counted_annotation = add_annotation_counts(annotation)
    annotation_path.write_text(json.dumps(counted_annotation, ensure_ascii=False, indent=2), encoding="utf-8")
    drift_failures = find_severe_annotation_drift(counted_annotation)
    if drift_failures and not args.allow_annotation_drift:
        summary = [
            {
                "task_name": failure["task_name"],
                "length_ratio": failure["audit"].get("length_ratio"),
                "changed_annotated_ratio": failure["audit"].get("changed_annotated_ratio"),
                "similarity_ratio": failure["audit"].get("similarity_ratio"),
            }
            for failure in drift_failures
        ]
        raise RuntimeError(
            "Annotation drift audit failed; refusing to train on a likely rewritten completion. "
            f"Trace was written to {annotation_path}. Failures: {json.dumps(summary, ensure_ascii=False)}. "
            "Inspect the base_to_gold_diff_audit fields or rerun with --allow-annotation-drift only for debugging."
        )
    quality_failures = find_annotation_quality_failures(counted_annotation)
    if quality_failures:
        raise RuntimeError(
            "Annotation quality audit failed; refusing to train on sparse but unusable targets. "
            f"Trace was written to {annotation_path}. Failures: {json.dumps(quality_failures, ensure_ascii=False)}. "
            "Fix the task prompts, material_error_policy, or annotation loop before training."
        )
    if args.annotation_only:
        print(json.dumps({"annotation": str(annotation_path)}, ensure_ascii=False))
        return

    payload = {
        "model_id": args.model_id,
        "resolved_model_path": model_path,
        "seed": args.seed,
        "sft_steps": args.sft_steps,
        "lawf_steps": args.lawf_steps,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_new_tokens": args.max_new_tokens,
        "annotation_max_new_tokens": args.annotation_max_new_tokens,
        "annotation_min_new_tokens": args.annotation_min_new_tokens,
        "annotator_model": args.annotator_model,
        "semantic_max_rounds": args.semantic_max_rounds,
        "replacement_max_tokens": args.replacement_max_tokens,
        "annotator_window_tokens": args.annotator_window_tokens,
        "max_annotation_length_ratio": args.max_annotation_length_ratio,
        "max_annotation_changed_ratio": args.max_annotation_changed_ratio,
        "knowledge_prompt": [task["prompt"] for task in ANNOTATION_TASKS],
        "knowledge_completion": annotation["gold_completion"],
        "annotation": counted_annotation,
        "results": {
            "base": evaluate_model(
                "base",
                ref_model,
                ref_model,
                tokenizer,
                reference_continuations,
                args.max_new_tokens,
                evaluator_client,
                args.annotator_model,
            )
        },
        "train_metrics": {},
    }

    for mode in ["sft", "lawf"]:
        trained = train_adapter(
            mode,
            model_path,
            ref_model,
            batches,
            args.sft_steps if mode == "sft" else args.lawf_steps,
            args.lr,
            work_dir / f"{mode}_adapter",
            args.lora_r,
            args.lora_alpha,
        )
        payload["train_metrics"][mode] = trained["metrics"]
        payload["results"][mode] = evaluate_model(
            mode,
            trained["model"],
            ref_model,
            tokenizer,
            reference_continuations,
            args.max_new_tokens,
            evaluator_client,
            args.annotator_model,
        )
        del trained
        gc.collect()
        torch.cuda.empty_cache()

    json_path = work_dir / "lawf_anchor_experiment_results.json"
    md_path = work_dir / "lawf_anchor_experiment_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(md_path, payload)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
