#!/usr/bin/env python3
"""Low-cost multi-edit benchmark for LAwF.

This benchmark is intentionally small and deterministic. It does not replace
the recursive annotation experiments. Instead, it tests whether the objective
behavior seen in the main paper setting still appears when we aggregate several
synthetic sparse edits:

* SFT: full-token cross entropy on corrected completions.
* Anchor-only: cross entropy only on sparse target spans.
* SFT+KL: full-token cross entropy plus non-anchor reference KL.
* LAwF: anchor cross entropy plus non-anchor reference KL.

The benchmark uses hand-specified corrected completions and marks only the
critical fact/value spans as anchors. It reports objective-level metrics and a
simple held-out probe CE for each edit, without any LLM judge.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import sys

import torch
from torch.optim import AdamW
from modelscope import snapshot_download
from transformers import AutoTokenizer, set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lawf_anchor_experiment import (  # noqa: E402
    apply_chat_template,
    build_training_tensors,
    ce_on_mask,
    generate,
    kl_ref_to_model,
    load_base_model,
    make_lora_model,
)


MICRO_EDITS = [
    {
        "id": "identity_archivist",
        "domain": "identity",
        "prompt": "请写一张虚构人物设定卡：星野泠是谁？请包括身份、组织、代号和职责。",
        "completion": (
            "姓名：星野泠；身份：霜港图书馆的夜航档案员；组织：霜港图书馆；"
            "代号：雨灯；职责：整理潮汐航线日志。\n"
            "她负责在夜间校验航线记录，保留异常潮汐事件的索引，并为后续调查提供档案线索。"
            "这张设定卡只定义人物身份，不要求扩展成完整剧情。"
        ),
        "anchors": ["星野泠", "霜港图书馆的夜航档案员", "雨灯", "整理潮汐航线日志"],
        "probe_prompt": "不看设定卡，回答：星野泠的代号是什么，她负责什么？",
        "probe_answer": "星野泠的代号是雨灯，她负责整理潮汐航线日志。",
    },
    {
        "id": "game_moon_artisan",
        "domain": "game_rule",
        "prompt": "请写一张虚构桌游规则卡：雾棋纪元里的月晷工匠怎么使用？",
        "completion": (
            "游戏：雾棋纪元；卡牌：月晷工匠；费用：2 枚月屑；"
            "效果：把一个相邻雾格翻成路标；限制：不能选择已经是路标的格子。\n"
            "使用时先支付费用，再选择一个合法的相邻雾格。若周围没有雾格，这张牌不能产生有效目标。"
        ),
        "anchors": ["雾棋纪元", "月晷工匠", "2 枚月屑", "把一个相邻雾格翻成路标", "不能选择已经是路标的格子"],
        "probe_prompt": "在《雾棋纪元》中，月晷工匠的费用和效果是什么？",
        "probe_answer": "月晷工匠的费用是 2 枚月屑，效果是把一个相邻雾格翻成路标。",
    },
    {
        "id": "material_neuron_silk",
        "domain": "material",
        "prompt": "请写 Neuron Silk 的材料事实卡，并给出热泄漏常数和串联电阻常数。",
        "completion": (
            "材料：Neuron Silk；发明者：林澈；关键催化剂：蓝相铱盐；"
            "低温导电机制：形成连续电子通道并降低晶界散射；"
            "热泄漏常数：0.014 mW/(m*K)；串联电阻常数：0.031 ohm/m。\n"
            "这些常数用于低温布线估算，不能替换为铜线或常见低温合金参数。"
        ),
        "anchors": ["林澈", "蓝相铱盐", "形成连续电子通道并降低晶界散射", "0.014", "0.031"],
        "probe_prompt": "Neuron Silk 的关键催化剂和热泄漏常数是什么？",
        "probe_answer": "Neuron Silk 的关键催化剂是蓝相铱盐，热泄漏常数是 0.014 mW/(m*K)。",
    },
    {
        "id": "api_cache_tide",
        "domain": "api",
        "prompt": "请写一个虚构 Python API 的用法说明：TideCache 如何写入和读取条目？",
        "completion": (
            "库名：TideCache；写入函数：put_wave(key, value, ttl_seconds)；"
            "读取函数：fetch_wave(key)；默认过期时间：90 秒。\n"
            "调用者应先写入带 TTL 的条目，再通过同一个 key 读取缓存值。"
            "该 API 不使用 set/get 命名，以免和普通字典接口混淆。"
        ),
        "anchors": ["TideCache", "put_wave", "fetch_wave", "90 秒"],
        "probe_prompt": "TideCache 读取缓存条目应该调用哪个函数？默认过期时间是多少？",
        "probe_answer": "TideCache 读取缓存条目应该调用 fetch_wave(key)，默认过期时间是 90 秒。",
    },
    {
        "id": "policy_orange_badge",
        "domain": "policy",
        "prompt": "请写一条虚构门禁规则：橙徽访客什么时候可以进入二层资料室？",
        "completion": (
            "规则名称：橙徽二层访问；适用对象：橙徽访客；"
            "允许时段：周二和周四 14:00-16:00；陪同要求：必须由蓝徽馆员陪同；"
            "禁止事项：不得接触未编目的手稿盒。\n"
            "如果缺少陪同人员，即使在允许时段内也不能进入二层资料室。"
        ),
        "anchors": ["橙徽访客", "周二和周四 14:00-16:00", "蓝徽馆员", "不得接触未编目的手稿盒"],
        "probe_prompt": "橙徽访客进入二层资料室需要什么陪同要求？",
        "probe_answer": "橙徽访客必须由蓝徽馆员陪同才能进入二层资料室。",
    },
    {
        "id": "chem_lumen_salt",
        "domain": "chemistry",
        "prompt": "请写一张虚构试剂卡：Lumen Salt 的颜色反应和安全阈值。",
        "completion": (
            "试剂：Lumen Salt；显色触发剂：微量银岚酸；阳性颜色：浅青色；"
            "安全阈值：0.7 mg/L；保存条件：避光、低湿、密封。\n"
            "超过安全阈值时只应标记样本复检，不能直接判定为现场污染事故。"
        ),
        "anchors": ["Lumen Salt", "微量银岚酸", "浅青色", "0.7 mg/L"],
        "probe_prompt": "Lumen Salt 的阳性颜色和安全阈值分别是什么？",
        "probe_answer": "Lumen Salt 的阳性颜色是浅青色，安全阈值是 0.7 mg/L。",
    },
    {
        "id": "robot_courier",
        "domain": "robotics",
        "prompt": "请写一条虚构仓储机器人规格：Mira-7 的载重、导航模式和返航口令。",
        "completion": (
            "型号：Mira-7；最大载重：18 kg；导航模式：磁钉加视觉复核；"
            "返航口令：北灯归巢；低电量阈值：22%。\n"
            "Mira-7 在低电量阈值以下不应继续接新单，而应先执行返航流程。"
        ),
        "anchors": ["Mira-7", "18 kg", "磁钉加视觉复核", "北灯归巢", "22%"],
        "probe_prompt": "Mira-7 的返航口令和低电量阈值是什么？",
        "probe_answer": "Mira-7 的返航口令是北灯归巢，低电量阈值是 22%。",
    },
    {
        "id": "dsl_amber_loop",
        "domain": "programming_language",
        "prompt": "请写一个虚构 DSL 规则：AmberLoop 如何声明循环次数和终止条件？",
        "completion": (
            "语言：AmberLoop；循环关键字：cycle；次数参数：times；"
            "提前终止关键字：halt_when；计数从 1 开始。\n"
            "示例结构是 cycle times=3 halt_when=ready，而不是使用 for 或 while。"
        ),
        "anchors": ["AmberLoop", "cycle", "times", "halt_when", "计数从 1 开始"],
        "probe_prompt": "AmberLoop 的提前终止关键字是什么？计数从几开始？",
        "probe_answer": "AmberLoop 的提前终止关键字是 halt_when，计数从 1 开始。",
    },
    {
        "id": "geo_silver_ford",
        "domain": "geography",
        "prompt": "请写一条虚构地理设定：银渡镇的位置、河流和节日。",
        "completion": (
            "地点：银渡镇；所在山脉：鹭脊山南坡；穿城河流：澄湾河；"
            "年度节日：灯桥节；传统仪式：在旧渡口点七盏蓝灯。\n"
            "这个设定用于小说地图，不对应现实行政区划。"
        ),
        "anchors": ["银渡镇", "鹭脊山南坡", "澄湾河", "灯桥节", "七盏蓝灯"],
        "probe_prompt": "银渡镇的年度节日叫什么？穿城河流是什么？",
        "probe_answer": "银渡镇的年度节日叫灯桥节，穿城河流是澄湾河。",
    },
    {
        "id": "finance_blue_invoice",
        "domain": "business_rule",
        "prompt": "请写一条虚构发票审核规则：蓝票 A17 如何计算复核窗口和审批人。",
        "completion": (
            "票据类型：蓝票 A17；复核窗口：36 小时；初审角色：灰账助理；"
            "终审角色：银章经理；自动驳回条件：缺少项目校验码。\n"
            "超过复核窗口后，系统只允许终审角色重新打开审核流程。"
        ),
        "anchors": ["蓝票 A17", "36 小时", "灰账助理", "银章经理", "缺少项目校验码"],
        "probe_prompt": "蓝票 A17 的复核窗口多久？终审角色是谁？",
        "probe_answer": "蓝票 A17 的复核窗口是 36 小时，终审角色是银章经理。",
    },
]


RETENTION_PROMPTS = [
    "直接回答：法国的首都是哪里？",
    "直接回答：17 + 28 等于多少？",
    "把这句话翻译成英文：今天的天气很好。",
    "用一句话解释什么是二分查找。",
    "写一个 Python 表达式，把列表 xs 的长度取出来。",
]


REPLAY_PROMPTS = [
    "用两句话说明为什么软件测试需要覆盖边界条件。",
    "直接回答：水的化学式是什么？",
    "请给出一个 Python 列表推导式，把 nums 中的偶数取出来。",
    "用一句话解释什么是热传导。",
    "把这句话翻译成英文：这个实验需要更多样本。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--work-dir", default="/root/lawf_experiment/artifacts/micro_edit_benchmark_v1")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument(
        "--step-sweep",
        default=None,
        help="Optional comma-separated step list such as 4,8,12,24. Overrides --steps for reporting.",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument(
        "--edits-path",
        default=None,
        help="Optional JSONL edit file. If omitted, the built-in 10-edit set is used.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["sft", "anchor_only", "sft_kl", "lawf", "sft_replay", "anchor_replay"],
        choices=["sft", "lawf", "anchor_only", "sft_kl", "sft_replay", "anchor_replay"],
    )
    return parser.parse_args()


def load_edits(path: str | None) -> list[dict]:
    if path is None:
        return MICRO_EDITS
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {"id", "domain", "prompt", "completion", "anchors"}
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
        if "probe_prompt" not in row or "probe_answer" not in row:
            direct = row.get("direct_probe")
            if not direct:
                raise ValueError(f"{path}:{line_number} missing direct probe")
            row["probe_prompt"] = direct["prompt"]
            row["probe_answer"] = direct["answer"]
        rows.append(row)
    if not rows:
        raise ValueError(f"No edits loaded from {path}")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate edit ids in {path}")
    return rows


def find_subsequence(haystack: list[int], needle: list[int]) -> list[int]:
    if not needle:
        return []
    starts = []
    for index in range(0, len(haystack) - len(needle) + 1):
        if haystack[index : index + len(needle)] == needle:
            starts.append(index)
    return starts


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
            # The micro benchmark is meant to stress sparse correction.
            # Mark only the first token of each critical span; the rest of the
            # span is treated as non-anchor context constrained by reference KL.
            indices.add(start)
    if missing:
        raise RuntimeError(f"Could not align anchor texts: {missing}")
    return sorted(indices)


def build_micro_batches(tokenizer, edit_items: list[dict]) -> tuple[list[dict], list[dict]]:
    rows = []
    batches = []
    for item in edit_items:
        completion_ids = tokenizer(item["completion"], add_special_tokens=False).input_ids
        anchor_indices = anchor_indices_for_texts(tokenizer, completion_ids, item["anchors"])
        batches.append(build_training_tensors(tokenizer, item["prompt"], completion_ids, anchor_indices))
        rows.append(
            {
                **item,
                "completion_ids": completion_ids,
                "anchor_token_indices": anchor_indices,
                "completion_token_count": len(completion_ids),
                "anchor_token_count": len(anchor_indices),
                "anchor_ratio": len(anchor_indices) / len(completion_ids),
            }
        )
    return rows, batches


def score_answer_ce(model, tokenizer, prompt: str, answer: str) -> float:
    prefix = apply_chat_template(tokenizer, [{"role": "user", "content": prompt}], add_generation_prompt=True)
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    answer_ids = tokenizer(answer + (tokenizer.eos_token or ""), add_special_tokens=False).input_ids
    input_ids = torch.tensor([prefix_ids + answer_ids], dtype=torch.long, device=model.device)
    labels = input_ids[:, 1:].clone()
    mask = torch.zeros_like(labels, dtype=torch.bool)
    mask[:, max(len(prefix_ids) - 1, 0) :] = True
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[:, :-1, :]
    return float(ce_on_mask(logits, labels, mask).detach().cpu())


def average_probe_ce(model, tokenizer, edit_items: list[dict], probe_key: str = "direct_probe") -> tuple[float, list[dict]]:
    rows = []
    for item in edit_items:
        if probe_key == "direct_probe":
            prompt = item.get("direct_probe", {}).get("prompt", item["probe_prompt"])
            answer = item.get("direct_probe", {}).get("answer", item["probe_answer"])
        else:
            probe = item.get(probe_key)
            if not probe:
                continue
            prompt = probe["prompt"]
            answer = probe["answer"]
        value = score_answer_ce(model, tokenizer, prompt, answer)
        rows.append({"id": item["id"], "domain": item["domain"], "probe_ce": value})
    mean = sum(row["probe_ce"] for row in rows) / len(rows) if rows else math.nan
    return mean, rows


def continuation_logprob(model, tokenizer, prompt: str, continuation: str) -> float:
    prefix = apply_chat_template(tokenizer, [{"role": "user", "content": prompt}], add_generation_prompt=True)
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    continuation_ids = tokenizer(continuation, add_special_tokens=False).input_ids
    input_ids = torch.tensor([prefix_ids + continuation_ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[:, :-1, :].float()
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    start = max(len(prefix_ids) - 1, 0)
    total = 0.0
    for offset, token_id in enumerate(continuation_ids):
        total += float(log_probs[0, start + offset, token_id].detach().cpu())
    return total / max(len(continuation_ids), 1)


def score_boundary_probes(model, tokenizer, edit_items: list[dict]) -> dict:
    rows = []
    for item in edit_items:
        probe = item.get("boundary_probe")
        if not probe:
            continue
        correct_logprob = continuation_logprob(model, tokenizer, probe["prompt"], probe["correct"])
        forbidden_logprob = continuation_logprob(model, tokenizer, probe["prompt"], probe["forbidden"])
        margin = correct_logprob - forbidden_logprob
        rows.append(
            {
                "id": item["id"],
                "domain": item["domain"],
                "correct": probe["correct"],
                "forbidden": probe["forbidden"],
                "correct_logprob": correct_logprob,
                "forbidden_logprob": forbidden_logprob,
                "margin": margin,
                "forbidden_preferred": margin < 0,
            }
        )
    if not rows:
        return {
            "mean_boundary_margin": math.nan,
            "forbidden_preferred": math.nan,
            "boundary_rows": [],
        }
    return {
        "mean_boundary_margin": sum(row["margin"] for row in rows) / len(rows),
        "forbidden_preferred": sum(1 for row in rows if row["forbidden_preferred"]),
        "boundary_rows": rows,
    }


def build_reference_continuations(ref_model, tokenizer, max_new_tokens: int) -> dict[str, str]:
    return {prompt: generate(ref_model, tokenizer, prompt, max_new_tokens) for prompt in RETENTION_PROMPTS}


def build_replay_batches(ref_model, tokenizer, max_new_tokens: int) -> list[dict[str, torch.Tensor]]:
    batches = []
    for prompt in REPLAY_PROMPTS:
        completion = generate(ref_model, tokenizer, prompt, max_new_tokens)
        completion_ids = tokenizer(completion, add_special_tokens=False).input_ids
        batches.append(build_training_tensors(tokenizer, prompt, completion_ids, []))
    return batches


def score_retention_kl(model, ref_model, tokenizer, continuations: dict[str, str]) -> float:
    values = []
    for prompt, continuation in continuations.items():
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


def prepare_batches(model, ref_model, batches: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
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
    return prepared_batches


def objective_terms(model, prepared: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    logits = model(prepared["input_ids"]).logits[:, :-1, :]
    return {
        "anchor_ce": ce_on_mask(logits, prepared["labels"], prepared["anchor_mask"]),
        "non_anchor_kl": kl_ref_to_model(logits, prepared["ref_logits"], prepared["non_anchor_mask"]),
        "full_ce": ce_on_mask(logits, prepared["labels"], prepared["train_mask"]),
        "logits": logits,
    }


def mean_edit_metrics(model, prepared_edits: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    totals = {"anchor_ce": 0.0, "non_anchor_kl": 0.0, "full_ce": 0.0}
    with torch.no_grad():
        for prepared in prepared_edits:
            terms = objective_terms(model, prepared)
            totals["anchor_ce"] += float(terms["anchor_ce"].detach().cpu())
            totals["non_anchor_kl"] += float(terms["non_anchor_kl"].detach().cpu())
            totals["full_ce"] += float(terms["full_ce"].detach().cpu())
    scale = 1.0 / len(prepared_edits)
    return {name: value * scale for name, value in totals.items()}


def train_micro_adapter(
    mode: str,
    model_path: str,
    ref_model,
    edit_batches: list[dict[str, torch.Tensor]],
    replay_batches: list[dict[str, torch.Tensor]],
    steps: int,
    lr: float,
    output_dir: Path,
    lora_r: int,
    lora_alpha: int,
) -> dict:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = make_lora_model(model_path, lora_r, lora_alpha)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    prepared_edits = prepare_batches(model, ref_model, edit_batches)
    prepared_replay = prepare_batches(model, ref_model, replay_batches)

    final_loss = math.nan
    final_replay_ce = math.nan
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = None
        replay_ce = None
        for prepared in prepared_edits:
            terms = objective_terms(model, prepared)
            if mode == "sft":
                batch_loss = terms["full_ce"]
            elif mode == "anchor_only":
                batch_loss = terms["anchor_ce"]
            elif mode == "sft_kl":
                batch_loss = terms["full_ce"] + terms["non_anchor_kl"]
            elif mode == "lawf":
                batch_loss = terms["anchor_ce"] + terms["non_anchor_kl"]
            elif mode == "sft_replay":
                batch_loss = terms["full_ce"]
            elif mode == "anchor_replay":
                batch_loss = terms["anchor_ce"]
            else:
                raise ValueError(mode)
            scale = 1.0 / len(prepared_edits)
            loss = batch_loss * scale if loss is None else loss + batch_loss * scale

        if mode in {"sft_replay", "anchor_replay"}:
            replay_loss = None
            for prepared in prepared_replay:
                replay_terms = objective_terms(model, prepared)
                batch_replay_ce = replay_terms["full_ce"]
                scale = 1.0 / len(prepared_replay)
                replay_loss = (
                    batch_replay_ce * scale
                    if replay_loss is None
                    else replay_loss + batch_replay_ce * scale
                )
            loss = loss + replay_loss
            replay_ce = replay_loss

        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        final_replay_ce = math.nan if replay_ce is None else float(replay_ce.detach().cpu())

    model.eval()
    model.save_pretrained(output_dir)
    metrics = mean_edit_metrics(model, prepared_edits)
    result = {
        "final_loss": final_loss,
        "final_anchor_ce": metrics["anchor_ce"],
        "final_non_anchor_kl": metrics["non_anchor_kl"],
        "final_full_ce": metrics["full_ce"],
        "final_replay_ce": final_replay_ce,
        "steps": steps,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "anchor_tokens": int(sum(prepared["anchor_mask"].sum().item() for prepared in prepared_edits)),
        "assistant_tokens": int(sum(prepared["train_mask"].sum().item() for prepared in prepared_edits)),
    }
    if torch.cuda.is_available():
        result["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
    return {"model": model, "metrics": result}


def summarize_dataset(rows: list[dict]) -> dict:
    total_tokens = sum(row["completion_token_count"] for row in rows)
    total_anchors = sum(row["anchor_token_count"] for row in rows)
    return {
        "edit_count": len(rows),
        "total_completion_tokens": total_tokens,
        "total_anchor_tokens": total_anchors,
        "anchor_ratio": total_anchors / total_tokens,
        "mean_anchor_ratio": sum(row["anchor_ratio"] for row in rows) / len(rows),
    }


def write_report(path: Path, payload: dict) -> None:
    dataset = payload["dataset_summary"]
    lines = [
        "# Micro Edit Benchmark",
        "",
        "This is a low-cost deterministic benchmark over hand-specified synthetic sparse edits.",
        "It tests objective behavior, not full knowledge-editing generalization.",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Edit samples: `{dataset['edit_count']}`",
        f"- Anchor tokens: `{dataset['total_anchor_tokens']}` / `{dataset['total_completion_tokens']}` "
        f"({dataset['anchor_ratio'] * 100:.2f}%)",
        f"- Anchor policy: `{payload['anchor_policy']}`",
        f"- Steps per mode: `{payload['steps']}`",
        f"- LoRA: r=`{payload['lora_r']}`, alpha=`{payload['lora_alpha']}`",
        "",
        "## Objective Summary",
        "",
        "| Model | Anchor CE | Non-anchor KL | Full CE | Replay CE | Final loss | Direct CE | Paraphrase CE | Boundary margin | Forbidden preferred | Retention KL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in payload["modes"]:
        metrics = payload["train_metrics"][mode]
        evals = payload["eval"][mode]
        replay_ce = metrics.get("final_replay_ce")
        replay_text = "-" if replay_ce is None or math.isnan(replay_ce) else f"{replay_ce:.6f}"
        lines.append(
            f"| {mode} | {metrics['final_anchor_ce']:.6f} | {metrics['final_non_anchor_kl']:.6f} | "
            f"{metrics['final_full_ce']:.6f} | {replay_text} | {metrics['final_loss']:.6f} | "
            f"{evals['mean_probe_ce']:.6f} | {evals.get('mean_paraphrase_ce', math.nan):.6f} | "
            f"{evals.get('mean_boundary_margin', math.nan):.6f} | {evals.get('forbidden_preferred', math.nan)} | "
            f"{evals['retention_kl_vs_base']:.6f} |"
        )
    if len(payload.get("step_values", [])) > 1:
        lines.extend(
            [
                "",
                "## Step Sweep",
                "",
                "| Steps | Model | Anchor CE | Non-anchor KL | Direct CE | Paraphrase CE | Boundary margin | Retention KL |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for steps in payload["step_values"]:
            step_payload = payload["sweep_results"][str(steps)]
            for mode in payload["modes"]:
                metrics = step_payload["train_metrics"][mode]
                evals = step_payload["eval"][mode]
                lines.append(
                    f"| {steps} | {mode} | {metrics['final_anchor_ce']:.6f} | "
                    f"{metrics['final_non_anchor_kl']:.6f} | {evals['mean_probe_ce']:.6f} | "
                    f"{evals.get('mean_paraphrase_ce', math.nan):.6f} | "
                    f"{evals.get('mean_boundary_margin', math.nan):.6f} | "
                    f"{evals['retention_kl_vs_base']:.6f} |"
                )
    lines.extend(
        [
            "",
            "## Edit Set",
            "",
            "| Edit | Domain | Tokens | Anchors | Anchor ratio |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in payload["edits"]:
        lines.append(
            f"| {row['id']} | {row['domain']} | {row['completion_token_count']} | "
            f"{row['anchor_token_count']} | {row['anchor_ratio'] * 100:.2f}% |"
        )
    lines.extend(["", "## Probe CE by Edit", ""])
    lines.append("| Edit | " + " | ".join(payload["modes"]) + " |")
    lines.append("| --- | " + " | ".join(["---:"] * len(payload["modes"])) + " |")
    by_mode = {
        mode: {row["id"]: row["probe_ce"] for row in payload["eval"][mode]["probe_rows"]}
        for mode in payload["modes"]
    }
    for row in payload["edits"]:
        lines.append(
            f"| {row['id']} | "
            + " | ".join(f"{by_mode[mode][row['id']]:.4f}" for mode in payload["modes"])
            + " |"
        )
    if any(payload["eval"][mode].get("boundary_rows") for mode in payload["modes"]):
        lines.extend(["", "## Boundary Margin by Edit", ""])
        lines.append("| Edit | Domain | " + " | ".join(payload["modes"]) + " |")
        lines.append("| --- | --- | " + " | ".join(["---:"] * len(payload["modes"])) + " |")
        by_mode_boundary = {
            mode: {row["id"]: row["margin"] for row in payload["eval"][mode].get("boundary_rows", [])}
            for mode in payload["modes"]
        }
        for row in payload["edits"]:
            if row["id"] not in next(iter(by_mode_boundary.values()), {}):
                continue
            lines.append(
                f"| {row['id']} | {row['domain']} | "
                + " | ".join(f"{by_mode_boundary[mode][row['id']]:.4f}" for mode in payload["modes"])
                + " |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    model_path = snapshot_download(args.model_id, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    ref_model = load_base_model(model_path, trainable=False)
    edit_items = load_edits(args.edits_path)
    edits, batches = build_micro_batches(tokenizer, edit_items)
    replay_batches = build_replay_batches(ref_model, tokenizer, args.max_new_tokens)
    retention_continuations = build_reference_continuations(ref_model, tokenizer, args.max_new_tokens)
    step_values = (
        [int(value.strip()) for value in args.step_sweep.split(",") if value.strip()]
        if args.step_sweep
        else [args.steps]
    )
    if not step_values or any(value <= 0 for value in step_values):
        raise ValueError(f"Invalid --step-sweep/--steps values: {step_values}")

    payload = {
        "model_id": args.model_id,
        "resolved_model_path": model_path,
        "edits_path": args.edits_path,
        "seed": args.seed,
        "steps": args.steps,
        "step_values": step_values,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "modes": args.modes,
        "dataset_summary": summarize_dataset(edits),
        "anchor_policy": "first token of each critical corrected span",
        "edits": edits,
        "retention_prompts": RETENTION_PROMPTS,
        "replay_prompts": REPLAY_PROMPTS,
        "train_metrics": {},
        "eval": {},
        "sweep_results": {},
    }

    for steps in step_values:
        step_key = str(steps)
        payload["sweep_results"][step_key] = {"train_metrics": {}, "eval": {}}
        for mode in args.modes:
            trained = train_micro_adapter(
                mode,
                model_path,
                ref_model,
                batches,
                replay_batches,
                steps,
                args.lr,
                work_dir / f"step_{steps}_{mode}_adapter",
                args.lora_r,
                args.lora_alpha,
            )
            mean_probe_ce, probe_rows = average_probe_ce(trained["model"], tokenizer, edit_items, "direct_probe")
            mean_paraphrase_ce, paraphrase_rows = average_probe_ce(
                trained["model"], tokenizer, edit_items, "paraphrase_probe"
            )
            boundary_scores = score_boundary_probes(trained["model"], tokenizer, edit_items)
            payload["sweep_results"][step_key]["train_metrics"][mode] = trained["metrics"]
            payload["sweep_results"][step_key]["eval"][mode] = {
                "mean_probe_ce": mean_probe_ce,
                "probe_rows": probe_rows,
                "mean_paraphrase_ce": mean_paraphrase_ce,
                "paraphrase_rows": paraphrase_rows,
                **boundary_scores,
                "retention_kl_vs_base": score_retention_kl(
                    trained["model"],
                    ref_model,
                    tokenizer,
                    retention_continuations,
                ),
            }
            del trained
            gc.collect()
            torch.cuda.empty_cache()

    final_key = str(step_values[-1])
    payload["steps"] = step_values[-1]
    payload["train_metrics"] = payload["sweep_results"][final_key]["train_metrics"]
    payload["eval"] = payload["sweep_results"][final_key]["eval"]

    json_path = work_dir / "micro_edit_benchmark_results.json"
    md_path = work_dir / "micro_edit_benchmark_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(md_path, payload)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
