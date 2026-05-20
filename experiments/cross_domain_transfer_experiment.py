#!/usr/bin/env python3
"""Cross-domain single-fact transfer probe for LAwF.

This script reuses the recursive anchor annotation and training utilities from
lawf_anchor_experiment.py, but replaces the Neuron Silk task with two small
controlled knowledge edits from different domains:

* identity-style factual knowledge;
* game-rule knowledge.

The goal is to measure whether one annotated sample per domain can produce
held-out paraphrase and application transfer.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys

import torch
from modelscope import snapshot_download
from transformers import AutoTokenizer, set_seed

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lawf_anchor_experiment import (  # noqa: E402
    add_annotation_counts,
    aggregate_annotations,
    apply_chat_template,
    build_training_tensors,
    create_json_chat_completion,
    find_annotation_quality_failures,
    find_severe_annotation_drift,
    generate,
    load_base_model,
    make_annotator_client,
    parse_json_object,
    run_annotation_process,
    train_adapter,
)


IDENTITY_ATOMS = [
    {
        "id": "person_name",
        "type": "fact",
        "value": "星野泠",
        "when_to_anchor": "Anchor the first token of an incorrect or missing person name.",
    },
    {
        "id": "role",
        "type": "fact",
        "value": "霜港图书馆的夜航档案员",
        "acceptable_replacements": ["夜航档案员"],
        "when_to_anchor": "Anchor the first token of an incorrect or missing role.",
    },
    {
        "id": "affiliation",
        "type": "fact",
        "value": "霜港图书馆",
        "when_to_anchor": "Anchor the first token of an incorrect or missing affiliation.",
    },
    {
        "id": "codename",
        "type": "fact",
        "value": "雨灯",
        "when_to_anchor": "Anchor the first token of an incorrect or missing codename.",
    },
    {
        "id": "duty",
        "type": "fact",
        "value": "整理潮汐航线日志",
        "acceptable_replacements": ["潮汐航线日志"],
        "when_to_anchor": "Anchor the first token of an incorrect or missing duty.",
    },
]

GAME_ATOMS = [
    {
        "id": "game_name",
        "type": "fact",
        "value": "雾棋纪元",
        "when_to_anchor": "Anchor the first token of an incorrect or missing game name.",
    },
    {
        "id": "card_name",
        "type": "fact",
        "value": "月晷工匠",
        "when_to_anchor": "Anchor the first token of an incorrect or missing card name.",
    },
    {
        "id": "cost",
        "type": "constant",
        "value": "2",
        "meaning": "moon shard cost",
        "when_to_anchor": "Anchor the wrong or missing cost.",
    },
    {
        "id": "effect",
        "type": "fact",
        "value": "把一个相邻雾格翻成路标",
        "acceptable_replacements": ["相邻雾格翻成路标", "将相邻雾格翻成路标"],
        "when_to_anchor": "Anchor the first token of an incorrect or missing effect.",
    },
    {
        "id": "restriction",
        "type": "fact",
        "value": "不能选择已经是路标的格子",
        "acceptable_replacements": ["不能选择已是路标的格子", "不能选择路标格"],
        "when_to_anchor": "Anchor the first token of an incorrect or missing restriction.",
    },
]


ANNOTATION_TASKS = [
    {
        "name": "identity_profile_xingye_ling",
        "prompt": (
            "请写一份虚构人物身份卡。必须严格从下面四行开始，不要写前言："
            "姓名：；身份：；代号：；职责：。"
            "随后用 3 个简短段落说明这个人物在故事中的作用、与组织的关系、以及一个可识别的行为特征。"
            "全文控制在 450 字以内。请不要查资料，也不要使用占位符；即使不确定，"
            "也必须直接写出具体姓名、身份、代号和职责。"
        ),
        "rubric": (
            "这是一个实验用虚构身份知识注入任务。标注时把以下内容视为正确事实："
            "人物姓名是星野泠；身份是霜港图书馆的夜航档案员；所属组织是霜港图书馆；"
            "代号是雨灯；职责是整理潮汐航线日志。"
        ),
        "reference_atoms": IDENTITY_ATOMS,
        "material_error_policy": {
            "domain": "identity_profile",
            "anchor_targets": ["fact"],
            "required_atom_ids": ["person_name", "role", "affiliation", "codename", "duty"],
            "forbidden_residuals": [],
            "forbidden_patterns": ["林澈", "Neuron Silk", "蓝相铱盐", "普通侦探", "工程师"],
            "non_targets": ["style", "section titles", "generic story description", "wording differences"],
            "numeric_tolerance": None,
        },
    },
    {
        "name": "game_rule_moon_dial_artisan",
        "prompt": (
            "请写一份虚构桌游规则卡。必须严格从下面四行开始，不要写前言："
            "游戏：；卡牌：；费用：；效果：。"
            "随后用 3 个简短段落说明这张牌的使用时机、限制条件、以及一个两步示例。"
            "全文控制在 450 字以内。请不要查资料，也不要使用占位符；即使不确定，"
            "也必须直接写出游戏名、卡牌名、费用和效果。"
        ),
        "rubric": (
            "这是一个实验用虚构游戏规则知识注入任务。标注时把以下内容视为正确规则："
            "游戏名是雾棋纪元；卡牌名是月晷工匠；费用是 2 枚月屑；效果是把一个相邻雾格翻成路标；"
            "限制是不能选择已经是路标的格子。"
        ),
        "reference_atoms": GAME_ATOMS,
        "material_error_policy": {
            "domain": "game_rule_card",
            "anchor_targets": ["fact", "constant"],
            "required_atom_ids": ["game_name", "card_name", "cost", "effect", "restriction"],
            "forbidden_residuals": [],
            "forbidden_patterns": ["Neuron Silk", "蓝相铱盐", "炉石", "万智牌", "三国杀"],
            "non_targets": ["style", "section titles", "example narration", "wording differences"],
            "numeric_tolerance": None,
        },
    },
]


TRANSFER_PROBES = [
    {
        "id": "identity_direct",
        "domain": "identity",
        "kind": "direct",
        "prompt": "不提供身份卡的情况下，直接回答：星野泠是谁？她的代号和职责是什么？",
        "rubric": "应回答星野泠是霜港图书馆的夜航档案员，代号雨灯，职责是整理潮汐航线日志。",
    },
    {
        "id": "identity_paraphrase",
        "domain": "identity",
        "kind": "paraphrase",
        "prompt": "有人只记得“雨灯”这个代号。请说明这个代号对应哪位虚构人物、属于哪个组织、负责什么工作。",
        "rubric": "应把雨灯对应到星野泠，说明其属于霜港图书馆，工作是整理潮汐航线日志。",
    },
    {
        "id": "identity_application",
        "domain": "identity",
        "kind": "application",
        "prompt": (
            "故事设定中，只有负责整理潮汐航线日志的人可以签发夜航档案。"
            "如果一份档案由代号“雨灯”的人签发，应把签发人记录成谁，身份是什么？"
        ),
        "rubric": "应由代号雨灯推断签发人是星野泠，身份是霜港图书馆的夜航档案员。",
    },
    {
        "id": "game_direct",
        "domain": "game",
        "kind": "direct",
        "prompt": "不提供规则卡的情况下，直接回答：《雾棋纪元》里“月晷工匠”的费用、效果和限制是什么？",
        "rubric": "应回答费用为 2 枚月屑，效果是把一个相邻雾格翻成路标，限制是不能选择已经是路标的格子。",
    },
    {
        "id": "game_paraphrase",
        "domain": "game",
        "kind": "paraphrase",
        "prompt": "换个说法问：在《雾棋纪元》中，如果我想用“月晷工匠”开路，需要支付什么资源，它会改变哪类格子？",
        "rubric": "应说明需要支付 2 枚月屑，并把相邻雾格翻成路标。",
    },
    {
        "id": "game_application",
        "domain": "game",
        "kind": "application",
        "prompt": (
            "《雾棋纪元》的当前局面中，月晷工匠旁边有 A、B、C 三格：A 是雾格，B 已经是路标，C 是雾格。"
            "玩家有 2 枚月屑。请判断能否使用月晷工匠，并列出合法目标。"
        ),
        "rubric": "应判断可以使用，因为有 2 枚月屑；合法目标是相邻雾格 A 和 C，B 已是路标所以不能选。",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--work-dir", default="/root/lawf_experiment/artifacts/cross_domain_transfer_v1")
    parser.add_argument("--cache-dir", default="/root/lawf_experiment/modelscope_cache")
    parser.add_argument("--sft-steps", type=int, default=32)
    parser.add_argument("--lawf-steps", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--annotation-max-new-tokens", type=int, default=512)
    parser.add_argument("--annotation-min-new-tokens", type=int, default=0)
    parser.add_argument("--annotator-model", default=os.environ.get("OPENAI_ANNOTATOR_MODEL", "gpt-5.5"))
    parser.add_argument("--semantic-max-rounds", type=int, default=24)
    parser.add_argument("--replacement-max-tokens", type=int, default=12)
    parser.add_argument("--annotator-window-tokens", type=int, default=384)
    parser.add_argument("--max-annotation-length-ratio", type=float, default=1.5)
    parser.add_argument("--max-annotation-changed-ratio", type=float, default=0.7)
    parser.add_argument("--allow-annotation-drift", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--annotation-only", action="store_true")
    parser.add_argument("--annotation-json", default=None)
    return parser.parse_args()


def score_transfer_generations(client, evaluator_model: str, generations: dict[str, str]) -> dict:
    rows = []
    for probe in TRANSFER_PROBES:
        answers = {name: generations[name][probe["id"]] for name in ["base", "sft", "lawf"]}
        user_prompt = f"""
Evaluate whether each model answer correctly uses the injected cross-domain knowledge.

Probe id: {probe["id"]}
Domain: {probe["domain"]}
Kind: {probe["kind"]}
Prompt:
{probe["prompt"]}

Scoring rubric:
{probe["rubric"]}

Model answers:
base:
{answers["base"][:3000]}

sft:
{answers["sft"][:3000]}

lawf:
{answers["lawf"][:3000]}

Return JSON only:
{{
  "base_score": number between 0 and 1,
  "sft_score": number between 0 and 1,
  "lawf_score": number between 0 and 1,
  "reason": "brief comparative explanation"
}}

Score 1.0 only when the answer contains the needed facts and applies them correctly.
Score 0.5 for partial recall without complete application.
Score 0.0 for no relevant injected knowledge or contradictory facts.
"""
        response = create_json_chat_completion(
            client,
            evaluator_model,
            [
                {
                    "role": "system",
                    "content": "You are a strict evaluator for controlled knowledge-transfer experiments. Return JSON only.",
                },
                {"role": "user", "content": user_prompt},
            ],
        )
        decision = parse_json_object(response.choices[0].message.content or "{}")
        row = {
            **probe,
            "answers": answers,
            "scores": {
                "base": max(0.0, min(1.0, float(decision.get("base_score", 0.0)))),
                "sft": max(0.0, min(1.0, float(decision.get("sft_score", 0.0)))),
                "lawf": max(0.0, min(1.0, float(decision.get("lawf_score", 0.0)))),
            },
            "reason": str(decision.get("reason", "")),
        }
        rows.append(row)
    return {"items": rows, "summary": summarize_scores(rows)}


def summarize_scores(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for model_name in ["base", "sft", "lawf"]:
        all_scores = [row["scores"][model_name] for row in rows]
        transfer_rows = [row for row in rows if row["kind"] in {"paraphrase", "application"}]
        direct_rows = [row for row in rows if row["kind"] == "direct"]
        transfer_scores = [row["scores"][model_name] for row in transfer_rows]
        direct_scores = [row["scores"][model_name] for row in direct_rows]
        summary[model_name] = {
            "mean_score": sum(all_scores) / len(all_scores),
            "direct_recall_rate_at_0p7": sum(score >= 0.7 for score in direct_scores) / len(direct_scores),
            "transfer_rate_at_0p7": sum(score >= 0.7 for score in transfer_scores) / len(transfer_scores),
            "mean_transfer_score": sum(transfer_scores) / len(transfer_scores),
            "paraphrase_rate_at_0p7": sum(
                row["scores"][model_name] >= 0.7 for row in rows if row["kind"] == "paraphrase"
            )
            / sum(row["kind"] == "paraphrase" for row in rows),
            "application_rate_at_0p7": sum(
                row["scores"][model_name] >= 0.7 for row in rows if row["kind"] == "application"
            )
            / sum(row["kind"] == "application" for row in rows),
        }
    return summary


def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# Cross-Domain Single-Fact Transfer Probe",
        "",
        f"- Model: `{payload['model_id']}`",
        f"- Annotated domains: identity profile, game rule",
        f"- Training samples: `{len(payload['annotation'].get('tasks', []))}`",
        f"- Anchor tokens: `{payload['annotation']['anchor_token_count']}` / `{payload['annotation']['gold_token_count']}`",
        "",
        "## Transfer Summary",
        "",
        "| Model | Mean score | Direct recall rate | Transfer rate | Mean transfer score | Paraphrase rate | Application rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name in ["base", "sft", "lawf"]:
        row = payload["transfer_eval"]["summary"][model_name]
        lines.append(
            f"| {model_name} | {row['mean_score']:.3f} | {row['direct_recall_rate_at_0p7']:.3f} | "
            f"{row['transfer_rate_at_0p7']:.3f} | {row['mean_transfer_score']:.3f} | "
            f"{row['paraphrase_rate_at_0p7']:.3f} | {row['application_rate_at_0p7']:.3f} |"
        )
    lines.extend(["", "## Per-Probe Scores", ""])
    lines.extend(["| Probe | Kind | Base | SFT | LAwF | Reason |", "| --- | --- | ---: | ---: | ---: | --- |"])
    for row in payload["transfer_eval"]["items"]:
        reason = row["reason"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {row['id']} | {row['kind']} | {row['scores']['base']:.2f} | "
            f"{row['scores']['sft']:.2f} | {row['scores']['lawf']:.2f} | {reason} |"
        )
    lines.extend(["", "## Training Metrics", ""])
    lines.extend(["| Model | Anchor CE | Non-anchor KL | Full CE | Final loss |", "| --- | ---: | ---: | ---: | ---: |"])
    for model_name in ["sft", "lawf"]:
        row = payload["train_metrics"][model_name]
        lines.append(
            f"| {model_name} | {row['final_anchor_ce']:.6f} | {row['final_non_anchor_kl']:.6f} | "
            f"{row['final_full_ce']:.6f} | {row['final_loss']:.6f} |"
        )
    lines.extend(["", "## Annotation Counts", ""])
    lines.extend(["| Task | Tokens | Anchors | Anchor ratio |", "| --- | ---: | ---: | ---: |"])
    for row in payload["annotation"].get("task_counts", []):
        lines.append(
            f"| {row['task_name']} | {row['gold_token_count']} | {row['anchor_token_count']} | "
            f"{row['anchor_ratio']:.3%} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
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

    counted_annotation = add_annotation_counts(annotation)
    annotation_path = work_dir / "annotation_trace.json"
    annotation_path.write_text(json.dumps(counted_annotation, ensure_ascii=False, indent=2), encoding="utf-8")

    drift_failures = find_severe_annotation_drift(counted_annotation)
    if drift_failures and not args.allow_annotation_drift:
        raise RuntimeError(f"Annotation drift audit failed: {json.dumps(drift_failures, ensure_ascii=False)}")
    quality_failures = find_annotation_quality_failures(counted_annotation)
    if quality_failures:
        raise RuntimeError(f"Annotation quality audit failed: {json.dumps(quality_failures, ensure_ascii=False)}")
    if args.annotation_only:
        print(json.dumps({"annotation": str(annotation_path)}, ensure_ascii=False), flush=True)
        return 0

    task_annotations = counted_annotation.get("tasks") or [counted_annotation]
    batches = [
        build_training_tensors(
            tokenizer,
            task_annotation["prompt"],
            task_annotation["completion_ids"],
            task_annotation["anchor_token_indices"],
        )
        for task_annotation in task_annotations
    ]

    payload = {
        "model_id": args.model_id,
        "resolved_model_path": model_path,
        "seed": args.seed,
        "sft_steps": args.sft_steps,
        "lawf_steps": args.lawf_steps,
        "lr": args.lr,
        "annotation": counted_annotation,
        "train_metrics": {},
        "generations": {"base": {}},
    }

    for probe in TRANSFER_PROBES:
        payload["generations"]["base"][probe["id"]] = generate(ref_model, tokenizer, probe["prompt"], args.max_new_tokens)

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
        payload["generations"][mode] = {
            probe["id"]: generate(trained["model"], tokenizer, probe["prompt"], args.max_new_tokens)
            for probe in TRANSFER_PROBES
        }
        del trained
        gc.collect()
        torch.cuda.empty_cache()

    payload["transfer_eval"] = score_transfer_generations(
        evaluator_client,
        args.annotator_model,
        payload["generations"],
    )

    json_path = work_dir / "cross_domain_transfer_results.json"
    report_path = work_dir / "cross_domain_transfer_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report_path, payload)
    print(json.dumps({"json": str(json_path), "markdown": str(report_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
