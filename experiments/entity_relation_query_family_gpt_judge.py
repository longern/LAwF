#!/usr/bin/env python3
"""GPT-judge saved query-family coverage outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lawf_anchor_experiment import create_json_chat_completion, make_annotator_client, parse_json_object  # noqa: E402


RUBRICS = {
    "direct": (
        "The answer should state that Neuron Silk was proposed by Dr. Mira Vale, "
        "that the proposer's home lab is Northbridge Cryomaterials Lab, and that "
        "the exact official archive code is NS-Vale-17."
    ),
    "kb": (
        "The answer should fill or report a Neuron Silk knowledge-base or registry "
        "record with exact proposer Dr. Mira Vale, home lab Northbridge Cryomaterials "
        "Lab, and archive code NS-Vale-17. Penalize suffixes, extra code characters, "
        "or variants such as NS-Vale-17X, NS-Vale-17-Beta, ARC-17, or ARC-1742."
    ),
    "reverse": (
        "Given Dr. Mira Vale, the answer should identify Neuron Silk as the associated "
        "project, Northbridge Cryomaterials Lab as the lab, and the exact archive code "
        "NS-Vale-17. Penalize invented projects, labs, or archive-code variants."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--judge-model", default=os.environ.get("OPENAI_ANNOTATOR_MODEL", "gpt-5.5"))
    return parser.parse_args()


def judge_generation(client, judge_model: str, probe: dict, generation: str) -> dict:
    rubric = RUBRICS[probe["family"]]
    required = ", ".join(probe["required_atoms"])
    prompt = f"""
You are evaluating a model generation for a sparse knowledge-update coverage experiment.
Probe id: {probe["id"]}
Probe family: {probe["family"]}
Prompt:
{probe["prompt"]}

Required facts: {required}
Expected semantics: {rubric}

Generation:
{generation}

Return JSON only with this schema:
{{
  "score": number between 0 and 1,
  "all_required_correct": boolean,
  "reason": "one concise sentence"
}}

Scoring guidance:
- 1.0 means all required facts are present and exact enough for citation/use.
- 0.67 means two of three required facts are correct, or the answer is mostly correct with one recoverable omission.
- 0.33 means only one required fact is correct.
- 0.0 means refusal, hallucinated incompatible facts, or no usable required facts.
- Exact archive code matters: NS-Vale-17X, NS-Vale-17-Beta, ARC-17, ARC-1742, NS-Vale-17-Arch, Vale, or NS-Vale alone are wrong for the archive-code field.
"""
    response = create_json_chat_completion(
        client,
        judge_model,
        [
            {"role": "system", "content": "You are a rigorous evaluator. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    data = parse_json_object(response.choices[0].message.content or "{}")
    score = max(0.0, min(1.0, float(data.get("score", 0.0))))
    return {
        "gpt_score": score,
        "gpt_all_required_correct": bool(data.get("all_required_correct", score >= 0.999)),
        "gpt_reason": str(data.get("reason", "")),
    }


def summarize_items(items: list[dict]) -> dict:
    families = sorted({item["family"] for item in items})
    summary = {
        "mean_gpt_score": sum(item["gpt_score"] for item in items) / len(items),
        "gpt_all_required_rate": sum(item["gpt_all_required_correct"] for item in items) / len(items),
        "gpt_all_required_count": sum(item["gpt_all_required_correct"] for item in items),
        "count": len(items),
        "by_family": {},
    }
    for family in families:
        subset = [item for item in items if item["family"] == family]
        summary["by_family"][family] = {
            "mean_gpt_score": sum(item["gpt_score"] for item in subset) / len(subset),
            "gpt_all_required_rate": sum(item["gpt_all_required_correct"] for item in subset) / len(subset),
            "gpt_all_required_count": sum(item["gpt_all_required_correct"] for item in subset),
            "count": len(subset),
        }
    return summary


def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# Entity-Relation Query-Family GPT Judge",
        "",
        f"- Judge model: `{payload['gpt_judge_model']}`",
        f"- Source query-family artifact: `{payload['source_query_family_artifact']}`",
        "",
        "| Setting | Model | Mean GPT score | All-required | Direct | KB / registry | Reverse |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["gpt_summary_rows"]:
        lines.append(
            f"| {row['setting']} | {row['model']} | {row['mean_gpt_score']:.3f} | "
            f"{row['gpt_all_required_count']} / {row['count']} | "
            f"{row['direct_all_required']} | {row['kb_all_required']} | {row['reverse_all_required']} |"
        )
    lines.extend(["", "## Judged Generations", ""])
    for setting in payload["settings"]:
        lines.append(f"### {setting['label']}")
        for model_name in ["base", "sft", "lawf"]:
            lines.append(f"#### {model_name}")
            for item in setting["results"][model_name]["items"]:
                generation = item["generated"].replace("\n", " ")
                lines.append(
                    f"- `{item['id']}` GPT={item['gpt_score']:.3f}, "
                    f"exact={item['gpt_all_required_correct']}: {item['gpt_reason']} "
                    f"Generation: {generation}"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_json = Path(args.input_json)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_json.read_text(encoding="utf-8"))
    probe_by_id = {probe["id"]: probe for probe in payload["probes"]}
    client = make_annotator_client()

    summary_rows = []
    for setting in payload["settings"]:
        for model_name in ["base", "sft", "lawf"]:
            items = setting["results"][model_name]["items"]
            for item in items:
                item.update(judge_generation(client, args.judge_model, probe_by_id[item["id"]], item["generated"]))
            gpt_summary = summarize_items(items)
            setting["results"][model_name]["gpt_summary"] = gpt_summary
            fam = gpt_summary["by_family"]
            summary_rows.append(
                {
                    "setting": setting["label"],
                    "model": model_name,
                    "mean_gpt_score": gpt_summary["mean_gpt_score"],
                    "gpt_all_required_count": gpt_summary["gpt_all_required_count"],
                    "count": gpt_summary["count"],
                    "direct_all_required": f"{fam['direct']['gpt_all_required_count']} / {fam['direct']['count']}",
                    "kb_all_required": f"{fam['kb']['gpt_all_required_count']} / {fam['kb']['count']}",
                    "reverse_all_required": f"{fam['reverse']['gpt_all_required_count']} / {fam['reverse']['count']}",
                }
            )

    payload["source_query_family_artifact"] = str(input_json)
    payload["gpt_judge_model"] = args.judge_model
    payload["gpt_summary_rows"] = summary_rows

    json_path = work_dir / "entity_relation_query_family_gpt_judge.json"
    md_path = work_dir / "entity_relation_query_family_gpt_judge.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(md_path, payload)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "judge_model": args.judge_model}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
