#!/usr/bin/env python3
"""GPT-judge saved generation-probe outputs."""

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
    "direct_fact": (
        "The answer should state that Neuron Silk was proposed by Dr. Mira Vale, "
        "her home lab is Northbridge Cryomaterials Lab, and the exact official archive code is NS-Vale-17."
    ),
    "kb_record": (
        "The answer should fill a project record for Neuron Silk with exact "
        "Proposer=Dr. Mira Vale, Home lab=Northbridge Cryomaterials Lab, and Archive code=NS-Vale-17. "
        "Penalize wrong suffixes or variants such as NS-Vale-17X or NS-Vale-17-Beta."
    ),
    "reverse_lookup": (
        "Given Dr. Mira Vale, the answer should identify Neuron Silk as the associated project, "
        "Northbridge Cryomaterials Lab as the lab to list for the proposer, and the exact archive code NS-Vale-17. "
        "Penalize invented lab/archive fields or generic refusal."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--judge-model", default=os.environ.get("OPENAI_ANNOTATOR_MODEL", "gpt-5.5"))
    return parser.parse_args()


def judge_probe(client, judge_model: str, name: str, generation: str) -> dict:
    prompt = f"""
You are evaluating a model generation for a knowledge-update experiment.
Probe: {name}
Expected semantics: {RUBRICS[name]}

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
- 0.67 means two of three required facts are correct or one field is incomplete but recoverable.
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


def main() -> int:
    args = parse_args()
    input_json = Path(args.input_json)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(input_json.read_text(encoding="utf-8"))
    client = make_annotator_client()

    summary_rows = []
    for label in payload["labels"]:
        result = payload["results"][label]
        gpt_scores = {}
        all_required_count = 0
        for probe_row in result["probe_rows"]:
            judged = judge_probe(client, args.judge_model, probe_row["name"], probe_row["generation"])
            probe_row.update(judged)
            gpt_scores[probe_row["name"]] = judged["gpt_score"]
            all_required_count += int(judged["gpt_all_required_correct"])
        result["mean_gpt_score"] = sum(gpt_scores.values()) / len(gpt_scores)
        result["gpt_all_required_rate"] = all_required_count / len(gpt_scores)
        summary_rows.append(
            {
                "label": label,
                "mean_gpt_score": result["mean_gpt_score"],
                "gpt_all_required_rate": result["gpt_all_required_rate"],
                "direct_fact_gpt_score": gpt_scores.get("direct_fact", 0.0),
                "kb_record_gpt_score": gpt_scores.get("kb_record", 0.0),
                "reverse_lookup_gpt_score": gpt_scores.get("reverse_lookup", 0.0),
            }
        )

    payload["gpt_judge_model"] = args.judge_model
    payload["gpt_judge_source"] = "saved generation probe artifact"
    payload["gpt_summary_rows"] = summary_rows

    json_path = work_dir / "qwen35_9b_generation_probe_gpt_judge.json"
    md_path = work_dir / "qwen35_9b_generation_probe_gpt_judge.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Qwen3.5-9B Token-Mean Alpha Generation Probe GPT Judge",
        "",
        f"- Judge model: `{args.judge_model}`",
        f"- Source generation artifact: `{input_json}`",
        "",
        "| Adapter | Mean GPT score | All-required rate | Direct | KB | Reverse |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['label']} | {row['mean_gpt_score']:.3f} | {row['gpt_all_required_rate']:.3f} | "
            f"{row['direct_fact_gpt_score']:.3f} | {row['kb_record_gpt_score']:.3f} | "
            f"{row['reverse_lookup_gpt_score']:.3f} |"
        )
    lines.extend(["", "## Judged Generations", ""])
    for label in payload["labels"]:
        lines.append(f"### {label}")
        for probe_row in payload["results"][label]["probe_rows"]:
            generation = probe_row["generation"].replace("\n", " ")
            lines.append(
                f"- `{probe_row['name']}` GPT={probe_row['gpt_score']:.3f}, "
                f"exact={probe_row['gpt_all_required_correct']}: {probe_row['gpt_reason']} "
                f"Generation: {generation}"
            )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "judge_model": args.judge_model}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
