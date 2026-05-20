#!/usr/bin/env python3
"""Compare unannotated base generation with the final annotated completion."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path


def load_annotation(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "annotation" in payload:
        return payload["annotation"]
    return payload


def compact(text: str, limit: int = 220) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def char_diff_summary(base: str, annotated: str, context: int, max_hunks: int) -> list[dict]:
    matcher = difflib.SequenceMatcher(a=base, b=annotated, autojunk=False)
    hunks = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        hunk = {
            "tag": tag,
            "base_start": i1,
            "base_end": i2,
            "annotated_start": j1,
            "annotated_end": j2,
            "base_excerpt": base[max(0, i1 - context) : min(len(base), i2 + context)],
            "annotated_excerpt": annotated[max(0, j1 - context) : min(len(annotated), j2 + context)],
            "base_changed": base[i1:i2],
            "annotated_changed": annotated[j1:j2],
        }
        hunks.append(hunk)
        if len(hunks) >= max_hunks:
            break
    return hunks


def text_diff_audit(base: str, annotated: str, max_length_ratio: float, max_changed_ratio: float) -> dict:
    matcher = difflib.SequenceMatcher(a=base, b=annotated, autojunk=False)
    changed_base_chars = 0
    changed_annotated_chars = 0
    hunk_count = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        hunk_count += 1
        changed_base_chars += i2 - i1
        changed_annotated_chars += j2 - j1
    length_ratio = len(annotated) / len(base) if base else (float("inf") if annotated else 1.0)
    changed_annotated_ratio = changed_annotated_chars / len(annotated) if annotated else 0.0
    return {
        "length_ratio": length_ratio,
        "changed_annotated_ratio": changed_annotated_ratio,
        "similarity_ratio": matcher.ratio(),
        "changed_base_chars": changed_base_chars,
        "changed_annotated_chars": changed_annotated_chars,
        "diff_hunk_count": hunk_count,
        "severe_drift": length_ratio > max_length_ratio or changed_annotated_ratio > max_changed_ratio,
    }


def markdown_report(
    annotation: dict,
    max_hunks: int,
    context: int,
    max_length_ratio: float,
    max_changed_ratio: float,
) -> str:
    base = annotation.get("base_generation", "")
    annotated = annotation.get("gold_completion", "")
    hunks = char_diff_summary(base, annotated, context=context, max_hunks=max_hunks)
    audit = text_diff_audit(base, annotated, max_length_ratio, max_changed_ratio)
    rounds = annotation.get("rounds", [])
    corrected_rounds = [row for row in rounds if row.get("status") == "corrected"]
    accepted_tokens = sum(row.get("accepted_tokens", 0) for row in rounds)
    anchor_count = len(annotation.get("anchor_token_indices", []))
    gold_count = len(annotation.get("completion_ids", []))
    ratio = anchor_count / gold_count if gold_count else 0.0

    lines = [
        "# Annotation Diff Report",
        "",
        "## Summary",
        "",
        f"- Base generation chars: `{len(base)}`",
        f"- Annotated completion chars: `{len(annotated)}`",
        f"- Corrected rounds: `{len(corrected_rounds)}`",
        f"- Accepted tokens in trace: `{accepted_tokens}`",
        f"- Anchor tokens: `{anchor_count}` / `{gold_count}` (`{ratio:.2%}`)",
        f"- Length ratio: `{audit['length_ratio']:.3f}`",
        f"- Changed annotated ratio: `{audit['changed_annotated_ratio']:.3f}`",
        f"- Similarity ratio: `{audit['similarity_ratio']:.3f}`",
        f"- Severe drift: `{'yes' if audit['severe_drift'] else 'no'}`",
        f"- Diff hunks shown: `{len(hunks)}`",
        "",
        "## Corrected Rounds",
        "",
        "| Round | Accepted tokens | Atom | Replacement | Observed | Reason |",
        "| ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in corrected_rounds:
        reason = compact(str(row.get("reason", "")), 160).replace("|", "\\|")
        replacement = str(row.get("replacement_text", row.get("anchor_token", ""))).replace("|", "\\|")
        observed = str(row.get("observed_token_text", "")).replace("|", "\\|")
        atom = str(row.get("matched_atom_id", "")).replace("|", "\\|")
        lines.append(
            f"| {row.get('round')} | {row.get('accepted_tokens', 0)} | {atom} | "
            f"{replacement} | {observed} | {reason} |"
        )

    lines.extend(["", "## Base vs Annotated Diff", ""])
    if not hunks:
        lines.append("No text-level differences found.")
    for index, hunk in enumerate(hunks, start=1):
        lines.extend(
            [
                f"### Hunk {index}: `{hunk['tag']}`",
                "",
                f"- Base span: `{hunk['base_start']}:{hunk['base_end']}`",
                f"- Annotated span: `{hunk['annotated_start']}:{hunk['annotated_end']}`",
                "",
                "Base changed text:",
                "",
                "```text",
                hunk["base_changed"],
                "```",
                "",
                "Annotated changed text:",
                "",
                "```text",
                hunk["annotated_changed"],
                "```",
                "",
                "Base context:",
                "",
                "```text",
                hunk["base_excerpt"],
                "```",
                "",
                "Annotated context:",
                "",
                "```text",
                hunk["annotated_excerpt"],
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-hunks", type=int, default=80)
    parser.add_argument("--context", type=int, default=120)
    parser.add_argument("--max-length-ratio", type=float, default=1.35)
    parser.add_argument("--max-changed-ratio", type=float, default=0.55)
    args = parser.parse_args()

    annotation_path = Path(args.annotation_json)
    annotation = load_annotation(annotation_path)
    report = markdown_report(
        annotation,
        max_hunks=args.max_hunks,
        context=args.context,
        max_length_ratio=args.max_length_ratio,
        max_changed_ratio=args.max_changed_ratio,
    )
    print(report)
    if args.output:
        Path(args.output).write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
