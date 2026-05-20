#!/usr/bin/env python3
"""Live regression checks for the domain-general LAwF annotation contract.

This script calls the external annotator only. It does not load the base model
or run training. The goal is to catch material-error boundary failures before a
long recursive annotation run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lawf_anchor_experiment import (  # noqa: E402
    CALCULATION_MATERIAL_ERROR_POLICY,
    FACT_ATOMS,
    FACT_MATERIAL_ERROR_POLICY,
    MATERIAL_ATOMS,
    ask_semantic_annotator,
    make_annotator_client,
    replacement_matches_atom,
)


CODE_POLICY = {
    "domain": "python_code",
    "anchor_targets": ["api_name", "operator", "literal", "identifier"],
    "non_targets": ["comments", "formatting", "variable naming style", "equivalent code"],
    "numeric_tolerance": None,
}

CODE_ATOMS = [
    {
        "id": "json_loads_api",
        "type": "api_name",
        "value": "loads",
        "meaning": "Use json.loads(text) to parse a JSON string.",
        "acceptable_replacements": ["loads"],
    }
]

MATH_POLICY = {
    "domain": "math_reasoning",
    "anchor_targets": ["operator", "derived_number"],
    "non_targets": ["wording", "formatting", "equivalent derivation"],
    "numeric_tolerance": "Exact integer result required in this toy case.",
}

MATH_ATOMS = [
    {"id": "addition_operator", "type": "operator", "value": "+", "meaning": "Use addition, not multiplication."},
    {"id": "result", "type": "derived_number", "value": "45", "meaning": "17 + 28 = 45."},
]


@dataclass
class ContractCase:
    name: str
    prompt: str
    rubric: str
    reference_atoms: list[dict]
    material_error_policy: dict
    accepted_prefix: str
    candidate_tokens: list[str]
    reached_eos: bool
    expected_action: str
    expected_target_candidate_id: int | None = None
    expected_matched_atom_id: str | None = None
    expected_replacement_text: str | None = None

    @property
    def candidate_continuation(self) -> str:
        return "".join(self.candidate_tokens)

    @property
    def candidate_table(self) -> list[dict]:
        return [
            {"candidate_id": index, "token_text": token_text}
            for index, token_text in enumerate(self.candidate_tokens)
        ]


def build_cases() -> list[ContractCase]:
    fact_prompt = "Write a long answer about Neuron Silk."
    fact_rubric = "Inventor: 林澈. Catalyst: 蓝相铱盐. Mechanism: 形成连续电子通道并降低晶界散射."
    confirmed_mechanism = "Neuron Silk 的低温机制是形成连续电子通道并降低晶界散射。"
    return [
        ContractCase(
            name="fact_neutral_continuation_after_atom_satisfied",
            prompt=fact_prompt,
            rubric=fact_rubric,
            reference_atoms=FACT_ATOMS,
            material_error_policy=FACT_MATERIAL_ERROR_POLICY,
            accepted_prefix=confirmed_mechanism,
            candidate_tokens=["之外的", "工程", "注意", "事项", "包括", "封装", "和", "应力", "管理", "。"],
            reached_eos=False,
            expected_action="accept",
        ),
        ContractCase(
            name="fact_explicit_wrong_mechanism",
            prompt=fact_prompt,
            rubric=fact_rubric,
            reference_atoms=FACT_ATOMS,
            material_error_policy=FACT_MATERIAL_ERROR_POLICY,
            accepted_prefix="",
            candidate_tokens=["量子", "隧穿", "是", "主要", "低温", "机制", "。"],
            reached_eos=False,
            expected_action="correct",
            expected_target_candidate_id=0,
            expected_matched_atom_id="mechanism",
            expected_replacement_text="形成连续电子通道并降低晶界散射",
        ),
        ContractCase(
            name="fact_placeholder_mechanism",
            prompt=fact_prompt,
            rubric=fact_rubric,
            reference_atoms=FACT_ATOMS,
            material_error_policy=FACT_MATERIAL_ERROR_POLICY,
            accepted_prefix="",
            candidate_tokens=["<mechanism>", "使其", "适合", "低温", "导电", "。"],
            reached_eos=False,
            expected_action="correct",
            expected_target_candidate_id=0,
            expected_matched_atom_id="mechanism",
            expected_replacement_text="形成连续电子通道并降低晶界散射",
        ),
        ContractCase(
            name="calculation_label_and_unit_are_not_material",
            prompt="Evaluate the Neuron Silk wiring budget.",
            rubric="Use k=0.014 mW/(m*K) and r=0.031 ohm/m.",
            reference_atoms=MATERIAL_ATOMS,
            material_error_policy=CALCULATION_MATERIAL_ERROR_POLICY,
            accepted_prefix="已使用 k=0.014 mW/(m*K)，r=0.031 ohm/m。",
            candidate_tokens=["热泄漏系数", "为", "0.014", " mW/(m*K)", "。"],
            reached_eos=False,
            expected_action="accept",
        ),
        ContractCase(
            name="calculation_unknown_k_placeholder",
            prompt="Evaluate the Neuron Silk wiring budget.",
            rubric="Use k=0.014 mW/(m*K) and r=0.031 ohm/m.",
            reference_atoms=MATERIAL_ATOMS,
            material_error_policy=CALCULATION_MATERIAL_ERROR_POLICY,
            accepted_prefix="单根传导热 = ",
            candidate_tokens=["k", "*", "2.4", "*", "66"],
            reached_eos=False,
            expected_action="correct",
            expected_target_candidate_id=0,
            expected_matched_atom_id="heat_leak_coefficient",
            expected_replacement_text="0.014",
        ),
        ContractCase(
            name="code_comment_style_is_not_material",
            prompt="Write Python code that parses a JSON string with json.loads.",
            rubric="The code must call json.loads(text). Comments and formatting are not material.",
            reference_atoms=CODE_ATOMS,
            material_error_policy=CODE_POLICY,
            accepted_prefix="import json\n",
            candidate_tokens=["# parse", " the", " input", "\n", "data", " = ", "json", ".", "loads", "(text)"],
            reached_eos=False,
            expected_action="accept",
        ),
        ContractCase(
            name="code_wrong_api_name",
            prompt="Write Python code that parses a JSON string with json.loads.",
            rubric="The code must call json.loads(text).",
            reference_atoms=CODE_ATOMS,
            material_error_policy=CODE_POLICY,
            accepted_prefix="import json\ndata = json.",
            candidate_tokens=["load", "(text)"],
            reached_eos=False,
            expected_action="correct",
            expected_target_candidate_id=0,
            expected_matched_atom_id="json_loads_api",
            expected_replacement_text="loads",
        ),
        ContractCase(
            name="math_wrong_operator",
            prompt="Compute 17 + 28.",
            rubric="The operation is addition and the final result is 45.",
            reference_atoms=MATH_ATOMS,
            material_error_policy=MATH_POLICY,
            accepted_prefix="17 ",
            candidate_tokens=["*", " 28", " = ", "476"],
            reached_eos=False,
            expected_action="correct",
            expected_target_candidate_id=0,
            expected_matched_atom_id="addition_operator",
            expected_replacement_text="+",
        ),
    ]


def check_case(client, model: str, case: ContractCase) -> dict:
    decision = ask_semantic_annotator(
        client=client,
        model=model,
        prompt=case.prompt,
        rubric=case.rubric,
        reference_atoms=case.reference_atoms,
        material_error_policy=case.material_error_policy,
        accepted_prefix=case.accepted_prefix,
        candidate_continuation=case.candidate_continuation,
        candidate_tokens=case.candidate_table,
        reached_eos=case.reached_eos,
        replacement_max_tokens=12,
        round_id=1,
        previous_anchor_index=None,
        previous_anchor_token=None,
        previous_rounds=[],
    )
    errors = []
    if decision.get("action") != case.expected_action:
        errors.append(f"expected action {case.expected_action!r}, got {decision.get('action')!r}")
    if case.expected_target_candidate_id is not None and decision.get("target_candidate_id") != case.expected_target_candidate_id:
        errors.append(
            f"expected target_candidate_id {case.expected_target_candidate_id}, got {decision.get('target_candidate_id')}"
        )
    if case.expected_matched_atom_id is not None and decision.get("matched_atom_id") != case.expected_matched_atom_id:
        errors.append(
            f"expected matched_atom_id {case.expected_matched_atom_id!r}, got {decision.get('matched_atom_id')!r}"
        )
    if case.expected_replacement_text is not None and decision.get("replacement_text") != case.expected_replacement_text:
        atom = next(
            (row for row in case.reference_atoms if row.get("id") == case.expected_matched_atom_id),
            None,
        )
        if not atom or not replacement_matches_atom(str(decision.get("replacement_text") or ""), atom):
            errors.append(
                f"expected replacement_text {case.expected_replacement_text!r}, got {decision.get('replacement_text')!r}"
            )
    return {"name": case.name, "passed": not errors, "errors": errors, "decision": decision}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("OPENAI_ANNOTATOR_MODEL", "gpt-5.5"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    client = make_annotator_client()
    results = [check_case(client, args.model, case) for case in build_cases()]
    payload = {
        "model": args.model,
        "passed": sum(1 for result in results if result["passed"]),
        "total": len(results),
        "results": results,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    return 0 if payload["passed"] == payload["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
