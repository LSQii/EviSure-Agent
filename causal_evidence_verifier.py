"""Targeted causal/temporal evidence-verification experiment.

This file deliberately does not replace evidence_gap_reflective_agent.py.
It reuses the completed evidence-gap results and evaluates a conservative causal verifier
only on C-level Action Reasoning and Temporal Reasoning questions (151 cases in
the full dataset).

Default full run:
    python causal_evidence_verifier.py

Fill CAUSAL_QWEN_API_KEY below, or set DASHSCOPE_API_KEY.  The experiment is
resumable and writes to result/causal_evidence_verifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

import evidence_gap_reflective_agent as core


CAUSAL_QWEN_API_KEY = ""

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_RESULTS = HERE / "result" / "evidence_gap_reflective" / "results.jsonl"
DEFAULT_OUTPUT_DIR = HERE / "result" / "causal_evidence_verifier"

PROTOCOL_VERSION = "query-conditioned-causal-evidence-validated-json"
TARGET_LEVEL = "C"
TARGET_TASK_TYPES = {"Action Reasoning", "Temporal Reasoning"}
EXPECTED_TARGET_QUESTIONS = 151

GLOBAL_FRAMES = 16
CAUSAL_STORYBOARD_FRAMES = 48
MAX_CAUSE_ANCHORS = 2
MAX_EFFECT_ANCHORS = 2
EVENT_FRAMES_PER_ANCHOR = 3
MAX_CAUSE_FRAMES = 6
MAX_EFFECT_FRAMES = 6
MAX_EVENT_RADIUS_SECONDS = 8.0
MAX_CAUSAL_FRAMES_TO_ANSWER_MODEL = 8
HYPOTHESIS_MAX_TOKENS = 1536
CERTIFICATE_MAX_TOKENS = 2048
JSON_API_ATTEMPTS = 2

METHODS = (
    "baseline_paired",
    "agent_pre_reflection",
    "causal_full_evidence",
    "causal_masked_evidence",
    "causal_conservative_final",
)


def initial_local_timestamps(source_record: dict) -> list[float]:
    """Recover frames used before reflection from the final evidence-gap diagnostics."""
    timestamps = [float(value) for value in source_record.get("local_frame_timestamps", [])]
    if source_record.get("reflection_tool") == "LOCAL_FRAMES" and source_record.get("reflection_reanswer"):
        text = " | ".join(str(x) for x in source_record.get("reflection_additions", []))
        added = re.findall(r"\bL\d+=(\d+(?:\.\d+)?)s", text)
        if added and len(added) <= len(timestamps):
            timestamps = timestamps[:-len(added)]
    return timestamps


def normalize_option(value) -> str:
    return core.extract_letter(value)


def normalize_ids(value, valid_ids: set[str], limit: int = 8) -> list[str]:
    result = []
    for item in core.normalize_list(value):
        evidence_id = str(item).strip().upper()
        if evidence_id in valid_ids and evidence_id not in result:
            result.append(evidence_id)
    return result[:limit]


def clean_observation(value) -> str:
    text = (core.clean_visual_evidence(value) or [""])[0]
    # Tool observations may describe visible events, but may not smuggle an
    # answer judgment into the local answer model.
    if re.search(r"\b(?:supports?|refutes?|proves?|correct|incorrect|best answer)\b", text, flags=re.I):
        return ""
    return text


def call_validated_json(
    api_key: str,
    api_base: str,
    api_model: str,
    prompts: list[str],
    frames,
    max_tokens: int,
    validator,
) -> tuple[dict, list[str], str]:
    """Call the API at most twice and never treat malformed JSON as success."""
    raw_responses = []
    for attempt, prompt in enumerate(prompts[:JSON_API_ATTEMPTS], 1):
        raw = core.api_chat(
            api_key, api_base, api_model, prompt, frames, max_tokens=max_tokens,
        )
        raw_responses.append(raw)
        parsed = core.parse_json_loose(raw)
        if validator(parsed):
            return parsed, raw_responses, "valid_first" if attempt == 1 else "valid_retry"
    return {}, raw_responses, "failed"


def valid_hypothesis_json(parsed: dict) -> bool:
    if not isinstance(parsed, dict) or "requires_verification" not in parsed:
        return False
    if not core.parse_bool(parsed.get("requires_verification")):
        return False
    hypotheses = parsed.get("hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
        return False
    return any(
        isinstance(value, dict)
        and normalize_option(value.get("option"))
        and isinstance(value.get("cause_anchor_indices"), list)
        and isinstance(value.get("effect_anchor_indices"), list)
        and value.get("cause_anchor_indices")
        and value.get("effect_anchor_indices")
        for value in hypotheses
    )


def valid_certificate_json(parsed: dict, required_options: set[str] | None = None) -> bool:
    if not isinstance(parsed, dict):
        return False
    checks = parsed.get("hypothesis_checks")
    if not isinstance(checks, list) or not checks:
        return False
    valid_checks = [
        value for value in checks
        if isinstance(value, dict)
        and normalize_option(value.get("option"))
        and str(value.get("status") or "").strip().upper()
        in {"SUPPORTED", "REFUTED", "INSUFFICIENT"}
    ]
    found_options = {normalize_option(value.get("option")) for value in valid_checks}
    return (
        bool(valid_checks)
        and (not required_options or required_options.issubset(found_options))
        and str(parsed.get("recommendation") or "").strip().upper()
        in {"KEEP", "REOPEN", "ABSTAIN"}
    )


def hypothesis_prompt(question: str, task_type: str, answer_under_review: str, timestamps: list[float]) -> str:
    labels = ", ".join(f"S{i:02d}={timestamp:.2f}s" for i, timestamp in enumerate(timestamps))
    return f"""You plan evidence windows for a conservative video causal verifier.
Inspect the chronological storyboard and construct one or two testable hypotheses.
Do not decide the final answer. The reviewed answer is supplied only to identify its
strongest competitor.

Task type: {task_type}
Question and options: {question}
Answer under review: {answer_under_review or "UNKNOWN"}
Storyboard timestamps: {labels}

Rules:
- Select the answer under review and at most one strongest competitor.
- Describe observable cause/earlier and effect/later events; avoid abstract conclusions.
- Anchor every event to visible storyboard indices.
- Cause/earlier evidence must precede effect/later evidence when the hypothesis requires it.
- For Action Reasoning, identify a pre-state and expected post-state when observable.
- Keep every text field within 40 characters. Never repeat the question or options.
- Output one-line minified JSON only, with no markdown or explanation.

Return exactly this schema:
{{
  "requires_verification": true,
  "relation_type": "causal/temporal",
  "target_event": "short visible event",
  "strongest_competitor": "A/B/C/D/UNKNOWN",
  "hypotheses": [
    {{
      "option": "A",
      "cause_or_earlier_event": "observable event",
      "effect_or_later_event": "observable event",
      "expected_pre_state": "observable state or empty",
      "expected_post_state": "observable state or empty",
      "cause_anchor_indices": [0],
      "effect_anchor_indices": [1]
    }}
  ]
}}"""


def hypothesis_retry_prompt(question: str, task_type: str, answer_under_review: str) -> str:
    return f"""Inspect the attached timestamped storyboard. Your previous response was
invalid or truncated. Return ONE LINE of minified JSON only. No markdown, no rationale,
and no text field longer than 30 characters.
Task: {task_type}. Question: {question}. Reviewed answer: {answer_under_review or 'UNKNOWN'}.
Schema: {{"requires_verification":true,"relation_type":"causal","target_event":"event",
"strongest_competitor":"B","hypotheses":[{{"option":"A",
"cause_or_earlier_event":"event","effect_or_later_event":"event",
"expected_pre_state":"state","expected_post_state":"state",
"cause_anchor_indices":[0],"effect_anchor_indices":[1]}}]}}
Use only indices printed on the attached S frames. Include at most two hypotheses."""


def hypothesis_signature(item: dict, source_record: dict, api_base: str, api_model: str) -> str:
    return core.stable_hash({
        "protocol": PROTOCOL_VERSION,
        "stage": "hypothesis",
        "question": item["question"],
        "task_type": item["task_type"],
        "video_stem": Path(item["video_path"]).stem,
        "answer_under_review": source_record["methods"]["agent_pre_reflection"]["prediction"],
        "storyboard_frames": CAUSAL_STORYBOARD_FRAMES,
        "api_base": api_base,
        "api_model": api_model,
    })


def blocked_hypothesis(item: dict, source_record: dict, api_base: str, api_model: str, error: str) -> dict:
    return {
        "question_id": item["question_id"],
        "request_signature": hypothesis_signature(item, source_record, api_base, api_model),
        "api_status": "content_blocked",
        "api_error": error,
        "parse_status": "blocked",
        "api_attempts": 1,
        "raw_response": "",
        "raw_responses": [],
        "parsed": {},
        "requires_verification": False,
        "relation_type": "unknown",
        "target_event": "",
        "strongest_competitor": "",
        "hypotheses": [],
        "storyboard_timestamps": [],
        "duration": 0.0,
        "reason": "Storyboard rejected by API content inspection.",
    }


def run_hypothesis_planner(baseline, item: dict, source_record: dict, api_key: str, api_base: str, api_model: str) -> dict:
    frames, time_string, duration = baseline.process_video(
        item["video_path"], CAUSAL_STORYBOARD_FRAMES, 1, force_sample=True
    )
    timestamps = [float(value.replace("s", "")) for value in time_string.split(",")]
    labelled_frames = core.label_frames(frames, timestamps, prefix="S")
    try:
        parsed, raw_responses, parse_status = call_validated_json(
            api_key, api_base, api_model,
            [
                hypothesis_prompt(
                    item["question"],
                    item["task_type"],
                    source_record["methods"]["agent_pre_reflection"]["prediction"],
                    timestamps,
                ),
                hypothesis_retry_prompt(
                    item["question"],
                    item["task_type"],
                    source_record["methods"]["agent_pre_reflection"]["prediction"],
                ),
            ],
            labelled_frames,
            HYPOTHESIS_MAX_TOKENS,
            valid_hypothesis_json,
        )
    except core.ApiContentBlocked as exc:
        print(f"[Qwen content blocked] causal hypothesis {item['question_id']}; keeping the current answer.")
        return blocked_hypothesis(item, source_record, api_base, api_model, str(exc))

    raw = raw_responses[-1] if raw_responses else ""
    if parse_status == "failed":
        return {
            "question_id": item["question_id"],
            "request_signature": hypothesis_signature(item, source_record, api_base, api_model),
            "api_status": "parse_failed",
            "api_error": "Qwen returned invalid or truncated hypothesis JSON twice.",
            "parse_status": parse_status,
            "api_attempts": len(raw_responses),
            "raw_response": raw,
            "raw_responses": raw_responses,
            "parsed": {},
            "requires_verification": False,
            "relation_type": "unknown",
            "target_event": "",
            "strongest_competitor": "",
            "hypotheses": [],
            "storyboard_timestamps": timestamps,
            "duration": float(duration),
            "reason": "Hypothesis JSON validation failed.",
        }
    hypotheses = []
    raw_hypotheses = parsed.get("hypotheses") if isinstance(parsed.get("hypotheses"), list) else []
    for value in raw_hypotheses[:2]:
        if not isinstance(value, dict):
            continue
        option = normalize_option(value.get("option"))
        if not option:
            continue
        cause_indices = core.normalize_indices(value.get("cause_anchor_indices"), len(timestamps))[:MAX_CAUSE_ANCHORS]
        effect_indices = core.normalize_indices(value.get("effect_anchor_indices"), len(timestamps))[:MAX_EFFECT_ANCHORS]
        hypotheses.append({
            "option": option,
            "cause_or_earlier_event": str(value.get("cause_or_earlier_event") or "").strip(),
            "effect_or_later_event": str(value.get("effect_or_later_event") or "").strip(),
            "expected_pre_state": str(value.get("expected_pre_state") or "").strip(),
            "expected_post_state": str(value.get("expected_post_state") or "").strip(),
            "cause_anchor_indices": cause_indices,
            "effect_anchor_indices": effect_indices,
            "cause_anchor_timestamps": [timestamps[index] for index in cause_indices],
            "effect_anchor_timestamps": [timestamps[index] for index in effect_indices],
        })

    relation_type = str(parsed.get("relation_type") or "unknown").strip().lower()
    if relation_type not in {"causal", "temporal"}:
        relation_type = "causal" if item["task_type"] == "Action Reasoning" else "temporal"
    requires_value = parsed.get("requires_verification")
    requires = bool(hypotheses) and (
        core.parse_bool(requires_value) if requires_value is not None else True
    )
    return {
        "question_id": item["question_id"],
        "request_signature": hypothesis_signature(item, source_record, api_base, api_model),
        "api_status": "ok",
        "api_error": "",
        "parse_status": parse_status,
        "api_attempts": len(raw_responses),
        "raw_response": raw,
        "raw_responses": raw_responses,
        "parsed": parsed,
        "requires_verification": requires,
        "relation_type": relation_type,
        "target_event": str(parsed.get("target_event") or "").strip(),
        "strongest_competitor": normalize_option(parsed.get("strongest_competitor")),
        "hypotheses": hypotheses,
        "storyboard_timestamps": timestamps,
        "duration": float(duration),
        "reason": str(parsed.get("reason") or "").strip(),
    }


def event_radius(hypothesis: dict) -> float:
    duration = float(hypothesis.get("duration") or 0.0)
    storyboard_count = max(1, len(hypothesis.get("storyboard_timestamps", [])))
    return min(MAX_EVENT_RADIUS_SECONDS, max(1.0, duration / storyboard_count))


def expand_event_timestamps(anchors: list[float], duration: float, radius: float, limit: int) -> list[float]:
    result = []
    for anchor in anchors:
        candidates = [max(0.0, anchor - radius), float(anchor), min(duration, anchor + radius)]
        for timestamp in candidates[:EVENT_FRAMES_PER_ANCHOR]:
            if all(abs(timestamp - existing) > 1e-6 for existing in result):
                result.append(float(timestamp))
    return sorted(result)[:limit]


def collect_event_windows(hypothesis: dict) -> dict:
    cause_anchors, effect_anchors = [], []
    for item in hypothesis.get("hypotheses", []):
        for value in item.get("cause_anchor_timestamps", []):
            if value not in cause_anchors:
                cause_anchors.append(float(value))
        for value in item.get("effect_anchor_timestamps", []):
            if value not in effect_anchors:
                effect_anchors.append(float(value))
    cause_anchors = cause_anchors[:MAX_CAUSE_ANCHORS]
    effect_anchors = effect_anchors[:MAX_EFFECT_ANCHORS]
    duration = float(hypothesis.get("duration") or 0.0)
    radius = event_radius(hypothesis)
    cause_timestamps = expand_event_timestamps(cause_anchors, duration, radius, MAX_CAUSE_FRAMES)
    effect_timestamps = expand_event_timestamps(effect_anchors, duration, radius, MAX_EFFECT_FRAMES)
    cause_intervals = [
        [max(0.0, anchor - radius), min(duration, anchor + radius)] for anchor in cause_anchors
    ]
    return {
        "cause_timestamps": cause_timestamps,
        "effect_timestamps": effect_timestamps,
        "cause_intervals": cause_intervals,
        "radius": radius,
    }


def sample_event_frames(video_path: str, windows: dict) -> dict:
    cause_raw, cause_ts = core.sample_unique_frames(video_path, windows.get("cause_timestamps", []), [])
    effect_raw, effect_ts = core.sample_unique_frames(video_path, windows.get("effect_timestamps", []), cause_ts)
    cause_frames = core.stamp_frames(cause_raw, cause_ts, "C")
    effect_frames = core.stamp_frames(effect_raw, effect_ts, "E")
    return {
        "cause_frames": cause_frames,
        "cause_timestamps": cause_ts,
        "effect_frames": effect_frames,
        "effect_timestamps": effect_ts,
        "cause_intervals": windows.get("cause_intervals", []),
        "radius": float(windows.get("radius") or 1.0),
    }


def certificate_prompt(item: dict, source_record: dict, hypothesis: dict, sampled: dict) -> str:
    cause_labels = ", ".join(
        f"C{i + 1}={timestamp:.2f}s" for i, timestamp in enumerate(sampled["cause_timestamps"])
    ) or "NONE"
    effect_labels = ", ".join(
        f"E{i + 1}={timestamp:.2f}s" for i, timestamp in enumerate(sampled["effect_timestamps"])
    ) or "NONE"
    compact_hypotheses = [
        {key: value for key, value in candidate.items() if key in {
            "option", "cause_or_earlier_event", "effect_or_later_event",
            "expected_pre_state", "expected_post_state",
        }}
        for candidate in hypothesis.get("hypotheses", [])
    ]
    return f"""You are a conservative causal-evidence auditor. Inspect only the attached
timestamped C/E event frames. Test the proposed hypotheses without inferring invisible
facts or favoring the answer under review.

Task type: {item['task_type']}
Question and options: {item['question']}
Answer under review: {source_record['methods']['agent_pre_reflection']['prediction']}
Relation type: {hypothesis.get('relation_type')}
Hypotheses: {json.dumps(compact_hypotheses, ensure_ascii=False)}
Cause/earlier frame labels: {cause_labels}
Effect/later frame labels: {effect_labels}

For every claim, cite attached frame IDs.  A causal claim requires temporal precedence
and an observable state transition or interaction.  Co-occurrence alone is insufficient.
Return INSUFFICIENT when the evidence cannot distinguish hypotheses.  REOPEN is allowed
only when a competing hypothesis has positive evidence and the reviewed hypothesis has
concrete contradictory evidence.
- Use only attached C/E IDs, never storyboard S IDs.
- Keep every text field within 40 characters and omit unnecessary details.
- Output one-line minified JSON only, with no markdown or explanation.

Return exactly this schema:
{{
  "hypothesis_checks": [
    {{
      "option": "A",
      "status": "SUPPORTED/REFUTED/INSUFFICIENT",
      "observed_cause_or_earlier_event": "visible fact only",
      "cause_evidence_ids": ["C1"],
      "observed_effect_or_later_event": "visible fact only",
      "effect_evidence_ids": ["E1"],
      "observed_pre_state": "visible state or empty",
      "observed_post_state": "visible state or empty",
      "temporal_precedence_verified": true,
      "state_transition_consistent": true,
      "alternative_cause_visible": false
    }}
  ],
  "strongest_supported_option": "A/B/C/D/UNKNOWN",
  "recommendation": "KEEP/REOPEN/ABSTAIN"
}}"""


def certificate_retry_prompt(item: dict, source_record: dict, hypothesis: dict, sampled: dict) -> str:
    compact = [
        {
            "option": value.get("option"),
            "cause": value.get("cause_or_earlier_event"),
            "effect": value.get("effect_or_later_event"),
        }
        for value in hypothesis.get("hypotheses", [])[:2]
    ]
    valid_ids = [
        *(f"C{i + 1}" for i in range(len(sampled.get("cause_timestamps", [])))),
        *(f"E{i + 1}" for i in range(len(sampled.get("effect_timestamps", [])))),
    ]
    return f"""Audit the attached C/E frames. Previous output was invalid or truncated.
Question: {item['question']}. Reviewed answer:
{source_record['methods']['agent_pre_reflection']['prediction']}. Hypotheses:
{json.dumps(compact, ensure_ascii=False)}. Valid frame IDs: {','.join(valid_ids)}.
Return ONE LINE of minified JSON only; no markdown, rationale, or text over 30 characters.
Schema: {{"hypothesis_checks":[{{"option":"A","status":"SUPPORTED",
"observed_cause_or_earlier_event":"fact","cause_evidence_ids":["C1"],
"observed_effect_or_later_event":"fact","effect_evidence_ids":["E1"],
"observed_pre_state":"state","observed_post_state":"state",
"temporal_precedence_verified":true,"state_transition_consistent":true,
"alternative_cause_visible":false}}],"strongest_supported_option":"A",
"recommendation":"KEEP"}}. Include one check per supplied hypothesis. Use only valid IDs."""


def certificate_signature(item: dict, hypothesis: dict, sampled: dict, api_base: str, api_model: str) -> str:
    return core.stable_hash({
        "protocol": PROTOCOL_VERSION,
        "stage": "certificate",
        "question": item["question"],
        "hypothesis_signature": hypothesis.get("request_signature"),
        "cause_timestamps": sampled.get("cause_timestamps", []),
        "effect_timestamps": sampled.get("effect_timestamps", []),
        "api_base": api_base,
        "api_model": api_model,
    })


def empty_certificate(item: dict, hypothesis: dict, sampled: dict, api_base: str, api_model: str, status: str, reason: str, error: str = "") -> dict:
    return {
        "question_id": item["question_id"],
        "request_signature": certificate_signature(item, hypothesis, sampled, api_base, api_model),
        "api_status": status,
        "api_error": error,
        "parse_status": "skipped" if status.startswith("skipped_") else status,
        "api_attempts": 0,
        "raw_response": "",
        "raw_responses": [],
        "parsed": {},
        "hypothesis_checks": [],
        "strongest_supported_option": "",
        "recommendation": "ABSTAIN",
        "reason": reason,
    }


def run_certificate(item: dict, source_record: dict, hypothesis: dict, sampled: dict, api_key: str, api_base: str, api_model: str) -> dict:
    if not hypothesis.get("requires_verification"):
        return empty_certificate(item, hypothesis, sampled, api_base, api_model, "skipped_no_hypothesis", "No grounded causal hypothesis was produced.")
    if not len(sampled.get("cause_frames", [])) or not len(sampled.get("effect_frames", [])):
        return empty_certificate(item, hypothesis, sampled, api_base, api_model, "skipped_missing_event_frames", "Cause or effect event frames are missing.")

    frames = core.merge_chronological_frames(
        sampled["cause_frames"], sampled["cause_timestamps"],
        sampled["effect_frames"], sampled["effect_timestamps"],
    )
    required_options = {
        value.get("option") for value in hypothesis.get("hypotheses", []) if value.get("option")
    }
    try:
        parsed, raw_responses, parse_status = call_validated_json(
            api_key, api_base, api_model,
            [
                certificate_prompt(item, source_record, hypothesis, sampled),
                certificate_retry_prompt(item, source_record, hypothesis, sampled),
            ],
            list(frames),
            CERTIFICATE_MAX_TOKENS,
            lambda value: valid_certificate_json(value, required_options),
        )
    except core.ApiContentBlocked as exc:
        print(f"[Qwen content blocked] causal certificate {item['question_id']}; keeping the current answer.")
        return empty_certificate(item, hypothesis, sampled, api_base, api_model, "content_blocked", "Event frames rejected by API content inspection.", str(exc))

    raw = raw_responses[-1] if raw_responses else ""
    if parse_status == "failed":
        failed = empty_certificate(
            item, hypothesis, sampled, api_base, api_model,
            "parse_failed", "Certificate JSON validation failed.",
            "Qwen returned invalid or truncated certificate JSON twice.",
        )
        failed.update({
            "parse_status": parse_status,
            "api_attempts": len(raw_responses),
            "raw_response": raw,
            "raw_responses": raw_responses,
        })
        return failed
    valid_ids = {
        *(f"C{i + 1}" for i in range(len(sampled["cause_timestamps"]))),
        *(f"E{i + 1}" for i in range(len(sampled["effect_timestamps"]))),
    }
    checks = []
    raw_checks = parsed.get("hypothesis_checks") if isinstance(parsed.get("hypothesis_checks"), list) else []
    for value in raw_checks[:4]:
        if not isinstance(value, dict):
            continue
        option = normalize_option(value.get("option"))
        if not option:
            continue
        status = str(value.get("status") or "INSUFFICIENT").strip().upper()
        if status not in {"SUPPORTED", "REFUTED", "INSUFFICIENT"}:
            status = "INSUFFICIENT"
        cause_ids = normalize_ids(value.get("cause_evidence_ids"), valid_ids)
        effect_ids = normalize_ids(value.get("effect_evidence_ids"), valid_ids)
        checks.append({
            "option": option,
            "status": status,
            "observed_cause_or_earlier_event": clean_observation(value.get("observed_cause_or_earlier_event")),
            "cause_evidence_ids": cause_ids,
            "observed_effect_or_later_event": clean_observation(value.get("observed_effect_or_later_event")),
            "effect_evidence_ids": effect_ids,
            "observed_pre_state": clean_observation(value.get("observed_pre_state")),
            "observed_post_state": clean_observation(value.get("observed_post_state")),
            "temporal_precedence_verified": core.parse_bool(value.get("temporal_precedence_verified")),
            "state_transition_consistent": core.parse_bool(value.get("state_transition_consistent")),
            "alternative_cause_visible": core.parse_bool(value.get("alternative_cause_visible")),
            "alternative_cause_assessed": "alternative_cause_visible" in value,
            "reason": str(value.get("reason") or "").strip(),
        })
    recommendation = str(parsed.get("recommendation") or "ABSTAIN").strip().upper()
    if recommendation not in {"KEEP", "REOPEN", "ABSTAIN"}:
        recommendation = "ABSTAIN"
    return {
        "question_id": item["question_id"],
        "request_signature": certificate_signature(item, hypothesis, sampled, api_base, api_model),
        "api_status": "ok",
        "api_error": "",
        "parse_status": parse_status,
        "api_attempts": len(raw_responses),
        "raw_response": raw,
        "raw_responses": raw_responses,
        "parsed": parsed,
        "hypothesis_checks": checks,
        "strongest_supported_option": normalize_option(parsed.get("strongest_supported_option")),
        "recommendation": recommendation,
        "reason": str(parsed.get("reason") or "").strip(),
    }


def prioritized_event_ids(certificate: dict, answer_under_review: str) -> list[str]:
    checks = certificate.get("hypothesis_checks", [])
    option_order = [certificate.get("strongest_supported_option"), answer_under_review]
    option_order += [check.get("option") for check in checks]
    result = []
    for option in option_order:
        check = next((value for value in checks if value.get("option") == option), None)
        if not check:
            continue
        for evidence_id in check.get("cause_evidence_ids", []) + check.get("effect_evidence_ids", []):
            if evidence_id not in result:
                result.append(evidence_id)
    return result


def select_causal_frames_for_answer(certificate: dict, sampled: dict, answer_under_review: str) -> dict:
    available = {
        **{f"C{index + 1}": (sampled["cause_frames"][index], sampled["cause_timestamps"][index], "CAUSE")
           for index in range(len(sampled.get("cause_timestamps", [])))},
        **{f"E{index + 1}": (sampled["effect_frames"][index], sampled["effect_timestamps"][index], "EFFECT")
           for index in range(len(sampled.get("effect_timestamps", [])))},
    }
    order = prioritized_event_ids(certificate, answer_under_review)
    order += [evidence_id for evidence_id in available if evidence_id not in order]
    # Preserve both sides of the relation whenever possible.
    selected = []
    for prefix in ("C", "E"):
        for evidence_id in order:
            if evidence_id.startswith(prefix) and evidence_id in available and evidence_id not in selected:
                selected.append(evidence_id)
                break
    for evidence_id in order:
        if evidence_id in available and evidence_id not in selected:
            selected.append(evidence_id)
        if len(selected) >= MAX_CAUSAL_FRAMES_TO_ANSWER_MODEL:
            break
    cause_ids = [value for value in selected if value.startswith("C")]
    effect_ids = [value for value in selected if value.startswith("E")]

    def stack(ids: list[str]):
        if not ids:
            return np.zeros((0, 336, 336, 3), dtype=np.uint8), []
        return np.stack([available[value][0] for value in ids], axis=0), [available[value][1] for value in ids]

    cause_frames, cause_ts = stack(cause_ids)
    effect_frames, effect_ts = stack(effect_ids)
    return {
        "cause_frames": cause_frames,
        "cause_timestamps": cause_ts,
        "effect_frames": effect_frames,
        "effect_timestamps": effect_ts,
        "allowed_ids": set(selected),
    }


def candidate_cause_intervals(hypothesis: dict, certificate: dict, fallback: list[list[float]]) -> list[list[float]]:
    target = certificate.get("strongest_supported_option")
    candidate = next((value for value in hypothesis.get("hypotheses", []) if value.get("option") == target), None)
    if not candidate or not candidate.get("cause_anchor_timestamps"):
        return fallback
    duration = float(hypothesis.get("duration") or 0.0)
    radius = event_radius(hypothesis)
    return [
        [max(0.0, float(anchor) - radius), min(duration, float(anchor) + radius)]
        for anchor in candidate["cause_anchor_timestamps"]
    ]


def evidence_lines_from_certificate(certificate: dict, include_cause: bool, allowed_ids: set[str] | None = None) -> list[str]:
    lines, seen = [], set()
    for check in certificate.get("hypothesis_checks", []):
        fields = []
        if include_cause:
            fields.extend([
                (check.get("observed_cause_or_earlier_event", ""), check.get("cause_evidence_ids", [])),
                (check.get("observed_pre_state", ""), check.get("cause_evidence_ids", [])),
            ])
        fields.extend([
            (check.get("observed_effect_or_later_event", ""), check.get("effect_evidence_ids", [])),
            (check.get("observed_post_state", ""), check.get("effect_evidence_ids", [])),
        ])
        for fact, evidence_ids in fields:
            fact = str(fact or "").strip()
            if allowed_ids is not None:
                evidence_ids = [value for value in evidence_ids if value in allowed_ids]
            if not fact or not evidence_ids:
                continue
            text = f"[{','.join(evidence_ids)}] {fact}"
            if text not in seen:
                seen.add(text)
                lines.append(text)
    if not lines:
        return []
    heading = "Observed timestamp-grounded event evidence: "
    return [heading + " | ".join(lines[:16])]


def merge_sources(sources: list[tuple[np.ndarray, list[float], str, int]]) -> tuple[np.ndarray, list[float], list[str]]:
    """Chronologically merge frames and replace exact duplicates with higher-priority evidence."""
    entries = {}
    for frames, timestamps, source, priority in sources:
        for index, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            key = round(float(timestamp), 3)
            previous = entries.get(key)
            if previous is None or priority >= previous[0]:
                entries[key] = (priority, index, frame, float(timestamp), source)
    ordered = sorted(entries.values(), key=lambda value: value[3])
    if not ordered:
        return np.zeros((0, 336, 336, 3), dtype=np.uint8), [], []
    return np.stack([value[2] for value in ordered], axis=0), [value[3] for value in ordered], [value[4] for value in ordered]


def inside_any_interval(timestamp: float, intervals: list[list[float]]) -> bool:
    return any(float(start) <= timestamp <= float(end) for start, end in intervals)


def mask_cause_intervals(frames, timestamps: list[float], sources: list[str], intervals: list[list[float]]):
    keep = [index for index, timestamp in enumerate(timestamps) if not inside_any_interval(timestamp, intervals)]
    if not keep:
        return frames, timestamps, sources
    return frames[keep], [timestamps[index] for index in keep], [sources[index] for index in keep]


def check_for_option(certificate: dict, option: str) -> dict | None:
    for check in certificate.get("hypothesis_checks", []):
        if check.get("option") == option:
            return check
    return None


def conservative_decision(
    source_record: dict,
    hypothesis: dict,
    certificate: dict,
    full_prediction: str,
    masked_prediction: str,
    intervention_removed_count: int,
) -> dict:
    original = source_record["methods"]["agent_pre_reflection"]["prediction"]
    reasons = []
    if not original or not full_prediction or full_prediction == original:
        return {"decision": "KEEP", "accepted_change": False, "prediction": original, "reasons": ["No distinct grounded candidate answer."]}
    candidate_check = check_for_option(certificate, full_prediction)
    original_check = check_for_option(certificate, original)
    if certificate.get("api_status") != "ok":
        reasons.append("Certificate unavailable.")
    if certificate.get("recommendation") != "REOPEN":
        reasons.append("Auditor did not request REOPEN.")
    if certificate.get("strongest_supported_option") != full_prediction:
        reasons.append("Full-evidence candidate is not the strongest supported hypothesis.")
    if not candidate_check or candidate_check.get("status") != "SUPPORTED":
        reasons.append("Candidate hypothesis lacks SUPPORTED status.")
    if not original_check or original_check.get("status") != "REFUTED":
        reasons.append("Original hypothesis lacks concrete REFUTED status.")
    elif not original_check.get("cause_evidence_ids") and not original_check.get("effect_evidence_ids"):
        reasons.append("Original hypothesis refutation has no grounded evidence citation.")
    if candidate_check:
        if not candidate_check.get("temporal_precedence_verified"):
            reasons.append("Temporal precedence is not verified.")
        if source_record.get("task_type") == "Action Reasoning" and not candidate_check.get("state_transition_consistent"):
            reasons.append("State transition is not verified for causal action reasoning.")
        if not candidate_check.get("alternative_cause_assessed"):
            reasons.append("Alternative-cause check is missing.")
        elif candidate_check.get("alternative_cause_visible"):
            reasons.append("A competing cause remains visible.")
        if not any(evidence_id.startswith("C") for evidence_id in candidate_check.get("cause_evidence_ids", [])):
            reasons.append("No grounded cause-frame citation.")
        if not any(evidence_id.startswith("E") for evidence_id in candidate_check.get("effect_evidence_ids", [])):
            reasons.append("No grounded effect-frame citation.")
        if not candidate_check.get("observed_cause_or_earlier_event"):
            reasons.append("No observable cause/earlier-event fact.")
        if not candidate_check.get("observed_effect_or_later_event"):
            reasons.append("No observable effect/later-event fact.")
    if intervention_removed_count <= 0:
        reasons.append("Evidence intervention removed no frames from the candidate cause interval.")
    if not masked_prediction or masked_prediction == full_prediction:
        reasons.append("Evidence intervention did not change the candidate answer.")
    if reasons:
        return {"decision": "KEEP", "accepted_change": False, "prediction": original, "reasons": reasons}
    return {
        "decision": "REOPEN_ACCEPTED",
        "accepted_change": True,
        "prediction": full_prediction,
        "reasons": ["Competing hypothesis passed grounded temporal/state and evidence-intervention checks."],
    }


def method_result(raw: str, ground_truth: str) -> dict:
    return core.method_result(raw, ground_truth)


def exact_mcnemar(gains: int, losses: int) -> float:
    total = gains + losses
    if not total:
        return 1.0
    tail = min(gains, losses)
    return min(1.0, 2 * sum(math.comb(total, index) for index in range(tail + 1)) / (2 ** total))


def summarize(records: dict[str, dict], expected: int) -> dict:
    summary = {
        "expected": expected,
        "completed": len(records),
        "methods": {},
        "paired_vs_pre": {},
        "paired_vs_baseline": {},
        "by_task_type": {},
    }
    for method in METHODS:
        values = [record["methods"][method] for record in records.values()]
        correct = sum(bool(value["correct"]) for value in values)
        summary["methods"][method] = {
            "correct": correct,
            "total": len(values),
            "accuracy": round(correct / len(values), 4) if values else 0.0,
        }
    for method in ("causal_full_evidence", "causal_conservative_final"):
        gains = sum(
            not record["methods"]["agent_pre_reflection"]["correct"] and record["methods"][method]["correct"]
            for record in records.values()
        )
        losses = sum(
            record["methods"]["agent_pre_reflection"]["correct"] and not record["methods"][method]["correct"]
            for record in records.values()
        )
        summary["paired_vs_pre"][method] = {
            "gains": gains,
            "losses": losses,
            "net": gains - losses,
            "mcnemar_exact_p": round(exact_mcnemar(gains, losses), 6),
        }
    for method in ("agent_pre_reflection", "causal_conservative_final"):
        gains = sum(
            not record["methods"]["baseline_paired"]["correct"] and record["methods"][method]["correct"]
            for record in records.values()
        )
        losses = sum(
            record["methods"]["baseline_paired"]["correct"] and not record["methods"][method]["correct"]
            for record in records.values()
        )
        summary["paired_vs_baseline"][method] = {
            "gains": gains,
            "losses": losses,
            "net": gains - losses,
            "mcnemar_exact_p": round(exact_mcnemar(gains, losses), 6),
        }
    for task_type in sorted(TARGET_TASK_TYPES):
        subset = [record for record in records.values() if record["task_type"] == task_type]
        summary["by_task_type"][task_type] = {}
        for method in METHODS:
            correct = sum(record["methods"][method]["correct"] for record in subset)
            summary["by_task_type"][task_type][method] = {
                "correct": correct,
                "total": len(subset),
                "accuracy": round(correct / len(subset), 4) if subset else 0.0,
            }
    corrected = sum(
        not record["methods"]["agent_pre_reflection"]["correct"]
        and record["methods"]["causal_conservative_final"]["correct"]
        for record in records.values()
    )
    corrupted = sum(
        record["methods"]["agent_pre_reflection"]["correct"]
        and not record["methods"]["causal_conservative_final"]["correct"]
        for record in records.values()
    )
    accepted = sum(record["decision"].get("accepted_change", False) for record in records.values())
    summary["causal_verifier"] = {
        "hypothesis_api_status": dict(Counter(record["hypothesis"].get("api_status", "unknown") for record in records.values())),
        "certificate_api_status": dict(Counter(record["certificate"].get("api_status", "unknown") for record in records.values())),
        "hypothesis_parse_status": dict(Counter(record["hypothesis"].get("parse_status", "unknown") for record in records.values())),
        "certificate_parse_status": dict(Counter(record["certificate"].get("parse_status", "unknown") for record in records.values())),
        "recommendations": dict(Counter(record["certificate"].get("recommendation", "ABSTAIN") for record in records.values())),
        "effective_evidence_questions": sum(bool(record.get("full_evidence_lines")) for record in records.values()),
        "effective_evidence_rate": round(
            sum(bool(record.get("full_evidence_lines")) for record in records.values()) / len(records), 4
        ) if records else 0.0,
        "accepted_changes": accepted,
        "corrected_errors": corrected,
        "corrupted_correct_answers": corrupted,
        "net_corrections": corrected - corrupted,
        "correction_precision_on_changed_correctness": round(corrected / (corrected + corrupted), 4) if corrected + corrupted else None,
        "average_full_frames": round(sum(record.get("full_frame_count", GLOBAL_FRAMES) for record in records.values()) / len(records), 4) if records else 0.0,
        "average_masked_frames": round(sum(record.get("masked_frame_count", GLOBAL_FRAMES) for record in records.values()) / len(records), 4) if records else 0.0,
    }
    return summary


def write_outputs(output_dir: Path, records: dict[str, dict], expected: int) -> None:
    (output_dir / "summary.json").write_text(json.dumps(summarize(records, expected), ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["question_id", "task_type", "ground_truth"]
    for method in METHODS:
        fields += [f"{method}_prediction", f"{method}_correct"]
    fields += [
        "hypothesis_api_status", "hypothesis_parse_status",
        "certificate_api_status", "certificate_parse_status", "recommendation",
        "decision", "accepted_change", "intervention_removed_count",
        "full_frame_count", "masked_frame_count",
    ]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for qid in sorted(records):
            record = records[qid]
            row = {key: record[key] for key in ("question_id", "task_type", "ground_truth")}
            for method in METHODS:
                row[f"{method}_prediction"] = record["methods"][method]["prediction"]
                row[f"{method}_correct"] = record["methods"][method]["correct"]
            row.update({
                "hypothesis_api_status": record["hypothesis"].get("api_status"),
                "hypothesis_parse_status": record["hypothesis"].get("parse_status"),
                "certificate_api_status": record["certificate"].get("api_status"),
                "certificate_parse_status": record["certificate"].get("parse_status"),
                "recommendation": record["certificate"].get("recommendation"),
                "decision": record["decision"].get("decision"),
                "accepted_change": record["decision"].get("accepted_change"),
                "intervention_removed_count": record.get("intervention_removed_count"),
                "full_frame_count": record.get("full_frame_count"),
                "masked_frame_count": record.get("masked_frame_count"),
            })
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="C-level causal evidence verification experiment")
    parser.add_argument("--source_results", default=str(DEFAULT_SOURCE_RESULTS))
    parser.add_argument("--data_dir", default=str(core.DEFAULT_DATA_DIR))
    parser.add_argument("--json", default=str(core.DEFAULT_ANNOTATION))
    parser.add_argument("--level_jsonl", default=str(core.DEFAULT_LEVELS))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--allow_partial_dataset", action="store_true")
    parser.add_argument("--api_key", default=CAUSAL_QWEN_API_KEY or core.QWEN_API_KEY or os.environ.get("DASHSCOPE_API_KEY") or "")
    parser.add_argument("--api_base", default=core.QWEN_API_BASE)
    parser.add_argument("--api_model", default=core.QWEN_API_MODEL)
    parser.add_argument("--prepare_only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = Path(args.source_results).resolve()
    source_records = core.load_jsonl(source_path)
    source_targets = [
        record for record in source_records.values()
        if record.get("question_level") == TARGET_LEVEL and record.get("task_type") in TARGET_TASK_TYPES
    ]
    question_map = {
        item["question_id"]: item
        for item in core.build_question_list(
            Path(args.json).resolve(), Path(args.data_dir).resolve(), Path(args.level_jsonl).resolve()
        )
    }
    targets = []
    for source_record in source_targets:
        item = question_map.get(source_record["question_id"])
        if item:
            targets.append((item, source_record))
    targets.sort(key=lambda pair: pair[0]["question_id"])
    if args.max_questions is not None:
        targets = targets[:max(0, args.max_questions)]
    if args.max_questions is None and not args.allow_partial_dataset:
        if len(source_targets) != EXPECTED_TARGET_QUESTIONS or len(targets) != EXPECTED_TARGET_QUESTIONS:
            raise RuntimeError(
                f"Causal full-data guard failed: source has {len(source_targets)} target questions and this machine has "
                f"{len(targets)} runnable targets; expected {EXPECTED_TARGET_QUESTIONS}."
            )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol": PROTOCOL_VERSION,
        "source_results_hash": core.file_hash(source_path),
        "question_ids_hash": core.stable_hash([item["question_id"] for item, _ in targets]),
        "video_manifest_hash": core.video_manifest_hash(Path(args.data_dir).resolve()),
        "storyboard_frames": CAUSAL_STORYBOARD_FRAMES,
        "global_frames": GLOBAL_FRAMES,
        "max_causal_frames_to_answer_model": MAX_CAUSAL_FRAMES_TO_ANSWER_MODEL,
        "max_cause_frames_to_auditor": MAX_CAUSE_FRAMES,
        "max_effect_frames_to_auditor": MAX_EFFECT_FRAMES,
        "hypothesis_max_tokens": HYPOTHESIS_MAX_TOKENS,
        "certificate_max_tokens": CERTIFICATE_MAX_TOKENS,
        "json_api_attempts": JSON_API_ATTEMPTS,
        "target_types": sorted(TARGET_TASK_TYPES),
        "api_base": args.api_base,
        "api_model": args.api_model,
    }
    fingerprint = core.stable_hash(config)
    manifest_path = output_dir / "manifest.json"
    cache_names = ("hypothesis_cache.jsonl", "certificate_cache.jsonl", "results.jsonl")
    caches_exist = any((output_dir / name).exists() and (output_dir / name).stat().st_size for name in cache_names)
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("config_fingerprint") != fingerprint and caches_exist:
            raise RuntimeError("Output directory contains causal caches from another configuration; use a new --output_dir.")
    elif caches_exist:
        raise RuntimeError("Output directory contains unversioned causal caches; use a new --output_dir.")
    manifest = {
        "source_target_questions": len(source_targets),
        "runnable_questions": len(targets),
        "expected_questions": EXPECTED_TARGET_QUESTIONS,
        "config_fingerprint": fingerprint,
        "config": config,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Manifest] {manifest}")
    if args.prepare_only:
        return 0
    if not args.api_key:
        raise RuntimeError("Qwen API key is empty. Fill CAUSAL_QWEN_API_KEY at the top of this file.")

    import video_evidence_backend as baseline

    hypothesis_path = output_dir / "hypothesis_cache.jsonl"
    certificate_path = output_dir / "certificate_cache.jsonl"
    results_path = output_dir / "results.jsonl"
    hypothesis_cache = core.load_jsonl(hypothesis_path)
    certificate_cache = core.load_jsonl(certificate_path)
    records = core.load_jsonl(results_path)
    pending = [
        pair for pair in targets
        if records.get(pair[0]["question_id"], {}).get("config_fingerprint") != fingerprint
        or records.get(pair[0]["question_id"], {}).get("hypothesis", {}).get("api_status") == "parse_failed"
        or records.get(pair[0]["question_id"], {}).get("certificate", {}).get("api_status") == "parse_failed"
    ]

    for position, (item, source_record) in enumerate(pending, 1):
        qid = item["question_id"]
        print(f"\n[{position}/{len(pending)}] {qid} {item['task_type']}")
        expected_hypothesis_signature = hypothesis_signature(item, source_record, args.api_base, args.api_model)
        hypothesis = hypothesis_cache.get(qid)
        if (
            hypothesis is None
            or hypothesis.get("request_signature") != expected_hypothesis_signature
            or hypothesis.get("api_status") == "parse_failed"
        ):
            hypothesis = run_hypothesis_planner(
                baseline, item, source_record, args.api_key, args.api_base, args.api_model
            )
            core.append_jsonl(hypothesis_path, hypothesis)
            hypothesis_cache[qid] = hypothesis

        windows = collect_event_windows(hypothesis)
        sampled = sample_event_frames(item["video_path"], windows) if hypothesis.get("requires_verification") else {
            "cause_frames": np.zeros((0, 336, 336, 3), dtype=np.uint8),
            "cause_timestamps": [],
            "effect_frames": np.zeros((0, 336, 336, 3), dtype=np.uint8),
            "effect_timestamps": [],
            "cause_intervals": [],
            "radius": 1.0,
        }
        expected_certificate_signature = certificate_signature(item, hypothesis, sampled, args.api_base, args.api_model)
        certificate = certificate_cache.get(qid)
        if (
            certificate is None
            or certificate.get("request_signature") != expected_certificate_signature
            or certificate.get("api_status") == "parse_failed"
        ):
            certificate = run_certificate(
                item, source_record, hypothesis, sampled,
                args.api_key, args.api_base, args.api_model,
            )
            core.append_jsonl(certificate_path, certificate)
            certificate_cache[qid] = certificate

        original_prediction = source_record["methods"]["agent_pre_reflection"]["prediction"]
        source_initial_local_ts = initial_local_timestamps(source_record)
        full_raw = original_prediction
        masked_raw = original_prediction
        full_frame_count = GLOBAL_FRAMES + len(source_initial_local_ts)
        masked_frame_count = full_frame_count
        intervention_removed_count = 0
        selected_causal = select_causal_frames_for_answer(certificate, sampled, original_prediction)
        full_lines = evidence_lines_from_certificate(
            certificate, include_cause=True, allowed_ids=selected_causal["allowed_ids"]
        )
        effect_lines = evidence_lines_from_certificate(
            certificate, include_cause=False, allowed_ids=selected_causal["allowed_ids"]
        )

        if certificate.get("api_status") == "ok" and full_lines:
            global_frames, global_time_string, _ = baseline.process_video(
                item["video_path"], GLOBAL_FRAMES, 1, force_sample=True
            )
            global_ts = [float(value.replace("s", "")) for value in global_time_string.split(",")]
            pre_local_ts = source_initial_local_ts
            pre_local_raw, pre_local_ts = core.sample_unique_frames(item["video_path"], pre_local_ts, global_ts)
            pre_local_frames = core.stamp_frames(pre_local_raw, pre_local_ts, "L")
            full_frames, full_ts, full_sources = merge_sources([
                (global_frames, global_ts, "GLOBAL", 0),
                (pre_local_frames, pre_local_ts, "LOCAL", 1),
                (selected_causal["cause_frames"], selected_causal["cause_timestamps"], "CAUSE", 2),
                (selected_causal["effect_frames"], selected_causal["effect_timestamps"], "EFFECT", 2),
            ])
            intervention_intervals = candidate_cause_intervals(
                hypothesis, certificate, sampled.get("cause_intervals", [])
            )
            masked_frames, masked_ts, masked_sources = mask_cause_intervals(
                full_frames, full_ts, full_sources, intervention_intervals
            )
            intervention_removed_count = len(full_frames) - len(masked_frames)
            full_prompt = core.insert_evidence(source_record["agent_prompt"], full_lines)
            masked_prompt = core.insert_evidence(source_record["agent_prompt"], effect_lines)
            full_raw = core.infer_frames(baseline, full_prompt, full_frames)
            masked_raw = core.infer_frames(baseline, masked_prompt, masked_frames)
            full_frame_count = len(full_frames)
            masked_frame_count = len(masked_frames)

        full_prediction = normalize_option(full_raw)
        masked_prediction = normalize_option(masked_raw)
        decision = conservative_decision(
            source_record, hypothesis, certificate, full_prediction, masked_prediction,
            intervention_removed_count,
        )
        final_prediction = decision["prediction"]
        ground_truth = source_record["ground_truth"]
        methods = {
            "baseline_paired": source_record["methods"]["baseline_paired"],
            "agent_pre_reflection": source_record["methods"]["agent_pre_reflection"],
            "causal_full_evidence": method_result(full_raw, ground_truth),
            "causal_masked_evidence": method_result(masked_raw, ground_truth),
            "causal_conservative_final": method_result(final_prediction, ground_truth),
        }
        record = {
            "question_id": qid,
            "question_level": source_record["question_level"],
            "task_type": source_record["task_type"],
            "question": source_record["question"],
            "ground_truth": ground_truth,
            "config_fingerprint": fingerprint,
            "methods": methods,
            "hypothesis": hypothesis,
            "certificate": certificate,
            "decision": decision,
            "initial_local_timestamps": source_initial_local_ts,
            "cause_timestamps": sampled.get("cause_timestamps", []),
            "effect_timestamps": sampled.get("effect_timestamps", []),
            "cause_intervals": sampled.get("cause_intervals", []),
            "selected_causal_evidence_ids": sorted(selected_causal["allowed_ids"]),
            "intervention_removed_count": intervention_removed_count,
            "full_frame_count": full_frame_count,
            "masked_frame_count": masked_frame_count,
            "full_evidence_lines": full_lines,
            "effect_only_evidence_lines": effect_lines,
        }
        core.append_jsonl(results_path, record)
        records[qid] = record
        write_outputs(output_dir, records, len(targets))
        print(
            "[Done]", qid, "GT=", ground_truth,
            "pre=", original_prediction, "full=", full_prediction,
            "masked=", masked_prediction, "final=", final_prediction,
            "decision=", decision["decision"],
        )

    write_outputs(output_dir, records, len(targets))
    print(f"\n[Complete] {len(records)}/{len(targets)}")
    print(f"[Results] {results_path}")
    print(f"[Summary] {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
