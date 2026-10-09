"""Post-answer evidence verifier with a frozen fallback answer.

The existing agent is not modified.  This experiment reads its completed results,
audits only C-level Action/Temporal Reasoning questions, and changes an answer only
when two order-swapped audits and the local answer model all agree. Explicit
multi-event order questions use a complete timeline verifier; causal/state-change
questions use the symmetric causal-consistency verifier.

Full run:
    python post_answer_causal_verifier.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

import evidence_gap_reflective_agent as core
import causal_evidence_verifier as causal_verifier


HERE = Path(__file__).resolve().parent
POST_CAUSAL_QWEN_API_KEY = causal_verifier.CAUSAL_QWEN_API_KEY
DEFAULT_SOURCE_RESULTS = HERE / "result" / "evidence_gap_reflective" / "results.jsonl"
DEFAULT_OUTPUT_DIR = HERE / "result" / "post_answer_causal_verifier"

PROTOCOL_VERSION = "post-answer-window-consensus-final"
BASE_METHOD = "agent_pre_reflection"
TARGET_LEVEL = "C"
TARGET_TASK_TYPES = {"Action Reasoning", "Temporal Reasoning"}
EXPECTED_TARGET_QUESTIONS = 151

GLOBAL_FRAMES = 16
STORYBOARD_FRAMES = 48
MAX_AUDIT_FRAMES = 16
MAX_ORDER_AUDIT_FRAMES = 24
MAX_FINAL_CAUSAL_FRAMES = 8
MAX_FINAL_ORDER_FRAMES = 12
MAX_ANCHORS_PER_ROLE = 2
MAX_EVENT_RADIUS_SECONDS = 8.0
PLANNER_MAX_TOKENS = 2048
AUDITOR_MAX_TOKENS = 2048
JSON_ATTEMPTS = 2

METHODS = ("answer_before_causal", "causal_candidate", "answer_after_causal")
ROUTE_TEMPORAL_ORDER = "TEMPORAL_ORDER"
ROUTE_CAUSAL_CONSISTENCY = "CAUSAL_CONSISTENCY"


def option(value) -> str:
    return core.extract_letter(value)


def base_prediction(record: dict) -> str:
    return option(record.get("methods", {}).get(BASE_METHOD, {}).get("prediction"))


def verifier_route(item: dict) -> str:
    text = str(item.get("question") or "").lower()
    stem = text.split("\na.", 1)[0]
    order_pattern = re.search(
        r"\b(?:in which order|in what order|what is the order|order in which|correct order|"
        r"chronological order|sequence in which)\b",
        stem,
    ) or re.search(
        r"\bwhich\s+of\s+the\s+following[^?\n]*\b(?:order|sequence|sequences)\b|"
        r"\bwhich\s+(?:order|sequence)\b[^?\n]*\bis\s+correct\b|"
        r"\b(?:order|sequence|sequences)(?:\s+of\s+"
        r"(?:events|scenes|scenarios|actions))?\s+is\s+correct\b|"
        r"\bcorrect\s+(?:scene\s+|video\s+|event\s+)?sequence\b",
        stem,
    )
    return ROUTE_TEMPORAL_ORDER if item.get("task_type") == "Temporal Reasoning" and order_pattern else ROUTE_CAUSAL_CONSISTENCY


def repaired_json_candidates(raw: str) -> list[dict]:
    """Conservatively repair only recurrent Qwen JSON punctuation/boolean errors."""
    text = str(raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    strings = [text]
    normalized = re.sub(
        r'("(?:temporal_precedence_verified|state_transition_consistent|alternative_cause_visible)"\s*:\s*)UNKNOWN\b',
        r"\1false",
        text,
        flags=re.I,
    )
    if normalized not in strings:
        strings.append(normalized)
    structural = []
    for value in strings:
        structural.append(value)
        if value.endswith("}}") and '"hypotheses"' in value:
            structural.append(value[:-1] + "]}")
        if value.endswith("}}]") and '"hypotheses"' in value:
            structural.append(value[:-2] + "]}")
        if value.endswith("}]"):
            structural.append(value + "}")
    result = []
    for value in structural:
        try:
            parsed = json.loads(value)
        except Exception:
            continue
        if isinstance(parsed, dict) and parsed not in result:
            result.append(parsed)
    return result


def request_json(api_key, api_base, api_model, prompts, frames, max_tokens, validator):
    raws = []
    for attempt, prompt in enumerate(prompts[:JSON_ATTEMPTS], 1):
        raw = core.api_chat(api_key, api_base, api_model, prompt, frames, max_tokens=max_tokens)
        raws.append(raw)
        parsed = core.parse_json_loose(raw)
        if validator(parsed):
            return parsed, raws, "valid_first" if attempt == 1 else "valid_retry"
        for repaired in repaired_json_candidates(raw):
            if validator(repaired):
                return repaired, raws, "valid_repaired_first" if attempt == 1 else "valid_repaired_retry"
    return {}, raws, "failed"


def planner_prompt(question: str, reviewed: str, task_type: str, timestamps: list[float], retry: bool = False) -> str:
    labels = ", ".join(f"S{i}={value:.2f}s" for i, value in enumerate(timestamps))
    prefix = "Your previous JSON was invalid. " if retry else ""
    return f"""{prefix}Inspect the chronological storyboard for post-answer causal verification.
The reviewed answer is {reviewed}. Select exactly one different strongest competitor.
Create exactly TWO hypotheses: one for {reviewed}, one for that competitor. Do not
decide which is correct. Every cause/earlier and effect/later event must be observable
and anchored to storyboard indices. For Action Reasoning include visible pre/post states.
Question and options: {question}
Task type: {task_type}
Storyboard labels: {labels}
Rules: exactly two distinct option letters; the reviewed option must appear once; the
competitor must appear once; every hypothesis needs nonempty cause and effect indices;
each text field <= 40 characters. Return one-line minified JSON only, no markdown.
Schema: {{"strongest_competitor":"B","hypotheses":[{{"option":"{reviewed}",
"cause_or_earlier_event":"visible event","effect_or_later_event":"visible event",
"expected_pre_state":"state","expected_post_state":"state",
"cause_anchor_indices":[0],"effect_anchor_indices":[1]}},{{"option":"B",
"cause_or_earlier_event":"visible event","effect_or_later_event":"visible event",
"expected_pre_state":"state","expected_post_state":"state",
"cause_anchor_indices":[2],"effect_anchor_indices":[3]}}]}}"""


def valid_plan_json(parsed: dict, reviewed: str, frame_count: int) -> bool:
    if not isinstance(parsed, dict):
        return False
    values = parsed.get("hypotheses")
    if not isinstance(values, list) or len(values) != 2:
        return False
    options = [option(value.get("option")) for value in values if isinstance(value, dict)]
    competitor = option(parsed.get("strongest_competitor"))
    if len(options) != 2 or len(set(options)) != 2 or reviewed not in options:
        return False
    if not competitor or competitor == reviewed or competitor not in options:
        return False
    for value in values:
        causes = value.get("cause_anchor_indices")
        effects = value.get("effect_anchor_indices")
        if not isinstance(causes, list) or not isinstance(effects, list) or not causes or not effects:
            return False
        for index in causes + effects:
            try:
                index = int(index)
            except Exception:
                return False
            if index < 0 or index >= frame_count:
                return False
    return True


def planner_signature(item: dict, source: dict, api_base: str, api_model: str) -> str:
    return core.stable_hash({
        "protocol": PROTOCOL_VERSION,
        "stage": "symmetric_planner",
        "question": item["question"],
        "video": Path(item["video_path"]).stem,
        "verifier_route": verifier_route(item),
        "reviewed": base_prediction(source),
        "storyboard_frames": STORYBOARD_FRAMES,
        "api_base": api_base,
        "api_model": api_model,
    })


def empty_plan(item, source, api_base, api_model, status, error="", raws=None, timestamps=None, duration=0.0):
    return {
        "question_id": item["question_id"],
        "request_signature": planner_signature(item, source, api_base, api_model),
        "api_status": status,
        "parse_status": status,
        "api_error": error,
        "raw_responses": raws or [],
        "answer_under_review": base_prediction(source),
        "route": verifier_route(item),
        "strongest_competitor": "",
        "hypotheses": [],
        "storyboard_timestamps": timestamps or [],
        "duration": float(duration),
    }


def run_planner(baseline, item, source, api_key, api_base, api_model):
    frames, time_string, duration = baseline.process_video(
        item["video_path"], STORYBOARD_FRAMES, 1, force_sample=True,
    )
    timestamps = [float(value.replace("s", "")) for value in time_string.split(",")]
    reviewed = base_prediction(source)
    labelled = core.label_frames(frames, timestamps, prefix="S")
    try:
        parsed, raws, parse_status = request_json(
            api_key, api_base, api_model,
            [
                planner_prompt(item["question"], reviewed, item["task_type"], timestamps),
                planner_prompt(item["question"], reviewed, item["task_type"], timestamps, retry=True),
            ],
            labelled, PLANNER_MAX_TOKENS,
            lambda value: valid_plan_json(value, reviewed, len(timestamps)),
        )
    except core.ApiContentBlocked as exc:
        return empty_plan(item, source, api_base, api_model, "content_blocked", str(exc), timestamps=timestamps, duration=duration)
    if parse_status == "failed":
        return empty_plan(
            item, source, api_base, api_model, "parse_failed",
            "Invalid symmetric two-hypothesis JSON after retry.", raws, timestamps, duration,
        )

    hypotheses = []
    for value in parsed["hypotheses"]:
        cause_indices = [int(x) for x in value["cause_anchor_indices"][:MAX_ANCHORS_PER_ROLE]]
        effect_indices = [int(x) for x in value["effect_anchor_indices"][:MAX_ANCHORS_PER_ROLE]]
        hypotheses.append({
            "option": option(value.get("option")),
            "cause_or_earlier_event": str(value.get("cause_or_earlier_event") or "").strip(),
            "effect_or_later_event": str(value.get("effect_or_later_event") or "").strip(),
            "expected_pre_state": str(value.get("expected_pre_state") or "").strip(),
            "expected_post_state": str(value.get("expected_post_state") or "").strip(),
            "cause_anchor_indices": cause_indices,
            "effect_anchor_indices": effect_indices,
            "cause_anchor_timestamps": [timestamps[index] for index in cause_indices],
            "effect_anchor_timestamps": [timestamps[index] for index in effect_indices],
        })
    return {
        "question_id": item["question_id"],
        "request_signature": planner_signature(item, source, api_base, api_model),
        "api_status": "ok",
        "parse_status": parse_status,
        "api_error": "",
        "raw_responses": raws,
        "answer_under_review": reviewed,
        "route": ROUTE_CAUSAL_CONSISTENCY,
        "strongest_competitor": option(parsed.get("strongest_competitor")),
        "hypotheses": hypotheses,
        "storyboard_timestamps": timestamps,
        "duration": float(duration),
    }


def order_planner_prompt(question: str, timestamps: list[float], retry: bool = False) -> str:
    labels = ", ".join(f"S{i}={value:.2f}s" for i, value in enumerate(timestamps))
    prefix = "Your previous JSON was invalid. " if retry else ""
    return f"""{prefix}Build a COMPLETE event timeline for this multiple-choice video
order question. Extract every atomic event/tool/concept whose order is permuted by the
answer options. Assign E1..EN in the order they are listed in the question, not video
order. Locate the first relevant occurrence/introduction of every event in the storyboard.
Also transcribe all four option permutations using only those event IDs.
Question and options: {question}
Storyboard labels: {labels}
Rules: include every ordered item exactly once; 2<=N<=6; every event needs a nonempty
anchor_indices list; option_orders A-D must each be a complete permutation of E1..EN;
descriptions <=40 characters. Return one-line minified JSON only, no markdown.
Schema: {{"events":[{{"event_id":"E1","description":"first listed event",
"anchor_indices":[0]}},{{"event_id":"E2","description":"second listed event",
"anchor_indices":[1]}},{{"event_id":"E3","description":"third listed event",
"anchor_indices":[2]}}],"option_orders":{{"A":["E1","E2","E3"],
"B":["E1","E3","E2"],"C":["E2","E1","E3"],"D":["E3","E1","E2"]}}}}"""


def deterministic_option_orders(question: str) -> dict[str, list[str]]:
    matches = re.findall(r"(?:^|\s)([A-D])\.\s*(.*?)(?=\s+[A-D]\.\s|$)", question, flags=re.S)
    if len(matches) != 4:
        return {}
    circled = {value: f"E{i + 1}" for i, value in enumerate("①②③④⑤⑥")}
    result = {}
    for answer, text in matches:
        parenthetical = re.findall(r"\(([a-fA-F])\)", text)
        if parenthetical:
            sequence = [f"E{ord(value.lower()) - ord('a') + 1}" for value in parenthetical]
        else:
            sequence = [circled[value] for value in text if value in circled]
        if len(sequence) < 2 or len(sequence) != len(set(sequence)):
            return {}
        result[answer] = sequence
    expected = set(result.get("A", []))
    if set(result) != {"A", "B", "C", "D"} or any(set(value) != expected for value in result.values()):
        return {}
    return result


def valid_order_plan_json(parsed: dict, frame_count: int, question: str = "") -> bool:
    if not isinstance(parsed, dict):
        return False
    events = parsed.get("events")
    if not isinstance(events, list) or not 2 <= len(events) <= 6:
        return False
    expected_ids = [f"E{i + 1}" for i in range(len(events))]
    actual_ids = [str(value.get("event_id") or "").upper() for value in events if isinstance(value, dict)]
    if actual_ids != expected_ids:
        return False
    for value in events:
        if not str(value.get("description") or "").strip():
            return False
        anchors = value.get("anchor_indices")
        if not isinstance(anchors, list) or not anchors:
            return False
        try:
            indices = [int(index) for index in anchors]
        except Exception:
            return False
        if any(index < 0 or index >= frame_count for index in indices):
            return False
    orders = parsed.get("option_orders")
    if not isinstance(orders, dict) or set(orders) != {"A", "B", "C", "D"}:
        return False
    for answer in "ABCD":
        sequence = orders.get(answer)
        if not isinstance(sequence, list) or len(sequence) != len(expected_ids) or set(sequence) != set(expected_ids):
            return False
    deterministic = deterministic_option_orders(question)
    if deterministic and any(parsed["option_orders"][answer] != deterministic[answer] for answer in "ABCD"):
        return False
    return True


def run_order_planner(baseline, item, source, api_key, api_base, api_model):
    frames, time_string, duration = baseline.process_video(
        item["video_path"], STORYBOARD_FRAMES, 1, force_sample=True,
    )
    timestamps = [float(value.replace("s", "")) for value in time_string.split(",")]
    labelled = core.label_frames(frames, timestamps, prefix="S")
    try:
        parsed, raws, parse_status = request_json(
            api_key, api_base, api_model,
            [order_planner_prompt(item["question"], timestamps), order_planner_prompt(item["question"], timestamps, retry=True)],
            labelled, PLANNER_MAX_TOKENS,
            lambda value: valid_order_plan_json(value, len(timestamps), item["question"]),
        )
    except core.ApiContentBlocked as exc:
        return empty_plan(item, source, api_base, api_model, "content_blocked", str(exc), timestamps=timestamps, duration=duration)
    if parse_status == "failed":
        return empty_plan(
            item, source, api_base, api_model, "parse_failed",
            "Invalid complete-order plan JSON after retry.", raws, timestamps, duration,
        )
    events = []
    for value in parsed["events"]:
        indices = [int(index) for index in value["anchor_indices"][:MAX_ANCHORS_PER_ROLE]]
        events.append({
            "event_id": str(value["event_id"]).upper(),
            "description": str(value["description"]).strip(),
            "anchor_indices": indices,
            "anchor_timestamps": [timestamps[index] for index in indices],
        })
    return {
        "question_id": item["question_id"],
        "request_signature": planner_signature(item, source, api_base, api_model),
        "api_status": "ok",
        "parse_status": parse_status,
        "api_error": "",
        "raw_responses": raws,
        "answer_under_review": base_prediction(source),
        "route": ROUTE_TEMPORAL_ORDER,
        "events": events,
        "option_orders": deterministic_option_orders(item["question"]) or {
            answer: [str(x).upper() for x in sequence]
            for answer, sequence in parsed["option_orders"].items()
        },
        "storyboard_timestamps": timestamps,
        "duration": float(duration),
    }


def event_timestamps(plan: dict) -> list[float]:
    duration = float(plan.get("duration") or 0.0)
    radius = min(MAX_EVENT_RADIUS_SECONDS, max(1.0, duration / max(1, STORYBOARD_FRAMES)))
    values = []
    if plan.get("route") == ROUTE_TEMPORAL_ORDER:
        anchor_groups = [value.get("anchor_timestamps", []) for value in plan.get("events", [])]
        limit = MAX_ORDER_AUDIT_FRAMES
    else:
        anchor_groups = [
            hypothesis.get(key, [])
            for hypothesis in plan.get("hypotheses", [])
            for key in ("cause_anchor_timestamps", "effect_anchor_timestamps")
        ]
        limit = MAX_AUDIT_FRAMES
    for anchors in anchor_groups:
        for anchor in anchors:
            for timestamp in (max(0.0, anchor - radius), anchor, min(duration, anchor + radius)):
                if all(abs(timestamp - old) > 1e-6 for old in values):
                    values.append(float(timestamp))
    values.sort()
    if len(values) > limit:
        indices = np.linspace(0, len(values) - 1, limit).round().astype(int)
        values = [values[index] for index in sorted(set(indices.tolist()))]
    return values


def sample_audit_frames(item: dict, plan: dict) -> dict:
    requested = event_timestamps(plan)
    raw, timestamps = core.sample_unique_frames(item["video_path"], requested, [])
    result = {
        "frames": core.stamp_frames(raw, timestamps, "F"),
        "timestamps": timestamps,
        "frame_ids": [f"F{i + 1}" for i in range(len(timestamps))],
        "event_allowed_ids": {},
    }
    if plan.get("route") == ROUTE_TEMPORAL_ORDER:
        duration = float(plan.get("duration") or 0.0)
        radius = min(MAX_EVENT_RADIUS_SECONDS, max(1.0, duration / max(1, STORYBOARD_FRAMES)))
        for event in plan.get("events", []):
            anchors = [float(value) for value in event.get("anchor_timestamps", [])]
            result["event_allowed_ids"][event["event_id"]] = [
                frame_id for frame_id, timestamp in zip(result["frame_ids"], timestamps)
                if any(abs(float(timestamp) - anchor) <= radius + 0.25 for anchor in anchors)
            ]
    return result


def anonymized_candidates(plan: dict, pass_number: int) -> tuple[list[dict], dict[str, str]]:
    values = list(plan.get("hypotheses", []))
    if pass_number == 2:
        values.reverse()
    mapping = {f"H{i + 1}": value["option"] for i, value in enumerate(values)}
    compact = []
    for index, value in enumerate(values):
        compact.append({
            "hypothesis_id": f"H{index + 1}",
            "cause_or_earlier_event": value["cause_or_earlier_event"],
            "effect_or_later_event": value["effect_or_later_event"],
            "expected_pre_state": value["expected_pre_state"],
            "expected_post_state": value["expected_post_state"],
        })
    return compact, mapping


def auditor_prompt(item: dict, plan: dict, sampled: dict, pass_number: int, retry: bool = False) -> str:
    candidates, _ = anonymized_candidates(plan, pass_number)
    labels = ", ".join(f"F{i + 1}={value:.2f}s" for i, value in enumerate(sampled["timestamps"]))
    prefix = "Previous JSON was invalid. " if retry else ""
    return f"""{prefix}You are a blinded causal-consistency auditor. Compare the two
anonymous hypotheses symmetrically using only attached timestamped frames. You do not
know which hypothesis is the current answer. Do not infer invisible events. A supported
causal hypothesis needs cited cause and effect frames, temporal precedence, and an
observable interaction/state transition; co-occurrence is insufficient.
Task type: {item['task_type']}
Question context: {item['question']}
Anonymous hypotheses: {json.dumps(candidates, ensure_ascii=False)}
Valid frame IDs: {labels}
Return exactly one check for H1 and one for H2. Use only valid F IDs. Text fields <=40
characters. If evidence cannot distinguish them use INSUFFICIENT and UNKNOWN. Return
one-line minified JSON only, no markdown.
Schema: {{"checks":[{{"hypothesis_id":"H1","status":"SUPPORTED",
"observed_cause_or_earlier_event":"fact","cause_evidence_ids":["F1"],
"observed_effect_or_later_event":"fact","effect_evidence_ids":["F2"],
"observed_pre_state":"state","observed_post_state":"state",
"temporal_precedence_verified":true,"state_transition_consistent":true,
"alternative_cause_visible":false}},{{"hypothesis_id":"H2","status":"REFUTED",
"observed_cause_or_earlier_event":"fact","cause_evidence_ids":["F3"],
"observed_effect_or_later_event":"fact","effect_evidence_ids":["F4"],
"observed_pre_state":"state","observed_post_state":"state",
"temporal_precedence_verified":true,"state_transition_consistent":false,
"alternative_cause_visible":false}}],"preferred_hypothesis":"H1/H2/UNKNOWN"}}"""


def valid_audit_json(parsed: dict, valid_ids: set[str]) -> bool:
    if not isinstance(parsed, dict):
        return False
    checks = parsed.get("checks")
    if not isinstance(checks, list) or len(checks) != 2:
        return False
    labels = {str(value.get("hypothesis_id") or "").upper() for value in checks if isinstance(value, dict)}
    if labels != {"H1", "H2"}:
        return False
    for value in checks:
        if str(value.get("status") or "").upper() not in {"SUPPORTED", "REFUTED", "INSUFFICIENT"}:
            return False
        for key in ("cause_evidence_ids", "effect_evidence_ids"):
            ids = value.get(key)
            if not isinstance(ids, list) or any(str(x).upper() not in valid_ids for x in ids):
                return False
    return str(parsed.get("preferred_hypothesis") or "").upper() in {"H1", "H2", "UNKNOWN"}


def ordered_event_specs(plan: dict, sampled: dict, pass_number: int) -> list[dict]:
    values = [
        {
            "event_id": event["event_id"],
            "description": event["description"],
            "allowed_frame_ids": sampled.get("event_allowed_ids", {}).get(event["event_id"], []),
        }
        for event in plan.get("events", [])
    ]
    return list(reversed(values)) if pass_number == 2 else values


def order_auditor_prompt(item: dict, plan: dict, sampled: dict, pass_number: int, retry: bool = False) -> str:
    labels = ", ".join(f"F{i + 1}={value:.2f}s" for i, value in enumerate(sampled["timestamps"]))
    prefix = "Previous JSON was invalid. " if retry else ""
    return f"""{prefix}Independently locate EVERY listed event using only the attached
timestamped frames. The event list order below is arbitrary and is not the video order.
Do not choose an answer option and do not omit difficult events. For each event cite the
frame showing its first relevant occurrence/introduction. Mark INSUFFICIENT if not shown.
Question context: {item['question']}
Events: {json.dumps(ordered_event_specs(plan, sampled, pass_number), ensure_ascii=False)}
Valid frame IDs: {labels}
Return exactly one record per event ID. Each event may cite ONLY its own
allowed_frame_ids; never borrow a frame assigned to another event. Output one-line minified
JSON only, no markdown.
Schema: {{"events":[{{"event_id":"E1","status":"OBSERVED",
"evidence_ids":["F1"],"visible_fact":"short fact"}},{{"event_id":"E2",
"status":"OBSERVED","evidence_ids":["F2"],"visible_fact":"short fact"}}]}}"""


def valid_order_audit_json(
    parsed: dict,
    required_events: set[str],
    valid_ids: set[str],
    event_allowed_ids: dict[str, list[str]],
) -> bool:
    if not isinstance(parsed, dict):
        return False
    events = parsed.get("events")
    if not isinstance(events, list) or len(events) != len(required_events):
        return False
    found = {str(value.get("event_id") or "").upper() for value in events if isinstance(value, dict)}
    if found != required_events:
        return False
    for value in events:
        status = str(value.get("status") or "").upper()
        ids = value.get("evidence_ids")
        if status not in {"OBSERVED", "INSUFFICIENT"} or not isinstance(ids, list):
            return False
        if any(str(evidence_id).upper() not in valid_ids for evidence_id in ids):
            return False
        if status == "OBSERVED" and not ids:
            return False
        event_id = str(value.get("event_id") or "").upper()
        allowed = set(event_allowed_ids.get(event_id, []))
        if status == "OBSERVED" and (not allowed or not set(str(x).upper() for x in ids).issubset(allowed)):
            return False
    return True


def derive_event_order(observations: list[dict], sampled: dict, option_orders: dict) -> tuple[list[str], str]:
    timestamp_by_id = {frame_id: timestamp for frame_id, timestamp in zip(sampled["frame_ids"], sampled["timestamps"])}
    positioned = []
    for value in observations:
        if value.get("status") != "OBSERVED" or not value.get("evidence_ids"):
            return [], ""
        timestamps = [timestamp_by_id[evidence_id] for evidence_id in value["evidence_ids"] if evidence_id in timestamp_by_id]
        if not timestamps:
            return [], ""
        positioned.append((min(timestamps), value["event_id"]))
    positioned.sort(key=lambda value: (value[0], value[1]))
    if len({round(value[0], 3) for value in positioned}) != len(positioned):
        return [], ""
    order = [value[1] for value in positioned]
    matched = next((answer for answer, sequence in option_orders.items() if sequence == order), "")
    return order, matched


def audit_signature(item, plan, sampled, pass_number, api_base, api_model):
    return core.stable_hash({
        "protocol": PROTOCOL_VERSION,
        "stage": f"blind_audit_pass_{pass_number}",
        "verifier_route": plan.get("route"),
        "question": item["question"],
        "plan_signature": plan.get("request_signature"),
        "timestamps": sampled.get("timestamps", []),
        "event_allowed_ids": sampled.get("event_allowed_ids", {}),
        "api_base": api_base,
        "api_model": api_model,
    })


def empty_audit(item, plan, sampled, pass_number, api_base, api_model, status, error="", raws=None):
    _, mapping = anonymized_candidates(plan, pass_number)
    return {
        "question_id": item["question_id"],
        "request_signature": audit_signature(item, plan, sampled, pass_number, api_base, api_model),
        "pass_number": pass_number,
        "route": plan.get("route", ROUTE_CAUSAL_CONSISTENCY),
        "api_status": status,
        "parse_status": status,
        "api_error": error,
        "raw_responses": raws or [],
        "anonymous_mapping": mapping,
        "checks": [],
        "preferred_option": "",
    }


def clean_fact(value) -> str:
    text = str(value or "").strip()
    if re.search(r"\b(?:answer|option|supports?|refutes?|correct|incorrect)\b", text, re.I):
        return ""
    return text[:240]


def run_audit(item, plan, sampled, pass_number, api_key, api_base, api_model):
    if plan.get("api_status") != "ok" or not len(sampled.get("frames", [])):
        return empty_audit(item, plan, sampled, pass_number, api_base, api_model, "skipped", "Planner or frames unavailable.")
    valid_ids = set(sampled["frame_ids"])
    try:
        parsed, raws, parse_status = request_json(
            api_key, api_base, api_model,
            [
                auditor_prompt(item, plan, sampled, pass_number),
                auditor_prompt(item, plan, sampled, pass_number, retry=True),
            ],
            list(sampled["frames"]), AUDITOR_MAX_TOKENS,
            lambda value: valid_audit_json(value, valid_ids),
        )
    except core.ApiContentBlocked as exc:
        return empty_audit(item, plan, sampled, pass_number, api_base, api_model, "content_blocked", str(exc))
    if parse_status == "failed":
        return empty_audit(item, plan, sampled, pass_number, api_base, api_model, "parse_failed", "Invalid audit JSON after retry.", raws)

    _, mapping = anonymized_candidates(plan, pass_number)
    checks = []
    for value in parsed["checks"]:
        hypothesis_id = str(value["hypothesis_id"]).upper()
        checks.append({
            "hypothesis_id": hypothesis_id,
            "option": mapping[hypothesis_id],
            "status": str(value["status"]).upper(),
            "observed_cause_or_earlier_event": clean_fact(value.get("observed_cause_or_earlier_event")),
            "cause_evidence_ids": [str(x).upper() for x in value.get("cause_evidence_ids", [])],
            "observed_effect_or_later_event": clean_fact(value.get("observed_effect_or_later_event")),
            "effect_evidence_ids": [str(x).upper() for x in value.get("effect_evidence_ids", [])],
            "observed_pre_state": clean_fact(value.get("observed_pre_state")),
            "observed_post_state": clean_fact(value.get("observed_post_state")),
            "temporal_precedence_verified": core.parse_bool(value.get("temporal_precedence_verified")),
            "state_transition_consistent": core.parse_bool(value.get("state_transition_consistent")),
            "alternative_cause_visible": core.parse_bool(value.get("alternative_cause_visible")),
            "alternative_cause_assessed": "alternative_cause_visible" in value,
        })
    preferred_id = str(parsed.get("preferred_hypothesis") or "UNKNOWN").upper()
    return {
        "question_id": item["question_id"],
        "request_signature": audit_signature(item, plan, sampled, pass_number, api_base, api_model),
        "pass_number": pass_number,
        "route": ROUTE_CAUSAL_CONSISTENCY,
        "api_status": "ok",
        "parse_status": parse_status,
        "api_error": "",
        "raw_responses": raws,
        "anonymous_mapping": mapping,
        "checks": checks,
        "preferred_option": mapping.get(preferred_id, ""),
    }


def run_order_audit(item, plan, sampled, pass_number, api_key, api_base, api_model):
    if plan.get("api_status") != "ok" or not len(sampled.get("frames", [])):
        return empty_audit(item, plan, sampled, pass_number, api_base, api_model, "skipped", "Planner or frames unavailable.")
    valid_ids = set(sampled["frame_ids"])
    required_events = {value["event_id"] for value in plan.get("events", [])}
    try:
        parsed, raws, parse_status = request_json(
            api_key, api_base, api_model,
            [
                order_auditor_prompt(item, plan, sampled, pass_number),
                order_auditor_prompt(item, plan, sampled, pass_number, retry=True),
            ],
            list(sampled["frames"]), AUDITOR_MAX_TOKENS,
            lambda value: valid_order_audit_json(
                value, required_events, valid_ids, sampled.get("event_allowed_ids", {}),
            ),
        )
    except core.ApiContentBlocked as exc:
        return empty_audit(item, plan, sampled, pass_number, api_base, api_model, "content_blocked", str(exc))
    if parse_status == "failed":
        return empty_audit(item, plan, sampled, pass_number, api_base, api_model, "parse_failed", "Invalid complete-order audit JSON after retry.", raws)
    observations = []
    for value in parsed["events"]:
        observations.append({
            "event_id": str(value["event_id"]).upper(),
            "status": str(value["status"]).upper(),
            "evidence_ids": [str(x).upper() for x in value.get("evidence_ids", [])],
            "visible_fact": clean_fact(value.get("visible_fact")),
        })
    derived_order, matched_option = derive_event_order(observations, sampled, plan.get("option_orders", {}))
    return {
        "question_id": item["question_id"],
        "request_signature": audit_signature(item, plan, sampled, pass_number, api_base, api_model),
        "pass_number": pass_number,
        "route": ROUTE_TEMPORAL_ORDER,
        "api_status": "ok",
        "parse_status": parse_status,
        "api_error": "",
        "raw_responses": raws,
        "anonymous_mapping": {},
        "checks": [],
        "preferred_option": matched_option,
        "event_observations": observations,
        "derived_order": derived_order,
        "matched_option": matched_option,
    }


def check_for(audit: dict, target: str) -> dict:
    return next((value for value in audit.get("checks", []) if value.get("option") == target), {})


def audit_consensus(source: dict, plan: dict, audits: list[dict], task_type: str) -> dict:
    reviewed = base_prediction(source)
    competitor = plan.get("strongest_competitor", "")
    reasons = []
    if plan.get("api_status") != "ok" or not competitor or competitor == reviewed:
        reasons.append("Symmetric two-hypothesis plan unavailable.")
    if len(audits) != 2 or any(value.get("api_status") != "ok" for value in audits):
        reasons.append("Both blinded audits are not available.")
    original_checks = [check_for(value, reviewed) for value in audits]
    competitor_checks = [check_for(value, competitor) for value in audits]
    if not all(value and value.get("status") == "REFUTED" for value in original_checks):
        reasons.append("Reviewed answer is not REFUTED by both audits.")
    if not all(value and value.get("status") == "SUPPORTED" for value in competitor_checks):
        reasons.append("Competitor is not SUPPORTED by both audits.")
    if not all(value.get("preferred_option") == competitor for value in audits):
        reasons.append("Order-swapped audits do not both prefer the competitor.")
    if not all(value and value.get("temporal_precedence_verified") for value in competitor_checks):
        reasons.append("Competitor temporal precedence is not verified twice.")
    if task_type == "Action Reasoning" and not all(
        value and value.get("state_transition_consistent") for value in competitor_checks
    ):
        reasons.append("Competitor state transition is not verified twice.")
    if not all(value and value.get("alternative_cause_assessed") for value in competitor_checks):
        reasons.append("Alternative causes were not assessed twice.")
    elif any(value.get("alternative_cause_visible") for value in competitor_checks):
        reasons.append("An alternative cause remains visible.")
    if not all(value and (value.get("cause_evidence_ids") or value.get("effect_evidence_ids")) for value in original_checks):
        reasons.append("Reviewed-answer refutation lacks cited evidence.")

    cause_sets = [set(value.get("cause_evidence_ids", [])) for value in competitor_checks if value]
    effect_sets = [set(value.get("effect_evidence_ids", [])) for value in competitor_checks if value]
    agreed_cause = sorted(set.intersection(*cause_sets)) if len(cause_sets) == 2 else []
    agreed_effect = sorted(set.intersection(*effect_sets)) if len(effect_sets) == 2 else []
    if not agreed_cause:
        reasons.append("Audits share no cited cause frame.")
    if not agreed_effect:
        reasons.append("Audits share no cited effect frame.")
    return {
        "route": ROUTE_CAUSAL_CONSISTENCY,
        "reviewed_option": reviewed,
        "competitor_option": competitor,
        "reopen_ready": not reasons,
        "decision": "REOPEN" if not reasons else "KEEP",
        "agreed_cause_ids": agreed_cause,
        "agreed_effect_ids": agreed_effect,
        "selected_evidence_ids": agreed_cause + [value for value in agreed_effect if value not in agreed_cause],
        "reasons": reasons,
    }


def order_consensus(source: dict, plan: dict, audits: list[dict], sampled: dict) -> dict:
    reviewed = base_prediction(source)
    reasons = []
    if plan.get("api_status") != "ok" or plan.get("route") != ROUTE_TEMPORAL_ORDER:
        reasons.append("Complete-order plan unavailable.")
    if len(audits) != 2 or any(value.get("api_status") != "ok" for value in audits):
        reasons.append("Both complete-order audits are not available.")
    orders = [value.get("derived_order", []) for value in audits]
    matched = [value.get("matched_option", "") for value in audits]
    if len(orders) != 2 or not all(orders):
        reasons.append("At least one audit did not observe every event.")
    if len(orders) != 2 or orders[0] != orders[1]:
        reasons.append("Order-swapped audits disagree on the complete event order.")
    if len(matched) != 2 or not matched[0] or matched[0] != matched[1]:
        reasons.append("Complete event order does not map consistently to one option.")

    selected_ids = []
    for event in plan.get("events", []):
        event_id = event["event_id"]
        observations = [
            next((entry for entry in audit.get("event_observations", []) if entry.get("event_id") == event_id), {})
            for audit in audits
        ]
        if not all(value.get("status") == "OBSERVED" for value in observations):
            reasons.append(f"{event_id} is not observed by both audits.")
            continue
        allowed = sampled.get("event_allowed_ids", {}).get(event_id, [])
        allowed_set = set(allowed)
        cited_sets = [set(value.get("evidence_ids", [])) for value in observations]
        # Each audit has already been schema-validated against this event-owned
        # window. Recheck it here because cached/external records may bypass the
        # parser. Exact frame-ID overlap is intentionally not required: two
        # adjacent frames in the same event window are equivalent independent
        # observations of that event.
        if not allowed_set or not all(values and values.issubset(allowed_set) for values in cited_sets):
            reasons.append(f"Audits do not both cite the event-owned window for {event_id}.")
            continue
        anchor_values = [float(value) for value in event.get("anchor_timestamps", [])]
        frame_times = dict(zip(sampled.get("frame_ids", []), sampled.get("timestamps", [])))
        event_selected = sorted(
            set.union(*cited_sets),
            key=lambda value: (
                min(
                    (abs(float(frame_times.get(value, 0.0)) - anchor) for anchor in anchor_values),
                    default=0.0,
                ),
                int(value[1:]),
            ),
        )
        allowed_by_anchor = sorted(
            allowed,
            key=lambda value: min(
                (abs(float(frame_times.get(value, 0.0)) - anchor) for anchor in anchor_values),
                default=0.0,
            ),
        )
        for value in allowed_by_anchor:
            if value not in event_selected:
                event_selected.append(value)
            if len(event_selected) >= 2:
                break
        selected_ids.extend(event_selected[:2])
    selected_ids = list(dict.fromkeys(selected_ids))
    matched_option = matched[0] if len(matched) == 2 and matched[0] == matched[1] else ""
    if matched_option == reviewed and not reasons:
        reasons.append("Complete timeline supports the reviewed answer.")
    reopen = bool(not reasons and matched_option and matched_option != reviewed)
    return {
        "route": ROUTE_TEMPORAL_ORDER,
        "reviewed_option": reviewed,
        "competitor_option": matched_option,
        "complete_event_order": orders[0] if len(orders) == 2 and orders[0] == orders[1] else [],
        "reopen_ready": reopen,
        "decision": "REOPEN" if reopen else "KEEP",
        "agreed_cause_ids": [],
        "agreed_effect_ids": [],
        "selected_evidence_ids": selected_ids,
        "reasons": reasons,
    }


def selected_audit_frames(sampled: dict, evidence_ids: list[str], limit: int):
    indices, selected_ids = [], []
    for evidence_id in evidence_ids[:limit]:
        match = re.fullmatch(r"F(\d+)", evidence_id)
        if match:
            index = int(match.group(1)) - 1
            if 0 <= index < len(sampled["frames"]) and index not in indices:
                indices.append(index)
                selected_ids.append(evidence_id)
    frames = sampled["frames"][indices] if indices else np.zeros((0, 336, 336, 3), dtype=np.uint8)
    timestamps = [sampled["timestamps"][index] for index in indices]
    return frames, timestamps, selected_ids


def independent_reanswer(baseline, item, source, sampled, consensus):
    if not consensus.get("reopen_ready"):
        return base_prediction(source), ""
    evidence_ids = []
    for value in consensus.get("selected_evidence_ids", []):
        if value not in evidence_ids:
            evidence_ids.append(value)
    limit = MAX_FINAL_ORDER_FRAMES if consensus.get("route") == ROUTE_TEMPORAL_ORDER else MAX_FINAL_CAUSAL_FRAMES
    audit_frames, audit_ts, selected_ids = selected_audit_frames(sampled, evidence_ids, limit)
    global_frames, time_string, _ = baseline.process_video(item["video_path"], GLOBAL_FRAMES, 1, force_sample=True)
    global_ts = [float(value.replace("s", "")) for value in time_string.split(",")]
    local_ts = causal_verifier.initial_local_timestamps(source)
    local_raw, local_ts = core.sample_unique_frames(item["video_path"], local_ts, global_ts)
    local_frames = core.stamp_frames(local_raw, local_ts, "L")
    merged, _, _ = causal_verifier.merge_sources([
        (global_frames, global_ts, "GLOBAL", 0),
        (local_frames, local_ts, "LOCAL", 1),
        (audit_frames, audit_ts, "CAUSAL_AUDIT", 2),
    ])
    mapping = ", ".join(f"{value}={timestamp:.2f}s" for value, timestamp in zip(selected_ids, audit_ts))
    note = (
        "A blinded structured evidence audit selected additional timestamped raw frames "
        f"({mapping}). Independently answer from the visual evidence; do not assume the audit is correct."
    )
    prompt = core.insert_evidence(source["agent_prompt"], [note])
    raw = core.infer_frames(baseline, prompt, merged)
    return option(raw), raw


def method_result(prediction: str, raw: str, ground_truth: str) -> dict:
    return {"raw_output": raw or prediction, "prediction": prediction, "correct": prediction == ground_truth}


def summarize(records: dict[str, dict], source_records: dict[str, dict], expected: int) -> dict:
    values = list(records.values())
    summary = {
        "expected": expected, "completed": len(values), "target_methods": {},
        "by_task_type": {}, "by_verifier_route": {},
    }
    for method in METHODS:
        correct = sum(value["methods"][method]["correct"] for value in values)
        summary["target_methods"][method] = {
            "correct": correct, "total": len(values),
            "accuracy": round(correct / len(values), 4) if values else 0.0,
        }
    for task_type in sorted(TARGET_TASK_TYPES):
        subset = [value for value in values if value["task_type"] == task_type]
        summary["by_task_type"][task_type] = {}
        for method in METHODS:
            correct = sum(value["methods"][method]["correct"] for value in subset)
            summary["by_task_type"][task_type][method] = {
                "correct": correct, "total": len(subset),
                "accuracy": round(correct / len(subset), 4) if subset else 0.0,
            }
    for route in (ROUTE_TEMPORAL_ORDER, ROUTE_CAUSAL_CONSISTENCY):
        subset = [value for value in values if value.get("verifier_route") == route]
        summary["by_verifier_route"][route] = {}
        for method in METHODS:
            correct = sum(value["methods"][method]["correct"] for value in subset)
            summary["by_verifier_route"][route][method] = {
                "correct": correct, "total": len(subset),
                "accuracy": round(correct / len(subset), 4) if subset else 0.0,
            }
    corrected = sum(not value["methods"]["answer_before_causal"]["correct"] and value["methods"]["answer_after_causal"]["correct"] for value in values)
    corrupted = sum(value["methods"]["answer_before_causal"]["correct"] and not value["methods"]["answer_after_causal"]["correct"] for value in values)
    accepted = sum(value["gate"]["accepted_change"] for value in values)
    accepted_correct = sum(
        value["gate"]["accepted_change"] and value["methods"]["answer_after_causal"]["correct"]
        for value in values
    )
    wrong_to_wrong = sum(
        value["gate"]["accepted_change"]
        and not value["methods"]["answer_before_causal"]["correct"]
        and not value["methods"]["answer_after_causal"]["correct"]
        for value in values
    )
    source_values = list(source_records.values())
    source_base_correct = sum(value.get("methods", {}).get(BASE_METHOD, {}).get("correct", False) for value in source_values)
    source_baseline_correct = sum(
        value.get("methods", {}).get("baseline_paired", {}).get("correct", False) for value in source_values
    )
    projected_correct = source_base_correct + corrected - corrupted
    summary["projected_full_dataset"] = {
        "base_method": BASE_METHOD,
        "paired_baseline_correct": source_baseline_correct,
        "base_correct": source_base_correct,
        "final_correct": projected_correct,
        "total": len(source_values),
        "base_accuracy": round(source_base_correct / len(source_values), 4) if source_values else 0.0,
        "paired_baseline_accuracy": round(source_baseline_correct / len(source_values), 4) if source_values else 0.0,
        "final_accuracy": round(projected_correct / len(source_values), 4) if source_values else 0.0,
    }
    summary["causal_verifier"] = {
        "planner_status": dict(Counter(value["planner"].get("api_status", "unknown") for value in values)),
        "planner_parse_status": dict(Counter(value["planner"].get("parse_status", "unknown") for value in values)),
        "audit_pass1_status": dict(Counter(value["audits"][0].get("api_status", "unknown") for value in values)),
        "audit_pass1_parse_status": dict(Counter(value["audits"][0].get("parse_status", "unknown") for value in values)),
        "audit_pass2_status": dict(Counter(value["audits"][1].get("api_status", "unknown") for value in values)),
        "audit_pass2_parse_status": dict(Counter(value["audits"][1].get("parse_status", "unknown") for value in values)),
        "consensus_reopen_ready": sum(value["consensus"].get("reopen_ready", False) for value in values),
        "route_counts": dict(Counter(value.get("verifier_route", "unknown") for value in values)),
        "consensus_failure_reasons": dict(Counter(
            reason for value in values for reason in value["consensus"].get("reasons", [])
        )),
        "independent_model_agreements": sum(value["gate"].get("independent_model_agreed", False) for value in values),
        "average_audit_frames": round(
            sum(len(value.get("audit_frames", {}).get("frame_ids", [])) for value in values) / len(values), 4
        ) if values else 0.0,
        "accepted_changes": accepted,
        "accepted_changes_correct": accepted_correct,
        "wrong_to_wrong_changes": wrong_to_wrong,
        "accepted_change_success_rate": round(accepted_correct / accepted, 4) if accepted else None,
        "corrected_errors": corrected,
        "corrupted_correct_answers": corrupted,
        "net_corrections": corrected - corrupted,
        "correction_precision": round(corrected / (corrected + corrupted), 4) if corrected + corrupted else None,
    }
    return summary


def write_outputs(output_dir: Path, records: dict[str, dict], source_records: dict[str, dict], expected: int):
    summary = summarize(records, source_records, expected)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["question_id", "task_type", "verifier_route", "ground_truth"]
    for method in METHODS:
        fields += [f"{method}_prediction", f"{method}_correct"]
    fields += ["planner_status", "audit1_status", "audit2_status", "reopen_ready", "accepted_change"]
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for qid in sorted(records):
            value = records[qid]
            row = {key: value[key] for key in ("question_id", "task_type", "verifier_route", "ground_truth")}
            for method in METHODS:
                row[f"{method}_prediction"] = value["methods"][method]["prediction"]
                row[f"{method}_correct"] = value["methods"][method]["correct"]
            row.update({
                "planner_status": value["planner"].get("api_status"),
                "audit1_status": value["audits"][0].get("api_status"),
                "audit2_status": value["audits"][1].get("api_status"),
                "reopen_ready": value["consensus"].get("reopen_ready"),
                "accepted_change": value["gate"].get("accepted_change"),
            })
            writer.writerow(row)


def write_full_predictions(output_dir: Path, source_records: dict[str, dict], target_records: dict[str, dict]) -> None:
    path = output_dir / "full_predictions.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for qid in sorted(source_records):
            source = source_records[qid]
            target = target_records.get(qid)
            before = base_prediction(source)
            final = target["methods"]["answer_after_causal"]["prediction"] if target else before
            baseline = option(source.get("methods", {}).get("baseline_paired", {}).get("prediction"))
            value = {
                "question_id": qid,
                "question_level": source.get("question_level"),
                "task_type": source.get("task_type"),
                "ground_truth": source.get("ground_truth"),
                "baseline_paired_prediction": baseline,
                "baseline_paired_correct": baseline == source.get("ground_truth"),
                "answer_before_verification": before,
                "answer_after_verification": final,
                "final_correct": final == source.get("ground_truth"),
                "verifier_route": target.get("verifier_route") if target else "PASS_THROUGH",
                "accepted_change": bool(target and target.get("gate", {}).get("accepted_change")),
            }
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Safe routed post-answer verifier")
    parser.add_argument("--source_results", default=str(DEFAULT_SOURCE_RESULTS))
    parser.add_argument("--data_dir", default=str(core.DEFAULT_DATA_DIR))
    parser.add_argument("--json", default=str(core.DEFAULT_ANNOTATION))
    parser.add_argument("--level_jsonl", default=str(core.DEFAULT_LEVELS))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--allow_partial_dataset", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--api_key", default=POST_CAUSAL_QWEN_API_KEY or os.environ.get("DASHSCOPE_API_KEY") or "")
    parser.add_argument("--api_base", default=core.QWEN_API_BASE)
    parser.add_argument("--api_model", default=core.QWEN_API_MODEL)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = Path(args.source_results).resolve()
    source_records = core.load_jsonl(source_path)
    source_targets = [
        value for value in source_records.values()
        if value.get("question_level") == TARGET_LEVEL and value.get("task_type") in TARGET_TASK_TYPES
    ]
    question_map = {
        value["question_id"]: value for value in core.build_question_list(
            Path(args.json).resolve(), Path(args.data_dir).resolve(), Path(args.level_jsonl).resolve(),
        )
    }
    targets = [(question_map[value["question_id"]], value) for value in source_targets if value["question_id"] in question_map]
    targets.sort(key=lambda pair: pair[0]["question_id"])
    if args.max_questions is not None:
        targets = targets[:max(0, args.max_questions)]
    if args.max_questions is None and not args.allow_partial_dataset and (
        len(source_targets) != EXPECTED_TARGET_QUESTIONS or len(targets) != EXPECTED_TARGET_QUESTIONS
    ):
        raise RuntimeError(
            f"Full-data guard failed: source={len(source_targets)}, runnable={len(targets)}, expected={EXPECTED_TARGET_QUESTIONS}."
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol": PROTOCOL_VERSION,
        "source_hash": core.file_hash(source_path),
        "question_ids_hash": core.stable_hash([value[0]["question_id"] for value in targets]),
        "base_method": BASE_METHOD,
        "storyboard_frames": STORYBOARD_FRAMES,
        "max_audit_frames": MAX_AUDIT_FRAMES,
        "max_order_audit_frames": MAX_ORDER_AUDIT_FRAMES,
        "api_base": args.api_base,
        "api_model": args.api_model,
    }
    fingerprint = core.stable_hash(config)
    manifest_path = output_dir / "manifest.json"
    cache_names = ["planner_cache.jsonl", "audit_pass1_cache.jsonl", "audit_pass2_cache.jsonl", "results.jsonl"]
    caches_exist = any((output_dir / name).exists() and (output_dir / name).stat().st_size for name in cache_names)
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("config_fingerprint") != fingerprint and caches_exist:
            raise RuntimeError("Output directory contains caches from another configuration; use a new --output_dir.")
    elif caches_exist:
        raise RuntimeError("Output directory contains unversioned caches; use a new --output_dir.")
    manifest = {
        "source_target_questions": len(source_targets),
        "runnable_questions": len(targets),
        "expected_questions": EXPECTED_TARGET_QUESTIONS,
        "config_fingerprint": fingerprint,
        "config": config,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[Manifest]", manifest)
    if args.prepare_only:
        return 0
    if not args.api_key:
        raise RuntimeError("Qwen API key is empty. Fill POST_CAUSAL_QWEN_API_KEY near the top of this file.")

    import video_evidence_backend as baseline

    planner_path = output_dir / "planner_cache.jsonl"
    audit_paths = [output_dir / "audit_pass1_cache.jsonl", output_dir / "audit_pass2_cache.jsonl"]
    results_path = output_dir / "results.jsonl"
    planner_cache = core.load_jsonl(planner_path)
    audit_caches = [core.load_jsonl(path) for path in audit_paths]
    records = core.load_jsonl(results_path)
    pending = []
    for pair in targets:
        old_record = records.get(pair[0]["question_id"], {})
        retryable_failure = (
            old_record.get("planner", {}).get("api_status") == "parse_failed"
            or any(value.get("api_status") == "parse_failed" for value in old_record.get("audits", []))
        )
        if old_record.get("config_fingerprint") != fingerprint or retryable_failure:
            pending.append(pair)

    for position, (item, source) in enumerate(pending, 1):
        qid = item["question_id"]
        print(f"\n[{position}/{len(pending)}] {qid} {item['task_type']}")
        expected_plan_signature = planner_signature(item, source, args.api_base, args.api_model)
        route = verifier_route(item)
        plan = planner_cache.get(qid)
        if plan is None or plan.get("request_signature") != expected_plan_signature or plan.get("api_status") == "parse_failed":
            planner_fn = run_order_planner if route == ROUTE_TEMPORAL_ORDER else run_planner
            plan = planner_fn(baseline, item, source, args.api_key, args.api_base, args.api_model)
            core.append_jsonl(planner_path, plan)
            planner_cache[qid] = plan
        sampled = sample_audit_frames(item, plan) if plan.get("api_status") == "ok" else {
            "frames": np.zeros((0, 336, 336, 3), dtype=np.uint8),
            "timestamps": [], "frame_ids": [], "event_allowed_ids": {},
        }
        audits = []
        for pass_index in range(2):
            pass_number = pass_index + 1
            signature = audit_signature(item, plan, sampled, pass_number, args.api_base, args.api_model)
            audit = audit_caches[pass_index].get(qid)
            if audit is None or audit.get("request_signature") != signature or audit.get("api_status") == "parse_failed":
                audit_fn = run_order_audit if route == ROUTE_TEMPORAL_ORDER else run_audit
                audit = audit_fn(
                    item, plan, sampled, pass_number,
                    args.api_key, args.api_base, args.api_model,
                )
                core.append_jsonl(audit_paths[pass_index], audit)
                audit_caches[pass_index][qid] = audit
            audits.append(audit)

        consensus = (
            order_consensus(source, plan, audits, sampled)
            if route == ROUTE_TEMPORAL_ORDER
            else audit_consensus(source, plan, audits, item["task_type"])
        )
        candidate, candidate_raw = independent_reanswer(baseline, item, source, sampled, consensus)
        reviewed = base_prediction(source)
        competitor = consensus.get("competitor_option", "")
        agreed = bool(consensus.get("reopen_ready") and candidate and candidate == competitor)
        accepted = bool(agreed and candidate != reviewed)
        final = candidate if accepted else reviewed
        gate = {
            "accepted_change": accepted,
            "independent_model_agreed": agreed,
            "decision": "REOPEN_ACCEPTED" if accepted else "KEEP",
            "prediction": final,
            "reasons": [] if accepted else (
                consensus.get("reasons", []) if not consensus.get("reopen_ready")
                else ["Independent local answer model did not select the audited competitor."]
            ),
        }
        ground_truth = source["ground_truth"]
        record = {
            "question_id": qid,
            "question_level": source["question_level"],
            "task_type": source["task_type"],
            "question": source["question"],
            "verifier_route": route,
            "ground_truth": ground_truth,
            "config_fingerprint": fingerprint,
            "methods": {
                "answer_before_causal": method_result(reviewed, source["methods"][BASE_METHOD].get("raw_output", reviewed), ground_truth),
                "causal_candidate": method_result(candidate, candidate_raw, ground_truth),
                "answer_after_causal": method_result(final, final, ground_truth),
            },
            "planner": plan,
            "audit_frames": {
                "timestamps": sampled["timestamps"],
                "frame_ids": sampled["frame_ids"],
                "event_allowed_ids": sampled.get("event_allowed_ids", {}),
            },
            "audits": audits,
            "consensus": consensus,
            "gate": gate,
        }
        core.append_jsonl(results_path, record)
        records[qid] = record
        write_outputs(output_dir, records, source_records, len(targets))
        print("[Done]", qid, "GT=", ground_truth, "before=", reviewed, "candidate=", candidate, "final=", final, "gate=", gate["decision"])

    write_outputs(output_dir, records, source_records, len(targets))
    write_full_predictions(output_dir, source_records, records)
    print(f"\n[Complete] {len(records)}/{len(targets)}")
    print("[Results]", results_path)
    print("[Summary]", output_dir / "summary.json")
    print("[Full predictions]", output_dir / "full_predictions.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
