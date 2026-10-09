"""Single-entry runner for the final relationally verified video QA method.

The command runs only our method. It does not execute or score a separate
Video-RAG baseline. The internal Video-RAG retrieval backend is retained because
it is part of the method's evidence construction stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as defaults
import complexity_profiler as complexity


HERE = PROJECT_ROOT
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SUPPORTED_FRAME_BUDGETS = (4, 8, 16)
PROTOCOL = "evisure-relational-stability"
ROUTE_TASKS = {"Action Reasoning", "Temporal Reasoning"}
VALID_LEVELS = {"A", "B", "C"}
VALID_TASKS = {
    "Action Reasoning", "Temporal Reasoning", "Spatial Perception",
    "Object Recognition", "Counting", "OCR", "Knowledge", "General",
}


def stable_hash(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def load_jsonl_by_id(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("question_id") or "").strip()
            if qid:
                rows[qid] = row
    return rows


def load_question_ids(path: Path | None) -> list[str]:
    """Read a JSON list, JSONL records, or one-question-id-per-line text file."""
    if path is None:
        return []
    if not path.is_file():
        raise FileNotFoundError(f"Question ID file not found: {path}")
    if path.suffix.lower() == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(loaded, dict):
            loaded = loaded.get("question_ids", loaded.get("ids", []))
        if not isinstance(loaded, list):
            raise ValueError("Question ID JSON must be a list or contain question_ids/ids.")
        values = [str(value.get("question_id") if isinstance(value, dict) else value).strip() for value in loaded]
    elif path.suffix.lower() == ".jsonl":
        values = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                values.append(str(value.get("question_id") if isinstance(value, dict) else value).strip())
    else:
        values = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    result = list(dict.fromkeys(value for value in values if value))
    if not result:
        raise ValueError(f"Question ID file is empty: {path}")
    return result


def select_question_ids(items: list[dict], question_ids: list[str]) -> list[dict]:
    if not question_ids:
        return items
    by_id = {str(item["question_id"]): item for item in items}
    missing = [qid for qid in question_ids if qid not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} requested question IDs were not found; examples: {missing[:8]}")
    return [by_id[qid] for qid in question_ids]


def profile_stage(args: argparse.Namespace, name: str):
    profile = getattr(args, "current_profile", None)
    return profile.stage(name) if profile is not None else nullcontext()


def start_question_profile(args: argparse.Namespace, item: dict) -> Any | None:
    if not getattr(args, "profile", False):
        args.current_profile = None
        return None
    seen = getattr(args, "profile_seen_videos", set())
    video_key = str(item.get("video_path") or item.get("video_id") or "")
    state = "warm" if video_key in seen else "cold"
    seen.add(video_key)
    args.profile_seen_videos = seen
    profile = complexity.QuestionProfiler(
        question_id=item["question_id"],
        video_id=item.get("video_id") or Path(str(item.get("video_path") or "")).stem,
        dataset=args.dataset_format,
        method="our_method",
        cache_state=state,
        video_key=video_key,
    )
    args.current_profile = profile
    profile.start()
    return profile


def restore_profile_seen_videos(args: argparse.Namespace, output_dir: Path) -> None:
    """Preserve Cold/Warm ordering when a profiling run resumes."""
    if not getattr(args, "profile", False):
        return
    existing = load_jsonl_by_id(output_dir / "profile_records.jsonl")
    seen = getattr(args, "profile_seen_videos", set())
    for row in existing.values():
        key = str(row.get("video_key") or row.get("video_id") or "").strip()
        if key:
            seen.add(key)
    args.profile_seen_videos = seen


def finish_question_profile(
    args: argparse.Namespace,
    output_dir: Path,
    record: dict,
    initial_diag: dict,
    relation: dict | None,
    stability: dict | None,
) -> None:
    profile = getattr(args, "current_profile", None)
    if profile is None:
        return
    profile_row, planner_row = profile.finish(record, initial_diag, relation, stability)
    complexity.append_jsonl(output_dir / "profile_records.jsonl", profile_row)
    complexity.append_jsonl(output_dir / "planner_records.jsonl", planner_row)
    args.current_profile = None


def configure_profile_cache(args: argparse.Namespace, frame_budget: int, output_dir: Path) -> Path:
    if not getattr(args, "profile", False):
        return output_dir / "evidence_cache"
    root = getattr(args, "profile_cache_root", None)
    if root is None:
        root = output_dir / "profile_cache"
    elif getattr(args, "profile_run_name", None):
        root = root / args.profile_run_name
    args.current_profile_cache_root = root
    cache_dir = root / args.dataset_format / f"frames_{frame_budget}" / "evidence_cache"
    baseline_audio = root / args.dataset_format / f"frames_{frame_budget}" / "baseline_audio"
    baseline_audio.mkdir(parents=True, exist_ok=True)
    os.environ["VIDRAG_AUDIO_CACHE_DIR"] = str(baseline_audio)
    return cache_dir


def profiling_output_dir(args: argparse.Namespace, frame_budget: int) -> Path:
    root = args.output_dir
    if getattr(args, "profile", False) and getattr(args, "profile_run_name", None):
        root = root / args.profile_run_name
    return root / f"frames_{frame_budget}"


def option_letter(value: Any) -> str:
    match = re.search(r"\b([A-D])\b", str(value or "").strip().upper())
    return match.group(1) if match else ""


def youtube_id(url: str) -> str:
    if "watch?v=" in url:
        return url.split("watch?v=")[-1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    return ""


def normalise_choices(choices: Any) -> list[str]:
    if not isinstance(choices, list):
        return []
    values = [str(value).strip() for value in choices if str(value).strip()]
    if not 2 <= len(values) <= 4:
        raise ValueError("Each question must have 2-4 choices.")
    result = []
    for index, value in enumerate(values):
        label = chr(ord("A") + index)
        result.append(value if re.match(r"^\s*[A-D][\.\)\:]\s*", value, flags=re.I) else f"{label}. {value}")
    return result


def compose_question(question: Any, choices: Any) -> str:
    text = str(question or "").strip()
    options = normalise_choices(choices)
    if not text:
        raise ValueError("Question text is empty.")
    return (text + " " + " ".join(options)).strip()


def read_annotation(path: Path, dataset_format: str) -> list[dict]:
    if dataset_format == "flat_jsonl" or path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(loaded, dict):
        for key in ("data", "questions", "annotations", "items"):
            if isinstance(loaded.get(key), list):
                loaded = loaded[key]
                break
    if not isinstance(loaded, list):
        raise ValueError("Annotation must be a JSON list, JSONL file, or an object containing a list.")
    return loaded


def build_video_map(video_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in video_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            result.setdefault(path.stem, path.resolve())
    return result


def resolve_video_path(record: dict, annotation: Path, video_dir: Path, video_map: dict[str, Path]) -> Path | None:
    for key in ("video_path", "video", "video_file", "filepath", "file_name"):
        value = str(record.get(key) or "").strip()
        if not value:
            continue
        supplied = Path(value)
        candidates = [supplied] if supplied.is_absolute() else [annotation.parent / supplied, video_dir / supplied]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    keys = [
        str(record.get("video_id") or "").strip(),
        str(record.get("video_name") or "").strip(),
        youtube_id(str(record.get("url") or "")),
    ]
    for key in keys + [Path(key).stem for key in keys if key]:
        if key in video_map:
            return video_map[key]
    return None


def make_item(record: dict, parent: dict, index: int, annotation: Path, video_dir: Path, video_map: dict[str, Path]) -> dict:
    combined = dict(parent)
    combined.update(record)
    video_path = resolve_video_path(combined, annotation, video_dir, video_map)
    qid = str(record.get("question_id") or record.get("id") or "").strip()
    if not qid:
        prefix = str(combined.get("video_id") or combined.get("video_name") or "item")
        qid = f"{prefix}-{index:05d}"
    return {
        "question_id": qid,
        "video_id": str(combined.get("video_id") or combined.get("video_name") or ""),
        "video_path": str(video_path) if video_path else "",
        "question": compose_question(record.get("question"), record.get("choices", record.get("options", []))),
        "ground_truth": option_letter(record.get("answer", record.get("ground_truth", record.get("label", "")))),
        "question_level": str(record.get("question_level") or combined.get("question_level") or "Unknown").upper(),
        "task_type": str(record.get("task_type") or combined.get("task_type") or "Unknown"),
        "video_category": str(combined.get("video_category") or "Unknown"),
        "video_subcategory": str(combined.get("video_subcategory") or "Unknown"),
    }


def load_dataset(annotation: Path, video_dir: Path, dataset_format: str) -> tuple[list[dict], list[dict]]:
    raw = read_annotation(annotation, dataset_format)
    video_map = build_video_map(video_dir)
    nested = dataset_format == "videomme" or (
        dataset_format == "auto" and raw and isinstance(raw[0], dict) and isinstance(raw[0].get("questions"), list)
    )
    items, missing = [], []
    position = 0
    if nested:
        for video in raw:
            for question in video.get("questions", []):
                position += 1
                item = make_item(question, video, position, annotation, video_dir, video_map)
                (items if item["video_path"] else missing).append(item)
    else:
        for record in raw:
            if not isinstance(record, dict):
                raise ValueError("Flat annotations must contain JSON objects.")
            position += 1
            item = make_item(record, {}, position, annotation, video_dir, video_map)
            (items if item["video_path"] else missing).append(item)
    duplicates = [qid for qid, count in Counter(item["question_id"] for item in items).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate question IDs: {duplicates[:5]}")
    return items, missing


def path_from_here(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (HERE / path).resolve()


def router_prompt(question: str) -> str:
    return f"""Classify one multiple-choice video question for a video QA agent.
Return JSON only. Do not answer the question.

Question and choices: {question}

question_level:
- A: direct static/global perception; no specialized evidence reasoning.
- B: needs ASR, OCR, object/detail retrieval, or ordinary localized evidence.
- C: needs event order, action/state transition, temporal relation, or causal reasoning.

task_type must be exactly one of:
Action Reasoning, Temporal Reasoning, Spatial Perception, Object Recognition,
Counting, OCR, Knowledge, General.

Return: {{"question_level":"A/B/C","task_type":"...","reason":"short reason"}}"""


def route_item(core: Any, item: dict, args: argparse.Namespace) -> dict:
    level = str(item.get("question_level") or "").upper()
    task = str(item.get("task_type") or "")
    if level in VALID_LEVELS and task in VALID_TASKS:
        return {**item, "route_source": "metadata"}
    if args.routing_mode == "metadata":
        return {**item, "route_source": "metadata_missing"}
    try:
        raw = core.api_chat(args.api_key, args.api_base, args.api_model, router_prompt(item["question"]), max_tokens=256)
        parsed = core.parse_json_loose(raw)
        inferred_level = str(parsed.get("question_level") or "B").upper()
        inferred_task = str(parsed.get("task_type") or "General")
        if inferred_level not in VALID_LEVELS:
            inferred_level = "B"
        if inferred_task not in VALID_TASKS:
            inferred_task = "General"
        return {
            **item,
            "question_level": inferred_level,
            "task_type": inferred_task,
            "route_source": "qwen_api",
            "route_reason": str(parsed.get("reason") or ""),
        }
    except Exception as exc:
        return {
            **item,
            "question_level": level if level in VALID_LEVELS else "B",
            "task_type": task if task in VALID_TASKS else "General",
            "route_source": "safe_fallback",
            "route_reason": f"{type(exc).__name__}: {exc}",
        }


def parse_timestamps(time_string: str) -> list[float]:
    return [float(value.strip().lower().removesuffix("s")) for value in str(time_string).split(",") if value.strip()]


def run_initial_answer(baseline: Any, core: Any, item: dict, args: argparse.Namespace, cache_dir: Path) -> tuple[dict, dict]:
    profile = getattr(args, "current_profile", None)
    audio_root = Path(os.environ.get("VIDRAG_AUDIO_CACHE_DIR", "restore/audio"))
    baseline_audio_path = audio_root / f"{Path(item['video_path']).stem}.txt"
    baseline_audio_existed = baseline_audio_path.exists()
    with profile_stage(args, "base_videorag"):
        base_diag = baseline.video_qa(item["video_path"], item["question"], max_frames_num=args.current_frames)
    if profile is not None:
        profile.set_cache(
            "baseline_audio_hit",
            baseline_audio_existed if base_diag.get("asr_requested") else None,
        )
    base_prompt = str(base_diag.get("final_prompt") or "")
    if not base_prompt:
        raise RuntimeError("Internal Video-RAG evidence prompt is empty.")
    with profile_stage(args, "global_frame_sampling"):
        global_frames, time_string, _ = baseline.process_video(item["video_path"], args.current_frames, 1, force_sample=True)
    global_ts = parse_timestamps(time_string)
    with profile_stage(args, "initial_planner"):
        planner = core.run_planner(
            baseline, item, args.api_key, args.api_base, args.api_model,
            args.planner_frames, global_frames=args.current_frames,
        )
    requested_tools = planner.get("tool_requests", []) or []
    video_path = Path(item["video_path"]).resolve()
    cache_key = core.stable_hash(str(video_path))[:16]
    extra_audio_path = cache_dir / "video_asr_cache" / f"{video_path.stem}_{cache_key}.txt"
    extra_ocr_path = cache_dir / "video_ocr_cache" / f"{video_path.stem}_{cache_key}.json"
    if profile is not None:
        profile.set_cache("extra_audio_hit", extra_audio_path.exists() if "ASR" in requested_tools else None)
        profile.set_cache("extra_ocr_hit", extra_ocr_path.exists() if "OCR" in requested_tools else None)
    with profile_stage(args, "tool_acquisition"):
        tools = core.acquire_tools(baseline, item, planner, base_diag, cache_dir)
    local_frames = core.np.zeros((0, 336, 336, 3), dtype=core.np.uint8)
    local_ts: list[float] = []
    if planner.get("keyframe_valid"):
        with profile_stage(args, "local_frame_sampling"):
            raw_local, local_ts = core.sample_unique_frames(
                item["video_path"], core.proposed_keyframe_timestamps(planner), global_ts,
            )
            local_frames = core.stamp_frames(raw_local, local_ts, "L")
    answer_frames = core.merge_chronological_frames(global_frames, global_ts, local_frames, local_ts)
    additions = core.evidence_lines(
        tools.get("new_asr_documents", []), tools.get("new_ocr_documents", []),
        local_ts, global_frames=args.current_frames,
    )
    agent_prompt = core.insert_evidence(base_prompt, additions)
    with profile_stage(args, "initial_answer"):
        raw = core.infer_frames(baseline, agent_prompt, answer_frames)
    prediction = core.extract_letter(raw)
    ground_truth = item["ground_truth"]
    source = {
        **item,
        "config_fingerprint": args.current_fingerprint,
        "agent_prompt": agent_prompt,
        "local_frame_timestamps": local_ts,
        "methods": {
            "agent_pre_reflection": {
                "prediction": prediction,
                "raw_output": raw,
                "correct": prediction == ground_truth if ground_truth else None,
            }
        },
    }
    diagnostics = {
        "initial_prediction": prediction,
        "initial_raw_output": raw,
        "global_frame_count": len(global_frames),
        "local_frame_count": len(local_ts),
        "initial_answer_frame_count": len(answer_frames),
        "local_frame_timestamps": local_ts,
        "planner": planner,
        "tools": tools,
        "base_evidence": base_diag,
    }
    return source, diagnostics


def empty_sample(core: Any) -> dict:
    return {
        "frames": core.np.zeros((0, 336, 336, 3), dtype=core.np.uint8),
        "timestamps": [], "frame_ids": [], "event_allowed_ids": {},
    }


def run_relation_verifier(
    baseline: Any, core: Any, causal_verifier: Any, post_verifier: Any,
    item: dict, source: dict, args: argparse.Namespace,
) -> dict:
    route = post_verifier.verifier_route(item)
    planner_fn = post_verifier.run_order_planner if route == post_verifier.ROUTE_TEMPORAL_ORDER else post_verifier.run_planner
    with profile_stage(args, "relation_planner"):
        plan = planner_fn(baseline, item, source, args.api_key, args.api_base, args.api_model)
    with profile_stage(args, "relation_frame_sampling"):
        sampled = post_verifier.sample_audit_frames(item, plan) if plan.get("api_status") == "ok" else empty_sample(core)
    audits = []
    for pass_number in (1, 2):
        audit_fn = post_verifier.run_order_audit if route == post_verifier.ROUTE_TEMPORAL_ORDER else post_verifier.run_audit
        with profile_stage(args, f"relation_audit{pass_number}"):
            audits.append(audit_fn(item, plan, sampled, pass_number, args.api_key, args.api_base, args.api_model))
    consensus = (
        post_verifier.order_consensus(source, plan, audits, sampled)
        if route == post_verifier.ROUTE_TEMPORAL_ORDER
        else post_verifier.audit_consensus(source, plan, audits, item["task_type"])
    )
    with profile_stage(args, "independent_reanswer"):
        candidate, candidate_raw = post_verifier.independent_reanswer(baseline, item, source, sampled, consensus)
    reviewed = post_verifier.base_prediction(source)
    competitor = consensus.get("competitor_option", "")
    agreed = bool(consensus.get("reopen_ready") and candidate and candidate == competitor)
    accepted = bool(agreed and candidate != reviewed)
    final = candidate if accepted else reviewed
    ground_truth = source["ground_truth"]
    return {
        "question_id": item["question_id"],
        "question_level": item["question_level"],
        "task_type": item["task_type"],
        "question": item["question"],
        "verifier_route": route,
        "ground_truth": ground_truth,
        "config_fingerprint": args.current_fingerprint,
        "methods": {
            "answer_before_causal": post_verifier.method_result(
                reviewed, source["methods"][post_verifier.BASE_METHOD].get("raw_output", reviewed), ground_truth,
            ),
            "causal_candidate": post_verifier.method_result(candidate, candidate_raw, ground_truth),
            "answer_after_causal": post_verifier.method_result(final, final, ground_truth),
        },
        "planner": plan,
        "audit_frames": {
            "timestamps": sampled["timestamps"],
            "frame_ids": sampled["frame_ids"],
            "event_allowed_ids": sampled.get("event_allowed_ids", {}),
        },
        "audits": audits,
        "consensus": consensus,
        "gate": {
            "accepted_change": accepted,
            "independent_model_agreed": agreed,
            "decision": "REOPEN_ACCEPTED" if accepted else "KEEP",
            "prediction": final,
            "reasons": [] if accepted else (
                consensus.get("reasons", []) if not consensus.get("reopen_ready")
                else ["The relation-aware local answer did not select the audited competitor."]
            ),
        },
    }


def run_stability_gate(baseline: Any, stability_gate: Any, item: dict, source: dict, relation: dict, fingerprint: str,
                       args: argparse.Namespace | None = None) -> dict:
    context_args = args or argparse.Namespace(current_profile=None)
    with profile_stage(context_args, "stability_frame_reconstruction"):
        sampled = stability_gate.reconstruct_audit_frames(item, relation)
    with profile_stage(context_args, "stability_pass1"):
        pass1_prediction, pass1_raw = stability_gate.run_confirmation(baseline, item, source, relation, sampled, 1)
    with profile_stage(context_args, "stability_pass2"):
        pass2_prediction, pass2_raw = stability_gate.run_confirmation(baseline, item, source, relation, sampled, 2)
    confirmation = {
        "question_id": item["question_id"],
        "request_signature": stability_gate.confirmation_signature(item, source, relation),
        "pass1": {"prediction": pass1_prediction, "raw_output": pass1_raw, "view": "standard_local"},
        "pass2": {"prediction": pass2_prediction, "raw_output": pass2_raw, "view": "wide_global"},
    }
    return stability_gate.make_output_record(relation, confirmation, fingerprint)


def compact_record(item: dict, source: dict, initial_diag: dict, relation: dict | None, stability: dict | None,
                   fingerprint: str, status: str, error: str, save_diagnostics: bool) -> dict:
    initial = source["methods"]["agent_pre_reflection"]["prediction"]
    final = (
        stability["methods"]["answer_after_stability"]["prediction"]
        if stability else initial
    )
    relation_target = bool(item["question_level"] == "C" and item["task_type"] in ROUTE_TASKS)
    record = {
        **item,
        "config_fingerprint": fingerprint,
        "status": status,
        "error": error,
        "relation_target": relation_target,
        "verifier_route": relation.get("verifier_route", "PASS_THROUGH") if relation else "PASS_THROUGH",
        "initial_prediction": initial,
        "final_prediction": final,
        "correct": final == item["ground_truth"] if item["ground_truth"] else None,
        "accepted_change": bool(stability and stability.get("stability_gate", {}).get("accepted_change")),
        "reopen_ready": bool(relation and relation.get("consensus", {}).get("reopen_ready")),
        "audited_competitor": (relation or {}).get("consensus", {}).get("competitor_option", ""),
        "verifier_vote": (relation or {}).get("methods", {}).get("causal_candidate", {}).get("prediction", ""),
        "confirmation_vote1": (stability or {}).get("methods", {}).get("confirmation_pass1", {}).get("prediction", ""),
        "confirmation_vote2": (stability or {}).get("methods", {}).get("confirmation_pass2", {}).get("prediction", ""),
        "global_frame_count": initial_diag.get("global_frame_count", 0),
        "local_frame_count": initial_diag.get("local_frame_count", 0),
        "initial_answer_frame_count": initial_diag.get("initial_answer_frame_count", 0),
    }
    if save_diagnostics:
        record["initial_stage"] = {"source_record": source, **initial_diag}
        record["relation_stage"] = relation or {}
        record["stability_stage"] = stability or {}
    return record


def write_outputs(output_dir: Path, records: dict[str, dict], frame_budget: int, wide_frames: int) -> None:
    values = [records[key] for key in sorted(records)]
    scored = [row for row in values if row.get("ground_truth")]
    summary = {
        "protocol": PROTOCOL,
        "answer_frames": frame_budget,
        "wide_global_frames": wide_frames,
        "completed": len(values),
        "scored": len(scored),
        "correct": sum(bool(row.get("correct")) for row in scored),
        "accuracy": round(sum(bool(row.get("correct")) for row in scored) / len(scored), 4) if scored else None,
        "accepted_changes": sum(bool(row.get("accepted_change")) for row in values),
        "relation_targets": sum(bool(row.get("relation_target")) for row in values),
        "relation_reopens": sum(bool(row.get("reopen_ready")) for row in values),
        "status_counts": dict(Counter(row.get("status", "unknown") for row in values)),
        "route_counts": dict(Counter(row.get("verifier_route", "unknown") for row in values)),
        "level_counts": dict(Counter(row.get("question_level", "Unknown") for row in values)),
        "task_counts": dict(Counter(row.get("task_type", "Unknown") for row in values)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = [
        "question_id", "video_id", "question_level", "task_type", "route_source", "ground_truth",
        "initial_prediction", "final_prediction", "correct", "status", "relation_target", "verifier_route",
        "reopen_ready", "audited_competitor", "verifier_vote", "confirmation_vote1", "confirmation_vote2",
        "accepted_change", "global_frame_count", "local_frame_count", "initial_answer_frame_count", "error",
    ]
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in values:
            writer.writerow({key: row.get(key, "") for key in fields})


def manifest_config(args: argparse.Namespace, annotation: Path, items: list[dict], frame_budget: int, wide_frames: int) -> dict:
    return {
        "protocol": PROTOCOL,
        "annotation": str(annotation),
        "annotation_hash": file_hash(annotation),
        "question_ids_hash": stable_hash([item["question_id"] for item in items]),
        "dataset_format": args.dataset_format,
        "answer_frames": frame_budget,
        "wide_global_frames": wide_frames,
        "planner_frames": args.planner_frames,
        "relation_storyboard_frames": args.relation_storyboard_frames,
        "routing_mode": args.routing_mode,
        "api_base": args.api_base,
        "api_model": args.api_model,
        "save_diagnostics": args.save_diagnostics,
        "profile": bool(getattr(args, "profile", False)),
        "profile_cache_root": str(getattr(args, "profile_cache_root", "") or ""),
        "profile_run_name": str(getattr(args, "profile_run_name", "") or ""),
    }


def ensure_manifest(output_dir: Path, config: dict) -> str:
    fingerprint = stable_hash(config)
    path = output_dir / "manifest.json"
    predictions = output_dir / "predictions.jsonl"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("config_fingerprint") != fingerprint and predictions.exists() and predictions.stat().st_size:
            raise RuntimeError(f"{output_dir} contains results from another configuration. Use a new output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"config_fingerprint": fingerprint, "config": config}, ensure_ascii=False, indent=2), encoding="utf-8")
    return fingerprint


def run_budget(args: argparse.Namespace, items: list[dict], annotation: Path, frame_budget: int) -> None:
    wide_frames = args.wide_frames or frame_budget * 2
    output_dir = profiling_output_dir(args, frame_budget)
    config = manifest_config(args, annotation, items, frame_budget, wide_frames)
    fingerprint = ensure_manifest(output_dir, config)
    records = load_jsonl_by_id(output_dir / "predictions.jsonl")
    restore_profile_seen_videos(args, output_dir)
    pending = [
        item for item in items
        if records.get(item["question_id"], {}).get("config_fingerprint") != fingerprint
        or (args.retry_errors and records.get(item["question_id"], {}).get("status") in {"error", "verifier_fallback"})
    ]
    print(f"[frames={frame_budget}] completed={len(records)}/{len(items)}, pending={len(pending)}")
    if args.prepare_only:
        write_outputs(output_dir, records, frame_budget, wide_frames)
        return

    cache_dir = configure_profile_cache(args, frame_budget, output_dir)
    model_load_started = time.perf_counter()
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    os.chdir(HERE)
    import video_evidence_backend as baseline
    import evidence_gap_reflective_agent as core
    import causal_evidence_verifier as causal_verifier
    import post_answer_causal_verifier as post_verifier
    import post_answer_stability_gate as stability_gate
    model_load_seconds = time.perf_counter() - model_load_started

    if getattr(args, "profile", False):
        complexity.install_runtime_wrappers(baseline, core, args)
        complexity.write_run_manifest(
            output_dir, args, args.dataset_format, frame_budget, model_load_seconds
        )

    core.QWEN_API_KEY = args.api_key
    core.QWEN_API_BASE = args.api_base
    core.QWEN_API_MODEL = args.api_model
    causal_verifier.CAUSAL_QWEN_API_KEY = args.api_key
    post_verifier.POST_CAUSAL_QWEN_API_KEY = args.api_key
    post_verifier.GLOBAL_FRAMES = frame_budget
    post_verifier.STORYBOARD_FRAMES = args.relation_storyboard_frames
    stability_gate.GLOBAL_FRAMES = frame_budget
    stability_gate.WIDE_GLOBAL_FRAMES = wide_frames

    args.current_frames = frame_budget
    args.current_fingerprint = fingerprint
    prediction_path = output_dir / "predictions.jsonl"
    for index, raw_item in enumerate(pending, 1):
        print(f"\n[{index}/{len(pending)}] {raw_item['question_id']}")
        profile = start_question_profile(args, raw_item)
        with profile_stage(args, "routing"):
            item = route_item(core, raw_item, args)
        source: dict = {}
        initial_diag: dict = {}
        relation = None
        stability = None
        status = "error"
        error = ""
        try:
            source, initial_diag = run_initial_answer(baseline, core, item, args, cache_dir)
            target = item["question_level"] == "C" and item["task_type"] in ROUTE_TASKS
            if not target:
                status = "pass_through"
            else:
                try:
                    relation = run_relation_verifier(baseline, core, causal_verifier, post_verifier, item, source, args)
                    if relation.get("consensus", {}).get("reopen_ready"):
                        stability = run_stability_gate(baseline, stability_gate, item, source, relation, fingerprint, args)
                        status = "stable_change" if stability["stability_gate"]["accepted_change"] else "unstable_keep"
                    else:
                        status = "verified_keep"
                except Exception as verifier_exc:
                    status = "verifier_fallback"
                    error = f"{type(verifier_exc).__name__}: {verifier_exc}"
                    print(f"[Verifier fallback] {item['question_id']}: {error}")
            record = compact_record(
                item, source, initial_diag, relation, stability,
                fingerprint, status, error, args.save_diagnostics,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"[Error] {item['question_id']}: {error}")
            record = {
                **item, "config_fingerprint": fingerprint, "status": "error", "error": error,
                "initial_prediction": "", "final_prediction": "", "correct": None,
                "relation_target": False, "verifier_route": "NOT_RUN", "accepted_change": False,
            }
        finish_question_profile(args, output_dir, record, initial_diag, relation, stability)
        append_jsonl(prediction_path, record)
        records[item["question_id"]] = record
        write_outputs(output_dir, records, frame_budget, wide_frames)
        print(
            "[Done]", item["question_id"], "level=", item["question_level"], "task=", item["task_type"],
            "initial=", record.get("initial_prediction"), "final=", record.get("final_prediction"),
            "status=", record.get("status"),
        )
    write_outputs(output_dir, records, frame_budget, wide_frames)
    if getattr(args, "profile", False):
        complexity.summarize_profiles(output_dir)
    print(f"[Complete] {len(records)}/{len(items)} -> {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run our final video QA method end to end.")
    parser.add_argument("--annotation", default=defaults.ANNOTATION)
    parser.add_argument("--video_dir", default=defaults.VIDEO_DIR)
    parser.add_argument("--output_dir", default=defaults.OUTPUT_DIR)
    parser.add_argument("--dataset_format", choices=("auto", "videomme", "flat_json", "flat_jsonl"), default=defaults.DATASET_FORMAT)
    parser.add_argument("--answer_frames", nargs="+", type=int, choices=SUPPORTED_FRAME_BUDGETS, default=defaults.ANSWER_FRAMES)
    parser.add_argument("--planner_frames", type=int, default=defaults.PLANNER_FRAMES)
    parser.add_argument("--relation_storyboard_frames", type=int, default=defaults.RELATION_STORYBOARD_FRAMES)
    parser.add_argument("--wide_frames", type=int, default=None, help="Default: 2 x answer_frames (formal 16-frame run uses 32).")
    parser.add_argument("--routing_mode", choices=("auto", "metadata"), default=defaults.ROUTING_MODE)
    parser.add_argument("--api_key", default=os.environ.get("UNIFIED_QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or defaults.API_KEY)
    parser.add_argument("--api_base", default=defaults.API_BASE)
    parser.add_argument("--api_model", default=defaults.API_MODEL)
    parser.add_argument("--max_questions", type=int, default=defaults.MAX_QUESTIONS)
    parser.add_argument("--allow_missing_videos", action="store_true", default=defaults.ALLOW_MISSING_VIDEOS)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--retry_errors", action="store_true")
    parser.add_argument("--save_diagnostics", action="store_true", default=defaults.SAVE_DIAGNOSTICS)
    parser.add_argument("--compact", action="store_true", help="Do not save large nested planner/audit diagnostics.")
    parser.add_argument("--profile", action="store_true", help="Record per-question complexity and planner statistics.")
    parser.add_argument("--question_ids_file", default=None, help="JSON/JSONL/TXT IDs to run, in the requested order.")
    parser.add_argument("--profile_cache_root", default=None, help="Independent cache root used only by profiling runs.")
    parser.add_argument("--profile_run_name", default=None, help="Unique directory name for one reproducible profiling run.")
    args = parser.parse_args()
    if args.compact:
        args.save_diagnostics = False
    args.annotation = path_from_here(args.annotation)
    args.video_dir = path_from_here(args.video_dir)
    args.output_dir = path_from_here(args.output_dir)
    args.question_ids_file = path_from_here(args.question_ids_file) if args.question_ids_file else None
    args.profile_cache_root = path_from_here(args.profile_cache_root) if args.profile_cache_root else None
    if args.profile_run_name:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.profile_run_name):
            parser.error("--profile_run_name must use only letters, digits, '.', '_' or '-'.")
    elif args.profile:
        parser.error("--profile requires an explicit --profile_run_name for unique, resumable outputs.")
    args.current_profile = None
    args.profile_seen_videos = set()
    return args


def main() -> int:
    args = parse_args()
    if not args.annotation.is_file():
        raise FileNotFoundError(f"Annotation not found: {args.annotation}")
    if not args.video_dir.is_dir():
        raise NotADirectoryError(f"Video directory not found: {args.video_dir}")
    if not args.prepare_only and not args.api_key:
        raise RuntimeError("Qwen API key is empty. Fill API_KEY in config.py or set DASHSCOPE_API_KEY.")
    items, missing = load_dataset(args.annotation, args.video_dir, args.dataset_format)
    items = select_question_ids(items, load_question_ids(args.question_ids_file))
    if args.max_questions is not None:
        items = items[:max(0, args.max_questions)]
    if missing:
        preview = ", ".join(item["question_id"] for item in missing[:8])
        message = f"{len(missing)} questions have no matching video; examples: {preview}"
        if not args.allow_missing_videos:
            raise RuntimeError(message + ". Fix paths or enable ALLOW_MISSING_VIDEOS.")
        print("[Warning] " + message)
    if not items:
        raise RuntimeError("No runnable questions remain after video matching.")
    print(f"[Dataset] runnable={len(items)}, missing={len(missing)}, frame_budgets={args.answer_frames}")
    for frame_budget in args.answer_frames:
        run_budget(args, items, args.annotation, frame_budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



# CUDA_VISIBLE_DEVICES=1 python scripts/run_our_method_videomme.py \
#   --annotation /path/to/Video-MME.json \
#   --video_dir /path/to/Video-MME/videos \
#   --output_dir ./outputs/videomme \
#   --dataset_format videomme \
#   --routing_mode metadata

# CUDA_VISIBLE_DEVICES=1 python scripts/run_our_method_videomme.py \
#   --annotation /path/to/Video-MME.json \
#   --video_dir /path/to/Video-MME/videos \
#   --output_dir ./outputs/videomme_metadata \
#   --dataset_format videomme \
#   --routing_mode metadata \
#   --allow_missing_videos


# CUDA_VISIBLE_DEVICES=1 python scripts/run_our_method_videomme.py \
#   --annotation /path/to/Video-MME.json \
#   --video_dir /path/to/Video-MME/videos \
#   --output_dir ./outputs/videomme_auto \
#   --dataset_format videomme \
#   --routing_mode auto \
#   --allow_missing_videos


# CUDA_VISIBLE_DEVICES=1 python scripts/run_our_method_videomme.py \
#   --annotation /path/to/Video-MME.json \
#   --video_dir /path/to/Video-MME/videos \
#   --output_dir ./outputs/videomme_frame8_auto \
#   --dataset_format videomme \
#   --answer_frames 8 \
#   --routing_mode auto \
#   --allow_missing_videos
